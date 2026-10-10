"""QR-code login (R-CH-003; protocol document sections 5.1, 5.2 and 8).

The flow: ask for a code (``get_bot_qrcode``), show it, then poll the scan status until the
person confirms on their phone.  The state machine handles every status the server can
report: ``wait``, ``scaned``, ``need_verifycode`` (the phone shows a number to type here),
``verify_code_blocked``, ``expired`` (a new code is fetched, up to three in all),
``scaned_but_redirect`` (continue on another host), ``binded_redirect`` and ``confirmed``.
Network trouble and gateway timeouts while waiting count as "still waiting".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit

from twin.channel.ilink.http import (
    TIMEOUT_DEFAULT_POLL_S,
    TIMEOUT_QR_S,
    IlinkError,
    IlinkHttp,
    IlinkHttpError,
    IlinkResponse,
    IlinkTransportError,
)
from twin.channel.ilink.wire import BOT_TYPE, DEFAULT_API_BASE
from twin.clock import Clock
from twin.ops.logging import get_logger

log = get_logger("twin.channel.ilink.login")

MAX_QR_CODES = 3
TOTAL_TIMEOUT_S = 480.0  # the official login command waits eight minutes
STATUS_TIMEOUT_S = TIMEOUT_DEFAULT_POLL_S + 5.0
POLL_GAP_S = 1.0
_HOST = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?(:\d{1,5})?$")


class LoginError(Exception):
    """The login could not be completed; the message says what to do."""


class LoginUI(Protocol):
    """What the login flow shows and asks (the terminal in production, a script in tests)."""

    def show_qr(self, content: str, *, number: int, total: int) -> None: ...

    def ask_verify_code(self, *, previous_was_wrong: bool) -> str: ...

    def info(self, message: str) -> None: ...


@dataclass(frozen=True)
class LoginResult:
    bot_token: str
    ilink_bot_id: str
    ilink_user_id: str | None
    api_base_url: str


def normalise_base_url(value: str | None, fallback: str) -> str:
    """The API host the login named; only HTTPS is accepted."""
    if not value:
        return fallback
    parts = urlsplit(value.strip())
    if parts.scheme != "https" or not parts.netloc or not _HOST.match(parts.netloc):
        raise LoginError("the server named an API address that is not a plain https host")
    return f"https://{parts.netloc}"


def redirect_base(host: str | None, current: str) -> str:
    """``scaned_but_redirect``: poll on ``https://<redirect_host>`` (unchanged if absent)."""
    if not host:
        return current
    cleaned = host.strip().removeprefix("https://").strip("/")
    if not _HOST.match(cleaned):
        raise LoginError("the server asked to continue on an invalid host")
    return f"https://{cleaned}"


class IlinkLogin:
    """Runs one login to completion."""

    def __init__(
        self,
        http: IlinkHttp,
        clock: Clock,
        ui: LoginUI,
        *,
        api_base_url: str = DEFAULT_API_BASE,
        local_token_list: tuple[str, ...] = (),
        total_timeout_s: float = TOTAL_TIMEOUT_S,
        max_codes: int = MAX_QR_CODES,
        poll_gap_s: float = POLL_GAP_S,
    ) -> None:
        self._http = http
        self._clock = clock
        self._ui = ui
        self._api_base = api_base_url
        self._local_tokens = list(local_token_list[:10])
        self._total_timeout_s = total_timeout_s
        self._max_codes = max_codes
        self._poll_gap_s = poll_gap_s

    async def run(self) -> LoginResult:
        deadline = self._clock.monotonic() + self._total_timeout_s
        poll_base = self._api_base
        issued = 0
        qrcode = ""
        verify_code: str | None = None
        asked_before = False
        while True:
            if not qrcode:
                if issued >= self._max_codes:
                    raise LoginError(
                        f"the login code expired {self._max_codes} times without being "
                        "confirmed; run the command again"
                    )
                qrcode = await self._new_code(issued + 1)
                issued += 1
                verify_code = None
                asked_before = False
            if self._clock.monotonic() >= deadline:
                raise LoginError("timed out waiting for the scan; run the command again")
            answer = await self._status(poll_base, qrcode, verify_code)
            if answer is None:  # network trouble, a read timeout or a gateway error: keep waiting
                await self._clock.sleep(self._poll_gap_s)
                continue
            status = str(answer.data.get("status", "wait"))
            if status == "confirmed":
                return self._result(answer)
            if status == "expired":
                self._ui.info("The login code expired; getting a new one.")
                qrcode = ""
                continue
            if status == "verify_code_blocked":
                self._ui.info("Too many wrong verification numbers; getting a new code.")
                qrcode = ""
                continue
            if status == "need_verifycode":
                verify_code = self._ui.ask_verify_code(previous_was_wrong=asked_before).strip()
                asked_before = True
                continue
            if status == "scaned":
                verify_code = None  # a correct number was accepted
                asked_before = False
                self._ui.info("Scanned: confirm the login on your phone.")
            elif status == "scaned_but_redirect":
                poll_base = redirect_base(_text(answer.data.get("redirect_host")), poll_base)
            elif status == "binded_redirect":
                raise LoginError(
                    "the server reports this bot is already bound to this installation but "
                    "no credentials are stored here; try again after a minute"
                )
            elif status != "wait":
                log.warning("unknown_login_status", status=status[:40])
            await self._clock.sleep(self._poll_gap_s)

    async def _new_code(self, number: int) -> str:
        try:
            answer = await self._http.post(
                "get_bot_qrcode",
                {"local_token_list": self._local_tokens},
                base_url=self._api_base,
                token=None,
                timeout_s=TIMEOUT_QR_S,
                query={"bot_type": str(BOT_TYPE)},
                with_base_info=False,
            )
        except IlinkError as exc:
            raise LoginError(f"cannot reach the login service: {exc}") from None
        qrcode = _text(answer.data.get("qrcode"))
        content = _text(answer.data.get("qrcode_img_content"))
        if not qrcode or not content:
            raise LoginError("the login service did not return a code")
        self._ui.show_qr(content, number=number, total=self._max_codes)
        return qrcode

    async def _status(
        self, base_url: str, qrcode: str, verify_code: str | None
    ) -> IlinkResponse | None:
        params = {"qrcode": qrcode}
        if verify_code:
            params["verify_code"] = verify_code
        try:
            return await self._http.get(
                "get_qrcode_status", params, base_url=base_url, timeout_s=STATUS_TIMEOUT_S
            )
        except (IlinkTransportError, IlinkHttpError):
            return None  # network trouble, a read timeout or a gateway error: still waiting
        except IlinkError as exc:
            log.warning("login_status_unreadable", error=str(exc))
            return None

    def _result(self, answer: IlinkResponse) -> LoginResult:
        token = _text(answer.data.get("bot_token"))
        bot_id = _text(answer.data.get("ilink_bot_id"))
        if not token or not bot_id:
            raise LoginError("the server confirmed the login but sent no credentials")
        base = normalise_base_url(_text(answer.data.get("baseurl")), self._api_base)
        return LoginResult(token, bot_id, _text(answer.data.get("ilink_user_id")), base)


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
