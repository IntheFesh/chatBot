"""Outgoing messages (R-CH-006, R-CH-007, R-CH-008, R-SAFE-006; protocol sections 5.4-5.7, 10.6).

Every send passes the same checks in the same order:

1. the recipient must be the bound user (otherwise :class:`RecipientNotAllowed`);
2. an image must be on the allow list (otherwise :class:`MediaNotAllowed`);
3. there must be a login and a ``context_token`` (otherwise a failed result, nothing is sent);
4. the safe window and quota must have room, unless a :class:`SendBypass` (the probe) allows
   the send; a window the platform already refused stays closed until the user writes again;
5. one request, one item, no automatic retry.  A refused or unknown outcome is reported, never
   repeated: a doubled bubble hurts more than a missing one.

Sends are serialised so bubbles keep their order and the quota is counted exactly.
"""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from twin.channel.base import (
    AuthState,
    BypassGrant,
    BypassRequest,
    MediaNotAllowed,
    OutboundKind,
    OutboundResult,
    SendBypass,
)
from twin.channel.binding import RecipientGuard
from twin.channel.ilink.auth import AuthGuard
from twin.channel.ilink.http import (
    TIMEOUT_QUICK_S,
    TIMEOUT_SEND_S,
    ApiAuth,
    IlinkError,
    IlinkHttp,
    IlinkHttpError,
    IlinkProtocolError,
    IlinkTransportError,
)
from twin.channel.ilink.media import CdnClient, MediaTransferError, sniff_image_mime
from twin.channel.ilink.store import ContextToken, IlinkStore
from twin.channel.ilink.wire import (
    ERR_SEND_REJECTED,
    ITEM_IMAGE,
    ITEM_TEXT,
    MESSAGE_BOT,
    STATE_FINISH,
    TEXT_CHUNK_LIMIT,
    TYPING_CANCEL,
    TYPING_START,
    as_text,
)
from twin.channel.policy import OutboundMediaPolicy, ensure_media_allowed
from twin.channel.window import SessionWindow
from twin.clock import Clock
from twin.llm.redaction import redact_text
from twin.ops.logging import get_logger

log = get_logger("twin.channel.ilink.outbound")

IMAGE_MIMES = {
    "image/png": "image/png",
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/gif": "image/gif",
    "image/webp": "image/webp",
}
TYPING_KEEPALIVE_S = 5.0
TYPING_MAX_S = 180.0
TICKET_TTL_S = 600.0
ERRMSG_LIMIT = 200


@dataclass(frozen=True)
class _Ready:
    """A send that passed the local checks."""

    user_id: str
    auth: ApiAuth
    context: ContextToken


class IlinkSender:
    """Sends text, images and typing notices to the bound user."""

    def __init__(
        self,
        *,
        http: IlinkHttp,
        cdn: CdnClient,
        store: IlinkStore,
        guard: AuthGuard,
        clock: Clock,
        recipients: RecipientGuard,
        media_policy: OutboundMediaPolicy,
        window_h: float,
        quota: int,
    ) -> None:
        self._http = http
        self._cdn = cdn
        self._store = store
        self._guard = guard
        self._clock = clock
        self._recipients = recipients
        self._media_policy = media_policy
        self._window_h = window_h
        self._quota = quota
        self._lock = asyncio.Lock()
        self._tickets: dict[str, tuple[str, float]] = {}
        self._typing_task: asyncio.Task[None] | None = None

    # ---------------------------------------------------------------- text

    async def send_text(
        self, text: str, *, recipient: str | None = None, bypass: SendBypass | None = None
    ) -> OutboundResult:
        user = self._recipients.resolve(recipient)
        if not text.strip():
            return OutboundResult.failure(OutboundKind.REJECTED, "empty_text")
        if len(text) > TEXT_CHUNK_LIMIT:
            return OutboundResult.failure(OutboundKind.REJECTED, "text_too_long")
        async with self._lock:
            ready = await self._prepare(user, "text", text, bypass)
            if isinstance(ready, OutboundResult):
                return ready
            item = {"type": ITEM_TEXT, "text_item": {"text": text}}
            return await self._post_message(ready, item, indexed_text=text)

    # --------------------------------------------------------------- image

    async def send_image(
        self,
        data: bytes | Path,
        mime: str,
        *,
        recipient: str | None = None,
        bypass: SendBypass | None = None,
    ) -> OutboundResult:
        user = self._recipients.resolve(recipient)
        raw = data if isinstance(data, bytes) else await asyncio.to_thread(Path(data).read_bytes)
        canonical = IMAGE_MIMES.get(mime.lower())
        if canonical is None:
            raise MediaNotAllowed(f"images of type {mime!r} are not sent")
        if sniff_image_mime(raw) != canonical:
            raise MediaNotAllowed("the bytes are not an image of the declared type")
        await asyncio.to_thread(ensure_media_allowed, self._media_policy, raw)
        async with self._lock:
            ready = await self._prepare(user, "image", None, bypass)
            if isinstance(ready, OutboundResult):
                return ready
            try:
                uploaded = await self._cdn.upload_image(raw, auth=ready.auth, to_user_id=user)
            except MediaTransferError as exc:
                if exc.auth_expired:
                    await self._guard.on_expired(source="getuploadurl", code=-14, errmsg=None)
                    return OutboundResult.failure(
                        OutboundKind.AUTH_EXPIRED, "needs_relogin", code=-14
                    )
                log.warning("image_upload_failed", reason=exc.reason)
                return OutboundResult.failure(
                    OutboundKind.UPLOAD_FAILED, exc.reason, http_status=exc.status
                )
            item = {
                "type": ITEM_IMAGE,
                "image_item": {
                    "media": {
                        "encrypt_query_param": uploaded.download_param,
                        "aes_key": uploaded.aes_key_b64,
                        "encrypt_type": 1,
                    },
                    "mid_size": uploaded.cipher_size,
                },
            }
            return await self._post_message(ready, item, indexed_text=None)

    # ------------------------------------------------------------ the gate

    async def _prepare(
        self, user: str, kind: Literal["text", "image"], text: str | None, bypass: SendBypass | None
    ) -> _Ready | OutboundResult:
        credentials = await asyncio.to_thread(self._store.credentials)
        if credentials is None:
            return OutboundResult.failure(OutboundKind.AUTH_EXPIRED, "not_logged_in")
        auth = await asyncio.to_thread(self._store.auth_record)
        if auth.state is AuthState.NEEDS_RELOGIN:
            return OutboundResult.failure(
                OutboundKind.AUTH_EXPIRED, "needs_relogin", code=auth.code
            )
        context = await asyncio.to_thread(self._store.context_token)
        if context is None:
            return OutboundResult.failure(OutboundKind.WINDOW_REJECTED, "no_context_token")
        window = SessionWindow(
            window_h=self._window_h,
            quota=self._quota,
            state=await asyncio.to_thread(self._store.window_state),
        )
        now = self._clock.now_utc()
        refusal = window.refusal_reason(now)
        if refusal == "session_expired":  # the platform already said no; wait for the user
            return OutboundResult.failure(
                OutboundKind.WINDOW_REJECTED, refusal, session_expired=True
            )
        grant: BypassGrant | None = None
        if bypass is not None:
            grant = bypass.authorize(BypassRequest(kind, text, now, refusal))
        elif refusal is not None:
            return OutboundResult.failure(OutboundKind.WINDOW_REJECTED, refusal)
        if grant is not None and grant.empty_context_token:  # the probe's optional experiment
            context = ContextToken("", context.received_at)
        return _Ready(user, ApiAuth(credentials.api_base_url, credentials.bot_token), context)

    # ---------------------------------------------------------- the request

    async def _post_message(
        self, ready: _Ready, item: dict[str, Any], *, indexed_text: str | None
    ) -> OutboundResult:
        client_id = (
            f"wechat-twin:{int(self._clock.now_utc().timestamp() * 1000)}-{secrets.token_hex(4)}"
        )
        body = {
            "msg": {
                "from_user_id": "",
                "to_user_id": ready.user_id,
                "client_id": client_id,
                "message_type": MESSAGE_BOT,
                "message_state": STATE_FINISH,
                "context_token": ready.context.token,
                "item_list": [item],
            }
        }
        try:
            response = await self._http.post(
                "sendmessage",
                body,
                base_url=ready.auth.base_url,
                token=ready.auth.token,
                timeout_s=TIMEOUT_SEND_S,
            )
        except IlinkTransportError as exc:
            if not exc.request_sent:
                return await self._failed(OutboundKind.NETWORK, exc.reason, count=False)
            return await self._failed(OutboundKind.AMBIGUOUS, exc.reason, count=True)
        except IlinkHttpError as exc:
            kind = OutboundKind.AMBIGUOUS if exc.status >= 500 else OutboundKind.REJECTED
            return await self._failed(
                kind, f"http_{exc.status}", count=kind is OutboundKind.AMBIGUOUS, status=exc.status
            )
        except IlinkProtocolError as exc:
            return await self._failed(OutboundKind.AMBIGUOUS, str(exc), count=True)
        if response.auth_expired:
            await self._guard.on_expired(
                source="sendmessage", code=response.error_code, errmsg=response.errmsg
            )
            return OutboundResult.failure(
                OutboundKind.AUTH_EXPIRED,
                "needs_relogin",
                code=response.error_code,
                errmsg=_clean(response.errmsg),
                client_id=client_id,
                ret=response.ret,
                errcode=response.errcode,
            )
        code = response.error_code
        if code == ERR_SEND_REJECTED:
            return await self._window_rejected(
                code, response.errmsg, client_id, ret=response.ret, errcode=response.errcode
            )
        if code is not None:
            return await self._failed(
                OutboundKind.REJECTED,
                f"ret_{code}",
                count=False,
                code=code,
                errmsg=response.errmsg,
                client_id=client_id,
                ret=response.ret,
                errcode=response.errcode,
            )
        message_id = as_text(response.data.get("message_id"))
        await asyncio.to_thread(self._record_sent)
        if message_id and indexed_text:
            await asyncio.to_thread(self._store.add_quote, message_id, indexed_text)
        return OutboundResult.success(message_id=message_id, client_id=client_id)

    def _record_sent(self) -> None:
        self._store.update_window(self._window_h, self._quota, lambda w: w.on_outbound(1))

    async def _window_rejected(
        self,
        code: int,
        errmsg: str | None,
        client_id: str,
        *,
        ret: int | None = None,
        errcode: int | None = None,
    ) -> OutboundResult:
        clean = _clean(errmsg)
        now = self._clock.now_utc()
        await asyncio.to_thread(
            self._store.update_window,
            self._window_h,
            self._quota,
            lambda w: w.mark_expired(now, code=code, errmsg=clean),
        )
        log.warning("send_rejected_by_platform", code=code)
        return OutboundResult.failure(
            OutboundKind.WINDOW_REJECTED,
            "platform_rejected",
            code=code,
            errmsg=clean,
            session_expired=True,
            client_id=client_id,
            ret=ret,
            errcode=errcode,
        )

    async def _failed(
        self,
        kind: OutboundKind,
        reason: str,
        *,
        count: bool,
        status: int | None = None,
        code: int | None = None,
        errmsg: str | None = None,
        client_id: str | None = None,
        ret: int | None = None,
        errcode: int | None = None,
    ) -> OutboundResult:
        clean = _clean(errmsg)
        now = self._clock.now_utc()

        def change(window: SessionWindow) -> None:
            if count:  # the message may have been delivered: use up the quota to be safe
                window.on_outbound(1)
            window.record_error(
                now,
                kind=kind.value,
                code=code if code is not None else status,
                errmsg=clean or reason,
            )

        await asyncio.to_thread(self._store.update_window, self._window_h, self._quota, change)
        log.warning("send_failed", kind=kind.value, reason=reason, code=code)
        return OutboundResult.failure(
            kind,
            reason,
            code=code,
            errmsg=clean,
            http_status=status,
            client_id=client_id,
            ret=ret,
            errcode=errcode,
        )

    # -------------------------------------------------------------- typing

    async def send_typing(self, active: bool, *, recipient: str | None = None) -> None:
        user = self._recipients.resolve(recipient)
        if not active:
            await self._stop_keepalive()
            cached = self._tickets.get(user)
            if cached is not None:
                await self._typing_request(user, cached[0], TYPING_CANCEL)
            return
        ticket = await self._ticket(user)
        if ticket is None:
            return
        await self._typing_request(user, ticket, TYPING_START)
        await self._stop_keepalive()
        self._typing_task = asyncio.create_task(self._keepalive(user, ticket))

    async def _keepalive(self, user: str, ticket: str) -> None:
        waited = 0.0
        while waited < TYPING_MAX_S:
            await self._clock.sleep(TYPING_KEEPALIVE_S)
            waited += TYPING_KEEPALIVE_S
            await self._typing_request(user, ticket, TYPING_START)
        await self._typing_request(user, ticket, TYPING_CANCEL)

    async def _stop_keepalive(self) -> None:
        task, self._typing_task = self._typing_task, None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def aclose(self) -> None:
        await self._stop_keepalive()

    async def _ticket(self, user: str) -> str | None:
        cached = self._tickets.get(user)
        now = self._clock.monotonic()
        if cached is not None and now - cached[1] < TICKET_TTL_S:
            return cached[0]
        credentials = await asyncio.to_thread(self._store.credentials)
        auth = await asyncio.to_thread(self._store.auth_record)
        if credentials is None or auth.state is not AuthState.OK:
            return None
        context = await asyncio.to_thread(self._store.context_token)
        body: dict[str, Any] = {"ilink_user_id": user}
        if context is not None:
            body["context_token"] = context.token
        try:
            response = await self._http.post(
                "getconfig",
                body,
                base_url=credentials.api_base_url,
                token=credentials.bot_token,
                timeout_s=TIMEOUT_QUICK_S,
            )
        except IlinkError as exc:
            log.debug("typing_ticket_unavailable", error=str(exc))
            return None
        if response.auth_expired:
            await self._guard.on_expired(
                source="getconfig", code=response.error_code, errmsg=response.errmsg
            )
            return None
        ticket = response.data.get("typing_ticket")
        if response.error_code is not None or not isinstance(ticket, str) or not ticket:
            return None
        self._tickets[user] = (ticket, now)
        return ticket

    async def _typing_request(self, user: str, ticket: str, status: int) -> None:
        credentials = await asyncio.to_thread(self._store.credentials)
        if credentials is None:
            return
        try:
            response = await self._http.post(
                "sendtyping",
                {"ilink_user_id": user, "typing_ticket": ticket, "status": status},
                base_url=credentials.api_base_url,
                token=credentials.bot_token,
                timeout_s=TIMEOUT_QUICK_S,
            )
        except IlinkError as exc:
            log.debug("typing_failed", error=str(exc))
            return
        if response.auth_expired:
            await self._guard.on_expired(
                source="sendtyping", code=response.error_code, errmsg=response.errmsg
            )


def _clean(errmsg: str | None) -> str | None:
    return redact_text(errmsg)[:ERRMSG_LIMIT] if errmsg else None
