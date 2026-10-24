"""HTTP access to the iLink API: headers, ``base_info``, timeouts and error mapping.

Everything that leaves this module goes to the official API host (or the host the login
named).  Tokens travel only in the ``Authorization`` header and are never logged; errors carry
the endpoint name and numbers, never URLs with query strings and never bodies.
"""

from __future__ import annotations

import base64
import json
import secrets
from dataclasses import dataclass
from typing import Any

import httpx

from twin import __version__
from twin.channel.ilink.wire import (
    APP_ID,
    CLIENT_VERSION,
    ERR_TOKEN_STALE,
)

BOT_AGENT = f"WechatTwin/{__version__}"

TIMEOUT_POLL_MARGIN_S = 5.0
TIMEOUT_DEFAULT_POLL_S = 35.0
TIMEOUT_SEND_S = 15.0
TIMEOUT_QUICK_S = 10.0
TIMEOUT_QR = httpx.Timeout(None, connect=TIMEOUT_QUICK_S)
"""``get_bot_qrcode``: no read timeout (the official client has none, protocol document section
2.4; a slow answer is not an error, the login has its own deadline), but a host that cannot be
reached at all still fails after the connect timeout (D-680)."""

MAX_RESPONSE_BYTES = 16 * 1024 * 1024
"""The largest API answer read (R18-A6, D-768).  The server decides how big an answer is; a
``getupdates`` batch is a few kilobytes, so anything near this is not a protocol answer."""

TimeoutSpec = float | httpx.Timeout | None


class IlinkError(Exception):
    """Base class of iLink transport errors."""


class IlinkTransportError(IlinkError):
    """The request failed before a usable answer arrived.

    ``request_sent`` is false when the connection could not even be made (so a resend is
    safe), true when the request may have reached the server (read timeout, dropped
    connection): the outcome is then unknown.
    """

    def __init__(self, endpoint: str, reason: str, *, request_sent: bool, timeout: bool) -> None:
        super().__init__(f"{endpoint}: {reason}")
        self.endpoint = endpoint
        self.reason = reason
        self.request_sent = request_sent
        self.timeout = timeout


class IlinkHttpError(IlinkError):
    """The server answered with a non-2xx status."""

    def __init__(self, endpoint: str, status: int) -> None:
        super().__init__(f"{endpoint}: HTTP {status}")
        self.endpoint = endpoint
        self.status = status


class IlinkProtocolError(IlinkError):
    """A 2xx answer that is not the JSON object the protocol promises."""

    def __init__(self, endpoint: str, reason: str) -> None:
        super().__init__(f"{endpoint}: {reason}")
        self.endpoint = endpoint


@dataclass(frozen=True)
class ApiAuth:
    """Where to send an authenticated request and with which token."""

    base_url: str
    token: str


@dataclass(frozen=True)
class IlinkResponse:
    """A parsed 2xx answer: the JSON object plus the ``ret``/``errcode`` convention."""

    status: int
    data: dict[str, Any]
    headers: httpx.Headers

    @property
    def ret(self) -> int | None:
        return _as_int(self.data.get("ret"))

    @property
    def errcode(self) -> int | None:
        return _as_int(self.data.get("errcode"))

    @property
    def errmsg(self) -> str | None:
        value = self.data.get("errmsg")
        return value if isinstance(value, str) and value else None

    @property
    def auth_expired(self) -> bool:
        return ERR_TOKEN_STALE in (self.ret, self.errcode)

    @property
    def error_code(self) -> int | None:
        """The error number to report: -14 wins, then ``errcode``, then ``ret``; 0/missing = ok."""
        if self.auth_expired:
            return ERR_TOKEN_STALE
        for value in (self.errcode, self.ret):
            if value:
                return value
        return None

    @property
    def ok(self) -> bool:
        return self.error_code is None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def random_uin() -> str:
    """``X-WECHAT-UIN``: a fresh random 32-bit number, as decimal text, base64-encoded."""
    return base64.b64encode(str(secrets.randbits(32)).encode("ascii")).decode("ascii")


def base_info() -> dict[str, str]:
    return {"channel_version": __version__, "bot_agent": BOT_AGENT}


class IlinkHttp:
    """Thin wrapper over one shared :class:`httpx.AsyncClient`."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(follow_redirects=False)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def reset_connections(self) -> bool:
        """Drop the pooled connections (after the machine slept they are dead).

        Only a client this object made is replaced; one that was handed in belongs to the caller
        and is left as it is.  Returns whether the connections were dropped.
        """
        if not self._owns_client:
            return False
        stale = self._client
        self._client = httpx.AsyncClient(follow_redirects=False)
        await stale.aclose()
        return True

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client

    def _headers(self, token: str | None, *, authorized: bool) -> dict[str, str]:
        headers = {
            "iLink-App-Id": APP_ID,
            "iLink-App-ClientVersion": CLIENT_VERSION,
        }
        if authorized:
            headers["Content-Type"] = "application/json"
            headers["AuthorizationType"] = "ilink_bot_token"
            headers["X-WECHAT-UIN"] = random_uin()
            if token:
                headers["Authorization"] = f"Bearer {token}"
        return headers

    async def post(
        self,
        endpoint: str,
        body: dict[str, Any],
        *,
        base_url: str,
        token: str | None,
        timeout_s: TimeoutSpec,
        query: dict[str, str] | None = None,
        with_base_info: bool = True,
    ) -> IlinkResponse:
        """``POST <base>/ilink/bot/<endpoint>`` with JSON; ``token=None`` sends no bearer."""
        payload = dict(body)
        if with_base_info:
            payload["base_info"] = base_info()
        url = f"{base_url.rstrip('/')}/ilink/bot/{endpoint}"
        return await self._send(
            "POST",
            endpoint,
            url,
            headers=self._headers(token, authorized=True),
            timeout_s=timeout_s,
            params=query,
            json_body=payload,
        )

    async def get(
        self,
        endpoint: str,
        params: dict[str, str],
        *,
        base_url: str,
        timeout_s: TimeoutSpec,
    ) -> IlinkResponse:
        """``GET <base>/ilink/bot/<endpoint>`` without credentials (the login status poll)."""
        url = f"{base_url.rstrip('/')}/ilink/bot/{endpoint}"
        return await self._send(
            "GET",
            endpoint,
            url,
            headers=self._headers(None, authorized=False),
            timeout_s=timeout_s,
            params=params,
            json_body=None,
        )

    async def _send(
        self,
        method: str,
        endpoint: str,
        url: str,
        *,
        headers: dict[str, str],
        timeout_s: TimeoutSpec,
        params: dict[str, str] | None,
        json_body: dict[str, Any] | None,
    ) -> IlinkResponse:
        try:
            async with self._client.stream(
                method,
                url,
                headers=headers,
                params=params,
                json=json_body,
                timeout=timeout_s,
            ) as response:
                if not 200 <= response.status_code < 300:
                    raise IlinkHttpError(endpoint, response.status_code)
                body = await read_limited(response, MAX_RESPONSE_BYTES)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            raise IlinkTransportError(
                endpoint, type(exc).__name__, request_sent=False, timeout=_is_timeout(exc)
            ) from None
        except httpx.TimeoutException as exc:
            raise IlinkTransportError(
                endpoint, type(exc).__name__, request_sent=True, timeout=True
            ) from None
        except httpx.HTTPError as exc:
            raise IlinkTransportError(
                endpoint, type(exc).__name__, request_sent=True, timeout=False
            ) from None
        except BodyTooLarge:
            raise IlinkProtocolError(endpoint, "the answer is too large") from None
        return IlinkResponse(response.status_code, _parse_object(endpoint, body), response.headers)


class BodyTooLarge(Exception):
    """A response body is larger than the caller accepts (see :func:`read_limited`)."""


async def read_limited(response: httpx.Response, limit: int) -> bytes:
    """The body of a streamed ``response``, refusing more than ``limit`` bytes (R18-A6, D-768).

    A declared ``Content-Length`` above the limit is refused before the first byte; otherwise
    reading stops - and :class:`BodyTooLarge` is raised - as soon as the bytes read pass the
    limit, so at most one chunk more than the limit is ever held.  The limit counts the decoded
    bytes (a compressed answer cannot inflate past it).
    """
    declared = response.headers.get("content-length", "")
    if declared.strip().isdigit() and int(declared) > limit:
        raise BodyTooLarge
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > limit:
            raise BodyTooLarge
        chunks.append(chunk)
    return b"".join(chunks)


def _is_timeout(exc: Exception) -> bool:
    return isinstance(exc, httpx.TimeoutException)


def _parse_object(endpoint: str, body: bytes) -> dict[str, Any]:
    if not body.strip():
        return {}
    try:
        data = json.loads(body)
    except ValueError:
        raise IlinkProtocolError(endpoint, "the answer is not JSON") from None
    if not isinstance(data, dict):
        raise IlinkProtocolError(endpoint, "the answer is not a JSON object")
    return data
