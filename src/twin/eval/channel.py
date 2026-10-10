"""``InMemoryChannel``: a channel that keeps everything in memory (R-EVAL-009).

The evaluation sandbox must never talk to WeChat: it replaces the channel with this one.  It
implements the whole :class:`~twin.channel.base.Channel` interface - it is a real channel, not a
stand-in for one - and keeps what the other channels keep:

* **one user, one id** (:data:`EVAL_USER_ID`); every send resolves its recipient through the
  same :class:`~twin.channel.binding.RecipientGuard`, naming anyone else raises
  :class:`~twin.channel.base.RecipientNotAllowed`;
* **the same picture allow list** (R-SAFE-006): bytes go out only when the media policy knows
  their SHA-256, and must be an image of the declared type, otherwise
  :class:`~twin.channel.base.MediaNotAllowed`;
* **quotes**: the channel accepts them (``supports_quote``), so the style measurements of the
  sandbox include the quote rate the running channel cannot show.

What it does instead of sending: every message is appended to :attr:`sent` (text, the quote that
went with it, or the SHA-256 and type of a picture) and handed to nobody.  A test or the sandbox
reads the log; :meth:`push` puts a message of the user into :meth:`incoming`.  There is no window
and no quota, since nothing is delivered.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from twin.channel.base import (
    AuthState,
    Channel,
    ChannelCapabilities,
    InboundMessage,
    MediaNotAllowed,
    MessageKind,
    OutboundKind,
    OutboundResult,
    QuoteTarget,
    SendBypass,
    SessionState,
)
from twin.channel.binding import RecipientGuard
from twin.channel.ilink.media import sniff_image_mime
from twin.channel.ilink.outbound import IMAGE_MIMES
from twin.channel.ilink.wire import TEXT_CHUNK_LIMIT
from twin.channel.policy import DenyAllMediaPolicy, OutboundMediaPolicy, ensure_media_allowed
from twin.clock import Clock

EVAL_USER_ID = "eval-sandbox-user"
QUOTA = 1_000_000  # nothing is delivered, so nothing runs out


@dataclass(frozen=True)
class SentMessage:
    """One send call that the channel accepted."""

    id: str
    kind: Literal["text", "image"]
    at: datetime
    text: str | None = None
    quote: QuoteTarget | None = None
    sha256: str | None = None
    mime: str | None = None


class InMemoryChannel(Channel):
    """A complete channel whose messages stay in memory (see the module description)."""

    name = "in_memory_channel"

    def __init__(
        self,
        *,
        clock: Clock,
        media_policy: OutboundMediaPolicy | None = None,
        user_id: str = EVAL_USER_ID,
        supports_quote: bool = True,
    ) -> None:
        self._clock = clock
        self._media_policy: OutboundMediaPolicy = media_policy or DenyAllMediaPolicy()
        self._user_id = user_id
        self._supports_quote = supports_quote
        self.guard = RecipientGuard(lambda: self._user_id)
        self.sent: list[SentMessage] = []
        self.typing = False
        self.started = False
        self._inbox: asyncio.Queue[InboundMessage | None] = asyncio.Queue()
        self._last_inbound_at: datetime | None = None
        self._outbound_since_inbound = 0
        self._counter = 0

    # ----------------------------------------------------------- lifecycle

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        if self.started:
            self.started = False
            self._inbox.put_nowait(None)

    # ------------------------------------------------------------- inbound

    def push(self, text: str, *, at: datetime | None = None) -> InboundMessage:
        """A text message of the user arrives (it is handed out by :meth:`incoming`)."""
        self._counter += 1
        moment = at if at is not None else self._clock.now_utc()
        message = InboundMessage(
            id=f"memory-in-{self._counter}", at=moment, kind=MessageKind.TEXT, text=text
        )
        self._last_inbound_at = moment
        self._outbound_since_inbound = 0
        self._inbox.put_nowait(message)
        return message

    async def incoming(self) -> AsyncIterator[InboundMessage]:
        while True:
            item = await self._inbox.get()
            if item is None:
                self._inbox.put_nowait(None)  # a second iteration also ends
                return
            yield item

    # ------------------------------------------------------------ outbound

    def _accepted(
        self,
        kind: Literal["text", "image"],
        *,
        text: str | None = None,
        quote: QuoteTarget | None = None,
        sha256: str | None = None,
        mime: str | None = None,
    ) -> OutboundResult:
        self._counter += 1
        message_id = f"memory-out-{self._counter}"
        self.sent.append(
            SentMessage(message_id, kind, self._clock.now_utc(), text, quote, sha256, mime)
        )
        self._outbound_since_inbound += 1
        self.typing = False
        return OutboundResult.success(message_id=message_id)

    async def send_text(
        self,
        text: str,
        quote: QuoteTarget | None = None,
        *,
        recipient: str | None = None,
        bypass: SendBypass | None = None,
    ) -> OutboundResult:
        self.guard.resolve(recipient)
        if not text.strip():
            return OutboundResult.failure(OutboundKind.REJECTED, "empty_text")
        if len(text) > TEXT_CHUNK_LIMIT:
            return OutboundResult.failure(OutboundKind.REJECTED, "text_too_long")
        return self._accepted("text", text=text, quote=quote)

    async def send_image(
        self,
        data: bytes | Path,
        mime: str,
        *,
        recipient: str | None = None,
        bypass: SendBypass | None = None,
    ) -> OutboundResult:
        self.guard.resolve(recipient)
        raw = data if isinstance(data, bytes) else await asyncio.to_thread(Path(data).read_bytes)
        canonical = IMAGE_MIMES.get(mime.lower())
        if canonical is None:
            raise MediaNotAllowed(f"images of type {mime!r} are not sent")
        if sniff_image_mime(raw) != canonical:
            raise MediaNotAllowed("the bytes are not an image of the declared type")
        digest = await asyncio.to_thread(ensure_media_allowed, self._media_policy, raw)
        return self._accepted("image", sha256=digest, mime=canonical)

    async def send_typing(self, active: bool, *, recipient: str | None = None) -> None:
        self.guard.resolve(recipient)
        self.typing = active

    # -------------------------------------------------------- introspection

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            supports_quote=self._supports_quote,
            supports_typing=True,
            gif_animated=None,
            proactive_window_h=None,
            outbound_quota=QUOTA,
            max_text_chars=TEXT_CHUNK_LIMIT,
        )

    def session_state(self) -> SessionState:
        return SessionState(
            auth=AuthState.OK,
            bound=True,
            last_inbound_at=self._last_inbound_at,
            outbound_since_inbound=self._outbound_since_inbound,
            expired=False,
            remaining_quota=QUOTA,
            window_remaining=None,
            has_context_token=True,
            extra={"in_memory": True, "typing": self.typing},
        )

    def clear(self) -> None:
        """Forget what was sent (the sandbox reads the log after every reply)."""
        self.sent.clear()
