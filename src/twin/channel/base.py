"""The channel abstraction (R-CH-001, R-CH-005).

A :class:`Channel` is the only way the rest of the program talks to the user's chat
application.  The WeChat implementation is :class:`twin.channel.ilink.channel.IlinkChannel`;
the terminal implementation (round 02, step 02d) satisfies the same interface, so the engine
and the proactive scheduler never know which one they are driving.

Rules that every implementation follows (CLAUDE.md rule 6, R-CH-007, R-SAFE-004, R-SAFE-006):

* a channel only ever talks to the single bound user; a send call that names any other
  recipient raises :class:`RecipientNotAllowed`;
* only bytes whose SHA-256 is on the outbound media allow list can be sent as images;
  anything else raises :class:`MediaNotAllowed`;
* a capability the protocol lacks is reported by :meth:`Channel.capabilities` and a call that
  needs it raises :class:`CapabilityNotSupported`; nothing is ever silently ignored.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol

from twin.clock import ensure_aware
from twin.storage.media import MediaKind

# ----------------------------------------------------------------- inbound


class MessageKind(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    VOICE = "voice"
    VIDEO = "video"
    FILE = "file"
    UNKNOWN = "unknown"


FLAG_VOICE_UNTRANSCRIBED = "voice_untranscribed"
FLAG_VIDEO_NO_COVER = "video_no_cover"
FLAG_MEDIA_UNAVAILABLE = "media_unavailable"
FLAG_UNKNOWN_ITEM_TYPE = "unknown_item_type"


@dataclass(frozen=True)
class MediaRef:
    """A media file held in the encrypted :class:`~twin.storage.media.MediaStore`."""

    sha256: str
    kind: MediaKind
    size: int
    mime: str | None = None
    file_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "kind": self.kind.value,
            "size": self.size,
            "mime": self.mime,
            "file_name": self.file_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MediaRef:
        return cls(
            sha256=str(data["sha256"]),
            kind=MediaKind(data["kind"]),
            size=int(data["size"]),
            mime=data.get("mime"),
            file_name=data.get("file_name"),
        )


@dataclass(frozen=True)
class QuoteInfo:
    """What the user quoted when replying to an older message.

    ``resolved`` is false when only the server id of the quoted message arrived and the
    channel no longer holds its text.
    """

    text: str | None = None
    title: str | None = None
    svr_id: str | None = None
    media_ref: MediaRef | None = None
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "title": self.title,
            "svr_id": self.svr_id,
            "media_ref": self.media_ref.to_dict() if self.media_ref else None,
            "resolved": self.resolved,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuoteInfo:
        media = data.get("media_ref")
        return cls(
            text=data.get("text"),
            title=data.get("title"),
            svr_id=data.get("svr_id"),
            media_ref=MediaRef.from_dict(media) if media else None,
            resolved=bool(data.get("resolved", False)),
        )


@dataclass(frozen=True)
class InboundMessage:
    """One message from the user: ``{id, at, kind, text, media_ref, quote}`` (R-CH-005).

    ``text`` holds the words for text messages, the cloud transcription for voice messages
    (``None`` plus :data:`FLAG_VOICE_UNTRANSCRIBED` when there is none) and the file name for
    files.  ``item_type`` is the protocol's item type number (kept for diagnostics only).
    """

    id: str
    at: datetime
    kind: MessageKind
    text: str | None = None
    media_ref: MediaRef | None = None
    quote: QuoteInfo | None = None
    flags: frozenset[str] = frozenset()
    item_type: int | None = None

    def __post_init__(self) -> None:
        ensure_aware(self.at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "at": self.at.isoformat(),
            "kind": self.kind.value,
            "text": self.text,
            "media_ref": self.media_ref.to_dict() if self.media_ref else None,
            "quote": self.quote.to_dict() if self.quote else None,
            "flags": sorted(self.flags),
            "item_type": self.item_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InboundMessage:
        media = data.get("media_ref")
        quote = data.get("quote")
        return cls(
            id=str(data["id"]),
            at=datetime.fromisoformat(data["at"]),
            kind=MessageKind(data["kind"]),
            text=data.get("text"),
            media_ref=MediaRef.from_dict(media) if media else None,
            quote=QuoteInfo.from_dict(quote) if quote else None,
            flags=frozenset(data.get("flags", ())),
            item_type=data.get("item_type"),
        )


# ---------------------------------------------------------------- outbound


class OutboundKind(StrEnum):
    OK = "ok"
    WINDOW_REJECTED = "window_rejected"  # session/window/quota refusal (local gate or server)
    AUTH_EXPIRED = "auth_expired"  # the login is no longer valid
    REJECTED = "rejected"  # the server answered with another error
    NETWORK = "network"  # the request never left this machine
    AMBIGUOUS = "ambiguous"  # sent, but the outcome is unknown (read timeout, 5xx)
    UPLOAD_FAILED = "upload_failed"


@dataclass(frozen=True)
class OutboundResult:
    """Outcome of one send call: success flag, protocol error code, session-expired flag."""

    ok: bool
    kind: OutboundKind
    reason: str = ""
    code: int | None = None
    errmsg: str | None = None
    http_status: int | None = None
    session_expired: bool = False
    message_id: str | None = None
    client_id: str | None = None

    @classmethod
    def success(
        cls, *, message_id: str | None = None, client_id: str | None = None
    ) -> OutboundResult:
        return cls(True, OutboundKind.OK, message_id=message_id, client_id=client_id)

    @classmethod
    def failure(cls, kind: OutboundKind, reason: str, **fields: Any) -> OutboundResult:
        if kind is OutboundKind.OK:
            raise ValueError("a failure cannot have the kind OK")
        return cls(False, kind, reason, **fields)


@dataclass(frozen=True)
class QuoteTarget:
    """The message a reply quotes (only for channels that can send quotes)."""

    message_id: str
    text: str


@dataclass(frozen=True)
class BypassRequest:
    """What a send that wants to skip the safe window and quota thresholds is about to do."""

    kind: Literal["text", "image"]
    text: str | None
    now: datetime
    gate_reason: str | None  # why the safe thresholds would have refused this send (or None)


class SendBypass(Protocol):
    """Authorises a send that skips the safe thresholds (the M0 probe, R-CH-009).

    ``authorize`` returns normally to allow the send and raises :class:`BypassRefused` to
    forbid it; it also writes the audit record.  The normal engine and proactive paths have no
    object of this kind.
    """

    def authorize(self, request: BypassRequest) -> None: ...


# ------------------------------------------------------- capabilities, state


@dataclass(frozen=True)
class ChannelCapabilities:
    """What this channel can do.  ``None`` means "not measured yet" (M0 fills it in)."""

    supports_quote: bool
    supports_typing: bool
    gif_animated: bool | None = None
    proactive_window_h: float | None = None
    outbound_quota: int | None = None
    max_text_chars: int | None = None


class AuthState(StrEnum):
    NOT_LOGGED_IN = "not_logged_in"
    OK = "ok"
    NEEDS_RELOGIN = "needs_relogin"


@dataclass(frozen=True)
class SessionState:
    """A snapshot of the login and of the conversation window (R-CH-001, R-CH-008)."""

    auth: AuthState
    bound: bool
    last_inbound_at: datetime | None
    outbound_since_inbound: int
    expired: bool
    remaining_quota: int
    window_remaining: timedelta | None
    has_context_token: bool
    extra: dict[str, Any] = field(default_factory=dict)


# -------------------------------------------------------------- exceptions


class ChannelError(Exception):
    """Base class of channel errors."""


class RecipientNotAllowed(ChannelError):
    """A send named someone other than the bound user (R-CH-007, R-SAFE-004)."""


class MediaNotAllowed(ChannelError):
    """The image is not on the outbound allow list (R-SAFE-006)."""


class CapabilityNotSupported(ChannelError):
    """The channel (or protocol) cannot do what was asked."""


class BypassRefused(ChannelError):
    """A :class:`SendBypass` did not authorise the send."""


# ---------------------------------------------------------------- interface


class Channel(ABC):
    """The interface the engine, the scheduler and the diagnostics use."""

    @abstractmethod
    async def start(self) -> None:
        """Begin receiving (background tasks, connections)."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop receiving and release everything; safe to call twice."""

    @abstractmethod
    def incoming(self) -> AsyncIterator[InboundMessage]:
        """Messages from the bound user, oldest first, each delivered at least once.

        A message counts as handed over when the consumer asks for the next one; a consumer
        that stops earlier receives the message again after a restart.
        """

    @abstractmethod
    async def send_text(
        self,
        text: str,
        quote: QuoteTarget | None = None,
        *,
        recipient: str | None = None,
        bypass: SendBypass | None = None,
    ) -> OutboundResult:
        """Send one text bubble to the bound user."""

    @abstractmethod
    async def send_image(
        self,
        data: bytes | Path,
        mime: str,
        *,
        recipient: str | None = None,
        bypass: SendBypass | None = None,
    ) -> OutboundResult:
        """Send one image whose bytes are on the outbound allow list."""

    @abstractmethod
    async def send_typing(self, active: bool, *, recipient: str | None = None) -> None:
        """Show (or clear) "typing..." to the bound user, where the channel supports it."""

    @abstractmethod
    def capabilities(self) -> ChannelCapabilities:
        """Static and measured abilities of the channel."""

    @abstractmethod
    def session_state(self) -> SessionState:
        """Current login and window state."""
