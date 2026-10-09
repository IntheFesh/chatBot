"""From an export message to a normalised record (R-IMP-007, R-IMP-009, R-SCOPE-002).

``renderType`` selects the :class:`~twin.ingest.events.Kind`; the numeric WeChat ``type`` is
used only when ``renderType`` is missing or unknown.  For each kind the "main text" is the
one field that carries the words (the content of a text, the transcript of a voice message,
the title of a link, ...), so event texts can be rendered later from the stored columns
alone.  The original message object is kept whole (``raw``).

``isSent`` decides who spoke: false is her, true is the user (R-SCOPE-002).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from twin.ingest.events import CallStatus, Kind
from twin.ingest.layout import truthy
from twin.ingest.schema import KNOWN_MESSAGE_KEYS, QUOTE_KEYS, ExportMessage
from twin.ingest.times import SourceTime, epoch_to_utc

RENDER_TYPE_KINDS: dict[str, Kind] = {
    "text": Kind.TEXT,
    "emoji": Kind.STICKER,
    "sticker": Kind.STICKER,
    "quote": Kind.QUOTE,
    "image": Kind.IMAGE,
    "voice": Kind.VOICE,
    "voip": Kind.CALL,
    "system": Kind.SYSTEM,
    "transfer": Kind.TRANSFER,
    "redpacket": Kind.REDPACKET,
    "link": Kind.LINK,
    "video": Kind.VIDEO,
    "file": Kind.FILE,
    "location": Kind.LOCATION,
    "chathistory": Kind.CHATHISTORY,
}
# WeChat's numeric message types, used when renderType says nothing usable
TYPE_CODE_KINDS: dict[int, Kind] = {
    1: Kind.TEXT,
    3: Kind.IMAGE,
    34: Kind.VOICE,
    43: Kind.VIDEO,
    47: Kind.STICKER,
    48: Kind.LOCATION,
    50: Kind.CALL,
    10000: Kind.SYSTEM,
    10002: Kind.SYSTEM,
}

_DURATION = re.compile(r"(\d{1,4}):(\d{2})(?::(\d{2}))?")
MAX_SECONDS = 24 * 3600


class InvalidMessage(ValueError):
    """The message cannot be imported; ``reason`` is a short code (never message content)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class MediaRequest:
    """One file a message refers to."""

    kind: str  # image | video_cover | voice | avatar | sticker | skip
    path: str | None
    md5: str | None
    file_id: str | None
    index: int
    declared_kind: str | None = None


@dataclass(slots=True)
class NormalizedMessage:
    id: str
    local_id: str | None
    server_id: str | None
    create_time: datetime
    sort_seq: int | None
    is_sent: bool
    kind: str
    render_type: str | None
    text: str | None
    raw: dict[str, Any]
    sticker_md5: str | None = None
    sticker_url: str | None = None
    quote: dict[str, Any] | None = None
    call_status: str | None = None
    call_duration_s: int | None = None
    voice_seconds: int | None = None
    has_transcript: bool = False
    media: list[MediaRequest] = field(default_factory=list)
    sender_avatar_path: str | None = None
    time_mismatch: bool = False
    render_type_known: bool = True
    unknown_fields: frozenset[str] = frozenset()


@dataclass(frozen=True)
class NormalizeContext:
    """What normalisation needs to know about the conversation."""

    conversation_username: str
    source_time: SourceTime


# --------------------------------------------------------------------- helpers


def as_text(value: Any) -> str | None:
    """A non-empty string, or ``None`` for anything else."""
    if isinstance(value, str) and value.strip():
        return value
    return None


def as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value == value and abs(value) < 2**62 else None
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError:
            try:
                number = float(text)
            except ValueError:
                return None
            return int(number) if number == number and abs(number) < 2**62 else None
    return None


def as_id(value: Any) -> str | None:
    """An identifier as text (numbers keep their digits, no exponent form)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value == int(value) else str(value)
    if isinstance(value, str):
        return value.strip() or None
    return None


def parse_voice_seconds(value: Any) -> int | None:
    """Seconds from ``voiceLength`` (milliseconds, usually a string); tolerant of junk."""
    milliseconds = as_int(value)
    if milliseconds is None or milliseconds <= 0:
        return None
    return min(MAX_SECONDS, max(1, round(milliseconds / 1000)))


def parse_call(content: str | None, voip_type: Any = None) -> tuple[str, int | None]:
    """``(call status, duration in seconds)`` from a call message (R-IMP-007)."""
    text = content or ""
    match = _DURATION.search(text)
    if match is not None:
        first, second, third = match.group(1), match.group(2), match.group(3)
        if third is None:
            seconds = int(first) * 60 + int(second)
        else:
            seconds = int(first) * 3600 + int(second) * 60 + int(third)
        if seconds > 0:
            return CallStatus.CONNECTED.value, min(seconds, MAX_SECONDS)
    if "其它设备" in text or "其他设备" in text:
        return CallStatus.OTHER_DEVICE.value, None
    if "取消" in text:
        return CallStatus.CANCELLED.value, None
    if "拒绝" in text:
        return CallStatus.REJECTED.value, None
    if "未应答" in text or "未接" in text or "无应答" in text:
        return CallStatus.MISSED.value, None
    return CallStatus.UNKNOWN.value, None


def kind_of(render_type: str | None, type_code: Any) -> tuple[Kind, bool]:
    """``(kind, recognised)`` for a message."""
    if render_type:
        key = render_type.casefold().replace("_", "").replace("-", "")
        kind = RENDER_TYPE_KINDS.get(key)
        if kind is not None:
            return kind, True
    code = as_int(type_code)
    if code is not None:
        by_code = TYPE_CODE_KINDS.get(code)
        if by_code is not None:
            return by_code, not render_type
    return Kind.UNKNOWN, not render_type


def classify_media(declared: str | None, parent: Kind) -> str:
    """The ``media_assets`` kind of an ``offlineMedia`` entry, or ``skip``."""
    name = (declared or "").casefold().replace("_", "").replace("-", "")
    if not name:
        return {Kind.IMAGE: "image", Kind.VOICE: "voice", Kind.STICKER: "sticker"}.get(
            parent, "skip"
        )
    if "avatar" in name or "head" in name:
        return "avatar"
    if "emoji" in name or "sticker" in name or "emotion" in name:
        return "sticker"
    if "thumb" in name or "cover" in name or "poster" in name:
        return "video_cover" if parent is Kind.VIDEO else "image"
    if "voice" in name or "audio" in name:
        return "voice"
    if "video" in name or "file" in name or "document" in name:
        return "skip"
    if "image" in name or "img" in name or "photo" in name or "pic" in name:
        return "image"
    return "skip"


def _media_requests(message: ExportMessage, parent: Kind) -> list[MediaRequest]:
    entries = message.offlineMedia
    if not entries:
        return []
    requests = [
        MediaRequest(
            classify_media(entry.kind, parent),
            entry.path,
            entry.md5,
            entry.fileId,
            index,
            entry.kind,
        )
        for index, entry in enumerate(entries)
    ]
    if parent is Kind.IMAGE:
        # a message with a full image does not need its thumbnail as well
        full = [r for r in requests if r.kind == "image" and "thumb" not in _lower(r.declared_kind)]
        if full:
            requests = [
                r if r.kind != "image" or r in full else _skipped(r, "thumbnail") for r in requests
            ]
    return requests


def _lower(value: str | None) -> str:
    return (value or "").casefold()


def _skipped(request: MediaRequest, _why: str) -> MediaRequest:
    return MediaRequest("skip", request.path, request.md5, request.file_id, request.index)


def _main_text(kind: Kind, message: ExportMessage) -> str | None:
    if kind in (Kind.TEXT, Kind.SYSTEM, Kind.CALL):
        return as_text(message.content)
    if kind is Kind.QUOTE:
        return as_text(message.content) or as_text(message.title)
    if kind is Kind.VOICE:
        return as_text(message.voiceTranscript)
    if kind in (Kind.LINK, Kind.FILE):
        return as_text(message.title) or as_text(message.content)
    if kind is Kind.LOCATION:
        return (
            as_text(message.locationPoiname)
            or as_text(message.locationLabel)
            or as_text(message.content)
        )
    return None


def _quote_of(message: ExportMessage) -> dict[str, Any] | None:
    found = {key: getattr(message, key) for key in QUOTE_KEYS if getattr(message, key) is not None}
    return found or None


# ------------------------------------------------------------------ main entry


def normalize_message(
    raw: dict[str, Any], message: ExportMessage, context: NormalizeContext
) -> NormalizedMessage:
    """Normalise one export message; raises :class:`InvalidMessage` if it cannot be stored."""
    local_id = as_id(message.localId)
    message_id = as_id(message.id)
    if message_id is None:
        if local_id is None:
            raise InvalidMessage("no_id")
        message_id = f"{context.conversation_username}:{local_id}"

    created = epoch_to_utc(message.createTime)
    mismatch = False
    if created is None:
        created = context.source_time.parse_local_text(message.createTimeText)
        if created is None:
            raise InvalidMessage("no_time")
    elif message.createTimeText:
        mismatch = context.source_time.disagrees(created, message.createTimeText)

    if message.isSent is not None:
        is_sent = truthy(message.isSent)
    elif message.senderUsername:
        is_sent = message.senderUsername != context.conversation_username
    else:
        raise InvalidMessage("no_sender")

    kind, recognised = kind_of(message.renderType, message.type)
    result = NormalizedMessage(
        id=message_id,
        local_id=local_id,
        server_id=as_id(message.serverId),
        create_time=created,
        sort_seq=as_int(message.sortSeq),
        is_sent=is_sent,
        kind=kind.value,
        render_type=message.renderType,
        text=_main_text(kind, message),
        raw=raw,
        quote=_quote_of(message),
        media=_media_requests(message, kind),
        sender_avatar_path=as_text(message.senderAvatarPath),
        time_mismatch=mismatch,
        render_type_known=recognised,
        unknown_fields=frozenset(raw.keys() - KNOWN_MESSAGE_KEYS),
    )
    if kind is Kind.STICKER:
        result.sticker_md5 = _md5_of(message.emojiMd5)
        result.sticker_url = as_text(message.emojiUrl)
    elif kind is Kind.VOICE:
        result.voice_seconds = parse_voice_seconds(message.voiceLength)
        result.has_transcript = result.text is not None
    elif kind is Kind.CALL:
        result.call_status, result.call_duration_s = parse_call(message.content, message.voipType)
    return result


def _md5_of(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().lower()
    return text if re.fullmatch(r"[0-9a-f]{32}", text) else None
