"""Tables of the chat-record import (round 03; R-STO-006, R-STO-007).

``conversations``, ``messages``, ``media_assets``, ``stickers``, ``sticker_uses`` and
``import_runs``.  They are defined in their own module and registered on the same
:class:`~twin.storage.models.Base` (``twin.storage.models`` imports this module last), so
migrations, key rotation and the schema checks see one metadata.

``messages`` holds only what an export contained.  Everything the bot itself says goes to
``bot_turns`` (a different table, round 09): a row of ``messages`` always names the export
it came from (``source_export_id`` is NOT NULL), so a bot message cannot be stored there by
accident, and style / retrieval / training queries read ``messages`` with ``is_sent`` false
only (R-STO-007).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from twin.storage.crypto import SealedBlob
from twin.storage.models import Base, TimestampMixin
from twin.storage.types import (
    UTCDateTime,
    encrypted_column,
    sealed_json,
    sealed_optional_text,
    sealed_text,
)

MESSAGE_KINDS = (
    "text",
    "sticker",
    "image",
    "quote",
    "voice",
    "video",
    "file",
    "link",
    "call",
    "transfer",
    "redpacket",
    "system",
    "location",
    "chathistory",
    "unknown",
)
MEDIA_KINDS = ("image", "video_cover", "voice", "avatar")
MEDIA_STATUSES = ("pending", "available", "missing", "corrupt")
STICKER_STATUSES = ("pending", "available", "md5_mismatch", "unavailable")
IMPORT_STATUSES = ("queued", "running", "interrupted", "failed", "done", "superseded")
IMPORT_PHASES = ("queued", "discover", "messages", "media", "stickers", "finalize", "hooks", "done")


def _in_list(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class Conversation(TimestampMixin, Base):
    """A chat the import has seen (only the target conversation is imported)."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    username_ct: Mapped[SealedBlob] = encrypted_column("username")
    username = sealed_text()
    display_name_ct: Mapped[SealedBlob | None] = encrypted_column("display_name", nullable=True)
    display_name = sealed_optional_text()
    is_group: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    her_avatar_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_avatar_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_message_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    last_message_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    last_export_id: Mapped[str | None] = mapped_column(String(128), nullable=True)


class ImportRun(TimestampMixin, Base):
    """One import (R-IMP-006): resumable state, counters and the post-import hooks."""

    __tablename__ = "import_runs"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    export_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    source_dir_ct: Mapped[SealedBlob] = encrypted_column("source_dir")
    source_dir = sealed_text()
    target_username_ct: Mapped[SealedBlob] = encrypted_column("target_username")
    target_username = sealed_text()
    conversation_id: Mapped[str | None] = mapped_column(
        String(26), ForeignKey("conversations.id"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    phase: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    processed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    inserted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duplicates: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    conflict_updated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    conflict_kept: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    invalid: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_sort_seq: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    last_message_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    speed_per_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    stats_ct: Mapped[SealedBlob] = encrypted_column("stats", json=True)
    stats = sealed_json()
    hooks_ct: Mapped[SealedBlob | None] = encrypted_column("hooks", json=True, nullable=True)
    hooks = sealed_json(optional=True)

    __table_args__ = (
        CheckConstraint(_in_list("status", IMPORT_STATUSES), name="status"),
        CheckConstraint(_in_list("phase", IMPORT_PHASES), name="phase"),
        Index("ix_import_runs_status", "status"),
        Index("ix_import_runs_fingerprint", "fingerprint"),
    )


class Message(TimestampMixin, Base):
    """A real message from an export (R-IMP-004, R-IMP-007)."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(26), ForeignKey("conversations.id"), nullable=False
    )
    create_time_utc: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    sort_seq: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    local_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    server_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_sent: Mapped[bool] = mapped_column(Boolean, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    render_type: Mapped[str | None] = mapped_column(String(48), nullable=True)
    text_ct: Mapped[SealedBlob | None] = encrypted_column("text", nullable=True)
    text = sealed_optional_text()
    raw_ct: Mapped[SealedBlob] = encrypted_column("raw", json=True)
    raw = sealed_json()
    sticker_md5: Mapped[str | None] = mapped_column(String(32), nullable=True)
    media_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    quote_ct: Mapped[SealedBlob | None] = encrypted_column("quote", json=True, nullable=True)
    quote = sealed_json(optional=True)
    call_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    call_duration_s: Mapped[int | None] = mapped_column(Integer, nullable=True)
    voice_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    has_transcript: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    source_export_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_exported_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    __table_args__ = (
        CheckConstraint(_in_list("kind", MESSAGE_KINDS), name="kind"),
        Index("ix_messages_conversation_time", "conversation_id", "create_time_utc"),
        Index("ix_messages_sent_kind", "is_sent", "kind"),
        Index("ix_messages_sticker_md5", "sticker_md5"),
    )


class MediaAsset(TimestampMixin, Base):
    """A media file belonging to a message (or an avatar), present or missing (R-IMP-008)."""

    __tablename__ = "media_assets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(26), ForeignKey("conversations.id"), nullable=False
    )
    message_id: Mapped[str | None] = mapped_column(
        String(128), ForeignKey("messages.id"), nullable=True
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    reason: Mapped[str | None] = mapped_column(String(48), nullable=True)
    orig_md5: Mapped[str | None] = mapped_column(String(64), nullable=True)
    file_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_path_ct: Mapped[SealedBlob | None] = encrypted_column("source_path", nullable=True)
    source_path = sealed_optional_text()
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    mime: Mapped[str | None] = mapped_column(String(32), nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    caption_ct: Mapped[SealedBlob | None] = encrypted_column("caption", nullable=True)
    caption = sealed_optional_text()
    caption_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    __table_args__ = (
        CheckConstraint(_in_list("kind", MEDIA_KINDS), name="kind"),
        CheckConstraint(_in_list("status", MEDIA_STATUSES), name="status"),
        Index("ix_media_assets_status", "status"),
        Index("ix_media_assets_sha256", "sha256"),
        Index("ix_media_assets_message_id", "message_id"),
        Index("ix_media_assets_kind_status", "kind", "status"),
    )


class Sticker(TimestampMixin, Base):
    """A sticker (animated or still picture) known by its MD5 (R-IMP-008, R-STK-002)."""

    __tablename__ = "stickers"

    md5: Mapped[str] = mapped_column(String(32), primary_key=True)
    url_ct: Mapped[SealedBlob | None] = encrypted_column("url", nullable=True)
    url = sealed_optional_text()
    source_path_ct: Mapped[SealedBlob | None] = encrypted_column("source_path", nullable=True)
    source_path = sealed_optional_text()
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    reason: Mapped[str | None] = mapped_column(String(48), nullable=True)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    mime: Mapped[str | None] = mapped_column(String(32), nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_attempt_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    her_uses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    user_uses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_used_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    __table_args__ = (
        CheckConstraint(_in_list("status", STICKER_STATUSES), name="status"),
        Index("ix_stickers_status", "status"),
        Index("ix_stickers_sha256", "sha256"),
    )


class StickerUse(TimestampMixin, Base):
    """One use of a sticker: who, when, in which message (R-IMP-008)."""

    __tablename__ = "sticker_uses"

    message_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("messages.id"), primary_key=True
    )
    sticker_md5: Mapped[str] = mapped_column(String(32), ForeignKey("stickers.md5"), nullable=False)
    conversation_id: Mapped[str] = mapped_column(
        String(26), ForeignKey("conversations.id"), nullable=False
    )
    by_her: Mapped[bool] = mapped_column(Boolean, nullable=False)
    used_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)

    __table_args__ = (
        Index("ix_sticker_uses_sticker_md5", "sticker_md5"),
        Index("ix_sticker_uses_used_at", "used_at"),
    )
