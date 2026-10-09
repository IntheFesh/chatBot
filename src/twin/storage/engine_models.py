"""Tables of the reply engine (round 09; R-STO-006, R-STO-007, R-ENG-001, R-ENG-011).

``bot_turns``
    the bot's own conversation, one row per message: every message the user sent to the bot
    (``direction = 'in'``) and every bubble the bot sent back (``'out'``), command replies
    included (``is_command``).  It is a different table from ``messages`` on purpose (R-STO-007):
    nothing the bot says can reach the style samples, the retrieval library or the training set,
    because those read ``messages`` through :mod:`twin.ingest.corpus` and never this table.
    The text (the user's words as they stand in the conversation, or the bubble), the media
    references, the plan and the per-stage numbers are sealed.  What belongs to a whole reply -
    backend, thinking, plan, cost, timings, post-processing actions - is stored once, on the first
    bubble of the reply (``bubble_index = 0``); the bubbles of a reply share ``reply_id``.  A
    reply the bot chose not to give is one ``no_reply`` row.  ``rejected_at`` marks a reply the
    user threw away (``/重来``); it stays on disk but no longer counts as conversation.
``conversation_state``
    the state machine of the one conversation (R-ENG-001): the state, the user messages waiting
    to be answered, the bubbles already sent, when the next one is due, and where the window of
    the recent conversation starts (R-MEM-001).  The engine writes it at every transition so a
    restart can go on from where it was.
``feedback``
    what the user said about a reply: ``redo`` (``/重来``) or ``not_like`` (``/不像``, optionally
    with the right wording).  Round 11 turns these into preference pairs.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, Float, ForeignKey, Index, Integer, String
from sqlalchemy import text as sql_text
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

DIRECTIONS = ("in", "out")
TURN_KINDS = ("text", "image", "voice", "video", "file", "sticker", "unknown", "no_reply")
BACKENDS = ("deepseek", "style", "hybrid", "fallback", "command", "safety")
STATES = ("IDLE", "COLLECTING", "DECIDING", "GENERATING", "SENDING")
FEEDBACK_TYPES = ("redo", "not_like")
MAIN_CONVERSATION = "main"


def _in_list(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class BotTurn(TimestampMixin, Base):
    """One message of the bot's conversation (R-ENG-011)."""

    __tablename__ = "bot_turns"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    direction: Mapped[str] = mapped_column(String(3), nullable=False)
    kind: Mapped[str] = mapped_column(String(10), nullable=False)
    text_ct: Mapped[SealedBlob] = encrypted_column("text")
    text = sealed_text()
    media_ct: Mapped[SealedBlob | None] = encrypted_column("media", json=True, nullable=True)
    media = sealed_json(optional=True)
    sticker_md5: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_command: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    external_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    reply_id: Mapped[str | None] = mapped_column(String(26), nullable=True)
    bubble_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    backend: Mapped[str | None] = mapped_column(String(10), nullable=True)
    thinking: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    plan_ct: Mapped[SealedBlob | None] = encrypted_column("plan", json=True, nullable=True)
    plan = sealed_json(optional=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    timings_ct: Mapped[SealedBlob | None] = encrypted_column("timings", json=True, nullable=True)
    timings = sealed_json(optional=True)
    actions_ct: Mapped[SealedBlob | None] = encrypted_column("actions", json=True, nullable=True)
    actions = sealed_json(optional=True)
    rejected_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    __table_args__ = (
        CheckConstraint(_in_list("direction", DIRECTIONS), name="direction"),
        CheckConstraint(_in_list("kind", TURN_KINDS), name="kind"),
        CheckConstraint("backend IS NULL OR " + _in_list("backend", BACKENDS), name="backend"),
        CheckConstraint("bubble_index IS NULL OR bubble_index >= 0", name="bubble_index"),
        Index("ix_bot_turns_at", "at"),
        Index("ix_bot_turns_direction_at", "direction", "at"),
        Index("ix_bot_turns_reply_id", "reply_id"),
        Index(
            "uq_bot_turns_external",
            "direction",
            "external_id",
            unique=True,
            sqlite_where=sql_text("external_id IS NOT NULL"),
        ),
    )


class ConversationState(TimestampMixin, Base):
    """The state of the conversation, persisted at every transition (R-ENG-001)."""

    __tablename__ = "conversation_state"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    state: Mapped[str] = mapped_column(String(10), nullable=False, default="IDLE")
    state_since: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    round_id: Mapped[str | None] = mapped_column(String(26), nullable=True)
    pending_ct: Mapped[SealedBlob | None] = encrypted_column("pending", json=True, nullable=True)
    pending = sealed_json(optional=True)
    sent_ct: Mapped[SealedBlob | None] = encrypted_column("sent", json=True, nullable=True)
    sent = sealed_json(optional=True)
    data_ct: Mapped[SealedBlob | None] = encrypted_column("data", json=True, nullable=True)
    data = sealed_json(optional=True)
    planned_send_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    window_start_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    last_inbound_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    last_outbound_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    extracted_through: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    __table_args__ = (CheckConstraint(_in_list("state", STATES), name="state"),)


class Feedback(TimestampMixin, Base):
    """What the user said about a reply (``/重来``, ``/不像``)."""

    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    type: Mapped[str] = mapped_column(String(10), nullable=False)
    reply_id: Mapped[str] = mapped_column(String(26), nullable=False)
    bot_turn_id: Mapped[str | None] = mapped_column(
        String(26), ForeignKey("bot_turns.id", ondelete="SET NULL"), nullable=True
    )
    correction_ct: Mapped[SealedBlob | None] = encrypted_column("correction", nullable=True)
    correction = sealed_optional_text()
    processed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    __table_args__ = (
        CheckConstraint(_in_list("type", FEEDBACK_TYPES), name="type"),
        Index("ix_feedback_reply_id", "reply_id"),
        Index("ix_feedback_type_processed", "type", "processed_at"),
    )
