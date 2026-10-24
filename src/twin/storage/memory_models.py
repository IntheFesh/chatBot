"""Tables of the memory (round 07; R-STO-006, R-MEM-002/003/005/006/010).

``facts``
    what the bot knows, one row per fact (R-MEM-003).  The text and the evidence references are
    sealed; subject, category, source, status and the dates are plain closed vocabularies and
    timestamps.  ``known_at`` is the earliest moment the fact could have been known (the time
    of the latest message that proves it, R-MEM-010); ``valid_from`` / ``valid_to`` bound the
    time it is true.  A fact replaced by a newer or better one keeps its row: ``superseded_by``
    names the replacement and ``superseded_at`` the moment it was replaced.  A candidate that a
    higher-priority source vetoed is kept with ``status = 'rejected'`` and ``rejected_by`` (it is
    never shown).  ``number`` is the number the user sees in ``/记忆`` and gives to ``/忘掉``.
    ``embedding_id`` / ``embed_version`` tie the fact to its row in the vector table.
``daily_summaries``
    the summary of one local day of one scope (``real`` records or the ``bot`` conversation,
    R-MEM-002).  A recomputed day gets a new ``version``; ``is_current`` marks the one in force.
    ``utc_start`` / ``utc_end`` are the bounds of the local day in the zone it was cut in.
``lifeline_events``
    the made-up life of the bot (R-MEM-005): what she did, where, in which mood.  Rows come from
    the daily plan (``plan``, round 08) or from details the bot let slip in conversation
    (``improvised``); all of them are bot inventions, so a real fact that contradicts one marks
    it ``invalidated`` (R-MEM-011).
``followups``
    things the user said that are worth asking about later (R-MEM-006).  ``created_at`` is the
    moment the commitment became known (for a replayed real record: the time of the message),
    ``closed_at`` the moment it stopped being open, so the state at any past moment can be read.
Each of the first four has ``rev``, a version counter the ORM raises on every update: the
in-memory snapshot of :mod:`twin.memory.corpus` reloads a row whose revision moved, which does not
depend on the clock.

``memory_replay_days``
    which local days of the real history the replay has processed, and from which messages
    (a hash of their ids and texts), so an interrupted replay resumes and a day whose messages
    changed is done again (R-MEM-010).  No text is stored.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
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

SUBJECTS = ("her", "user", "both", "other")
CATEGORIES = (
    "life",
    "preference",
    "plan",
    "anniversary",
    "nickname",
    "relation",
    "work_study",
    "other",
)
SOURCES = ("real_record", "user_said", "bot_invented", "user_command")
FACT_STATUSES = ("active", "rejected")
RECURRENCES = ("none", "yearly", "monthly")
SUMMARY_SCOPES = ("real", "bot")
LIFELINE_SOURCES = ("plan", "improvised")
LIFELINE_STATUSES = ("active", "invalidated")
FOLLOWUP_STATUSES = ("open", "done", "cancelled", "expired")
FOLLOWUP_ORIGINS = ("real_record", "bot_session", "user_command")

# R-MEM-004: a real record outranks what the user said, which outranks what the bot invented;
# the user's own command (/记住) is a deliberate correction and outranks everything.
SOURCE_PRIORITY = {"bot_invented": 1, "user_said": 2, "real_record": 3, "user_command": 4}
# sources that exist only because the bot conversation exists (R-MEM-010: empty before it started)
BOT_CONVERSATION_SOURCES = frozenset({"user_said", "bot_invented", "user_command"})


def _in_list(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class Fact(TimestampMixin, Base):
    """One fact the bot remembers (R-MEM-003)."""

    __tablename__ = "facts"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    rev: Mapped[int] = mapped_column(Integer, nullable=False)
    __mapper_args__ = {"version_id_col": rev}  # noqa: RUF012 - read by SQLAlchemy as it is
    number: Mapped[int] = mapped_column(Integer, nullable=False)
    subject: Mapped[str] = mapped_column(String(8), nullable=False)
    category: Mapped[str] = mapped_column(String(16), nullable=False)
    text_ct: Mapped[SealedBlob] = encrypted_column("text")
    text = sealed_text()
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(8), nullable=False, default="active")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.8)
    importance: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    known_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    valid_from: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    valid_to: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    event_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    recurrence: Mapped[str] = mapped_column(String(8), nullable=False, default="none")
    superseded_by: Mapped[str | None] = mapped_column(
        String(26), ForeignKey("facts.id", ondelete="SET NULL"), nullable=True
    )
    superseded_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    rejected_by: Mapped[str | None] = mapped_column(
        String(26), ForeignKey("facts.id", ondelete="SET NULL"), nullable=True
    )
    evidence_ct: Mapped[SealedBlob | None] = encrypted_column("evidence", json=True, nullable=True)
    evidence = sealed_json(optional=True)
    embedding_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    embed_version: Mapped[str | None] = mapped_column(String(200), nullable=True)

    __table_args__ = (
        CheckConstraint(_in_list("subject", SUBJECTS), name="subject"),
        CheckConstraint(_in_list("category", CATEGORIES), name="category"),
        CheckConstraint(_in_list("source", SOURCES), name="source"),
        CheckConstraint(_in_list("status", FACT_STATUSES), name="status"),
        CheckConstraint(_in_list("recurrence", RECURRENCES), name="recurrence"),
        CheckConstraint("importance >= 1 AND importance <= 5", name="importance"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence"),
        UniqueConstraint("number", name="uq_facts_number"),
        Index("ix_facts_known_at", "known_at"),
        Index("ix_facts_status_source", "status", "source"),
        Index("ix_facts_event_date", "event_date"),
    )


class DailySummary(TimestampMixin, Base):
    """The summary of one local day, per scope, versioned (R-MEM-002)."""

    __tablename__ = "daily_summaries"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    rev: Mapped[int] = mapped_column(Integer, nullable=False)
    __mapper_args__ = {"version_id_col": rev}  # noqa: RUF012 - read by SQLAlchemy as it is
    scope: Mapped[str] = mapped_column(String(8), nullable=False)
    local_date: Mapped[date] = mapped_column(Date, nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False)
    utc_start: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    utc_end: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    text_ct: Mapped[SealedBlob] = encrypted_column("text")
    text = sealed_text()
    embedding_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    embed_version: Mapped[str | None] = mapped_column(String(200), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    input_hash: Mapped[str | None] = mapped_column(String(40), nullable=True)
    template_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    __table_args__ = (
        CheckConstraint(_in_list("scope", SUMMARY_SCOPES), name="scope"),
        UniqueConstraint("scope", "local_date", "version", name="uq_daily_summaries_day_version"),
        Index("ix_daily_summaries_scope_date_current", "scope", "local_date", "is_current"),
    )


class LifelineEvent(TimestampMixin, Base):
    """A made-up event of the bot's day (R-MEM-005)."""

    __tablename__ = "lifeline_events"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    rev: Mapped[int] = mapped_column(Integer, nullable=False)
    __mapper_args__ = {"version_id_col": rev}  # noqa: RUF012 - read by SQLAlchemy as it is
    local_date: Mapped[date] = mapped_column(Date, nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False)
    start_local: Mapped[str | None] = mapped_column(String(5), nullable=True)
    end_local: Mapped[str | None] = mapped_column(String(5), nullable=True)
    activity_ct: Mapped[SealedBlob] = encrypted_column("activity")
    activity = sealed_text()
    place_ct: Mapped[SealedBlob | None] = encrypted_column("place", nullable=True)
    place = sealed_optional_text()
    mood_ct: Mapped[SealedBlob | None] = encrypted_column("mood", nullable=True)
    mood = sealed_optional_text()
    detail_ct: Mapped[SealedBlob | None] = encrypted_column("detail", nullable=True)
    detail = sealed_optional_text()
    source: Mapped[str] = mapped_column(String(10), nullable=False, default="plan")
    status: Mapped[str] = mapped_column(String(12), nullable=False, default="active")
    consistency_checked_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    invalidated_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    invalidated_by: Mapped[str | None] = mapped_column(String(26), nullable=True)
    fact_id: Mapped[str | None] = mapped_column(String(26), nullable=True)
    shared_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    shared_reply_id: Mapped[str | None] = mapped_column(String(26), nullable=True)

    __table_args__ = (
        CheckConstraint(_in_list("source", LIFELINE_SOURCES), name="source"),
        CheckConstraint(_in_list("status", LIFELINE_STATUSES), name="status"),
        Index("ix_lifeline_events_local_date", "local_date"),
        Index("ix_lifeline_events_fact_id", "fact_id"),
    )


class Followup(TimestampMixin, Base):
    """Something to ask about later (R-MEM-006)."""

    __tablename__ = "followups"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    rev: Mapped[int] = mapped_column(Integer, nullable=False)
    __mapper_args__ = {"version_id_col": rev}  # noqa: RUF012 - read by SQLAlchemy as it is
    text_ct: Mapped[SealedBlob] = encrypted_column("text")
    text = sealed_text()
    due_at_utc: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    window_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=240)
    source_turn_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(10), nullable=False, default="open")
    closed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    close_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    origin: Mapped[str] = mapped_column(String(12), nullable=False, default="bot_session")
    evidence_ct: Mapped[SealedBlob | None] = encrypted_column("evidence", json=True, nullable=True)
    evidence = sealed_json(optional=True)
    fact_id: Mapped[str | None] = mapped_column(String(26), nullable=True)

    __table_args__ = (
        CheckConstraint(_in_list("status", FOLLOWUP_STATUSES), name="status"),
        CheckConstraint(_in_list("origin", FOLLOWUP_ORIGINS), name="origin"),
        CheckConstraint("window_minutes >= 0", name="window_minutes"),
        Index("ix_followups_status_due", "status", "due_at_utc"),
        Index("ix_followups_fact_id", "fact_id"),
    )


class MemoryReplayDay(TimestampMixin, Base):
    """One local day of the real history the replay has processed (R-MEM-010)."""

    __tablename__ = "memory_replay_days"

    local_date: Mapped[date] = mapped_column(Date, primary_key=True)
    input_hash: Mapped[str] = mapped_column(String(40), nullable=False)
    lines: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    facts_added: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    followups_added: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    summary_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    batch_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    replayed_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
