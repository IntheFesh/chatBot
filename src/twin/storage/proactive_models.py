"""Tables of the proactive messages (round 10; R-STO-006, R-PRO-008, R-EVAL-005).

``proactive_candidates``
    the messages the scheduler has decided to consider: the ones whose time is fixed by her day
    (the wake-up greeting, a meal, goodnight) and the follow-ups that came due.  A candidate waits
    here until it is sent, dropped or void; ``planned_at`` is when it wants to go and
    ``window_end`` the last moment it is still worth sending.  Candidates that come from the random
    draw of a tick (silence, sharing, the edge of sleep) are decided at once and only appear in the
    log.  ``key`` names the slot of the day a candidate fills (``greeting``, ``meal:lunch``,
    ``bedtime``, ``followup:<id>``), so a slot is made once per local day and zone, whatever happens
    to the day plan.
``proactive_log``
    one row per decision, the audit trail that ``twin proactive log`` and ``twin eval proactive``
    read: a message that went out (``sent``), one the hard constraints refused (``rejected``,
    ``reason`` is the closed code), one the planner chose not to send (``declined``), one that
    could not be made or sent (``failed``), one that came too late (``expired``) or was dropped
    (``dropped``), and, once per local day, the marker that the scheduler was running with the
    range of that day (``opened``).  The plain columns carry everything that is counted - the local
    date, her state, the chase number, the range and quota in force; the planner's reason, the
    bubbles and the channel's result are sealed.
``ratings``
    what ``/评分`` writes: a score of 1 to 5 with an optional sealed note.

Nothing here is read by the style samples, the retrieval library or the training set.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    Float,
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
)

CANDIDATE_KINDS = ("followup", "greeting", "meal", "bedtime", "silence", "share", "edge")
CANDIDATE_STATUSES = ("pending", "sent", "expired", "dropped", "declined", "superseded")
LOG_KINDS = (*CANDIDATE_KINDS, "day")
LOG_OUTCOMES = ("sent", "rejected", "declined", "failed", "expired", "dropped", "opened")
RATING_SOURCES = ("command",)


def _in_list(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class ProactiveCandidate(TimestampMixin, Base):
    """A message waiting to be considered (see the module description)."""

    __tablename__ = "proactive_candidates"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    local_date: Mapped[date] = mapped_column(Date, nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False)
    key: Mapped[str] = mapped_column(String(48), nullable=False)
    kind: Mapped[str] = mapped_column(String(10), nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False)
    planned_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    window_end: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    status: Mapped[str] = mapped_column(String(10), nullable=False, default="pending")
    status_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    last_reason: Mapped[str | None] = mapped_column(String(24), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    plan_id: Mapped[str | None] = mapped_column(String(26), nullable=True)
    detail_ct: Mapped[SealedBlob | None] = encrypted_column("detail", json=True, nullable=True)
    detail = sealed_json(optional=True)

    __table_args__ = (
        CheckConstraint(_in_list("kind", CANDIDATE_KINDS), name="kind"),
        CheckConstraint(_in_list("status", CANDIDATE_STATUSES), name="status"),
        CheckConstraint("attempts >= 0", name="attempts"),
        UniqueConstraint("local_date", "timezone", "key", name="uq_proactive_candidates_slot"),
        Index("ix_proactive_candidates_status_planned", "status", "planned_at"),
    )


class ProactiveLog(TimestampMixin, Base):
    """One decision of the scheduler (see the module description)."""

    __tablename__ = "proactive_log"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    candidate_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    local_date: Mapped[date] = mapped_column(Date, nullable=False)
    local_at: Mapped[str] = mapped_column(String(16), nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(10), nullable=False)
    outcome: Mapped[str] = mapped_column(String(10), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(24), nullable=True)
    her_state: Mapped[str | None] = mapped_column(String(12), nullable=True)
    chase_seq: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bubbles_sent: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    quota_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    range_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    range_max: Mapped[int | None] = mapped_column(Integer, nullable=True)
    enabled: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    candidate_id: Mapped[str | None] = mapped_column(String(26), nullable=True)
    followup_id: Mapped[str | None] = mapped_column(String(26), nullable=True)
    reply_id: Mapped[str | None] = mapped_column(String(26), nullable=True)
    backend: Mapped[str | None] = mapped_column(String(10), nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    plan_reason_ct: Mapped[SealedBlob | None] = encrypted_column("plan_reason", nullable=True)
    plan_reason = sealed_optional_text()
    content_ct: Mapped[SealedBlob | None] = encrypted_column("content", json=True, nullable=True)
    content = sealed_json(optional=True)
    result_ct: Mapped[SealedBlob | None] = encrypted_column("result", json=True, nullable=True)
    result = sealed_json(optional=True)

    __table_args__ = (
        CheckConstraint(_in_list("kind", LOG_KINDS), name="kind"),
        CheckConstraint(_in_list("outcome", LOG_OUTCOMES), name="outcome"),
        CheckConstraint("chase_seq >= 0", name="chase_seq"),
        CheckConstraint("bubbles_sent >= 0", name="bubbles_sent"),
        Index("ix_proactive_log_local_date_outcome", "local_date", "outcome"),
        Index("ix_proactive_log_at", "at"),
    )


class Rating(TimestampMixin, Base):
    """A score the user gave with ``/评分`` (R-EVAL-005)."""

    __tablename__ = "ratings"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    local_date: Mapped[date] = mapped_column(Date, nullable=False)
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String(12), nullable=False, default="command")
    note_ct: Mapped[SealedBlob | None] = encrypted_column("note", nullable=True)
    note = sealed_optional_text()

    __table_args__ = (
        CheckConstraint("score >= 1 AND score <= 5", name="score"),
        CheckConstraint(_in_list("source", RATING_SOURCES), name="source"),
        Index("ix_ratings_at", "at"),
    )


__all__ = [
    "CANDIDATE_KINDS",
    "CANDIDATE_STATUSES",
    "LOG_KINDS",
    "LOG_OUTCOMES",
    "ProactiveCandidate",
    "ProactiveLog",
    "Rating",
]
