"""Reading and writing ``proactive_candidates``, ``proactive_log`` and ``ratings`` (R-PRO-008).

Three small repositories; the scheduler, the sender, the audit and the commands go through them
and nothing else touches the tables.  Every write is one short transaction (``BEGIN IMMEDIATE``
through :meth:`~twin.storage.db.Database.transaction`), so a process that stops anywhere leaves
the books consistent: a message is written to the log the moment its first bubble is out and
completed when the last one is.

Views are plain frozen records with the sealed columns already decrypted; ``with_text=False``
leaves the planner's reason and the content out (the CLI shows no text unless asked).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from twin.clock import Clock, ensure_aware
from twin.schedule.events import CandidatesExpired
from twin.schedule.proactive.types import Candidate, Reason, TriggerKind
from twin.storage.db import Database
from twin.storage.proactive_models import ProactiveCandidate, ProactiveLog, Rating

# ------------------------------------------------------------------------- candidates


@dataclass(frozen=True)
class CandidateRow:
    """A row of ``proactive_candidates``."""

    id: str
    local_date: date
    timezone: str
    key: str
    kind: TriggerKind
    priority: int
    planned_at: datetime
    window_end: datetime
    status: str
    status_at: datetime | None
    last_reason: str | None
    attempts: int
    plan_id: str | None
    detail: Mapping[str, Any]

    def candidate(self) -> Candidate:
        return Candidate(
            self.kind,
            self.key,
            self.planned_at,
            self.window_end,
            id=self.id,
            attempts=self.attempts,
            detail=self.detail,
        )


@dataclass(frozen=True)
class NewCandidate:
    """A slot to fill: what the scheduler makes of her day or of a follow-up."""

    local_date: date
    timezone: str
    key: str
    kind: TriggerKind
    priority: int
    planned_at: datetime
    window_end: datetime
    plan_id: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)


def _candidate_row(row: ProactiveCandidate) -> CandidateRow:
    return CandidateRow(
        id=row.id,
        local_date=row.local_date,
        timezone=row.timezone,
        key=row.key,
        kind=TriggerKind(row.kind),
        priority=row.priority,
        planned_at=row.planned_at,
        window_end=row.window_end,
        status=row.status,
        status_at=row.status_at,
        last_reason=row.last_reason,
        attempts=row.attempts,
        plan_id=row.plan_id,
        detail=dict(row.detail or {}),
    )


class CandidateStore:
    """The candidates that wait (see the module description)."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def add(self, new: NewCandidate) -> CandidateRow | None:
        """Store a slot; ``None`` if the day already has one under this key (made once per day)."""
        try:
            with self._db.transaction(bump_state=False) as session:
                exists = session.scalar(
                    select(ProactiveCandidate.id).where(
                        ProactiveCandidate.local_date == new.local_date,
                        ProactiveCandidate.timezone == new.timezone,
                        ProactiveCandidate.key == new.key,
                    )
                )
                if exists is not None:
                    return None
                row = ProactiveCandidate(
                    local_date=new.local_date,
                    timezone=new.timezone,
                    key=new.key,
                    kind=new.kind.value,
                    priority=new.priority,
                    planned_at=ensure_aware(new.planned_at),
                    window_end=ensure_aware(new.window_end),
                    status="pending",
                    attempts=0,
                    plan_id=new.plan_id,
                    detail=dict(new.detail) or None,
                )
                session.add(row)
                session.flush()
                return _candidate_row(row)
        except IntegrityError:  # another process made the same slot a moment earlier
            return None

    def has_key(self, key: str) -> bool:
        """Whether any day has a candidate under ``key`` (a follow-up is considered once)."""
        with self._db.session() as session:
            found = session.scalar(
                select(ProactiveCandidate.id).where(ProactiveCandidate.key == key).limit(1)
            )
            return found is not None

    def get(self, candidate_id: str) -> CandidateRow | None:
        with self._db.session() as session:
            row = session.get(ProactiveCandidate, candidate_id)
            return _candidate_row(row) if row is not None else None

    def pending(self) -> list[CandidateRow]:
        """Every candidate that still waits, the earliest first."""
        stmt = (
            select(ProactiveCandidate)
            .where(ProactiveCandidate.status == "pending")
            .order_by(ProactiveCandidate.planned_at, ProactiveCandidate.priority)
        )
        with self._db.session() as session:
            return [_candidate_row(row) for row in session.scalars(stmt)]

    def for_day(self, local_date: date, timezone: str) -> list[CandidateRow]:
        stmt = (
            select(ProactiveCandidate)
            .where(
                ProactiveCandidate.local_date == local_date,
                ProactiveCandidate.timezone == timezone,
            )
            .order_by(ProactiveCandidate.planned_at)
        )
        with self._db.session() as session:
            return [_candidate_row(row) for row in session.scalars(stmt)]

    def mark(
        self, candidate_id: str, status: str, *, at: datetime, reason: Reason | None = None
    ) -> None:
        """Move a candidate to a final status (``sent``, ``expired``, ...)."""
        with self._db.transaction(bump_state=False) as session:
            row = session.get(ProactiveCandidate, candidate_id)
            if row is None:
                return
            row.status = status
            row.status_at = ensure_aware(at)
            if reason is not None:
                row.last_reason = reason.value

    def note_reason(self, candidate_id: str, reason: Reason) -> None:
        """Remember why the candidate was refused last (the log is written once per reason)."""
        with self._db.transaction(bump_state=False) as session:
            row = session.get(ProactiveCandidate, candidate_id)
            if row is not None:
                row.last_reason = reason.value

    def reschedule(self, candidate_id: str, planned_at: datetime) -> None:
        """Try again later: the new time, one more attempt, no refusal remembered."""
        with self._db.transaction(bump_state=False) as session:
            row = session.get(ProactiveCandidate, candidate_id)
            if row is not None:
                row.planned_at = ensure_aware(planned_at)
                row.attempts += 1
                row.last_reason = None

    def expire(self, *, before: datetime, at: datetime) -> list[CandidateRow]:
        """Void the pending candidates whose own time window ended before ``before``."""
        return self._void(lambda row: row.window_end < before, at)

    def expire_covered(self, event: CandidatesExpired) -> list[CandidateRow]:
        """Void the pending candidates a schedule event covers (restart, wake-up, zone switch)."""
        return self._void(lambda row: event.covers(row.planned_at), event.at)

    def _void(
        self, covered: Callable[[ProactiveCandidate], bool], at: datetime
    ) -> list[CandidateRow]:
        voided: list[CandidateRow] = []
        with self._db.transaction(bump_state=False) as session:
            rows = session.scalars(
                select(ProactiveCandidate).where(ProactiveCandidate.status == "pending")
            )
            for row in rows:
                if covered(row):
                    row.status = "expired"
                    row.status_at = ensure_aware(at)
                    voided.append(_candidate_row(row))
        return voided


# ----------------------------------------------------------------------------- the log


@dataclass(frozen=True)
class LogEntry:
    """A row of ``proactive_log`` (the sealed columns decrypted, or ``None``)."""

    id: str
    at: datetime
    candidate_at: datetime
    local_date: date
    local_at: str
    timezone: str
    kind: str
    outcome: str
    reason: str | None
    her_state: str | None
    chase_seq: int
    bubbles_sent: int
    quota_total: int | None
    range_min: int | None
    range_max: int | None
    enabled: bool | None
    candidate_id: str | None
    followup_id: str | None
    reply_id: str | None
    backend: str | None
    cost_usd: float | None
    plan_reason: str | None
    content: Mapping[str, Any] | None
    result: Mapping[str, Any] | None


@dataclass(frozen=True)
class NewLog:
    """What the scheduler writes for a decision."""

    at: datetime
    candidate_at: datetime
    local_date: date
    local_at: str
    timezone: str
    kind: str
    outcome: str
    reason: str | None = None
    her_state: str | None = None
    chase_seq: int = 0
    bubbles_sent: int = 0
    quota_total: int | None = None
    range_min: int | None = None
    range_max: int | None = None
    enabled: bool | None = None
    candidate_id: str | None = None
    followup_id: str | None = None
    reply_id: str | None = None
    backend: str | None = None
    cost_usd: float | None = None
    plan_reason: str | None = None
    content: Mapping[str, Any] | None = None
    result: Mapping[str, Any] | None = None


def _log_entry(row: ProactiveLog, *, with_text: bool = True) -> LogEntry:
    return LogEntry(
        id=row.id,
        at=row.at,
        candidate_at=row.candidate_at,
        local_date=row.local_date,
        local_at=row.local_at,
        timezone=row.timezone,
        kind=row.kind,
        outcome=row.outcome,
        reason=row.reason,
        her_state=row.her_state,
        chase_seq=row.chase_seq,
        bubbles_sent=row.bubbles_sent,
        quota_total=row.quota_total,
        range_min=row.range_min,
        range_max=row.range_max,
        enabled=row.enabled,
        candidate_id=row.candidate_id,
        followup_id=row.followup_id,
        reply_id=row.reply_id,
        backend=row.backend,
        cost_usd=row.cost_usd,
        plan_reason=row.plan_reason if with_text else None,
        content=dict(row.content) if with_text and row.content else None,
        result=dict(row.result) if row.result else None,
    )


class ProactiveLogStore:
    """The audit trail (see the module description)."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def add(self, new: NewLog) -> LogEntry:
        with self._db.transaction(bump_state=False) as session:
            row = ProactiveLog(
                at=ensure_aware(new.at),
                candidate_at=ensure_aware(new.candidate_at),
                local_date=new.local_date,
                local_at=new.local_at,
                timezone=new.timezone,
                kind=new.kind,
                outcome=new.outcome,
                reason=new.reason,
                her_state=new.her_state,
                chase_seq=new.chase_seq,
                bubbles_sent=new.bubbles_sent,
                quota_total=new.quota_total,
                range_min=new.range_min,
                range_max=new.range_max,
                enabled=new.enabled,
                candidate_id=new.candidate_id,
                followup_id=new.followup_id,
                reply_id=new.reply_id,
                backend=new.backend,
                cost_usd=new.cost_usd,
                plan_reason=new.plan_reason,
                content=dict(new.content) if new.content is not None else None,
                result=dict(new.result) if new.result is not None else None,
            )
            session.add(row)
            session.flush()
            return _log_entry(row)

    def update(
        self,
        log_id: str,
        *,
        bubbles_sent: int | None = None,
        reply_id: str | None = None,
        content: Mapping[str, Any] | None = None,
        result: Mapping[str, Any] | None = None,
        backend: str | None = None,
        cost_usd: float | None = None,
    ) -> None:
        """Complete a row that was written when the first bubble went out."""
        with self._db.transaction(bump_state=False) as session:
            row = session.get(ProactiveLog, log_id)
            if row is None:
                return
            if bubbles_sent is not None:
                row.bubbles_sent = bubbles_sent
            if reply_id is not None:
                row.reply_id = reply_id
            if content is not None:
                row.content = dict(content)
            if result is not None:
                row.result = dict(result)
            if backend is not None:
                row.backend = backend
            if cost_usd is not None:
                row.cost_usd = cost_usd

    def get(self, log_id: str) -> LogEntry | None:
        with self._db.session() as session:
            row = session.get(ProactiveLog, log_id)
            return _log_entry(row) if row is not None else None

    # ---------------------------------------------------------------------- counts

    def count_sent_on(self, local_date: date) -> int:
        stmt = select(func.count()).where(
            ProactiveLog.outcome == "sent", ProactiveLog.local_date == local_date
        )
        with self._db.session() as session:
            return int(session.scalar(stmt) or 0)

    def count_sent_since(self, since: datetime, *, state: str | None = None) -> int:
        stmt = select(func.count()).where(
            ProactiveLog.outcome == "sent", ProactiveLog.at >= ensure_aware(since)
        )
        if state is not None:
            stmt = stmt.where(ProactiveLog.her_state == state)
        with self._db.session() as session:
            return int(session.scalar(stmt) or 0)

    def last_sent(self) -> LogEntry | None:
        stmt = (
            select(ProactiveLog)
            .where(ProactiveLog.outcome == "sent")
            .order_by(ProactiveLog.at.desc(), ProactiveLog.id.desc())
            .limit(1)
        )
        with self._db.session() as session:
            row = session.scalars(stmt).first()
            return _log_entry(row, with_text=False) if row is not None else None

    def last_sent_before(self, moment: datetime) -> LogEntry | None:
        """The last message that went out before ``moment``."""
        stmt = (
            select(ProactiveLog)
            .where(ProactiveLog.outcome == "sent", ProactiveLog.at < ensure_aware(moment))
            .order_by(ProactiveLog.at.desc(), ProactiveLog.id.desc())
            .limit(1)
        )
        with self._db.session() as session:
            row = session.scalars(stmt).first()
            return _log_entry(row, with_text=False) if row is not None else None

    def sent_since(self, since: datetime) -> list[LogEntry]:
        """The messages sent at or after ``since``, oldest first."""
        stmt = (
            select(ProactiveLog)
            .where(ProactiveLog.outcome == "sent", ProactiveLog.at >= ensure_aware(since))
            .order_by(ProactiveLog.at, ProactiveLog.id)
        )
        with self._db.session() as session:
            return [_log_entry(row, with_text=False) for row in session.scalars(stmt)]

    def opened(self, local_date: date, timezone: str) -> LogEntry | None:
        stmt = (
            select(ProactiveLog)
            .where(
                ProactiveLog.outcome == "opened",
                ProactiveLog.local_date == local_date,
                ProactiveLog.timezone == timezone,
            )
            .limit(1)
        )
        with self._db.session() as session:
            row = session.scalars(stmt).first()
            return _log_entry(row, with_text=False) if row is not None else None

    def last_rejection(self, kind: str, since: datetime) -> LogEntry | None:
        stmt = (
            select(ProactiveLog)
            .where(
                ProactiveLog.outcome == "rejected",
                ProactiveLog.kind == kind,
                ProactiveLog.at >= ensure_aware(since),
            )
            .order_by(ProactiveLog.at.desc())
            .limit(1)
        )
        with self._db.session() as session:
            row = session.scalars(stmt).first()
            return _log_entry(row, with_text=False) if row is not None else None

    # --------------------------------------------------------------------- reading

    def entries(
        self,
        *,
        first_day: date | None = None,
        last_day: date | None = None,
        since: datetime | None = None,
        outcomes: Iterable[str] | None = None,
        with_text: bool = False,
        limit: int | None = None,
    ) -> list[LogEntry]:
        """Rows of the log, oldest first, between two local days (both included)."""
        stmt = select(ProactiveLog).order_by(ProactiveLog.at, ProactiveLog.id)
        if first_day is not None:
            stmt = stmt.where(ProactiveLog.local_date >= first_day)
        if last_day is not None:
            stmt = stmt.where(ProactiveLog.local_date <= last_day)
        if since is not None:
            stmt = stmt.where(ProactiveLog.at >= ensure_aware(since))
        if outcomes is not None:
            stmt = stmt.where(ProactiveLog.outcome.in_(list(outcomes)))
        if limit is not None:
            stmt = stmt.limit(limit)
        with self._db.session() as session:
            return [_log_entry(row, with_text=with_text) for row in session.scalars(stmt)]


# ---------------------------------------------------------------------------- ratings


@dataclass(frozen=True)
class RatingRow:
    id: str
    at: datetime
    local_date: date
    score: int
    note: str | None


class RatingStore:
    """What ``/评分`` writes (R-EVAL-005)."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def add(self, score: int, note: str | None, *, at: datetime, local_date: date) -> RatingRow:
        if not 1 <= score <= 5:
            raise ValueError("a score is 1 to 5")
        with self._db.transaction(bump_state=False) as session:
            row = Rating(
                at=ensure_aware(at),
                local_date=local_date,
                score=score,
                source="command",
                note=note or None,
            )
            session.add(row)
            session.flush()
            return RatingRow(row.id, row.at, row.local_date, row.score, row.note)

    def between(self, start: datetime, end: datetime) -> list[RatingRow]:
        """Ratings with ``start <= at < end``, oldest first."""
        stmt = (
            select(Rating)
            .where(Rating.at >= ensure_aware(start), Rating.at < ensure_aware(end))
            .order_by(Rating.at, Rating.id)
        )
        with self._db.session() as session:
            return [
                RatingRow(r.id, r.at, r.local_date, r.score, r.note) for r in session.scalars(stmt)
            ]

    def recent(self, limit: int = 20) -> list[RatingRow]:
        stmt = select(Rating).order_by(Rating.at.desc(), Rating.id.desc()).limit(limit)
        with self._db.session() as session:
            return [
                RatingRow(r.id, r.at, r.local_date, r.score, r.note) for r in session.scalars(stmt)
            ]


def average(ratings: Sequence[RatingRow]) -> float | None:
    """The mean score, or ``None`` when there is none."""
    if not ratings:
        return None
    return sum(r.score for r in ratings) / len(ratings)
