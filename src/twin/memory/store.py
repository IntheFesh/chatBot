"""Reading and writing the memory tables (round 07; R-STO-002, R-STO-006).

:class:`MemoryStore` is the only code that touches ``facts``, ``daily_summaries``,
``lifeline_events``, ``followups`` and ``memory_replay_days``.  It returns the plain records of
:mod:`twin.memory.records`; the rows themselves (and their sealed text) never leave this module.
Every method is one short transaction.  Nothing here knows about models, vectors or the
priority rules: those live in the modules above (:mod:`twin.memory.conflict`,
:mod:`twin.memory.writer`).

Two settings are kept here as well: the moment the bot's conversation began
(``memory.bot_online_at``, set by whoever first writes something of the bot's own) and nothing
else; counters and caches are in memory only.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import delete, func, select
from sqlalchemy.engine import CursorResult

from twin.memory.records import FactRecord, FollowupRecord, LifelineRecord, SummaryRecord
from twin.storage.memory_models import (
    DailySummary,
    Fact,
    Followup,
    LifelineEvent,
    MemoryReplayDay,
)
from twin.storage.settings_store import get_setting, put_setting

if TYPE_CHECKING:
    from twin.services import Services

BOT_ONLINE_KEY = "memory.bot_online_at"
SQLITE_VARIABLE_LIMIT = 500


@dataclass(frozen=True)
class NewFact:
    """The fields of a fact about to be stored."""

    subject: str
    category: str
    text: str
    source: str
    known_at: datetime
    confidence: float = 0.8
    importance: int = 3
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    event_date: date | None = None
    recurrence: str = "none"
    evidence: dict[str, Any] | None = None
    status: str = "active"
    superseded_by: str | None = None
    superseded_at: datetime | None = None
    rejected_by: str | None = None


@dataclass(frozen=True)
class NewFollowup:
    text: str
    due_at: datetime
    window_minutes: int
    created_at: datetime
    origin: str = "bot_session"
    source_turn_id: str | None = None
    evidence: dict[str, Any] | None = None
    fact_id: str | None = None


@dataclass(frozen=True)
class NewEvent:
    local_date: date
    timezone: str
    activity: str
    source: str = "plan"
    start_local: str | None = None
    end_local: str | None = None
    place: str | None = None
    mood: str | None = None
    detail: str | None = None
    fact_id: str | None = None


@dataclass(frozen=True)
class ReplayDayRecord:
    local_date: date
    input_hash: str
    lines: int
    facts_added: int
    followups_added: int
    summary_version: int | None
    batch_id: str | None
    replayed_at: datetime


@dataclass(frozen=True)
class StoreSignature:
    """Cheap fingerprint of the tables: changes whenever any memory row changes."""

    parts: tuple[tuple[str, int, int, str], ...] = field(default_factory=tuple)


def _chunks[T](items: Sequence[T], size: int = SQLITE_VARIABLE_LIMIT) -> Iterable[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class MemoryStore:
    """Database access of the memory (see the module description)."""

    def __init__(self, services: Services) -> None:
        self._services = services
        self._db = services.db

    # ------------------------------------------------------------------ facts

    def add_fact(self, new: NewFact, *, at: datetime | None = None) -> FactRecord:
        """Store a fact and give it the next number the user can name in ``/忘掉``."""
        now = at or self._services.clock.now_utc()
        with self._db.transaction() as session:
            number = int(session.scalar(select(func.coalesce(func.max(Fact.number), 0))) or 0) + 1
            row = Fact(
                number=number,
                subject=new.subject,
                category=new.category,
                text=new.text,
                source=new.source,
                status=new.status,
                confidence=new.confidence,
                importance=new.importance,
                known_at=new.known_at,
                valid_from=new.valid_from,
                valid_to=new.valid_to,
                event_date=new.event_date,
                recurrence=new.recurrence,
                superseded_by=new.superseded_by,
                superseded_at=new.superseded_at,
                rejected_by=new.rejected_by,
                evidence=new.evidence,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            session.flush()
            return FactRecord.from_row(row)

    def fact(self, fact_id: str) -> FactRecord | None:
        with self._db.session() as session:
            row = session.get(Fact, fact_id)
            return FactRecord.from_row(row) if row else None

    def facts(self, ids: Sequence[str]) -> list[FactRecord]:
        found: list[FactRecord] = []
        with self._db.session() as session:
            for chunk in _chunks(list(ids)):
                found.extend(
                    FactRecord.from_row(row)
                    for row in session.scalars(select(Fact).where(Fact.id.in_(list(chunk))))
                )
        return found

    def all_facts(self) -> list[FactRecord]:
        with self._db.session() as session:
            stmt = (
                select(Fact).order_by(Fact.known_at, Fact.number).execution_options(yield_per=2000)
            )
            return [FactRecord.from_row(row) for row in session.scalars(stmt)]

    def update_fact(self, fact_id: str, **changes: Any) -> FactRecord:
        """Change columns of a fact (``text`` and ``evidence`` are sealed again)."""
        with self._db.transaction() as session:
            row = session.get(Fact, fact_id)
            if row is None:
                raise KeyError(f"no fact {fact_id}")
            for name, value in changes.items():
                if not hasattr(Fact, name):
                    raise AttributeError(f"facts have no column {name!r}")
                setattr(row, name, value)
            session.flush()
            return FactRecord.from_row(row)

    def supersede_fact(self, old_id: str, new_id_: str, *, at: datetime) -> FactRecord:
        """Mark ``old_id`` as replaced by ``new_id_`` from the moment ``at`` on."""
        with self._db.transaction() as session:
            row = session.get(Fact, old_id)
            if row is None:
                raise KeyError(f"no fact {old_id}")
            row.superseded_by = new_id_
            row.superseded_at = at
            if row.valid_to is None or row.valid_to > at:
                row.valid_to = at
            session.flush()
            return FactRecord.from_row(row)

    def delete_facts(self, ids: Sequence[str]) -> list[str]:
        """Remove facts for good; facts they had replaced become current again.

        Returns the ids of the facts that were restored.  Facts that a deleted fact had vetoed
        stay rejected (they remain a record of what was refused) but lose the link.
        """
        restored: list[str] = []
        with self._db.transaction() as session:
            wanted = set(ids)
            for chunk in _chunks(list(wanted)):
                for row in session.scalars(select(Fact).where(Fact.superseded_by.in_(list(chunk)))):
                    if row.id in wanted:
                        continue
                    if row.valid_to is not None and row.valid_to == row.superseded_at:
                        row.valid_to = None
                    row.superseded_by = None
                    row.superseded_at = None
                    restored.append(row.id)
                for row in session.scalars(select(Fact).where(Fact.rejected_by.in_(list(chunk)))):
                    row.rejected_by = None
            session.flush()
            for chunk in _chunks(list(wanted)):
                session.execute(delete(Fact).where(Fact.id.in_(list(chunk))))
        return restored

    def set_fact_embedding(self, updates: dict[str, tuple[str | None, str | None]]) -> None:
        """``{fact id: (embedding id, encoding)}``; ``(None, None)`` clears the link."""
        if not updates:
            return
        with self._db.transaction(bump_state=False) as session:
            for chunk in _chunks(list(updates)):
                for fact in session.scalars(select(Fact).where(Fact.id.in_(list(chunk)))):
                    fact.embedding_id, fact.embed_version = updates[fact.id]

    # -------------------------------------------------------------- summaries

    def add_summary(
        self,
        *,
        scope: str,
        local_date: date,
        timezone: str,
        utc_start: datetime,
        utc_end: datetime,
        text: str,
        input_hash: str | None,
        template_version: str | None,
    ) -> SummaryRecord:
        """Store a summary as the next version of its day; older versions stop being current."""
        now = self._services.clock.now_utc()
        with self._db.transaction() as session:
            latest = session.scalar(
                select(func.max(DailySummary.version)).where(
                    DailySummary.scope == scope, DailySummary.local_date == local_date
                )
            )
            for old in session.scalars(
                select(DailySummary).where(
                    DailySummary.scope == scope,
                    DailySummary.local_date == local_date,
                    DailySummary.is_current.is_(True),
                )
            ):
                old.is_current = False
            row = DailySummary(
                scope=scope,
                local_date=local_date,
                timezone=timezone,
                utc_start=utc_start,
                utc_end=utc_end,
                text=text,
                version=int(latest or 0) + 1,
                is_current=True,
                input_hash=input_hash,
                template_version=template_version,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            session.flush()
            return SummaryRecord.from_row(row)

    def current_summary(self, scope: str, local_date: date) -> SummaryRecord | None:
        with self._db.session() as session:
            row = session.scalars(
                select(DailySummary).where(
                    DailySummary.scope == scope,
                    DailySummary.local_date == local_date,
                    DailySummary.is_current.is_(True),
                )
            ).first()
            return SummaryRecord.from_row(row) if row else None

    def current_summaries(self) -> list[SummaryRecord]:
        with self._db.session() as session:
            rows = session.scalars(
                select(DailySummary)
                .where(DailySummary.is_current.is_(True))
                .order_by(DailySummary.local_date, DailySummary.scope)
            )
            return [SummaryRecord.from_row(row) for row in rows]

    def summary_versions(self, scope: str, local_date: date) -> list[SummaryRecord]:
        with self._db.session() as session:
            rows = session.scalars(
                select(DailySummary)
                .where(DailySummary.scope == scope, DailySummary.local_date == local_date)
                .order_by(DailySummary.version)
            )
            return [SummaryRecord.from_row(row) for row in rows]

    def set_summary_embedding(self, updates: dict[str, tuple[str | None, str | None]]) -> None:
        if not updates:
            return
        with self._db.transaction(bump_state=False) as session:
            for chunk in _chunks(list(updates)):
                for summary in session.scalars(
                    select(DailySummary).where(DailySummary.id.in_(list(chunk)))
                ):
                    summary.embedding_id, summary.embed_version = updates[summary.id]

    # -------------------------------------------------------------- lifeline

    def add_event(self, new: NewEvent, *, at: datetime | None = None) -> LifelineRecord:
        now = at or self._services.clock.now_utc()
        with self._db.transaction() as session:
            row = LifelineEvent(
                local_date=new.local_date,
                timezone=new.timezone,
                start_local=new.start_local,
                end_local=new.end_local,
                activity=new.activity,
                place=new.place,
                mood=new.mood,
                detail=new.detail,
                source=new.source,
                status="active",
                fact_id=new.fact_id,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            session.flush()
            return LifelineRecord.from_row(row)

    def event(self, event_id: str) -> LifelineRecord | None:
        with self._db.session() as session:
            row = session.get(LifelineEvent, event_id)
            return LifelineRecord.from_row(row) if row else None

    def events(
        self, *, day: date | None = None, include_invalid: bool = False
    ) -> list[LifelineRecord]:
        stmt = select(LifelineEvent).order_by(
            LifelineEvent.local_date, LifelineEvent.start_local, LifelineEvent.created_at
        )
        if day is not None:
            stmt = stmt.where(LifelineEvent.local_date == day)
        if not include_invalid:
            stmt = stmt.where(LifelineEvent.status == "active")
        with self._db.session() as session:
            return [LifelineRecord.from_row(row) for row in session.scalars(stmt)]

    def invalidate_event(self, event_id: str, *, by: str | None, at: datetime) -> bool:
        with self._db.transaction() as session:
            row = session.get(LifelineEvent, event_id)
            if row is None or row.status == "invalidated":
                return False
            row.status = "invalidated"
            row.invalidated_at = at
            row.invalidated_by = by
            session.flush()
            return True

    def delete_events(self, ids: Sequence[str]) -> int:
        removed = 0
        with self._db.transaction() as session:
            for chunk in _chunks(list(ids)):
                result = session.execute(
                    delete(LifelineEvent).where(LifelineEvent.id.in_(list(chunk)))
                )
                removed += int(cast("CursorResult[Any]", result).rowcount or 0)
        return removed

    def mark_events_shared(self, ids: Sequence[str], *, at: datetime, reply_id: str | None) -> int:
        """Note that a proactive message told the user about these entries (round 10).

        An entry that was told before keeps its first mark: throwing the later message away
        (``/重来``) must not make her forget that an earlier one told it.
        """
        marked = 0
        with self._db.transaction() as session:
            for chunk in _chunks(list(ids)):
                for row in session.scalars(
                    select(LifelineEvent).where(
                        LifelineEvent.id.in_(list(chunk)), LifelineEvent.shared_at.is_(None)
                    )
                ):
                    row.shared_at = at
                    row.shared_reply_id = reply_id
                    marked += 1
        return marked

    def unshare_events(self, ids: Sequence[str]) -> int:
        """Take the "already told" mark off these entries (``/重来`` threw the message away)."""
        cleared = 0
        with self._db.transaction() as session:
            for chunk in _chunks(list(ids)):
                for row in session.scalars(
                    select(LifelineEvent).where(
                        LifelineEvent.id.in_(list(chunk)), LifelineEvent.shared_at.is_not(None)
                    )
                ):
                    row.shared_at = None
                    row.shared_reply_id = None
                    cleared += 1
        return cleared

    def stamp_events_checked(self, ids: Sequence[str], at: datetime) -> None:
        with self._db.transaction() as session:
            for chunk in _chunks(list(ids)):
                for row in session.scalars(
                    select(LifelineEvent).where(LifelineEvent.id.in_(list(chunk)))
                ):
                    row.consistency_checked_at = at

    def events_of_fact(self, fact_ids: Sequence[str]) -> list[LifelineRecord]:
        found: list[LifelineRecord] = []
        with self._db.session() as session:
            for chunk in _chunks(list(fact_ids)):
                found.extend(
                    LifelineRecord.from_row(row)
                    for row in session.scalars(
                        select(LifelineEvent).where(LifelineEvent.fact_id.in_(list(chunk)))
                    )
                )
        return found

    # ------------------------------------------------------------- followups

    def add_followup(self, new: NewFollowup) -> FollowupRecord:
        with self._db.transaction() as session:
            row = Followup(
                text=new.text,
                due_at_utc=new.due_at,
                window_minutes=new.window_minutes,
                source_turn_id=new.source_turn_id,
                status="open",
                origin=new.origin,
                evidence=new.evidence,
                fact_id=new.fact_id,
                created_at=new.created_at,
                updated_at=self._services.clock.now_utc(),
            )
            session.add(row)
            session.flush()
            return FollowupRecord.from_row(row)

    def followup(self, followup_id: str) -> FollowupRecord | None:
        with self._db.session() as session:
            row = session.get(Followup, followup_id)
            return FollowupRecord.from_row(row) if row else None

    def followups(self, *, only_open: bool = False) -> list[FollowupRecord]:
        stmt = select(Followup).order_by(Followup.due_at_utc, Followup.created_at)
        if only_open:
            stmt = stmt.where(Followup.status == "open")
        with self._db.session() as session:
            return [FollowupRecord.from_row(row) for row in session.scalars(stmt)]

    def close_followup(
        self, followup_id: str, *, status: str, at: datetime, reason: str | None
    ) -> FollowupRecord | None:
        """Close an open follow-up (``done`` / ``cancelled`` / ``expired``) as of ``at``."""
        if status == "open":
            raise ValueError("a follow-up is closed with done, cancelled or expired")
        with self._db.transaction() as session:
            row = session.get(Followup, followup_id)
            if row is None or row.status != "open":
                return None
            row.status = status
            row.closed_at = at
            row.close_reason = reason
            session.flush()
            return FollowupRecord.from_row(row)

    def delete_followups(self, ids: Sequence[str]) -> int:
        removed = 0
        with self._db.transaction() as session:
            for chunk in _chunks(list(ids)):
                result = session.execute(delete(Followup).where(Followup.id.in_(list(chunk))))
                removed += int(cast("CursorResult[Any]", result).rowcount or 0)
        return removed

    def followups_of_fact(self, fact_ids: Sequence[str]) -> list[FollowupRecord]:
        found: list[FollowupRecord] = []
        with self._db.session() as session:
            for chunk in _chunks(list(fact_ids)):
                found.extend(
                    FollowupRecord.from_row(row)
                    for row in session.scalars(
                        select(Followup).where(Followup.fact_id.in_(list(chunk)))
                    )
                )
        return found

    # ---------------------------------------------------------- replay days

    def replay_days(self) -> dict[date, ReplayDayRecord]:
        with self._db.session() as session:
            return {
                row.local_date: ReplayDayRecord(
                    row.local_date,
                    row.input_hash,
                    row.lines,
                    row.facts_added,
                    row.followups_added,
                    row.summary_version,
                    row.batch_id,
                    row.replayed_at,
                )
                for row in session.scalars(select(MemoryReplayDay))
            }

    def mark_replayed(
        self,
        local_date: date,
        *,
        input_hash: str,
        lines: int,
        facts_added: int,
        followups_added: int,
        summary_version: int | None,
        batch_id: str | None,
    ) -> None:
        now = self._services.clock.now_utc()
        with self._db.transaction() as session:
            row = session.get(MemoryReplayDay, local_date)
            if row is None:
                row = MemoryReplayDay(
                    local_date=local_date,
                    input_hash=input_hash,
                    lines=lines,
                    facts_added=facts_added,
                    followups_added=followups_added,
                    summary_version=summary_version,
                    batch_id=batch_id,
                    replayed_at=now,
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
            else:
                row.input_hash = input_hash
                row.lines = lines
                row.facts_added = facts_added
                row.followups_added = followups_added
                row.summary_version = summary_version
                row.batch_id = batch_id
                row.replayed_at = now

    # ---------------------------------------------------------------- bot era

    def bot_online_at(self) -> datetime | None:
        """When the bot's conversation began (``None`` while nothing of it exists)."""
        with self._db.session() as session:
            raw = get_setting(session, BOT_ONLINE_KEY)
        return datetime.fromisoformat(raw) if isinstance(raw, str) else None

    def mark_bot_online(self, at: datetime) -> bool:
        """Record the start of the bot's conversation; an earlier recorded start is kept."""
        current = self.bot_online_at()
        if current is not None and current <= at:
            return False
        with self._db.transaction(bump_state=False) as session:
            return put_setting(
                session,
                BOT_ONLINE_KEY,
                at.isoformat(),
                clock=self._services.clock,
                by="memory",
                record_history=False,
            )

    # --------------------------------------------------- change detection (corpus)

    def fact_stamps(self) -> dict[str, int]:
        """``{id: revision}`` of the facts that can be shown (not vetoed); no text is read."""
        with self._db.session() as session:
            rows = session.execute(select(Fact.id, Fact.rev).where(Fact.status == "active"))
            return {str(row_id): int(rev) for row_id, rev in rows}

    def summary_stamps(self) -> dict[str, int]:
        """``{id: revision}`` of the current summaries."""
        with self._db.session() as session:
            rows = session.execute(
                select(DailySummary.id, DailySummary.rev).where(DailySummary.is_current.is_(True))
            )
            return {str(row_id): int(rev) for row_id, rev in rows}

    def followup_stamps(self) -> dict[str, int]:
        with self._db.session() as session:
            rows = session.execute(select(Followup.id, Followup.rev))
            return {str(row_id): int(rev) for row_id, rev in rows}

    def event_stamps(self) -> dict[str, int]:
        with self._db.session() as session:
            rows = session.execute(select(LifelineEvent.id, LifelineEvent.rev))
            return {str(row_id): int(rev) for row_id, rev in rows}

    def summaries(self, ids: Sequence[str]) -> list[SummaryRecord]:
        found: list[SummaryRecord] = []
        with self._db.session() as session:
            for chunk in _chunks(list(ids)):
                found.extend(
                    SummaryRecord.from_row(row)
                    for row in session.scalars(
                        select(DailySummary).where(DailySummary.id.in_(list(chunk)))
                    )
                )
        return found

    def followups_by_ids(self, ids: Sequence[str]) -> list[FollowupRecord]:
        found: list[FollowupRecord] = []
        with self._db.session() as session:
            for chunk in _chunks(list(ids)):
                found.extend(
                    FollowupRecord.from_row(row)
                    for row in session.scalars(select(Followup).where(Followup.id.in_(list(chunk))))
                )
        return found

    def events_by_ids(self, ids: Sequence[str]) -> list[LifelineRecord]:
        found: list[LifelineRecord] = []
        with self._db.session() as session:
            for chunk in _chunks(list(ids)):
                found.extend(
                    LifelineRecord.from_row(row)
                    for row in session.scalars(
                        select(LifelineEvent).where(LifelineEvent.id.in_(list(chunk)))
                    )
                )
        return found

    # -------------------------------------------------------------- bookkeeping

    def signature(self) -> StoreSignature:
        """Row count, sum of the row revisions and newest id of each table.

        Every change of a row raises its revision (SQLAlchemy's version counter), so any insert,
        update or delete moves this fingerprint - whatever the clock says.
        """
        parts: list[tuple[str, int, int, str]] = []
        with self._db.session() as session:
            for model in (Fact, DailySummary, LifelineEvent, Followup):
                count, total, newest = session.execute(
                    select(func.count(), func.coalesce(func.sum(model.rev), 0), func.max(model.id))
                ).one()
                parts.append((model.__tablename__, int(count), int(total), str(newest or "")))
        return StoreSignature(tuple(parts))

    def counts(self) -> dict[str, int]:
        with self._db.session() as session:
            return {
                model.__tablename__: int(
                    session.scalar(select(func.count()).select_from(model)) or 0
                )
                for model in (Fact, DailySummary, LifelineEvent, Followup, MemoryReplayDay)
            }

    def fact_counts_by_source(self) -> dict[str, int]:
        with self._db.session() as session:
            rows = session.execute(
                select(Fact.source, func.count())
                .where(Fact.status == "active", Fact.superseded_by.is_(None))
                .group_by(Fact.source)
            )
            return {str(source): int(count) for source, count in rows}
