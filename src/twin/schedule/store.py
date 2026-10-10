"""The schedule's tables and small settings: plans, time zone history, salt, last greeting.

``PlanStore`` is the only code that reads or writes ``daily_plans``: it turns rows into
:class:`~twin.schedule.plan_model.DailyPlan` and back, finds the plan that decides a moment and
marks the plans a newer plan takes over from (``supersede_from``).  ``TimezoneHistory`` lists the
switches of R-SCH-002.  ``InstallSalt`` is the random salt of the plan seeds, created on first use
and kept in the sealed ``settings`` table; ``GreetingLog`` remembers when the last wake-up
greeting went out (the 18-hour rule, R-SCH-002), which round 10 records with
:meth:`GreetingLog.record`.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from sqlalchemy import or_, select, update

from twin.clock import Clock, ensure_aware
from twin.schedule.plan_model import DailyPlan
from twin.storage.db import Database
from twin.storage.schedule_models import DailyPlanRow, TimezoneChange
from twin.storage.settings_store import get_setting, put_setting

SALT_KEY = "schedule.install_salt"
GREETING_KEY = "schedule.last_wake_greeting"
SALT_BYTES = 16


def _plan(row: DailyPlanRow) -> DailyPlan:
    return DailyPlan.from_json(
        row.plan,
        id=row.id,
        created_at=row.created_at,
        superseded_by=row.superseded_by,
        superseded_at=row.superseded_at,
        lifeline_job_id=row.lifeline_job_id,
        lifeline_done_at=row.lifeline_done_at,
        summary_queued_at=row.summary_queued_at,
    )


def _latest_first() -> tuple[Any, ...]:
    return (
        DailyPlanRow.effective_from.desc(),
        DailyPlanRow.created_at.desc(),
        DailyPlanRow.id.desc(),
    )


class PlanStore:
    """Repository over ``daily_plans``."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def add(self, plan: DailyPlan) -> DailyPlan:
        """Store ``plan`` (a new id is made when it has none) and return the stored plan."""
        with self._db.transaction() as session:
            row = DailyPlanRow(
                local_date=plan.local_date,
                timezone=plan.timezone,
                day_type=plan.day_type,
                plan=plan.to_json(),
                seed=plan.seed,
                effective_from=plan.effective_from,
                ends_at=plan.ends_at,
                wake_at=plan.wake,
                reason=plan.reason,
                inputs_hash=plan.inputs_hash,
                superseded_by=plan.superseded_by,
                superseded_at=plan.superseded_at,
                lifeline_job_id=plan.lifeline_job_id,
                lifeline_done_at=plan.lifeline_done_at,
                summary_queued_at=plan.summary_queued_at,
                created_at=plan.created_at,
                updated_at=plan.created_at,
                **({"id": plan.id} if plan.id else {}),
            )
            session.add(row)
            session.flush()
            return _plan(row)

    def get(self, plan_id: str) -> DailyPlan | None:
        with self._db.session() as session:
            row = session.get(DailyPlanRow, plan_id)
            return _plan(row) if row else None

    def current_for(self, day: date, zone_key: str) -> DailyPlan | None:
        """The newest plan of ``day`` in ``zone_key`` that nothing has taken over from."""
        stmt = (
            select(DailyPlanRow)
            .where(
                DailyPlanRow.local_date == day,
                DailyPlanRow.timezone == zone_key,
                DailyPlanRow.superseded_at.is_(None),
            )
            .order_by(*_latest_first())
            .limit(1)
        )
        with self._db.session() as session:
            row = session.scalars(stmt).first()
            return _plan(row) if row else None

    def covering(self, moment: datetime) -> DailyPlan | None:
        """The plan that decided the state at ``moment`` (the newest one in force then)."""
        at = ensure_aware(moment)
        stmt = (
            select(DailyPlanRow)
            .where(
                DailyPlanRow.effective_from <= at,
                DailyPlanRow.ends_at > at,
                or_(DailyPlanRow.superseded_at.is_(None), DailyPlanRow.superseded_at > at),
            )
            .order_by(*_latest_first())
            .limit(1)
        )
        with self._db.session() as session:
            row = session.scalars(stmt).first()
            return _plan(row) if row else None

    def for_date(self, day: date) -> list[DailyPlan]:
        """Every plan made for local date ``day``, oldest first (replaced ones included)."""
        stmt = (
            select(DailyPlanRow)
            .where(DailyPlanRow.local_date == day)
            .order_by(DailyPlanRow.created_at, DailyPlanRow.id)
        )
        with self._db.session() as session:
            return [_plan(row) for row in session.scalars(stmt)]

    def in_force_after(self, moment: datetime) -> list[DailyPlan]:
        """Plans that are not replaced and still decide something after ``moment``."""
        at = ensure_aware(moment)
        stmt = (
            select(DailyPlanRow)
            .where(DailyPlanRow.ends_at > at, DailyPlanRow.superseded_at.is_(None))
            .order_by(DailyPlanRow.effective_from, DailyPlanRow.id)
        )
        with self._db.session() as session:
            return [_plan(row) for row in session.scalars(stmt)]

    def recent(self, limit: int = 20) -> list[DailyPlan]:
        stmt = select(DailyPlanRow).order_by(DailyPlanRow.created_at.desc()).limit(limit)
        with self._db.session() as session:
            return [_plan(row) for row in session.scalars(stmt)]

    def supersede_from(
        self, moment: datetime, by: str, *, keep_before: date, zone_key: str
    ) -> list[str]:
        """Mark the plans that still decide something after ``moment`` as replaced by ``by``.

        Plans of earlier days in the same time zone stay: their nights end before the new plan's
        and the new plan takes the night over from them (the newest plan of a moment wins).
        """
        at = ensure_aware(moment)
        with self._db.transaction() as session:
            ids = list(
                session.scalars(
                    select(DailyPlanRow.id).where(
                        DailyPlanRow.id != by,
                        DailyPlanRow.ends_at > at,
                        DailyPlanRow.superseded_at.is_(None),
                        or_(
                            DailyPlanRow.local_date >= keep_before,
                            DailyPlanRow.timezone != zone_key,
                        ),
                    )
                )
            )
            if ids:
                session.execute(
                    update(DailyPlanRow)
                    .where(DailyPlanRow.id.in_(ids))
                    .values(superseded_by=by, superseded_at=at, updated_at=self._clock.now_utc())
                )
            return ids

    def mark_lifeline_queued(self, plan_id: str, job_id: str) -> None:
        self._set(plan_id, lifeline_job_id=job_id, lifeline_done_at=None)

    def mark_lifeline_done(self, plan_id: str, at: datetime) -> None:
        self._set(plan_id, lifeline_done_at=ensure_aware(at))

    def mark_summary_queued(self, plan_id: str, at: datetime) -> None:
        self._set(plan_id, summary_queued_at=ensure_aware(at))

    def carry_over(
        self,
        to_id: str,
        *,
        lifeline_job_id: str | None,
        lifeline_done_at: datetime | None,
        summary_queued_at: datetime | None,
    ) -> None:
        """Give a rebuilt plan the bookkeeping of the plan it replaces (jobs that already ran)."""
        values: dict[str, object] = {}
        if lifeline_job_id is not None:
            values["lifeline_job_id"] = lifeline_job_id
            values["lifeline_done_at"] = lifeline_done_at
        if summary_queued_at is not None:
            values["summary_queued_at"] = summary_queued_at
        if values:
            self._set(to_id, **values)

    def _set(self, plan_id: str, **values: object) -> None:
        with self._db.transaction() as session:
            session.execute(
                update(DailyPlanRow)
                .where(DailyPlanRow.id == plan_id)
                .values(updated_at=self._clock.now_utc(), **values)
            )


# ------------------------------------------------------------------ time zones


@dataclass(frozen=True)
class SwitchRecord:
    """One row of ``timezone_history``."""

    id: str
    changed_at: datetime
    from_timezone: str
    to_timezone: str
    source: str
    plan_id: str | None
    last_greeting_at: datetime | None


def _switch(row: TimezoneChange) -> SwitchRecord:
    return SwitchRecord(
        row.id,
        row.changed_at,
        row.from_timezone,
        row.to_timezone,
        row.source,
        row.plan_id,
        row.last_greeting_at,
    )


class TimezoneHistory:
    """Repository over ``timezone_history`` (R-SCH-002)."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def add(
        self,
        *,
        at: datetime,
        old: str,
        new: str,
        source: str,
        plan_id: str | None,
        last_greeting_at: datetime | None,
    ) -> SwitchRecord:
        now = self._clock.now_utc()
        with self._db.transaction() as session:
            row = TimezoneChange(
                changed_at=ensure_aware(at),
                from_timezone=old,
                to_timezone=new,
                source=source,
                plan_id=plan_id,
                last_greeting_at=last_greeting_at,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            session.flush()
            return _switch(row)

    def link_plan(self, switch_id: str, plan_id: str) -> None:
        with self._db.transaction() as session:
            session.execute(
                update(TimezoneChange)
                .where(TimezoneChange.id == switch_id)
                .values(plan_id=plan_id, updated_at=self._clock.now_utc())
            )

    def recent(self, limit: int = 20) -> list[SwitchRecord]:
        """Newest first."""
        stmt = (
            select(TimezoneChange)
            .order_by(TimezoneChange.changed_at.desc(), TimezoneChange.id.desc())
            .limit(limit)
        )
        with self._db.session() as session:
            return [_switch(row) for row in session.scalars(stmt)]

    def latest(self) -> SwitchRecord | None:
        found = self.recent(1)
        return found[0] if found else None

    def after(self, switch_id: str | None) -> list[SwitchRecord]:
        """Switches recorded after ``switch_id`` (all of them for ``None``), oldest first."""
        stmt = select(TimezoneChange).order_by(TimezoneChange.id)
        if switch_id is not None:
            stmt = stmt.where(TimezoneChange.id > switch_id)
        with self._db.session() as session:
            return [_switch(row) for row in session.scalars(stmt)]


# --------------------------------------------------------------- small settings


class InstallSalt:
    """The random salt of the plan seeds: made once per installation, never shown in a log."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def peek(self) -> str | None:
        """The salt, or ``None`` before the first plan was made (reading never writes)."""
        with self._db.session() as session:
            value = get_setting(session, SALT_KEY)
        return str(value) if value else None

    def get(self) -> str:
        """The salt; the first call makes it."""
        found = self.peek()
        if found is not None:
            return found
        with self._db.transaction(bump_state=False) as session:
            value = get_setting(session, SALT_KEY)
            if not value:
                value = secrets.token_hex(SALT_BYTES)
                put_setting(
                    session, SALT_KEY, value, clock=self._clock, by="schedule", record_history=False
                )
        return str(value)


class GreetingLog:
    """When the last wake-up greeting was sent (R-SCH-002 needs it, round 10 writes it)."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def last(self) -> datetime | None:
        with self._db.session() as session:
            value = get_setting(session, GREETING_KEY)
        return ensure_aware(datetime.fromisoformat(value)) if value else None

    def record(self, at: datetime) -> None:
        """Remember that a wake-up greeting went out at ``at`` (never moves backwards)."""
        moment = ensure_aware(at)
        previous = self.last()
        if previous is not None and previous >= moment:
            return
        with self._db.transaction(bump_state=False) as session:
            put_setting(
                session,
                GREETING_KEY,
                moment.isoformat(),
                clock=self._clock,
                by="schedule",
                record_history=False,
            )
