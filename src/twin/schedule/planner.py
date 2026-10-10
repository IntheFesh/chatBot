"""The daily planner: makes, keeps and replaces the day plans, answers "what is she doing" (R-SCH).

:class:`DailyPlanner` is the synchronous core of the schedule.  It owns the rules for when a
plan is made and what replaces what; the application component (:mod:`twin.schedule.component`)
calls it in a worker thread and publishes the events it returns.

When a plan is made
    ``ensure(day)`` makes the plan of a local day that has none, for the whole day, the first time
    anything asks (the scheduler does at local 00:05, the application at start-up).  The seed is a
    hash of the date and the installation's salt, so the plan made at 00:05 and the plan made at
    noon are the same plan.  ``refresh()`` is for a moment when the plan may no longer fit the
    world (start-up, waking from sleep, a changed routine or holiday): it compares a fingerprint of
    the plan's inputs and, only if they differ, makes a new plan from now on.

What replaces what (R-SCH-002)
    A new plan that starts at ``T`` supersedes every plan that still decides something after ``T``
    (same day or, after a time zone switch, the old zone's), from ``T`` on: the past stays as it
    was.  The state of a past moment is read from the plan that was in force then.

Time zone switch
    ``switch_timezone`` writes the new zone to the runtime settings, records the switch, and plans
    the rest of the new local day from this moment: the morning and the night of the new zone are
    drawn afresh, and if it is night there already, she goes to bed now (the onset is clipped to
    the moment of the switch) instead of sleeping from an hour that is already behind.  A wake-up
    greeting is not sent again within 18 hours of the last one: that is decided here when the plan
    is made (``greeting``), so the plan of the next morning is already marked.

Everything is read through the injected clock and the time service; nothing here asks the
machine for the time.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twin.clock import Clock
from twin.config.settings import ScheduleConfig
from twin.ops.logging import get_logger
from twin.profile.activity_model import ActivityModel
from twin.schedule.events import (
    CandidatesExpired,
    PlanRebuilt,
    ScheduleEvent,
    TimezoneSwitched,
)
from twin.schedule.plan_builder import (
    PlanRequest,
    QuotaRange,
    build_plan,
    fingerprint,
    model_fingerprint,
    plan_seed,
)
from twin.schedule.plan_model import DailyPlan, HerState, Segment
from twin.schedule.store import GreetingLog, InstallSalt, PlanStore, SwitchRecord, TimezoneHistory
from twin.schedule.time_service import PlanUnavailableError, TimeService
from twin.schedule.wallclock import day_bounds_utc
from twin.storage.db import Database

log = get_logger("twin.schedule.planner")

REMEMBERED_PLANS = 8
STRETCH_STEPS = 4
ONE_MICROSECOND = timedelta(microseconds=1)
PREVIEW_SALT = "preview"


class TimezoneError(ValueError):
    """A time zone name that is not an IANA zone known to ``tzdata``."""


@dataclass(frozen=True)
class PlanOutcome:
    """The plan in force after a call, whether the call made a new one, and what to announce."""

    plan: DailyPlan
    changed: bool
    superseded: tuple[str, ...] = ()
    events: tuple[ScheduleEvent, ...] = ()


@dataclass(frozen=True)
class SwitchOutcome:
    """What a time zone switch did."""

    changed: bool
    old_timezone: str
    new_timezone: str
    record: SwitchRecord | None = None
    plan: DailyPlan | None = None
    events: tuple[ScheduleEvent, ...] = ()


@dataclass(frozen=True)
class GreetingDecision:
    """Whether the wake-up greeting may go out now (round 10 asks before it sends)."""

    allowed: bool
    reason: str
    plan_id: str | None = None


def validate_zone(name: str) -> ZoneInfo:
    """The zone called ``name``; anything but a real IANA name is an error."""
    text = name.strip()
    if not text or text.startswith(("/", ".")) or ".." in text:
        raise TimezoneError(f"not an IANA time zone name: {name!r} (for example Asia/Shanghai)")
    try:
        return ZoneInfo(text)
    except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
        raise TimezoneError(
            f"unknown IANA time zone {name!r} (for example America/Chicago or Asia/Shanghai)"
        ) from exc


class DailyPlanner:
    """Makes, stores and reads the day plans (see the module description)."""

    def __init__(
        self,
        *,
        db: Database,
        clock: Clock,
        time_service: TimeService,
        config: ScheduleConfig,
        model_source: Callable[[], ActivityModel | None],
        quota_source: Callable[[], QuotaRange],
        set_timezone: Callable[[str, str], None],
        drop_caches: Sequence[Callable[[], None]] = (),
        store: PlanStore | None = None,
        salt: InstallSalt | None = None,
        greetings: GreetingLog | None = None,
        history: TimezoneHistory | None = None,
    ) -> None:
        self._clock = clock
        self._time = time_service
        self._config = config
        self._model_source = model_source
        self._quota_source = quota_source
        self._set_timezone = set_timezone
        self._drop_caches = tuple(drop_caches)
        self.store = store or PlanStore(db, clock)
        self.salt = salt or InstallSalt(db, clock)
        self.greetings = greetings or GreetingLog(db, clock)
        self.history = history or TimezoneHistory(db, clock)
        self._remembered: list[DailyPlan] = []
        # the event loop reads states while a worker thread makes plans: the list of remembered
        # plans has its own small lock, and the making of plans is done by one thread at a time
        self._memory = threading.Lock()
        self._making = threading.RLock()

    # ----------------------------------------------------------------- caches

    def invalidate(self) -> None:
        """Forget what was read (another process changed the tables, or the zone changed)."""
        with self._memory:
            self._remembered.clear()
        for drop in self._drop_caches:
            drop()

    def _remember(self, plan: DailyPlan) -> None:
        with self._memory:
            kept = [p for p in self._remembered if p.id != plan.id]
            kept.append(plan)
            self._remembered = kept[-REMEMBERED_PLANS:]

    def _recall(self, moment: datetime) -> DailyPlan | None:
        """The remembered plan that decides ``moment`` (the newest one that covers it)."""
        with self._memory:
            candidates = [p for p in self._remembered if p.covers(moment)]
        if not candidates:
            return None
        return max(candidates, key=lambda p: (p.effective_from, p.created_at))

    # ----------------------------------------------------------------- inputs

    def _load_model(self) -> tuple[ActivityModel | None, str]:
        model = self._model_source()
        return model, model_fingerprint(model)

    def _inputs_hash(self, day: date, zone: ZoneInfo, routine_hash: str) -> str:
        quota = self._quota_source()
        return fingerprint(
            zone.key,
            day.isoformat(),
            self._time.day_type(day),
            self._time.day_type(day + timedelta(days=1)),
            routine_hash,
            [quota.minimum, quota.maximum, quota.enabled],
            [self._config.min_awake_h, self._config.meal_jitter_min],
        )

    def _request(
        self,
        day: date,
        zone: ZoneInfo,
        *,
        effective_from: datetime,
        reason: str,
        same_day: DailyPlan | None,
        model: ActivityModel | None,
        routine_hash: str,
        salt: str,
    ) -> PlanRequest:
        return PlanRequest(
            day=day,
            zone=zone,
            day_type=self._time.day_type(day),
            next_day_type=self._time.day_type(day + timedelta(days=1)),
            seed=plan_seed(day, salt),
            model=model,
            effective_from=effective_from,
            go_to_bed_at=effective_from if reason == "timezone_switch" else None,
            quota=self._quota_source(),
            last_greeting_at=self.greetings.last(),
            previous_day=self.store.current_for(day - timedelta(days=1), zone.key),
            same_day=same_day,
            reason=reason,
            config=self._config,
            inputs_hash=self._inputs_hash(day, zone, routine_hash),
            routine_hash=routine_hash,
            created_at=self._clock.now_utc(),
        )

    # ------------------------------------------------------------------ making

    def _create(
        self,
        day: date,
        *,
        reason: str,
        effective_from: datetime | None,
        same_day: DailyPlan | None,
        loaded: tuple[ActivityModel | None, str] | None = None,
    ) -> PlanOutcome:
        zone = self._time.bot_timezone()
        day_start, _ = day_bounds_utc(day, zone)
        start = day_start if effective_from is None else max(effective_from, day_start)
        model, routine_hash = loaded or self._load_model()
        request = self._request(
            day,
            zone,
            effective_from=start,
            reason=reason,
            same_day=same_day,
            model=model,
            routine_hash=routine_hash,
            salt=self.salt.get(),
        )
        stored = self.store.add(build_plan(request))
        replaced: list[str] = []
        if same_day is not None or reason == "timezone_switch":
            replaced = self.store.supersede_from(
                start, stored.id, keep_before=day, zone_key=zone.key
            )
            stored = self._carry(stored, replaced)
        self._remember(stored)
        log.info(
            "day_plan_made",
            day=day.isoformat(),
            zone=zone.key,
            reason=reason,
            plan=stored.id,
            replaced=len(replaced),
        )
        events: tuple[ScheduleEvent, ...] = ()
        if replaced:
            events = (
                PlanRebuilt(
                    at=request.created_at,
                    plan_id=stored.id,
                    reason=reason,
                    superseded=tuple(replaced),
                ),
            )
        return PlanOutcome(stored, True, tuple(replaced), events)

    def _carry(self, plan: DailyPlan, replaced: Sequence[str]) -> DailyPlan:
        """Keep the record of the daily jobs that already ran for this day and this zone.

        A lifeline that was drawn for the day stays: she may have told the user about it, and a
        corrected routine does not rewrite what already happened.  The plan of a new local date
        (after a time zone switch) starts without any.
        """
        olds = [p for p in (self.store.get(i) for i in replaced) if p is not None]
        same = [p for p in olds if p.local_date == plan.local_date and p.timezone == plan.timezone]
        if not same:
            return plan
        newest = max(same, key=lambda p: (p.effective_from, p.created_at))
        queued = max((p.summary_queued_at for p in same if p.summary_queued_at), default=None)
        self.store.carry_over(
            plan.id,
            lifeline_job_id=newest.lifeline_job_id,
            lifeline_done_at=newest.lifeline_done_at,
            summary_queued_at=queued,
        )
        return self.store.get(plan.id) or plan

    def ensure(self, day: date | None = None, *, reason: str = "demand") -> DailyPlan:
        """The plan of a local day; made for the whole day if there is none (R-SCH-004)."""
        zone = self._time.bot_timezone()
        target = day if day is not None else self._time.local_date()
        found = self.store.current_for(target, zone.key)
        if found is not None:
            self._remember(found)
            return found
        with self._making:
            found = self.store.current_for(target, zone.key)  # another thread may have just made it
            if found is not None:
                self._remember(found)
                return found
            return self._create(target, reason=reason, effective_from=None, same_day=None).plan

    def refresh(self, reason: str, *, force: bool = False) -> PlanOutcome:
        """Make today's plan current (start-up, waking, a changed routine) - R-SCH-005.

        No plan yet: the plan of the whole day.  A plan whose inputs have not changed is kept
        (the same seed would draw the same plan).  Otherwise a new plan takes over from now on.
        """
        with self._making:
            now = self._clock.now_utc()
            zone = self._time.bot_timezone()
            day = self._time.local_date(now)
            current = self.store.current_for(day, zone.key)
            if current is None:
                return self._create(day, reason=reason, effective_from=None, same_day=None)
            loaded = self._load_model()
            if not force and current.inputs_hash == self._inputs_hash(day, zone, loaded[1]):
                self._remember(current)
                return PlanOutcome(current, False)
            return self._create(
                day, reason=reason, effective_from=now, same_day=current, loaded=loaded
            )

    # --------------------------------------------------------------- time zone

    def switch_timezone(self, new_name: str, *, source: str = "cli") -> SwitchOutcome:
        """Move the bot to another time zone and plan the rest of the new local day (R-SCH-002)."""
        new_zone = validate_zone(new_name)
        with self._making:
            return self._switch(new_zone, source)

    def _switch(self, new_zone: ZoneInfo, source: str) -> SwitchOutcome:
        old_zone = self._time.bot_timezone()
        if new_zone.key == old_zone.key:
            return SwitchOutcome(False, old_zone.key, new_zone.key)
        now = self._clock.now_utc()
        last_greeting = self.greetings.last()
        self._set_timezone(new_zone.key, source)
        self.invalidate()
        record = self.history.add(
            at=now,
            old=old_zone.key,
            new=new_zone.key,
            source=source,
            plan_id=None,
            last_greeting_at=last_greeting,
        )
        outcome = self._create(
            self._time.local_date(now), reason="timezone_switch", effective_from=now, same_day=None
        )
        self.history.link_plan(record.id, outcome.plan.id)
        record = replace(record, plan_id=outcome.plan.id)
        log.info(
            "timezone_switched",
            old=old_zone.key,
            new=new_zone.key,
            source=source,
            plan=outcome.plan.id,
        )
        events: tuple[ScheduleEvent, ...] = (
            CandidatesExpired(at=now, cutoff=now, reason="timezone_switch", include_future=True),
            TimezoneSwitched(
                at=now,
                old_timezone=old_zone.key,
                new_timezone=new_zone.key,
                switch_id=record.id,
                plan_id=outcome.plan.id,
            ),
            *outcome.events,
        )
        return SwitchOutcome(True, old_zone.key, new_zone.key, record, outcome.plan, events)

    # ----------------------------------------------------------------- reading

    def _plan_at(self, moment: datetime, *, create: bool) -> DailyPlan | None:
        best = self._recall(moment)
        if best is not None:
            return best
        loaded = self.store.covering(moment)
        if loaded is not None:
            self._remember(loaded)
            return loaded
        if not create:
            return None
        day = self._time.local_date(moment)
        if abs((day - self._time.local_date()).days) > 1:
            raise PlanUnavailableError(
                f"no day plan decides {moment.isoformat()}, and plans are only made for "
                "yesterday, today and tomorrow"
            )
        plan = self.ensure(day, reason="demand")
        if not plan.covers(moment):
            raise PlanUnavailableError(f"the plan of {day.isoformat()} does not cover {moment}")
        return plan

    def plan_at(self, moment: datetime) -> DailyPlan:
        """The plan that decides ``moment`` (made on first use for yesterday to tomorrow)."""
        plan = self._plan_at(moment, create=True)
        if plan is None:
            raise PlanUnavailableError(f"no day plan decides {moment.isoformat()}")
        return plan

    def state_at(self, moment: datetime) -> HerState:
        """What she is doing at ``moment``, read from the plan (R-SCH-001)."""
        plan = self.plan_at(moment)
        segment = plan.segment_at(moment)
        if segment is None:
            raise PlanUnavailableError(f"plan {plan.id} has no state at {moment.isoformat()}")
        since, until = self._stretch(segment)
        return HerState(segment.kind, since, until, plan.id, segment.busy)

    def _same_state(self, segment: Segment, other: Segment | None) -> bool:
        return other is not None and other.kind == segment.kind and other.busy == segment.busy

    def _stretch(self, segment: Segment) -> tuple[datetime, datetime]:
        """How long the state lasts, also across the border between two plans."""
        since, until = segment.start, segment.end
        for _ in range(STRETCH_STEPS):
            following = self._plan_at(until, create=False)
            other = following.segment_at(until) if following is not None else None
            if not self._same_state(segment, other) or other is None:
                break
            until = other.end
        for _ in range(STRETCH_STEPS):
            moment = since - ONE_MICROSECOND
            earlier = self._plan_at(moment, create=False)
            other = earlier.segment_at(moment) if earlier is not None else None
            if not self._same_state(segment, other) or other is None:
                break
            since = other.start
        return since, until

    # ------------------------------------------------------------------ greeting

    def greeting_decision(self, now: datetime | None = None) -> GreetingDecision:
        """May the wake-up greeting go out at ``now``? (R-PRO-004, R-SCH-002)."""
        moment = now if now is not None else self._clock.now_utc()
        plan = self._plan_at(moment, create=False)
        if plan is None:
            return GreetingDecision(False, "no_plan")
        window = plan.greeting
        if not window.allowed or window.earliest is None or window.latest is None:
            return GreetingDecision(False, window.reason, plan.id)
        last = self.greetings.last()
        if plan.wake is not None and last is not None and last >= plan.wake:
            return GreetingDecision(False, "already_sent_after_this_wake_up", plan.id)
        if last is not None and window.earliest - last < timedelta(
            hours=self._config.greeting_min_gap_h
        ):
            return GreetingDecision(False, "within_min_gap_of_the_last_greeting", plan.id)
        if moment < window.earliest:
            return GreetingDecision(False, "too_early", plan.id)
        if moment > window.latest:
            return GreetingDecision(False, "window_over", plan.id)
        return GreetingDecision(True, "ok", plan.id)

    def record_wake_greeting(self, at: datetime) -> None:
        """Round 10 calls this when it has sent the wake-up greeting."""
        self.greetings.record(at)

    # ------------------------------------------------------------------ preview

    def preview(self, day: date) -> DailyPlan:
        """The plan ``day`` would get, without storing anything (for read-only commands).

        With the installation's salt this is exactly the plan that will be made; before the salt
        exists (nothing has been planned yet) a fixed stand-in is used and the result only shows
        the shape of a day.
        """
        zone = self._time.bot_timezone()
        model, routine_hash = self._load_model()
        salt = self.salt.peek() or PREVIEW_SALT
        day_start, _ = day_bounds_utc(day, zone)
        request = self._request(
            day,
            zone,
            effective_from=day_start,
            reason="preview",
            same_day=None,
            model=model,
            routine_hash=routine_hash,
            salt=salt,
        )
        return build_plan(request)
