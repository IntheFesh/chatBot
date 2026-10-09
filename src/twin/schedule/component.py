"""The schedule as an application component: day plans, daily jobs, wake-ups (R-SCH-002..005).

``ScheduleComponent`` is what keeps the schedule running inside ``twin run``:

start-up and waking (R-SCH-005)
    :meth:`ScheduleComponent.resume` loads or makes today's plan (a plan whose inputs changed is
    replaced from now on), declares the proactive candidates planned for a time that has already
    passed void - they are not sent late - and announces the resume, so that the channel connects
    again and the engine answers what arrived while the machine was away.  The messages the user
    sent meanwhile need nothing here: they wait in the channel's inbox and are answered with the
    usual delay by round 09.

every ``schedule.tick_s`` seconds (:meth:`ScheduleComponent.tick`)
    * makes the plan of the new local day at 00:MM (``schedule.plan_minute``, default 00:05);
    * when she wakes, queues the life line of the day (R-MEM-005);
    * ``schedule.summary_lead_min`` before she wakes - or in the nearest off-peak window, but not
      after two hours - queues the summaries of the day before (R-MEM-002);
    * publishes the time zone switches that another process made (``twin timezone set``).

when the state watcher reports a change (within two seconds)
    drops what was read from the database (the holiday calendar, the plans; the zone name is read
    afresh on every call anyway) and checks that today's plan still fits the routine and the
    proactive range.

``switch_timezone`` is the in-process route of ``/时区`` (round 11).  All events go to
:class:`~twin.schedule.events.ScheduleEvents`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING

from twin.app import Application, ComponentHealth, HealthStatus, TaskSupervisor
from twin.ops.logging import get_logger
from twin.ops.power_events import PowerEvent, PowerEventMonitor
from twin.ops.state_watch import StateWatcher
from twin.schedule.events import (
    CandidatesExpired,
    Resumed,
    ResumeKind,
    ScheduleEvent,
    ScheduleEvents,
    TimezoneSwitched,
)
from twin.schedule.jobs import queue_lifeline
from twin.schedule.planner import PlanOutcome, SwitchOutcome
from twin.schedule.service import ScheduleKit, peak_calendar_for, schedule_kit
from twin.schedule.summary import queue_previous_day_summaries, summary_times
from twin.schedule.wallclock import day_bounds_utc

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.schedule.component")

COMPONENT_NAME = "schedule"


@dataclass
class TickReport:
    """What one look at the clock found due."""

    plan_made: bool = False
    lifeline_job: str | None = None
    summary_jobs: list[str] = field(default_factory=list)
    summaries_queued: bool = False
    events: list[ScheduleEvent] = field(default_factory=list)


class ScheduleComponent:
    """Runs the schedule next to the job worker and the channel."""

    name = COMPONENT_NAME
    depends_on: Sequence[str] = ("state_watcher",)

    def __init__(
        self,
        services: Services,
        watcher: StateWatcher | None = None,
        *,
        kit: ScheduleKit | None = None,
    ) -> None:
        self._services = services
        self._watcher = watcher
        self.kit = kit or schedule_kit(services)
        self._config = services.settings.schedule
        self._supervisor = TaskSupervisor(self.name, services.clock, services.alerts)
        self._unsubscribe: Callable[[], None] | None = None
        self._seen_switch: str | None = None
        self._lock = asyncio.Lock()

    @property
    def events(self) -> ScheduleEvents:
        return self.kit.events

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        latest = await asyncio.to_thread(self.kit.planner.history.latest)
        self._seen_switch = latest.id if latest else None
        if self._watcher is not None:
            self._unsubscribe = self._watcher.subscribe(self._on_state_change)
        await self.resume("startup")
        await self.tick()
        self._supervisor.spawn("tick", self._loop, restart_on_exit=True)

    async def stop(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        await self._supervisor.stop()

    def health(self) -> ComponentHealth:
        supervised = self._supervisor.health()
        if supervised.status is not HealthStatus.OK:
            return supervised
        return ComponentHealth(HealthStatus.OK)

    async def _loop(self) -> None:
        clock = self._services.clock
        while True:
            await clock.sleep(self._config.tick_s)
            await self.tick()

    # ------------------------------------------------------------- resume (R-SCH-005)

    async def resume(self, kind: ResumeKind, gap_s: float = 0.0) -> PlanOutcome:
        """Start-up or waking: today's plan is current, late candidates are void, say so."""
        async with self._lock:
            now = self._services.clock.now_utc()
            outcome = await asyncio.to_thread(self.kit.planner.refresh, kind)
        # announced outside the lock: a subscriber may call back into the component
        await self.kit.events.publish(CandidatesExpired(at=now, cutoff=now, reason=kind))
        for event in outcome.events:
            await self.kit.events.publish(event)
        await self.kit.events.publish(
            Resumed(at=now, kind=kind, gap_s=gap_s, plan_id=outcome.plan.id)
        )
        log.info(
            "schedule_resumed",
            kind=kind,
            plan=outcome.plan.id,
            rebuilt=outcome.changed,
            gap_s=round(gap_s),
        )
        return outcome

    async def on_power_event(self, event: PowerEvent) -> None:
        """The machine woke up (the power monitor calls this)."""
        await self.resume("wake", event.gap_s)
        await self.tick()

    # ------------------------------------------------------------------------ tick

    def _unseen_switches(self) -> list[ScheduleEvent]:
        """Time zone switches made by another process, as events (each is published once)."""
        records = self.kit.planner.history.after(self._seen_switch)
        events: list[ScheduleEvent] = []
        for record in records:
            events.append(
                CandidatesExpired(
                    at=record.changed_at,
                    cutoff=record.changed_at,
                    reason="timezone_switch",
                    include_future=True,
                )
            )
            events.append(
                TimezoneSwitched(
                    at=record.changed_at,
                    old_timezone=record.from_timezone,
                    new_timezone=record.to_timezone,
                    switch_id=record.id,
                    plan_id=record.plan_id,
                )
            )
        if records:
            self._seen_switch = records[-1].id
        return events

    def _tick(self) -> TickReport:
        services = self._services
        planner = self.kit.planner
        store = planner.store
        time = self.kit.time
        report = TickReport()
        now = services.clock.now_utc()
        zone = time.bot_timezone()
        today = time.local_date(now)
        plan = store.current_for(today, zone.key)
        if plan is None:
            start, _ = day_bounds_utc(today, zone)
            due = start + timedelta(minutes=self._config.plan_minute)
            if now >= due or store.covering(now) is None:
                plan = planner.ensure(today, reason="daily")
                report.plan_made = True
        if plan is not None:
            wake = plan.wake if plan.wake is not None else plan.effective_from
            if plan.lifeline_job_id is None and now >= wake:
                job = queue_lifeline(services, plan)
                store.mark_lifeline_queued(plan.id, job)
                report.lifeline_job = job
            if plan.summary_queued_at is None:
                times = summary_times(
                    wake,
                    calendar=peak_calendar_for(services),
                    offpeak_discount=services.settings.pricing.offpeak_multiplier < 1.0,
                    config=self._config,
                )
                if now >= times.trigger_at:
                    report.summary_jobs = queue_previous_day_summaries(services, plan, times)
                    store.mark_summary_queued(plan.id, now)
                    report.summaries_queued = True
        report.events.extend(self._unseen_switches())
        return report

    async def tick(self) -> TickReport:
        """Do what is due now (see the module description)."""
        async with self._lock:
            report = await asyncio.to_thread(self._tick)
        for event in report.events:
            await self.kit.events.publish(event)
        return report

    # ------------------------------------------------------------- state changes

    async def _on_state_change(self, _old: int, _new: int) -> None:
        async with self._lock:
            outcome = await asyncio.to_thread(self._after_state_change)
        for event in outcome:
            await self.kit.events.publish(event)

    def _after_state_change(self) -> list[ScheduleEvent]:
        self.kit.drop_caches()
        result = self.kit.planner.refresh("settings_changed")
        return [*result.events, *self._unseen_switches()]

    # ------------------------------------------------------------------ time zone

    async def switch_timezone(self, name: str, *, source: str = "command") -> SwitchOutcome:
        """Switch the bot's time zone (R-SCH-002) and announce it."""
        async with self._lock:
            outcome = await asyncio.to_thread(self.kit.planner.switch_timezone, name, source=source)
            if outcome.record is not None:
                self._seen_switch = outcome.record.id
        for event in outcome.events:
            await self.kit.events.publish(event)
        if outcome.changed:
            await self.tick()
        return outcome

    async def rebuild(self, reason: str = "manual", *, force: bool = True) -> PlanOutcome:
        """Make today's plan again from now (``twin plan rebuild``, a changed routine)."""
        async with self._lock:
            outcome = await asyncio.to_thread(self.kit.planner.refresh, reason, force=force)
        for event in outcome.events:
            await self.kit.events.publish(event)
        return outcome


def register_schedule(
    application: Application,
    services: Services,
    watcher: StateWatcher,
    *,
    reconnect: Callable[[], Awaitable[None]] | None = None,
) -> tuple[ScheduleComponent, PowerEventMonitor]:
    """Add the schedule and the power event monitor to ``application``.

    ``reconnect`` (the channel component's) is called when the machine woke up from sleep.
    """
    component = ScheduleComponent(services, watcher)
    application.register(component)
    config = services.settings.schedule
    monitor = PowerEventMonitor(
        services.clock,
        component.on_power_event,
        tick_s=config.power_tick_s,
        jump_ticks=config.power_jump_ticks,
        alerts=services.alerts,
    )
    application.register(monitor)
    if reconnect is not None:

        async def reconnect_after_wake(event: Resumed) -> None:
            if event.kind == "wake":
                await reconnect()

        component.kit.events.subscribe(Resumed, reconnect_after_wake)
    return component, monitor
