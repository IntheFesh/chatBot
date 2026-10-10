"""The schedule component: the 00:05 plan, the daily jobs, restart and wake-up, outside switches.

R-SCH-004 (the plan at 00:05 or at start-up), R-MEM-005 and R-MEM-002 (the life line when she
wakes, the summaries before), R-SCH-005 (no late messages after a restart or a wake-up), R-SCH-002
(a switch made by another process is announced), R-SCH-001 (a changed time zone is noticed within
two seconds).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta

import pytest

from tests.support.clock import ManualClock
from tests.support.routine import Rig, fixed_model, make_model, sleep_window
from tests.support.synth_chat import MessageWriter
from tests.support.waiting import wait_until
from twin.app import Application, HealthStatus
from twin.config.runtime import BOT_TIMEZONE
from twin.ops.components import build_application
from twin.ops.jobs import JobQueue
from twin.ops.power_events import PowerEvent
from twin.ops.state_watch import StateWatcher
from twin.profile.overrides import RoutineOverrides
from twin.schedule.component import ScheduleComponent, register_schedule
from twin.schedule.events import CandidatesExpired, PlanRebuilt, Resumed, TimezoneSwitched
from twin.schedule.jobs import LIFELINE_JOB
from twin.services import Services
from twin.storage.db import Database, WritePolicy, use_write_policy

FRIDAY = date(2026, 10, 9)
SHANGHAI = "Asia/Shanghai"
CHICAGO = "America/Chicago"


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


@pytest.fixture
def rig(services: Services, clock: ManualClock) -> Rig:
    clock.set_time(utc(2026, 10, 9, 5, 0))  # 00:00 on Friday in Chicago
    return Rig.build(services, clock, fixed_model())


@pytest.fixture
def component(rig: Rig, services: Services) -> ScheduleComponent:
    return ScheduleComponent(services, kit=rig.kit)


def jobs(services: Services, kind: str | None = None) -> list:  # type: ignore[type-arg]
    return JobQueue(services.db, services.clock).list_jobs(job_type=kind, limit=100)


class Recorder:
    """Subscribes to every event type and keeps what it hears, in order."""

    def __init__(self, component: ScheduleComponent) -> None:
        self.events: list[object] = []
        for kind in (CandidatesExpired, PlanRebuilt, Resumed, TimezoneSwitched):
            component.events.subscribe(kind, self.events.append)

    def of(self, kind: type) -> list:  # type: ignore[type-arg]
        return [e for e in self.events if isinstance(e, kind)]


# ------------------------------------------------------------------- the daily plan


async def test_the_plan_of_the_new_day_is_made_at_five_past_midnight(
    rig: Rig, component: ScheduleComponent
) -> None:
    rig.planner.ensure(FRIDAY)  # Friday's plan runs on into Saturday morning
    rig.move_to(rig.at(2026, 10, 10, 0, 4))
    report = await component.tick()
    assert (
        not report.plan_made and rig.planner.store.current_for(date(2026, 10, 10), CHICAGO) is None
    )
    assert rig.kit.time.her_state().kind == "deep_sleep"  # Friday's plan still decides the night
    rig.move_to(rig.at(2026, 10, 10, 0, 5))
    report = await component.tick()
    assert report.plan_made
    saturday = rig.planner.store.current_for(date(2026, 10, 10), CHICAGO)
    assert saturday is not None and saturday.reason == "daily" and saturday.day_type == "weekend"
    again = await component.tick()
    assert not again.plan_made  # once


async def test_a_machine_with_no_plan_at_all_does_not_wait_for_five_past(
    rig: Rig, component: ScheduleComponent
) -> None:
    report = await component.tick()  # 00:00 on a fresh install: nothing covers the moment
    assert report.plan_made
    assert rig.planner.store.current_for(FRIDAY, CHICAGO) is not None


async def test_start_up_loads_or_makes_the_plan_of_today(
    rig: Rig, component: ScheduleComponent
) -> None:
    rig.move_to(rig.at(2026, 10, 9, 15))
    recorder = Recorder(component)
    await component.start()
    try:
        plan = rig.planner.store.current_for(FRIDAY, CHICAGO)
        assert plan is not None and plan.reason == "startup"
        (resumed,) = recorder.of(Resumed)
        assert resumed.kind == "startup" and resumed.plan_id == plan.id
        assert component.health().status is HealthStatus.OK
    finally:
        await component.stop()
    restarted = ScheduleComponent(component._services, kit=rig.kit)
    await restarted.start()
    try:
        assert len(rig.planner.store.for_date(FRIDAY)) == 1  # the plan was found, not made again
    finally:
        await restarted.stop()


# ---------------------------------------------------------------------- daily jobs


async def test_the_life_line_is_queued_once_when_she_wakes(
    rig: Rig, component: ScheduleComponent, services: Services
) -> None:
    plan = rig.planner.ensure(FRIDAY)
    rig.move_to(rig.at(2026, 10, 9, 7, 29))
    assert (await component.tick()).lifeline_job is None
    assert jobs(services, LIFELINE_JOB) == []
    rig.move_to(rig.at(2026, 10, 9, 7, 30))
    report = await component.tick()
    (job,) = jobs(services, LIFELINE_JOB)
    assert report.lifeline_job == job.id
    assert job.payload == {"date": "2026-10-09", "plan_id": plan.id} and not job.offpeak_only
    stored = rig.planner.store.get(plan.id)
    assert stored is not None and stored.lifeline_job_id == job.id
    rig.move_to(rig.at(2026, 10, 9, 12))
    assert (await component.tick()).lifeline_job is None
    fresh = ScheduleComponent(services, kit=rig.kit)  # a restart does not queue it again
    assert (await fresh.tick()).lifeline_job is None
    assert len(jobs(services, LIFELINE_JOB)) == 1


async def test_a_plan_made_after_she_woke_queues_the_life_line_at_once(
    rig: Rig, component: ScheduleComponent, services: Services
) -> None:
    rig.move_to(rig.at(2026, 10, 9, 15))
    report = await component.tick()
    assert report.plan_made and report.lifeline_job is not None


async def test_without_a_routine_the_life_line_waits_for_nothing(
    rig: Rig, component: ScheduleComponent
) -> None:
    rig.use(None)
    report = await component.tick()
    assert report.lifeline_job is not None  # no wake-up is planned: the plan's start is the cue


async def test_the_summaries_are_queued_an_hour_before_she_wakes(
    rig: Rig, component: ScheduleComponent, services: Services
) -> None:
    writer = MessageWriter(services)
    writer.add(utc(2026, 10, 8, 20, 0), True, "text", "第一句")  # Thursday 15:00 in Chicago
    writer.add(utc(2026, 10, 8, 20, 5), False, "text", "第二句")
    writer.store()
    rig.planner.ensure(FRIDAY)
    rig.move_to(rig.at(2026, 10, 9, 6, 29))  # wake-up 07:30, lead 06:30
    assert not (await component.tick()).summaries_queued
    rig.move_to(rig.at(2026, 10, 9, 6, 30))
    report = await component.tick()
    assert report.summaries_queued and len(report.summary_jobs) == 1
    (job,) = jobs(services, "memory_summary")
    assert job.payload["date"] == "2026-10-08" and job.payload["scope"] == "real"
    assert job.offpeak_only and job.deadline == rig.at(2026, 10, 9, 9, 30)  # two hours after waking
    assert not (await component.tick()).summaries_queued
    assert len(jobs(services, "memory_summary")) == 1


async def test_a_day_without_records_queues_no_summary_but_is_not_looked_at_again(
    rig: Rig, component: ScheduleComponent, services: Services
) -> None:
    plan = rig.planner.ensure(FRIDAY)
    rig.move_to(rig.at(2026, 10, 9, 7))
    report = await component.tick()
    assert report.summaries_queued and report.summary_jobs == []
    assert jobs(services, "memory_summary") == []
    stored = rig.planner.store.get(plan.id)
    assert stored is not None and stored.summary_queued_at == rig.clock.now_utc()


async def test_the_tick_loop_does_the_work_when_the_clock_reaches_it(
    rig: Rig, component: ScheduleComponent, services: Services
) -> None:
    rig.planner.ensure(FRIDAY)
    rig.move_to(rig.at(2026, 10, 9, 7, 20))
    await component.start()
    try:
        await wait_until(lambda: rig.clock.pending_sleepers >= 1)
        assert jobs(services, LIFELINE_JOB) == []
        await rig.clock.advance(600)  # 07:30 passes inside the loop
        # a tick works in a thread, so the loop may have looked at the clock before it moved on:
        # whenever it is asleep again (its last tick is finished) it is given the next tick
        for _ in range(5):
            await wait_until(lambda: rig.clock.pending_sleepers >= 1)
            if jobs(services, LIFELINE_JOB):
                break
            await rig.clock.advance(rig.services.settings.schedule.tick_s)
        assert len(jobs(services, LIFELINE_JOB)) == 1
    finally:
        await component.stop()


# --------------------------------------------------------- restart and wake-up


class Candidates:
    """The proactive candidates round 10 will keep: planned times and whether they are void."""

    def __init__(self, component: ScheduleComponent, planned: dict[str, datetime]) -> None:
        self.planned = planned
        self.void: set[str] = set()
        component.events.subscribe(CandidatesExpired, self.on_expired)

    def on_expired(self, event: CandidatesExpired) -> None:
        self.void |= {name for name, at in self.planned.items() if event.covers(at)}


async def test_candidates_planned_for_a_time_that_has_passed_are_not_sent_late(
    rig: Rig, component: ScheduleComponent
) -> None:
    now = rig.at(2026, 10, 9, 15)
    rig.move_to(now)
    book = Candidates(
        component,
        {
            "an hour ago": now - timedelta(hours=1),
            "just now": now - timedelta(seconds=1),
            "right now": now,
            "in half an hour": now + timedelta(minutes=30),
        },
    )
    await component.resume("startup")
    assert book.void == {"an hour ago", "just now"}  # later ones keep their turn


async def test_waking_from_sleep_does_the_same_and_reports_how_long_it_was(
    rig: Rig, component: ScheduleComponent
) -> None:
    rig.planner.ensure(FRIDAY)
    rig.move_to(rig.at(2026, 10, 9, 12))
    recorder = Recorder(component)
    book = Candidates(
        component, {"missed": rig.at(2026, 10, 9, 11), "later": rig.at(2026, 10, 9, 13)}
    )
    await component.resume("wake", 7200.0)
    assert book.void == {"missed"}
    (expired,) = recorder.of(CandidatesExpired)
    assert expired.reason == "wake" and expired.cutoff == rig.clock.now_utc()
    (resumed,) = recorder.of(Resumed)
    assert resumed.kind == "wake" and resumed.gap_s == 7200.0
    assert recorder.events.index(expired) < recorder.events.index(resumed)


async def test_waking_across_midnight_makes_the_plan_of_the_new_day(
    rig: Rig, component: ScheduleComponent
) -> None:
    rig.planner.ensure(FRIDAY)
    rig.move_to(rig.at(2026, 10, 9, 23, 50))
    await component.tick()
    rig.move_to(rig.at(2026, 10, 10, 9))  # the machine slept for nine hours
    outcome = await component.resume("wake", 9 * 3600)
    assert outcome.changed and outcome.plan.local_date == date(2026, 10, 10)
    assert outcome.plan.reason == "wake" and rig.kit.time.her_state().kind == "free"


async def test_waking_rebuilds_a_plan_that_no_longer_fits_the_routine(
    rig: Rig, component: ScheduleComponent
) -> None:
    first = rig.planner.ensure(FRIDAY)
    rig.move_to(rig.at(2026, 10, 9, 12))
    rig.use(make_model({"all": sleep_window(22.0, 6.0)}))  # corrected while the machine slept
    recorder = Recorder(component)
    outcome = await component.resume("wake", 3600.0)
    assert outcome.changed and outcome.superseded == (first.id,)
    assert outcome.plan.night is not None
    assert outcome.plan.night.onset == rig.at(2026, 10, 9, 22)
    (rebuilt,) = recorder.of(PlanRebuilt)
    assert rebuilt.plan_id == outcome.plan.id and rebuilt.reason == "wake"
    unchanged = await component.resume("wake", 60.0)
    assert not unchanged.changed  # nothing moved: the plan stays


async def test_a_power_event_runs_the_wake_flow_and_the_daily_work(
    rig: Rig, component: ScheduleComponent, services: Services
) -> None:
    rig.planner.ensure(FRIDAY)
    rig.move_to(rig.at(2026, 10, 9, 12))
    recorder = Recorder(component)
    await component.on_power_event(PowerEvent("resume", "clock_jump", rig.clock.now_utc(), 5400.0))
    assert [e.gap_s for e in recorder.of(Resumed)] == [5400.0]
    assert len(jobs(services, LIFELINE_JOB)) == 1  # she woke while the machine slept


async def test_the_channel_is_asked_to_reconnect_only_after_a_wake_up(
    rig: Rig, services: Services
) -> None:
    calls: list[str] = []

    async def reconnect() -> None:
        calls.append("reconnect")

    application = Application()
    build, watcher = build_application(services)
    del build
    component, _monitor = register_schedule(application, services, watcher, reconnect=reconnect)
    await component.resume("startup")
    assert calls == []
    await component.resume("wake", 30.0)
    assert calls == ["reconnect"]


def test_the_schedule_and_the_power_monitor_join_the_application(
    rig: Rig, services: Services
) -> None:
    application, watcher = build_application(services)
    component, monitor = register_schedule(application, services, watcher)
    names = [c.name for c in application.start_order()]
    assert {"schedule", "power_events", "state_watcher"} <= set(names)
    assert names.index("state_watcher") < names.index("schedule")
    assert component.name == "schedule" and monitor.name == "power_events"
    config = services.settings.schedule
    assert monitor.threshold_s == config.power_tick_s * config.power_jump_ticks


# ------------------------------------------------------------ changes from outside


def another_process_changes(db: Database, work) -> None:  # type: ignore[no-untyped-def]
    """What a LIGHT CLI command does: write, bumping the state version in the same transaction."""
    other = Database(db.path, clock=db.clock)
    try:
        with use_write_policy(WritePolicy(bump_state=True)), other.transaction():
            work()
    finally:
        other.dispose()


async def test_a_zone_set_by_another_process_is_effective_within_two_seconds(
    rig: Rig, services: Services, clock: ManualClock
) -> None:
    watcher = StateWatcher(services.db, clock)
    component = ScheduleComponent(services, watcher, kit=rig.kit)
    task = asyncio.create_task(watcher.run())
    try:
        await wait_until(lambda: clock.pending_sleepers >= 1)
        assert rig.kit.time.bot_timezone().key == CHICAGO
        watcher_unsubscribe = watcher.subscribe(component._on_state_change)
        try:
            services.runtime.set(BOT_TIMEZONE, SHANGHAI, by="cli")  # as `twin timezone set` would
            assert rig.kit.time.bot_timezone().key == SHANGHAI  # read on every call
            assert rig.kit.time.local_date() == FRIDAY  # 13:00 there, still Friday
            assert rig.planner.store.current_for(FRIDAY, SHANGHAI) is None
            await clock.advance(2.0)  # one poll interval: the plans follow the zone
            await wait_until(lambda: rig.planner.store.current_for(FRIDAY, SHANGHAI) is not None)
        finally:
            watcher_unsubscribe()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_a_switch_made_by_another_process_is_announced_once(
    rig: Rig, services: Services, clock: ManualClock
) -> None:
    watcher = StateWatcher(services.db, clock)
    component = ScheduleComponent(services, watcher, kit=rig.kit)
    await watcher.poll_once()
    await component.start()
    recorder = Recorder(component)
    try:
        rig.move_to(rig.at(2026, 10, 9, 14))
        another_process_changes(
            services.db, lambda: rig.planner.switch_timezone(SHANGHAI, source="cli")
        )
        await watcher.poll_once()
        (switched,) = recorder.of(TimezoneSwitched)
        assert (switched.old_timezone, switched.new_timezone) == (CHICAGO, SHANGHAI)
        expired = [e for e in recorder.of(CandidatesExpired) if e.reason == "timezone_switch"]
        assert len(expired) == 1 and expired[0].covers(rig.clock.now_utc() + timedelta(days=1))
        await component.tick()
        await watcher.poll_once()
        assert len(recorder.of(TimezoneSwitched)) == 1  # not announced again
        assert rig.kit.time.bot_timezone().key == SHANGHAI
        assert rig.kit.time.her_state().kind in {"sleep_edge", "deep_sleep"}  # 03:00 in Shanghai
    finally:
        await component.stop()


async def test_a_switch_made_inside_the_application_is_announced_once(
    rig: Rig, services: Services, clock: ManualClock
) -> None:
    watcher = StateWatcher(services.db, clock)
    component = ScheduleComponent(services, watcher, kit=rig.kit)
    await watcher.poll_once()
    await component.start()
    recorder = Recorder(component)
    try:
        rig.move_to(rig.at(2026, 10, 9, 10))
        outcome = await component.switch_timezone(SHANGHAI, source="command")
        assert outcome.changed
        assert len(recorder.of(TimezoneSwitched)) == 1
        await watcher.poll_once()  # the version moved: the same switch is not announced again
        assert len(recorder.of(TimezoneSwitched)) == 1 and len(recorder.of(PlanRebuilt)) == 1
        same = await component.switch_timezone(SHANGHAI)
        assert not same.changed and len(recorder.of(TimezoneSwitched)) == 1
        # the day of the new zone is planned and its life line is queued when she wakes there
        plan = rig.planner.store.current_for(date(2026, 10, 9), SHANGHAI)
        assert plan is not None and plan.reason == "timezone_switch"
    finally:
        await component.stop()


async def test_a_corrected_routine_replaces_the_plan_when_the_state_changes(
    rig: Rig, services: Services, clock: ManualClock
) -> None:
    watcher = StateWatcher(services.db, clock)
    component = ScheduleComponent(services, watcher, kit=rig.kit)
    await watcher.poll_once()
    await component.start()
    recorder = Recorder(component)
    try:
        rig.move_to(rig.at(2026, 10, 9, 10))
        before = rig.planner.store.current_for(FRIDAY, CHICAGO)
        assert before is not None
        rig.use(make_model({"all": sleep_window(22.0, 6.0)}))
        another_process_changes(
            services.db,
            lambda: RoutineOverrides(services.db, clock).add_holiday(
                date(2026, 10, 12), date(2026, 10, 12)
            ),
        )
        await watcher.poll_once()
        after = rig.planner.store.current_for(FRIDAY, CHICAGO)
        assert after is not None and after.id != before.id and after.reason == "settings_changed"
        assert len(recorder.of(PlanRebuilt)) == 1
    finally:
        await component.stop()
