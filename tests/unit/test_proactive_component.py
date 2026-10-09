"""The scheduler as a component of the application: its ticks, its wiring (R-ARCH-001)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from tests.support.clock import ManualClock
from tests.support.engine_harness import build_harness
from tests.support.proactive_world import World
from twin.app import Application, HealthStatus
from twin.engine.command_port import CommandContext
from twin.engine.component import EngineComponent
from twin.schedule.events import CandidatesExpired
from twin.schedule.proactive.component import (
    COMPONENT_NAME,
    OFFSET_MARGIN_S,
    ProactiveComponent,
    build_scheduler,
    next_tick_at,
    register_proactive,
    tick_offset_s,
)
from twin.schedule.proactive.scheduler import TickReport
from twin.services import Services

TICK_S = 300.0
DAY = date(2026, 10, 9)


# ------------------------------------------------------------------------ the grid


def test_the_offset_is_a_fixed_draw_of_the_day_inside_the_tick() -> None:
    first = tick_offset_s("salt", DAY, TICK_S)
    assert first == tick_offset_s("salt", DAY, TICK_S)
    assert OFFSET_MARGIN_S <= first <= TICK_S - OFFSET_MARGIN_S
    days = {tick_offset_s("salt", DAY + timedelta(days=n), TICK_S) for n in range(30)}
    assert len(days) > 20  # another offset on most days
    assert tick_offset_s("other salt", DAY, TICK_S) != first
    assert all(OFFSET_MARGIN_S <= v <= TICK_S - OFFSET_MARGIN_S for v in days)


def test_a_very_short_tick_still_has_an_offset() -> None:
    assert 0 < tick_offset_s("salt", DAY, 15.0) < 15.0


@pytest.mark.parametrize("seconds", [0, 1, 37, 299, 300, 301, 12_345])
def test_the_next_tick_is_the_first_one_on_the_grid_after_now(seconds: int) -> None:
    offset = 42.5
    now = datetime(2026, 10, 9, 12, 0, tzinfo=UTC) + timedelta(seconds=seconds)
    following = next_tick_at(now, TICK_S, offset)
    assert following > now and following - now <= timedelta(seconds=TICK_S)
    assert (following.timestamp() - offset) % TICK_S == pytest.approx(0, abs=1e-6)
    assert following.tzinfo is not None


def test_a_moment_that_is_a_tick_waits_for_the_next_one() -> None:
    offset = 40.0
    on_grid = datetime.fromtimestamp(1_790_000_000 // 300 * 300 + offset, tz=UTC)
    assert next_tick_at(on_grid, TICK_S, offset) == on_grid + timedelta(seconds=TICK_S)


# ----------------------------------------------------------------------- the loop


class Counting:
    """A scheduler tick that only notes when it was called."""

    def __init__(self, clock: ManualClock, fail_first: bool = False) -> None:
        self.clock = clock
        self.times: list[datetime] = []
        self.fail_first = fail_first

    async def __call__(self) -> TickReport:
        self.times.append(self.clock.now_utc())
        if self.fail_first and len(self.times) == 1:
            raise RuntimeError("one bad tick")
        return TickReport(at=self.clock.now_utc())


async def test_the_component_ticks_at_once_and_then_on_the_aligned_grid(calm: World) -> None:
    counting = Counting(calm.clock)
    calm.scheduler.tick = counting  # type: ignore[method-assign]
    calm.go_to(calm.at(10, 0))
    component = ProactiveComponent(calm.scheduler, calm.services, calm.rig.kit)
    assert component.name == COMPONENT_NAME and tuple(component.depends_on) == (
        "schedule",
        "engine",
    )
    await component.start()
    try:
        await calm.clock.settle()
        assert len(counting.times) == 1 and component.ticks == 1  # it opens the day at once
        for _ in range(4):
            await calm.clock.advance(TICK_S + 1)
            await calm.clock.settle()
        assert len(counting.times) >= 4
        offset = tick_offset_s(calm.rig.planner.salt.get(), DAY, TICK_S)
        for moment in counting.times[1:]:
            assert (moment.timestamp() - offset) % TICK_S < 2.0 or TICK_S - (
                (moment.timestamp() - offset) % TICK_S
            ) < 2.0, moment
        assert component.health().status == HealthStatus.OK
    finally:
        await component.stop()


async def test_a_component_without_a_first_tick_waits_for_the_grid(calm: World) -> None:
    counting = Counting(calm.clock)
    calm.scheduler.tick = counting  # type: ignore[method-assign]
    component = ProactiveComponent(calm.scheduler, calm.services, calm.rig.kit, first_tick=False)
    await component.start()
    try:
        await calm.clock.settle()
        assert counting.times == []
        await calm.clock.advance(TICK_S + 1)
        await calm.clock.settle()
        assert len(counting.times) == 1
    finally:
        await component.stop()


async def test_one_bad_tick_does_not_end_the_schedule(calm: World) -> None:
    counting = Counting(calm.clock, fail_first=True)
    calm.scheduler.tick = counting  # type: ignore[method-assign]
    component = ProactiveComponent(calm.scheduler, calm.services, calm.rig.kit)
    await component.start()
    try:
        await calm.clock.settle()
        assert component.ticks == 1  # the failed one counts, and nothing was raised
        await calm.clock.advance(TICK_S + 1)
        await calm.clock.settle()
        assert len(counting.times) == 2 and component.ticks == 2
    finally:
        await component.stop()


async def test_stopping_the_component_stops_the_ticks(calm: World) -> None:
    counting = Counting(calm.clock)
    calm.scheduler.tick = counting  # type: ignore[method-assign]
    component = ProactiveComponent(calm.scheduler, calm.services, calm.rig.kit)
    await component.start()
    await calm.clock.settle()
    await component.stop()
    before = len(counting.times)
    await calm.clock.advance(3 * TICK_S)
    await calm.clock.settle()
    assert len(counting.times) == before


# --------------------------------------------------------------------------- wiring


async def test_registering_adds_the_component_the_events_and_the_rating_command(
    calm: World,
) -> None:
    application = Application()
    engine_component = EngineComponent(calm.engine, calm.channel, calm.services)
    component = register_proactive(application, calm.services, engine_component)
    assert application.components[COMPONENT_NAME] is component
    assert isinstance(component, ProactiveComponent)
    before = component.scheduler.epoch
    now = calm.clock.now_utc()
    await calm.rig.kit.events.publish(CandidatesExpired(at=now, cutoff=now, reason="wake"))
    assert component.scheduler.epoch == before + 1  # its own subscription heard it
    router = engine_component.router
    assert router is not None
    outcome = await router.handle("/评分 5 好", CommandContext(now, "in-1"))
    assert outcome is not None and "记下了：5/5。" in outcome.reply
    assert [r.score for r in calm.ratings.recent(5)] == [5]
    help_text = await router.handle("/帮助", CommandContext(now, "in-2"))
    assert help_text is not None and "/评分 <1-5> [备注]" in help_text.reply


async def test_a_plan_change_makes_the_scheduler_forget_what_it_worked_out(calm: World) -> None:
    application = Application()
    component = register_proactive(
        application, calm.services, EngineComponent(calm.engine, calm.channel, calm.services)
    )
    scheduler = component.scheduler
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.go_to(calm.at(10, 0))
    await scheduler.tick()
    assert scheduler._today is not None  # type: ignore[attr-defined]
    outcome = calm.rig.planner.switch_timezone("Asia/Shanghai")
    await calm.publish(outcome.events)
    assert scheduler._today is None  # type: ignore[attr-defined]


async def test_the_scheduler_needs_an_engine_made_by_build_engine(
    services: Services, clock: ManualClock
) -> None:
    harness = build_harness(services, clock)
    assert harness.engine.kit is None
    with pytest.raises(RuntimeError, match="build_engine"):
        build_scheduler(services, EngineComponent(harness.engine, harness.channel, services))


def test_the_status_source_is_made_without_an_engine(calm: World) -> None:
    from twin.schedule.proactive.component import proactive_status_for

    source = proactive_status_for(calm.services, calm.channel.session_state)
    assert source() is not None
    calm.channel.session_state = lambda: (_ for _ in ()).throw(RuntimeError("no channel"))  # type: ignore[method-assign]
    again: Any = proactive_status_for(calm.services, calm.channel.session_state)()
    assert again is not None and again.blocked is None  # a channel that cannot say hides nothing
