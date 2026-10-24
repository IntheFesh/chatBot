"""The start of ``twin run`` does not wait for the proactive scheduler's first tick (R-NFR-003).

The first tick of the scheduler opens the day: it asks the planner, whose calls have timeouts of
60 to 180 seconds and retries.  ``Application.start`` starts the components one after the other and
has no start timeout, so a component that waits for that tick at its start holds up every component
after it - and the bot does not receive messages for minutes (R-NFR-003: under 30 seconds).  Here
the first tick is blocked for good, as a planner that never answers would block it, and the
application of ``twin run`` must still start completely and answer the user (D-730).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from tests.integration.life.conftest import WorldFactory
from tests.support.proactive_world import opening_curve, proactive_model
from twin.schedule.proactive.scheduler import TickReport

pytestmark = pytest.mark.integration

START_LIMIT_S = 30.0  # R-NFR-003, in real seconds: the start itself takes milliseconds


async def test_a_first_proactive_tick_that_never_ends_does_not_hold_up_the_application(
    make_world: WorldFactory,
) -> None:
    """R17-36: every component starts, and the bot answers, while the first tick is blocked."""
    world = await make_world(
        datetime(2026, 10, 9, 16, 0, tzinfo=UTC),  # 11:00 in Chicago, she is up
        model=proactive_model(opening_curve(base=0.02)),
        start_application=False,
    )
    world.deepseek.book.when("你好", "嗨")
    entered = asyncio.Event()
    gate = asyncio.Event()
    cancelled: list[bool] = []

    async def planner_that_never_answers() -> TickReport:
        entered.set()
        try:
            await gate.wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return TickReport(at=world.now)

    application = world.assembly.application
    world.assembly.proactive.scheduler.tick = planner_that_never_answers  # type: ignore[method-assign]
    await asyncio.wait_for(world.start(), timeout=START_LIMIT_S)  # before the fix: it never ends

    assert entered.is_set() and not gate.is_set()  # the first tick is running, and blocked
    started = {component.name for component in application._started}  # type: ignore[attr-defined]
    assert started == set(application.components)  # the ones after "proactive" are up as well
    order = [component.name for component in application.start_order()]
    assert order.index("proactive") < len(order) - 1  # there are components that follow it

    await world.say("你好")  # the engine receives and answers while the planner is still silent
    await world.run_until_idle()
    assert [said.text for said in world.persona_said] == ["嗨"]
    assert not gate.is_set() and cancelled == []

    await application.stop()  # a component that is stopped ends the blocked tick, cleanly
    world.started = False
    assert cancelled == [True]
