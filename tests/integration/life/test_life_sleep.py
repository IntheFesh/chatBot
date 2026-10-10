"""The computer sleeps for three hours and wakes up (R-SCH-005, R-ARCH-004).

A sleeping machine stops everything and its wall clock goes on: when it wakes up, the wall clock
has moved three hours more than the monotonic clock did (``clock.jump_wall``).  The application of
``twin run`` notices (the power monitor's clock-jump fallback; on Windows also
``WM_POWERBROADCAST``), and on the WeChat channel - the platform keeps what the user wrote while
it slept - the wake-up flow does what the specification says:

* a message of her own that was planned for the time of the sleep is void: it is not sent three
  hours late, and the log says it was interrupted;
* the channel connects again (a second ``notifystart``) and the messages the user sent meanwhile
  are fetched and answered - with her usual delay, from the moment of the wake-up on, not at once;
* the plan of the day is current again: when the sleep crossed midnight the plan of the new day is
  made on waking, and she is asleep in it - nothing goes out in the deep night;
* the user is told by an alert, once, that the machine was asleep.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from tests.integration.life.conftest import WorldFactory
from tests.support.life_checks import (
    assert_bot_text_not_in_her_data,
    assert_clean_screen,
    assert_never_in_deep_sleep,
    assert_screen_matches_records,
)
from tests.support.life_world import LifeWorld
from tests.support.proactive_world import opening_curve, proactive_model

pytestmark = pytest.mark.integration

FRIDAY, SATURDAY = date(2026, 10, 9), date(2026, 10, 10)
SLEEP = timedelta(hours=3)
PEAKS = {31: 0.5, 32: 0.5, 48: 0.6, 49: 0.6, 73: 0.6, 74: 0.6, 90: 0.5, 91: 0.5, 92: 0.5}
PEAKS |= {93: 0.5, 94: 0.5, 95: 0.5}
QUIET_WINDOW = timedelta(seconds=15)


def announcements(world: LifeWorld) -> int:
    assert world.double is not None
    return len([name for name, _ in world.double.requests if name == "msg/notifystart"])


async def sleep_the_machine(world: LifeWorld, *, message_after: timedelta, text: str) -> datetime:
    """Three hours pass in the wall clock; the user writes ``message_after`` into them."""
    assert world.double is not None
    await world.clock.settle()
    world.clock.jump_wall(message_after.total_seconds())
    world.double.user_types(text)  # the platform keeps it; nobody is home to fetch it
    world.clock.jump_wall((SLEEP - message_after).total_seconds())
    woke = world.now
    taken = world.handled
    await world.clock.step(5)  # the power monitor's next look
    # the channel connects again and fetches what the platform kept (a few polls, a second each)
    await world.run_until(woke + timedelta(minutes=2), until=lambda: world.handled > taken)
    assert world.handled > taken, "the message the platform kept was never fetched"
    return woke


async def test_a_wake_up_in_the_evening_voids_what_was_planned_and_answers_with_a_delay(
    make_world: WorldFactory,
) -> None:
    world = await make_world(
        datetime(2026, 10, 9, 21, 30, tzinfo=UTC),  # 16:30 in Chicago
        model=proactive_model(opening_curve(base=0.02, peaks=PEAKS)),
        platform="ilink",
    )
    await world.say("我去开会啦")
    await world.run_until(world.local(17, 30))
    plan_before = world.plan().id
    planned = list(world.assembly.proactive.scheduler.candidates.pending())
    dinner = [c for c in planned if c.kind == "meal" and c.planned_at > world.local(18, 0)]
    assert dinner and dinner[0].window_end < world.local(18, 45)  # asked about dinner at ~18:20
    assert announcements(world) == 1
    sent_before = len(world.proactive_rows(outcomes=["sent"]))

    # ---- the machine sleeps from 17:30 to 20:30; at 18:00 he wrote ------------------------------
    woke = await sleep_the_machine(world, message_after=timedelta(minutes=30), text="在吗")
    assert woke == world.local(20, 30)
    await world.run_until_idle()
    voided = [r for r in world.proactive_rows(outcomes=["expired"]) if r.reason == "interrupted"]
    assert dinner[0].kind == "meal" and len(voided) == 1 and voided[0].kind == "meal"
    assert voided[0].at >= woke  # decided at the wake-up, not before
    meals = [r for r in world.proactive_rows(outcomes=["sent"]) if r.kind == "meal"]
    assert not [r for r in meals if r.local_date == FRIDAY]  # dinner was not asked after the fact
    assert len(world.proactive_rows(outcomes=["sent"])) >= sent_before
    assert announcements(world) == 2  # the channel connected again
    assert ("system_resumed", "warning") in world.alerts()
    assert world.plan().id == plan_before  # the plan was current: nothing to rebuild

    # ---- the message of 18:00 is answered with her usual delay, counted from the wake-up ---------
    answer = world.persona_said[-1]
    assert answer.text == "嗯嗯" and answer.at >= woke + QUIET_WINDOW
    assert answer.at - woke < timedelta(minutes=30)  # a normal delay, not three hours of it
    inbound = world.in_rows()[-1]
    assert inbound.text == "在吗" and inbound.at == world.local(18, 0)  # it keeps its own time

    assert_clean_screen(world)
    assert_never_in_deep_sleep(world)
    assert_screen_matches_records(world)
    assert_bot_text_not_in_her_data(world)
    assert world.deepseek.unexpected == []


async def test_a_wake_up_after_midnight_makes_the_plan_of_the_new_day(
    make_world: WorldFactory,
) -> None:
    world = await make_world(
        datetime(2026, 10, 10, 3, 0, tzinfo=UTC),  # Friday 22:00 in Chicago
        model=proactive_model(opening_curve(base=0.02, peaks=PEAKS)),
        platform="ilink",
    )
    await world.say("晚安啦")
    await world.run_until(world.local(23, 0))
    assert not world.kit.planner.store.current_for(SATURDAY, "America/Chicago")  # not yet
    friday = world.plan(FRIDAY)
    onset, wake = friday.night.onset, friday.night.wake
    assert onset < world.local(1, 0, on=SATURDAY) and wake > world.local(6, 0, on=SATURDAY)

    # ---- asleep from 23:00 to 02:00: the schedule never saw midnight -----------------------------
    woke = await sleep_the_machine(world, message_after=timedelta(minutes=60), text="睡了吗")
    assert woke == world.local(2, 0, on=SATURDAY)
    await world.run_for(minutes=10)
    saturday = world.plan(SATURDAY)  # made at the wake-up, for the day that began while it slept
    assert saturday.created_at >= woke and saturday.local_date == SATURDAY
    assert saturday.morning is not None and saturday.morning.wake == wake  # the same night
    assert world.kit.time.her_state(world.now).kind == "deep_sleep"
    sent = [r for r in world.proactive_rows(outcomes=["sent"]) if r.at >= world.local(23, 0)]
    assert sent == []  # the good night planned for 23:15 is not sent at 02:00

    # ---- his message waits for her morning -------------------------------------------------------
    assert [s for s in world.persona_said if s.at >= world.local(23, 0)] == []
    await world.run_until_idle()
    answer = world.persona_said[-1]
    assert answer.at >= wake and answer.at - wake <= timedelta(minutes=40)
    assert world.decision() is None  # the round is over
    voided = [r for r in world.proactive_rows() if r.reason == "interrupted"]
    assert voided and all(r.at >= woke for r in voided)
    assert_clean_screen(world)
    assert_never_in_deep_sleep(world)
    assert_screen_matches_records(world)
    assert_bot_text_not_in_her_data(world)
