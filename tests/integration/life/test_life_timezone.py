"""The time zone changes at noon: Chicago to Shanghai (R-SCH-002, R-SCH-004, R-PRO-003).

The user has come home and tells her, by ``/时区``, that the bot lives in Shanghai now.  At noon in
Chicago it is one o'clock in the night of the next day in Shanghai.  The application of ``twin run``
runs on through the switch:

* the command is answered at once and the zone is stored (``settings``, ``timezone_history``);
* today's plan is made again from this moment, in the new zone: she is not woken for the occasion -
  she goes to sleep at once (the night is clipped to the switch) and wakes at the hour of the new
  zone's night;
* the candidates of the old plan (the dinner she might have asked about) are void;
* she does not greet him again in the morning: the last greeting is less than 18 hours old;
* nothing of hers goes out in the new deep sleep, and what he writes meanwhile is answered when she
  wakes - at a Shanghai hour, not a Chicago one.
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
from tests.support.proactive_world import opening_curve, proactive_model
from twin.config.runtime import BOT_TIMEZONE

pytestmark = pytest.mark.integration

FRIDAY = date(2026, 10, 9)
CHICAGO, SHANGHAI = "America/Chicago", "Asia/Shanghai"


async def test_the_zone_switch_at_noon_neither_greets_twice_nor_wakes_her(
    make_world: WorldFactory,
) -> None:
    world = await make_world(
        datetime(2026, 10, 9, 3, 0, tzinfo=UTC),  # Thursday 22:00 in Chicago
        model=proactive_model(opening_curve(base=0.02)),
    )
    await world.say("我先睡啦，晚安")
    await world.run_until(world.local(11, 55, on=FRIDAY))
    greeted = [r for r in world.proactive_rows(outcomes=["sent"]) if r.kind == "greeting"]
    assert len(greeted) == 1 and greeted[0].timezone == CHICAGO  # the morning greeting, in Chicago

    # ---- noon: the switch --------------------------------------------------------------------
    await world.run_until(world.local(12, 0, on=FRIDAY))
    switch_at = world.now
    await world.say("/时区 Asia/Shanghai")
    answer = world.system_said[-1]
    assert answer.at == switch_at  # a command is answered the moment it arrives
    assert "America/Chicago 切到 Asia/Shanghai" in answer.text
    assert "2026-10-10 周六 01:00" in answer.text  # one o'clock in the night, in Shanghai
    assert "她现在：快睡着或刚醒" in answer.text
    assert world.services.runtime.get(BOT_TIMEZONE) == SHANGHAI
    record = world.kit.planner.history.latest()
    assert record is not None
    assert (record.from_timezone, record.to_timezone) == (CHICAGO, SHANGHAI)
    assert record.changed_at == switch_at and record.last_greeting_at == greeted[0].at

    # ---- the plan is made again, from this moment, in the new zone ---------------------------
    plan = world.plan(date(2026, 10, 10))
    assert plan.timezone == SHANGHAI and plan.reason == "timezone_switch"
    assert plan.effective_from == switch_at and plan.id == record.plan_id
    assert plan.morning is not None and plan.morning.clipped and plan.morning.onset == switch_at
    assert plan.morning.wake.astimezone(world.zone).date() == date(2026, 10, 10)
    assert plan.morning.wake - switch_at > timedelta(hours=5)  # a whole night, as hers are
    assert world.kit.time.her_state(switch_at + timedelta(minutes=1)).asleep
    assert not plan.greeting.allowed and "min_gap" in plan.greeting.reason  # no second greeting
    assert plan.night is not None and plan.night.onset > plan.morning.wake
    voided = [
        r for r in world.proactive_rows(outcomes=["expired"]) if r.reason == "timezone_switch"
    ]
    assert voided and all(r.at == switch_at for r in voided)  # what was planned is void

    # ---- she sleeps in the new zone; his message waits for her waking ------------------------
    await world.run_for(minutes=40)
    await world.say("在吗")
    wake = plan.morning.wake
    await world.run_for(seconds=60)
    decision = world.decision()
    assert decision is not None and decision.mode == "asleep" and decision.send_at >= wake
    await world.run_until(wake + timedelta(hours=3))
    after = [s for s in world.persona_said if s.at > switch_at]
    assert after, "she answered nothing"
    assert all(s.at >= wake for s in after)  # not one bubble in the new night
    assert wake + timedelta(minutes=5) <= after[0].at <= wake + timedelta(minutes=40)
    kinds = [r.kind for r in world.proactive_rows(outcomes=["sent"]) if r.at > switch_at]
    assert "greeting" not in kinds  # he was greeted at 07:40 Chicago time, 12 hours ago
    assert all(r.timezone == SHANGHAI for r in world.proactive_rows() if r.at >= switch_at)
    local_wake = wake.astimezone(world.zone)
    assert local_wake.tzinfo is not None and local_wake.utcoffset() == timedelta(hours=8)

    # ---- what holds for any story ------------------------------------------------------------
    assert_clean_screen(world)
    assert_never_in_deep_sleep(world)
    assert_screen_matches_records(world)
    assert_bot_text_not_in_her_data(world)
    assert world.deepseek.unexpected == []


async def test_a_reply_that_waits_when_the_zone_is_switched_is_not_sent_in_the_new_night(
    make_world: WorldFactory,
) -> None:
    """R17-32 (D-722): a reply queued for 15:20 Chicago time, then the bot moves to Shanghai.

    At 13:20 he pauses her for two hours and writes; her reply is queued for the end of the pause
    (a few seconds after it, 15:20).  At half past one he switches the bot to Shanghai, where it
    is half past two in the night and she sleeps until the morning: 15:20 in Chicago is 04:20
    there.  The reply waits for her waking (plus the jitter) instead of going out at 15:20 Chicago
    time into the new deep sleep - which is what it did before the engine looked at her state
    again.
    """
    world = await make_world(
        datetime(2026, 10, 9, 18, 20, tzinfo=UTC),  # 13:20 in Chicago
        model=proactive_model(opening_curve(base=0.02)),
    )
    await world.say("/暂停 2小时")
    await world.say("在吗")
    await world.run_for(seconds=30)
    waiting = world.decision()
    assert waiting is not None and waiting.mode == "paused" and not waiting.woke_up
    assert waiting.send_at >= world.local(15, 20) and waiting.send_at - world.local(15, 20) < (
        timedelta(minutes=2)
    )
    await world.run_for(minutes=10)

    # ---- the switch: it is 02:30 of the next night in Shanghai ---------------------------------
    switch_at = world.now
    await world.say("/时区 Asia/Shanghai")
    plan = world.plan(date(2026, 10, 10))
    assert plan.timezone == SHANGHAI and plan.effective_from == switch_at
    assert plan.morning is not None and plan.morning.clipped
    wake = plan.morning.wake
    assert world.kit.time.her_state(waiting.send_at).kind == "deep_sleep"  # 15:20 is in her night
    await world.run_for(seconds=60)
    moved = world.decision()
    assert moved is not None and moved.woke_up and moved.send_at >= wake + timedelta(minutes=5)
    assert moved.wake_at == wake and moved.paused_until == waiting.paused_until
    assert world.persona_said == []

    # ---- the time she was going to answer passes: she sleeps -----------------------------------
    await world.run_until(waiting.send_at + timedelta(minutes=30))
    assert world.persona_said == []
    await world.run_until(wake + timedelta(hours=1))
    said = [s for s in world.persona_said if s.at > switch_at]
    assert said, "she never answered"
    assert said[0].at >= moved.send_at >= wake + timedelta(minutes=5)

    # ---- what holds for any story --------------------------------------------------------------
    assert_clean_screen(world)
    assert_never_in_deep_sleep(world)
    assert_screen_matches_records(world)
    assert_bot_text_not_in_her_data(world)
    assert world.deepseek.unexpected == []


async def test_without_the_switch_the_same_reply_goes_out_at_its_time(
    make_world: WorldFactory,
) -> None:
    """R17-32 (the control): the same story without the switch: the reply comes at 15:20."""
    world = await make_world(
        datetime(2026, 10, 9, 18, 20, tzinfo=UTC),
        model=proactive_model(opening_curve(base=0.02)),
    )
    await world.say("/暂停 2小时")
    await world.say("在吗")
    await world.run_for(seconds=30)
    waiting = world.decision()
    assert waiting is not None and waiting.mode == "paused"
    await world.run_for(minutes=10)
    await world.run_until(waiting.send_at + timedelta(minutes=2))
    said = world.persona_said
    assert said and abs((said[0].at - waiting.send_at).total_seconds()) < 60
    assert said[0].at.astimezone(world.zone).hour == 15
    assert_never_in_deep_sleep(world)
    assert world.deepseek.unexpected == []
