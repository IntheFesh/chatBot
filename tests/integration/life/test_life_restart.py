"""The process is killed in the middle of a reply and started again (R-ENG-001, R-CH-004).

This one runs on the WeChat channel: the platform and the user's phone are made up
(``tests/support/wechat_double.py``), the channel is the production ``IlinkChannel`` with its
stored cursor, inbox and window.  "Killed" means that every task of the application ends where it
is and nothing is cleaned up - no ``stop`` of any component - and the next process starts on the
same database and files.  What the user's phone shows is the evidence:

* killed between two bubbles of a reply: after the restart the rest comes, once each, in order,
  without a second request to the model, and not the moment the machine is back;
* killed inside a send - the platform has delivered the bubble, nobody has heard back: the bubble
  is not said again (an unknown outcome is never repeated), and the rest of the reply follows;
* a message of the user that came while the process was dead interrupts the rest of the reply, and
  the new answer knows what was already said;
* a message that the platform holds, or that was fetched but not yet taken by the engine, is
  answered once after the restart - never lost, never twice.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from tests.integration.life.conftest import WorldFactory
from tests.support.life_checks import (
    assert_clean_screen,
    assert_never_in_deep_sleep,
    assert_screen_matches_records,
)
from tests.support.life_world import LifeWorld
from tests.support.proactive_world import opening_curve, proactive_model

pytestmark = pytest.mark.integration

COUNT = ("一", "二", "三", "四")


async def counting_world(make_world: WorldFactory) -> LifeWorld:
    world = await make_world(
        datetime(2026, 10, 9, 16, 0, tzinfo=UTC),  # 11:00 in Chicago, she is up
        model=proactive_model(opening_curve(base=0.02)),
        platform="ilink",
    )
    world.deepseek.book.when("数数", *COUNT)
    return world


def phone(world: LifeWorld) -> list[str]:
    assert world.double is not None
    return world.double.texts()


async def until_shown(world: LifeWorld, count: int) -> None:
    await world.run_until(
        world.now + timedelta(minutes=5), until=lambda: len(phone(world)) >= count
    )
    assert len(phone(world)) == count


def stored_indexes(world: LifeWorld) -> list[tuple[str, int | None]]:
    return [(row.text, row.bubble_index) for row in world.out_rows()]


async def test_killed_between_two_bubbles_the_rest_follows_once_and_in_order(
    make_world: WorldFactory,
) -> None:
    world = await counting_world(make_world)
    await world.say("数数")
    await until_shown(world, 2)
    assert phone(world) == ["一", "二"]
    assert world.assembly.engine.engine.snapshot().state == "SENDING"
    await world.kill()
    dead_until = world.now + timedelta(minutes=5)
    await world.clock.step(300)  # five minutes pass with nobody home
    assert phone(world) == ["一", "二"]
    await world.restart()
    restarted = world.now
    await world.run_for(minutes=5)
    assert phone(world) == list(COUNT)  # each bubble once, in order
    assert stored_indexes(world) == [(text, number) for number, text in enumerate(COUNT)]
    assert world.deepseek.calls["reply"] == 1  # the reply was not written again
    assert world.double is not None
    third = next(d for d in world.double.phone.delivered if d.text == "三")
    assert third.at - restarted >= timedelta(seconds=1)  # not the moment the machine is back
    assert world.now >= dead_until
    assert len({row.reply_id for row in world.out_rows()}) == 1  # one reply, four bubbles
    assert_screen_matches_records(world)
    assert_clean_screen(world)
    assert_never_in_deep_sleep(world)
    assert world.deepseek.unexpected == []


async def test_killed_inside_a_send_the_bubble_is_not_said_twice(make_world: WorldFactory) -> None:
    world = await counting_world(make_world)
    assert world.double is not None
    me = asyncio.current_task()

    def die_with_the_second_bubble_on_the_phone(_: object) -> None:
        if len(phone(world)) == 2:  # the platform has shown it; the answer is on its way back
            for task in asyncio.all_tasks() - world.before_start - {me}:
                task.cancel()

    world.double.after_delivery = die_with_the_second_bubble_on_the_phone
    await world.say("数数")
    await until_shown(world, 2)
    await world.kill()
    world.double.after_delivery = None
    stored = world.assembly.engine.engine.snapshot()
    assert [item["text"] for item in stored.sent] == [
        "一"
    ]  # "二" was never noted: the process died
    await world.clock.step(300)
    await world.restart()
    await world.run_for(minutes=5)
    assert phone(world) == list(COUNT)  # "二" once: it counts as sent, the rest goes on after it
    assert stored_indexes(world) == [(text, number) for number, text in enumerate(COUNT)]
    first = world.out_rows()[0]
    assert any(a["step"] == "bubble_in_doubt" for a in first.actions or [])
    assert world.deepseek.calls["reply"] == 1 and world.double.refused == 0
    assert_screen_matches_records(world)
    assert_clean_screen(world)


async def test_a_message_that_came_while_the_process_was_dead_interrupts_the_rest(
    make_world: WorldFactory,
) -> None:
    world = await counting_world(make_world)
    assert world.double is not None
    world.deepseek.book.when("别数了", "好的不数了")
    await world.say("数数")
    await until_shown(world, 2)
    await world.kill()
    world.double.user_types("别数了")  # the platform keeps it until the bot asks for it
    await world.clock.step(120)
    await world.restart()
    await world.run_until_idle()
    assert phone(world) == ["一", "二", "好的不数了"]  # the rest of the count is not said
    prompt = world.deepseek.of_kind("reply")[-1].last_user
    assert "一" in prompt and "二" in prompt and "别数了" in prompt  # she knows what she said
    assert world.deepseek.calls["reply"] == 2
    assert_screen_matches_records(world)
    assert_clean_screen(world)


@pytest.mark.parametrize("steps_before_kill", [0, 1])
async def test_a_message_is_answered_once_whatever_stage_it_had_reached(
    make_world: WorldFactory, steps_before_kill: int
) -> None:
    """0: the platform has it, nobody fetched it.  1: fetched and stored, not yet answered."""
    world = await make_world(
        datetime(2026, 10, 9, 16, 0, tzinfo=UTC),
        model=proactive_model(opening_curve(base=0.02)),
        platform="ilink",
    )
    assert world.double is not None
    world.deepseek.book.when("在吗", "在呀")
    world.double.user_types("在吗")
    for _ in range(steps_before_kill):
        await world.clock.step(1.0)  # the poll fetches it
    await world.kill()
    assert phone(world) == []
    await world.clock.step(60)
    await world.restart()
    await world.run_until_idle()  # her own delay may be a few minutes
    assert phone(world) == ["在呀"]  # once
    assert world.deepseek.calls["reply"] == 1
    assert [r.text for r in world.in_rows()] == ["在吗"]
    assert_screen_matches_records(world)
