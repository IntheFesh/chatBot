"""A restart or a wake-up goes on from the stored state, per state (R-ENG-001, R-SCH-005)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.engine_extras import restart
from tests.support.engine_harness import (
    START,
    FixedDay,
    Harness,
    ScriptedChannel,
    ScriptedWriter,
    build_harness,
    make_draft,
    run_to_idle,
    wait_for_state,
)
from tests.support.waiting import wait_until
from twin.engine.roundstate import RoundData
from twin.schedule.events import Resumed
from twin.services import Services
from twin.storage.engine_models import BotTurn

LOCAL_MIDNIGHT = START + timedelta(hours=17)
WAKE = START + timedelta(hours=24)


@pytest.fixture
async def rig(services: Services, clock: ManualClock) -> AsyncIterator[Harness]:
    harness = build_harness(services, clock)
    await harness.engine.start()
    yield harness
    await harness.engine.stop()


def at_state(harness: Harness, state: str) -> bool:
    return harness.engine.snapshot().state == state


def out_texts(harness: Harness) -> list[str]:
    with harness.services.db.session() as session:
        rows = session.scalars(
            select(BotTurn)
            .where(BotTurn.direction == "out", BotTurn.kind != "no_reply")
            .order_by(BotTurn.at, BotTurn.id)
        )
        return [row.text for row in rows]


# ------------------------------------------------------------------------ IDLE


async def test_idle_with_a_waiting_message_starts_collecting(
    services: Services, clock: ManualClock
) -> None:
    first = build_harness(services, clock)  # the message arrives before the engine runs
    await first.message("在吗")
    snap = first.engine.snapshot()
    assert snap.state == "IDLE" and len(snap.pending) == 1
    second = await restart(first)
    try:
        await run_to_idle(second.engine, clock)
        assert second.channel.texts == ["好的"]
    finally:
        await second.engine.stop()


# ------------------------------------------------------------------ COLLECTING


async def test_collecting_starts_its_quiet_window_again_and_takes_in_the_backlog(
    rig: Harness,
) -> None:
    await rig.message("第一条")
    await wait_until(lambda: rig.engine.waiting_until == START + timedelta(seconds=15))
    second = await restart(rig, clock_jump_s=600)  # ten minutes of downtime
    try:
        now = rig.clock.now_utc()
        await wait_until(lambda: second.engine.waiting_until == now + timedelta(seconds=15))
        await second.message("补发的一条")  # the channel hands over what piled up meanwhile
        await run_to_idle(second.engine, second.clock)
        assert second.writer.contexts[0].user_text == "第一条\n补发的一条"
        assert second.writer.calls == 1 and second.channel.texts == ["好的"]
    finally:
        await second.engine.stop()


# ------------------------------------------------------------------- DECIDING


async def test_deciding_keeps_a_planned_time_that_lies_ahead(
    services: Services, clock: ManualClock
) -> None:
    clock.set_time(LOCAL_MIDNIGHT)
    night = FixedDay(
        (START - timedelta(days=1), "free"),
        (START + timedelta(hours=15), "deep_sleep"),
        (START + timedelta(hours=23, minutes=30), "sleep_edge"),
        (WAKE, "free"),
    )
    first = build_harness(services, clock, day=night)
    await first.engine.start()
    await first.message("睡了吗")
    await run_to_idle(first.engine, clock, until=lambda: at_state(first, "DECIDING"))
    planned = first.engine.snapshot().planned_send_at
    assert planned is not None and planned > WAKE
    second = await restart(first, clock_jump_s=3 * 3600)
    try:
        assert second.engine.snapshot().planned_send_at == planned  # the queue survives
        await wait_until(lambda: second.engine.waiting_until == planned)
        assert second.channel.texts == []
        await run_to_idle(second.engine, clock)
        assert second.writer.contexts[0].woke_up
        sent = next(o for o in second.channel.out if o.kind == "text")
        assert sent.at >= planned
    finally:
        await second.engine.stop()


async def test_deciding_draws_again_when_the_planned_time_has_passed(rig: Harness) -> None:
    await rig.message("在吗")
    await run_to_idle(rig.engine, rig.clock, until=lambda: at_state(rig, "DECIDING"))
    old = rig.engine.snapshot().planned_send_at
    second = await restart(rig, clock_jump_s=7200)  # the machine slept for two hours
    try:
        woke = second.clock.now_utc()
        assert old is not None and old < woke
        await wait_until(lambda: second.engine.waiting_until is not None)
        planned = second.engine.snapshot().planned_send_at
        assert planned is not None and planned > woke  # not the moment the machine wakes
        assert planned - woke >= timedelta(seconds=20)  # her usual latency, from now
        assert second.writer.calls == 0
        await run_to_idle(second.engine, second.clock)
        first_bubble = next(o for o in second.channel.out if o.kind == "text")
        assert first_bubble.at - woke >= timedelta(seconds=20)
    finally:
        await second.engine.stop()


# ----------------------------------------------------------------- GENERATING


async def test_generating_goes_back_to_deciding_for_a_fresh_draw(rig: Harness) -> None:
    rig.writer.gate = asyncio.Event()
    await rig.message("在吗")
    await run_to_idle(rig.engine, rig.clock, until=rig.writer.started.is_set)
    assert at_state(rig, "GENERATING")
    second = await restart(rig, clock_jump_s=300)
    try:
        woke = second.clock.now_utc()
        await wait_for_state(second.engine, "DECIDING")
        await wait_until(lambda: second.engine.waiting_until is not None)
        planned = second.engine.snapshot().planned_send_at
        assert planned is not None and planned - woke >= timedelta(seconds=20)
        assert RoundData.of(second.engine.snapshot()).answering == ()
        await run_to_idle(second.engine, second.clock)
        assert second.channel.texts == ["好的"] and second.writer.calls == 1
    finally:
        await second.engine.stop()


# -------------------------------------------------------------------- SENDING


async def test_sending_goes_on_with_the_bubbles_that_were_not_out(rig: Harness) -> None:
    rig.writer.add(make_draft("一", "二", "三"))
    await rig.message("数数")
    await run_to_idle(rig.engine, rig.clock, until=lambda: rig.channel.texts == ["一"])
    killed = rig.engine.snapshot()
    stored = RoundData.of(killed).outgoing
    assert killed.state == "SENDING" and stored is not None
    assert [b.text for b in stored.unsent] == ["二", "三"] and killed.pending == ()
    assert [item["text"] for item in killed.sent] == ["一"]
    second = await restart(rig, clock_jump_s=60)
    try:
        resumed = second.clock.now_utc()
        await run_to_idle(second.engine, second.clock)
        assert second.channel.texts == ["二", "三"]
        assert second.writer.calls == 0  # nothing is written again
        assert out_texts(second) == ["一", "二", "三"]
        gaps = [o.at for o in second.channel.out if o.kind == "text"]
        assert gaps[0] - resumed >= timedelta(seconds=1)  # not the moment the machine wakes
        with second.services.db.session() as session:
            replies = {
                row.reply_id for row in session.scalars(select(BotTurn)) if row.direction == "out"
            }
        assert len(replies) == 1  # one reply, three bubbles
        assert second.engine.snapshot().state == "IDLE"
    finally:
        await second.engine.stop()


async def test_sending_with_a_message_that_came_meanwhile_continues_like_an_interruption(
    rig: Harness,
) -> None:
    rig.writer.add(make_draft("一", "二", "三"))
    await rig.message("数数")
    await run_to_idle(rig.engine, rig.clock, until=lambda: rig.channel.texts == ["一"])
    second = await restart(rig, clock_jump_s=60)
    try:
        second.writer.add(make_draft("好的不数了"))
        await second.message("别数了")
        await run_to_idle(second.engine, second.clock)
        assert second.channel.texts == ["好的不数了"]
        context = second.writer.contexts[0]
        assert context.already_said == ("一",) and context.user_text == "别数了"
    finally:
        await second.engine.stop()


async def test_a_kill_after_the_last_bubble_leaves_nothing_to_send(rig: Harness) -> None:
    rig.writer.add(make_draft("一"))
    await rig.message("嗨")
    await run_to_idle(rig.engine, rig.clock, until=lambda: rig.channel.texts == ["一"])
    second = await restart(rig)
    try:
        await run_to_idle(second.engine, second.clock)
        assert second.channel.texts == [] and second.engine.snapshot().state == "IDLE"
        assert out_texts(second) == ["一"]
    finally:
        await second.engine.stop()


# ------------------------------------------------------------------- Resumed


async def test_a_wake_up_keeps_a_planned_time_that_is_still_ahead(rig: Harness) -> None:
    await rig.message("在吗")
    await run_to_idle(rig.engine, rig.clock, until=lambda: at_state(rig, "DECIDING"))
    planned = rig.engine.snapshot().planned_send_at
    await rig.engine.on_resumed(Resumed(rig.clock.now_utc(), "wake", 30.0, None))
    await asyncio.sleep(0.05)
    assert rig.engine.snapshot().planned_send_at == planned
    await run_to_idle(rig.engine, rig.clock)
    assert rig.channel.texts == ["好的"]


async def test_a_wake_up_after_the_planned_time_draws_again_not_a_burst_of_replies(
    rig: Harness,
) -> None:
    await rig.message("在吗")
    await run_to_idle(rig.engine, rig.clock, until=lambda: at_state(rig, "DECIDING"))
    old = rig.engine.snapshot().planned_send_at
    rig.clock.jump_wall(7200)  # the machine slept; the wall clock moved, the sleepers did not
    woke = rig.clock.now_utc()
    await rig.engine.on_resumed(Resumed(woke, "wake", 7200.0, None))
    await wait_until(
        lambda: (
            (rig.engine.snapshot().planned_send_at or woke) > woke
            and rig.engine.snapshot().planned_send_at != old
        )
    )
    assert rig.channel.texts == []
    await run_to_idle(rig.engine, rig.clock)
    first = next(o for o in rig.channel.out if o.kind == "text")
    assert first.at - woke >= timedelta(seconds=20)


async def test_a_wake_up_cancels_a_generation_whose_connection_is_gone(rig: Harness) -> None:
    rig.writer.gate = asyncio.Event()
    await rig.message("在吗")
    await run_to_idle(rig.engine, rig.clock, until=rig.writer.started.is_set)
    await rig.engine.on_resumed(Resumed(rig.clock.now_utc(), "wake", 600.0, None))
    await wait_until(lambda: rig.writer.cancelled == 1)
    await wait_for_state(rig.engine, "DECIDING")
    rig.writer.gate.set()
    await run_to_idle(rig.engine, rig.clock)
    assert rig.channel.texts == ["好的"] and rig.writer.calls == 2


async def test_a_wake_up_while_sending_goes_on_with_the_next_bubble(rig: Harness) -> None:
    rig.writer.add(make_draft("一", "二"))
    await rig.message("数数")
    await run_to_idle(rig.engine, rig.clock, until=lambda: rig.channel.texts == ["一"])
    await rig.engine.on_resumed(Resumed(rig.clock.now_utc(), "wake", 600.0, None))
    await run_to_idle(rig.engine, rig.clock)
    assert rig.channel.texts == ["一", "二"] and rig.writer.calls == 1


async def test_the_start_event_of_the_schedule_is_harmless_for_an_idle_engine(rig: Harness) -> None:
    await rig.engine.on_resumed(Resumed(rig.clock.now_utc(), "startup", 0.0, None))
    await asyncio.sleep(0.05)
    assert rig.engine.snapshot().state == "IDLE"
    rig.writer.add(make_draft("好"))
    await rig.message("在吗")
    await run_to_idle(rig.engine, rig.clock)
    assert rig.channel.texts == ["好"]


async def test_two_engines_in_a_row_share_one_channel_history(
    services: Services, clock: ManualClock
) -> None:
    channel = ScriptedChannel(clock)
    first = build_harness(services, clock, channel=channel, writer=ScriptedWriter(make_draft("一")))
    await first.engine.start()
    await first.message("嗨")
    await run_to_idle(first.engine, clock)
    second = await restart(first, channel=channel, writer=ScriptedWriter(make_draft("二")))
    try:
        await second.message("再嗨")
        await run_to_idle(second.engine, clock)
        assert [turn.text for turn in second.writer.contexts[0].history] == ["嗨", "一"]
        assert channel.texts == ["一", "二"]
    finally:
        await second.engine.stop()
