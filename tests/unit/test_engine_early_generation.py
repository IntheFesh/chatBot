"""The reply is made before her delay ends, so that it appears when the delay does (D-723).

Her measured latency contains the time she types; the reply used to be started when the delay was
over, so the first bubble came after the delay *plus* the generation *plus* the typing.  Now
the engine keeps a running estimate of how long a reply needs from the start of its generation
until its first bubble could appear (``generation_lead_s``: the generation, the silent second and
the typing), begins the reply that long before the drawn moment, stores that moment with the reply
(``Outgoing.due_at``) and lets the sender await the first bubble silently until it is time to
type it (``BubbleSender.send(first_due_at=...)``).

The pacing of these tests has no randomness: her latency is 60 s, reading two characters 0.5 s, so
the delay D of a message "在吗" is 60.5 s from its arrival; typing the draft "好的" takes 1 s; the
writer takes G seconds of the manual clock.  A reply of 8 s of generation has a lead of
8 + 1 + 1 = 10 s.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta

import pytest

from tests.support.clock import ManualClock
from tests.support.engine_extras import restart
from tests.support.engine_harness import (
    Harness,
    ScriptedWriter,
    build_harness,
    decision_of,
    make_draft,
    reference_pacing,
    run_to_idle,
)
from tests.support.waiting import wait_until
from twin.config.runtime import ENGINE_PAUSED_UNTIL
from twin.engine.pacing import MIN_INTERVAL_S
from twin.engine.roundstate import RoundData
from twin.engine.state_store import ConversationSnapshot
from twin.services import Services

DELAY = 60.5  # her latency (60 s) and the time to read "在吗"
GENERATION = 8.0
TYPING = 1.0  # "好的" at half a second a character
LEAD = GENERATION + MIN_INTERVAL_S + TYPING  # what the estimate is after one such reply


def make_writer(clock: ManualClock, takes_s: float = GENERATION) -> ScriptedWriter:
    return ScriptedWriter(clock=clock, takes_s=takes_s)


@pytest.fixture
async def world(services: Services, clock: ManualClock) -> AsyncIterator[Harness]:
    harness = build_harness(
        services, clock, writer=make_writer(clock), pacing=reference_pacing(latency_s=60.0)
    )
    await harness.engine.start()
    yield harness
    await harness.engine.stop()


async def run(harness: Harness, **options: object) -> None:
    await run_to_idle(harness.engine, harness.clock, writer=harness.writer, **options)  # type: ignore[arg-type]


async def warm_up(harness: Harness) -> None:
    """One reply is made, so that the engine knows how long a reply takes."""
    await harness.message("在吗")
    await run(harness)
    assert harness.engine.generation_lead_s == LEAD
    assert harness.channel.texts == ["好的"]


def first_bubble(harness: Harness, number: int = 1) -> datetime:
    texts = [o for o in harness.channel.out if o.kind == "text"]
    return texts[number - 1].at


async def test_the_first_reply_of_a_process_is_made_when_her_delay_is_over(world: Harness) -> None:
    """R17-34 (cold start): nothing is known of how long a reply takes, so nothing begins early."""
    anchor = world.clock.now_utc()
    assert world.engine.generation_lead_s == 0.0
    await world.message("在吗")
    await run(world)
    assert world.writer.began == [anchor + timedelta(seconds=DELAY)]  # begun when the delay is over
    # late by the generation: the silent second, then the typing - as before this change plus the
    # second that every late reply waits
    assert first_bubble(world) - anchor == timedelta(
        seconds=DELAY + GENERATION + MIN_INTERVAL_S + TYPING
    )


async def test_a_reply_made_in_advance_appears_at_the_moment_her_delay_ends(world: Harness) -> None:
    """R17-34: the first bubble of the second reply is out at D, not at D + G + typing."""
    await warm_up(world)
    await world.clock.advance(3600)  # an hour of quiet: the first reply is long over
    anchor = world.clock.now_utc()
    await world.message("在吗")
    await run(world)
    # the generation began its lead before the drawn moment (D - 10 s) ...
    assert world.writer.began[1] == anchor + timedelta(seconds=DELAY - LEAD)
    # ... and the first bubble is out at D (to within the second that the sender keeps),
    # not at D + G + typing = D + 9 s
    appeared = first_bubble(world, 2) - anchor
    assert abs(appeared - timedelta(seconds=DELAY)) <= timedelta(seconds=MIN_INTERVAL_S)
    assert appeared == timedelta(seconds=DELAY)  # in fact exactly: the estimate was exact
    # "typing" shows during the last second only; before that she was silent
    typing_on = next(
        o.at for o in world.channel.out if o.kind == "typing" and o.at > world.writer.began[1]
    )
    assert typing_on - anchor == timedelta(seconds=DELAY - TYPING)
    assert world.channel.texts == ["好的", "好的"]
    row = world.store.latest_reply()
    assert row and row[0].text == "好的"


async def test_a_generation_slower_than_the_lead_makes_the_reply_late_by_the_overrun_only(
    world: Harness,
) -> None:
    """R17-34: the reply that takes 20 s instead of 8 is 12 s late (not 20 s), and teaches."""
    await warm_up(world)
    await world.clock.advance(3600)
    world.writer.takes_s = 20.0
    anchor = world.clock.now_utc()
    await world.message("在吗")
    await run(world)
    ready = DELAY - LEAD + 20.0  # begun 10 s early, 20 s long: 10 s after the delay
    assert first_bubble(world, 2) - anchor == timedelta(
        seconds=ready + MIN_INTERVAL_S + TYPING  # the least wait, then the typing
    )
    # the estimate moves towards what it took: 0.7 * 10 + 0.3 * (20 + 1 + 1)
    assert world.engine.generation_lead_s == pytest.approx(0.7 * LEAD + 0.3 * 22.0)


async def test_the_lead_is_capped_and_a_failure_or_silence_does_not_teach(world: Harness) -> None:
    """R17-34: at most 30 s before the delay ends; only a reply that was made counts."""
    world.writer.takes_s = 100.0
    await world.message("在吗")
    await run(world)
    assert world.engine.generation_lead_s == 30.0  # a minute and a half, cut at the ceiling
    world.writer.takes_s = 5.0
    world.writer.add(make_draft(fallback="backend_error"), make_draft(no_reply=True))
    await world.clock.advance(3600)
    await world.message("在吗")
    await run(world)  # the failure is tried again later: the retry is a silence (no reply)
    assert world.engine.generation_lead_s == 30.0  # neither a failure nor a silence changed it


async def test_a_message_while_the_reply_is_made_in_advance_joins_the_round_and_time_stands(
    world: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R17-34: inside her delay a message merges (R-ENG-004); it does not start the round again."""
    await warm_up(world)
    await world.clock.advance(3600)
    anchor = world.clock.now_utc()
    await world.message("在吗")
    world.writer.started.clear()
    await run(world, until=world.writer.started.is_set)
    planned = world.engine.snapshot().planned_send_at
    assert planned is None  # the state is GENERATING: the planned time is in the decision
    decision = decision_of(world.engine)
    assert decision is not None and decision.send_at == anchor + timedelta(seconds=DELAY)
    begun = decision.send_at - timedelta(seconds=LEAD)
    assert abs(world.clock.now_utc() - begun) < timedelta(milliseconds=10)  # D - lead
    visited: list[str] = []
    original = world.state.transition

    def record(state: str, **fields: object) -> ConversationSnapshot:
        visited.append(state)
        return original(state, **fields)

    monkeypatch.setattr(world.state, "transition", record)
    await world.clock.advance(2)  # 2 s into the generation: still inside her delay
    await world.message("还有一件事")
    await wait_until(lambda: world.writer.cancelled == 1)
    await run(world)
    assert visited[:2] == [
        "DECIDING",
        "GENERATING",
    ]  # not back to COLLECTING and a new quiet window
    assert "COLLECTING" not in visited
    assert world.writer.contexts[-1].user_text == "在吗\n还有一件事"
    assert world.channel.texts == ["好的", "好的"]  # one reply to both
    out = world.store.latest_reply()
    assert out and {"step": "generation_cancelled", "count": 1} in (out[0].actions or [])
    # the drawn time stood (the message came after 80 % of the wait, but the short pause after it
    # ends before the drawn time): no new draw, no new quiet window and first delay - the reply is
    # made again at once, 2 s after the round began, and shown after its own 8 s, the least wait
    # and the typing
    assert world.writer.began[-1] == anchor + timedelta(seconds=DELAY - LEAD + 2)
    assert first_bubble(world, 2) - anchor == timedelta(
        seconds=DELAY - LEAD + 2 + GENERATION + MIN_INTERVAL_S + TYPING
    )


async def test_a_message_while_the_early_reply_waits_to_be_shown_drops_it_and_joins_the_round(
    world: Harness,
) -> None:
    """R17-34: the draft that is ready early and awaited silently is dropped like any other draft
    when the user writes - and the round goes back to her delay, not to a new one."""
    await warm_up(world)
    await world.clock.advance(3600)
    world.writer.takes_s = 2.0  # this reply is ready at D - 8 s and waits for its time
    anchor = world.clock.now_utc()
    await world.message("在吗")
    await run(world, until=lambda: world.engine.snapshot().state == "SENDING")
    stored = RoundData.of(world.engine.snapshot()).outgoing
    assert stored is not None and stored.due_at == anchor + timedelta(seconds=DELAY)
    assert world.channel.texts == ["好的"]  # only the first reply so far
    await world.clock.advance(2)
    await world.message("等一下")  # inside her delay, with the draft waiting to be shown
    await run(world)
    assert world.channel.texts == ["好的", "好的"]  # one reply to both, not two
    assert world.writer.contexts[-1].user_text == "在吗\n等一下"
    assert world.writer.calls == 3  # the first reply, the dropped draft, the answer to both
    assert first_bubble(world, 2) - anchor == timedelta(seconds=DELAY)  # at her time, not before
    out = world.store.latest_reply()
    assert out and {"step": "generation_cancelled", "count": 1} in (out[0].actions or [])


async def test_a_reply_made_in_advance_survives_a_restart_and_appears_at_its_time(
    services: Services, clock: ManualClock
) -> None:
    """R17-34: ``Outgoing.due_at`` is stored: the process dies while the draft waits to be shown."""
    first = build_harness(
        services,
        clock,
        writer=make_writer(clock),
        pacing=reference_pacing(latency_s=60.0),
    )
    await first.engine.start()
    await warm_up(first)
    await first.clock.advance(3600)
    anchor = first.clock.now_utc()
    await first.message("在吗")
    await run(first, until=lambda: first.engine.snapshot().state == "SENDING")
    due = anchor + timedelta(seconds=DELAY)
    stored = RoundData.of(first.engine.snapshot()).outgoing
    assert stored is not None and stored.due_at == due and stored.reply_id is None
    assert clock.now_utc() < due  # the draft is ready and waits
    second = await restart(
        first,
        writer=ScriptedWriter(clock=clock, takes_s=GENERATION),
        pacing=reference_pacing(latency_s=60.0),
    )
    try:
        await run_to_idle(second.engine, clock)
        assert second.writer.calls == 0  # nothing is written again
        assert second.channel.texts == ["好的"]
        assert first_bubble(second) - anchor == timedelta(seconds=DELAY)  # at the same moment
    finally:
        await second.engine.stop()


async def test_a_reply_that_is_late_after_a_restart_waits_the_least_and_types(
    services: Services, clock: ManualClock
) -> None:
    """R17-34: the moment has passed while the process was away: a second, then the typing."""
    first = build_harness(
        services,
        clock,
        writer=make_writer(clock),
        pacing=reference_pacing(latency_s=60.0),
    )
    await first.engine.start()
    await warm_up(first)
    await first.clock.advance(3600)
    await first.message("在吗")
    await run(first, until=lambda: first.engine.snapshot().state == "SENDING")
    second = await restart(
        first,
        clock_jump_s=300,
        writer=ScriptedWriter(),
        pacing=reference_pacing(latency_s=60.0),
    )
    try:
        woke = clock.now_utc()
        await run_to_idle(second.engine, clock)
        assert second.writer.calls == 0 and second.channel.texts == ["好的"]
        assert first_bubble(second) - woke == timedelta(seconds=MIN_INTERVAL_S + TYPING)
    finally:
        await second.engine.stop()


async def test_a_pause_is_not_cut_short_by_the_lead(services: Services, clock: ManualClock) -> None:
    """R17-34: nothing is made while the pause lasts, however early the lead would begin."""
    harness = build_harness(
        services,
        clock,
        writer=make_writer(clock),
        pacing=reference_pacing(latency_s=60.0),
    )
    await harness.engine.start()
    try:
        await warm_up(harness)
        await clock.advance(3600)
        until = clock.now_utc() + timedelta(seconds=120)
        harness.services.runtime.set(ENGINE_PAUSED_UNTIL, until)
        await harness.message("在吗")
        await run(harness)
        began = harness.writer.began[1]
        assert timedelta(0) <= began - until < timedelta(milliseconds=10)  # not until - lead
        assert harness.channel.texts == ["好的", "好的"]
    finally:
        await harness.engine.stop()


async def test_a_message_after_the_moment_still_cancels_the_generation_as_before(
    world: Harness,
) -> None:
    """R17-34: past her delay the reply is "being written": a message starts the round again."""
    world.writer.gate = asyncio.Event()
    await world.message("在吗")
    await run(world, until=world.writer.started.is_set)  # cold start: begun at the end of the delay
    assert world.engine.snapshot().state == "GENERATING"
    await world.message("算了")
    await wait_until(lambda: world.writer.cancelled == 1)
    await wait_until(lambda: world.engine.snapshot().state == "COLLECTING")
    assert RoundData.of(world.engine.snapshot()).decision is None
