"""A reply that waits when the time zone is switched (R-ENG-003, R-SCH-004, D-233 1, D-722).

The reply was drawn for the plan of that moment.  ``/时区`` changes the plan under it: a reply due
in ten minutes may now fall into her deep sleep.  The engine looks at her state again on every
pass of DECIDING and before the generation starts (``ConversationEngine._rejudge``) and queues
the reply to her waking plus the jitter, exactly like a message that arrives while she sleeps.
The day in these tests is :class:`~tests.support.engine_harness.ZonedDay`: asleep from midnight
to 07:00 by the clock of a zone that the test flips; 16:00 UTC is 11:00 in Chicago and midnight
in Shanghai.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from tests.support.clock import ManualClock
from tests.support.engine_harness import (
    START,
    Harness,
    ZonedDay,
    build_harness,
    decision_of,
    make_draft,
    reference_pacing,
    run_to_idle,
    waiting_for,
)
from tests.support.waiting import wait_until
from twin.engine.decision import Decision
from twin.services import Services

CHICAGO, SHANGHAI = "America/Chicago", "Asia/Shanghai"
FRIDAY_16 = START + timedelta(hours=4)  # 11:00 in Chicago, 00:00 (Saturday) in Shanghai
SHANGHAI_WAKE = datetime(2026, 10, 9, 23, 0, tzinfo=UTC)  # 07:00 Saturday in Shanghai
SLOW = reference_pacing(latency_s=600.0)  # her usual answer takes ten minutes


@pytest.fixture
async def world(services: Services, clock: ManualClock) -> AsyncIterator[Harness]:
    clock.set_time(FRIDAY_16)
    harness = build_harness(services, clock, day=ZonedDay(CHICAGO), pacing=SLOW)
    await harness.engine.start()
    yield harness
    await harness.engine.stop()


def moved(harness: Harness) -> bool:
    decision = decision_of(harness.engine)
    return decision is not None and decision.woke_up


async def switch_zone(harness: Harness, name: str) -> None:
    """``/时区``: her day moves with the zone, and the engine hears of it like of any setting."""
    assert isinstance(harness.day, ZonedDay)
    harness.day.zone = name
    await harness.engine.on_state_change(0, 1)


async def wait_for_the_reply_to_be_planned(harness: Harness) -> datetime:
    await harness.message("在吗")
    await run_to_idle(
        harness.engine, harness.clock, until=lambda: harness.engine.snapshot().state == "DECIDING"
    )
    planned = harness.engine.snapshot().planned_send_at
    assert planned is not None
    return planned


async def test_a_reply_waiting_for_its_time_waits_for_her_waking_after_a_switch_puts_her_to_sleep(
    world: Harness,
) -> None:
    """R17-32: the reply drawn for 11:10 in Chicago is not sent at 00:10 in Shanghai."""
    planned = await wait_for_the_reply_to_be_planned(world)
    before = decision_of(world.engine)
    assert before is not None and before.mode == "free" and not before.woke_up
    assert timedelta(minutes=10) <= planned - FRIDAY_16 <= timedelta(minutes=11)
    await waiting_for(world.engine, planned)
    await world.clock.advance(300)  # 11:05 in Chicago: halfway through her wait
    await switch_zone(world, SHANGHAI)  # 00:05 of the next night there
    await wait_until(lambda: moved(world))
    queued = decision_of(world.engine)
    assert queued is not None and queued.mode == "asleep" and queued.wake_at == SHANGHAI_WAKE
    assert (
        SHANGHAI_WAKE + timedelta(minutes=5)
        <= queued.send_at
        <= SHANGHAI_WAKE + timedelta(minutes=40)
    )
    assert world.engine.snapshot().planned_send_at == queued.send_at > planned
    assert world.channel.texts == [] and world.writer.calls == 0

    await run_to_idle(world.engine, world.clock)
    sent = next(o for o in world.channel.out if o.kind == "text")
    assert sent.at >= queued.send_at  # not one bubble before she is up
    assert world.writer.contexts[0].woke_up  # and she says she has just woken
    assert world.channel.texts == ["好的"]
    assert world.engine.snapshot().state == "IDLE"


async def test_without_a_switch_the_reply_goes_out_at_its_planned_time(world: Harness) -> None:
    """R17-32 (the control): nothing changes for a reply whose plan did not change."""
    planned = await wait_for_the_reply_to_be_planned(world)
    await waiting_for(world.engine, planned)
    await world.clock.advance(300)
    await world.engine.on_state_change(0, 1)  # a setting changed - but not the plan
    await wait_until(
        lambda: not world.engine.pending_wake and world.engine.waiting_until == planned
    )
    assert decision_of(world.engine) is not None and not moved(world)  # looked at, left as it was
    await run_to_idle(world.engine, world.clock)
    sent = next(o for o in world.channel.out if o.kind == "text")
    assert abs((sent.at - planned).total_seconds()) <= 5.0  # at her time, in Chicago's morning
    assert not world.writer.contexts[0].woke_up
    assert world.channel.texts == ["好的"]


async def test_a_retry_that_a_switch_puts_into_her_sleep_waits_for_her_waking_too(
    world: Harness,
) -> None:
    """R17-32 (R-ENG-010): the late retry of a failed reply is moved like the first try."""
    world.writer.add(make_draft(fallback="backend_error"), make_draft("早呀"))
    await world.message("在吗")
    await run_to_idle(
        world.engine,
        world.clock,
        until=lambda: (decision_of(world.engine) or Decision("free", START, START)).mode == "retry",
    )
    retry = decision_of(world.engine)
    assert retry is not None and retry.mode == "retry" and world.writer.calls == 1
    await switch_zone(world, SHANGHAI)  # the retry (2 to 10 minutes from now) falls into the night
    await wait_until(lambda: moved(world))
    queued = decision_of(world.engine)
    assert queued is not None and queued.mode == "asleep" and queued.wake_at == SHANGHAI_WAKE
    assert queued.send_at >= SHANGHAI_WAKE + timedelta(minutes=5)
    assert world.channel.texts == []
    await run_to_idle(world.engine, world.clock)
    sent = next(o for o in world.channel.out if o.kind == "text")
    assert sent.at >= queued.send_at and world.channel.texts == ["早呀"]
    assert world.writer.calls == 2  # the second try was made after she woke


async def test_a_reply_queued_for_her_waking_keeps_its_time_when_a_switch_wakes_her(
    services: Services, clock: ManualClock
) -> None:
    """R17-32: only the way into sleep is looked at; a queue keeps its absolute time (D-233 1)."""
    clock.set_time(FRIDAY_16)
    harness = build_harness(services, clock, day=ZonedDay(SHANGHAI), pacing=SLOW)
    await harness.engine.start()
    try:
        planned = await wait_for_the_reply_to_be_planned(harness)
        queued = decision_of(harness.engine)
        assert queued is not None and queued.woke_up and planned >= SHANGHAI_WAKE
        await waiting_for(harness.engine, planned)
        await switch_zone(harness, CHICAGO)  # it is 11:00 in Chicago: she is up
        await wait_until(
            lambda: not harness.engine.pending_wake and harness.engine.waiting_until == planned
        )  # the engine looked again, and went on waiting for the same moment
        assert harness.engine.snapshot().planned_send_at == planned  # the time stands
        assert harness.channel.texts == []
        await run_to_idle(harness.engine, clock)
        sent = next(o for o in harness.channel.out if o.kind == "text")
        assert sent.at >= planned
    finally:
        await harness.engine.stop()


async def test_a_time_that_was_moved_once_is_not_moved_again_and_the_driver_does_not_spin(
    services: Services, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R17-32: a decision that again lands in her sleep (a pause that ends at the edge of it, say)
    is looked at once; the second look does not start a third draw, and the reply goes out."""
    clock.set_time(FRIDAY_16)
    harness = build_harness(services, clock, day=ZonedDay(SHANGHAI), pacing=SLOW)
    drawn: list[datetime] = []
    real = harness.engine._decider.decide  # type: ignore[attr-defined]

    def stubborn(pacing: object, situation: object) -> Decision:
        decision = real(pacing, situation)  # type: ignore[arg-type]
        drawn.append(decision.send_at)
        # whatever she would do, the answer is ten minutes from now - and that is in her sleep
        return replace(
            decision,
            mode="free",
            woke_up=False,
            wake_at=None,
            send_at=harness.clock.now_utc() + timedelta(minutes=10),
        )

    monkeypatch.setattr(harness.engine._decider, "decide", stubborn)  # type: ignore[attr-defined]
    await harness.engine.start()
    try:
        await wait_for_the_reply_to_be_planned(harness)
        await run_to_idle(harness.engine, clock)
        assert harness.channel.texts == ["好的"]
        assert len(drawn) == 2  # the first draw and the one look of ``_rejudge``; no third
    finally:
        await harness.engine.stop()
