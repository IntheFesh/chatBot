"""A running channel probe has the channel to itself: the engine holds (R17-4, D-684, D-685).

The M0 probe counts what the platform lets the bot send after a message of the user.  While a plan
runs (``probe_active``) the engine stores what the user writes, still screens it for a crisis, and
then waits: no decision, no draft, no bubble, no answer to a command.  When the probe ends the round
is timed afresh - she has "just seen" the messages - and answered once.  The answer to a crisis is
the one send that does not wait, and it goes out as an urgent message of the channel.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest

from tests.support.clock import ManualClock
from tests.support.engine_harness import (
    Harness,
    ScriptedCrisis,
    build_harness,
    make_draft,
    run_to_idle,
    wait_for_state,
    waiting_for,
)
from tests.support.waiting import wait_until
from twin.engine.command_port import CommandContext, CommandOutcome
from twin.engine.machine import PROBE_RECHECK_S
from twin.engine.roundstate import RoundData
from twin.services import Services


class Probe:
    """The switch the engine asks: is a probe running?"""

    def __init__(self) -> None:
        self.on = True
        self.asked = 0

    def __call__(self) -> bool:
        self.asked += 1
        return self.on


class RecordingCommands:
    """A command router that records what it is asked to run."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def handle(self, text: str, context: CommandContext) -> CommandOutcome | None:
        self.calls.append(text)
        return CommandOutcome("⚙️ 好的")


@pytest.fixture
def probe() -> Probe:
    return Probe()


@pytest.fixture
async def rig(
    services: Services, clock: ManualClock, probe: Probe
) -> AsyncIterator[tuple[Harness, RecordingCommands]]:
    commands = RecordingCommands()
    harness = build_harness(
        services,
        clock,
        probe_active=probe,
        commands=commands,
        crisis=ScriptedCrisis("不想活"),
    )
    await harness.engine.start()
    yield harness, commands
    await harness.engine.stop()


async def held(harness: Harness) -> None:
    """Run until the round is waiting for the probe (its look-again wait is the 60 s one)."""
    await run_to_idle(
        harness.engine,
        harness.clock,
        until=lambda: (
            harness.engine.snapshot().state == "COLLECTING"
            and harness.clock.now_utc()
            > harness.engine.snapshot().last_inbound_at + timedelta(seconds=20)
        ),
    )
    await waiting_for(harness.engine, harness.clock.now_utc() + timedelta(seconds=PROBE_RECHECK_S))


async def test_a_message_is_stored_not_answered_and_answered_soon_after_the_probe(
    rig: tuple[Harness, RecordingCommands], probe: Probe
) -> None:
    """The user's message waits in COLLECTING for as long as the probe runs, however long."""
    harness, _ = rig
    await harness.message("在吗")
    await held(harness)
    for _ in range(3):  # an hour and more goes by: the round keeps waiting, one look a minute
        await harness.clock.advance(1200)
        await waiting_for(
            harness.engine, harness.clock.now_utc() + timedelta(seconds=PROBE_RECHECK_S)
        )
    snapshot = harness.engine.snapshot()
    assert snapshot.state == "COLLECTING" and len(snapshot.pending) == 1
    assert harness.writer.calls == 0 and harness.channel.out == []
    assert RoundData.of(snapshot).decision is None

    probe.on = False  # the probe ends: the state watcher tells the engine
    ended = harness.clock.now_utc()
    await harness.engine.on_state_change(1, 2)
    await run_to_idle(harness.engine, harness.clock)
    first = next(o for o in harness.channel.out if o.kind == "text")
    assert first.at - ended < timedelta(seconds=10)  # "just seen", not as if it were old news
    assert harness.channel.texts == ["好的"] and harness.writer.calls == 1


async def test_a_second_message_during_the_probe_joins_the_same_round(
    rig: tuple[Harness, RecordingCommands], probe: Probe
) -> None:
    harness, _ = rig
    await harness.message("在吗")
    await held(harness)
    await harness.message("还在吗")
    await held(harness)
    assert len(harness.engine.snapshot().pending) == 2 and harness.writer.calls == 0
    probe.on = False
    await harness.engine.on_state_change(1, 2)
    await run_to_idle(harness.engine, harness.clock)
    assert harness.writer.calls == 1 and harness.channel.texts == ["好的"]  # one answer for both


async def test_a_decision_made_before_the_probe_is_taken_back(
    rig: tuple[Harness, RecordingCommands], probe: Probe
) -> None:
    """The probe starts while she waits for her time to answer: the planned time is dropped."""
    harness, _ = rig
    probe.on = False
    await harness.message("在吗")
    await run_to_idle(
        harness.engine,
        harness.clock,
        until=lambda: harness.engine.snapshot().state == "DECIDING",
    )
    planned = harness.engine.snapshot().planned_send_at
    assert planned is not None
    await waiting_for(harness.engine, planned)
    probe.on = True  # the probe plan is stored (the CLI bumps the state version)
    await harness.engine.on_state_change(1, 2)
    await wait_for_state(harness.engine, "COLLECTING")
    snapshot = harness.engine.snapshot()
    assert snapshot.planned_send_at is None and RoundData.of(snapshot).decision is None
    await harness.clock.advance(7200)  # past the time she had planned
    assert harness.writer.calls == 0 and harness.channel.out == []
    probe.on = False
    await harness.engine.on_state_change(1, 2)
    await run_to_idle(harness.engine, harness.clock)
    assert harness.channel.texts == ["好的"]


async def test_a_reply_that_is_ready_when_the_probe_starts_is_not_sent(
    rig: tuple[Harness, RecordingCommands], probe: Probe
) -> None:
    """The first bubble is where she can still hold back (like ``/暂停``): nothing is sent."""
    harness, _ = rig
    probe.on = False
    harness.writer.gate = asyncio.Event()
    harness.writer.add(make_draft("好呀"), make_draft("来啦"))
    await harness.message("在吗")
    await run_to_idle(harness.engine, harness.clock, until=harness.writer.started.is_set)
    probe.on = True  # the probe starts while the reply is being written
    harness.writer.gate.set()
    await wait_until(
        lambda: (
            harness.engine.snapshot().state == "COLLECTING"
            and harness.writer.calls == 1
            and harness.engine.waiting_until is not None
        ),
        limit_s=10.0,
        interval=0.002,
    )
    assert harness.channel.out == []  # the ready reply was not sent
    probe.on = False
    await harness.engine.on_state_change(1, 2)
    await run_to_idle(harness.engine, harness.clock)
    assert harness.channel.texts == ["来啦"] and harness.writer.calls == 2


async def test_a_command_during_the_probe_is_stored_but_neither_run_nor_answered(
    rig: tuple[Harness, RecordingCommands], probe: Probe
) -> None:
    harness, commands = rig
    await harness.message("/状态")
    await asyncio.sleep(0.05)
    assert commands.calls == [] and harness.channel.out == []
    assert harness.engine.snapshot().pending == ()  # it is not chat either
    probe.on = False
    await harness.message("/状态")  # after the probe it is a command again
    assert commands.calls == ["/状态"] and harness.channel.texts == ["⚙️ 好的"]


async def test_a_crisis_is_answered_during_the_probe_and_as_an_urgent_send(
    rig: tuple[Harness, RecordingCommands], probe: Probe
) -> None:
    """R-SAFE-001 beats the measurement (D4): the answer is not held; the channel is told it is
    urgent (it lets it through and voids the probe's attempt)."""
    harness, _ = rig
    await harness.message("我真的不想活了")
    await run_to_idle(harness.engine, harness.clock)
    assert harness.channel.texts == ["我想先停一下。", "你现在还好吗？", "也可以联系 988。"]
    assert harness.channel.urgent == harness.channel.texts  # every bubble of it
    assert harness.writer.calls == 0
    assert harness.engine.snapshot().state == "IDLE"


async def test_an_ordinary_reply_is_never_urgent(services: Services, clock: ManualClock) -> None:
    harness = build_harness(services, clock)
    await harness.engine.start()
    try:
        await harness.message("在吗")
        await run_to_idle(harness.engine, harness.clock)
        assert harness.channel.texts == ["好的"] and harness.channel.urgent == []
    finally:
        await harness.engine.stop()


async def test_without_a_probe_source_the_engine_holds_nothing(
    services: Services, clock: ManualClock
) -> None:
    harness = build_harness(services, clock)  # no ``probe_active`` (the console channel)
    await harness.engine.start()
    try:
        await harness.message("在吗")
        await run_to_idle(harness.engine, harness.clock)
        assert harness.channel.texts == ["好的"]
    finally:
        await harness.engine.stop()
