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
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.engine_harness import (
    START,
    Harness,
    ScriptedCrisis,
    build_harness,
    make_draft,
    run_to_idle,
    wait_for_state,
    waiting_for,
)
from tests.support.waiting import wait_until
from twin.commands.parse import parse_command
from twin.commands.registry import CommandCall, CommandRegistry, CommandSpec
from twin.commands.router import CommandRouter
from twin.engine.command_port import CommandContext, CommandOutcome
from twin.engine.machine import PROBE_RECHECK_S
from twin.engine.roundstate import RoundData
from twin.services import Services
from twin.storage.engine_models import BotTurn


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

    def recognises(self, text: str) -> bool:
        return parse_command(text) is not None  # the strict reading: no table here

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
    # an hour goes by: the round keeps waiting, one look a minute.  The clock moves one look at
    # a time, each after the engine is waiting again (``waiting_for``): a jump over many looks
    # left a look that was still in a thread on a loaded machine with a wake-up in the past
    for _ in range(int(3600 // PROBE_RECHECK_S)):
        await harness.clock.advance(PROBE_RECHECK_S)
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
    # R18-F13: not a fixed 50 ms - the command is decided inside handle_message; drain waits until
    # nothing is in flight or pending, so nothing can still be on its way to the channel
    await harness.settle()
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


# ------------------------------------------------- R18-B7 (D-754): chat that looks like a command


def router_with_status(ran: list[str]) -> CommandRouter:
    """The real router (its parser and table), with ``/状态`` the one command it knows."""

    async def status(call: CommandCall) -> str:
        ran.append(call.name)
        return "一切正常"

    registry = CommandRegistry()
    registry.register(
        CommandSpec("状态", "基本", "状态", "/状态", "/状态", status, aliases=("status",))
    )
    return CommandRouter(registry)


def inbound_rows(harness: Harness) -> dict[str, bool]:
    with harness.services.db.session() as session:
        rows = session.scalars(select(BotTurn).where(BotTurn.direction == "in"))
        return {row.text: row.is_command for row in rows}


async def test_r18_b7_chat_that_starts_with_a_slash_waits_in_the_round_during_a_probe(
    services: Services, clock: ManualClock, probe: Probe
) -> None:
    """R18-B7: before, every message that began with a slash was stored as a command and dropped
    while a probe ran - WeChat's "/::)" and "/ 吃饭" were never answered.  Now the router's own
    reading (the one ``handle`` uses) decides: chat waits in the round, a command is held."""
    ran: list[str] = []
    harness = build_harness(services, clock, probe_active=probe, commands=router_with_status(ran))
    await harness.engine.start()
    try:
        await harness.message("/状态")  # a command: stored, not run, not made up later (D-686)
        await harness.message("/::)")  # WeChat's old emoticon: chat
        await harness.message("/ 吃饭")  # a slash and a space before an unknown word: chat
        await held(harness)
        rows = inbound_rows(harness)
        assert rows == {"/状态": True, "/::)": False, "/ 吃饭": False}
        assert len(harness.engine.snapshot().pending) == 2 and ran == []
        probe.on = False
        await harness.engine.on_state_change(1, 2)
        await run_to_idle(harness.engine, harness.clock)
        assert harness.channel.texts == ["好的"] and harness.writer.calls == 1
        assert [item.text for item in harness.writer.contexts[0].inbound] == ["/::)", "/ 吃饭"]
        assert ran == []  # the command is not run after the probe either
    finally:
        await harness.engine.stop()


async def test_r18_b7_a_yes_without_an_open_question_is_chat_during_a_probe(
    services: Services, clock: ManualClock, probe: Probe
) -> None:
    """R18-B7: "是" is held only as the answer to an open question; otherwise it is chat."""

    class NoQuestion:
        async def is_confirmation(self, text: str, at: object) -> bool:
            return False  # nothing is open

        async def confirm(self, context: CommandContext) -> CommandOutcome | None:
            raise AssertionError("never asked without an open question")

        async def propose(self, answered: object) -> str | None:
            return None

        async def withdraw(self, asked_since: object) -> None:
            return None

    harness = build_harness(services, clock, probe_active=probe)
    harness.engine.attach_corrections(NoQuestion())
    await harness.engine.start()
    try:
        await harness.message("是")
        await held(harness)
        assert inbound_rows(harness) == {"是": False}
        probe.on = False
        await harness.engine.on_state_change(1, 2)
        await run_to_idle(harness.engine, harness.clock)
        assert harness.channel.texts == ["好的"]
    finally:
        await harness.engine.stop()


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.one_of(
        st.text(alphabet="/／ 　:：，,状态吃饭思考status)-8x", max_size=12),
        st.sampled_from(["/状态", "／状态", "/ 状态", "/status", "/状态，看看", "/::)", "/ 吃饭"]),
    )
)
async def test_r18_b7_the_probe_time_reading_is_the_reading_of_the_router(text: str) -> None:
    """R18-B7 (rule 8): ``recognises`` says "command" exactly when ``handle`` answers one."""
    router = router_with_status([])
    context = CommandContext(at=START, inbound_id="t")
    assert router.recognises(text) is ((await router.handle(text, context)) is not None)
