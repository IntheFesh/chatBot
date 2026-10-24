"""What the engine gives the proactive scheduler: a count of arrivals and a way to wait."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from tests.support.clock import ManualClock
from tests.support.engine_extras import ScriptedCommands
from tests.support.engine_harness import Harness, build_harness
from twin.engine.command_port import CommandOutcome
from twin.services import Services


@pytest.fixture
async def rig(services: Services, clock: ManualClock) -> AsyncIterator[Harness]:
    commands = ScriptedCommands(状态=CommandOutcome("⚙️ 一切正常"))
    harness = build_harness(services, clock, commands=commands)
    await harness.engine.start()
    yield harness
    await harness.engine.stop()


async def test_every_message_of_the_user_is_one_arrival_and_a_command_is_none(
    rig: Harness,
) -> None:
    assert rig.engine.arrivals == 0
    await rig.message("在吗")
    assert rig.engine.arrivals == 1
    await rig.message("/状态")  # a command is answered at once and is no one talking to her
    assert rig.engine.arrivals == 1
    await rig.message("还在吗")
    assert rig.engine.arrivals == 2


async def test_waiting_ends_the_moment_a_message_arrives(rig: Harness) -> None:
    since = rig.engine.arrivals
    waiting = asyncio.ensure_future(rig.engine.wait_for_arrival(since, 600.0))
    await rig.clock.settle()
    assert not waiting.done()
    await rig.message("在吗")
    assert await asyncio.wait_for(waiting, 5.0) is True


async def test_waiting_ends_at_the_time_when_nobody_writes(rig: Harness) -> None:
    waiting = asyncio.ensure_future(rig.engine.wait_for_arrival(rig.engine.arrivals, 30.0))
    await rig.clock.settle()
    await rig.clock.advance(29.0)
    await rig.clock.settle()
    assert not waiting.done()
    await rig.clock.advance(2.0)
    assert await asyncio.wait_for(waiting, 5.0) is False


async def test_a_message_that_came_before_the_wait_began_counts_at_once(rig: Harness) -> None:
    since = rig.engine.arrivals
    await rig.message("在吗")
    assert await rig.engine.wait_for_arrival(since, 600.0) is True


async def test_no_time_to_wait_means_no_wait(rig: Harness) -> None:
    assert await rig.engine.wait_for_arrival(rig.engine.arrivals, 0.0) is False
    assert await rig.engine.wait_for_arrival(rig.engine.arrivals, -5.0) is False


async def test_the_snapshot_is_the_stored_state(rig: Harness) -> None:
    snapshot = rig.engine.snapshot()
    assert snapshot.state == "IDLE" and snapshot.pending == ()
