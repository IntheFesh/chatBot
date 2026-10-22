"""A channel probe is running: the bot says nothing of its own (R17-4, D-684..D-686; R-CH-009).

The M0 probe counts what the platform lets the bot send after a message of the user.  Every reply,
every message of her own and every answer to a command would be counted with it and spoil the
measurement.  So while a probe plan runs, in the application of ``twin run`` on the WeChat channel
(the platform and the phone are made up):

* the user's messages are stored but not answered, and not read as chat: a command is not run;
* she writes nothing first: no greeting, no meal, no good night - the log says why;
* everything that reaches the phone is a ``[测试]`` message of the probe;
* the one exception is a crisis (R-SAFE-001): the answer goes out, out of the role, and the attempt
  it disturbed is void - the probe says so and asks for a fresh message;
* when the probe ends the round that waited is decided afresh and answered after a short pause.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from tests.integration.life.conftest import WorldFactory
from tests.support.life_checks import assert_bot_text_not_in_her_data
from tests.support.life_world import LifeWorld
from tests.support.proactive_world import opening_curve, proactive_model
from twin.channel.base import TEST_PREFIX
from twin.channel.probe.model import PlanStatus, ProbeOptions
from twin.channel.probe.store import ProbeStore
from twin.services import Services

pytestmark = pytest.mark.integration

FRIDAY = date(2026, 10, 9)
PEAKS = {48: 0.6, 49: 0.6, 73: 0.6, 74: 0.6}  # lunch and dinner
CRISIS = "我真的不想活了"


def let_her_chase(services: Services) -> None:
    services.settings.proactive.max_chase = 6


async def probe_world(make_world: WorldFactory) -> LifeWorld:
    return await make_world(
        datetime(2026, 10, 9, 15, 0, tzinfo=UTC),  # Friday 10:00 in Chicago
        model=proactive_model(opening_curve(base=0.02, peaks=PEAKS)),
        platform="ilink",
        platform_quota=40,  # the probe sends past the safe count: the platform must let it
        quota=30,
        configure=let_her_chase,
    )


def phone(world: LifeWorld) -> list[str]:
    assert world.double is not None
    return world.double.texts()


def of_the_bot(world: LifeWorld) -> list[str]:
    """What the phone shows that the probe did not send."""
    return [text for text in phone(world) if not text.startswith(TEST_PREFIX)]


async def test_without_a_probe_she_answers_and_writes_first_in_the_same_story(
    make_world: WorldFactory,
) -> None:
    """The control: the story of the next test, with no probe, is not silent (so it proves it)."""
    world = await probe_world(make_world)
    world.deepseek.book.when("在吗", "我在呀")
    await world.say("在吗")
    await world.run_until(world.local(21, 0))
    assert of_the_bot(world)[0] == "我在呀"
    assert world.proactive_rows(outcomes=["sent"])
    assert world.deepseek.calls["reply"] >= 1


async def test_the_bot_is_silent_while_the_probe_runs_and_answers_when_it_ends(
    make_world: WorldFactory,
) -> None:
    """R17-4: replies, her own messages and the answers to commands are held for the probe."""
    world = await probe_world(make_world)
    world.deepseek.book.when("在吗", "我在呀")
    store = ProbeStore(world.services.db, world.clock)
    store.create(ProbeOptions(max_messages=3))

    # ---- the user writes (it is the fresh message the probe asked for), and the day goes by ------
    await world.say("在吗")
    await world.run_until(world.local(21, 0))  # eleven hours: lunch and dinner come and go
    shown = phone(world)
    assert shown and all(text.startswith(TEST_PREFIX) for text in shown), shown
    assert world.deepseek.calls["reply"] == 0 and world.deepseek.calls["proactive"] == 0
    assert world.proactive_rows(outcomes=["sent"]) == []
    snapshot = world.assembly.engine.engine.snapshot()
    assert snapshot.state == "COLLECTING" and len(snapshot.pending) == 1  # held, not lost

    # ---- the status of the probe says it, from another process ---------------------------------
    code, output = world.cli("channel", "probe", "status")
    assert code == 0 and "bot PAUSED" in output

    # ---- a command during the probe is stored and neither run nor answered -----------------------
    await world.say("/状态")
    await world.run_for(minutes=5)
    assert all(text.startswith(TEST_PREFIX) for text in phone(world))
    assert len(world.assembly.engine.engine.snapshot().pending) == 1  # the command is not chat

    # ---- the probe ends: the round that waited is decided afresh and answered --------------------
    ended = world.now
    store.finish(PlanStatus.STOPPED, "the scenario ends it")
    await world.run_for(minutes=10)
    assert of_the_bot(world) == ["我在呀"]  # once, and nothing for the command
    answered = [s for s in world.persona_said if not s.text.startswith(TEST_PREFIX)]
    assert ended < answered[0].at <= ended + timedelta(minutes=10)
    assert world.deepseek.calls["reply"] == 1
    assert world.system_said == [] and world.assembly.engine.engine.snapshot().state == "IDLE"
    assert_bot_text_not_in_her_data(world)
    assert world.deepseek.unexpected == []


async def test_she_does_not_write_first_while_the_probe_runs_even_if_he_is_quiet(
    make_world: WorldFactory,
) -> None:
    """R17-4: the proactive rule (R-PRO-003) refuses with ``probe_running``; the log says so."""
    world = await probe_world(make_world)
    store = ProbeStore(world.services.db, world.clock)
    store.create(ProbeOptions(max_messages=3))
    await world.run_until(world.local(14, 0))  # the probe waits for a message of his; lunch passes
    assert all(text.startswith(TEST_PREFIX) for text in phone(world))
    assert world.proactive_rows(outcomes=["sent"]) == []
    refused = {r.reason for r in world.proactive_rows(outcomes=["rejected"])}
    assert "probe_running" in refused  # the log says why she did not write

    # ---- the probe is over: she writes first again -------------------------------------------
    store.finish(PlanStatus.STOPPED, "the scenario ends it")
    ended = world.now
    await world.run_until(world.local(23, 30))
    later = [r for r in world.proactive_rows(outcomes=["sent"]) if r.at > ended]
    assert later, "she never wrote first again"
    assert_bot_text_not_in_her_data(world)


async def test_a_crisis_is_answered_during_the_probe_and_the_attempt_is_void(
    make_world: WorldFactory,
) -> None:
    """R17-4 (D4): safety wins over the measurement; the probe says the attempt must be redone."""
    world = await probe_world(make_world)
    world.deepseek.book.when("在吗", "我在呀")
    store = ProbeStore(world.services.db, world.clock)
    world.watch.add("probe/runner.py")  # the probe sets the pace of this story
    store.create(ProbeOptions(max_messages=3))
    await world.run_for(seconds=30)  # it announces the step and waits for a message of his
    await world.say("在吗")
    await world.run_for(seconds=30)  # the attempt is under way: the first test message is out
    plan = store.load()
    assert plan is not None and plan.steps[0].attempts[0].outcome is None
    assert plan.steps[0].attempts[0].phase.value == "running"

    await world.say(CRISIS)
    await world.run_for(minutes=2)
    spoken = "\n".join(of_the_bot(world))
    assert "988" in spoken and "我想先停一下" in spoken  # out of the role, with the help line
    assert world.deepseek.calls["reply"] == 0 and world.deepseek.calls["crisis"] == 1
    assert "我在呀" not in spoken  # the persona stays quiet, only the safety answer went out

    plan = store.load()
    assert plan is not None and plan.status is PlanStatus.RUNNING
    first = plan.steps[0].attempts[0]
    assert first.outcome == "void" and first.void_reason is not None
    assert "safety" in first.void_reason
    assert len(plan.steps[0].attempts) == 2  # the probe asks for a fresh message again
    assert any(entry["decision"] == "urgent" for entry in store.audit())
    code, output = world.cli("channel", "probe", "status")
    assert code == 0 and "void" in output and "safety" in output

    # ---- and the rest of the day is as silent as before --------------------------------------
    world.watch.discard("probe/runner.py")
    answers = len(of_the_bot(world))
    await world.run_until(world.local(21, 0))
    assert len(of_the_bot(world)) == answers
    assert world.proactive_rows(outcomes=["sent"]) == []
    assert_bot_text_not_in_her_data(world)
