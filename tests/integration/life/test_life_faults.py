"""Faults in the middle of a conversation never end the application (R-ARCH-004, R-ENG-010).

"An exception in the handling of one conversation must not make the main loop exit; it is logged,
raised as an alert, and degraded to a delayed reply."  The application of ``twin run`` is made to
meet the faults a long run meets, and the conversation goes on after each:

* **the model is down** (HTTP 503 on every request): the reply is tried again later, three times,
  minutes apart; then one short answer of her own words and an alert - never an error text, and
  when the model is back the next message is answered as always;
* **the generation raises** (a bug in the prompt, a corrupt example library): it is the same as a
  model that is down - the reply is tried again later, and in the end she says one short word;
* **a step of the engine's loop raises** (a bug, a full disk, a corrupt row): the loop catches it,
  raises the alert ``engine_error`` (with the class of the error, not its message), waits a few
  seconds and does the step again - the reply comes as if nothing had happened;
* **a step raises again and again**: after five failures in a row the round is given up with a
  critical alert, the conversation is idle again - and the next message is answered normally;
* **the proactive planner's model is down**: she writes nothing on her own, the schedule goes on,
  and the next candidate is tried when the model is back.

What the user sees is only ever her own words (or silence); the details are in the alerts and the
log.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select

from tests.integration.life.conftest import WorldFactory
from tests.support.life_checks import (
    assert_bot_text_not_in_her_data,
    assert_clean_screen,
    assert_never_in_deep_sleep,
    assert_screen_matches_records,
)
from tests.support.life_world import LifeWorld
from tests.support.proactive_world import opening_curve, proactive_model
from twin.engine.machine import ConversationEngine
from twin.engine.pipeline import ReplyPipeline
from twin.storage.models import Alert

pytestmark = pytest.mark.integration

HER_ANSWERS = ("好的呀", "嗯嗯嗯", "知道啦")
SECRET = "LEAK-MARKER-4711 /home/user/secret.db"  # what an exception message could carry


async def faulty_world(make_world: WorldFactory, **options: Any) -> LifeWorld:
    return await make_world(
        datetime(2026, 10, 9, 16, 0, tzinfo=UTC),  # 11:00 in Chicago, she is up
        model=proactive_model(opening_curve(base=0.02)),
        her_answers=HER_ANSWERS,
        **options,
    )


def alerts_of(world: LifeWorld, category: str) -> list[Alert]:
    with world.services.db.session() as session:
        rows = list(session.scalars(select(Alert).where(Alert.category == category)))
        session.expunge_all()
    return rows


def nothing_of_the_trouble_is_shown(world: LifeWorld) -> None:
    assert_clean_screen(world)
    for item in world.said:
        assert SECRET not in item.text and "LEAK-MARKER-4711" not in item.text
    for row in world.rows():
        assert "LEAK-MARKER-4711" not in row.text
    with world.services.db.session() as session:
        for alert in session.scalars(select(Alert)):
            # the class of the error is in an alert, not its text
            assert "LEAK-MARKER-4711" not in f"{alert.title} {alert.detail}"


async def test_the_model_down_ends_in_one_natural_answer_and_then_all_is_well(
    make_world: WorldFactory,
) -> None:
    world = await faulty_world(make_world)
    world.deepseek.book.when("在吗", "在呀")
    world.deepseek.fail("reply", 503, times=500)

    await world.say("在吗")
    asked = world.now
    await world.run_until_idle()
    said = world.persona_said
    assert len(said) == 1 and said[0].text in HER_ANSWERS
    assert said[0].at - asked >= timedelta(minutes=3 * 2)  # three late tries, 2 to 10 minutes apart
    assert world.deepseek.calls["reply"] >= 4  # the first try and the three late ones
    row = world.out_rows()[0]
    assert row.backend == "fallback"
    assert ("reply_failed", "warning") in world.alerts()
    assert not alerts_of(world, "engine_error")  # a failing model is not a failing engine

    # ---- the model is back: the next message is answered as always -------------------------------
    world.deepseek.failures.clear()
    await world.say("在吗")
    await world.run_until_idle()
    assert world.persona_said[-1].text == "在呀" and world.out_rows()[-1].backend == "deepseek"
    nothing_of_the_trouble_is_shown(world)
    assert_screen_matches_records(world)
    assert_never_in_deep_sleep(world)
    assert_bot_text_not_in_her_data(world)
    assert world.deepseek.unexpected == []


async def test_a_generation_that_raises_is_a_late_reply_not_a_dead_loop(
    make_world: WorldFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = await faulty_world(make_world)
    world.deepseek.book.when("在吗", "在呀")
    original = ReplyPipeline.run
    raised: list[datetime] = []

    async def flaky(self: ReplyPipeline, *args: Any, **kwargs: Any) -> Any:
        if len(raised) < 2:
            raised.append(world.now)
            raise RuntimeError(f"example library is malformed: {SECRET}")
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(ReplyPipeline, "run", flaky)
    await world.say("在吗")
    asked = world.now
    await world.run_until_idle()
    assert len(raised) == 2 and world.deepseek.calls["reply"] == 1  # the third time it worked
    assert raised[1] - raised[0] >= timedelta(minutes=2)  # a late try is 2 to 10 minutes later
    assert [s.text for s in world.persona_said] == ["在呀"]  # said once, late
    assert world.persona_said[0].at - asked >= timedelta(minutes=4)
    assert not alerts_of(world, "reply_failed")  # it did not come to the last resort
    assert not alerts_of(world, "engine_error")  # and the loop did not even notice
    nothing_of_the_trouble_is_shown(world)
    assert_screen_matches_records(world)
    assert_bot_text_not_in_her_data(world)
    assert world.deepseek.unexpected == []


async def test_a_generation_that_always_raises_ends_in_one_short_word_and_an_alert(
    make_world: WorldFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = await faulty_world(make_world)
    world.deepseek.book.when("在吗", "在呀")
    original = ReplyPipeline.run
    failing = True
    raised: list[datetime] = []

    async def broken(self: ReplyPipeline, *args: Any, **kwargs: Any) -> Any:
        if failing:
            raised.append(world.now)
            raise RuntimeError(f"cannot read the example library: {SECRET}")
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(ReplyPipeline, "run", broken)
    await world.say("在吗")
    await world.run_until_idle()
    assert len(raised) == 4  # the first try and the three late ones
    said = world.persona_said
    assert len(said) == 1 and said[0].text in HER_ANSWERS
    assert ("reply_failed", "warning") in world.alerts()
    assert world.out_rows()[0].backend == "fallback"

    # ---- the fault is gone: the next message is answered -----------------------------------------
    failing = False
    await world.say("在吗")
    await world.run_until_idle()
    assert world.persona_said[-1].text == "在呀"
    nothing_of_the_trouble_is_shown(world)
    assert_screen_matches_records(world)
    assert_bot_text_not_in_her_data(world)
    assert world.deepseek.unexpected == []


async def test_a_step_of_the_loop_that_raises_is_done_again(
    make_world: WorldFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = await faulty_world(make_world)
    world.deepseek.book.when("在吗", "在呀")
    original = ConversationEngine._quiet_window_s
    raised: list[datetime] = []

    async def flaky(self: ConversationEngine) -> float:
        if len(raised) < 2:
            raised.append(world.now)
            raise RuntimeError(f"disk image is malformed: {SECRET}")
        return await original(self)

    monkeypatch.setattr(ConversationEngine, "_quiet_window_s", flaky)
    await world.say("在吗")
    await world.run_until_idle()
    assert len(raised) == 2
    assert raised[1] - raised[0] >= timedelta(seconds=5)  # a pause between the tries
    assert [s.text for s in world.persona_said] == ["在呀"]  # said once
    alerts = alerts_of(world, "engine_error")
    assert alerts and all(a.severity == "warning" for a in alerts)
    assert alerts[0].detail == {"error": "RuntimeError"}  # the class of the error, nothing else
    assert world.assembly.engine.health().status.name == "OK"
    nothing_of_the_trouble_is_shown(world)
    assert_screen_matches_records(world)
    assert_bot_text_not_in_her_data(world)
    assert world.deepseek.unexpected == []


async def test_a_step_that_always_raises_gives_the_round_up_and_the_loop_lives_on(
    make_world: WorldFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = await faulty_world(make_world)
    world.deepseek.book.when("在吗", "在呀")
    original = ConversationEngine._quiet_window_s
    failing = True
    raised: list[datetime] = []

    async def broken(self: ConversationEngine) -> float:
        if failing:
            raised.append(world.now)
            raise RuntimeError(f"cannot read the settings: {SECRET}")
        return await original(self)

    monkeypatch.setattr(ConversationEngine, "_quiet_window_s", broken)
    await world.say("在吗")
    await world.run_until_idle()
    assert len(raised) == 5  # five failures in a row, then the round is given up
    critical = [a for a in alerts_of(world, "engine_error") if a.severity == "critical"]
    assert len(critical) == 1 and critical[0].detail == {"error": "RuntimeError"}
    assert world.persona_said == []  # nothing was said - and nothing about the trouble either
    assert world.idle() and world.assembly.engine.health().status.name == "OK"

    # ---- the fault is gone: the next message is answered -----------------------------------------
    failing = False
    await world.say("在吗")
    await world.run_until_idle()
    assert [s.text for s in world.persona_said] == ["在呀"]
    nothing_of_the_trouble_is_shown(world)
    assert_screen_matches_records(world)
    assert_never_in_deep_sleep(world)
    assert_bot_text_not_in_her_data(world)
    assert world.deepseek.unexpected == []


async def test_the_planner_down_means_silence_and_the_schedule_goes_on(
    make_world: WorldFactory,
) -> None:
    world = await make_world(
        datetime(2026, 10, 9, 15, 0, tzinfo=UTC),  # 10:00 in Chicago
        model=proactive_model(opening_curve(base=0.02, peaks={48: 0.6, 49: 0.6, 73: 0.6, 74: 0.6})),
        her_answers=HER_ANSWERS,
    )
    world.deepseek.fail("proactive", 503, times=500)
    await world.say("我去上课啦")  # the platform lets her write first after a message of his

    await world.run_until(world.local(15, 0))  # lunch and the hours after it
    assert world.deepseek.calls["proactive"] >= 1  # the planner was asked, and failed
    assert world.proactive_rows(outcomes=["sent"]) == []  # nothing of her own
    assert [s.text for s in world.persona_said] == ["嗯嗯"]  # the answer to his message only
    assert not alerts_of(world, "engine_error")
    tick = world.assembly.proactive.scheduler
    assert tick is not None  # the scheduler is alive and keeps planning
    health = world.assembly.proactive.health()
    assert health.status.name == "OK"

    # ---- the model is back: what the day still holds is sent -------------------------------------
    world.deepseek.failures.clear()
    await world.run_until(world.local(21, 0))
    assert world.proactive_rows(outcomes=["sent"]), "she never wrote again after the model was back"
    assert all(r.at >= world.local(15, 0) for r in world.proactive_rows(outcomes=["sent"]))
    nothing_of_the_trouble_is_shown(world)
    assert_screen_matches_records(world)
    assert_never_in_deep_sleep(world)
    assert_bot_text_not_in_her_data(world)
    assert world.deepseek.unexpected == []
