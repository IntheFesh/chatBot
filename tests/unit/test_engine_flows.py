"""Her day shapes the reply: asleep, busy, paused, and the memory after it (R-ENG-003/004)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.engine_extras import Routine, Window
from tests.support.engine_harness import (
    START,
    FixedDay,
    Harness,
    build_harness,
    make_draft,
    run_to_idle,
)
from tests.support.synthetic import us_phone
from tests.support.waiting import wait_until
from twin.config.runtime import ENGINE_PAUSED_UNTIL, SHOW_THINKING
from twin.engine.pacing import PacingModel
from twin.engine.roundstate import RoundData
from twin.engine.turns import OutboundBubble, ReplyMeta
from twin.profile.distribution import EmpiricalDistribution
from twin.schedule.plan_model import BusySpan, busy_ref
from twin.services import Services
from twin.storage.engine_models import BotTurn

LOCAL_MIDNIGHT = START + timedelta(hours=17)  # 00:00 in Chicago
WAKE = START + timedelta(hours=24)  # 07:00 in Chicago the next morning


def night_day() -> FixedDay:
    return FixedDay(
        (START - timedelta(days=1), "free"),
        (START + timedelta(hours=14, minutes=30), "sleep_edge"),
        (START + timedelta(hours=15), "deep_sleep"),
        (START + timedelta(hours=23, minutes=30), "sleep_edge"),
        (WAKE, "free"),
    )


@pytest.fixture
async def sleeping(services: Services, clock: ManualClock) -> AsyncIterator[Harness]:
    clock.set_time(LOCAL_MIDNIGHT)
    harness = build_harness(services, clock, day=night_day())
    await harness.engine.start()
    yield harness
    await harness.engine.stop()


def out_rows(harness: Harness) -> list[BotTurn]:
    with harness.services.db.session() as session:
        found = list(
            session.scalars(
                select(BotTurn).where(BotTurn.direction == "out").order_by(BotTurn.at, BotTurn.id)
            )
        )
        session.expunge_all()
    return found


# ---------------------------------------------------------------------------- asleep


async def test_a_message_in_the_night_is_answered_after_she_wakes(sleeping: Harness) -> None:
    sleeping.writer.add(make_draft("早呀", "刚醒"))
    await sleeping.message("睡了吗")
    await run_to_idle(
        sleeping.engine,
        sleeping.clock,
        until=lambda: sleeping.engine.snapshot().state == "DECIDING",
    )
    decision = RoundData.of(sleeping.engine.snapshot()).decision
    assert decision is not None and decision.mode == "asleep" and decision.woke_up
    assert decision.wake_at == WAKE
    assert WAKE + timedelta(minutes=5) <= decision.send_at <= WAKE + timedelta(minutes=40)
    assert sleeping.channel.texts == []
    await run_to_idle(sleeping.engine, sleeping.clock)
    first = next(o for o in sleeping.channel.out if o.kind == "text")
    assert first.at >= decision.send_at  # not a second before she is up
    assert sleeping.writer.contexts[0].woke_up  # the prompt says she has just woken up
    assert sleeping.channel.texts == ["早呀", "刚醒"]


async def test_more_messages_in_the_night_join_the_same_morning_round(sleeping: Harness) -> None:
    await sleeping.message("睡了吗")
    await run_to_idle(
        sleeping.engine,
        sleeping.clock,
        until=lambda: sleeping.engine.snapshot().state == "DECIDING",
    )
    planned = sleeping.engine.snapshot().planned_send_at
    await sleeping.clock.advance(3 * 3600)  # three hours later: far from 80 % of the wait
    await sleeping.message("晚安哦")
    await wait_until(lambda: RoundData.of(sleeping.engine.snapshot()).decided_for == 2)
    assert sleeping.engine.snapshot().planned_send_at == planned
    sleeping.writer.add(make_draft("早"))
    await run_to_idle(sleeping.engine, sleeping.clock)
    assert sleeping.writer.contexts[0].user_text == "睡了吗\n晚安哦"
    assert sleeping.writer.calls == 1


async def test_a_message_just_before_she_wakes_adds_a_short_pause(sleeping: Harness) -> None:
    await sleeping.message("睡了吗")
    await run_to_idle(
        sleeping.engine,
        sleeping.clock,
        until=lambda: sleeping.engine.snapshot().state == "DECIDING",
    )
    planned = sleeping.engine.snapshot().planned_send_at
    assert planned is not None
    await sleeping.clock.advance((planned - sleeping.clock.now_utc()).total_seconds() - 2.0)
    await sleeping.message("在不在")  # two seconds before she would start: past 80 %
    await wait_until(lambda: RoundData.of(sleeping.engine.snapshot()).decided_for == 2)
    extended = sleeping.engine.snapshot().planned_send_at
    assert extended is not None and extended > planned
    assert extended == sleeping.clock.now_utc() + timedelta(seconds=0.5 + 4.0)


# ------------------------------------------------------------------------------ busy


async def test_when_she_is_busy_the_delay_is_drawn_from_the_busy_window(
    services: Services, clock: ManualClock
) -> None:
    window = Window(EmpiricalDistribution.from_counter({900.0: 1}))
    routine = Routine([window])
    span = BusySpan(
        START - timedelta(hours=1),
        START + timedelta(hours=4),
        "07:00–11:00",
        "inferred",
        busy_ref("workday", None, 0),
    )
    day = FixedDay(
        (START - timedelta(days=1), "free"), (START - timedelta(hours=1), "busy"), busy=span
    )
    pacing = PacingModel(
        latency_all=EmpiricalDistribution.from_samples([5.0] * 5),
        burst_gap=EmpiricalDistribution.from_samples([4.0] * 5),
        activity=routine,  # type: ignore[arg-type]
    )
    harness = build_harness(services, clock, day=day, pacing=pacing)
    await harness.engine.start()
    try:
        await harness.message("在忙吗")
        await run_to_idle(
            harness.engine, clock, until=lambda: harness.engine.snapshot().state == "DECIDING"
        )
        decision = RoundData.of(harness.engine.snapshot()).decision
        assert decision is not None and decision.mode == "busy"
        assert decision.delay_s == pytest.approx(900.0)  # her busy latency, minutes not seconds
        await run_to_idle(harness.engine, clock)
        assert harness.channel.texts == ["好的"]
    finally:
        await harness.engine.stop()


# ---------------------------------------------------------------------------- paused


@pytest.fixture
async def rig(services: Services, clock: ManualClock) -> AsyncIterator[Harness]:
    harness = build_harness(services, clock)
    await harness.engine.start()
    yield harness
    await harness.engine.stop()


async def test_during_a_pause_there_is_no_reply_and_afterwards_she_has_just_seen_it(
    rig: Harness,
) -> None:
    until = rig.clock.now_utc() + timedelta(hours=1)
    rig.services.runtime.set(ENGINE_PAUSED_UNTIL, until)
    await rig.message("在吗")
    await run_to_idle(
        rig.engine, rig.clock, until=lambda: rig.engine.snapshot().state == "DECIDING"
    )
    decision = RoundData.of(rig.engine.snapshot()).decision
    assert decision is not None and decision.mode == "paused" and decision.paused_until == until
    await rig.clock.advance(1800)
    assert rig.channel.texts == [] and rig.writer.calls == 0
    await run_to_idle(rig.engine, rig.clock)
    first = next(o for o in rig.channel.out if o.kind == "text")
    # the pause ends, then reading (0.5 s) and her pause (4 s) and typing: not her usual 20 s
    assert first.at - until < timedelta(seconds=10)
    assert rig.channel.texts == ["好的"]


async def test_lifting_the_pause_early_makes_her_answer_soon(rig: Harness) -> None:
    rig.services.runtime.set(ENGINE_PAUSED_UNTIL, rig.clock.now_utc() + timedelta(hours=5))
    await rig.message("在吗")
    await run_to_idle(
        rig.engine, rig.clock, until=lambda: rig.engine.snapshot().state == "DECIDING"
    )
    await rig.clock.advance(600)
    resumed = rig.clock.now_utc()
    rig.services.runtime.set(ENGINE_PAUSED_UNTIL, None)  # /恢复
    await rig.engine.on_state_change(1, 2)
    await run_to_idle(rig.engine, rig.clock)
    first = next(o for o in rig.channel.out if o.kind == "text")
    assert first.at - resumed < timedelta(seconds=10)  # just seen, not her usual twenty seconds
    assert rig.channel.texts == ["好的"]


def decision_mode(harness: Harness) -> str | None:
    decision = RoundData.of(harness.engine.snapshot()).decision
    return decision.mode if decision is not None else None


async def test_a_pause_set_while_she_waits_holds_the_round_back(rig: Harness) -> None:
    await rig.message("在吗")
    await run_to_idle(
        rig.engine, rig.clock, until=lambda: rig.engine.snapshot().state == "DECIDING"
    )
    until = rig.clock.now_utc() + timedelta(hours=2)
    rig.services.runtime.set(ENGINE_PAUSED_UNTIL, until)
    await rig.engine.on_state_change(1, 2)
    await wait_until(lambda: decision_mode(rig) == "paused")
    await rig.clock.advance(3600)
    assert rig.writer.calls == 0 and rig.channel.texts == []
    await run_to_idle(rig.engine, rig.clock)
    assert rig.channel.texts == ["好的"]


async def test_a_pause_set_just_as_the_time_comes_is_respected(rig: Harness) -> None:
    await rig.message("在吗")
    await run_to_idle(
        rig.engine, rig.clock, until=lambda: rig.engine.snapshot().state == "DECIDING"
    )
    planned = rig.engine.snapshot().planned_send_at
    assert planned is not None
    until = planned + timedelta(hours=1)
    rig.services.runtime.set(ENGINE_PAUSED_UNTIL, until)  # no state watcher call this time
    await rig.clock.advance((planned - rig.clock.now_utc()).total_seconds())
    await wait_until(lambda: decision_mode(rig) == "paused")
    assert rig.writer.calls == 0 and rig.engine.snapshot().state == "DECIDING"


# ------------------------------------------------------------------------- the user's end


async def test_thinking_is_shown_as_a_system_message_when_the_user_asked_for_it(
    rig: Harness,
) -> None:
    rig.services.runtime.set(SHOW_THINKING, True)
    number = us_phone()
    reasoning = f"她在想：对方的电话是 {number}，" + "想" * 600
    rig.writer.add(make_draft("好呀", reasoning=reasoning))
    await rig.message("在吗")
    await run_to_idle(rig.engine, rig.clock)
    assert rig.channel.texts[0] == "好呀"
    shown = rig.channel.texts[1]
    assert shown.startswith("⚙️ 思考：")
    assert number not in shown and "[手机号" in shown  # redacted
    assert len(shown) <= len("⚙️ 思考：") + 500  # cut to 500 characters
    last = out_rows(rig)[-1]
    assert last.is_command and last.backend == "command"  # a system message, not conversation


async def test_thinking_stays_hidden_by_default(rig: Harness) -> None:
    rig.writer.add(make_draft("好呀", reasoning="她在想"))
    await rig.message("在吗")
    await run_to_idle(rig.engine, rig.clock)
    assert rig.channel.texts == ["好呀"]


# ------------------------------------------------------------------------------ memory


async def test_a_conversation_is_handed_to_the_extractor_after_the_quiet_minutes(
    rig: Harness,
) -> None:
    rig.writer.add(make_draft("好呀"))
    await rig.message("我周五有面试")
    await run_to_idle(rig.engine, rig.clock)
    await wait_until(lambda: rig.engine.snapshot().last_outbound_at is not None)
    await rig.clock.advance(29 * 60)
    assert rig.queued == []  # she spoke 29 minutes ago: the conversation is not over yet
    await rig.clock.advance(2 * 60)
    await wait_until(lambda: len(rig.queued) == 1)
    assert [(m.role, m.text) for m in rig.queued[0]] == [("user", "我周五有面试"), ("bot", "好呀")]
    assert rig.engine.snapshot().extracted_through is not None
    assert await rig.engine.extract_if_quiet() is False  # handed over once


async def test_the_extractor_gets_only_what_is_new_and_never_a_command(rig: Harness) -> None:
    rig.writer.add(make_draft("好呀"), make_draft("嗯"))
    await rig.message("第一天的事")
    await run_to_idle(rig.engine, rig.clock)
    await rig.clock.advance(31 * 60)
    await wait_until(lambda: len(rig.queued) == 1)
    rig.store.add_bubble(
        OutboundBubble("⚙️ 状态正常", rig.clock.now_utc()),
        meta=ReplyMeta("command"),
        is_command=True,
    )
    await rig.message("第二天的事")
    await run_to_idle(rig.engine, rig.clock)
    await rig.clock.advance(31 * 60)
    await wait_until(lambda: len(rig.queued) == 2)
    assert [m.text for m in rig.queued[1]] == ["第二天的事", "嗯"]


async def test_nothing_is_queued_for_a_conversation_that_has_nothing_new(
    services: Services, clock: ManualClock
) -> None:
    harness = build_harness(services, clock)  # the engine is not started: no loop of its own
    assert await harness.engine.extract_if_quiet() is False  # no conversation at all
    harness.store.add_inbound(at=clock.now_utc(), kind="text", text="我周五有面试")
    harness.state.update(last_inbound_at=clock.now_utc())
    assert await harness.engine.extract_if_quiet() is False  # not quiet yet
    clock.tick(30 * 60)
    assert await harness.engine.extract_if_quiet() is True
    assert [m.text for m in harness.queued[0]] == ["我周五有面试"]
    assert await harness.engine.extract_if_quiet() is False
