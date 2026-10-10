"""A reply that fails is never an error text; a crisis leaves the role (R-ENG-010, R-SAFE-001)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.engine_harness import (
    START,
    FixedDay,
    Harness,
    ScriptedCrisis,
    ScriptedWriter,
    build_harness,
    make_draft,
    run_to_idle,
)
from tests.support.waiting import wait_until
from twin.engine.fallback import ShortAnswers
from twin.engine.roundstate import RoundData
from twin.services import Services
from twin.storage.engine_models import BotTurn

SHORT = ("嗯嗯", "好的", "哈哈")
FORBIDDEN = (
    "Traceback",
    "Error",
    "error",
    "Exception",
    "exception",
    "sk-",
    "boom",
    "system prompt",
    "提示词",
    "拒绝",
    "抱歉",
    "无法",
    "timeout",
    "Timeout",
)


@pytest.fixture
async def rig(services: Services, clock: ManualClock) -> AsyncIterator[Harness]:
    harness = build_harness(services, clock, answers=SHORT)
    await harness.engine.start()
    yield harness
    await harness.engine.stop()


def stored_out(harness: Harness) -> list[BotTurn]:
    with harness.services.db.session() as session:
        found = list(
            session.scalars(
                select(BotTurn).where(BotTurn.direction == "out").order_by(BotTurn.at, BotTurn.id)
            )
        )
        session.expunge_all()
    return found


def every_outgoing_text(harness: Harness) -> list[str]:
    """What went through the channel and what was stored: nothing else may carry a word."""
    return [*harness.channel.texts, *(row.text for row in stored_out(harness))]


def failing(reason: str = "backend_error") -> Any:
    return make_draft(fallback=reason)  # type: ignore[arg-type]


# ------------------------------------------------------------------------- retries


async def test_a_failed_reply_is_tried_again_two_to_ten_minutes_later(rig: Harness) -> None:
    rig.writer.add(failing(), make_draft("好呀"))
    await rig.message("在吗")
    await run_to_idle(rig.engine, rig.clock, until=lambda: rig.writer.calls == 1)
    await wait_until(lambda: RoundData.of(rig.engine.snapshot()).retries == 1)
    failed_at = rig.clock.now_utc()
    snap = rig.engine.snapshot()
    decision = RoundData.of(snap).decision
    assert snap.state == "DECIDING" and decision is not None and decision.mode == "retry"
    assert timedelta(minutes=2) <= decision.send_at - failed_at <= timedelta(minutes=10)
    assert rig.channel.texts == []
    await run_to_idle(rig.engine, rig.clock)
    assert rig.channel.texts == ["好呀"] and rig.writer.calls == 2
    first = next(row for row in stored_out(rig) if row.bubble_index == 0)
    assert {"step": "retried_later", "count": 1} in first.actions


async def test_after_three_failed_retries_a_short_answer_of_hers_goes_out_with_an_alert(
    rig: Harness,
) -> None:
    rig.writer.add(*(failing() for _ in range(4)))
    await rig.message("在吗")
    await run_to_idle(rig.engine, rig.clock)
    assert rig.writer.calls == 4  # the first try and three retries
    assert len(rig.channel.texts) == 1 and rig.channel.texts[0] in SHORT
    assert rig.alerts.categories == ["reply_failed"]
    detail = rig.alerts.raised[0][2]
    assert detail is not None and detail["reason"] == "backend_error" and detail["answered"]
    row = next(r for r in stored_out(rig) if r.kind == "text")
    assert row.backend == "fallback"
    assert {"step": "fallback_answer", "detail": "backend_error"} in row.actions


@pytest.mark.parametrize(
    "answer",
    [
        RuntimeError("boom: Traceback (most recent call last) sk-secret-key"),
        TimeoutError("timeout after 60s"),
        failing("refused"),
        failing("violations"),
        failing("empty"),
        make_draft(),  # a reply without a single bubble
        asyncio.CancelledError(),  # the pipeline task cancelled from inside
    ],
    ids=["exception", "timeout", "refused", "violations", "empty", "no-bubbles", "cancelled"],
)
async def test_whatever_goes_wrong_no_error_text_ever_reaches_the_user(
    rig: Harness, answer: Any
) -> None:
    rig.writer.add(*(answer for _ in range(4)))
    await rig.message("在吗")
    await run_to_idle(rig.engine, rig.clock)
    texts = every_outgoing_text(rig)
    assert texts and set(texts) <= set(SHORT) | {"在吗"}
    for text in texts:
        assert not any(word in text for word in FORBIDDEN)
    assert rig.alerts.categories == ["reply_failed"]


async def test_a_message_that_comes_while_she_waits_to_retry_joins_the_round(rig: Harness) -> None:
    rig.writer.add(failing(), make_draft("好呀"))
    await rig.message("在吗")
    await run_to_idle(rig.engine, rig.clock, until=lambda: rig.writer.calls == 1)
    await wait_until(lambda: RoundData.of(rig.engine.snapshot()).retries == 1)
    await rig.message("还在吗")
    await run_to_idle(rig.engine, rig.clock)
    assert rig.writer.contexts[1].user_text == "在吗\n还在吗"
    assert rig.channel.texts == ["好呀"]


async def test_without_any_short_answer_of_hers_nothing_is_sent_but_the_alert_says_so(
    services: Services, clock: ManualClock
) -> None:
    harness = build_harness(services, clock, short_answers=ShortAnswers(()))
    await harness.engine.start()
    try:
        harness.writer.add(*(failing() for _ in range(4)))
        await harness.message("在吗")
        await run_to_idle(harness.engine, clock)
        assert harness.channel.texts == [] and harness.engine.snapshot().state == "IDLE"
        category, title, detail = harness.alerts.raised[0]
        assert category == "reply_failed" and "no short answer" in title
        assert detail is not None and detail["answered"] is False
    finally:
        await harness.engine.stop()


def test_short_answers_are_the_words_she_says_most_often() -> None:
    phrases = {  # the payload of a stored profile version: the sentences of each party
        "her": {
            "sentences": [
                ["嗯嗯", 90],
                ["好的", 60],
                ["今天晚上我们去吃火锅好不好", 40],  # too long
                ["[图片]", 30],  # a media placeholder
                ["ok", 25],  # not words
                ["在吗?", 20],  # punctuation
                ["哈哈哈", 10],
                ["坏", "x"],  # broken entry
            ]
        },
        "user": {"sentences": [["晚安啦", 500]]},  # what he says is never an answer of hers
    }
    answers = ShortAnswers.from_phrases(phrases)
    assert list(answers.texts()) == ["嗯嗯", "好的", "哈哈哈"] and bool(answers)
    assert not ShortAnswers.from_phrases(None) and ShortAnswers.from_phrases({}).texts() == []
    assert ShortAnswers.from_phrases({"her": None, "user": {}}).texts() == []
    import random

    counts: dict[str, int] = {}
    rng = random.Random(1)
    for _ in range(600):
        picked = answers.pick(rng)
        assert picked is not None
        counts[picked] = counts.get(picked, 0) + 1
    assert counts["嗯嗯"] > counts["好的"] > counts["哈哈哈"]  # in proportion to her use
    assert ShortAnswers(()).pick(rng) is None


# ------------------------------------------------------------------------ silence


async def test_an_empty_draft_is_a_failure_not_silence(rig: Harness) -> None:
    rig.writer.add(make_draft(), make_draft("好呀"))
    await rig.message("在吗")
    await run_to_idle(rig.engine, rig.clock)
    assert rig.channel.texts == ["好呀"] and rig.writer.calls == 2


# --------------------------------------------------------------------------- crisis


async def test_a_crisis_is_answered_out_of_the_role_at_once_even_in_the_middle_of_the_night(
    services: Services, clock: ManualClock
) -> None:
    clock.set_time(START + timedelta(hours=17))  # midnight in Chicago
    night = FixedDay(
        (START - timedelta(days=1), "free"),
        (START + timedelta(hours=15), "deep_sleep"),
        (START + timedelta(hours=24), "free"),
    )
    harness = build_harness(services, clock, day=night, crisis=ScriptedCrisis("不想活"))
    await harness.engine.start()
    try:
        began = clock.now_utc()
        await harness.message("我真的不想活了")
        await run_to_idle(harness.engine, clock)
        assert harness.channel.texts == ["我想先停一下。", "你现在还好吗？", "也可以联系 988。"]
        assert harness.writer.calls == 0  # no persona, no model
        assert all(o.at - began <= timedelta(seconds=15) for o in harness.channel.out)
        rows = [row for row in stored_out(harness) if row.kind == "text"]
        assert {row.backend for row in rows if row.backend} == {"safety"}
        assert harness.engine.snapshot().state == "IDLE" and harness.engine.snapshot().pending == ()
    finally:
        await harness.engine.stop()


async def test_a_keyword_the_judgement_dismisses_is_an_ordinary_message(
    services: Services, clock: ManualClock
) -> None:
    harness = build_harness(services, clock, crisis=ScriptedCrisis("笑死", confirm=False))
    await harness.engine.start()
    try:
        harness.writer.add(make_draft("哈哈"))
        await harness.message("笑死我了")
        await run_to_idle(harness.engine, clock)
        assert harness.crisis.handled == [["笑死我了"]]
        assert harness.channel.texts == ["哈哈"] and harness.writer.calls == 1
    finally:
        await harness.engine.stop()


async def test_the_messages_of_a_round_are_screened_together_and_once(
    services: Services, clock: ManualClock
) -> None:
    crisis = ScriptedCrisis("不想活", confirm=False)
    harness = build_harness(services, clock, crisis=crisis)
    await harness.engine.start()
    try:
        await harness.message("在吗")
        await harness.message("不想活了吗，这个游戏")
        await run_to_idle(harness.engine, clock)
        assert crisis.handled == [["在吗", "不想活了吗，这个游戏"]]  # one call for the round
    finally:
        await harness.engine.stop()


async def test_a_crisis_in_a_later_message_is_answered_at_once_and_ends_the_round(
    services: Services, clock: ManualClock
) -> None:
    harness = build_harness(services, clock, crisis=ScriptedCrisis("想死"))
    await harness.engine.start()
    try:
        await harness.message("在吗")
        await run_to_idle(
            harness.engine, clock, until=lambda: harness.engine.snapshot().state == "DECIDING"
        )
        await harness.message("我有时候想死")  # while she is still "reading" the first one
        await run_to_idle(harness.engine, clock)
        assert harness.channel.texts[0] == "我想先停一下。" and harness.writer.calls == 0
        assert len(harness.channel.texts) == 3
        assert harness.engine.snapshot().pending == ()
    finally:
        await harness.engine.stop()


async def test_a_crisis_screen_only_reads_what_the_user_wrote_or_said(
    services: Services, clock: ManualClock
) -> None:
    from twin.channel.base import MessageKind

    crisis = ScriptedCrisis("想死")
    harness = build_harness(services, clock, crisis=crisis)
    await harness.engine.start()
    try:
        await harness.message("[图片：一张写着想死的涂鸦]", kind=MessageKind.IMAGE)
        await run_to_idle(harness.engine, clock)
        assert crisis.handled == [] and harness.writer.calls == 1  # a picture is not his words
    finally:
        await harness.engine.stop()


async def test_a_thrown_away_reply_is_not_repeated_by_the_pipeline_when_the_model_refuses(
    rig: Harness,
) -> None:
    rig.writer.add(failing("refused"), make_draft("好呀"))
    await rig.message("在吗")
    await run_to_idle(rig.engine, rig.clock)
    assert rig.channel.texts == ["好呀"]
    assert rig.alerts.categories == []  # a refusal that a later try overcame needs no alert


async def test_the_backend_failure_log_never_has_the_text_of_the_conversation(
    rig: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    caplog.set_level(logging.DEBUG)
    rig.writer.add(RuntimeError("it broke"), make_draft("秘密的回复"))
    await rig.message("秘密的消息")
    await run_to_idle(rig.engine, rig.clock)
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "秘密的消息" not in logged and "秘密的回复" not in logged
    assert ScriptedWriter is not None
