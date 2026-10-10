"""Corrections in plain words: noticed, asked about, recorded on the user's "是" (R-LRN-002).

The engine runs on the manual clock with a scripted writer and the real tables; the model that
judges a remark is the real client behind ``respx``.  The persona answers the remark like any
message; only afterwards does the system ask whether to record it, and only a "是" within the
window - before she has said anything else - records it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import httpx
import pytest
import respx
from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.deepseek import API, TEST_KEY, error, ok, request_json
from tests.support.engine_extras import ScriptedCommands
from tests.support.engine_harness import (
    ClockData,
    Harness,
    build_harness,
    make_draft,
    run_to_idle,
)
from tests.support.waiting import wait_until
from twin.commands import texts
from twin.engine import machine
from twin.engine.command_port import CommandContext, CommandOutcome
from twin.engine.feedback import FeedbackStore
from twin.engine.rounds import RoundStore
from twin.engine.turns import BotTurnMessages, BotTurnStore
from twin.learning.corrections import (
    CorrectionService,
    PendingCorrection,
    is_confirmation_word,
    looks_like_correction,
)
from twin.learning.dislike import NotLikeRecorder
from twin.learning.pairs import PreferencePairStore
from twin.learning.sample import SampleBuilder
from twin.llm.runtime import DEEPSEEK_SECRET, LlmRuntime, build_llm_runtime
from twin.services import Services
from twin.storage.engine_models import BotTurn

QUESTION = texts.PREFIX + texts.CORRECTION_ASK
DONE = texts.PREFIX + texts.CORRECTION_DONE


# ---------------------------------------------------------------------------- the rules


@pytest.mark.parametrize(
    "text",
    [
        "她不会这么说",
        "她不会这么说话的",
        "她不会这样说啦",
        "她才不会这么说呢",
        "她根本不会那样讲话",
        "你说话不像她",
        "你说话一点都不像她！",
        "这不像她",
        "不像她",
        "别这么说话",
        "别这样说话",
        "不要这样说话了",
        "她不是这么说话的",
        "不是她的说话方式",
        "你的语气不对",
        "说话方式不像",
        " 她　不会这么说 ",
        "她不会这么说，她会说哈哈哈",
    ],
)
def test_a_remark_about_the_wording_is_noticed_by_the_local_rule(text: str) -> None:
    assert looks_like_correction(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "你好呀",
        "今天好累呀",
        "她不会做饭",
        "她今天不会来了",
        "别这样",
        "我想她了",
        "你说话真有意思",
        "说话算数",
        "/不像 她会说好呀",
        "她不会这么说" + "啊" * 220,  # a long message is a conversation, not a remark
    ],
)
def test_ordinary_chat_is_not_taken_for_a_correction(text: str) -> None:
    assert not looks_like_correction(text)


@pytest.mark.parametrize("text", ["是", "是的", "是。", " 是 ", "是！", "确认", "是的。"])
def test_the_yes_is_a_short_yes(text: str) -> None:
    assert is_confirmation_word(text)


@pytest.mark.parametrize("text", ["", "不是", "是不是", "好", "嗯", "对", "是的呀你说得对", "否"])
def test_anything_else_is_not_the_yes(text: str) -> None:
    assert not is_confirmation_word(text)


# ----------------------------------------------------------------------------- the flow


@dataclass
class Judge:
    """The model that judges a remark: what it answers and what it was asked."""

    answer: Callable[[], httpx.Response]
    asked: list[dict[str, Any]]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.asked.append(request_json(request))
        return self.answer()


def yes() -> httpx.Response:
    return ok(content=json.dumps({"is_correction": True}))


def no() -> httpx.Response:
    return ok(content=json.dumps({"is_correction": False}))


@pytest.fixture
def judge() -> Judge:
    return Judge(yes, [])


@pytest.fixture
def api(judge: Judge) -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        router.post(API).mock(side_effect=judge)
        yield router


@dataclass
class Rig:
    harness: Harness
    service: CorrectionService
    llm: LlmRuntime
    recorder: NotLikeRecorder

    def new_service(self) -> CorrectionService:
        """A service over the same tables (the settings are read when it is made)."""
        services, clock = self.harness.services, self.harness.clock
        return CorrectionService(
            services,
            self.llm.client,
            self.recorder,
            BotTurnStore(services.db, clock),
            RoundStore(services.db),
        )

    async def say(self, text: str, *, advance: float = 0.0) -> None:
        """The user writes ``text`` and the engine answers it completely."""
        if advance:
            self.harness.clock.tick(advance)
        await self.harness.message(text)
        await run_to_idle(self.harness.engine, self.harness.clock)

    @property
    def said(self) -> list[str]:
        return self.harness.channel.texts

    def replies(self, *lines: str) -> None:
        self.harness.writer.add(*(make_draft(*line.split("|")) for line in lines))


@pytest.fixture
async def rig(services: Services, clock: ManualClock, api: respx.MockRouter) -> AsyncIterator[Rig]:
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    services.runtime.initialize()
    llm = build_llm_runtime(services)
    commands = ScriptedCommands(**{"不像 她不会这么说": CommandOutcome(texts.PREFIX + "记下了")})
    harness = build_harness(services, clock, commands=commands)
    store = BotTurnStore(services.db, clock)
    recorder = NotLikeRecorder(
        store,
        FeedbackStore(services.db, clock),
        PreferencePairStore(services.db, clock),
        SampleBuilder(services, BotTurnMessages(services.db), ClockData(clock)),
    )
    service = CorrectionService(services, llm.client, recorder, store, RoundStore(services.db))
    harness.engine.attach_corrections(service)
    await harness.engine.start()
    yield Rig(harness, service, llm, recorder)
    await harness.engine.stop()
    await llm.client.aclose()


def stored(rig: Rig) -> list[BotTurn]:
    with rig.harness.services.db.session() as session:
        rows = list(session.scalars(select(BotTurn).order_by(BotTurn.at, BotTurn.id)))
        session.expunge_all()
    return rows


async def first_exchange(rig: Rig) -> str:
    """The user says something, she answers; returns the id of her reply."""
    rig.replies("哈哈 辛苦啦|早点休息")
    await rig.say("今天好累呀")
    reply = rig.harness.store.latest_reply()
    assert reply and reply[0].reply_id
    return reply[0].reply_id


async def test_the_remark_is_answered_like_any_message_and_then_asked_about(
    rig: Rig, judge: Judge
) -> None:
    await first_exchange(rig)
    assert judge.asked == []  # ordinary chat never reaches the model
    rig.replies("好吧 我改改")
    await rig.say("她不会这么说")
    assert rig.said == ["哈哈 辛苦啦", "早点休息", "好吧 我改改", QUESTION]
    (asked,) = judge.asked
    prompt = asked["messages"][1]["content"]
    assert "哈哈 辛苦啦\n早点休息" in prompt and "她不会这么说" in prompt
    last = stored(rig)[-1]
    assert (last.direction, last.is_command, last.backend, last.text) == (
        "out",
        True,
        "command",
        QUESTION,
    )
    assert rig.harness.writer.calls == 2  # the question did not cost another reply


async def test_the_yes_records_the_verdict_on_the_reply_the_remark_was_about(rig: Rig) -> None:
    reply_id = await first_exchange(rig)
    rig.replies("好吧 我改改")
    await rig.say("她不会这么说")
    calls = rig.harness.writer.calls
    await rig.say("是", advance=60)
    assert rig.said[-1] == DONE and rig.harness.writer.calls == calls  # she was not asked
    (record,) = rig.recorder.feedback_of(reply_id)
    assert (record.type, record.correction) == ("not_like", None)
    assert PreferencePairStore(rig.harness.services.db, rig.harness.clock).count() == 0
    rows = stored(rig)
    yes_row = next(r for r in rows if r.direction == "in" and r.text == "是")
    assert yes_row.is_command and rows[-1].is_command  # neither is conversation


async def test_neither_the_question_nor_the_yes_reach_the_prompt_or_the_memory(rig: Rig) -> None:
    await first_exchange(rig)
    rig.replies("好吧 我改改")
    await rig.say("她不会这么说")
    await rig.say("是", advance=30)
    rig.replies("嗯嗯")
    await rig.say("晚安", advance=60)
    history = [turn.text for turn in rig.harness.writer.contexts[-1].history]
    assert all(text != "是" and "不像" not in text for text in history)
    assert "她不会这么说" in "\n".join(history)  # the remark itself is a message of the chat
    await rig.harness.clock.advance(31 * 60)
    await wait_until(lambda: len(rig.harness.queued) >= 1)
    sent = [m.text for batch in rig.harness.queued for m in batch]
    assert "是" not in sent and QUESTION not in sent and DONE not in sent


async def test_a_yes_after_the_window_is_a_plain_message(rig: Rig) -> None:
    reply_id = await first_exchange(rig)
    rig.replies("好吧 我改改")
    await rig.say("她不会这么说")
    rig.replies("是什么呀")
    await rig.say("是", advance=61 * 60)
    assert rig.said[-1] == "是什么呀"
    assert rig.recorder.feedback_of(reply_id) == []
    assert not next(r for r in stored(rig) if r.text == "是").is_command


async def test_the_window_is_a_setting(rig: Rig) -> None:
    rig.harness.services.settings.commands.confirm_window_min = 5
    rig.harness.engine.attach_corrections(rig.new_service())
    reply_id = await first_exchange(rig)
    rig.replies("好吧 我改改")
    await rig.say("她不会这么说")
    rig.replies("哦")
    await rig.say("是", advance=6 * 60)
    assert rig.recorder.feedback_of(reply_id) == []


async def test_a_yes_after_she_has_said_something_else_answers_that_and_not_the_question(
    rig: Rig,
) -> None:
    reply_id = await first_exchange(rig)
    rig.replies("好吧 我改改")
    await rig.say("她不会这么说")
    rig.replies("你想听什么呀")
    await rig.say("你再说一句", advance=30)  # she answers: the question is no longer the last word
    rig.replies("好的好的")
    await rig.say("是", advance=30)
    assert rig.said[-1] == "好的好的" and rig.recorder.feedback_of(reply_id) == []


async def test_nothing_is_recorded_without_the_yes(rig: Rig) -> None:
    reply_id = await first_exchange(rig)
    rig.replies("好吧 我改改")
    await rig.say("她不会这么说")
    assert rig.recorder.feedback_of(reply_id) == [] and rig.service.pending() is not None


async def test_a_yes_with_no_question_open_is_an_ordinary_yes(rig: Rig) -> None:
    await first_exchange(rig)
    rig.replies("真的吗")
    await rig.say("是", advance=30)
    assert rig.said[-1] == "真的吗" and rig.service.pending() is None


async def test_the_question_is_asked_once_and_the_yes_closes_it(rig: Rig) -> None:
    reply_id = await first_exchange(rig)
    rig.replies("好吧 我改改")
    await rig.say("她不会这么说")
    await rig.say("是", advance=30)
    assert rig.service.pending() is None
    rig.replies("嗯")
    await rig.say("是", advance=30)  # a second yes confirms nothing
    assert len(rig.recorder.feedback_of(reply_id)) == 1


async def test_the_open_question_survives_a_restart(rig: Rig) -> None:
    reply_id = await first_exchange(rig)
    rig.replies("好吧 我改改")
    await rig.say("她不会这么说")
    again = rig.new_service()
    pending = again.pending()
    assert isinstance(pending, PendingCorrection) and pending.reply_id == reply_id
    assert await again.is_confirmation("是", rig.harness.clock.now_utc() + timedelta(minutes=1))


async def test_a_newer_question_replaces_the_older_one(rig: Rig) -> None:
    await first_exchange(rig)
    rig.replies("好吧 我改改")
    await rig.say("她不会这么说")
    older = rig.service.pending()
    rig.replies("这样呢")
    await rig.say("你说话不像她", advance=30)
    newer = rig.service.pending()
    assert older is not None and newer is not None and newer.reply_id != older.reply_id


# --------------------------------------------------------------------- when nothing is asked


async def test_a_remark_the_model_does_not_confirm_is_not_asked_about(
    rig: Rig, judge: Judge
) -> None:
    judge.answer = no
    await first_exchange(rig)
    rig.replies("嗯嗯")
    await rig.say("她不会这么说")
    assert rig.said[-1] == "嗯嗯" and len(judge.asked) == 1 and rig.service.pending() is None


@pytest.mark.parametrize("failure", ["down", "garbage"])
async def test_a_model_that_cannot_answer_costs_the_question_and_nothing_else(
    rig: Rig, judge: Judge, failure: str
) -> None:
    judge.answer = (
        (lambda: error(400, "boom")) if failure == "down" else (lambda: ok(content="not json"))
    )
    await first_exchange(rig)
    rig.replies("嗯嗯")
    await rig.say("她不会这么说")
    assert rig.said[-1] == "嗯嗯" and rig.service.pending() is None


async def test_without_a_key_nothing_is_asked(rig: Rig, judge: Judge) -> None:
    rig.harness.services.secrets.delete(DEEPSEEK_SECRET)
    await first_exchange(rig)
    rig.replies("嗯嗯")
    await rig.say("她不会这么说")
    assert rig.said[-1] == "嗯嗯" and judge.asked == []


async def test_a_remark_before_any_reply_has_nothing_to_be_about(rig: Rig, judge: Judge) -> None:
    rig.replies("你好呀")
    await rig.say("她不会这么说")
    assert rig.said == ["你好呀"] and judge.asked == []


async def test_a_reply_already_marked_is_not_asked_about_again(rig: Rig, judge: Judge) -> None:
    reply_id = await first_exchange(rig)
    rig.recorder.record(None, reply_id=reply_id)
    rig.replies("嗯嗯")
    await rig.say("她不会这么说")
    assert rig.said[-1] == "嗯嗯" and judge.asked == []


async def test_the_reply_out_of_the_role_is_not_asked_about(rig: Rig, judge: Judge) -> None:
    rig.harness.writer.add(make_draft("我想先停一下|你现在还好吗", backend="safety"))
    await rig.say("今天好累呀")
    rig.replies("嗯嗯")
    await rig.say("她不会这么说")
    assert judge.asked == [] and rig.service.pending() is None


async def test_the_detection_can_be_switched_off(rig: Rig, judge: Judge) -> None:
    rig.harness.services.settings.learning.detect_corrections = False
    await first_exchange(rig)
    rig.replies("嗯嗯")
    await rig.say("她不会这么说")
    assert rig.said[-1] == "嗯嗯" and judge.asked == []


async def test_a_command_is_never_a_remark(rig: Rig, judge: Judge) -> None:
    await first_exchange(rig)
    await rig.harness.message("/不像 她不会这么说")  # the router takes it: no round, no question
    assert rig.said[-1] == texts.PREFIX + "记下了" and judge.asked == []


# ------------------------------------------------------------ the engine and a failing port


class BrokenPort:
    """A correction service that fails in each of its three places."""

    def __init__(self) -> None:
        self.confirmed = True
        self.calls: list[str] = []

    async def is_confirmation(self, text: str, at: datetime) -> bool:
        self.calls.append("is_confirmation")
        return self.confirmed and text == "是"

    async def confirm(self, context: CommandContext) -> CommandOutcome | None:
        self.calls.append("confirm")
        raise RuntimeError("the service is down")

    async def propose(self, answered: Sequence[str]) -> str | None:
        self.calls.append("propose")
        raise RuntimeError("the service is down")


async def test_a_failing_service_never_costs_the_user_a_reply(
    services: Services, clock: ManualClock
) -> None:
    harness = build_harness(services, clock)
    port = BrokenPort()
    harness.engine.attach_corrections(port)
    assert harness.engine.corrections is port
    await harness.engine.start()
    try:
        harness.writer.add(make_draft("嗯嗯"), make_draft("什么呀"))
        await harness.message("你好")
        await run_to_idle(harness.engine, harness.clock)
        await harness.message("是")  # taken for a yes, but the service fails: it is chat
        await run_to_idle(harness.engine, harness.clock)
        assert harness.channel.texts == ["嗯嗯", "什么呀"]
        assert "propose" in port.calls and "confirm" in port.calls
        yes_row = next(r for r in _rows(harness) if r.text == "是")
        assert not yes_row.is_command  # the message was queued as chat after all
    finally:
        await harness.engine.stop()


def _rows(harness: Harness) -> list[BotTurn]:
    with harness.services.db.session() as session:
        found = list(session.scalars(select(BotTurn).order_by(BotTurn.at, BotTurn.id)))
        session.expunge_all()
    return found


class StuckPort(BrokenPort):
    """A correction service whose question never comes."""

    async def propose(self, answered: Sequence[str]) -> str | None:
        self.calls.append("propose")
        await asyncio.Event().wait()
        return None


async def test_a_question_that_takes_too_long_never_holds_the_conversation_up(
    services: Services, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(machine, "CORRECTION_TIMEOUT_S", 0.05)
    harness = build_harness(services, clock)
    harness.engine.attach_corrections(StuckPort())
    await harness.engine.start()
    try:
        harness.writer.add(make_draft("嗯嗯"))
        await harness.message("你好")
        await run_to_idle(harness.engine, harness.clock)
        assert harness.channel.texts == ["嗯嗯"] and harness.engine.snapshot().state == "IDLE"
    finally:
        await harness.engine.stop()


async def test_the_engine_can_say_a_system_message_on_its_own(
    services: Services, clock: ManualClock
) -> None:
    harness = build_harness(services, clock)
    await harness.engine.start()
    try:
        await harness.engine.notify("导入完成")
        await harness.engine.notify(texts.PREFIX + "已经有前缀")
        assert harness.channel.texts == [texts.PREFIX + "导入完成", texts.PREFIX + "已经有前缀"]
        assert [(r.is_command, r.backend) for r in _rows(harness)] == [
            (True, "command"),
            (True, "command"),
        ]
    finally:
        await harness.engine.stop()


async def test_a_yes_for_a_reply_that_is_gone_is_told_so_and_closes_the_question(rig: Rig) -> None:
    await first_exchange(rig)
    now = rig.harness.clock.now_utc()
    rig.service._store(PendingCorrection("no-such-reply", now))
    outcome = await rig.service.confirm(CommandContext(now + timedelta(seconds=30), "in-1"))
    assert outcome == CommandOutcome(texts.PREFIX + texts.NOT_LIKE_NOTHING)
    assert rig.service.pending() is None  # the question is closed whether or not it recorded
    assert await rig.service.confirm(CommandContext(now + timedelta(seconds=40), "in-2")) is None
