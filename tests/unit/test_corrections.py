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
from hypothesis import given
from hypothesis import strategies as st
from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.deepseek import API, TEST_KEY, error, ok, request_json
from tests.support.embedding import HashingBackend
from tests.support.engine_extras import ScriptedCommands
from tests.support.engine_harness import (
    ClockData,
    Harness,
    build_harness,
    make_draft,
    run_to_idle,
)
from tests.support.memory import make_memory
from tests.support.waiting import wait_until
from twin.channel.base import OutboundKind, OutboundResult
from twin.commands import texts
from twin.engine import machine
from twin.engine.command_port import CommandContext, CommandOutcome
from twin.engine.feedback import FeedbackStore
from twin.engine.rounds import RoundStore
from twin.engine.turns import BotTurnMessages, BotTurnStore
from twin.learning.corrections import (
    KIND_FACT,
    KIND_NOT_LIKE,
    CorrectionService,
    FactProposer,
    PendingCorrection,
    PendingStore,
    is_confirmation_word,
    is_rejection_word,
    looks_like_correction,
)
from twin.learning.dislike import NotLikeRecorder
from twin.learning.pairs import PreferencePairStore
from twin.learning.sample import SampleBuilder
from twin.llm.runtime import DEEPSEEK_SECRET, LlmRuntime, build_llm_runtime
from twin.memory.manage import MemoryManager
from twin.memory.memory import Memory
from twin.services import Services
from twin.storage.engine_models import BotTurn
from twin.storage.settings_store import get_setting, put_setting

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


# R17-23 (D-673): the local rule is a recall-oriented pre-filter, DeepSeek's correction_check is
# the judge, so it must let the usual ways of saying "this is not how she talks" through
RECALLED_REMARKS = [
    "她不会说这种话",
    "她从来不说这种话",
    "她不会用这个词",
    "她不会用这种语气",
    "她说话没这么正式",
    "她说话才不这样",
    "她的语气不是这样的",
    "她没这么客气",
    "她不是这种语气",
    "不是她会说的",
    "这不是她说的",
    "不像真人",
    "这不像人话",
    "听起来好假",
    "听起来不像她",
    "听起来像机器人",
    "你像客服一样",
    "ai味太重了",
    "书面腔",
    "太正式了",
    "太书面了吧",
    "这么客气干嘛",
    "好生硬啊",
    "有点啰嗦",
    "太机械了",
    "好假",
    "语气不对",
    "语气太冲了",
    "措辞不太对",
    "用词不对",
    "说法有点怪",
    "别用这种语气",
    "别用这个词",
    "不要说这种话",
    "别这么客气",
    "换个语气",
    "换种说法吧",
    "她平时都回哈哈",
    "她一般会说哈哈哈",
    "她应该说好呀",
]


@pytest.mark.parametrize("text", RECALLED_REMARKS)
def test_the_usual_ways_of_saying_she_does_not_talk_like_that_are_noticed(text: str) -> None:
    """R17-23: fails before the fix (the rule only knew "不会这么/这样/那么/那样/用这种 + 说")."""
    assert looks_like_correction(text), text


@pytest.mark.parametrize(
    "text",
    [
        "她不会来了",
        "哈哈哈",
        "她说话声音好听",
        "她不会游泳",
        "她一般不吃辣",
        "她平时都在学校",
        "太好吃了",
        "今天太累了",
        "这个太贵了",
        "语气词太多了",
        "我们说话算数",
        "我不会用这个软件",
        "她不是我同学",
        "我想去吃饭",
        "你像我哥",
        "换个地方吃饭吧",
        "我真假分不清",
    ],
)
def test_ordinary_remarks_are_not_taken_for_a_correction_either(text: str) -> None:
    assert not looks_like_correction(text), text


@given(
    st.sampled_from(RECALLED_REMARKS),
    st.lists(
        st.sampled_from(["", "", " ", "\u3000", "，", "。", "！", "?", "~", "…"]), min_size=60
    ),
)
def test_spacing_punctuation_and_width_do_not_hide_a_remark(text: str, fillers: list[str]) -> None:
    """R17-23 hypothesis: the rule reads the message without white space and punctuation."""
    spread = "".join(fillers[i] + char for i, char in enumerate(text))
    assert looks_like_correction(spread), spread
    assert looks_like_correction(text.upper().replace("AI", "ＡＩ"))


@given(
    st.text(
        alphabet=list("今天好累呀吃饭睡觉我们去玩哈嗯哦的了吗呢吧a1 ，。！"),
        min_size=0,
        max_size=40,
    )
)
def test_chat_without_a_word_about_her_way_of_talking_is_not_a_remark(text: str) -> None:
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
    memory: Memory

    def new_service(self) -> CorrectionService:
        """A service over the same tables (the settings are read when it is made)."""
        services, clock = self.harness.services, self.harness.clock
        return CorrectionService(
            services,
            self.llm.client,
            self.recorder,
            BotTurnStore(services.db, clock),
            RoundStore(services.db),
            memory=MemoryManager(self.memory, self.llm.client),
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
async def rig(
    services: Services, clock: ManualClock, api: respx.MockRouter, embedder: HashingBackend
) -> AsyncIterator[Rig]:
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
    memory = make_memory(services)
    service = CorrectionService(
        services,
        llm.client,
        recorder,
        store,
        RoundStore(services.db),
        memory=MemoryManager(memory, llm.client),
    )
    harness.engine.attach_corrections(service)
    await harness.engine.start()
    yield Rig(harness, service, llm, recorder, memory)
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

    async def withdraw(self, asked_since: datetime) -> None:
        self.calls.append("withdraw")
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


# ------------------------------------------------- R17-11 (D-674): a fact in a /不像 wording

FACT = "我们是2019年在上海认识的"
REMEMBERED = texts.PREFIX + "记住了（第 1 条）：" + FACT + texts.REMEMBER_AS_WRITTEN


def remembered(rig: Rig) -> list[tuple[str, str]]:
    """The facts of the memory as ``(text, source)``."""
    listing = MemoryManager(rig.memory).list_items()
    return [(item.text, item.source) for item in listing.items]


async def ask_about_a_fact(rig: Rig, reply_id: str, wording: str = FACT) -> None:
    """What ``/不像 <说法>`` leaves behind when the wording is a fact: the open question."""
    services, clock = rig.harness.services, rig.harness.clock
    proposer = FactProposer(
        services,
        rig.llm.client,
        BotTurnStore(services.db, clock),
        PendingStore(services),
    )
    assert await proposer.propose(wording, reply_id) == texts.FACT_ASK
    pending = rig.service.pending()
    assert pending is not None and (pending.kind, pending.text) == (KIND_FACT, wording)


async def test_the_yes_to_the_fact_question_remembers_the_wording_once(rig: Rig) -> None:
    """R17-11: before the fix nothing ever called ``MemoryManager.remember`` for a correction."""
    rig.harness.services.settings.memory.fact_extraction = False  # the wording is stored as it is
    reply_id = await first_exchange(rig)
    await ask_about_a_fact(rig, reply_id)
    calls = rig.harness.writer.calls
    await rig.say("是", advance=60)
    assert rig.said[-1] == REMEMBERED and rig.harness.writer.calls == calls
    assert remembered(rig) == [(FACT, "user_command")]
    assert rig.service.pending() is None
    assert rig.recorder.feedback_of(reply_id) == []  # a fact question is not a "不像" verdict
    yes_row = next(r for r in stored(rig) if r.direction == "in" and r.text == "是")
    assert yes_row.is_command and stored(rig)[-1].is_command  # neither is conversation
    rig.replies("真的吗")
    await rig.say("是", advance=30)  # a second yes confirms nothing: it is a plain message
    assert rig.said[-1] == "真的吗" and len(remembered(rig)) == 1


async def test_a_fact_question_lapses_after_the_window_and_nothing_is_remembered(rig: Rig) -> None:
    rig.harness.services.settings.memory.fact_extraction = False
    reply_id = await first_exchange(rig)
    await ask_about_a_fact(rig, reply_id)
    rig.replies("是什么呀")
    await rig.say("是", advance=61 * 60)
    assert rig.said[-1] == "是什么呀" and remembered(rig) == []
    assert not next(r for r in stored(rig) if r.text == "是").is_command


async def test_a_fact_question_inside_the_window_is_still_open_at_59_minutes(rig: Rig) -> None:
    rig.harness.services.settings.memory.fact_extraction = False
    reply_id = await first_exchange(rig)
    await ask_about_a_fact(rig, reply_id)
    await rig.say("是", advance=59 * 60)
    assert rig.said[-1] == REMEMBERED


@pytest.mark.parametrize("word", ["不用", "算了", "不是", "否", "不用了！", "不要"])
async def test_a_no_closes_the_question_and_remembers_nothing(rig: Rig, word: str) -> None:
    rig.harness.services.settings.memory.fact_extraction = False
    reply_id = await first_exchange(rig)
    await ask_about_a_fact(rig, reply_id)
    calls = rig.harness.writer.calls
    await rig.say(word, advance=30)
    assert rig.said[-1] == texts.PREFIX + texts.CORRECTION_REJECTED
    assert rig.harness.writer.calls == calls  # she was not asked: it was not conversation
    assert remembered(rig) == [] and rig.service.pending() is None
    assert next(r for r in stored(rig) if r.direction == "in" and r.text == word).is_command


async def test_a_no_closes_the_not_like_question_too(rig: Rig) -> None:
    reply_id = await first_exchange(rig)
    rig.replies("好吧 我改改")
    await rig.say("她不会这么说")
    await rig.say("算了", advance=30)
    assert rig.said[-1] == texts.PREFIX + texts.CORRECTION_REJECTED
    assert rig.recorder.feedback_of(reply_id) == [] and rig.service.pending() is None


async def test_a_no_with_no_question_open_is_an_ordinary_message(rig: Rig) -> None:
    await first_exchange(rig)
    rig.replies("好吧")
    await rig.say("不用", advance=30)
    assert rig.said[-1] == "好吧"
    assert not next(r for r in stored(rig) if r.text == "不用").is_command


async def test_a_fact_question_is_closed_when_she_has_said_something_in_between(rig: Rig) -> None:
    rig.harness.services.settings.memory.fact_extraction = False
    reply_id = await first_exchange(rig)
    await ask_about_a_fact(rig, reply_id)
    rig.replies("你想听什么呀")
    await rig.say("你再说一句", advance=30)
    rig.replies("好的好的")
    await rig.say("是", advance=30)
    assert rig.said[-1] == "好的好的" and remembered(rig) == []


async def test_the_open_fact_question_survives_a_restart(rig: Rig) -> None:
    reply_id = await first_exchange(rig)
    await ask_about_a_fact(rig, reply_id)
    again = rig.new_service()
    pending = again.pending()
    assert pending is not None and (pending.kind, pending.text) == (KIND_FACT, FACT)
    assert await again.is_confirmation("是", rig.harness.clock.now_utc() + timedelta(minutes=1))
    assert await again.is_confirmation("不用", rig.harness.clock.now_utc() + timedelta(minutes=1))


async def test_a_newer_question_of_the_other_kind_replaces_the_fact_question(rig: Rig) -> None:
    reply_id = await first_exchange(rig)
    await ask_about_a_fact(rig, reply_id)
    rig.replies("好吧 我改改")
    await rig.say("她不会这么说", advance=30)
    pending = rig.service.pending()
    assert pending is not None and (pending.kind, pending.text) == (KIND_NOT_LIKE, None)


def test_a_question_stored_before_facts_were_asked_about_reads_as_a_not_like_question(
    rig: Rig,
) -> None:
    services = rig.harness.services
    store = PendingStore(services)
    moment = rig.harness.clock.now_utc()
    with services.db.transaction(bump_state=False) as session:
        put_setting(
            session,
            "learning.correction.pending",
            {"reply_id": "r-1", "asked_at": moment.isoformat()},
            clock=services.clock,
            record_history=False,
        )
    assert store.get() == PendingCorrection("r-1", moment, KIND_NOT_LIKE, None)
    for broken in (
        {"reply_id": "r-1", "asked_at": moment.isoformat(), "kind": "other"},
        {"reply_id": "r-1", "asked_at": moment.isoformat(), "kind": KIND_FACT},  # no wording
        {"reply_id": "r-1", "asked_at": "yesterday"},
        {"asked_at": moment.isoformat()},
        "not a dict",
    ):
        with services.db.transaction(bump_state=False) as session:
            put_setting(
                session,
                "learning.correction.pending",
                broken,
                clock=services.clock,
                record_history=False,
            )
        assert store.get() is None, broken
    store.put(PendingCorrection("r-2", moment, KIND_FACT, FACT))
    with services.db.session() as session:
        raw = get_setting(session, "learning.correction.pending")
    assert raw == {"reply_id": "r-2", "asked_at": moment.isoformat(), "kind": "fact", "text": FACT}
    store.put(None)
    assert store.get() is None


class FailingMemory:
    """A memory manager whose ``remember`` fails once and then works."""

    def __init__(self, inner: MemoryManager) -> None:
        self.inner = inner
        self.failed = False

    async def remember(self, text: str) -> Any:
        if not self.failed:
            self.failed = True
            raise RuntimeError("the memory is down")
        return await self.inner.remember(text)


async def test_a_memory_that_fails_leaves_the_fact_question_open_for_another_yes(rig: Rig) -> None:
    rig.harness.services.settings.memory.fact_extraction = False
    reply_id = await first_exchange(rig)
    await ask_about_a_fact(rig, reply_id)
    broken = FailingMemory(MemoryManager(rig.memory))
    service = rig.new_service()
    service._memory = broken  # type: ignore[assignment]
    now = rig.harness.clock.now_utc() + timedelta(seconds=30)
    with pytest.raises(RuntimeError, match="memory is down"):
        await service.confirm(CommandContext(now, "in-1"))
    assert service.pending() is not None and remembered(rig) == []
    outcome = await service.confirm(CommandContext(now, "in-2"))
    assert outcome == CommandOutcome(REMEMBERED) and service.pending() is None


@pytest.mark.parametrize("text", ["不用", "算了", "不是", "否", " 不用了。", "不要"])
def test_the_no_is_a_short_no(text: str) -> None:
    assert is_rejection_word(text) and not is_confirmation_word(text)


@pytest.mark.parametrize("text", ["", "是", "好", "不是吧", "不用管我", "我不要", "否则呢"])
def test_anything_else_is_not_the_no(text: str) -> None:
    assert not is_rejection_word(text)


async def test_a_yes_for_a_reply_that_is_gone_is_told_so_and_closes_the_question(rig: Rig) -> None:
    await first_exchange(rig)
    now = rig.harness.clock.now_utc()
    rig.service._store(PendingCorrection("no-such-reply", now))
    outcome = await rig.service.confirm(CommandContext(now + timedelta(seconds=30), "in-1"))
    assert outcome == CommandOutcome(texts.PREFIX + texts.NOT_LIKE_NOTHING)
    assert rig.service.pending() is None  # the question is closed whether or not it recorded
    assert await rig.service.confirm(CommandContext(now + timedelta(seconds=40), "in-2")) is None


# ------------------------------------------- R18-B6 (D-755): a question that never reached him

REFUSALS = [
    OutboundResult.failure(OutboundKind.REJECTED, "bad_request"),
    OutboundResult.failure(OutboundKind.NETWORK, "connect_error"),
    OutboundResult.failure(OutboundKind.WINDOW_REJECTED, "window_elapsed"),
]


@pytest.mark.parametrize("refusal", REFUSALS, ids=lambda r: r.kind.value)
async def test_r18_b6_a_question_the_channel_does_not_deliver_is_withdrawn(
    rig: Rig, judge: Judge, refusal: OutboundResult
) -> None:
    """R18-B6: her reply goes out, the question after it does not.

    Before: the question stayed open (and a refused or failed one was even written down as said),
    so the user's next "是" - an answer to her, not to a question he never saw - was swallowed as
    the confirmation and recorded a "不像".  Now nothing is written down, the question is closed,
    and "是" is chat.
    """
    reply_id = await first_exchange(rig)
    rig.replies("好吧 我改改")
    rig.harness.channel.results.extend([OutboundResult.success(message_id="o1"), refusal])
    await rig.say("她不会这么说")
    assert len(judge.asked) == 1  # it was a correction, and the question was tried
    assert rig.said[-1] == "好吧 我改改"
    assert rig.service.pending() is None
    assert QUESTION not in [row.text for row in stored(rig)]  # no "said" row of it either
    rig.replies("嗯？")
    await rig.say("是", advance=30)
    assert rig.said[-1] == "嗯？"  # answered by her, as chat
    assert rig.recorder.feedback_of(reply_id) == []
    yes_row = next(r for r in stored(rig) if r.direction == "in" and r.text == "是")
    assert not yes_row.is_command


async def test_r18_b6_a_question_with_an_unknown_outcome_counts_as_asked(
    rig: Rig, judge: Judge
) -> None:
    """R18-B6: the channel cannot say whether it arrived: it may be on his phone, it stays open."""
    reply_id = await first_exchange(rig)
    rig.replies("好吧 我改改")
    rig.harness.channel.results.extend(
        [
            OutboundResult.success(message_id="o1"),
            OutboundResult.failure(OutboundKind.AMBIGUOUS, "read_timeout"),
        ]
    )
    await rig.say("她不会这么说")
    assert rig.service.pending() is not None
    assert QUESTION in [row.text for row in stored(rig)]
    await rig.say("是", advance=30)
    assert rig.said[-1] == DONE
    assert [f.type for f in rig.recorder.feedback_of(reply_id)] == ["not_like"]


async def test_r18_b6_a_command_answer_that_carries_a_question_and_is_not_delivered_withdraws_it(
    rig: Rig,
) -> None:
    """R18-B6: ``/不像 <说法>`` stores the fact question while it runs (D-674); when its answer
    is refused the question is closed too, and a later "是" is chat."""
    reply_id = await first_exchange(rig)
    commands = rig.harness.commands
    pending = PendingStore(rig.harness.services)

    def ask(text: str, context: Any) -> None:
        moment = rig.harness.clock.now_utc()
        pending.put(PendingCorrection(reply_id, moment, KIND_FACT, FACT))

    commands.before = ask
    rig.harness.channel.results.append(OutboundResult.failure(OutboundKind.REJECTED, "x"))
    await rig.harness.message("/不像 她不会这么说")
    assert rig.service.pending() is None
    rig.replies("嗯？")
    await rig.say("是", advance=30)
    assert rig.said[-1] == "嗯？" and remembered(rig) == []


async def test_r18_b6_an_older_question_is_not_withdrawn_by_a_system_message_that_fails(
    rig: Rig,
) -> None:
    """R18-B6: only the question asked for the message that failed is closed, not one before."""
    reply_id = await first_exchange(rig)
    rig.service._store(PendingCorrection(reply_id, rig.harness.clock.now_utc()))
    rig.harness.clock.tick(5)
    rig.harness.channel.results.append(OutboundResult.failure(OutboundKind.REJECTED, "x"))
    await rig.harness.engine.notify("导入完成")
    assert rig.service.pending() is not None
    assert texts.PREFIX + "导入完成" not in [row.text for row in stored(rig)]


class QuestionThenBroken(BrokenPort):
    """A service that asks its question and then cannot close it."""

    async def propose(self, answered: Sequence[str]) -> str | None:
        self.calls.append("propose")
        return QUESTION


async def test_r18_b6_a_withdrawal_that_fails_never_costs_the_user_a_reply(
    services: Services, clock: ManualClock
) -> None:
    """R18-B6: the question was refused and the service cannot close it: the talk goes on."""
    harness = build_harness(services, clock)
    port = QuestionThenBroken()
    harness.engine.attach_corrections(port)
    await harness.engine.start()
    try:
        harness.writer.add(make_draft("嗯嗯"), make_draft("什么呀"))
        harness.channel.results.extend(
            [
                OutboundResult.success(message_id="o1"),
                OutboundResult.failure(OutboundKind.REJECTED, "bad_request"),
            ]
        )
        await harness.message("你好")
        await run_to_idle(harness.engine, harness.clock)
        assert port.calls[-2:] == ["propose", "withdraw"]
        port.confirmed = False
        await harness.message("在吗")
        await run_to_idle(harness.engine, harness.clock)
        assert harness.channel.texts[:2] == ["嗯嗯", "什么呀"]  # (then the question, delivered)
    finally:
        await harness.engine.stop()
