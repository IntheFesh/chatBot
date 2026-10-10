"""The engine end to end: terminal channel, real pipeline, scripted DeepSeek (round 09-3).

What is real: the local console channel, the stores, the live data source (profile, routine,
memory, retrieval of a synthetic conversation), the prompt, the DeepSeek client (HTTP intercepted
by ``respx``), the post-processing, the state machine and its pacing.  What is scripted: the manual
clock, her day (asleep, busy) and the words DeepSeek answers with.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections import deque
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import httpx
import pytest
import respx
from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.console import RecordingOutput, ScriptedInput
from tests.support.deepseek import API, TEST_KEY, ok
from tests.support.embedding import HashingBackend
from tests.support.engine_harness import START, FixedDay, bubbles_written, run_to_idle
from tests.support.synth_chat import ChatSpec, build_chat
from tests.support.waiting import wait_until
from twin.channel.local import TYPING_TEXT, LocalConsoleChannel
from twin.engine.component import EngineComponent, build_engine
from twin.engine.machine import ConversationEngine
from twin.engine.roundstate import RoundData
from twin.llm.runtime import DEEPSEEK_SECRET, build_llm_runtime
from twin.memory.recent import BotMessage
from twin.profile.api import load_activity_model, load_profile
from twin.profile.builder import rebuild
from twin.retrieval.embedder import EmbeddingService
from twin.retrieval.indexer import run_index
from twin.retrieval.query import ExampleQuery, ExampleRetriever, QueryTurn
from twin.schedule.plan_model import BusySpan, busy_ref
from twin.schedule.service import schedule_kit
from twin.services import Services
from twin.storage.engine_models import BotTurn
from twin.storage.models import Alert, CostLedger

MARKER = "端到端专属暗号"
LOCAL_MIDNIGHT = START + timedelta(hours=17)
WAKE = START + timedelta(hours=24)


@dataclass
class World:
    """A running engine on the terminal channel with DeepSeek answering from a script."""

    services: Services
    clock: ManualClock
    channel: LocalConsoleChannel
    input: ScriptedInput
    output: RecordingOutput
    engine: ConversationEngine
    component: EngineComponent
    api: respx.MockRouter
    route: respx.Route
    replies: deque[str] = field(default_factory=deque)
    extracted: list[list[BotMessage]] = field(default_factory=list)
    gate: asyncio.Event | None = None
    started: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def bot_lines(self) -> list[str]:
        return [line for line in self.output.lines if line.startswith("bot:")]

    def requests(self) -> list[dict[str, Any]]:
        return [json.loads(call.request.content) for call in self.route.calls]

    def say(self, text: str) -> None:
        self.input.feed(text)

    async def said(self, count: int) -> None:
        """Wait until ``count`` messages of the user are stored *and queued* by the engine.

        A stored row is not yet a queued message: the engine writes the row first and queues it a
        thread-hop later, and a test that runs the engine to idle in between sees "idle, nothing
        waiting" - which is what a slow disk (Windows) shows every time.  The component counts a
        message as handled when ``handle_message`` has returned, i.e. after it is queued.
        """
        await wait_until(lambda: self.component.handled >= count)

    def rows(self) -> list[BotTurn]:
        with self.services.db.session() as session:
            found = list(session.scalars(select(BotTurn).order_by(BotTurn.at, BotTurn.id)))
            session.expunge_all()
        return found


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
async def world(
    services: Services,
    clock: ManualClock,
    embedder: HashingBackend,
    api: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[World]:
    services.settings.retrieval.model = embedder.info.model
    build_chat(services, ChatSpec(days=40))
    rebuild(services, "all")  # her profile and routine, from the synthetic conversation
    run_index(services)
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    schedule_kit(services).time.attach_states(FixedDay((START - timedelta(days=1), "free")))
    extracted: list[list[BotMessage]] = []
    monkeypatch.setattr(
        "twin.engine.component.queue_bot_extraction",
        lambda _services, turns: extracted.append(list(turns)) or "job",
    )
    inp, out = ScriptedInput(), RecordingOutput()
    channel = LocalConsoleChannel.from_services(services, input=inp, output=out)
    llm = build_llm_runtime(services)
    engine = build_engine(services, channel, runtime=llm, rng=random.Random(2))
    component = EngineComponent(engine, channel, services, restart_dispatch=False)
    replies: deque[str] = deque()
    started = asyncio.Event()
    holder: dict[str, World] = {}

    async def answer(request: httpx.Request) -> httpx.Response:
        started.set()
        mine = holder["w"]
        if mine.gate is not None:
            await mine.gate.wait()
        text = replies.popleft() if replies else "好的"
        return ok(content=text, prompt=400, hit=300, completion_tokens=12)

    route = api.post(API).mock(side_effect=answer)
    built = World(
        services, clock, channel, inp, out, engine, component, api, route, replies, extracted
    )
    built.started = started
    holder["w"] = built
    await channel.start()
    await component.start()
    yield built
    await component.stop()
    await channel.stop()
    await llm.client.aclose()


# --------------------------------------------------------------------- the whole way


async def test_a_conversation_from_the_terminal_to_the_memory(world: World) -> None:
    world.replies.append("好呀\n我也想去")
    world.say("周末去看电影吧")
    await world.said(1)
    await run_to_idle(world.engine, world.clock)
    assert world.bot_lines == ["bot: 好呀", "bot: 我也想去"]
    assert TYPING_TEXT in world.output.lines  # she showed that she was typing
    rows = world.rows()
    assert [(r.direction, r.kind, r.text) for r in rows] == [
        ("in", "text", "周末去看电影吧"),
        ("out", "text", "好呀"),
        ("out", "text", "我也想去"),
    ]
    first = rows[1]
    assert first.backend == "deepseek" and first.cost_usd and first.cost_usd > 0
    assert first.timings["delay"] > 0 and first.timings["generate"] >= 0
    assert any(a["step"] == "decided_free" for a in first.actions)
    last_prompt = world.requests()[0]["messages"][-1]["content"]
    assert "周末去看电影吧" in last_prompt and "此刻" in last_prompt
    # the conversation is handed to the memory after thirty quiet minutes (R-MEM-007); her own
    # delay can be longer than that, then the message of the user goes first and her reply later
    await world.clock.advance(31 * 60)
    conversation = ["周末去看电影吧", "好呀", "我也想去"]
    await wait_until(lambda: len(handed_over(world)) == 3)
    assert handed_over(world) == conversation  # each message once, in order


def handed_over(world: World) -> list[str]:
    return [message.text for batch in world.extracted for message in batch]


async def test_a_second_round_has_the_first_in_its_history(world: World) -> None:
    world.replies.extend(["好呀", "那就六点"])
    world.say("周末去看电影吧")
    await world.said(1)
    await run_to_idle(world.engine, world.clock)
    world.say("几点好呢")
    await world.said(2)
    await run_to_idle(world.engine, world.clock)
    second = world.requests()[1]["messages"]
    roles = [m["role"] for m in second]
    assert roles[0] == "system" and roles[-3:] == ["user", "assistant", "user"]
    assert second[-3]["content"] == "周末去看电影吧" and second[-2]["content"] == "好呀"
    assert "几点好呢" in second[-1]["content"]


# ------------------------------------------------------------------------ her day


async def test_a_message_in_the_night_is_answered_after_she_wakes_like_someone_who_just_woke(
    world: World,
) -> None:
    schedule_kit(world.services).time.attach_states(
        FixedDay(
            (START - timedelta(days=1), "free"),
            (START + timedelta(hours=15), "deep_sleep"),
            (START + timedelta(hours=23, minutes=30), "sleep_edge"),
            (WAKE, "free"),
        )
    )
    world.clock.set_time(LOCAL_MIDNIGHT)
    world.replies.append("早呀 刚醒")
    world.say("睡了吗")
    await world.said(1)
    await run_to_idle(
        world.engine, world.clock, until=lambda: world.engine.snapshot().state == "DECIDING"
    )
    assert world.bot_lines == []
    decision = RoundData.of(world.engine.snapshot()).decision
    assert decision is not None and decision.mode == "asleep" and decision.wake_at == WAKE
    await run_to_idle(world.engine, world.clock)
    assert world.clock.now_utc() >= WAKE + timedelta(minutes=5)
    assert world.bot_lines and "刚醒" in world.requests()[0]["messages"][-1]["content"]


async def test_when_she_is_busy_the_delay_comes_from_that_busy_window_of_her_routine(
    world: World,
) -> None:
    model = load_activity_model(world.services, "live")
    assert model is not None
    windows = model.busy_windows("workday")
    assert windows, "the synthetic conversation has slow afternoons"
    span = BusySpan(
        START - timedelta(hours=1),
        START + timedelta(hours=6),
        windows[0].label(),
        "inferred",
        busy_ref("workday", None, 0),
    )
    schedule_kit(world.services).time.attach_states(
        FixedDay(
            (START - timedelta(days=1), "free"), (START - timedelta(hours=1), "busy"), busy=span
        )
    )
    world.say("在忙吗")
    await world.said(1)
    await run_to_idle(
        world.engine, world.clock, until=lambda: world.engine.snapshot().state == "DECIDING"
    )
    decision = RoundData.of(world.engine.snapshot()).decision
    latency = windows[0].latency
    assert decision is not None and decision.mode == "busy"
    assert latency.minimum() <= decision.delay_s <= latency.maximum()
    assert decision.delay_s >= 60.0  # minutes, not the seconds of a free afternoon
    await run_to_idle(world.engine, world.clock)
    assert world.bot_lines == ["bot: 好的"]


# ------------------------------------------------------------ cancelling and cutting in


async def test_a_message_while_the_reply_is_being_written_cancels_it_and_both_are_answered(
    world: World,
) -> None:
    world.gate = asyncio.Event()
    world.say("周末去看电影吧")
    await world.said(1)
    await run_to_idle(world.engine, world.clock, until=world.started.is_set)
    world.say("算了我们明天去")
    await world.said(2)
    await wait_until(lambda: world.engine.snapshot().state == "COLLECTING")
    world.gate.set()
    world.replies.append("好呀")
    await run_to_idle(world.engine, world.clock)
    assert world.bot_lines == ["bot: 好呀"]
    last_prompt = world.requests()[-1]["messages"][-1]["content"]
    assert "周末去看电影吧" in last_prompt and "算了我们明天去" in last_prompt
    with world.services.db.session() as session:
        ledger_rows = list(session.scalars(select(CostLedger)))
    assert len(ledger_rows) == 1  # only the request that came back is billed


async def test_a_message_while_she_sends_keeps_what_is_out_and_writes_the_rest_again(
    world: World,
) -> None:
    world.replies.extend(["第一条\n第二条\n第三条", "好的 接着说"])
    world.say("给我讲个故事")
    await world.said(1)
    await run_to_idle(world.engine, world.clock, until=bubbles_written(world.engine, 1))
    world.say("等等")
    await world.said(2)
    await run_to_idle(world.engine, world.clock)
    assert world.bot_lines[0] == "bot: 第一条" and "bot: 第二条" not in world.bot_lines
    again = world.requests()[1]["messages"][-1]["content"]
    assert "第一条" in again and "不要重复" in again and "等等" in again


async def test_a_kill_in_the_middle_of_sending_is_resumed_from_the_same_database(
    world: World,
) -> None:
    world.replies.append("一\n二\n三")
    world.say("数数")
    await world.said(1)
    await run_to_idle(world.engine, world.clock, until=bubbles_written(world.engine, 1))
    await world.component.stop()  # the process dies here; the channel is the platform's
    stored = world.engine.snapshot()
    assert stored.state == "SENDING" and [s["text"] for s in stored.sent] == ["一"]
    revived = build_engine(world.services, world.channel, rng=random.Random(3))
    component = EngineComponent(revived, world.channel, world.services, restart_dispatch=False)
    await component.start()
    try:
        await run_to_idle(revived, world.clock)
        assert [line for line in world.bot_lines if line != "bot: 一"] == ["bot: 二", "bot: 三"]
        assert len(world.requests()) == 1  # nothing was written again
        with world.services.db.session() as session:
            replies = {r.reply_id for r in session.scalars(select(BotTurn)) if r.direction == "out"}
        assert len(replies) == 1
    finally:
        await component.stop()


# ------------------------------------------------------------------------- quota


async def test_when_the_platform_leaves_few_messages_the_bubbles_are_merged(
    world: World, services: Services
) -> None:
    await world.component.stop()
    await world.channel.stop()
    world.output.lines.clear()
    tight_input = ScriptedInput()
    tight = LocalConsoleChannel.from_services(
        services, input=tight_input, output=world.output, quota=4
    )
    engine = build_engine(services, tight, rng=random.Random(4))
    component = EngineComponent(engine, tight, services, restart_dispatch=False)
    await tight.start()
    await component.start()
    try:
        world.replies.append("一\n二\n三\n四\n五")
        tight_input.feed("说五句")
        await wait_until(lambda: component.handled >= 1)
        await run_to_idle(engine, world.clock)
        assert len(world.bot_lines) == 2  # 4 left - 2 kept for what she starts = 2 bubbles
        first = next(r for r in world.rows() if r.direction == "out")
        assert any(a["step"] == "quota_merge" for a in first.actions)
        assert "bot: 一" in world.bot_lines[0] and "五" in world.bot_lines[-1]
    finally:
        await component.stop()
        await tight.stop()


# ---------------------------------------------------------------------------- safety


async def test_a_crisis_is_answered_out_of_the_role_with_the_help_line_and_an_alert(
    world: World,
) -> None:
    world.replies.append('{"is_crisis": true, "severity": "high", "reason": "says so"}')
    world.say("我真的不想活了")
    await world.said(1)
    await run_to_idle(world.engine, world.clock)
    said = "\n".join(world.bot_lines)
    assert "988" in said  # Chicago: the American help line
    assert len(world.requests()) == 1  # the judgement; no persona reply was written
    assert "is_crisis" in world.requests()[0]["messages"][-1]["content"]
    first = next(r for r in world.rows() if r.direction == "out")
    assert first.backend == "safety"
    with world.services.db.session() as session:
        alerts = [(a.category, a.severity) for a in session.scalars(select(Alert))]
    assert ("crisis_detected", "critical") in alerts


async def test_an_ordinary_conversation_logs_no_words_of_it(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    world.replies.append("秘密回复")
    world.say("秘密消息")
    await world.said(1)
    await run_to_idle(world.engine, world.clock)
    logged = "\n".join(str(record.__dict__) for record in caplog.records)
    assert "秘密消息" not in logged and "秘密回复" not in logged


# --------------------------------------------------------- what she says stays hers


async def test_what_the_bot_said_never_becomes_what_she_sounds_like(
    world: World, embedder: HashingBackend
) -> None:
    world.replies.extend([f"{MARKER}回复一", f"{MARKER}回复二"])
    world.say(f"{MARKER}用户一")
    await world.said(1)
    await run_to_idle(world.engine, world.clock)
    world.say(f"{MARKER}用户二")
    await world.said(2)
    await run_to_idle(world.engine, world.clock)
    assert len([r for r in world.rows() if MARKER in r.text]) == 4
    rebuild(world.services, "all")
    profile = load_profile(world.services, "live")
    assert profile is not None
    assert MARKER not in json.dumps(profile.phrases() or {}, ensure_ascii=False)
    run_index(world.services)
    retriever = ExampleRetriever(world.services, EmbeddingService(embedder))
    asked = [QueryTurn(False, f"{MARKER}用户一"), QueryTurn(True, f"{MARKER}回复一")]
    examples = await retriever.query(ExampleQuery(asked, 720, "workday", None, 8))
    assert examples
    shown = " ".join(
        line.text
        for example in examples
        for line in (*(item for turn in example.context for item in turn.lines), *example.reply)
    )
    assert MARKER not in shown
