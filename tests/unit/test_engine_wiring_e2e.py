"""The reply engine with its command router and the style model, end to end (round 09 step 4).

What is real: the terminal channel, the stores, the live data source (profile, routine, memory,
retrieval of a synthetic conversation), the prompts, the DeepSeek client (HTTP intercepted by
``respx``), the post-processing, the state machine, the command router of ``twin.commands``, the
style runtime with its backends and backend selector.  What is scripted: the manual clock, her
day, the words DeepSeek answers with, and the style model - a client the test controls
(``ScriptedStyleClient``) or, in one test, the real HTTP client against the local test server.
"""

from __future__ import annotations

import json
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
from tests.support.style_models import ScriptedStyleClient, down, register_model
from tests.support.style_server import StyleServer, llama_defaults
from tests.support.synth_chat import ChatSpec, build_chat
from tests.support.waiting import wait_until
from twin.channel.local import TYPING_TEXT, LocalConsoleChannel
from twin.commands import texts
from twin.config.runtime import (
    BACKEND_ACTIVE,
    BACKEND_FALLBACK,
    SHOW_THINKING,
    THINKING_CHAT,
)
from twin.engine.component import EngineComponent, build_engine
from twin.engine.machine import ConversationEngine
from twin.engine.style_runtime import StyleRuntime
from twin.llm.runtime import DEEPSEEK_SECRET, LlmRuntime, build_llm_runtime
from twin.memory.recent import BotMessage
from twin.profile.builder import rebuild
from twin.profile.persona import compose
from twin.profile.persona.store import PersonaStore
from twin.retrieval.indexer import run_index
from twin.schedule.service import schedule_kit
from twin.services import Services
from twin.storage.engine_models import BotTurn, Feedback
from twin.storage.models import Alert

PREFIX = texts.PREFIX
CONTINUATION = " " * 5  # how the terminal indents the further lines of a message
PLAN = json.dumps(
    {
        "reply": True,
        "intent": "接住对方的话",
        "facts_to_use": [],
        "tone": "随意",
        "bubble_hint": "一条短句",
        "sticker_hint": "",
    },
    ensure_ascii=False,
)
CARD_MARKER = "训练时的口头禅是嘿嘿"
ASSISTANT_OPENER = "<|im_start|>assistant\n"


@dataclass
class World:
    """A running engine on the terminal channel; DeepSeek and the style model follow scripts."""

    services: Services
    clock: ManualClock
    channel: LocalConsoleChannel
    input: ScriptedInput
    output: RecordingOutput
    engine: ConversationEngine
    component: EngineComponent
    llm: LlmRuntime
    style: StyleRuntime
    style_client: ScriptedStyleClient | None
    route: respx.Route
    replies: deque[str] = field(default_factory=deque)
    reasoning: deque[str | None] = field(default_factory=deque)
    extracted: list[list[BotMessage]] = field(default_factory=list)
    started: int = 0

    @property
    def bot_lines(self) -> list[str]:
        """The messages of the bot as the terminal shows them: ``bot: <text>``, one per message.

        The channel writes a message of several lines as ``bot: first`` and the other lines
        indented under it; they are put back together here.
        """
        messages: list[str] = []
        for line in self.output.lines:
            if line.startswith("bot:"):
                messages.append(line)
            elif line.startswith(CONTINUATION) and messages:
                messages[-1] += "\n" + line[len(CONTINUATION) :]
        return messages

    @property
    def command_lines(self) -> list[str]:
        return [line for line in self.bot_lines if line.startswith(f"bot: {PREFIX}")]

    def requests(self) -> list[dict[str, Any]]:
        return [json.loads(call.request.content) for call in self.route.calls]

    def say(self, text: str) -> None:
        self.input.feed(text)

    async def said(self, count: int) -> None:
        """Wait until ``count`` messages of the user are handled: stored, answered or queued.

        Stored is not queued: the engine writes the row first and queues the message a thread-hop
        later (a slash message is shown to the command router in between), and a test that runs
        the engine to idle then sees "idle, nothing waiting".  The component counts a message as
        handled when ``handle_message`` has returned - for ``/重来`` that includes queueing the
        messages of the reply again.
        """
        await wait_until(lambda: self.component.handled >= count)

    async def chat(self, text: str, count: int) -> None:
        """The user writes a message (the ``count``-th); returns once the engine queued it."""
        self.say(text)
        await self.said(count)

    async def command(self, text: str, *, count: int, answers: int) -> str:
        """Send a command and wait for its answer: the number of messages said, answers expected."""
        self.say(text)
        await self.said(count)
        await wait_until(lambda: len(self.command_lines) >= answers)
        return self.command_lines[answers - 1].removeprefix("bot: ")

    def rows(self) -> list[BotTurn]:
        with self.services.db.session() as session:
            found = list(session.scalars(select(BotTurn).order_by(BotTurn.at, BotTurn.id)))
            session.expunge_all()
        return found

    def feedback(self) -> list[tuple[str, str]]:
        with self.services.db.session() as session:
            return [(row.type, row.reply_id) for row in session.scalars(select(Feedback))]

    def alerts(self) -> list[tuple[str, str]]:
        with self.services.db.session() as session:
            rows = session.scalars(select(Alert).order_by(Alert.created_at, Alert.id))
            return [(row.category, row.severity) for row in rows]


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


def stored_card(services: Services) -> None:
    """The pre-holdout card the style model below was trained with (version v1)."""
    text = (
        compose.stats_block(["几乎不用逗号"])
        + compose.auto_block(f"### 风格\n- {CARD_MARKER}\n\n### 基本情况\n- 一件事实\n\n")
        + compose.manual_block([], None)
    )
    PersonaStore(services.db, services.clock).add_version("pre_holdout", text, reason="generate")


async def assemble(
    services: Services,
    clock: ManualClock,
    embedder: HashingBackend,
    api: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    *,
    style_client: ScriptedStyleClient | None,
) -> World:
    """Everything of ``twin chat --local``: a synthetic past and the engine ``build_engine`` makes.

    ``style_client`` replaces the style model's transport; ``None`` leaves what the settings say
    (the HTTP client of ``style_model.endpoint``).
    """
    services.settings.retrieval.model = embedder.info.model
    build_chat(services, ChatSpec(days=40))
    rebuild(services, "all")  # her profile and routine, from the synthetic conversation
    run_index(services)
    services.runtime.initialize()  # as at the start of the application
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
    style = StyleRuntime.from_services(services, llm, client=style_client)
    engine = build_engine(services, channel, runtime=llm, style=style, rng=random.Random(2))
    component = EngineComponent(engine, channel, services, restart_dispatch=False, style=style)
    replies: deque[str] = deque()
    reasoning: deque[str | None] = deque()
    holder: dict[str, World] = {}

    async def answer(request: httpx.Request) -> httpx.Response:
        holder["w"].started += 1
        text = replies.popleft() if replies else "好的"
        why = reasoning.popleft() if reasoning else None
        return ok(content=text, reasoning=why, prompt=400, hit=300, completion_tokens=12)

    route = api.post(API).mock(side_effect=answer)
    world = World(
        services,
        clock,
        channel,
        inp,
        out,
        engine,
        component,
        llm,
        style,
        style_client,
        route,
        replies,
        reasoning,
        extracted,
    )
    holder["w"] = world
    await channel.start()
    await component.start()
    return world


async def close(world: World) -> None:
    await world.component.stop()
    await world.channel.stop()
    await world.llm.client.aclose()


@pytest.fixture
async def world(
    services: Services,
    clock: ManualClock,
    embedder: HashingBackend,
    api: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[World]:
    built = await assemble(
        services, clock, embedder, api, monkeypatch, style_client=ScriptedStyleClient()
    )
    yield built
    await close(built)


def handed_over(world: World) -> list[str]:
    """What went to the memory so far, batch after batch."""
    return [message.text for batch in world.extracted for message in batch]


def register_style_model(services: Services) -> None:
    """A registered, active, gate-passed model locked to card v1 (what activating one makes)."""
    stored_card(services)
    register_model(services, persona_version="v1")


# ------------------------------------------------------------------ the command path


async def test_a_command_is_answered_at_once_in_the_system_voice_and_costs_nothing(
    world: World,
) -> None:
    before = world.clock.now_utc()
    reply = await world.command("/帮助", count=1, answers=1)
    assert reply.startswith(PREFIX)
    for name in ("/帮助", "/状态", "/思考", "/显示思考", "/后端", "/重来"):
        assert name in reply
    assert world.clock.now_utc() == before  # no quiet window, no delay of hers: not a second
    assert TYPING_TEXT not in world.output.lines  # she did not "type" a system message
    assert world.route.call_count == 0  # no model was asked
    assert world.engine.snapshot().state == "IDLE" and world.engine.snapshot().pending == ()
    rows = world.rows()
    assert [(r.direction, r.is_command, r.backend) for r in rows] == [
        ("in", True, None),
        ("out", True, "command"),
    ]
    assert rows[0].text == "/帮助" and rows[1].text == reply


async def test_command_replies_reach_the_user_exactly_as_the_router_wrote_them(
    world: World,
) -> None:
    """The persona post-processing (full stops, comma sentences, AI phrases) never touches them."""
    think = await world.command("/思考 开", count=1, answers=1)
    assert think == PREFIX + texts.THINK_SET.format(mode="开")  # the full stop stays
    show = await world.command("/显示思考 开", count=2, answers=2)
    assert show == PREFIX + texts.SHOW_SET_ON
    refused = await world.command("/后端 hybrid", count=3, answers=3)
    assert refused == PREFIX + texts.BACKEND_REFUSED.format(
        reason=texts.BACKEND_REASONS["not_registered"].format(detail="")
    )  # commas and full stops inside, and not split into bubbles
    assert "，" in refused and refused.endswith("。")
    unknown = await world.command("/没有这条指令", count=4, answers=4)
    assert unknown.startswith(PREFIX) and "/帮助" in unknown  # a slash command is never chat
    assert world.route.call_count == 0
    stored = {row.text for row in world.rows() if row.direction == "out"}
    assert {think, show, refused, unknown} <= stored  # what was sent is what was stored


async def test_commands_never_reach_the_prompt_the_memory_or_the_training_rows(
    world: World,
) -> None:
    await world.command("/帮助", count=1, answers=1)
    world.replies.append("好呀")
    await world.chat("周末去看电影吧", 2)
    await run_to_idle(world.engine, world.clock)
    await world.command("/思考 自动", count=3, answers=2)
    world.replies.append("六点吧")
    await world.chat("几点好呢", 4)
    await run_to_idle(world.engine, world.clock)
    second = world.requests()[1]["messages"]
    assert [m["role"] for m in second][-3:] == ["user", "assistant", "user"]
    prompt = json.dumps(second, ensure_ascii=False)
    assert "/帮助" not in prompt and "/思考" not in prompt and "⚙️" not in prompt
    await world.clock.advance(31 * 60)  # thirty quiet minutes: the conversation goes to the memory
    await wait_until(lambda: sum(len(batch) for batch in world.extracted) >= 4)
    handed = [message.text for batch in world.extracted for message in batch]
    assert handed == ["周末去看电影吧", "好呀", "几点好呢", "六点吧"]
    marked = {row.text: row.is_command for row in world.rows()}
    assert marked["/帮助"] and marked["/思考 自动"] and not marked["周末去看电影吧"]


async def test_an_ordinary_message_goes_the_normal_way_beside_the_commands(world: World) -> None:
    await world.command("/帮助", count=1, answers=1)
    world.replies.append("好呀\n我也想去")
    await world.chat("周末去看电影吧", 2)
    await run_to_idle(world.engine, world.clock)
    assert world.bot_lines[1:] == ["bot: 好呀", "bot: 我也想去"]
    assert TYPING_TEXT in world.output.lines  # the normal path types, the command path does not
    out = [r for r in world.rows() if r.direction == "out" and not r.is_command]
    assert [r.text for r in out] == ["好呀", "我也想去"] and out[0].backend == "deepseek"
    assert out[0].timings["delay"] > 0  # her own delay, which a command never has


@pytest.mark.parametrize("text", ["/::)", "/:8-)", "/home/user", "/ 思考", "/"])
async def test_a_slash_that_is_not_a_command_is_just_a_message(world: World, text: str) -> None:
    world.replies.append("哈哈")
    await world.chat(text, 1)
    await run_to_idle(world.engine, world.clock)
    assert world.bot_lines == ["bot: 哈哈"]
    rows = world.rows()
    assert [(r.direction, r.is_command) for r in rows] == [("in", False), ("out", False)]


# ---------------------------------------------------------------- /状态 and /思考


async def test_status_shows_the_cache_hit_rate_the_backend_and_the_suggested_quiet_window(
    world: World,
) -> None:
    world.replies.append("好呀")
    await world.chat("周末去看电影吧", 1)
    await run_to_idle(world.engine, world.clock)
    report = await world.command("/状态", count=2, answers=1)
    assert "缓存命中率 75%" in report  # 300 of 400 prompt tokens came from the cache (R-LLM-010)
    assert "后端：deepseek" in report and "思考模式：关；显示思考：关" in report
    window = await world.engine.quiet_window()
    assert (
        texts.STATUS_QUIET_WINDOW.format(
            configured="15", suggested=f"{window.suggested_s:.0f}", adaptive="关"
        )
        in report
    )
    assert window.suggested_s >= 15
    assert "风格模型：还没有登记" in report and "重训提醒：暂无" in report
    assert "平台窗口" in report and "预算级别" in report


async def test_think_on_switches_the_next_reply_to_thinking_and_show_thinking_adds_the_thoughts(
    world: World,
) -> None:
    await world.command("/思考 开", count=1, answers=1)
    await world.command("/显示思考 开", count=2, answers=2)
    assert world.services.runtime.get(THINKING_CHAT) == "on"
    assert world.services.runtime.get(SHOW_THINKING) is True  # the key the engine reads
    world.replies.append("好呀")
    world.reasoning.append("她问我周末的安排，我想去")
    await world.chat("周末去看电影吧", 3)
    await run_to_idle(world.engine, world.clock)
    assert world.requests()[0]["thinking"] == {"type": "enabled"}
    assert world.bot_lines[-2:] == ["bot: 好呀", f"bot: {PREFIX}思考：她问我周末的安排，我想去"]
    thoughts = world.rows()[-1]
    assert thoughts.is_command and thoughts.text.startswith(f"{PREFIX}思考：")
    await world.command("/显示思考 关", count=4, answers=3)
    await world.command("/思考 关", count=5, answers=4)
    world.replies.append("好的")
    world.reasoning.append("不该被看到")
    await world.chat("那就这样", 6)
    await run_to_idle(world.engine, world.clock)
    assert world.requests()[-1]["thinking"] == {"type": "disabled"}
    assert not any("不该被看到" in line for line in world.output.lines)


async def test_a_style_backend_cannot_be_chosen_without_a_registered_model(world: World) -> None:
    refused = await world.command("/后端 hybrid", count=1, answers=1)
    assert "还没有登记风格模型" in refused
    assert world.services.runtime.get(BACKEND_ACTIVE) == "deepseek"
    world.replies.append("好呀")
    await world.chat("在吗", 2)
    await run_to_idle(world.engine, world.clock)
    assert [r.backend for r in world.rows() if r.direction == "out" and not r.is_command] == [
        "deepseek"
    ]
    assert world.style_client is not None and world.style_client.prompts == []


# -------------------------------------------------------------------------- /重来


async def test_redo_throws_the_last_reply_away_and_writes_it_again(world: World) -> None:
    world.replies.extend(["第一版", "第二版"])
    await world.chat("周末去看电影吧", 1)
    await run_to_idle(world.engine, world.clock)
    assert world.bot_lines == ["bot: 第一版"]
    asked = world.clock.now_utc()
    reply = await world.command("/重来", count=2, answers=1)
    assert reply == PREFIX + texts.REDO_DONE  # said at once ...
    assert world.clock.now_utc() == asked
    await run_to_idle(world.engine, world.clock)
    assert world.bot_lines == ["bot: 第一版", f"bot: {PREFIX}{texts.REDO_DONE}", "bot: 第二版"]
    rows = world.rows()
    first = next(r for r in rows if r.text == "第一版")
    again = next(r for r in rows if r.text == "第二版")
    assert first.rejected_at is not None and again.rejected_at is None
    assert world.feedback() == [("redo", first.reply_id)]
    second_request = world.requests()[1]["messages"]
    assert "周末去看电影吧" in second_request[-1]["content"]
    assert "第一版" not in json.dumps(second_request, ensure_ascii=False)  # not in the prompt
    await world.clock.advance(31 * 60)
    # her own delay can be longer than the quiet minutes: then his message is handed over first
    # and her reply in a batch of its own - either way, the same two messages in the same order
    await wait_until(lambda: len(handed_over(world)) >= 2)
    assert handed_over(world) == ["周末去看电影吧", "第二版"]  # the thrown-away reply is not there


async def test_a_second_redo_before_the_new_reply_is_written_finds_nothing_to_redo(
    world: World,
) -> None:
    world.replies.extend(["第一版", "第二版"])
    await world.chat("在吗", 1)
    await run_to_idle(world.engine, world.clock)
    await world.command("/重来", count=2, answers=1)
    again = await world.command("/重来", count=3, answers=2)
    assert again == PREFIX + texts.REDO_NOTHING
    await run_to_idle(world.engine, world.clock)
    assert [r.text for r in world.rows() if not r.is_command and r.direction == "out"] == [
        "第一版",
        "第二版",
    ]
    assert len(world.feedback()) == 1  # one verdict, not two


async def test_redo_before_any_reply_only_says_so(world: World) -> None:
    reply = await world.command("/重来", count=1, answers=1)
    assert reply == PREFIX + texts.REDO_NOTHING
    assert world.route.call_count == 0 and world.feedback() == []


async def test_redo_while_the_reply_is_still_being_sent_drops_the_rest_and_writes_a_new_one(
    world: World,
) -> None:
    world.replies.extend(["一\n二\n三", "新的一条"])
    await world.chat("数数", 1)
    await run_to_idle(world.engine, world.clock, until=bubbles_written(world.engine, 1))
    assert world.engine.snapshot().state == "SENDING"
    reply = await world.command("/重来", count=2, answers=1)
    assert reply == PREFIX + texts.REDO_DONE
    await run_to_idle(world.engine, world.clock)
    assert "bot: 二" not in world.bot_lines and "bot: 三" not in world.bot_lines
    assert world.bot_lines[-1] == "bot: 新的一条"
    rows = [r for r in world.rows() if r.direction == "out" and not r.is_command]
    assert [(r.text, r.rejected_at is not None) for r in rows] == [
        ("一", True),
        ("新的一条", False),
    ]
    assert any(a["step"] == "send_stopped_by_redo" for a in rows[0].actions)
    assert len(world.requests()) == 2 and world.engine.snapshot().state == "IDLE"
    assert world.feedback() == [("redo", rows[0].reply_id)]


async def test_a_reply_given_out_of_the_role_to_a_crisis_is_never_redone(world: World) -> None:
    world.replies.append('{"is_crisis": true, "severity": "high", "reason": "says so"}')
    await world.chat("我真的不想活了", 1)
    await run_to_idle(world.engine, world.clock)
    assert "988" in "\n".join(world.bot_lines)
    refused = await world.command("/重来", count=2, answers=1)
    assert refused == PREFIX + texts.REDO_SAFETY
    assert world.feedback() == []
    rows = [r for r in world.rows() if r.direction == "out" and not r.is_command]
    assert rows and rows[0].backend == "safety" and all(r.rejected_at is None for r in rows)
    assert len(world.requests()) == 1  # nothing was asked of the model again


# --------------------------------------------------------- the style model in the engine


async def test_hybrid_has_deepseek_plan_and_the_style_model_write_the_reply(world: World) -> None:
    register_style_model(world.services)
    chosen = await world.command("/后端 hybrid", count=1, answers=1)
    assert chosen.startswith(PREFIX + texts.BACKEND_SET.format(name="hybrid"))
    assert world.services.runtime.get(BACKEND_ACTIVE) == "hybrid"
    client = world.style_client
    assert client is not None
    client.replies[:] = ["好呀\n我也想去"]
    world.replies.append(PLAN)  # DeepSeek's only job here: the plan
    await world.chat("周末去看电影吧", 2)
    await run_to_idle(world.engine, world.clock)
    assert world.bot_lines[-2:] == ["bot: 好呀", "bot: 我也想去"]
    assert len(world.requests()) == 1 and "reply_plan" not in json.dumps(world.requests()[0])
    (prompt,) = client.prompts
    assert prompt.text.startswith("<|im_start|>system\n") and prompt.text.endswith(ASSISTANT_OPENER)
    assert CARD_MARKER in prompt.text and "周末去看电影吧" in prompt.text
    assert "接住对方的话" in prompt.text  # the plan went into the style prompt
    out = [r for r in world.rows() if r.direction == "out" and not r.is_command]
    assert out[0].backend == "hybrid" and out[0].plan is not None
    report = await world.command("/状态", count=3, answers=2)
    assert "后端：hybrid" in report and "运行正常" in report


async def test_style_backend_writes_the_reply_without_asking_deepseek(world: World) -> None:
    register_style_model(world.services)
    await world.command("/后端 style", count=1, answers=1)
    assert world.style_client is not None
    world.style_client.replies[:] = ["好呀"]
    await world.chat("在吗", 2)
    await run_to_idle(world.engine, world.clock)
    assert world.bot_lines[-1] == "bot: 好呀" and world.route.call_count == 0
    out = [r for r in world.rows() if r.direction == "out" and not r.is_command]
    assert [(r.backend, r.cost_usd) for r in out] == [("style", 0.0)]


async def test_a_style_model_that_dies_falls_back_to_deepseek_and_comes_back_when_healthy(
    world: World,
) -> None:
    register_style_model(world.services)
    await world.command("/后端 style", count=1, answers=1)
    client = world.style_client
    assert client is not None
    client.replies[:] = [down("connection refused")]
    world.replies.append("我接着说")
    await world.chat("在吗", 2)
    await run_to_idle(world.engine, world.clock)
    # the reply is not lost: the pipeline turned to DeepSeek, and the selector fell back
    assert world.bot_lines[-1] == "bot: 我接着说"
    fallback = world.services.runtime.get(BACKEND_FALLBACK)
    assert (
        fallback is not None and fallback["requested"] == "style" and fallback["reason"] == "error"
    )
    assert ("style_model_down", "warning") in world.alerts()
    report = await world.command("/状态", count=3, answers=2)
    assert "风格模型不可用，眼下由 deepseek 回复" in report
    calls = len(client.prompts)
    world.replies.append("还是我来回")
    await world.chat("还在吗", 4)
    await run_to_idle(world.engine, world.clock)
    assert world.bot_lines[-1] == "bot: 还是我来回"
    assert len(client.prompts) == calls  # the style model is left alone while it is out
    # the server is back: ten healthy minutes (the clock is the manual one) bring the model back
    client.healthy = True
    client.replies[:] = ["好呀"]
    world.clock.tick(31)
    await world.style.selector.tick()  # healthy from now on
    assert world.services.runtime.get(BACKEND_FALLBACK) is not None
    world.clock.tick(10 * 60 + 1)
    await world.chat("回来了吗", 5)
    await run_to_idle(world.engine, world.clock)
    assert world.services.runtime.get(BACKEND_FALLBACK) is None
    assert world.bot_lines[-1] == "bot: 好呀"
    assert ("style_model_down", "info") in world.alerts()
    out = [r for r in world.rows() if r.direction == "out" and not r.is_command]
    assert [r.backend for r in out] == ["deepseek", "deepseek", "style"]
    again = await world.command("/状态", count=6, answers=3)
    assert "后端：style" in again and "眼下由 deepseek" not in again


async def test_a_style_model_whose_outputs_keep_being_refused_falls_back_to_deepseek(
    world: World,
) -> None:
    register_style_model(world.services)
    await world.command("/后端 style", count=1, answers=1)
    client = world.style_client
    assert client is not None
    client.replies[:] = ["<think>想一想</think>"] * 8  # nothing but a thinking block, every time
    for number, text in enumerate(("在吗", "在不在"), start=1):
        world.replies.append(f"备用第{number}条")
        await world.chat(text, number + 1)
        await run_to_idle(world.engine, world.clock)
    fallback = world.services.runtime.get(BACKEND_FALLBACK)
    assert fallback is not None and fallback["reason"] == "violations"
    assert world.bot_lines[-2:] == ["bot: 备用第1条", "bot: 备用第2条"]


async def test_a_style_model_that_is_down_when_asked_for_is_never_chosen(world: World) -> None:
    register_style_model(world.services)
    assert world.style_client is not None
    world.style_client.healthy = False
    refused = await world.command("/后端 hybrid", count=1, answers=1)
    assert refused.startswith(PREFIX + "没有切换：") and "不健康" in refused
    assert world.services.runtime.get(BACKEND_ACTIVE) == "deepseek"


# --------------------------------------------------- the settings' own style model over HTTP


@pytest.fixture
def style_server() -> Iterator[StyleServer]:
    server = StyleServer()
    llama_defaults(server, "好呀")
    server.start()
    yield server
    server.stop()


async def test_the_engine_reaches_the_configured_style_server_and_falls_back_when_it_stops(
    services: Services,
    clock: ManualClock,
    embedder: HashingBackend,
    api: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    style_server: StyleServer,
) -> None:
    services.settings.style_model.endpoint = style_server.url
    register_style_model(services)
    api.route(host="127.0.0.1").pass_through()  # the style server is a real local one
    built = await assemble(services, clock, embedder, api, monkeypatch, style_client=None)
    try:
        chosen = await built.command("/后端 hybrid", count=1, answers=1)
        assert chosen.startswith(PREFIX + texts.BACKEND_SET.format(name="hybrid"))
        assert "/health" in style_server.paths()
        built.replies.append(PLAN)
        await built.chat("在吗", 2)
        await run_to_idle(built.engine, built.clock)
        assert built.bot_lines[-1] == "bot: 好呀"
        sent = style_server.last("/completion").json
        assert sent["prompt"].endswith(ASSISTANT_OPENER) and "<|im_end|>" in sent["stop"]
        style_server.set("GET", "/health", status=503, body={"status": "loading"})
        style_server.set("POST", "/completion", status=503, body={"error": "gone"})
        built.clock.tick(31)  # the next look at the server is due
        built.replies.append("换我来回")
        await built.chat("还在吗", 3)
        await run_to_idle(built.engine, built.clock)
        assert built.bot_lines[-1] == "bot: 换我来回"
        assert built.services.runtime.get(BACKEND_FALLBACK) is not None
        assert ("style_model_down", "warning") in built.alerts()
    finally:
        await close(built)


async def test_the_component_closes_the_style_model_when_it_stops(world: World) -> None:
    assert world.style_client is not None and not world.style_client.closed
    await world.component.stop()
    assert world.style_client.closed
