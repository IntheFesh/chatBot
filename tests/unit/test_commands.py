"""The commands of round 09 and the router that serves them (R-CMD-001, R-CMD-002, R-CMD-003).

Parsing is tested as a pure function; the router and the commands run on a real database
(``services``), the manual clock and the scripted style model of ``tests/support``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest

from tests.support.clock import ManualClock
from tests.support.embedding import HashingBackend
from tests.support.memory import add_event, add_fact, make_memory
from tests.support.style_models import ScriptedStyleClient, register_model
from twin.commands import texts
from twin.commands.parse import fold, is_command_text, parse_command
from twin.commands.registry import (
    CommandCall,
    CommandRegistry,
    CommandSpec,
    UsageError,
)
from twin.commands.router import CommandRouter, with_prefix
from twin.config.runtime import BACKEND_ACTIVE, BACKEND_FALLBACK, SHOW_THINKING, THINKING_CHAT
from twin.engine.backend_select import BackendSelector
from twin.engine.command_port import CommandContext, CommandOutcome, CommandPort
from twin.engine.feedback import FeedbackStore
from twin.engine.style_models import StyleModels
from twin.engine.turns import BotTurnStore, OutboundBubble, ReplyMeta
from twin.llm.ledger import LedgerRecord
from twin.llm.runtime import LlmRuntime, build_llm_runtime
from twin.llm.types import CostBreakdown, Usage
from twin.memory.manage import MemoryManager
from twin.memory.store import NewEvent
from twin.services import Services
from twin.storage.training_models import ModelRegistryEntry

PREFIX = "⚙️ "
COMMANDS = ("帮助", "状态", "思考", "显示思考", "后端", "重来")


# ---------------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    ("text", "name", "args"),
    [
        ("/思考 开", "思考", "开"),
        ("／思考 开", "思考", "开"),
        ("/思考:开", "思考", "开"),
        ("/思考：开", "思考", "开"),
        ("/思考 ： 开", "思考", "开"),
        ("  /思考　开  ", "思考", "开"),
        ("/思考   自动  ", "思考", "自动"),
        ("/Status", "status", ""),
        ("／ＳＴＡＴＵＳ", "status", ""),
        ("/后端 DeepSeek", "后端", "DeepSeek"),
        ("/帮助", "帮助", ""),
        ("/记住 她的生日是 3月3日：别忘了", "记住", "她的生日是 3月3日：别忘了"),
    ],
)
def test_a_command_is_recognised_with_full_width_marks_colons_and_spaces(
    text: str, name: str, args: str
) -> None:
    parsed = parse_command(text)
    assert parsed is not None and (parsed.name, parsed.args) == (name, args)
    assert is_command_text(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "你好",
        "/",
        "//",
        "/ 思考",
        "/::)",
        "/:8-)",
        "/home/user/x",
        "/s/foo/bar/",
        "a /思考",
    ],
)
def test_chat_is_not_a_command(text: str) -> None:
    assert parse_command(text) is None and not is_command_text(text)


def test_names_and_choices_are_compared_without_case_and_width() -> None:
    assert fold(" ＤｅｅｐＳｅｅｋ ") == "deepseek" and fold("ＯＮ") == "on"


def test_a_slash_command_that_is_not_in_the_table_is_still_a_command() -> None:
    parsed = parse_command("/思考开")  # no space between name and argument: one unknown name
    assert parsed is not None and parsed.name == "思考开"


# --------------------------------------------------------------------------- registry


async def nothing(call: CommandCall) -> str:
    return "好"


def spec(name: str = "测试", **changes: object) -> CommandSpec:
    values: dict[str, object] = {
        "name": name,
        "group": "基本",
        "summary": "测试用",
        "syntax": f"/{name}",
        "example": f"/{name}",
        "handler": nothing,
    }
    values.update(changes)
    return CommandSpec(**values)  # type: ignore[arg-type]


def test_the_registry_refuses_a_name_that_is_taken_and_groups_in_a_fixed_order() -> None:
    registry = CommandRegistry()
    registry.register(spec("甲", group="记忆", aliases=("a",)))
    registry.register(spec("乙", group="基本"))
    registry.register(spec("丙", group="别的组"))
    registry.register(spec("丁", group="基本"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec("A"))  # an alias, folded
    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec("甲"))
    assert [name for name, _ in registry.groups()] == ["基本", "记忆", "别的组"]
    assert [s.name for s in registry.specs()] == ["甲", "乙", "丙", "丁"]
    found = registry.find("a")
    assert found is not None and found.name == "甲" and registry.find("无") is None


def test_a_choice_command_accepts_its_spellings_and_nothing_else() -> None:
    chooser = spec(choices={"on": ("开", "打开", "ON"), "off": ("关",)})
    assert [chooser.choose(text) for text in ("开", " 打开 ", "on", "Ｏｎ", "关", "off")] == [
        "on",
        "on",
        "on",
        "on",
        "off",
        "off",
    ]
    for bad in ("", "也许", "开关"):
        with pytest.raises(UsageError):
            chooser.choose(bad)
    with pytest.raises(UsageError, match="no fixed choices"):
        spec().choose("开")


# ------------------------------------------------------------------------------ router


@pytest.fixture
async def llm(services: Services) -> AsyncIterator[LlmRuntime]:
    built = build_llm_runtime(services)
    yield built
    await built.client.aclose()


class World:
    """The router with everything it works with, for the tests below."""

    def __init__(self, services: Services, llm: LlmRuntime, clock: ManualClock) -> None:
        services.runtime.initialize()
        self.services, self.llm, self.clock = services, llm, clock
        self.client = ScriptedStyleClient()
        self.selector = BackendSelector(
            runtime=services.runtime,
            models=StyleModels(services.db),
            client=self.client,
            config=services.settings.backend,
            clock=clock,
            alerts=services.alerts,
            limits=llm.budget.limits,
        )
        self.turns = BotTurnStore(services.db, clock)
        self.feedback = FeedbackStore(services.db, clock)
        self.memory = make_memory(services)
        self.router = CommandRouter.from_services(
            services,
            llm,
            selector=self.selector,
            turns=self.turns,
            feedback=self.feedback,
            memory=self.memory,
        )

    async def say(self, text: str) -> CommandOutcome:
        outcome = await self.router.handle(text, CommandContext(self.clock.now_utc(), "in-1"))
        assert outcome is not None, f"{text!r} was not taken for a command"
        return outcome

    def setting(self, spec_: object) -> object:
        return self.services.runtime.get(spec_)  # type: ignore[call-overload]


@pytest.fixture
def world(
    services: Services, llm: LlmRuntime, clock: ManualClock, embedder: HashingBackend
) -> World:
    return World(services, llm, clock)


async def test_chat_goes_through_and_every_command_answers_in_the_voice_of_the_system(
    world: World,
) -> None:
    assert await world.router.handle("你好呀", CommandContext(world.clock.now_utc(), "i")) is None
    port: CommandPort = world.router  # the engine's view of it
    assert isinstance(port, CommandRouter)
    for text in ("/帮助", "/状态", "/思考 自动", "/显示思考 关", "/后端 deepseek", "/xyz"):
        outcome = await world.say(text)
        assert outcome.reply.startswith(PREFIX) and not outcome.redo
    assert with_prefix("已有") == PREFIX + "已有" and with_prefix(PREFIX + "x") == PREFIX + "x"
    assert texts.PREFIX == PREFIX == "⚙️ "


async def test_a_command_of_a_later_round_is_unknown_until_it_registers_and_then_it_is_listed(
    world: World,
) -> None:
    before = await world.say("/暂停 2小时")
    assert "没有这个指令：/暂停" in before.reply and "/暂停" not in before.reply.split("\n")[1]
    called: list[str] = []

    async def pause(call: CommandCall) -> str:
        called.append(call.args)
        return f"暂停到{call.args}"

    world.router.register(
        CommandSpec(
            name="暂停",
            group="作息与主动",
            summary="暂停回复与主动",
            syntax="/暂停 <时长>",
            example="/暂停 2小时",
            handler=pause,
            aliases=("pause",),
        )
    )
    outcome = await world.say("／暂停：2小时")
    assert outcome.reply == PREFIX + "暂停到2小时" and called == ["2小时"]
    assert (await world.say("/PAUSE 1小时")).reply == PREFIX + "暂停到1小时"
    help_text = (await world.say("/帮助")).reply
    assert "【作息与主动】\n/暂停 <时长> — 暂停回复与主动" in help_text


async def test_an_unknown_command_gets_the_help_summary_and_is_not_chat(world: World) -> None:
    for text in ("/不像 她不会这么说", "/记住 她喜欢猫", "/zzz", "/思考开"):
        outcome = await world.say(text)
        assert outcome.reply.startswith(PREFIX + "没有这个指令：/")
        assert "可用的指令：/帮助 /状态 /思考 /显示思考 /后端 /重来" in outcome.reply
        assert "发 /帮助 看每个指令的用法和例子。" in outcome.reply and not outcome.redo


async def test_help_lists_the_commands_that_exist_by_group_and_only_those(world: World) -> None:
    reply = (await world.say("/帮助")).reply
    lines = reply.split("\n")
    assert lines[0] == PREFIX + texts.HELP_HEADER
    assert "【基本】" in lines and "【对话与生成】" in lines
    assert lines.index("【基本】") < lines.index("【对话与生成】")
    for name in COMMANDS:
        assert f"/{name}" in reply
    for later in ("/不像", "/记住", "/忘掉", "/记忆", "/评分", "/时区", "/暂停", "/主动", "/作息"):
        assert later not in reply  # not implemented yet, so not offered
    assert "/思考 开|关|自动 — " in reply and "/后端 deepseek|style|hybrid — " in reply


async def test_help_for_one_command_gives_syntax_purpose_example_and_other_names(
    world: World,
) -> None:
    reply = (await world.say("/帮助 /思考")).reply
    assert "/思考 开|关|自动\n作用：" in reply and "例如：/思考 自动" in reply
    assert "也可以写成：/think、/thinking" in reply
    assert (await world.say("／help：重来")).reply.startswith(PREFIX + "/重来\n作用：")
    assert "没有这个指令：/不存在" in (await world.say("/帮助 不存在")).reply


async def test_thinking_mode_is_set_in_every_spelling_and_a_wrong_one_gets_the_usage(
    world: World,
) -> None:
    for text, mode in (
        ("/思考 开", "on"),
        ("/思考:关", "off"),
        ("／思考　自动", "auto"),
        ("/Think ON", "on"),
    ):
        outcome = await world.say(text)
        assert world.setting(THINKING_CHAT) == mode and "思考模式已设为" in outcome.reply
    assert "自动 = 你提问" in (await world.say("/思考 自动")).reply
    for bad in ("/思考", "/思考 也许", "/思考 开 关"):
        outcome = await world.say(bad)
        assert outcome.reply.startswith(PREFIX + "参数不对：")
        assert "用法：/思考 开|关|自动\n例如：/思考 自动" in outcome.reply
    assert world.setting(THINKING_CHAT) == "auto"  # the wrong ones changed nothing
    history = world.services.runtime.history(THINKING_CHAT)
    assert history and history[-1].by == "command"


async def test_thinking_that_the_budget_has_switched_off_is_said_so(world: World) -> None:
    world.services.settings.budget.daily_usd = 0.01
    world.llm.ledger.record(
        LedgerRecord(
            "deepseek",
            "deepseek-flash",
            "reply",
            Usage(prompt_tokens=1, completion_tokens=1, cache_miss_tokens=1),
            CostBreakdown(0.05, 0.0, 0.0, True, 1.0),
            False,
            1,
            world.clock.now_utc(),
        )
    )
    world.llm.budget.status(force=True)
    assert texts.THINK_BUDGET in (await world.say("/思考 开")).reply
    assert texts.THINK_BUDGET not in (await world.say("/思考 关")).reply


async def test_showing_the_thinking_is_a_switch(world: World) -> None:
    assert world.setting(SHOW_THINKING) is False
    assert (await world.say("/显示思考 开")).reply == PREFIX + texts.SHOW_SET_ON
    assert world.setting(SHOW_THINKING) is True
    assert (await world.say("/显示思考 关")).reply == PREFIX + texts.SHOW_SET_OFF
    assert world.setting(SHOW_THINKING) is False
    assert "用法：/显示思考 开|关" in (await world.say("/显示思考 自动")).reply


async def test_the_backend_may_always_go_back_to_deepseek_and_a_fallback_ends_with_it(
    world: World,
) -> None:
    register_model(world.services)
    world.services.runtime.set(BACKEND_ACTIVE, "style", by="command")
    world.client.healthy = False
    await world.selector.choose()
    assert world.setting(BACKEND_FALLBACK) is not None
    reply = (await world.say("/后端 deepseek")).reply
    assert reply == PREFIX + texts.BACKEND_SET.format(name="deepseek")
    assert world.setting(BACKEND_ACTIVE) == "deepseek" and world.setting(BACKEND_FALLBACK) is None


async def test_the_style_backends_are_refused_with_the_reason_until_a_model_is_fit(
    world: World,
) -> None:
    async def refused(text: str) -> str:
        outcome = await world.say(text)
        assert world.setting(BACKEND_ACTIVE) == "deepseek"  # nothing changed
        assert outcome.reply.startswith(PREFIX + "没有切换：")
        return outcome.reply

    assert "还没有登记风格模型" in await refused("/后端 style")
    register_model(world.services, run_id="r1", active=False, gate_passed=None)
    assert "还没有启用" in await refused("/后端 hybrid")
    register_model(world.services, run_id="r2", gate_passed=False)
    gate = await refused("/后端 style")
    assert "没有通过上线门槛" in gate and "twin model activate --force" in gate
    assert "r2 Q5_K_M" in gate
    with world.services.db.transaction(bump_state=False) as session:
        row = session.get(ModelRegistryEntry, "r2-Q5_K_M")
        assert row is not None
        row.gate_passed = True
    world.client.healthy, world.client.detail = False, "the server is down"
    assert "连不上或不健康（the server is down）" in await refused("/后端 style")
    world.client.healthy = True
    outcome = await world.say("/后端 Hybrid")
    assert (
        outcome.reply == PREFIX + texts.BACKEND_SET.format(name="hybrid") + texts.BACKEND_STYLE_HINT
    )
    assert world.setting(BACKEND_ACTIVE) == "hybrid"
    assert (await world.say("/后端 style")).reply.startswith(PREFIX + "生成后端已切到「style」")
    assert "用法：/后端 deepseek|style|hybrid" in (await world.say("/后端 gpt")).reply


async def test_a_command_that_fails_says_so_without_a_word_of_the_chat_and_is_logged_without_it(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    async def broken(call: CommandCall) -> str:
        raise RuntimeError("秘密的聊天内容 " + call.args)

    world.router.register(spec("坏", handler=broken))
    with caplog.at_level(logging.DEBUG):
        outcome = await world.say("/坏 暗号甲")
    assert outcome.reply == PREFIX + texts.FAILED
    assert "暗号甲" not in outcome.reply and "秘密" not in outcome.reply
    assert any(record.getMessage() == "command_failed" for record in caplog.records)
    assert "秘密的聊天内容" not in caplog.text and "暗号甲" not in caplog.text


async def test_a_handler_may_return_an_outcome_and_the_prefix_is_added_once(world: World) -> None:
    async def outcome(call: CommandCall) -> CommandOutcome:
        return CommandOutcome("两行\n第二行", redo=True)

    world.router.register(spec("成", handler=outcome))
    result = await world.say("/成")
    assert result == CommandOutcome(PREFIX + "两行\n第二行", True)


# ------------------------------------------------------------------------- /重来


def one_exchange(world: World) -> tuple[str, list[str]]:
    """The user said something and she answered with two bubbles; returns the reply id and rows."""
    now = world.clock.now_utc()
    world.turns.add_inbound(at=now, kind="text", text="在吗")
    rows = world.turns.add_reply(
        [
            OutboundBubble("在呢", now + timedelta(seconds=5)),
            OutboundBubble("怎么啦", now + timedelta(seconds=9)),
        ],
        ReplyMeta("deepseek"),
    )
    assert rows[0].reply_id is not None
    return rows[0].reply_id, [row.id for row in rows]


async def test_redo_throws_the_reply_away_records_the_verdict_and_asks_for_a_new_one(
    world: World,
) -> None:
    world.clock.tick(60)
    reply_id, row_ids = one_exchange(world)
    world.clock.tick(60)
    outcome = await world.say("/重来")
    assert outcome.redo and outcome.reply == PREFIX + texts.REDO_DONE
    rows = world.turns.reply(reply_id)
    assert all(row.rejected_at is not None for row in rows)
    assert world.turns.latest_reply() == [] and not any(r.counts_as_conversation for r in rows)
    (record,) = world.feedback.for_reply(reply_id)
    assert record.type == "redo" and record.bot_turn_id == row_ids[0]
    assert [r.id for r in world.feedback.unprocessed("redo")] == [record.id]


async def test_redo_twice_does_not_reach_back_to_the_reply_before(world: World) -> None:
    first_id, _ = one_exchange(world)
    world.clock.tick(120)
    world.turns.add_inbound(at=world.clock.now_utc(), kind="text", text="还在吗")
    world.clock.tick(5)
    second = world.turns.add_reply(
        [OutboundBubble("在", world.clock.now_utc())], ReplyMeta("deepseek")
    )
    world.clock.tick(5)
    assert (await world.say("/重来")).redo
    again = await world.say("/重来")  # the new reply is not written yet
    assert again.reply == PREFIX + texts.REDO_NOTHING and not again.redo
    assert world.turns.reply(first_id)[0].rejected_at is None  # the older reply is untouched
    assert world.turns.reply(second[0].reply_id or "")[0].rejected_at is not None
    assert len(world.feedback.unprocessed()) == 1


async def test_redo_has_nothing_to_redo_before_the_first_reply_or_after_a_new_message(
    world: World,
) -> None:
    assert (await world.say("/重来")).reply == PREFIX + texts.REDO_NOTHING
    one_exchange(world)
    world.clock.tick(30)
    world.turns.add_inbound(at=world.clock.now_utc(), kind="text", text="对了")  # not answered yet
    assert (await world.say("/重来")).reply == PREFIX + texts.REDO_NOTHING
    assert world.feedback.unprocessed() == []


async def test_redo_never_throws_away_the_answer_given_out_of_the_role_to_a_crisis(
    world: World,
) -> None:
    now = world.clock.now_utc()
    world.turns.add_inbound(at=now, kind="text", text="我真的不想活了")
    rows = world.turns.add_reply(
        [OutboundBubble("我很担心你", now + timedelta(seconds=2))], ReplyMeta("safety")
    )
    world.clock.tick(30)
    outcome = await world.say("/重来")
    assert outcome == CommandOutcome(PREFIX + texts.REDO_SAFETY, False)
    assert world.turns.reply(rows[0].reply_id or "")[0].rejected_at is None
    assert world.feedback.unprocessed() == []  # no verdict is recorded either


async def test_redo_also_takes_back_what_the_reply_made_up(world: World) -> None:
    _, row_ids = one_exchange(world)
    now = world.clock.now_utc()
    invented = add_fact(
        world.memory,
        "她今天去了新开的咖啡店",
        now,
        source="bot_invented",
        evidence={"kind": "bot_turn", "ids": [row_ids[0]]},
    )
    shared = add_fact(
        world.memory,
        "她养了一只猫",
        now,
        source="bot_invented",
        evidence={"kind": "bot_turn", "ids": [row_ids[1], "older-row"]},
    )
    real = add_fact(
        world.memory, "她喜欢吃火锅", now, evidence={"kind": "bot_turn", "ids": row_ids}
    )
    said = add_fact(
        world.memory,
        "对方说他下周出差",
        now,
        source="user_said",
        evidence={"kind": "bot_turn", "ids": row_ids},
    )
    world.memory.store.add_event(
        NewEvent(
            local_date=now.date(),
            timezone="America/Chicago",
            activity="去咖啡店",
            source="improvised",
            fact_id=invented.id,
        )
    )
    world.memory.refresh()
    world.clock.tick(30)
    outcome = await world.say("/重来")
    assert (
        outcome.redo and texts.REDO_UNDONE.format(count=2) in outcome.reply
    )  # a fact, a life line entry
    world.memory.refresh()
    remaining = {fact.text for fact in world.memory.corpus.facts.values()}
    assert remaining == {"她养了一只猫", "她喜欢吃火锅", "对方说他下周出差"}
    assert not [e for e in world.memory.corpus.events.values() if e.active]
    assert shared.id in world.memory.corpus.facts and real.id in world.memory.corpus.facts
    assert said.id in world.memory.corpus.facts


def test_forgetting_what_a_reply_made_up_needs_a_reply_to_name_and_leaves_the_rest(
    services: Services, embedder: HashingBackend
) -> None:
    memory = make_memory(services)
    manager = MemoryManager(memory)
    assert manager.forget_derived_from([]).deleted == ()
    add_fact(
        memory, "一条事实", services.clock.now_utc(), source="bot_invented", evidence={"kind": "x"}
    )
    assert manager.forget_derived_from(["a"]).deleted == ()  # a fact without cited rows stays
    add_event(memory, services.clock.now_utc().date(), "与它无关的事")
    assert len(memory.corpus.facts) == 1
