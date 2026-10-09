"""The whole command table at once: help, tolerance, usage and the system voice (R-CMD-001..003).

Every command of R-CMD-002 is in the table of the application's router (``/评分`` belongs to
round 10, which registers it itself; a stub registered here stands in for it).  The tests walk the
table, so a command added later is covered by the same rules without a new line here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from tests.support.clock import ManualClock
from tests.support.commands_world import PREFIX, CommandWorld, open_world
from tests.support.embedding import HashingBackend
from twin.commands import texts
from twin.commands.parse import is_command_text
from twin.commands.registry import CommandCall, CommandSpec
from twin.engine.command_port import CommandContext
from twin.services import Services

START = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
SPEC_TABLE = [
    "帮助",
    "状态",
    "思考",
    "显示思考",
    "时区",
    "暂停",
    "恢复",
    "主动",
    "作息",
    "后端",
    "重来",
    "不像",
    "记住",
    "忘掉",
    "记忆",
    "评分",
    "费用",
    "导入",
]
NEED_ARGUMENTS = (
    "时区",
    "暂停",
    "主动",
    "作息",
    "记住",
    "忘掉",
    "导入",
    "思考",
    "显示思考",
    "后端",
)


async def rating_stub(call: CommandCall) -> str:
    return f"评分 {call.args}"


@pytest.fixture
async def world(
    services: Services, clock: ManualClock, embedder: HashingBackend
) -> AsyncIterator[CommandWorld]:
    async with open_world(services, clock, start=START) as built:
        built.router.register(
            CommandSpec(
                name="评分",
                group="学习与评分",
                summary="对最近一周主动消息与整体体验打分",
                syntax="/评分 <1-5> [备注]",
                example="/评分 4 挺像的",
                handler=rating_stub,
            )
        )
        yield built


async def test_every_command_of_the_spec_table_is_registered(world: CommandWorld) -> None:
    registered = {spec.name for spec in world.router.registry.specs()}
    assert registered == set(SPEC_TABLE)
    assert len(world.router.registry.specs()) == len(SPEC_TABLE)


async def test_the_help_lists_every_command_in_its_group_with_one_line_each(
    world: CommandWorld,
) -> None:
    reply = await world.reply("/帮助")
    lines = reply.split("\n")
    for spec in world.router.registry.specs():
        assert f"{spec.syntax} — {spec.summary}" in lines, spec.name
    groups = [line for line in lines if line.startswith("【")]
    assert groups == [
        "【基本】",
        "【对话与生成】",
        "【作息与主动】",
        "【记忆】",
        "【学习与评分】",
        "【费用与导入】",
    ]
    assert len([line for line in lines if " — " in line]) == len(SPEC_TABLE)


@pytest.mark.parametrize("name", SPEC_TABLE)
async def test_the_help_of_one_command_gives_syntax_purpose_and_example(
    world: CommandWorld, name: str
) -> None:
    spec = world.router.registry.find(name)
    assert spec is not None
    for spelling in (f"/帮助 {name}", f"/帮助 /{name}", f"／帮助：{name}", f"/help {name}"):
        reply = await world.reply(spelling) if "help" not in spelling else None
        if reply is None:
            continue  # the English name of the help is /help, which the terminal takes itself
        assert reply.split("\n")[:3] == [
            spec.syntax,
            f"作用：{spec.summary}",
            f"例如：{spec.example}",
        ]


@pytest.mark.parametrize("name", SPEC_TABLE)
async def test_the_example_of_every_command_is_recognised_and_answered_in_system_voice(
    world: CommandWorld, name: str
) -> None:
    spec = world.router.registry.find(name)
    assert spec is not None and is_command_text(spec.example)
    outcome = await world.router.handle(spec.example, CommandContext(START, "in-1"))
    assert outcome is not None and outcome.reply.startswith(PREFIX)
    assert not outcome.reply.startswith(PREFIX + texts.UNKNOWN.format(name=name))
    assert "Traceback" not in outcome.reply and texts.FAILED not in outcome.reply


@pytest.mark.parametrize("name", [n for n in SPEC_TABLE if n not in {"导入", "记住", "忘掉"}])
async def test_every_command_is_understood_with_a_full_width_slash_colon_and_spaces(
    world: CommandWorld, name: str
) -> None:
    spec = world.router.registry.find(name)
    assert spec is not None
    words = spec.example.split(" ", 1)
    arguments = words[1] if len(words) > 1 else ""
    plain = await world.reply(spec.example)
    tolerant = await world.reply(f"　 ／{name}：　{arguments}  ")
    assert not tolerant.startswith("没有这个指令")
    if name in {"帮助", "状态", "费用", "记忆", "评分"}:
        assert tolerant == plain  # a read: the same answer however it is written


@pytest.mark.parametrize("name", NEED_ARGUMENTS)
async def test_a_command_that_needs_an_argument_answers_the_usage_when_it_gets_none(
    world: CommandWorld, name: str
) -> None:
    spec = world.router.registry.find(name)
    assert spec is not None
    reply = await world.reply(f"/{name}")
    assert reply.startswith(("参数不对：", "用法：")), reply
    assert f"用法：{spec.syntax}" in reply and f"例如：{spec.example}" in reply


async def test_a_command_never_puts_a_stack_trace_into_the_chat(world: CommandWorld) -> None:
    async def broken(call: CommandCall) -> str:
        raise RuntimeError("secret detail that must not be shown")

    world.router.register(
        CommandSpec("坏了", "基本", "总是出错", "/坏了", "/坏了", broken),
    )
    reply = (await world.say("/坏了")).reply
    assert reply == PREFIX + texts.FAILED and "secret" not in reply


async def test_the_aliases_are_the_english_names_of_the_commands(world: CommandWorld) -> None:
    pairs = {
        "timezone": "时区",
        "tz": "时区",
        "pause": "暂停",
        "resume": "恢复",
        "proactive": "主动",
        "routine": "作息",
        "remember": "记住",
        "forget": "忘掉",
        "memory": "记忆",
        "notlike": "不像",
        "cost": "费用",
        "import": "导入",
    }
    for alias, name in pairs.items():
        found = world.router.registry.find(alias)
        assert found is not None and found.name == name, alias


async def test_chat_that_starts_with_a_slash_but_names_no_command_is_still_chat(
    world: CommandWorld,
) -> None:
    for text in ("/::)", "/:8-)", "/home/user/x", "你好 /状态", "a/时区 北京"):
        assert await world.router.handle(text, CommandContext(START, "in-1")) is None


async def test_no_command_of_the_table_asks_for_the_bot_to_be_always_at_hand(
    world: CommandWorld,
) -> None:
    names = " ".join(spec.name + spec.summary for spec in world.router.registry.specs())
    for forbidden in ("随叫随到", "随时待命", "一直在线"):
        assert forbidden not in names
