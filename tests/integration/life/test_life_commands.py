"""Every command of the table, at least once, in a conversation (R-CMD-001 to R-CMD-003).

The user sends the whole table to the running application of ``twin run`` - with the arguments he
would give - and the test checks each answer, what the command changed, and, at the end, what holds
for all of them: the answers come in the system voice and at once, nothing about a command reaches
a prompt, the memory or the examples, and the table of the router has no command that this story
did not use (a command added later fails this test until it is given a line here).

The story is one morning in Chicago: he asks what she is doing at the weekend, throws the answer
away, teaches her how to say it, pauses her, changes her routine, and looks at the cost.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import select

from tests.integration.life.conftest import WorldFactory
from tests.support.life_checks import (
    assert_clean_screen,
    assert_never_in_deep_sleep,
    assert_screen_matches_records,
)
from tests.support.life_world import LifeWorld
from tests.support.proactive_world import opening_curve, proactive_model
from twin.commands import texts
from twin.commands.parse import parse_command
from twin.config.runtime import (
    BACKEND_ACTIVE,
    ENGINE_PAUSED_UNTIL,
    PROACTIVE_DAILY_MAX,
    PROACTIVE_DAILY_MIN,
    PROACTIVE_ENABLED,
    SHOW_THINKING,
    THINKING_CHAT,
)
from twin.storage.engine_models import Feedback
from twin.storage.models import Alert

pytestmark = pytest.mark.integration

PREFIX = texts.PREFIX
SPEC_TABLE = {
    "帮助", "状态", "思考", "显示思考", "时区", "暂停", "恢复", "主动", "作息",
    "后端", "重来", "不像", "记住", "忘掉", "记忆", "评分", "费用", "导入",
}  # fmt: skip


class Desk:
    """The user's keyboard: sends commands, remembers which ones, and hands back the answer."""

    def __init__(self, world: LifeWorld) -> None:
        self.world = world
        self.sent: list[str] = []
        self.used: set[str] = set()

    async def __call__(self, text: str) -> str:
        found = parse_command(text)
        assert found is not None, f"{text!r} is not a command"
        spec = self.world.assembly.engine.router.registry.find(found.name)  # type: ignore[union-attr]
        assert spec is not None, f"{text!r}: no such command"
        self.used.add(spec.name)
        self.sent.append(text)
        before = len(self.world.system_said)
        await self.world.say(text)
        assert len(self.world.system_said) == before + 1, f"{text!r} was not answered"
        answer = self.world.system_said[-1].text
        assert answer.startswith(PREFIX) and texts.FAILED not in answer
        return answer.removeprefix(PREFIX)


async def test_every_command_is_answered_and_does_what_it_says(make_world: WorldFactory) -> None:
    world = await make_world(
        datetime(2026, 10, 9, 16, 0, tzinfo=UTC),  # 11:00 in Chicago
        model=proactive_model(opening_curve(base=0.02)),
    )
    desk = Desk(world)
    book, runtime = world.deepseek.book, world.services.runtime
    router = world.assembly.engine.router
    assert router is not None
    book.queue.extend(["看电影", "逛街"])
    book.reasoning = "她问我周末的安排，我想去看电影"

    # ---- basics: help and status -------------------------------------------------------------
    listing = await desk("/帮助")
    for spec in router.registry.specs():
        assert f"/{spec.name}" in listing  # the table lists every command
    one = await desk("/帮助 思考")
    assert "/思考 开|关|自动" in one and "例如：/思考 自动" in one
    status = await desk("/状态")
    for line in ("时区：America/Chicago", "她此刻：", "后端：deepseek", "主动消息：", "平台窗口："):
        assert line in status
    assert world.deepseek.calls["reply"] == 0 and world.deepseek.calls["proactive"] == 0

    # ---- thinking, and showing it -------------------------------------------------------------
    assert await desk("/思考 开") == texts.THINK_SET.format(mode="开")
    assert (await desk("/思考 自动")).startswith(texts.THINK_SET.format(mode="自动"))
    assert await desk("/思考   关") == texts.THINK_SET.format(mode="关")  # extra white space
    assert runtime.get(THINKING_CHAT) == "off"
    await desk("/思考 开")
    assert await desk("/显示思考 开") == texts.SHOW_SET_ON and runtime.get(SHOW_THINKING) is True
    await world.say("周末干嘛")
    await world.run_until_idle()
    said = world.said[-2:]
    assert said[0].text == "看电影"
    assert said[1].kind == "system" and said[1].text == f"{PREFIX}思考：她问我周末的安排，我想去看电影"
    assert world.deepseek.of_kind("reply")[-1].body["thinking"] == {"type": "enabled"}
    assert await desk("/显示思考 关") == texts.SHOW_SET_OFF
    await desk("/思考 关")

    # ---- the backend ------------------------------------------------------------------------------
    assert await desk("/后端 deepseek") == texts.BACKEND_SET.format(name="deepseek")
    refused = await desk("/后端 style")  # no model is registered: refused, nothing changes
    assert "还没有登记风格模型" in refused and runtime.get(BACKEND_ACTIVE) == "deepseek"

    # ---- redo: the reply is thrown away and written again --------------------------------------
    assert await desk("/重来") == texts.REDO_DONE
    await world.run_until_idle()
    assert world.persona_said[-1].text == "逛街"
    first = [r for r in world.out_rows() if r.text == "看电影"][0]
    assert first.rejected_at is not None  # a negative example, and not a message of hers any more
    with world.services.db.session() as session:
        assert [row.type for row in session.scalars(select(Feedback))] == ["redo"]

    # ---- "not like her": a negative label, and a correction that becomes a pair ------------------
    taught = await desk("/不像 那你早点睡吧")
    assert texts.NOT_LIKE_DONE in taught and "已存为一对对照" in taught
    assert (await desk("/不像")).startswith(texts.NOT_LIKE_DONE)
    book.queue.append("好的")

    # ---- time zone, pause, resume --------------------------------------------------------------------
    assert (await desk("/时区 查看")).startswith("当前时区：America/Chicago")
    assert "America/Chicago" in await desk("/时区：查看")  # the colon of a Chinese keyboard
    paused = await desk("/暂停 2小时")
    assert "暂停到 10-09 周五" in paused and runtime.get(ENGINE_PAUSED_UNTIL) is not None
    spoken = len(world.persona_said)
    await world.say("在吗")
    await world.run_for(minutes=90)
    assert len(world.persona_said) == spoken  # nothing, while she is paused (R-ENG-003)
    assert await desk("/恢复") == texts.RESUME_DONE
    assert runtime.get(ENGINE_PAUSED_UNTIL) is None
    await world.run_until_idle()
    assert world.persona_said[-1].text == "好的" and len(world.persona_said) == spoken + 1
    assert await desk("/恢复") == texts.RESUME_NOT_PAUSED

    # ---- how often she writes first -------------------------------------------------------------------
    assert "每天 2-5 条" in await desk("/主动 2-5")
    assert (runtime.get(PROACTIVE_DAILY_MIN), runtime.get(PROACTIVE_DAILY_MAX)) == (2, 5)
    assert "已关闭" in await desk("/主动 关") and runtime.get(PROACTIVE_ENABLED) is False
    assert "已开启" in await desk("/主动 开") and runtime.get(PROACTIVE_ENABLED) is True
    wrong = await desk("/主动 9-1")  # a wrong argument: the usage, never a trace
    assert "用法：/主动 <最少>-<最多>|开|关" in wrong and "例如：/主动 2-5" in wrong

    # ---- her routine --------------------------------------------------------------------------------------
    assert "第 1 条" in await desk("/作息 睡 01:00-08:30")
    assert "第 2 条" in await desk("/作息 忙 周一 09:00-11:00")
    assert "第 3 条" in await desk("/作息 假期 2026-10-12..2026-10-14")
    routine = await desk("/作息 查看")
    assert "1. 睡眠 01:00–08:30" in routine and "2. 忙碌 周一 09:00–11:00" in routine
    assert "3. 节假日 2026-10-12 至 2026-10-14" in routine
    assert "已删除第 1 条" in await desk("/作息 删除 1")
    assert "睡眠" not in await desk("/作息 查看")

    # ---- memory ------------------------------------------------------------------------------------------------
    assert (await desk("/记住 她下周三有考试")).startswith("记住了（第 1 条）：她下周三有考试")
    assert (await desk("/记忆")).startswith("1. 她下周三有考试")
    assert (await desk("/记忆 考试")).startswith("1. 她下周三有考试")
    assert "第 1 条：她下周三有考试" in await desk("/忘掉 1")
    assert await desk("/记忆") == "还没有记住什么。"

    # ---- rating, cost, import --------------------------------------------------------------------------------
    assert "记下了：4/5" in await desk("/评分 4 晚安发得很自然")
    assert "分数要在 1 到 5 之间" in await desk("/评分 9")
    today = await desk("/费用 今天")
    assert "今天（2026-10-09）的费用" in today and "预算级别：0" in today
    assert "2026-10 的费用" in await desk("/费用 本月")
    missing = await desk("/导入 D:\\聊天记录\\不存在的文件夹")
    assert missing.startswith("找不到这个文件夹")
    assert not [kind for kind, *_ in world.job_rows() if "import" in kind]  # none queued

    # ---- the table's own edges: unknown commands, other spellings ------------------------------------------
    desk.sent.append("/没有这条")
    await world.say("/没有这条")  # not in the table: the summary of what is, never chat
    unknown = world.system_said[-1].text
    assert "没有这个指令：/没有这条" in unknown and "/帮助" in unknown
    assert (await desk("／状态")).startswith("当前状态")  # the full-width slash
    spoken = len(world.persona_said)
    await world.say("/::)")  # not a command: a WeChat emoticon, chat
    await world.run_until_idle()
    assert len(world.persona_said) == spoken + 1

    # ---- the whole table was used --------------------------------------------------------------------------------
    table = {spec.name for spec in router.registry.specs()}
    assert desk.used == table, f"not used: {table - desk.used}"
    assert table == SPEC_TABLE  # and it is the table of the specification (R-CMD-002)

    # ---- what holds for every command ---------------------------------------------------------------------------
    rows = world.rows()
    commands = [row for row in rows if row.is_command]
    assert [r.text for r in commands if r.direction == "in"] == desk.sent
    assert all(r.backend == "command" for r in commands if r.direction == "out")
    chat = [r for r in rows if not r.is_command]  # a command is never taken for chat
    assert not [r for r in chat if r.text.startswith(PREFIX) or parse_command(r.text)]
    for call in world.deepseek.log:  # no command and no system message reaches any model
        if call.kind != "rules":  # (the fixed text of that one names /重来 and /不像)
            assert not [text for text in desk.sent if text in call.text], call.kind
        assert PREFIX not in call.text, call.kind
    await world.run_for(minutes=45)  # the quiet half hour: the conversation goes to the memory
    await world.drain_jobs()
    for call in world.deepseek.of_kind("memory"):
        assert not [text for text in desk.sent if text in call.text and "/记住" in text]
    assert_clean_screen(world)
    assert_screen_matches_records(world)
    assert_never_in_deep_sleep(world)
    assert world.deepseek.unexpected == []
    with world.services.db.session() as session:
        assert not [a for a in session.scalars(select(Alert)) if a.category == "engine_error"]
    assert date(2026, 10, 9) == world.today() and world.now > datetime(2026, 10, 9, tzinfo=UTC)
    assert timedelta(0) < world.now - datetime(2026, 10, 9, 16, 0, tzinfo=UTC)
