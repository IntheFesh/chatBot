"""The commands and the learning in the built engine (R-CMD-001, R-LRN-001, R-ARCH-006).

``register_engine`` wires the engine the way ``twin run`` does: the router with the commands of
rounds 09 and 11, the correction service, the report of imports started from the chat and the weekly
look at the correction rules.  The writer is scripted; the tables, the clock and the router are
real.  A command is answered at once, in the system voice, outside the round, and neither it nor
its answer is conversation, memory or learning.
"""

from __future__ import annotations

import random
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import respx
from sqlalchemy import select

from tests.fixtures.synth_export import SynthExport
from tests.support.clock import ManualClock
from tests.support.deepseek import API, error
from tests.support.embedding import HashingBackend
from tests.support.engine_harness import ScriptedChannel, ScriptedWriter, make_draft, run_to_idle
from tests.support.ingest import make_export, run_import
from tests.support.waiting import wait_until
from twin.app import Application
from twin.commands import texts
from twin.commands.import_report import ImportReportComponent
from twin.config.runtime import ENGINE_PAUSED_UNTIL, TARGET_USERNAME
from twin.engine.component import EngineComponent, register_engine
from twin.ingest.runs import latest_run
from twin.learning.component import LearningComponent
from twin.learning.corrections import CorrectionService
from twin.llm.runtime import DEEPSEEK_SECRET
from twin.ops.jobs import JobQueue
from twin.services import Services
from twin.storage.engine_models import BotTurn

PREFIX = texts.PREFIX
START = datetime(2026, 10, 9, 18, 0, tzinfo=UTC)

Running = tuple[Application, EngineComponent, ScriptedChannel, ScriptedWriter]


@pytest.fixture(autouse=True)
def deepseek_is_down() -> Iterator[respx.MockRouter]:
    """No test of this file reaches the network: DeepSeek answers every call with an error."""
    with respx.mock(assert_all_called=False) as router:
        router.post(API).mock(return_value=error(400, "not in this test"))
        yield router


@pytest.fixture
async def running(
    services: Services, clock: ManualClock, embedder: HashingBackend
) -> AsyncIterator[Running]:
    services.secrets.set(DEEPSEEK_SECRET, "synthetic-test-key-0001")
    clock.set_time(START)
    channel = ScriptedChannel(clock)
    writer = ScriptedWriter()
    application = Application()
    component = register_engine(
        application, services, channel, pipeline=writer, rng=random.Random(3)
    )
    await component.start()
    yield application, component, channel, writer
    await component.stop()


def rows(services: Services) -> list[BotTurn]:
    with services.db.session() as session:
        found = list(session.scalars(select(BotTurn).order_by(BotTurn.at, BotTurn.id)))
        session.expunge_all()
    return found


async def say(running: Running, text: str) -> None:
    _application, component, channel, _writer = running
    seen = component.handled
    channel.push(text)
    await wait_until(lambda: component.handled == seen + 1)


async def test_the_application_has_every_part_of_the_round(running: Running) -> None:
    application, component, _channel, _writer = running
    assert isinstance(application.components["import_report"], ImportReportComponent)
    assert isinstance(application.components["learning"], LearningComponent)
    assert isinstance(component.engine.corrections, CorrectionService)
    router = component.router
    assert router is not None
    names = {spec.name for spec in router.registry.specs()}
    assert {
        "时区",
        "暂停",
        "恢复",
        "主动",
        "作息",
        "记住",
        "忘掉",
        "记忆",
        "不像",
        "费用",
        "导入",
    } <= names
    assert "评分" not in names  # round 10 adds its own


@pytest.mark.parametrize(
    "command",
    [
        "/时区 查看",
        "/暂停 30分钟",
        "/恢复",
        "/主动 2-4",
        "/作息 查看",
        "/作息 睡 01:00-08:30",
        "/记住 她喜欢蓝色",
        "/记忆",
        "/忘掉 1",
        "/不像",
        "/费用",
        "/导入 /nowhere/at/all",
        "／时区：　北京",
    ],
)
async def test_a_command_is_answered_at_once_in_system_voice_and_is_never_conversation(
    running: Running, clock: ManualClock, services: Services, command: str
) -> None:
    _application, component, channel, writer = running
    await say(running, command)
    [answer] = channel.out
    assert answer.text is not None and answer.text.startswith(PREFIX)
    assert answer.at == START  # not a second of typing or delay
    assert writer.calls == 0 and component.engine.snapshot().state == "IDLE"
    stored = rows(services)
    assert [(r.direction, r.is_command) for r in stored] == [("in", True), ("out", True)]
    assert stored[1].backend == "command" and stored[1].text == answer.text


async def test_no_command_and_no_answer_reaches_the_memory_the_prompt_or_the_learning(
    running: Running, clock: ManualClock, services: Services
) -> None:
    _application, component, channel, writer = running
    for command in ("/记住 她的暗号是蓝色海豚", "/忘掉 蓝色海豚", "/不像 她会说好呀", "/费用"):
        await say(running, command)
        clock.tick(30)
    writer.add(make_draft("好呀"))
    await say(running, "在吗")
    await run_to_idle(component.engine, clock)
    assert writer.contexts[0].history == ()  # the prompt has no command, no answer
    await clock.advance(31 * 60)
    queue = JobQueue(services.db, clock)
    await wait_until(lambda: bool(queue.list_jobs(job_type="memory_extract")))
    [job] = queue.list_jobs(job_type="memory_extract")
    sent = [turn["text"] for turn in job.payload["turns"]]
    assert sent == ["在吗", "好呀"]  # the extractor reads the chat, not "/记住 ..." or its answers
    stored = [r for r in rows(services) if r.is_command]
    assert len(stored) == 8 and all(r.backend in (None, "command") for r in stored)
    assert services.db is not None and channel.texts[0].startswith(PREFIX)


async def test_a_pause_holds_the_reply_back_until_it_ends_and_resume_ends_it_at_once(
    running: Running, clock: ManualClock, services: Services
) -> None:
    _application, component, channel, writer = running
    await say(running, "/暂停 2小时")
    assert services.runtime.get(ENGINE_PAUSED_UNTIL) == START + timedelta(hours=2)
    writer.add(make_draft("刚看到"))
    await say(running, "在吗")
    await run_to_idle(component.engine, clock)
    replied = next(out for out in channel.out if out.text == "刚看到")
    assert replied.at >= START + timedelta(hours=2)  # nothing before the pause ends
    writer.add(make_draft("在的"))
    await say(running, "/暂停 3小时")
    await say(running, "/恢复")
    assert services.runtime.get(ENGINE_PAUSED_UNTIL) is None
    asked = clock.now_utc()
    await say(running, "还在吗")
    await run_to_idle(component.engine, clock)
    again = next(out for out in channel.out if out.text == "在的")
    assert asked <= again.at < asked + timedelta(minutes=30)  # not held back by the old pause


async def test_the_end_of_an_import_started_from_the_chat_is_said_once(
    running: Running, services: Services, tmp_path: Path
) -> None:
    application, _component, channel, _writer = running
    export: SynthExport = make_export(tmp_path, target_messages=30, other_conversations=0)
    services.runtime.set(TARGET_USERNAME, export.target_username, by="test")
    await say(running, f"/导入 {export.root}")
    assert channel.texts[-1].startswith(PREFIX + "已开始导入")
    report = application.components["import_report"]
    assert isinstance(report, ImportReportComponent)
    assert await report.poll_once() == 0  # still queued
    run = latest_run(services.db)
    assert run is not None
    run_import(services, export, run_id=run.id)
    assert await report.poll_once() == 1
    assert channel.texts[-1].startswith(PREFIX + "导入完成：新增 30 条")
    assert await report.poll_once() == 0  # said once
    last = rows(services)[-1]
    assert last.is_command and last.backend == "command"
    for sentence in export.texts:
        assert sentence not in channel.texts[-1]


async def test_the_zone_command_moves_the_clock_of_the_replies(
    running: Running, services: Services
) -> None:
    _application, _component, channel, _writer = running
    await say(running, "/时区 北京")
    assert "时区已从 America/Chicago 切到 Asia/Shanghai" in (channel.texts[-1])
    await say(running, "/时区 查看")
    assert channel.texts[-1].startswith(PREFIX + "当前时区：Asia/Shanghai")
    await say(running, "/时区 火星")
    assert "不认识这个时区" in channel.texts[-1]


async def test_every_command_works_in_the_terminal_chat_without_a_schedule_component(
    services: Services, embedder: HashingBackend
) -> None:
    """``twin chat --local`` has no schedule component: the commands use the schedule kit."""
    import asyncio

    from tests.support.console import RecordingOutput, ScriptedInput
    from twin.channel.chat import run_local_chat

    services.secrets.set(DEEPSEEK_SECRET, "synthetic-test-key-0001")
    lines = [
        "/帮助",
        "／时区：北京",
        "/主动 2-4",
        "/作息 睡 01:00-08:30",
        "/作息 查看",
        "/记住 她喜欢蓝色",
        "/记忆",
        "/费用",
        "/暂停 1小时",
        "/恢复",
    ]
    inp, out = ScriptedInput(), RecordingOutput()
    chat = asyncio.ensure_future(
        run_local_chat(services, input=inp, output=out, signals=False, drain=True)
    )

    def system_lines() -> list[str]:
        said = [line.removeprefix("bot: ") for line in out.lines if line.startswith("bot: ")]
        return [line for line in said if line.startswith(PREFIX)]

    for number, line in enumerate(lines, start=1):  # one at a time: each opens the window afresh
        inp.feed(line)
        await wait_until(lambda number=number: len(system_lines()) >= number)
    inp.close()
    await asyncio.wait_for(chat, timeout=60)
    system = system_lines()
    assert len(system) == len(lines)
    text = "\n".join(out.lines)  # a long answer shows its first line with the label
    assert "/时区 <IANA 名称>|查看" in text and "/作息 睡 <HH:MM>-<HH:MM>" in text
    assert "时区已从 America/Chicago 切到 Asia/Shanghai" in text
    assert "主动消息每天 2-4 条，已设好" in text
    assert "睡眠时间已设为每天 01:00–08:30" in text and "1. 睡眠 01:00–08:30" in text
    assert "记住了（第 1 条）：她喜欢蓝色" in text and "1. 她喜欢蓝色" in text
    assert "合计 $0.0000" in text and "已恢复" in text
