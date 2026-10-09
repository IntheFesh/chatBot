"""Commands are answered at once, outside the round and the memory (R-CMD-001, R-ENG-011)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.engine_extras import ScriptedCommands
from tests.support.engine_harness import (
    Harness,
    build_harness,
    make_draft,
    run_to_idle,
)
from tests.support.waiting import wait_until
from twin.config.runtime import ENGINE_PAUSED_UNTIL
from twin.engine.command_port import CommandContext, CommandOutcome
from twin.engine.state_store import ConversationSnapshot
from twin.services import Services
from twin.storage.engine_models import BotTurn

HELP = "⚙️ 帮助：/状态 /思考"


def rows(harness: Harness) -> list[BotTurn]:
    with harness.services.db.session() as session:
        found = list(session.scalars(select(BotTurn).order_by(BotTurn.at, BotTurn.id)))
        session.expunge_all()
    return found


@pytest.fixture
async def rig(services: Services, clock: ManualClock) -> AsyncIterator[Harness]:
    commands = ScriptedCommands(帮助=CommandOutcome(HELP), 状态=CommandOutcome("⚙️ 一切正常"))
    harness = build_harness(services, clock, commands=commands)
    await harness.engine.start()
    yield harness
    await harness.engine.stop()


async def test_a_command_is_answered_at_once_without_any_delay(rig: Harness) -> None:
    await rig.message("/帮助")
    assert rig.channel.texts == [HELP]  # the clock has not moved a second
    assert rig.clock.now_utc() == rig.channel.out[0].at
    assert rig.writer.calls == 0 and rig.engine.snapshot().state == "IDLE"
    assert rig.engine.snapshot().pending == ()


async def test_the_command_and_its_answer_are_marked_and_never_conversation(rig: Harness) -> None:
    await rig.message("/帮助")
    stored = rows(rig)
    assert [(r.direction, r.is_command, r.backend) for r in stored] == [
        ("in", True, None),
        ("out", True, "command"),
    ]
    assert stored[1].text == HELP
    rig.writer.add(make_draft("好呀"))
    await rig.message("在吗")
    await run_to_idle(rig.engine, rig.clock)
    context = rig.writer.contexts[0]
    assert context.history == ()  # the command and its answer are not in the prompt
    await rig.clock.advance(31 * 60)
    await wait_until(lambda: len(rig.queued) == 1)
    assert [m.text for m in rig.queued[0]] == ["在吗", "好呀"]  # nor in the memory


async def test_the_router_gets_the_text_and_the_message_it_came_in(rig: Harness) -> None:
    message = rig.channel.push("/状态", at=rig.clock.now_utc())
    await rig.engine.handle_message(message)
    text, context = rig.commands.seen[0]
    assert text == "/状态" and isinstance(context, CommandContext)
    assert context.at == message.at
    stored = rows(rig)[0]
    assert context.inbound_id == stored.id and stored.external_id == message.id


async def test_a_full_width_slash_reaches_the_router_too(rig: Harness) -> None:
    await rig.message("／帮助")
    assert rig.channel.texts == [HELP] and rig.commands.seen[0][0] == "／帮助"


async def test_a_command_works_during_a_pause_and_does_not_touch_the_round(rig: Harness) -> None:
    rig.services.runtime.set(ENGINE_PAUSED_UNTIL, rig.clock.now_utc() + timedelta(hours=1))
    rig.writer.gate = asyncio.Event()
    await rig.message("在吗")
    await run_to_idle(
        rig.engine, rig.clock, until=lambda: rig.engine.snapshot().state == "DECIDING"
    )
    pending = rig.engine.snapshot().pending
    await rig.message("/状态")
    assert rig.channel.texts == ["⚙️ 一切正常"]  # answered although she is paused
    assert rig.engine.snapshot().pending == pending and rig.engine.snapshot().state == "DECIDING"


async def test_a_command_during_generation_does_not_cancel_it(rig: Harness) -> None:
    rig.writer.gate = asyncio.Event()
    await rig.message("在吗")
    await run_to_idle(rig.engine, rig.clock, until=rig.writer.started.is_set)
    await rig.message("/帮助")
    assert rig.channel.texts == [HELP] and rig.writer.cancelled == 0
    rig.writer.gate.set()
    rig.writer.add(make_draft("好呀"))
    await run_to_idle(rig.engine, rig.clock)
    assert rig.channel.texts == [HELP, "好呀"] and rig.writer.calls == 1


async def test_text_that_starts_with_a_slash_but_is_no_command_is_chat(rig: Harness) -> None:
    rig.writer.add(make_draft("哈哈"))
    await rig.message("/摸摸你的头")
    await run_to_idle(rig.engine, rig.clock)
    assert rig.channel.texts == ["哈哈"] and rig.writer.contexts[0].user_text == "/摸摸你的头"
    stored = rows(rig)[0]
    assert stored.is_command is False  # unmarked again once the router said it is none


async def test_without_a_router_a_slash_message_is_chat(
    services: Services, clock: ManualClock
) -> None:
    harness = build_harness(services, clock)
    await harness.engine.start()
    try:
        await harness.message("/帮助")
        await run_to_idle(harness.engine, clock)
        assert harness.writer.contexts[0].user_text == "/帮助"
        assert rows(harness)[0].is_command is False
    finally:
        await harness.engine.stop()


async def test_a_router_that_breaks_is_answered_never_read_as_chat(rig: Harness) -> None:
    rig.commands.error = RuntimeError("the router fell over")
    await rig.message("/帮助")
    assert rig.channel.texts == ["⚙️ 这条指令没有处理成功，请稍后再试。"]
    assert rows(rig)[0].is_command and rig.writer.calls == 0


async def test_only_text_can_be_a_command(rig: Harness) -> None:
    from twin.channel.base import MessageKind

    rig.writer.add(make_draft("看到了"))
    await rig.message("/帮助", kind=MessageKind.VOICE)  # a voice message that transcribes so
    await run_to_idle(rig.engine, rig.clock)
    assert rig.commands.seen == [] and rig.channel.texts == ["看到了"]


# ------------------------------------------------------------------------------ /重来


def reject_latest(harness: Harness) -> Any:
    """What the router does for ``/重来``: throw the last reply away."""

    def before(text: str, context: CommandContext) -> None:
        if text == "/重来":
            latest = harness.store.latest_reply()
            assert latest and latest[0].reply_id
            harness.store.reject_reply(latest[0].reply_id)

    return before


async def test_redo_answers_the_same_messages_again_after_the_command_reply(
    services: Services, clock: ManualClock
) -> None:
    commands = ScriptedCommands(重来=CommandOutcome("⚙️ 好的，重新来一次", redo=True))
    harness = build_harness(services, clock, commands=commands)
    commands.before = reject_latest(harness)
    await harness.engine.start()
    try:
        harness.writer.add(make_draft("第一版", "回复"), make_draft("第二版"))
        await harness.message("周末去看电影吧")
        await run_to_idle(harness.engine, clock)
        assert harness.channel.texts == ["第一版", "回复"]
        await harness.message("/重来")
        assert harness.channel.texts[-1] == "⚙️ 好的，重新来一次"  # at once
        await run_to_idle(harness.engine, clock)
        assert harness.channel.texts[-2:] == ["⚙️ 好的，重新来一次", "第二版"]
        second = harness.writer.contexts[1]
        assert second.user_text == "周末去看电影吧"  # the same message
        assert second.history == ()  # the rejected reply is not in the prompt
        again = next(row for row in rows(harness) if row.text == "第二版")
        assert again.backend == "deepseek" and again.rejected_at is None
        first = next(row for row in rows(harness) if row.text == "第一版")
        assert first.rejected_at is not None
    finally:
        await harness.engine.stop()


async def test_redo_skips_her_first_delay_but_not_her_pace(
    services: Services, clock: ManualClock
) -> None:
    commands = ScriptedCommands(重来=CommandOutcome("⚙️ 重来", redo=True))
    harness = build_harness(services, clock, commands=commands)
    commands.before = reject_latest(harness)
    await harness.engine.start()
    try:
        harness.writer.add(make_draft("一"), make_draft("二", "三"))
        await harness.message("在吗")
        await run_to_idle(harness.engine, clock)
        asked = clock.now_utc()
        await harness.message("/重来")
        await run_to_idle(harness.engine, clock)
        text_times = [
            o.at for o in harness.channel.out if o.kind == "text" and o.text in ("二", "三")
        ]
        assert text_times[0] - asked < timedelta(seconds=10)  # no quiet window, no 20 s latency
        assert text_times[1] > text_times[0]  # but the second bubble still waits for her pause
    finally:
        await harness.engine.stop()


async def test_redo_with_nothing_to_redo_only_says_so(
    services: Services, clock: ManualClock
) -> None:
    commands = ScriptedCommands(重来=CommandOutcome("⚙️ 还没有可以重来的回复", redo=True))
    harness = build_harness(services, clock, commands=commands)
    await harness.engine.start()
    try:
        await harness.message("/重来")
        assert harness.channel.texts == ["⚙️ 还没有可以重来的回复"]
        assert harness.engine.snapshot().state == "IDLE" and harness.writer.calls == 0
    finally:
        await harness.engine.stop()


async def test_redo_while_a_round_is_under_way_is_not_stacked_on_it(
    services: Services, clock: ManualClock
) -> None:
    commands = ScriptedCommands(重来=CommandOutcome("⚙️ 重来", redo=True))
    harness = build_harness(services, clock, commands=commands)
    await harness.engine.start()
    try:
        harness.writer.add(make_draft("一"))
        await harness.message("在吗")
        await run_to_idle(harness.engine, clock)
        await harness.message("再问一句")  # a new round is collecting
        before: ConversationSnapshot = harness.engine.snapshot()
        await harness.message("/重来")
        assert harness.engine.snapshot().pending == before.pending  # nothing was added to it
    finally:
        await harness.engine.stop()
