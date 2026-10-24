"""Before her profile, routine and persona card exist the bot does not talk (R18-B8, D-870, D-871).

The engine stores what the user writes, says once (at most every six hours for the same missing
things, the book kept in the database) what is missing and what stands in the way, and holds the
round; a crisis is still answered at once and commands still work.  Once ready the messages that
piled up are one round, timed afresh ("just seen") and answered by her plan - when she is asleep,
after she wakes.  The harness leaves the ready data out (``ready=False``): the production check
(:class:`~twin.engine.readiness.ReadinessGate`) runs on an empty database, nothing is switched off.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.engine_harness import (
    Harness,
    ScriptedCrisis,
    build_harness,
    decision_of,
    make_draft,
    run_to_idle,
)
from tests.support.readiness import install_profile, install_ready, install_routine
from tests.support.routine import Rig, student_model
from tests.support.waiting import wait_until
from twin.commands.parse import parse_command
from twin.engine.command_port import CommandContext, CommandOutcome
from twin.engine.machine import READY_RECHECK_S, SYSTEM_PREFIX
from twin.engine.readiness import NOTICE_KEY, NoticeBook, Readiness
from twin.services import Services
from twin.storage.engine_models import BotTurn
from twin.storage.settings_store import get_setting

ALL_MISSING = "画像、作息模型、人设卡"


class RecordingCommands:
    """A command router that records what it is asked to run."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def recognises(self, text: str) -> bool:
        return parse_command(text) is not None

    async def handle(self, text: str, context: CommandContext) -> CommandOutcome | None:
        self.calls.append(text)
        return CommandOutcome("⚙️ 状态：好的")


@pytest.fixture
async def rig(services: Services, clock: ManualClock) -> AsyncIterator[Harness]:
    harness = build_harness(
        services, clock, ready=False, commands=RecordingCommands(), crisis=ScriptedCrisis("不想活")
    )
    await harness.engine.start()
    yield harness
    await harness.engine.stop()


async def held(harness: Harness) -> None:
    """Run until the round waits for readiness: past the quiet window, the driver waits for its
    long look-again (``READY_RECHECK_S``) and has nothing left to see."""
    engine, clock = harness.engine, harness.clock
    await run_to_idle(
        engine,
        clock,
        until=lambda: (
            engine.waiting_until is not None
            and not engine.pending_wake
            and engine.waiting_until - clock.now_utc() > timedelta(seconds=READY_RECHECK_S / 2)
        ),
    )


async def hold_for(harness: Harness, seconds: float) -> None:
    """Let ``seconds`` of the manual clock go by while the round is held, one look at a time.

    Every look reads the database a few times; hours of looks on a slow disk take real seconds
    (``SLOWDB_MS=30``), so the real bound is generous - the clock's time costs nothing.
    """
    end = harness.clock.now_utc() + timedelta(seconds=seconds)
    await run_to_idle(
        harness.engine,
        harness.clock,
        until=lambda: harness.clock.now_utc() >= end,
        limit_s=20.0 + seconds / READY_RECHECK_S,
    )
    await held(harness)


def notices(harness: Harness) -> list[str]:
    return [text for text in harness.channel.texts if text.startswith(SYSTEM_PREFIX)]


def bot_rows(services: Services) -> list[tuple[str, bool]]:
    """``(direction, is_command)`` of every row of the bot's conversation, oldest first."""
    with services.db.session() as session:
        rows = session.scalars(select(BotTurn).order_by(BotTurn.at, BotTurn.id))
        return [(row.direction, row.is_command) for row in rows]


async def test_r18_b8_a_message_before_the_bot_is_ready_is_stored_and_not_answered(
    rig: Harness, services: Services
) -> None:
    """R18-B8: no reply; one system message that says what is missing and how to move on."""
    await rig.message("在吗")
    await held(rig)
    await hold_for(rig, 3 * 3600)  # hours go by: the round keeps waiting, nothing of hers is said
    snapshot = rig.engine.snapshot()
    assert snapshot.state == "COLLECTING" and len(snapshot.pending) == 1
    assert rig.writer.calls == 0
    (notice,) = rig.channel.texts
    assert notice.startswith(SYSTEM_PREFIX) and f"缺 {ALL_MISSING}" in notice
    assert "`twin import <导出目录>`" in notice  # nothing was imported yet: how to start
    assert "画像与作息：导入完成后自动计算" in notice
    assert "在吗" not in notice
    assert bot_rows(services) == [("in", False), ("out", True)]  # the notice is a command row


async def test_r18_b8_the_same_missing_things_are_told_at_most_every_six_hours(
    rig: Harness,
) -> None:
    """R18-B8: a second message within six hours gets no second notice, and the hours that pass
    bring none by themselves; a message six hours or more after the notice gets one again."""
    await rig.message("在吗")
    await held(rig)
    await hold_for(rig, 2 * 3600)
    await rig.message("怎么不理我")
    await held(rig)
    assert len(notices(rig)) == 1 and len(rig.engine.snapshot().pending) == 2
    await hold_for(rig, 5 * 3600)  # more than six hours since the notice, no new message
    assert len(notices(rig)) == 1
    await rig.message("还在吗")
    await held(rig)
    assert len(notices(rig)) == 2 and rig.writer.calls == 0
    assert len(rig.engine.snapshot().pending) == 3


async def test_r18_b8_the_book_of_notices_survives_a_restart(
    services: Services, clock: ManualClock, rig: Harness
) -> None:
    """R18-B8: the time of the notice is in the database: a new process does not say it again."""
    await rig.message("在吗")
    await held(rig)
    assert len(notices(rig)) == 1
    stored = get_setting_value(services)
    assert set(stored) == {"profile,routine,persona"}
    await rig.engine.stop()
    again = build_harness(services, clock, ready=False)
    await again.engine.start()
    try:
        await hold_for(again, 60)  # its quiet window starts afresh, then it holds again
        assert again.channel.texts == [] and again.writer.calls == 0
        assert len(again.engine.snapshot().pending) == 1
    finally:
        await again.engine.stop()


def get_setting_value(services: Services) -> dict[str, str]:
    with services.db.session() as session:
        value = get_setting(session, NOTICE_KEY, {})
    assert isinstance(value, dict)
    return value


async def test_r18_b8_a_different_set_of_missing_things_is_told_when_it_changes(
    rig: Harness, services: Services
) -> None:
    """R18-B8: the profile and routine are computed (``state_version`` moves): the card is the
    one thing left, and that is a new kind of notice - with what the card waits for."""
    await rig.message("在吗")
    await held(rig)
    install_profile(services)
    install_routine(services)
    await rig.engine.on_state_change(1, 2)
    await wait_until(lambda: len(notices(rig)) == 2, limit_s=10.0, interval=0.002)
    await held(rig)
    first, second = notices(rig)
    assert f"缺 {ALL_MISSING}" in first
    assert "缺 人设卡。" in second and "画像与作息" not in second
    assert "人设卡：她的消息只有 0 条，不够生成描述" in second
    assert rig.writer.calls == 0


async def test_r18_b8_once_ready_the_backlog_is_one_round_answered_as_just_seen(
    rig: Harness, services: Services
) -> None:
    """R18-B8: the profile, routine and card are written and the engine is told: the messages
    that piled up get one answer, soon after that moment (she has just seen them)."""
    await rig.message("在吗")
    await held(rig)
    await hold_for(rig, 3 * 3600)
    await rig.message("人呢")
    await held(rig)
    install_ready(services)
    ready_at = rig.clock.now_utc()
    await rig.engine.on_state_change(1, 2)
    await run_to_idle(rig.engine, rig.clock)
    assert rig.writer.calls == 1
    assert rig.channel.texts[-1] == "好的" and len(notices(rig)) == 1
    reply = next(o for o in rig.channel.out if o.kind == "text" and o.text == "好的")
    assert reply.at - ready_at < timedelta(seconds=30)  # "just seen", not old news
    context = rig.writer.contexts[-1]
    assert [item.text for item in context.inbound] == ["在吗", "人呢"]  # one round for both


async def test_r18_b8_a_crisis_before_the_bot_is_ready_is_answered_at_once(rig: Harness) -> None:
    """R-SAFE-001 stays first: the crisis screen runs before the hold, the answer goes out."""
    await rig.message("我不想活了")
    await run_to_idle(rig.engine, rig.clock)
    assert rig.writer.calls == 0 and rig.crisis.handled == [["我不想活了"]]
    assert rig.channel.texts == ["我想先停一下。", "你现在还好吗？", "也可以联系 988。"]
    assert rig.engine.snapshot().state == "IDLE"  # the crisis ended the round: no notice


async def test_r18_b8_commands_work_before_the_bot_is_ready(rig: Harness) -> None:
    """D-871: commands need no profile - they are answered at once, no notice, no round."""
    await rig.message("/状态")
    assert rig.channel.texts == ["⚙️ 状态：好的"]
    assert rig.engine.snapshot().pending == ()


async def test_r18_b8_a_step_that_keeps_failing_before_readiness_says_nothing_of_hers(
    services: Services, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-871: the short answer of the last resort (D-550) is hers too: not before readiness."""
    harness = build_harness(services, clock, ready=False, answers=("嗯嗯嗯",))
    await harness.engine.start()
    try:

        def broken(texts: object) -> int:
            raise RuntimeError("always")

        monkeypatch.setattr(harness.crisis, "screen", broken)
        await harness.message("在吗")
        await run_to_idle(harness.engine, harness.clock)
        assert harness.channel.texts == [] and harness.writer.calls == 0
        assert "engine_error" in harness.alerts.categories
    finally:
        await harness.engine.stop()


async def test_r18_b8_ready_while_she_is_in_deep_sleep_waits_for_her_waking(
    services: Services, clock: ManualClock
) -> None:
    """R18-B8 with the real day plan: the card is approved at night; she answers after waking."""
    model = student_model()  # sleeps 23:30-07:30 on workdays (Chicago)
    rig = Rig.build(services, clock, model, salt="readiness")
    night = rig.at(2026, 10, 9, 1, 0)  # Friday 01:00 in Chicago: deep sleep
    clock.set_time(night)
    harness = build_harness(services, clock, ready=False, day=rig.kit.time)  # type: ignore[arg-type]
    await harness.engine.start()
    try:
        harness.writer.add(make_draft("刚醒，怎么啦"))
        await harness.message("睡了吗")
        await held(harness)
        (notice,) = notices(harness)
        assert f"缺 {ALL_MISSING}" in notice
        await hold_for(harness, 3600)
        install_ready(services, model=model)
        await harness.engine.on_state_change(1, 2)
        await run_to_idle(
            harness.engine, harness.clock, until=lambda: decision_of(harness.engine) is not None
        )
        decision = decision_of(harness.engine)
        assert decision is not None and decision.mode == "asleep" and decision.woke_up
        plan = rig.planner.ensure(night.astimezone(rig.zone()).date())
        assert plan.morning is not None
        wake = plan.morning.wake
        assert decision.wake_at == wake and decision.send_at > wake
        await run_to_idle(harness.engine, harness.clock)
        texts = [o for o in harness.channel.out if o.kind == "text"]
        reply = next(o for o in texts if o.text == "刚醒，怎么啦")
        assert reply.at > wake and harness.writer.calls == 1
    finally:
        await harness.engine.stop()


# ----------------------------------------------------------------------- the notice book


def test_r18_b8_the_notice_book_is_due_once_per_kind_and_six_hours(services: Services) -> None:
    """D-871: never told -> due; within six hours -> not; after six hours only with a new
    message of his; another set of missing things is another kind."""
    book = NoticeBook(services)
    every = Readiness(("profile", "routine", "persona"))
    card = Readiness(("persona",))
    told = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
    assert book.due(every, told, told)
    book.record(every, told)
    assert book.last(every.key) == told
    assert not book.due(every, told + timedelta(hours=5, minutes=59), told + timedelta(hours=1))
    assert not book.due(every, told + timedelta(hours=6), told - timedelta(minutes=1))
    assert not book.due(every, told + timedelta(hours=6), None)
    # he wrote two hours after it: nothing then, and nothing at six hours by itself
    assert not book.due(every, told + timedelta(hours=6), told + timedelta(hours=2))
    assert book.due(every, told + timedelta(hours=6), told + timedelta(hours=6))
    assert book.due(card, told + timedelta(minutes=1), told)
