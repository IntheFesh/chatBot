"""``/评分``, the proactive line of ``/状态``, ``/重来`` and the life line's "told" marks."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date, timedelta

import pytest
from sqlalchemy import text

from tests.support.clock import ManualClock
from tests.support.embedding import HashingBackend
from tests.support.engine_harness import Harness, build_harness
from tests.support.memory import make_memory
from tests.support.proactive_world import World, proactive_model
from tests.support.routine import Rig
from tests.support.style_models import ScriptedStyleClient
from twin.commands import texts
from twin.commands.rating import RatingCommand, parse_rating, rating_command
from twin.commands.registry import CommandRegistry, UsageError
from twin.commands.router import CommandRouter
from twin.commands.status import ProactiveStatus
from twin.config.runtime import PROACTIVE_DAILY_MAX, PROACTIVE_DAILY_MIN, PROACTIVE_ENABLED
from twin.engine.backend_select import BackendSelector
from twin.engine.command_port import CommandContext
from twin.engine.component import command_port_for
from twin.engine.style_models import StyleModels
from twin.engine.turns import BotTurnStore, OutboundBubble, ReplyMeta
from twin.llm.runtime import LlmRuntime, build_llm_runtime
from twin.memory.lifeline import SHARED_STEP, LifelineStore, PlannedEvent, shared_ids_of
from twin.schedule.proactive.component import proactive_status_for
from twin.schedule.proactive.settings import daily_range, is_enabled, is_paused
from twin.schedule.proactive.store import RatingStore
from twin.schedule.proactive.types import TriggerKind
from twin.schedule.service import KIT_KEY
from twin.services import Services
from twin.storage.state import read_state_version

PREFIX = "⚙️ "


# -------------------------------------------------------------------------- the argument


@pytest.mark.parametrize(
    ("args", "score", "note"),
    [
        ("4", 4, ""),
        ("４", 4, ""),
        ("四", 4, ""),
        ("4分", 4, ""),
        ("4/5", 4, ""),
        ("4 / 5", 4, ""),
        ("4星", 4, ""),
        ("四个星", 4, ""),
        ("五星", 5, ""),
        ("1", 1, ""),
        ("5 晚安发得很自然", 5, "晚安发得很自然"),
        ("4：有点频繁", 4, "有点频繁"),
        ("4，有点频繁", 4, "有点频繁"),
        ("4, ok", 4, "ok"),
        ("4分 不错", 4, "不错"),
        ("  ３　还行  ", 3, "还行"),
        (": 2 太多了", 2, "太多了"),
        ("3\n第二行也算备注", 3, "第二行也算备注"),
    ],
)
def test_the_score_is_read_forgivingly_and_the_rest_is_the_note(
    args: str, score: int, note: str
) -> None:
    assert parse_rating(args) == (score, note)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ("", ""),
        ("   ", ""),
        ("很好", texts.RATING_NO_NUMBER),
        ("0", texts.RATING_RANGE),
        ("6", texts.RATING_RANGE),
        ("10", texts.RATING_RANGE),
        ("-1", texts.RATING_NO_NUMBER),
        ("4.5", texts.RATING_NOT_WHOLE),
        ("４.５", texts.RATING_NOT_WHOLE),
    ],
)
def test_a_score_that_is_missing_or_outside_one_to_five_is_refused(
    args: str, expected: str
) -> None:
    with pytest.raises(UsageError) as caught:
        parse_rating(args)
    assert str(caught.value) == expected


# ------------------------------------------------------------------------------ the router


class Commands:
    """A router with ``/评分`` and the ratings it writes, on the real test database."""

    def __init__(self, services: Services, clock: ManualClock) -> None:
        self.services, self.clock = services, clock
        rig = Rig.build(services, clock, proactive_model())
        services.extras[KIT_KEY] = rig.kit
        self.time = rig.kit.time
        self.store = RatingStore(services.db, clock)
        self.router = CommandRouter(CommandRegistry())
        self.router.register(rating_command(self.store, clock, self.time))

    async def say(self, line: str) -> str:
        outcome = await self.router.handle(line, CommandContext(self.clock.now_utc(), "in-1"))
        assert outcome is not None, line
        return outcome.reply

    def rows(self) -> list[tuple[int, str | None]]:
        return [(r.score, r.note) for r in self.store.recent(50)][::-1]


@pytest.fixture
def commands(services: Services, clock: ManualClock) -> Commands:
    return Commands(services, clock)


async def test_a_rating_is_written_at_once_and_answered_in_the_voice_of_the_system(
    commands: Commands,
) -> None:
    reply = await commands.say("/评分 4 晚安发得很自然")
    assert reply.startswith(PREFIX) and "记下了：4/5。" in reply and "备注也存好了。" in reply
    assert "最近 7 天一共评过 1 次，平均 4.0 分。" in reply
    assert commands.rows() == [(4, "晚安发得很自然")]
    row = commands.store.recent(1)[0]
    assert row.local_date == commands.time.local_date(commands.clock.now_utc())


async def test_a_rating_without_a_note_stores_none_and_the_week_is_averaged(
    commands: Commands,
) -> None:
    await commands.say("/评分 5")
    commands.clock.tick(3600)
    reply = await commands.say("／评分：３")
    assert "最近 7 天一共评过 2 次，平均 4.0 分。" in reply and "备注" not in reply
    assert commands.rows() == [(5, None), (3, None)]
    commands.clock.tick(8 * 86400)
    later = await commands.say("/rate 2")  # the alias; the old ones are out of the week
    assert "一共评过 1 次，平均 2.0 分" in later


async def test_a_bad_rating_is_answered_with_the_usage_and_nothing_is_written(
    commands: Commands,
) -> None:
    for line in ("/评分", "/评分 0", "/评分 6", "/评分 很好", "/评分 4.5"):
        reply = await commands.say(line)
        assert reply.startswith(PREFIX) and "/评分 <1-5> [备注]" in reply, line
    assert commands.rows() == []
    assert texts.RATING_RANGE in await commands.say("/评分 7")


async def test_the_note_is_sealed_in_the_database(commands: Commands) -> None:
    await commands.say("/评分 4 这句备注不应该以明文存放")
    with commands.services.db.session() as session:
        stored = session.execute(text("SELECT note, score FROM ratings")).one()
    assert stored.score == 4
    assert b"\xe5\xa4\x87\xe6\xb3\xa8" not in bytes(stored.note)
    assert "不应该" not in repr(stored.note)


async def test_the_store_refuses_a_score_outside_the_range(commands: Commands) -> None:
    with pytest.raises(ValueError, match="1 to 5"):
        commands.store.add(6, None, at=commands.clock.now_utc(), local_date=date(2026, 10, 9))


def test_the_command_is_a_member_of_the_rating_group_with_an_example() -> None:
    spec = rating_command(object(), object(), object())  # type: ignore[arg-type]
    assert spec.name == "评分" and spec.group == "学习与评分" and "rate" in spec.aliases
    assert spec.example == "/评分 4 晚安发得很自然"
    assert isinstance(spec.handler.__self__, RatingCommand)  # type: ignore[attr-defined]


@pytest.fixture
async def chat(services: Services, clock: ManualClock) -> AsyncIterator[tuple[Harness, Commands]]:
    commands = Commands(services, clock)
    harness = build_harness(services, clock, commands=commands.router)
    await harness.engine.start()
    yield harness, commands
    await harness.engine.stop()


async def test_a_rating_is_a_command_and_never_reaches_the_conversation_or_the_memory(
    chat: tuple[Harness, Commands],
) -> None:
    harness, commands = chat
    version = _version(commands.services)
    await harness.message("/评分 5 很自然")
    assert harness.channel.texts[0].startswith(PREFIX + "记下了：5/5。")
    assert harness.writer.calls == 0 and harness.engine.snapshot().state == "IDLE"
    assert commands.rows() == [(5, "很自然")]
    from sqlalchemy import select

    from twin.storage.engine_models import BotTurn

    with harness.services.db.session() as session:
        stored = [
            (r.direction, r.is_command)
            for r in session.scalars(select(BotTurn).order_by(BotTurn.at, BotTurn.id))
        ]
    assert stored == [("in", True), ("out", True)]
    assert version <= _version(commands.services)


def _version(services: Services) -> int:
    with services.db.session() as session:
        return read_state_version(session)


# --------------------------------------------------------------------------- /状态


async def test_the_status_line_of_a_running_scheduler(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    await calm.tick_at(calm.at(0, 10))
    source = proactive_status_for(calm.services, calm.channel.session_state)
    calm.put(TriggerKind.MEAL, calm.at(12, 0))
    calm.go_to(calm.at(9, 0))
    status = source()
    assert isinstance(status, ProactiveStatus)
    assert (status.low, status.high, status.enabled) == (1, 6, True)
    assert status.sent_today == 0 and status.quota == 6 and status.blocked is None
    assert status.next_kind in ("起床问候", "饭点") and status.next_at is not None
    assert status.next_at >= calm.clock.now_utc()


async def test_the_status_counts_what_went_out_today_and_names_what_blocks_it(
    calm: World,
) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.put(TriggerKind.SHARE, calm.at(10, 0))
    await calm.tick_at(calm.at(10, 5))
    calm.go_to(calm.at(10, 10))
    source = proactive_status_for(calm.services, calm.channel.session_state)
    assert source().sent_today == 1  # type: ignore[union-attr]
    calm.go_to(calm.at(23, 0))  # the window of the platform is over by then
    blocked = source()
    assert blocked is not None and blocked.blocked == "被窗口抑制"


async def test_the_status_without_a_plan_still_reports_the_range_and_a_channel_that_cannot_say(
    calm: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    from twin.schedule.time_service import PlanUnavailableError

    def no_plan(_moment: object) -> object:
        raise PlanUnavailableError("no plan")

    def no_channel() -> object:
        raise RuntimeError("the channel is gone")

    monkeypatch.setattr(calm.rig.kit.planner, "plan_at", no_plan)
    source = proactive_status_for(calm.services, no_channel)  # type: ignore[arg-type]
    status = source()
    assert status is not None and status.quota is None and status.blocked is None
    assert (status.low, status.high) == (1, 6)


async def test_the_status_follows_the_switch_and_the_range(calm: World) -> None:
    source = proactive_status_for(calm.services, calm.channel.session_state)
    calm.services.runtime.set(PROACTIVE_ENABLED, False, by="test")
    calm.services.runtime.set(PROACTIVE_DAILY_MIN, 2, by="test")
    calm.services.runtime.set(PROACTIVE_DAILY_MAX, 4, by="test")
    status = source()
    assert status is not None and (status.low, status.high, status.enabled) == (2, 4, False)


async def test_the_proactive_line_of_the_status_report(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.go_to(calm.at(9, 0))
    parts = calm.engine.kit
    assert parts is not None
    router = command_port_for(
        calm.services,
        calm.engine,
        llm=parts.llm,
        style=parts.style,
        channel=calm.channel,
        memory=parts.data.memory,
        proactive=proactive_status_for(calm.services, calm.channel.session_state),
    )
    outcome = await router.handle("/状态", CommandContext(calm.clock.now_utc(), "s1"))
    assert outcome is not None
    line = next(row for row in outcome.reply.splitlines() if row.startswith("主动消息"))
    assert line.startswith("主动消息：每天 1-6 条（开），今天已发 0 条；今天计划 6 条")
    without = command_port_for(
        calm.services,
        calm.engine,
        llm=parts.llm,
        style=parts.style,
        channel=calm.channel,
        memory=parts.data.memory,
    )
    plain = await without.handle("/状态", CommandContext(calm.clock.now_utc(), "s2"))
    assert plain is not None and texts.STATUS_PROACTIVE_OFF in plain.reply


# ----------------------------------------------------------------- the settings of round 11


def test_the_range_comes_from_the_settings_or_the_configuration(services: Services) -> None:
    services.runtime.initialize()
    config = services.settings.proactive
    assert daily_range(services.runtime, config) == (config.daily_min, config.daily_max)
    services.runtime.set(PROACTIVE_DAILY_MIN, 3, by="test")
    assert daily_range(services.runtime, config) == (3, max(3, config.daily_max))
    services.runtime.set(PROACTIVE_DAILY_MAX, 2, by="test")
    assert daily_range(services.runtime, config) == (3, 3)  # a maximum below the minimum is raised
    services.runtime.set(PROACTIVE_DAILY_MAX, 5, by="test")
    assert daily_range(services.runtime, config) == (3, 5)


def test_the_switch_and_the_pause(services: Services, clock: ManualClock) -> None:
    from twin.config.runtime import ENGINE_PAUSED_UNTIL, PAUSED

    services.runtime.initialize()
    now = clock.now_utc()
    assert is_enabled(services.runtime) and not is_paused(services.runtime, now)
    services.runtime.set(PROACTIVE_ENABLED, False, by="test")
    assert not is_enabled(services.runtime)
    services.runtime.set(ENGINE_PAUSED_UNTIL, now + timedelta(minutes=30), by="test")
    assert is_paused(services.runtime, now)
    assert not is_paused(services.runtime, now + timedelta(minutes=31))  # it ended by itself
    services.runtime.set(ENGINE_PAUSED_UNTIL, None, by="test")
    services.runtime.set(PAUSED, True, by="test")
    assert is_paused(services.runtime, now)


# ------------------------------------------------------------------ the life line's marks


class Redo:
    """The router of the application with a life line that a proactive message has told."""

    def __init__(self, services: Services, clock: ManualClock) -> None:
        services.runtime.initialize()
        self.llm: LlmRuntime = build_llm_runtime(services)
        self.clock = clock
        self.turns = BotTurnStore(services.db, clock)
        self.memory = make_memory(services)
        selector = BackendSelector(
            runtime=services.runtime,
            models=StyleModels(services.db),
            client=ScriptedStyleClient(),
            config=services.settings.backend,
            clock=clock,
            alerts=services.alerts,
            limits=self.llm.budget.limits,
        )
        self.router = CommandRouter.from_services(
            services, self.llm, selector=selector, turns=self.turns, memory=self.memory
        )
        self.lifeline = LifelineStore(self.memory)

    async def say(self, line: str) -> str:
        outcome = await self.router.handle(line, CommandContext(self.clock.now_utc(), "r1"))
        assert outcome is not None
        return outcome.reply


@pytest.fixture
async def redo(
    services: Services, clock: ManualClock, embedder: HashingBackend
) -> AsyncIterator[Redo]:
    built = Redo(services, clock)
    yield built
    await built.llm.client.aclose()


def told_day(redo: Redo) -> tuple[date, list[str]]:
    day = redo.clock.now_utc().date()
    entries = redo.lifeline.replace_plan(
        day,
        [
            PlannedEvent("在图书馆看文献", "09:00", "11:00"),
            PlannedEvent("去食堂吃饭", "11:30", "12:00"),
            PlannedEvent("午睡", "13:00", "14:00"),
        ],
    )
    return day, [entry.id for entry in entries]


async def test_redo_of_a_proactive_message_takes_its_marks_off_the_life_line(redo: Redo) -> None:
    day, ids = told_day(redo)
    reply = redo.turns.add_reply(
        [OutboundBubble("刚从图书馆出来", redo.clock.now_utc() + timedelta(seconds=1))],
        ReplyMeta(
            "deepseek",
            actions=({"step": "proactive", "count": 1}, {"step": SHARED_STEP, "ids": ids[:2]}),
        ),
    )
    reply_id = reply[0].reply_id or ""
    assert redo.lifeline.mark_shared(ids[:2], reply_id=reply_id) == 2
    assert [e.shared for e in redo.lifeline.day(day)] == [True, True, False]
    redo.clock.tick(30)
    answer = await redo.say("/重来")
    assert texts.REDO_UNSHARED.format(count=2) in answer
    assert [e.shared for e in redo.lifeline.day(day)] == [False, False, False]
    assert [e.activity for e in redo.lifeline.unshared(day)] == [
        "在图书馆看文献",
        "去食堂吃饭",
        "午睡",
    ]


async def test_redo_of_an_ordinary_reply_leaves_the_marks_alone(redo: Redo) -> None:
    day, ids = told_day(redo)
    redo.lifeline.mark_shared(ids[:1])
    now = redo.clock.now_utc()
    redo.turns.add_inbound(at=now, kind="text", text="在吗")
    redo.turns.add_reply(
        [OutboundBubble("在呢", now + timedelta(seconds=3))], ReplyMeta("deepseek")
    )
    redo.clock.tick(30)
    answer = await redo.say("/重来")
    assert texts.REDO_UNSHARED.format(count=1) not in answer
    assert [e.shared for e in redo.lifeline.day(day)] == [True, False, False]


# ------------------------------------------------------------- the data layer of the marks


def test_marking_entries_as_told_and_taking_the_mark_off(
    services: Services, clock: ManualClock, embedder: HashingBackend
) -> None:
    memory = make_memory(services)
    lifeline = LifelineStore(memory)
    day = date(2026, 10, 9)
    first, second = lifeline.replace_plan(
        day, [PlannedEvent("看书", "09:00", "10:00"), PlannedEvent("跑步", "17:00", "18:00")]
    )
    assert not first.shared and first.shared_at is None
    assert lifeline.mark_shared([]) == 0 and lifeline.unshare([]) == 0
    assert lifeline.mark_shared([first.id], reply_id="reply-1") == 1
    marked = next(e for e in lifeline.day(day) if e.id == first.id)
    assert marked.shared and marked.shared_reply_id == "reply-1" and marked.shared_at is not None
    assert [e.id for e in lifeline.unshared(day)] == [second.id]
    assert lifeline.mark_shared([first.id]) == 0  # already told: not counted twice
    assert lifeline.unshare([first.id, second.id]) == 1
    assert lifeline.unshared(day) == lifeline.day(day)
    assert lifeline.unshare([first.id]) == 0


def test_a_replan_of_the_day_keeps_what_was_told_of_an_improvised_entry(
    services: Services, embedder: HashingBackend
) -> None:
    memory = make_memory(services)
    lifeline = LifelineStore(memory)
    day = date(2026, 10, 9)
    added = lifeline.add_improvised(day, PlannedEvent("路上买了杯奶茶", "11:10", "11:20"))
    lifeline.mark_shared([added.id])
    lifeline.replace_plan(day, [PlannedEvent("看书", "09:00", "10:00")])
    kept = next(e for e in lifeline.day(day) if e.id == added.id)
    assert kept.shared and kept.source == "improvised"


def test_the_shared_ids_are_read_back_from_the_actions_of_a_reply() -> None:
    actions = [
        {"step": "proactive", "count": 1},
        {"step": SHARED_STEP, "count": 2, "ids": ["a", "b", ""]},
        {"step": SHARED_STEP, "ids": ["b", "c"]},
        {"step": SHARED_STEP, "ids": "not a list"},
        {"step": "other", "ids": ["z"]},
    ]
    assert shared_ids_of(actions) == ["a", "b", "c"]
    assert shared_ids_of([]) == []
