"""``/时区 /暂停 /恢复 /主动 /作息`` (R-CMD-002, R-SCH-002, R-ACT-005, R-PRO-002).

The router runs on the real tables with the schedule of ``tests/support/routine`` (she sleeps
23:30-07:30 in Chicago, busy 13:00-17:00 on workdays) and the manual clock, which starts on Friday
9 October 2026 at 07:00 Chicago time.  Every answer starts with the system prefix and none of them
passes through the persona.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tests.support.clock import ManualClock
from tests.support.commands_world import PREFIX, CommandWorld, open_world
from tests.support.embedding import HashingBackend
from twin.commands import texts
from twin.config.runtime import (
    BOT_TIMEZONE,
    ENGINE_PAUSED_UNTIL,
    PROACTIVE_DAILY_MAX,
    PROACTIVE_DAILY_MIN,
    PROACTIVE_ENABLED,
)
from twin.profile.overrides import RoutineOverrides
from twin.schedule.events import PlanRebuilt, TimezoneSwitched
from twin.services import Services

CHICAGO = ZoneInfo("America/Chicago")
START = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)  # 07:00 in Chicago


@pytest.fixture
async def world(
    services: Services, clock: ManualClock, embedder: HashingBackend
) -> AsyncIterator[CommandWorld]:
    async with open_world(services, clock, start=START) as built:
        yield built


def zone(world: CommandWorld) -> str:
    return str(world.services.runtime.get(BOT_TIMEZONE))


# ------------------------------------------------------------------------------ /时区


async def test_the_zone_command_shows_the_zone_the_local_time_and_what_she_is_doing(
    world: CommandWorld,
) -> None:
    reply = await world.reply("/时区 查看")
    assert reply.startswith("当前时区：America/Chicago，当地时间 2026-10-09 周五 07:00")
    assert "她现在：" in reply and "到 07:30" in reply
    assert (await world.reply("／时区：查看")) == reply  # full-width slash and colon
    assert await world.reply("/tz show") == reply  # the English name and spelling


@pytest.mark.parametrize(
    "spelling",
    ["北京", "上海", "中国", "国内", "Beijing", "asia/shanghai", "ASIA/SHANGHAI", " 北京 "],
)
async def test_the_zone_is_switched_by_alias_or_name_and_remembered_as_a_command(
    world: CommandWorld, spelling: str
) -> None:
    seen: list[TimezoneSwitched] = []

    async def note(event: TimezoneSwitched) -> None:
        seen.append(event)

    world.rig.kit.events.subscribe(TimezoneSwitched, note)
    reply = await world.reply(f"/时区 {spelling}")
    assert "时区已从 America/Chicago 切到 Asia/Shanghai" in reply
    assert "当地时间 2026-10-09 周五 20:00" in reply and "她现在：" in reply
    assert zone(world) == "Asia/Shanghai"
    assert world.services.runtime.history(BOT_TIMEZONE)[-1].by == "command"
    record = world.rig.planner.history.latest()
    assert record is not None and (record.to_timezone, record.source) == (
        "Asia/Shanghai",
        "command",
    )
    assert [(e.old_timezone, e.new_timezone) for e in seen] == [
        ("America/Chicago", "Asia/Shanghai")
    ]


async def test_a_switch_plans_the_rest_of_the_day_in_the_new_zone_and_back(
    world: CommandWorld,
) -> None:
    await world.reply("/时区 上海")
    plan = world.rig.planner.plan_at(world.clock.now_utc())
    assert plan.timezone == "Asia/Shanghai" and plan.reason == "timezone_switch"
    await world.reply("/时区 芝加哥")
    assert zone(world) == "America/Chicago"
    assert world.rig.planner.plan_at(world.clock.now_utc()).timezone == "America/Chicago"
    assert [r.to_timezone for r in world.rig.planner.history.recent(5)] == [
        "America/Chicago",
        "Asia/Shanghai",
    ]


async def test_switching_to_the_zone_she_is_in_changes_nothing(world: CommandWorld) -> None:
    reply = await world.reply("/时区 芝加哥")
    assert reply.startswith("已经是 America/Chicago 了，当地时间 2026-10-09 周五 07:00")
    assert world.rig.planner.history.latest() is None


@pytest.mark.parametrize("bad", ["", "火星", "Mars/Olympus", "../etc/passwd", "UTC+8x", "/"])
async def test_a_zone_that_does_not_exist_gets_the_usage_and_changes_nothing(
    world: CommandWorld, bad: str
) -> None:
    reply = await world.reply(f"/时区 {bad}")
    assert "用法：/时区 <IANA 名称>|查看" in reply and "例如：/时区 北京" in reply
    if bad:
        assert reply.startswith("参数不对：不认识这个时区") or "不认识这个时区" in reply
    assert zone(world) == "America/Chicago" and world.rig.planner.history.latest() is None


# ------------------------------------------------------------------- /暂停 and /恢复


def paused_until(world: CommandWorld) -> datetime | None:
    value = world.services.runtime.get(ENGINE_PAUSED_UNTIL)
    return value if value is None else value.astimezone(UTC)


@pytest.mark.parametrize(
    ("text", "ends"),
    [
        ("2小时", START + timedelta(hours=2)),
        ("２小时", START + timedelta(hours=2)),
        ("30分钟", START + timedelta(minutes=30)),
        ("一个半小时", START + timedelta(minutes=90)),
        ("到明早", START + timedelta(hours=1)),  # 07:00 now: the morning that is coming is 08:00
        ("到22:00", START + timedelta(hours=15)),
        ("到晚上10点", START + timedelta(hours=15)),
        ("：2小时", START + timedelta(hours=2)),  # a colon after the name is a separator
    ],
)
async def test_a_pause_is_stored_as_an_absolute_utc_moment(
    world: CommandWorld, text: str, ends: datetime
) -> None:
    reply = await world.reply(f"/暂停{text}" if text.startswith("：") else f"/暂停 {text}")
    assert reply.startswith("好的，暂停到 ") and "America/Chicago" in reply
    assert paused_until(world) == ends
    assert world.services.runtime.history(ENGINE_PAUSED_UNTIL)[-1].by == "command"


async def test_until_tomorrow_morning_at_night_is_the_morning_of_the_next_day(
    world: CommandWorld,
) -> None:
    world.clock.set_time(datetime(2026, 10, 10, 3, 30, tzinfo=UTC))  # 22:30 in Chicago
    reply = await world.reply("/暂停 到明早")
    assert "10-10 周六 08:00" in reply
    assert paused_until(world) == datetime(2026, 10, 10, 13, 0, tzinfo=UTC)


@pytest.mark.parametrize("bad", ["", "很久", "30", "0分钟", "999小时", "8天", "到25:00", "到"])
async def test_a_pause_that_cannot_be_read_gets_the_usage_and_changes_nothing(
    world: CommandWorld, bad: str
) -> None:
    reply = await world.reply(f"/暂停 {bad}")
    assert "用法：/暂停 <时长>" in reply and "例如：/暂停 2小时" in reply
    assert paused_until(world) is None


async def test_the_reasons_for_a_refused_pause_are_given(world: CommandWorld) -> None:
    assert "最长只能暂停 168 小时" in await world.reply("/暂停 200小时")
    assert "没看懂这个时长" in await world.reply("/暂停 很久")
    assert "必须在现在之后" in await world.reply("/暂停 0分钟")
    world.services.settings.commands.pause_max_h = 24
    assert "最长只能暂停 24 小时" in await world.reply("/暂停 2天")


async def test_resume_ends_the_pause_and_says_when_there_was_none(world: CommandWorld) -> None:
    assert await world.reply("/恢复") == texts.RESUME_NOT_PAUSED
    await world.reply("/暂停 3小时")
    assert await world.reply("／恢复") == texts.RESUME_DONE and paused_until(world) is None
    assert await world.reply("/resume") == texts.RESUME_NOT_PAUSED


async def test_a_pause_that_has_run_out_is_cleared_by_resume(world: CommandWorld) -> None:
    await world.reply("/暂停 1小时")
    world.clock.tick(2 * 3600)
    assert await world.reply("/恢复") == texts.RESUME_NOT_PAUSED
    assert paused_until(world) is None


# ------------------------------------------------------------------------------ /主动


def quota(world: CommandWorld) -> tuple[int, int, bool]:
    runtime = world.services.runtime
    return (
        int(runtime.get(PROACTIVE_DAILY_MIN)),
        int(runtime.get(PROACTIVE_DAILY_MAX)),
        bool(runtime.get(PROACTIVE_ENABLED)),
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [("2-5", (2, 5)), ("２－５", (2, 5)), ("0-0", (0, 0)), ("2~7", (2, 7)), ("3-12", (3, 12))],
)
async def test_the_proactive_range_is_set_within_the_limits(
    world: CommandWorld, text: str, expected: tuple[int, int]
) -> None:
    reply = await world.reply(f"/主动 {text}")
    assert reply.startswith(f"主动消息每天 {expected[0]}-{expected[1]} 条，已设好")
    assert quota(world)[:2] == expected
    assert world.services.runtime.history(PROACTIVE_DAILY_MAX)[-1].by == "command"


@pytest.mark.parametrize("bad", ["", "5-2", "0-13", "13-14", "a-b", "2", "2-3-4", "-1-3", "自动"])
async def test_a_proactive_range_that_is_not_allowed_gets_the_usage_and_changes_nothing(
    world: CommandWorld, bad: str
) -> None:
    before = quota(world)
    reply = await world.reply(f"/主动 {bad}")
    assert "用法：/主动 <最少>-<最多>|开|关" in reply and "例如：/主动 2-5" in reply
    assert quota(world) == before


async def test_the_proactive_messages_are_switched_on_and_off(world: CommandWorld) -> None:
    assert quota(world)[2] is True
    assert await world.reply("/主动 关") == texts.PROACTIVE_OFF and quota(world)[2] is False
    assert "现在是关着的" in await world.reply("/主动 2-4")
    assert await world.reply("/主动 开") == texts.PROACTIVE_ON and quota(world) == (2, 4, True)
    assert await world.reply("/proactive off") == texts.PROACTIVE_OFF


async def test_the_two_bounds_are_written_so_that_min_never_exceeds_max(
    world: CommandWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = world.services.runtime
    steps: list[tuple[int, int]] = []
    original = type(runtime).set

    def watching(self: object, spec: object, value: object, *, by: str = "user") -> bool:
        changed = original(self, spec, value, by=by)  # type: ignore[arg-type]
        steps.append((int(runtime.get(PROACTIVE_DAILY_MIN)), int(runtime.get(PROACTIVE_DAILY_MAX))))
        return changed

    monkeypatch.setattr(type(runtime), "set", watching)
    await world.reply("/主动 8-10")  # the new minimum is above the old maximum (6)
    await world.reply("/主动 1-2")  # the new maximum is below the old minimum (8)
    await world.reply("/主动 1-5")
    assert steps and all(low <= high for low, high in steps)
    assert quota(world)[:2] == (1, 5)


async def test_the_range_reaches_the_plan_of_today(world: CommandWorld) -> None:
    first = world.rig.planner.ensure(world.rig.kit.time.local_date(), reason="daily")
    rebuilt: list[PlanRebuilt] = []

    async def note(event: PlanRebuilt) -> None:
        rebuilt.append(event)

    world.rig.kit.events.subscribe(PlanRebuilt, note)
    await world.reply("/主动 3-3")
    plan = world.rig.planner.plan_at(world.clock.now_utc())
    assert plan.id != first.id and (plan.quota.minimum, plan.quota.maximum) == (3, 3)
    assert rebuilt and rebuilt[-1].reason == "proactive_changed"
    await world.reply("/主动 关")
    assert world.rig.planner.plan_at(world.clock.now_utc()).quota.total == 0


# ------------------------------------------------------------------------------ /作息


def overrides(world: CommandWorld) -> RoutineOverrides:
    return RoutineOverrides(world.services.db, world.clock)


def wake_local(world: CommandWorld) -> str:
    plan = world.rig.planner.plan_at(world.clock.now_utc())
    assert plan.wake is not None
    return f"{plan.wake.astimezone(CHICAGO):%H:%M}"


async def test_a_sleep_correction_is_stored_and_the_rest_of_today_is_planned_again(
    world: CommandWorld,
) -> None:
    assert wake_local(world) == "07:30"
    reply = await world.reply("/作息 睡 01:00-08:30")
    assert reply.startswith(
        "睡眠时间已设为每天 01:00–08:30（America/Chicago本地时间），记为第 1 条。"
    )
    assert reply.endswith(texts.ROUTINE_REBUILT)
    [item] = overrides(world).entries()
    assert item.kind == "sleep" and (item.params["start"], item.params["end"]) == ("01:00", "08:30")
    assert wake_local(world) == "08:30"  # the plan of today was made again with the correction


async def test_busy_time_and_holidays_are_stored_in_every_spelling_the_table_names(
    world: CommandWorld,
) -> None:
    busy = await world.reply("/作息 忙 周一至周五 09:00-11:00")
    assert busy.startswith("已记下：周一、周二、周三、周四、周五 09:00–11:00 她在忙，记为第 1 条。")
    again = await world.reply("／作息　忙　周六、周日　10:00 - 12:00")  # full-width, extra spaces
    assert "周六、周日 10:00–12:00" in again and "第 2 条" in again
    single = await world.reply("/作息 忙 周三 14:00至15:00")
    assert "周三 14:00–15:00" in single
    holiday = await world.reply("/作息 假期 2026-10-09")
    assert holiday.startswith("已记下：2026-10-09 至 2026-10-09 按假期算，记为第 4 条。")
    ranged = await world.reply("/作息 假期 2026-10-12..2026-10-14")
    assert "2026-10-12 至 2026-10-14" in ranged
    kinds = [(v.kind, v.params) for v in overrides(world).entries()]
    assert [k for k, _ in kinds] == ["busy", "busy", "busy", "holiday", "holiday"]
    assert kinds[0][1]["weekdays"] == [0, 1, 2, 3, 4]
    assert world.rig.planner.plan_at(world.clock.now_utc()).day_type == "holiday"


async def test_the_corrections_are_listed_with_numbers_and_deleted_by_number(
    world: CommandWorld,
) -> None:
    assert await world.reply("/作息 查看") == texts.ROUTINE_EMPTY
    await world.reply("/作息 睡 01:00-08:30")
    await world.reply("/作息 假期 2026-10-09")
    listing = await world.reply("/作息 查看")
    lines = listing.split("\n")
    assert lines[0] == texts.ROUTINE_LIST_HEADER
    assert lines[1].startswith("1. 睡眠 01:00–08:30") and lines[2].startswith(
        "2. 节假日 2026-10-09"
    )
    assert wake_local(world) == "08:30"
    removed = await world.reply("/作息 删除 1")
    assert removed.startswith("已删除第 1 条：睡眠 01:00–08:30") and removed.endswith(
        texts.ROUTINE_REBUILT
    )
    assert [v.kind for v in overrides(world).entries()] == ["holiday"]
    assert (await world.reply("/作息 查看")).split("\n")[1].startswith("1. 节假日")
    assert wake_local(world) != "08:30"  # the plan went back to what the data says


async def test_a_disabled_correction_is_listed_as_such(world: CommandWorld) -> None:
    await world.reply("/作息 睡 01:00-08:30")
    [item] = overrides(world).entries()
    overrides(world).set_enabled(item.id, False)
    assert texts.ROUTINE_LIST_DISABLED in await world.reply("/作息 查看")


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "睡",
        "睡 25:00-26:00",
        "睡 01:00-01:00",
        "睡 01:00",
        "忙",
        "忙 周一",
        "忙 周九 09:00-10:00",
        "忙 周一 11:00-09:00",
        "假期",
        "假期 abc",
        "假期 2026-10-07..2026-10-01",
        "删除",
        "删除 x",
        "删除 0",
        "别的 1",
    ],
)
async def test_a_bad_correction_gets_the_usage_and_stores_nothing(
    world: CommandWorld, bad: str
) -> None:
    reply = await world.reply(f"/作息 {bad}")
    assert reply.startswith(("参数不对：", "用法：")) and "/作息 睡 <HH:MM>-<HH:MM>" in reply
    assert overrides(world).entries() == []


async def test_deleting_a_number_that_does_not_exist_says_so(world: CommandWorld) -> None:
    await world.reply("/作息 睡 01:00-08:30")
    reply = await world.reply("/作息 删除 7")
    assert texts.ROUTINE_NO_SUCH.format(number=7) in reply and len(overrides(world).entries()) == 1


async def test_a_plan_that_cannot_be_made_again_now_is_said_not_hidden(
    world: CommandWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object, **kwargs: object) -> object:
        raise RuntimeError("no plan")

    monkeypatch.setattr(world.rig.planner, "refresh", refuse)
    reply = await world.reply("/作息 睡 01:00-08:30")
    assert reply.endswith(texts.ROUTINE_REBUILD_LATER) and len(overrides(world).entries()) == 1


async def test_the_answers_are_system_voice_and_never_ask_the_persona(world: CommandWorld) -> None:
    for text in ("/时区 查看", "/暂停 1小时", "/恢复", "/主动 1-3", "/作息 查看"):
        outcome = await world.say(text)
        assert outcome.reply.startswith(PREFIX) and not outcome.redo


# --------------------------------------------------------- through the schedule component


async def test_inside_the_application_the_commands_go_through_the_schedule_component(
    world: CommandWorld,
) -> None:
    from twin.commands.router import CommandRouter
    from twin.commands.routine_commands import ComponentSchedule
    from twin.engine.command_port import CommandContext
    from twin.schedule.component import ScheduleComponent

    component = ScheduleComponent(world.services, kit=world.rig.kit)
    router = CommandRouter.from_services(
        world.services,
        world.llm,
        selector=world.selector,
        turns=world.turns,
        feedback=world.feedback,
        memory=world.memory,
        schedule=ComponentSchedule(component, world.rig.kit),
        data=world.views,
    )
    switched: list[TimezoneSwitched] = []
    rebuilt: list[PlanRebuilt] = []

    async def note_switch(event: TimezoneSwitched) -> None:
        switched.append(event)

    async def note_rebuild(event: PlanRebuilt) -> None:
        rebuilt.append(event)

    component.events.subscribe(TimezoneSwitched, note_switch)
    component.events.subscribe(PlanRebuilt, note_rebuild)

    async def reply(text: str) -> str:
        outcome = await router.handle(text, CommandContext(world.clock.now_utc(), "in-1"))
        assert outcome is not None
        return outcome.reply

    assert "时区已从 America/Chicago 切到 Asia/Shanghai" in await reply("/时区 北京")
    assert [(e.old_timezone, e.new_timezone) for e in switched] == [
        ("America/Chicago", "Asia/Shanghai")
    ]
    await reply("/主动 3-3")
    await reply("/作息 睡 01:00-08:30")
    assert {e.reason for e in rebuilt} >= {"proactive_changed", "routine_changed"}
    plan = world.rig.planner.plan_at(world.clock.now_utc())
    assert plan.timezone == "Asia/Shanghai" and (plan.quota.minimum, plan.quota.maximum) == (3, 3)
