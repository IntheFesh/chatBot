"""The local days of the memory and the way a date is worded (R-MEM-002, R-ACT-001, R-NFR-005).

A summary belongs to a local day.  Hers are cut in the zone she was in (with the periods she
travelled), the bot's in the zone the bot lives in - and when that zone is switched, from then on.
The days of a daylight saving change are 23 or 25 hours long, never 24.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from tests.support.clock import ManualClock
from tests.support.memory import CHICAGO, fact_record, followup_record, summary_record, utc
from twin.config.settings import TzRange
from twin.ingest.times import SourceTime
from twin.memory.localdate import BOT, REAL, MemoryClock
from twin.memory.render import (
    BOT_FACT_NOTE,
    day_label,
    fact_line,
    followup_line,
    heading,
    summary_line,
)
from twin.schedule.time_service import BotTimeService

SHANGHAI = "Asia/Shanghai"


class Zone:
    """The bot's zone, which a test can switch (what ``time.bot_timezone`` is in a running bot)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self) -> str:
        return self.name


def calendar(clock: ManualClock, bot: Zone, ranges: list[TzRange] | None = None) -> MemoryClock:
    return MemoryClock(SourceTime(CHICAGO, ranges or []), BotTimeService(clock, bot))


# ------------------------------------------------------------------------------ her days


def test_her_days_are_cut_in_her_zone_and_the_change_of_the_clocks_makes_them_25_or_23_hours(
    clock: ManualClock,
) -> None:
    memory = calendar(clock, Zone(CHICAGO))
    start, end = memory.real_bounds(date(2026, 11, 1))  # the clocks go back at 2 o'clock
    assert (start, end) == (utc(2026, 11, 1, 5), utc(2026, 11, 2, 6))
    assert end - start == timedelta(hours=25)
    start, end = memory.real_bounds(date(2026, 3, 8))  # and forward in the spring
    assert end - start == timedelta(hours=23)
    start, end = memory.real_bounds(date(2026, 10, 9))
    assert end - start == timedelta(hours=24) and start == utc(2026, 10, 9, 5)


def test_a_message_after_midnight_in_utc_still_belongs_to_the_evening_before(
    clock: ManualClock,
) -> None:
    memory = calendar(clock, Zone(CHICAGO))
    moment = utc(2026, 11, 2, 3, 30)  # 21:30 on 1 November in Chicago
    assert memory.real_date(moment) == date(2026, 11, 1)
    assert memory.real_zone(moment).key == CHICAGO
    first, last = memory.real_bounds(date(2026, 11, 1))
    assert first <= moment < last


def test_the_periods_she_travelled_are_cut_in_the_zone_she_was_in(clock: ManualClock) -> None:
    trip = TzRange.model_validate(
        {"from": date(2026, 6, 1), "to": date(2026, 6, 30), "tz": SHANGHAI}
    )
    memory = calendar(clock, Zone(CHICAGO), [trip])
    start, end = memory.real_bounds(date(2026, 6, 10))
    assert (start, end) == (utc(2026, 6, 9, 16), utc(2026, 6, 10, 16))  # a Shanghai day
    assert memory.real_zone(utc(2026, 6, 10, 20)).key == SHANGHAI
    assert memory.real_date(utc(2026, 6, 10, 20)) == date(2026, 6, 11)  # already 04:00 there
    assert memory.zone_key_for_day(REAL, date(2026, 6, 10)) == SHANGHAI
    assert memory.zone_key_for_day(REAL, date(2026, 7, 1)) == CHICAGO  # home again


# ------------------------------------------------------------------------------ the bot's days


def test_the_bots_days_follow_the_zone_it_lives_in_and_move_when_it_is_switched(
    clock: ManualClock,
) -> None:
    where = Zone(CHICAGO)
    memory = calendar(clock, where)
    moment = utc(2026, 11, 1, 18)  # noon in Chicago, 02:00 the next day in Shanghai
    assert memory.bot_zone().key == CHICAGO and memory.bot_date(moment) == date(2026, 11, 1)
    start, end = memory.bot_bounds(date(2026, 11, 1))
    assert end - start == timedelta(hours=25)  # the day the clocks went back

    where.name = SHANGHAI  # `/时区 Asia/Shanghai`
    assert memory.bot_zone().key == SHANGHAI and memory.bot_date(moment) == date(2026, 11, 2)
    start, end = memory.bot_bounds(date(2026, 11, 1))
    assert (start, end) == (utc(2026, 10, 31, 16), utc(2026, 11, 1, 16))  # a plain Shanghai day
    assert memory.zone_key_for_day(BOT, date(2026, 11, 1)) == SHANGHAI
    assert memory.real_date(moment) == date(2026, 11, 1)  # hers did not move with the bot's


def test_either_scope_is_asked_the_same_way(clock: ManualClock) -> None:
    where = Zone(SHANGHAI)
    memory = calendar(clock, where)
    moment = utc(2026, 11, 1, 18)
    assert memory.date_of(REAL, moment) == date(2026, 11, 1)  # Chicago, noon
    assert memory.date_of(BOT, moment) == date(2026, 11, 2)  # Shanghai, 02:00
    assert (
        memory.zone_of(REAL, moment).key == CHICAGO and memory.zone_of(BOT, moment).key == SHANGHAI
    )
    assert memory.bounds_of(REAL, date(2026, 11, 1)) == memory.real_bounds(date(2026, 11, 1))
    assert memory.bounds_of(BOT, date(2026, 11, 1)) == memory.bot_bounds(date(2026, 11, 1))
    assert memory.source_time.zone_on(date(2026, 11, 1)).key == CHICAGO


# ------------------------------------------------------------------------------ the wording


def test_a_day_is_named_with_its_weekday() -> None:
    assert day_label(date(2026, 11, 1)) == "11月1日周日"
    assert day_label(date(2026, 3, 9)) == "3月9日周一"
    assert heading("今天要留意的") == "【今天要留意的】"


def test_a_follow_up_is_worded_on_the_wall_clock_of_the_zone_it_is_shown_in() -> None:
    due = datetime(2026, 11, 1, 7, 30, tzinfo=UTC)  # after the clocks went back in Chicago
    followup = followup_record(text="她要考试", due_at=due)
    assert followup_line(followup, ZoneInfo(CHICAGO)) == "她要考试（11月1日 01:30）"
    assert followup_line(followup, ZoneInfo(SHANGHAI)) == "她要考试（11月1日 15:30）"
    late = followup_record(text="打电话", due_at=utc(2026, 11, 1, 23, 45))
    assert (
        followup_line(late, ZoneInfo(SHANGHAI)) == "打电话（11月2日 07:45）"
    )  # the next day there


def test_a_fact_says_how_far_off_its_date_is_and_whose_word_it_is() -> None:
    fact = fact_record(subject="her", text="她有考试")
    assert fact_line(fact, None) == "她：她有考试"
    assert fact_line(fact, 0).endswith("（就是今天）") and fact_line(fact, 1).endswith(
        "（就是明天）"
    )
    assert fact_line(fact, -1).endswith("（是昨天）") and fact_line(fact, 3).endswith("（3天后）")
    assert (
        fact_line(fact, 8) == "她：她有考试" and fact_line(fact, -2) == "她：她有考试"
    )  # not near
    invented = fact_record(subject="user", text="我养了猫", source="bot_invented")
    assert fact_line(invented, None) == "对方：我养了猫" + BOT_FACT_NOTE
    assert fact_line(fact_record(subject="", text="下雨"), None) == "下雨"


def test_a_summary_says_which_conversation_it_is_of() -> None:
    real = summary_record(local_date=date(2026, 11, 1), scope="real", text="聊了考试")
    bot = summary_record(local_date=date(2026, 11, 1), scope="bot", text="聊了晚饭")
    assert summary_line(real) == "11月1日周日（聊天记录）：聊了考试"
    assert summary_line(bot) == "11月1日周日（和对方的聊天）：聊了晚饭"
