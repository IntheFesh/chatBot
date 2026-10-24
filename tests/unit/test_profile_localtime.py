"""Local clock positions and day types of the routine model (R-ACT-001, R-ACT-002)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from twin.config.settings import TimeConfig, TzRange
from twin.ingest.times import SourceTime
from twin.llm.pricing import PeakCalendar
from twin.profile.localtime import (
    LocalClock,
    format_minute,
    parse_clock,
    slot_of_minute,
)
from twin.schedule.daytype import DayTypeCalendar


def clock_for(zone: str = "America/Chicago", ranges: list[TzRange] | None = None) -> LocalClock:
    config = TimeConfig(source_timezone=zone, source_timezone_ranges=ranges or [])
    return LocalClock(SourceTime.from_config(config))


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


def test_slots_are_quarter_hours_of_the_local_clock() -> None:
    assert slot_of_minute(0) == 0 and slot_of_minute(14.99) == 0 and slot_of_minute(15) == 1
    assert slot_of_minute(23 * 60 + 59) == 95
    assert format_minute(8 * 60 + 30) == "08:30" and format_minute(1500) == "01:00"
    assert parse_clock("8:05") == 485 and parse_clock("23:59") == 1439
    for bad in ("24:00", "8", "ab:cd", "12:60", "-1:00"):
        with pytest.raises(ValueError, match="clock time"):
            parse_clock(bad)


def test_utc_is_converted_to_the_local_wall_clock() -> None:
    stamp = clock_for().stamp(utc(2026, 8, 5, 13, 45, 30))  # CDT, UTC-5
    assert stamp.day == date(2026, 8, 5) and stamp.zone == "America/Chicago"
    assert stamp.slot == 35 and stamp.minute == pytest.approx(8 * 60 + 45.5)
    winter = clock_for().stamp(utc(2026, 1, 5, 13, 45))  # CST, UTC-6
    assert winter.slot == 31 and winter.minute == 7 * 60 + 45


def test_a_message_just_after_midnight_belongs_to_the_next_local_day() -> None:
    stamp = clock_for().stamp(utc(2026, 8, 5, 5, 10))  # 00:10 CDT
    assert stamp.day == date(2026, 8, 5) and stamp.slot == 0
    before = clock_for().stamp(utc(2026, 8, 5, 4, 50))  # 23:50 the day before
    assert before.day == date(2026, 8, 4) and before.slot == 95
    assert stamp.wall - before.wall == pytest.approx(20.0)


def test_spring_forward_in_chicago_skips_the_missing_hour() -> None:
    # 2026-03-08: 02:00 CST becomes 03:00 CDT, at 08:00 UTC
    clock = clock_for()
    last_before = clock.stamp(utc(2026, 3, 8, 7, 59))
    first_after = clock.stamp(utc(2026, 3, 8, 8, 0))
    assert (last_before.slot, first_after.slot) == (7, 12)  # 01:59 then 03:00
    assert first_after.minute == 3 * 60 and first_after.day == date(2026, 3, 8)


def test_fall_back_in_chicago_shows_the_repeated_hour_twice() -> None:
    # 2026-11-01: 02:00 CDT becomes 01:00 CST, at 07:00 UTC
    clock = clock_for()
    first = clock.stamp(utc(2026, 11, 1, 6, 30))  # 01:30 CDT
    second = clock.stamp(utc(2026, 11, 1, 7, 30))  # 01:30 CST
    assert first.slot == second.slot == 6
    assert clock.stamp(utc(2026, 11, 1, 8, 30)).slot == 10  # 02:30 CST


def test_zone_ranges_switch_the_clock_by_date() -> None:
    ranges = [TzRange(from_=date(2026, 8, 10), to=date(2026, 8, 20), tz="Asia/Shanghai")]
    clock = clock_for("America/Chicago", ranges)
    inside = clock.stamp(utc(2026, 8, 12, 0, 30))  # 08:30 in Shanghai (UTC+8)
    assert inside.zone == "Asia/Shanghai" and inside.day == date(2026, 8, 12)
    assert inside.minute == 8 * 60 + 30
    outside = clock.stamp(utc(2026, 8, 25, 0, 30))  # 19:30 the day before in Chicago
    assert outside.zone == "America/Chicago" and outside.day == date(2026, 8, 24)
    assert outside.minute == 19 * 60 + 30


# --------------------------------------------------------------------- day types


def test_chinese_calendar_decides_for_china_zones() -> None:
    calendar = DayTypeCalendar()
    shanghai = "Asia/Shanghai"
    assert calendar.day_type(date(2026, 10, 1), shanghai) == "holiday"  # National Day
    assert calendar.day_type(date(2026, 10, 4), shanghai) == "holiday"  # a Sunday inside it
    assert calendar.day_type(date(2026, 10, 10), shanghai) == "workday"  # a working Saturday
    assert calendar.day_type(date(2026, 10, 17), shanghai) == "weekend"
    assert calendar.day_type(date(2026, 10, 19), shanghai) == "workday"
    assert calendar.day_type(date(2026, 10, 19), "America/Chicago") == "workday"
    assert calendar.day_type(date(2026, 10, 1), "America/Chicago") == "workday"  # US: no holiday


def test_us_federal_holidays_decide_for_us_zones() -> None:
    calendar = DayTypeCalendar()
    chicago = "America/Chicago"
    assert calendar.day_type(date(2026, 9, 7), chicago) == "holiday"  # Labor Day
    assert calendar.day_type(date(2026, 7, 3), chicago) == "holiday"  # July 4 observed
    assert calendar.day_type(date(2026, 9, 5), chicago) == "weekend"
    assert calendar.day_type(date(2026, 9, 8), chicago) == "workday"
    assert calendar.days_between(date(2026, 9, 1), date(2026, 9, 14), chicago) == {
        "workday": 9,
        "weekend": 4,
        "holiday": 1,  # Labor Day only
    }


def test_other_zones_use_the_weekday_only() -> None:
    calendar = DayTypeCalendar()
    assert calendar.day_type(date(2026, 12, 25), "Europe/Paris") == "workday"
    assert calendar.day_type(date(2026, 12, 26), "Europe/Paris") == "weekend"
    configured = DayTypeCalendar(zone_countries={"Europe/Paris": "US"})
    assert configured.day_type(date(2026, 12, 25), "Europe/Paris") == "holiday"


def test_years_the_library_does_not_know_fall_back_to_weekdays_like_the_price_table() -> None:
    fallbacks: list[int] = []
    calendar = DayTypeCalendar(on_fallback=fallbacks.append)
    assert calendar.day_type(date(2031, 10, 1), "Asia/Shanghai") == "workday"  # a Wednesday
    assert calendar.day_type(date(2031, 10, 4), "Asia/Shanghai") == "weekend"
    assert fallbacks == [2031]  # the same one-warning-per-year fallback as PeakCalendar


def test_the_calendar_is_the_round_one_module() -> None:
    seen: list[date] = []

    def workday(day: date) -> bool:
        seen.append(day)
        return day.day % 2 == 0

    custom = PeakCalendar(workday=workday)
    calendar = DayTypeCalendar(peak_calendar=custom)
    assert calendar.day_type(date(2026, 3, 4), "Asia/Shanghai") == "workday"
    assert calendar.day_type(date(2026, 3, 5), "Asia/Shanghai") == "weekend"
    assert seen == [date(2026, 3, 4), date(2026, 3, 5)]
    assert calendar.with_holiday_ranges([]).day_type(date(2026, 3, 4), "Asia/Shanghai") == "workday"


def test_manual_holiday_ranges_win_over_the_calendar() -> None:
    calendar = DayTypeCalendar(holiday_ranges=[(date(2026, 8, 10), date(2026, 8, 12))])
    assert calendar.day_type(date(2026, 8, 11), "America/Chicago") == "holiday"
    assert calendar.day_type(date(2026, 8, 13), "America/Chicago") == "workday"
    extended = DayTypeCalendar().with_holiday_ranges([(date(2026, 8, 13), date(2026, 8, 13))])
    assert extended.day_type(date(2026, 8, 13), "Asia/Shanghai") == "holiday"
    with pytest.raises(ValueError, match="before it starts"):
        DayTypeCalendar(holiday_ranges=[(date(2026, 8, 12), date(2026, 8, 10))])
