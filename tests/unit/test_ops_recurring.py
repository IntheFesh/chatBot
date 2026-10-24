"""When something that repeats was last due, on a time zone's clock (R-OPS-003/005/006)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from twin.ops.recurring import Daily, Monthly, Weekly, is_due, latest_due

CHICAGO = ZoneInfo("America/Chicago")
SHANGHAI = ZoneInfo("Asia/Shanghai")


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


def test_the_daily_moment_is_today_or_yesterday() -> None:
    rule = Daily(4)
    # 2026-10-09 05:00 Chicago (CDT, UTC-5) = 10:00 UTC: today's 04:00 has passed
    assert latest_due(rule, utc(2026, 10, 9, 10), CHICAGO) == utc(2026, 10, 9, 9)
    # 03:30 Chicago: it is still yesterday's 04:00
    assert latest_due(rule, utc(2026, 10, 9, 8, 30), CHICAGO) == utc(2026, 10, 8, 9)
    # exactly on the moment counts
    assert latest_due(rule, utc(2026, 10, 9, 9), CHICAGO) == utc(2026, 10, 9, 9)


def test_the_moment_follows_the_zone_of_the_bot() -> None:
    now = utc(2026, 10, 9, 10)
    assert latest_due(Daily(4), now, SHANGHAI) == utc(2026, 10, 8, 20)  # 04:00 Beijing = 20:00 UTC
    assert latest_due(Daily(4), now, CHICAGO) != latest_due(Daily(4), now, SHANGHAI)


def test_a_daylight_saving_change_moves_the_utc_time_not_the_wall_clock() -> None:
    # Chicago falls back on 2026-11-01 at 02:00 CDT (07:00 UTC): 04:00 is 09:00 UTC
    # before the change and 10:00 UTC after it
    before = latest_due(Daily(4), utc(2026, 10, 31, 12), CHICAGO)
    after = latest_due(Daily(4), utc(2026, 11, 1, 12), CHICAGO)
    assert before == utc(2026, 10, 31, 9)
    assert after == utc(2026, 11, 1, 10)
    assert after - before == timedelta(hours=25)  # the day of the change has 25 hours


def test_a_moment_inside_the_spring_gap_is_judged_as_an_instant() -> None:
    # 02:30 does not exist in Chicago on 2026-03-08 (the clocks jump from 02:00 CST to 03:00 CDT);
    # the moment is 08:30 UTC, and at 08:10 UTC (03:10 CDT) it has not come yet
    rule = Daily(2, 30)
    assert latest_due(rule, utc(2026, 3, 8, 8, 10), CHICAGO) == utc(2026, 3, 7, 8, 30)
    assert latest_due(rule, utc(2026, 3, 8, 8, 31), CHICAGO) == utc(2026, 3, 8, 8, 30)


def test_the_weekly_moment_is_the_last_sunday_half_past_three() -> None:
    rule = Weekly(weekday=6, hour=3, minute=30)
    # Friday 2026-10-09 noon UTC: the last Sunday was 2026-10-04, 03:30 CDT = 08:30 UTC
    assert latest_due(rule, utc(2026, 10, 9, 12), CHICAGO) == utc(2026, 10, 4, 8, 30)
    # Sunday 03:00 Chicago: not yet, still last week's
    assert latest_due(rule, utc(2026, 10, 11, 8, 0), CHICAGO) == utc(2026, 10, 4, 8, 30)
    assert latest_due(rule, utc(2026, 10, 11, 8, 30), CHICAGO) == utc(2026, 10, 11, 8, 30)


def test_the_monthly_moment_is_the_first_at_nine() -> None:
    rule = Monthly(1, 9)
    assert latest_due(rule, utc(2026, 10, 9, 12), CHICAGO) == utc(2026, 10, 1, 14)
    # the morning of the 1st before 09:00 local: the previous month's 1st
    assert latest_due(rule, utc(2026, 10, 1, 13), CHICAGO) == utc(2026, 9, 1, 14)
    # across a year
    assert latest_due(rule, utc(2027, 1, 1, 10), CHICAGO) == utc(2026, 12, 1, 15)


def test_a_day_that_a_month_does_not_have_is_its_last_day() -> None:
    rule = Monthly(31, 9)
    assert latest_due(rule, utc(2026, 3, 5, 12), CHICAGO) == utc(2026, 2, 28, 15)


@pytest.mark.parametrize(
    ("last_run", "due"),
    [
        (None, True),
        (utc(2026, 10, 9, 8), True),  # before today's 04:00 Chicago (09:00 UTC)
        (utc(2026, 10, 9, 9), False),  # served exactly then
        (utc(2026, 10, 9, 11), False),
    ],
)
def test_a_moment_is_served_once(last_run: datetime | None, due: bool) -> None:
    assert is_due(Daily(4), utc(2026, 10, 9, 12), CHICAGO, last_run) is due
