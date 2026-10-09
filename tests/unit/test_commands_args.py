"""Reading the arguments of the commands: durations, clock ranges, dates, zones, quotas (R-CMD-003).

The rules are pure functions of their text (and of the moment they are told), so they are tested
without a chat; the commands that use them are tested in ``test_commands_routine.py``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from twin.commands.args import (
    ArgumentError,
    cn_number,
    parse_clock_range,
    parse_date_range,
    parse_one_date,
    parse_pause,
    parse_quota,
    resolve_zone,
    split_range,
)

CHICAGO = ZoneInfo("America/Chicago")


def at(hour: int, minute: int = 0, *, day: int = 9) -> datetime:
    """A moment on the clock of Chicago (Friday 9 October 2026 by default)."""
    return datetime(2026, 10, day, hour, minute, tzinfo=CHICAGO)


def pause_end(text: str, now: datetime) -> datetime:
    return parse_pause(text, now).until.astimezone(CHICAGO)


# -------------------------------------------------------------------------- numbers


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("2", 2.0),
        ("2.5", 2.5),
        ("两", 2.0),
        ("十", 10.0),
        ("十二", 12.0),
        ("三十五", 35.0),
        ("二十", 20.0),
        ("零", 0.0),
        ("九", 9.0),
    ],
)
def test_chinese_and_arabic_numbers_are_read(text: str, value: float) -> None:
    assert cn_number(text) == value


@pytest.mark.parametrize("text", ["", "abc", "百", "十十", "二三", "三十五六"])
def test_what_is_not_a_small_number_is_none(text: str) -> None:
    assert cn_number(text) is None


# --------------------------------------------------------------------------- pausing


@pytest.mark.parametrize(
    ("text", "minutes"),
    [
        ("2小时", 120),
        ("２小时", 120),  # full-width digit
        ("2 小时", 120),
        ("两小时", 120),
        ("2个小时", 120),
        ("半小时", 30),
        ("半个小时", 30),
        ("一个半小时", 90),
        ("2.5小时", 150),
        ("30分钟", 30),
        ("三十分钟", 30),
        ("45分", 45),
        ("90min", 90),
        ("2h", 120),
        ("2小时30分钟", 150),
        ("1天", 1440),
        ("半天", 720),
    ],
)
def test_a_length_of_time_ends_that_long_after_now(text: str, minutes: int) -> None:
    now = at(10)
    assert parse_pause(text, now).until == now.astimezone(UTC) + timedelta(minutes=minutes)


def test_a_pause_is_an_absolute_utc_moment_across_the_change_of_the_clock() -> None:
    now = datetime(2026, 11, 1, 0, 30, tzinfo=CHICAGO)  # the clocks go back at 02:00 that night
    until = parse_pause("3小时", now).until
    assert until.tzinfo is UTC and until - now.astimezone(UTC) == timedelta(hours=3)


def test_until_tomorrow_morning_is_the_next_morning_hour() -> None:
    assert pause_end("到明早", at(22, 30)) == at(8, day=10)
    assert pause_end("到明天早上", at(22, 30)) == at(8, day=10)
    assert pause_end("到早上", at(22, 30)) == at(8, day=10)
    # after midnight the morning that is coming is the one of the same calendar day
    assert pause_end("到明早", at(1, 30)) == at(8)
    # exactly at the hour it is the next one
    assert pause_end("到明早", at(8)) == at(8, day=10)


def test_the_morning_hour_is_a_setting() -> None:
    assert parse_pause("到明早", at(22), morning_hour=9).until == at(9, day=10).astimezone(UTC)


@pytest.mark.parametrize(
    ("text", "hour", "minute"),
    [
        ("到22:00", 22, 0),
        ("到22：00", 22, 0),
        ("到晚上10点", 22, 0),
        ("到下午3点", 15, 0),
        ("到10点半", 10, 30),
        ("到9点30分", 9, 30),
        ("到中午12点", 12, 0),
        ("到凌晨1点", 1, 0),
        ("到早上9点", 9, 0),
    ],
)
def test_until_a_time_of_day_is_the_next_time_the_clock_shows_it(
    text: str, hour: int, minute: int
) -> None:
    ended = pause_end(text, at(11))
    assert (ended.hour, ended.minute) == (hour, minute)
    assert ended > at(11) and ended - at(11) <= timedelta(days=1)


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("", "unreadable"),
        ("很久", "unreadable"),
        ("30", "unreadable"),
        ("到", "unreadable"),
        ("到25:00", "unreadable"),
        ("到明晚八点半以后", "unreadable"),
        ("0分钟", "not_future"),
        ("999小时", "too_long"),
        ("8天", "too_long"),
    ],
)
def test_a_pause_that_cannot_be_read_or_is_not_allowed_says_why(text: str, code: str) -> None:
    with pytest.raises(ArgumentError) as caught:
        parse_pause(text, at(10))
    assert str(caught.value) == code


def test_the_longest_pause_is_a_setting() -> None:
    assert parse_pause("48小时", at(10), max_hours=48).until
    with pytest.raises(ArgumentError, match="too_long"):
        parse_pause("49小时", at(10), max_hours=48)


# ----------------------------------------------------------------- clock ranges, dates


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("01:00-09:30", "01:00", "09:30"),
        ("1:00-9:30", "01:00", "09:30"),
        ("01：00～09：00", "01:00", "09:00"),
        ("23:30至07:30", "23:30", "07:30"),
        ("23:30到7:30", "23:30", "07:30"),
        ("  13:00 - 17:00 ", "13:00", "17:00"),
        ("下午1点到晚上9点", "13:00", "21:00"),
        ("13点-17点半", "13:00", "17:30"),
    ],
)
def test_a_clock_range_is_two_times_in_24_hour_form(text: str, start: str, end: str) -> None:
    assert parse_clock_range(text) == (start, end)


@pytest.mark.parametrize("text", ["", "01:00", "01:00-", "-09:00", "25:00-26:00", "a-b", "1-2-3"])
def test_a_bad_clock_range_is_refused(text: str) -> None:
    with pytest.raises(ArgumentError):
        parse_clock_range(text)


def test_a_range_splits_on_the_usual_dashes() -> None:
    assert split_range("a–b") == ("a", "b")
    assert split_range("a～b") == ("a", "b")


TODAY = date(2026, 10, 9)


@pytest.mark.parametrize(
    ("text", "first", "last"),
    [
        ("2026-10-01", date(2026, 10, 1), date(2026, 10, 1)),
        ("2026/10/1", date(2026, 10, 1), date(2026, 10, 1)),
        ("2026年10月1日", date(2026, 10, 1), date(2026, 10, 1)),
        ("2026-10-01..2026-10-07", date(2026, 10, 1), date(2026, 10, 7)),
        ("2026-10-01 ~ 2026-10-07", date(2026, 10, 1), date(2026, 10, 7)),
        ("10月1日至10月7日", date(2026, 10, 1), date(2026, 10, 7)),
        ("10/1到10/7", date(2026, 10, 1), date(2026, 10, 7)),
        ("12月24日", date(2026, 12, 24), date(2026, 12, 24)),
    ],
)
def test_a_holiday_is_one_date_or_two(text: str, first: date, last: date) -> None:
    assert parse_date_range(text, TODAY) == (first, last)


def test_a_date_without_a_year_is_the_next_one_but_a_recent_one_stays_this_year() -> None:
    assert parse_one_date("10月1日", TODAY) == date(2026, 10, 1)  # eight days ago: this year
    assert parse_one_date("1月2日", TODAY) == date(2027, 1, 2)  # long past: next year
    assert parse_one_date("9月20日", TODAY) == date(2026, 9, 20)  # within a month


@pytest.mark.parametrize(
    "text",
    ["", "明天", "2026-13-01", "2026-02-30", "10月", "2026-10-07..2026-10-01", "a..b", "..", "x至"],
)
def test_a_bad_date_is_refused(text: str) -> None:
    with pytest.raises(ArgumentError):
        parse_date_range(text, TODAY)


# --------------------------------------------------------------------- zones, quotas


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("芝加哥", "America/Chicago"),
        ("北京", "Asia/Shanghai"),
        ("上海", "Asia/Shanghai"),
        ("中国", "Asia/Shanghai"),
        ("国内", "Asia/Shanghai"),
        ("北京时间", "Asia/Shanghai"),
        ("纽约", "America/New_York"),
        ("Chicago", "America/Chicago"),
        ("ＢＥＩＪＩＮＧ", "Asia/Shanghai"),
        ("asia/shanghai", "Asia/Shanghai"),
        ("America/CHICAGO", "America/Chicago"),
        (" Asia/Tokyo ", "Asia/Tokyo"),
    ],
)
def test_a_zone_is_an_alias_or_an_iana_name_in_any_case(text: str, name: str) -> None:
    assert resolve_zone(text) == name


@pytest.mark.parametrize("text", ["", "火星", "Mars/Olympus", "../etc/passwd", "UTC+8x"])
def test_an_unknown_zone_is_refused(text: str) -> None:
    with pytest.raises(ArgumentError):
        resolve_zone(text)


@pytest.mark.parametrize(
    ("text", "low", "high"),
    [
        ("2-5", 2, 5),
        ("0-0", 0, 0),
        ("1~6", 1, 6),
        ("３－５", 3, 5),
        ("0-12", 0, 12),
        ("4至4", 4, 4),
    ],
)
def test_a_quota_is_two_numbers_in_order(text: str, low: int, high: int) -> None:
    assert parse_quota(text) == (low, high)


@pytest.mark.parametrize(("text", "code"), [("5-2", "out_of_range"), ("0-13", "out_of_range")])
def test_a_quota_outside_the_limits_says_so(text: str, code: str) -> None:
    with pytest.raises(ArgumentError) as caught:
        parse_quota(text)
    assert str(caught.value) == code


@pytest.mark.parametrize("text", ["", "2", "a-b", "2-", "-5", "2-3-4", "开"])
def test_a_quota_that_is_not_two_numbers_is_unreadable(text: str) -> None:
    with pytest.raises(ArgumentError) as caught:
        parse_quota(text)
    assert str(caught.value) == "unreadable"
