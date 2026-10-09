"""Reading Chinese date and time phrases against the moment of the conversation (R-MEM-006/007)."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from twin.memory.timeparse import (
    PERIOD_DEFAULTS,
    parse_iso_date,
    parse_local_iso,
    parse_number,
    parse_when,
)

CHICAGO = ZoneInfo("America/Chicago")
BEIJING = ZoneInfo("Asia/Shanghai")
# Wednesday 4 March 2026, 20:30 in Chicago (standard time, UTC-6)
REFERENCE = datetime(2026, 3, 4, 20, 30, tzinfo=CHICAGO)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("7", 7),
        ("十", 10),
        ("十一", 11),
        ("二十", 20),
        ("二十三", 23),
        ("三十一", 31),
        ("两", 2),
        ("零", 0),
        ("", None),
        ("百", None),
    ],
)
def test_numbers_are_read_from_digits_and_chinese_numerals(text: str, expected: int | None) -> None:
    assert parse_number(text) == expected


@pytest.mark.parametrize(
    ("phrase", "day", "clock"),
    [
        ("明天下午三点考试", date(2026, 3, 5), time(15, 0)),
        ("后天上午10点半", date(2026, 3, 6), time(10, 30)),
        ("大后天", date(2026, 3, 7), None),
        ("昨天", date(2026, 3, 3), None),
        ("周五晚上八点", date(2026, 3, 6), time(20, 0)),
        ("下周三", date(2026, 3, 11), None),
        ("下下周一", date(2026, 3, 16), None),
        ("这周日下午2点10分", date(2026, 3, 8), time(14, 10)),
        ("上周五", date(2026, 2, 27), None),
        ("周末", date(2026, 3, 7), None),
        ("下周末", date(2026, 3, 14), None),
        ("3月15日", date(2026, 3, 15), None),
        ("十二月二十五号", date(2026, 12, 25), None),
        ("2026年4月1日上午9点", date(2026, 4, 1), time(9, 0)),
        ("2026-04-01 09:00", date(2026, 4, 1), time(9, 0)),
        ("3/9", date(2026, 3, 9), None),
        ("下个月5号", date(2026, 4, 5), None),
        ("三天后", date(2026, 3, 7), None),
        ("两周后", date(2026, 3, 18), None),
        ("一个月后", date(2026, 4, 4), None),
        ("中午1点", date(2026, 3, 4), time(13, 0)),
        ("今晚12点", date(2026, 3, 5), time(0, 0)),
    ],
)
def test_phrases_are_read_against_the_reference_day(
    phrase: str, day: date, clock: time | None
) -> None:
    when = parse_when(phrase, REFERENCE)
    assert when is not None
    assert (when.day, when.clock) == (day, clock)


def test_a_bare_weekday_means_the_next_one_and_a_bare_clock_time_means_today() -> None:
    assert parse_when("周三", REFERENCE).day == date(2026, 3, 11)  # type: ignore[union-attr]
    assert parse_when("周四", REFERENCE).day == date(2026, 3, 5)  # type: ignore[union-attr]
    only_time = parse_when("15:30", REFERENCE)
    assert only_time is not None
    assert (only_time.day, only_time.clock, only_time.day_given) == (
        date(2026, 3, 4),
        time(15, 30),
        False,
    )


@pytest.mark.parametrize(
    ("phrase", "hour"),
    [
        ("三点", 15),
        ("八点", 8),
        ("十二点", 12),
        ("下午三点", 15),
        ("晚上八点", 20),
        ("早上八点", 8),
    ],
)
def test_an_hour_without_a_part_of_the_day_is_read_the_way_people_mean_it(
    phrase: str, hour: int
) -> None:
    when = parse_when(phrase, REFERENCE)
    assert when is not None and when.clock is not None and when.clock.hour == hour


def test_a_part_of_the_day_alone_has_a_usual_time_and_the_word_for_tonight_implies_one() -> None:
    evening = parse_when("明晚", REFERENCE)
    assert evening is not None
    assert (evening.day, evening.clock, evening.period) == (date(2026, 3, 5), None, "晚上")
    assert evening.at(CHICAGO, default=time(9, 0)).time() == PERIOD_DEFAULTS["晚上"]
    day_only = parse_when("下周三", REFERENCE)
    assert day_only is not None
    assert day_only.at(CHICAGO, default=time(9, 0)).time() == time(9, 0)
    assert day_only.has_time is False


def test_text_without_a_time_is_not_a_time() -> None:
    assert parse_when("今天天气不错吧", REFERENCE) is not None  # "今天" is a day
    assert parse_when("没有时间", REFERENCE) is None
    assert parse_when("", REFERENCE) is None


def test_the_reference_must_have_a_time_zone() -> None:
    with pytest.raises(ValueError, match="time zone"):
        parse_when("明天", datetime(2026, 3, 4, 12, 0))  # noqa: DTZ001 - the point of the test


def test_tomorrow_afternoon_is_a_different_instant_in_chicago_and_in_beijing() -> None:
    """The same message instant, the same words: the local day and clock decide (R-MEM-007)."""
    said_at = datetime(
        2026, 3, 5, 2, 30, tzinfo=UTC
    )  # 20:30 on 4 March in Chicago, 10:30 on 5 March in Beijing
    chicago = parse_when("明天下午三点", said_at.astimezone(CHICAGO))
    beijing = parse_when("明天下午三点", said_at.astimezone(BEIJING))
    assert chicago is not None and beijing is not None
    assert chicago.day == date(2026, 3, 5) and beijing.day == date(2026, 3, 6)
    due_chicago = chicago.at(CHICAGO, default=time(9, 0)).astimezone(UTC)
    due_beijing = beijing.at(BEIJING, default=time(9, 0)).astimezone(UTC)
    assert due_chicago == datetime(2026, 3, 5, 21, 0, tzinfo=UTC)
    assert due_beijing == datetime(2026, 3, 6, 7, 0, tzinfo=UTC)
    assert due_beijing - due_chicago == timedelta(hours=10)


def test_the_daylight_saving_change_is_respected() -> None:
    """Sunday 8 March 2026 is the day Chicago changes to daylight time."""
    saturday = datetime(2026, 3, 7, 12, 0, tzinfo=CHICAGO)
    when = parse_when("明天下午2点", saturday)
    assert when is not None
    assert when.at(CHICAGO, default=time(9, 0)).astimezone(UTC) == datetime(
        2026, 3, 8, 19, 0, tzinfo=UTC
    )  # UTC-5 after the change


def test_a_date_far_in_the_past_this_year_means_next_year() -> None:
    november = datetime(2026, 11, 20, 12, 0, tzinfo=CHICAGO)
    when = parse_when("3月5号", november)
    assert when is not None and when.day == date(2027, 3, 5)
    recent = parse_when("11月1号", november)
    assert recent is not None and recent.day == date(2026, 11, 1)  # a few weeks back stays


def test_impossible_dates_are_not_dates() -> None:
    assert parse_when("2月30日", REFERENCE) is None


def test_iso_helpers() -> None:
    assert parse_iso_date("2026-03-05") == date(2026, 3, 5)
    assert parse_iso_date("2026-03-05 10:00") == date(2026, 3, 5)
    assert parse_iso_date("soon") is None
    parsed = parse_local_iso("2026-03-05 15:00", CHICAGO)
    assert parsed is not None and parsed.utcoffset() is not None
    assert parsed.astimezone(UTC) == datetime(2026, 3, 5, 21, 0, tzinfo=UTC)
    assert parse_local_iso("2026-03-05T15:00", BEIJING) == datetime(
        2026, 3, 5, 15, 0, tzinfo=BEIJING
    )
    assert parse_local_iso("2026-03-05", CHICAGO) == datetime(2026, 3, 5, 0, 0, tzinfo=CHICAGO)
    zoned = parse_local_iso("2026-03-05T15:00+00:00", CHICAGO)
    assert zoned is not None and zoned.astimezone(UTC) == datetime(2026, 3, 5, 15, 0, tzinfo=UTC)
    assert parse_local_iso("tomorrow", CHICAGO) is None
