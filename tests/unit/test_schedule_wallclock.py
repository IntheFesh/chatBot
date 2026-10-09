"""Clock times and instants across daylight saving time (R-SCH-003).

Chicago changes its clock on 2026-03-08 (02:00 becomes 03:00, a 23-hour day) and on 2026-11-01
(02:00 becomes 01:00, a 25-hour day).  The rules are written in ``twin.schedule.wallclock``: a time
that does not exist is read with the offset before the change (an hour later in real time), a time
that exists twice is the first one, and the states of a day never leave a hole or overlap.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from zoneinfo import ZoneInfo

from hypothesis import given
from hypothesis import strategies as st

from twin.schedule.wallclock import (
    WallKind,
    carry,
    clock_text,
    day_bounds_utc,
    day_length,
    day_start_utc,
    minute_of_day,
    next_transition,
    resolve,
    to_utc,
)

CHICAGO = ZoneInfo("America/Chicago")
SHANGHAI = ZoneInfo("Asia/Shanghai")
SPRING = date(2026, 3, 8)
AUTUMN = date(2026, 11, 1)


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


def test_an_ordinary_day_maps_clock_times_one_to_one() -> None:
    found = resolve(date(2026, 10, 9), 7 * 60 + 30, CHICAGO)  # CDT, UTC-5
    assert found.kind is WallKind.NORMAL
    assert found.utc == utc(2026, 10, 9, 12, 30)
    assert found.reading(CHICAGO).strftime("%H:%M") == "07:30"
    assert to_utc(date(2026, 10, 9), 7 * 60 + 30, SHANGHAI) == utc(2026, 10, 8, 23, 30)


def test_a_time_that_does_not_exist_is_read_with_the_offset_before_the_change() -> None:
    """02:30 on 2026-03-08 is never shown: it happens at 03:30, an hour later, never earlier."""
    found = resolve(SPRING, 2 * 60 + 30, CHICAGO)
    assert found.kind is WallKind.GAP
    assert found.utc == utc(2026, 3, 8, 8, 30)  # 02:30 CST
    assert found.reading(CHICAGO).strftime("%H:%M") == "03:30"
    before = resolve(SPRING, 1 * 60 + 59, CHICAGO)
    after = resolve(SPRING, 3 * 60, CHICAGO)
    assert before.kind is WallKind.NORMAL and after.kind is WallKind.NORMAL
    assert before.utc == utc(2026, 3, 8, 7, 59) and after.utc == utc(2026, 3, 8, 8, 0)
    edges = [resolve(SPRING, m, CHICAGO).kind for m in (120, 150, 179, 180)]
    assert edges == [WallKind.GAP, WallKind.GAP, WallKind.GAP, WallKind.NORMAL]
    # two times inside the gap keep their distance
    first, second = to_utc(SPRING, 2 * 60 + 10, CHICAGO), to_utc(SPRING, 2 * 60 + 40, CHICAGO)
    assert second - first == timedelta(minutes=30)


def test_a_time_that_exists_twice_is_the_first_one() -> None:
    """01:30 on 2026-11-01 is shown twice; the plan uses the first (still daylight time)."""
    found = resolve(AUTUMN, 90, CHICAGO)
    assert found.kind is WallKind.REPEATED
    assert found.utc == utc(2026, 11, 1, 6, 30)  # 01:30 CDT, not 07:30 UTC (01:30 CST)
    assert found.reading(CHICAGO).strftime("%H:%M %Z") == "01:30 CDT"
    assert resolve(AUTUMN, 59, CHICAGO).kind is WallKind.NORMAL  # 00:59
    assert resolve(AUTUMN, 60, CHICAGO).kind is WallKind.REPEATED
    assert resolve(AUTUMN, 119, CHICAGO).kind is WallKind.REPEATED
    assert resolve(AUTUMN, 120, CHICAGO).kind is WallKind.NORMAL  # 02:00 happens once (CST)
    assert to_utc(AUTUMN, 120, CHICAGO) == utc(2026, 11, 1, 8, 0)


def test_the_days_of_the_change_are_23_and_25_hours_long() -> None:
    assert day_length(SPRING, CHICAGO) == timedelta(hours=23)
    assert day_length(AUTUMN, CHICAGO) == timedelta(hours=25)
    assert day_length(date(2026, 10, 9), CHICAGO) == timedelta(hours=24)
    assert day_length(SPRING, SHANGHAI) == timedelta(hours=24)
    start, end = day_bounds_utc(AUTUMN, CHICAGO)
    assert (start, end) == (utc(2026, 11, 1, 5, 0), utc(2026, 11, 2, 6, 0))
    assert day_start_utc(AUTUMN + timedelta(days=1), CHICAGO) == end


def test_minutes_beyond_a_day_carry_into_the_date() -> None:
    assert carry(date(2026, 10, 9), -30) == (date(2026, 10, 8), 1410)
    assert carry(date(2026, 10, 9), 1500) == (date(2026, 10, 10), 60)
    assert carry(date(2026, 10, 9), 0) == (date(2026, 10, 9), 0)
    found = resolve(date(2026, 10, 9), -30, CHICAGO)
    assert found.day == date(2026, 10, 8) and found.minute == 1410
    assert found.utc == utc(2026, 10, 9, 4, 30)


def test_a_midnight_that_does_not_exist_is_the_moment_of_the_change() -> None:
    """Havana changes its clock at midnight: the day starts when the clock jumps."""
    havana = ZoneInfo("America/Havana")
    start = day_start_utc(date(2026, 3, 8), havana)
    assert start.astimezone(havana).strftime("%H:%M") == "01:00"
    previous_end = day_bounds_utc(date(2026, 3, 7), havana)[1]
    assert previous_end == start


def test_the_wall_clock_helpers_read_the_instant_back() -> None:
    moment = utc(2026, 11, 1, 7, 30)  # 01:30 CST, the second pass through the hour
    assert clock_text(moment, CHICAGO) == "01:30"
    assert minute_of_day(moment, CHICAGO) == 90
    assert minute_of_day(utc(2026, 10, 9, 12, 30, 30), CHICAGO) == 7 * 60 + 30.5


@given(
    day=st.dates(min_value=date(2026, 1, 1), max_value=date(2027, 12, 31)),
    minutes=st.lists(st.integers(min_value=-300, max_value=2000), min_size=2, max_size=8),
)
def test_instants_never_run_backwards_when_the_clock_time_grows(
    day: date, minutes: list[int]
) -> None:
    ordered = sorted(minutes)
    for zone in (CHICAGO, SHANGHAI, ZoneInfo("Europe/London")):
        instants = [to_utc(day, minute, zone) for minute in ordered]
        assert instants == sorted(instants)


def test_states_of_a_day_tile_it_exactly_on_the_days_of_the_change() -> None:
    """Boundaries given in clock times: no hole, no overlap, 23 and 25 hours in total."""
    for day, hours in ((SPRING, 23), (AUTUMN, 25), (date(2026, 10, 9), 24)):
        marks = [0, 90, 150, 480, 840, 1320, 1440]  # clock times of state changes
        instants = [to_utc(day, minute, CHICAGO) for minute in marks]
        spans = list(pairwise(instants))
        assert all(end >= start for start, end in spans)
        assert sum((end - start for start, end in spans), timedelta()) == timedelta(hours=hours)
        assert spans[0][0] == day_bounds_utc(day, CHICAGO)[0]
        assert spans[-1][1] == day_bounds_utc(day, CHICAGO)[1]


def test_the_next_clock_change_is_found_to_the_minute() -> None:
    change = next_transition(CHICAGO, utc(2026, 10, 9))
    assert change is not None
    assert change.at == utc(2026, 11, 1, 7, 0)  # 02:00 CDT
    assert (change.before, change.after) == ("CDT", "CST")
    assert change.shift == timedelta(hours=-1)
    spring = next_transition(CHICAGO, utc(2026, 11, 1, 7, 0))
    assert spring is not None and spring.at == utc(2027, 3, 14, 8, 0)
    assert spring.shift == timedelta(hours=1)
    assert next_transition(SHANGHAI, utc(2026, 10, 9)) is None
