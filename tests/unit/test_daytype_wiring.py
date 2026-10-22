"""One day-type calendar for the whole container (R-ACT-002, R-LLM-007).

R17-40a (D-733).  ``pricing.extra_peak_dates`` / ``extra_offpeak_dates`` (a holiday the library
does not know, a make-up working day it has not been told) and the holiday ranges set with
``twin routine add holiday`` decide whether a day is a workday, weekend or holiday.  Three places
built their own ``DayTypeCalendar`` without the container's holiday calendar - the profile build,
the example windows and the life line - so the same day was a workday for the time service and a
weekend for the profile.  They now ask ``day_type_calendar_for(services)``, the one the kit is made
of.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from tests.support.clock import ManualClock
from tests.support.embedding import HashingBackend
from tests.support.memory import make_memory
from twin.config.runtime import BOT_TIMEZONE
from twin.llm.pricing import PeakCalendar
from twin.memory.lifeline import LifelineStore
from twin.profile.builder import BuildContext
from twin.profile.overrides import RoutineOverrides
from twin.retrieval.windows import LocalPlace
from twin.schedule.daytype import DayTypeCalendar
from twin.schedule.service import peak_calendar_for, schedule_kit
from twin.services import Services

SHANGHAI = "Asia/Shanghai"
WEDNESDAY = date(2026, 6, 17)  # an ordinary working day, which the owner declares a day off
SATURDAY = date(2026, 7, 4)  # an ordinary Saturday, which the owner declares a working day
SUNDAY = date(2026, 7, 5)  # the next day, plain
HOLIDAY_FIRST, HOLIDAY_LAST = date(2026, 8, 10), date(2026, 8, 12)  # Mon-Wed, set by hand
DAYS = (WEDNESDAY, SATURDAY, SUNDAY, date(2026, 8, 11), date(2026, 8, 13), date(2026, 6, 18))


@pytest.fixture
def configured(services: Services, clock: ManualClock) -> Services:
    """A container whose owner told the price table about two days, and the routine about three."""
    pricing = services.settings.pricing
    pricing.extra_offpeak_dates = [WEDNESDAY]
    pricing.extra_peak_dates = [SATURDAY]
    services.settings.time.source_timezone = SHANGHAI  # her records were made in China
    RoutineOverrides(services.db, clock).add_holiday(HOLIDAY_FIRST, HOLIDAY_LAST)
    services.runtime.set(BOT_TIMEZONE, SHANGHAI, by="test")
    return services


def expected(services: Services) -> dict[date, str]:
    """What the kit's time service says (the reference every other place must agree with)."""
    kit = schedule_kit(services)
    return {day: kit.time.day_type(day) for day in DAYS}


def test_the_reference_knows_the_extra_dates_and_the_holiday_range(configured: Services) -> None:
    reference = expected(configured)
    assert reference[WEDNESDAY] == "weekend"  # declared off-peak: not a working day
    assert reference[SATURDAY] == "workday"  # declared peak: a working day
    assert reference[SUNDAY] == "weekend"
    assert reference[date(2026, 8, 11)] == "holiday"  # inside the range set by hand
    assert reference[date(2026, 8, 13)] == "workday"  # the day after it
    assert reference[date(2026, 6, 18)] == "workday"


def test_day_type_calendar_for_is_what_the_kit_is_made_of(configured: Services) -> None:
    from twin.schedule.service import day_type_calendar_for

    calendar = day_type_calendar_for(configured)
    assert isinstance(calendar, DayTypeCalendar)
    assert {day: calendar.day_type(day, SHANGHAI) for day in DAYS} == expected(configured)
    # the holiday calendar is the container's one, with its extra dates and its fallback alert
    assert calendar._china is peak_calendar_for(configured)  # type: ignore[attr-defined]
    assert isinstance(calendar._china, PeakCalendar)  # type: ignore[attr-defined]
    # a new one on every call: it holds the ranges as they are now
    RoutineOverrides(configured.db, configured.clock).add_holiday(
        date(2026, 9, 1), date(2026, 9, 1)
    )
    assert day_type_calendar_for(configured).day_type(date(2026, 9, 1), SHANGHAI) == "holiday"
    assert calendar.day_type(date(2026, 9, 1), SHANGHAI) == "workday"


def test_the_profile_build_agrees_with_the_time_service(configured: Services) -> None:
    """R17-40a: ``BuildContext.create`` made a calendar without the extra dates."""
    context = BuildContext.create(configured)
    assert {day: context.calendar.day_type(day, SHANGHAI) for day in DAYS} == expected(configured)


def test_the_example_windows_agree_with_the_time_service(configured: Services) -> None:
    """R17-40a: ``LocalPlace.create`` stamped windows with a day type the schedule would not."""
    place = LocalPlace.create(configured)
    for day, kind in expected(configured).items():
        noon = datetime(day.year, day.month, day.day, 4, 0, tzinfo=UTC)  # 12:00 in Shanghai
        assert place.of(noon) == (48, kind), day
        assert place.stamp(noon).day == day


def test_the_life_line_agrees_with_the_time_service(
    configured: Services, embedder: HashingBackend
) -> None:
    """R17-40a: ``LifelineStore`` without a time service classified days with a bare calendar."""
    store = LifelineStore(make_memory(configured))
    for day, kind in expected(configured).items():
        today, tomorrow = store._day_types(day)  # type: ignore[attr-defined]
        assert today == kind, day
        assert tomorrow == schedule_kit(configured).time.day_type(
            date.fromordinal(day.toordinal() + 1)
        )


def test_a_us_zone_ignores_the_chinese_extras_but_still_has_its_ranges(
    services: Services, clock: ManualClock
) -> None:
    """The extra dates are the price table's, i.e. Beijing's: a Chicago day is not moved by them."""
    from twin.schedule.service import day_type_calendar_for

    services.settings.pricing.extra_offpeak_dates = [WEDNESDAY]
    RoutineOverrides(services.db, clock).add_holiday(HOLIDAY_FIRST, HOLIDAY_LAST)
    calendar = day_type_calendar_for(services)
    assert calendar.day_type(WEDNESDAY, "America/Chicago") == "workday"
    assert calendar.day_type(date(2026, 8, 11), "America/Chicago") == "holiday"
    assert calendar.day_type(WEDNESDAY, SHANGHAI) == "weekend"
