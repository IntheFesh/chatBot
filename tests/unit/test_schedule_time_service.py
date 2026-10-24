"""The time service: the one source of "now" and of the day's type (R-SCH-001, R-SCOPE-005).

Every test injects the clock; the zone is a callable so that a switch of the bot's time zone can be
tested without a database.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from tests.support.clock import ManualClock
from twin.schedule.daytype import DayTypeCalendar
from twin.schedule.time_service import BotTimeService, Cached, PlanUnavailableError

CHICAGO = "America/Chicago"
SHANGHAI = "Asia/Shanghai"


def make(clock: ManualClock, zone: list[str] | None = None, **kwargs: object) -> BotTimeService:
    names = zone if zone is not None else [CHICAGO]
    return BotTimeService(clock, lambda: names[0], **kwargs)  # type: ignore[arg-type]


def test_now_and_the_local_date_follow_the_injected_clock_and_the_current_zone(
    clock: ManualClock,
) -> None:
    clock.set_time(datetime(2026, 10, 9, 3, 30, tzinfo=UTC))  # 22:30 on the 8th in Chicago
    zone = [CHICAGO]
    service = make(clock, zone)
    assert service.now_utc() == datetime(2026, 10, 9, 3, 30, tzinfo=UTC)
    assert service.local_date() == date(2026, 10, 8)
    assert service.now_local().strftime("%Y-%m-%d %H:%M %Z") == "2026-10-08 22:30 CDT"
    zone[0] = SHANGHAI  # the next call sees the new zone
    assert service.local_date() == date(2026, 10, 9)
    assert service.now_local().strftime("%H:%M %Z") == "11:30 CST"
    assert service.bot_timezone().key == SHANGHAI
    assert service.local_date(datetime(2026, 1, 1, 17, 0, tzinfo=UTC)) == date(2026, 1, 2)


def test_a_naive_instant_is_refused(clock: ManualClock) -> None:
    with pytest.raises(ValueError, match="naive datetime"):
        make(clock).local_date(datetime(2026, 10, 9, 12, 0))  # noqa: DTZ001 - the point of the test


def test_day_and_month_bounds_have_the_length_of_the_local_day_and_month(
    clock: ManualClock,
) -> None:
    service = make(clock)
    start, end = service.day_bounds_utc(date(2026, 3, 8))
    assert (start, end) == (
        datetime(2026, 3, 8, 6, 0, tzinfo=UTC),
        datetime(2026, 3, 9, 5, 0, tzinfo=UTC),
    )
    assert end - start == timedelta(hours=23)
    start, end = service.day_bounds_utc(date(2026, 11, 1))
    assert end - start == timedelta(hours=25)
    march = service.month_bounds_utc(date(2026, 3, 17))
    assert march == (
        datetime(2026, 3, 1, 6, 0, tzinfo=UTC),
        datetime(2026, 4, 1, 5, 0, tzinfo=UTC),
    )
    december = service.month_bounds_utc(date(2026, 12, 31))
    assert december == (
        datetime(2026, 12, 1, 6, 0, tzinfo=UTC),
        datetime(2027, 1, 1, 6, 0, tzinfo=UTC),
    )


def test_a_clock_time_becomes_an_instant_by_the_daylight_saving_rules(clock: ManualClock) -> None:
    service = make(clock)
    assert service.local_to_utc(date(2026, 3, 8), 150) == datetime(2026, 3, 8, 8, 30, tzinfo=UTC)
    assert service.local_to_utc(date(2026, 11, 1), 90) == datetime(2026, 11, 1, 6, 30, tzinfo=UTC)
    assert service.local_to_utc(date(2026, 10, 9), 450) == datetime(2026, 10, 9, 12, 30, tzinfo=UTC)


def test_day_types_follow_the_country_of_the_current_zone(clock: ManualClock) -> None:
    zone = [CHICAGO]
    service = make(clock, zone)
    assert service.day_type(date(2026, 10, 9)) == "workday"  # a Friday
    assert service.day_type(date(2026, 10, 10)) == "weekend"  # a Saturday
    assert service.day_type(date(2026, 10, 12)) == "holiday"  # Columbus Day (US federal)
    assert service.day_type(date(2026, 11, 26)) == "holiday"  # Thanksgiving
    assert service.day_type(date(2026, 12, 25)) == "holiday"
    zone[0] = SHANGHAI
    assert service.day_type(date(2026, 10, 12)) == "workday"  # nothing special in China
    assert service.day_type(date(2026, 10, 1)) == "holiday"  # National Day


def test_a_weekend_made_a_working_day_in_china_is_a_workday(clock: ManualClock) -> None:
    service = make(clock, [SHANGHAI])
    assert date(2026, 10, 10).weekday() == 5  # a Saturday
    assert service.day_type(date(2026, 10, 10)) == "workday"  # the make-up day for National Day
    assert service.day_type(date(2026, 2, 28)) == "workday"  # ... and for the Spring Festival
    assert service.day_type(date(2026, 10, 11)) == "weekend"  # an ordinary Sunday after it
    assert service.day_type(date(2026, 2, 15)) == "holiday"  # the Spring Festival on a Sunday
    chicago = make(clock, [CHICAGO])
    assert chicago.day_type(date(2026, 10, 10)) == "weekend"  # a Saturday is a Saturday there


def test_a_year_the_holiday_library_does_not_know_falls_back_to_the_weekdays(
    clock: ManualClock,
) -> None:
    warned: list[int] = []
    calendar = DayTypeCalendar(on_fallback=warned.append)
    service = make(clock, [SHANGHAI], calendar=lambda: calendar)
    assert date(2099, 6, 1).weekday() == 0
    assert service.day_type(date(2099, 6, 1)) == "workday"
    assert service.day_type(date(2099, 6, 6)) == "weekend"
    assert service.day_type(date(2099, 6, 7)) == "weekend"
    assert warned == [2099]  # one warning per year, as the price table's calendar does


def test_the_calendar_of_the_price_table_is_the_calendar_of_the_day_types(
    clock: ManualClock,
) -> None:
    """One module for peak hours and holidays: extra dates change both (R-LLM-007)."""
    from twin.llm.pricing import PeakCalendar

    peak = PeakCalendar(extra_offpeak_dates=[date(2026, 10, 14)])  # a Wednesday made a day off
    calendar = DayTypeCalendar(peak_calendar=peak)
    service = make(clock, [SHANGHAI], calendar=lambda: calendar)
    assert not peak.is_workday(date(2026, 10, 14))
    assert service.day_type(date(2026, 10, 14)) == "weekend"


def test_holiday_ranges_set_by_hand_turn_those_days_into_holidays(clock: ManualClock) -> None:
    calendar = DayTypeCalendar(holiday_ranges=[(date(2026, 10, 13), date(2026, 10, 15))])
    service = make(clock, calendar=lambda: calendar)
    assert [service.day_type(date(2026, 10, d)) for d in (12, 13, 14, 15, 16)] == [
        "holiday",
        "holiday",
        "holiday",
        "holiday",
        "workday",
    ]
    assert service.next_day_type(date(2026, 10, 12)) == "holiday"


def test_a_cached_value_is_read_once_until_it_is_dropped() -> None:
    loads = []

    def load() -> str:
        loads.append(1)
        return f"value-{len(loads)}"

    cached = Cached(load)
    assert (cached(), cached()) == ("value-1", "value-1")
    cached.invalidate()
    assert cached() == "value-2" and len(loads) == 2


def test_her_state_needs_the_day_plans(clock: ManualClock) -> None:
    service = make(clock)
    with pytest.raises(PlanUnavailableError, match="no day plans"):
        service.her_state()

    class Fixed:
        def state_at(self, moment: datetime) -> object:
            return ("asked", moment)

    service.attach_states(Fixed())  # type: ignore[arg-type]
    assert service.her_state() == ("asked", clock.now_utc())  # type: ignore[comparison-overlap]
    later = clock.now_utc() + timedelta(hours=2)
    assert service.her_state(later) == ("asked", later)  # type: ignore[comparison-overlap]
