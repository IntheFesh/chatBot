"""Day types for the routine model: workday, weekend or holiday (R-ACT-002).

The type of a calendar day depends on where she was that day (R-ACT-001): for a China time
zone it is decided by ``chinese_calendar`` (statutory holidays and compensatory working days
included), for a US time zone by the US federal holidays of the ``holidays`` package, and for
any other zone by the weekday alone.

The Chinese part is the *same calendar module as the price table's peak hours* (round 01,
:class:`twin.llm.pricing.PeakCalendar`): the same ``is_workday`` call, the same fallback to
"Monday to Friday" for years the library does not cover, the same one-warning-per-year
logging.  Nothing is decided twice.

Holiday date ranges set by hand (``twin routine add holiday``) turn every day in the range
into a holiday, whatever the calendar says.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import date, timedelta
from typing import Literal

from holidays import HolidayBase, country_holidays

from twin.llm.pricing import PeakCalendar

DayType = Literal["workday", "weekend", "holiday"]
DAY_TYPES: tuple[DayType, ...] = ("workday", "weekend", "holiday")

CHINA = "CN"
UNITED_STATES = "US"

# zones whose country the calendar knows without configuration; `safety.timezone_country`
# (R-CFG-004) extends and overrides this table
BUILTIN_ZONE_COUNTRIES: Mapping[str, str] = {
    "Asia/Shanghai": CHINA,
    "Asia/Chongqing": CHINA,
    "Asia/Harbin": CHINA,
    "Asia/Urumqi": CHINA,
    "PRC": CHINA,
    "America/New_York": UNITED_STATES,
    "America/Detroit": UNITED_STATES,
    "America/Chicago": UNITED_STATES,
    "America/Denver": UNITED_STATES,
    "America/Phoenix": UNITED_STATES,
    "America/Los_Angeles": UNITED_STATES,
    "America/Anchorage": UNITED_STATES,
    "Pacific/Honolulu": UNITED_STATES,
    "America/Indiana/Indianapolis": UNITED_STATES,
    "US/Eastern": UNITED_STATES,
    "US/Central": UNITED_STATES,
    "US/Mountain": UNITED_STATES,
    "US/Pacific": UNITED_STATES,
}


class DayTypeCalendar:
    """Classifies a local calendar date given the zone she was in."""

    def __init__(
        self,
        *,
        zone_countries: Mapping[str, str] | None = None,
        holiday_ranges: Iterable[tuple[date, date]] = (),
        peak_calendar: PeakCalendar | None = None,
        on_fallback: Callable[[int], None] | None = None,
    ) -> None:
        self._countries = {**BUILTIN_ZONE_COUNTRIES, **(zone_countries or {})}
        self._ranges = tuple(holiday_ranges)
        for start, end in self._ranges:
            if end < start:
                raise ValueError(f"holiday range {start}..{end} ends before it starts")
        self._china = peak_calendar or PeakCalendar(on_fallback=on_fallback)
        self._us_by_year: dict[int, HolidayBase] = {}
        self._cache: dict[tuple[date, str], DayType] = {}

    def with_holiday_ranges(self, ranges: Iterable[tuple[date, date]]) -> DayTypeCalendar:
        """A calendar that also treats ``ranges`` as holidays (shares the Chinese calendar)."""
        return DayTypeCalendar(
            zone_countries=self._countries, holiday_ranges=ranges, peak_calendar=self._china
        )

    def country_of(self, zone_key: str) -> str | None:
        return self._countries.get(zone_key)

    def _us_holidays(self, year: int) -> HolidayBase:
        found = self._us_by_year.get(year)
        if found is None:
            found = country_holidays("US", years=year)
            self._us_by_year[year] = found
        return found

    def _classify(self, day: date, zone_key: str) -> DayType:
        if any(start <= day <= end for start, end in self._ranges):
            return "holiday"
        country = self.country_of(zone_key)
        weekend = day.weekday() >= 5
        if country == CHINA:
            if self._china.is_workday(day):
                return "workday"
            return "holiday" if self._china.public_holiday(day) else "weekend"
        if country == UNITED_STATES:
            if day in self._us_holidays(day.year):
                return "holiday"
            return "weekend" if weekend else "workday"
        return "weekend" if weekend else "workday"

    def day_type(self, day: date, zone_key: str) -> DayType:
        key = (day, zone_key)
        found = self._cache.get(key)
        if found is None:
            found = self._classify(day, zone_key)
            self._cache[key] = found
        return found

    def days_between(self, start: date, end: date, zone_key: str) -> dict[DayType, int]:
        """How many days of each type lie in ``[start, end]`` (for tests and reports)."""
        counts: dict[DayType, int] = dict.fromkeys(DAY_TYPES, 0)
        day = start
        while day <= end:
            counts[self.day_type(day, zone_key)] += 1
            day += timedelta(days=1)
        return counts
