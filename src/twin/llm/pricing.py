"""Prices, peak hours and the off-peak policy (R-LLM-006, R-LLM-007).

Peak hours (DeepSeek pricing page, checked 2026-10-09): on a Beijing working day the hours
01:00-04:00 and 06:00-10:00 UTC; everything else is off-peak and billed at
``pricing.offpeak_multiplier`` times the listed price.  "Working day" is decided by
``chinese_calendar.is_workday`` (public holidays are not working days, compensatory working
days are).  The library only covers a few years; outside its range the calendar falls back to
"Monday to Friday", merges ``pricing.extra_offpeak_dates`` / ``extra_peak_dates``, logs one
warning per year and raises one alert, and ``twin doctor`` reports it.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import chinese_calendar

from twin.clock import ensure_aware
from twin.config.settings import ModelPrice, Settings
from twin.llm.errors import UnknownModelError
from twin.llm.official import LEGACY_MODEL_ALIASES, PEAK_HOURS_UTC
from twin.llm.types import CostBreakdown, Usage
from twin.ops.jobs import OffPeakPolicy, set_offpeak_policy
from twin.ops.logging import get_logger

log = get_logger("twin.llm.pricing")

BEIJING = ZoneInfo("Asia/Shanghai")
SEARCH_HORIZON_DAYS = 60
_PER_MILLION = 1_000_000.0

WorkdayFn = Callable[[date], bool]


@dataclass(frozen=True)
class OffPeakWindow:
    """A half-open interval ``[start, end)`` of off-peak time (both aware UTC)."""

    start: datetime
    end: datetime

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment < self.end


@dataclass(frozen=True)
class CalendarCoverage:
    """Which years the holiday calendar knows (``twin doctor``, R-OPS-009)."""

    covered_years: tuple[int, ...]
    missing_years: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_years


class PeakCalendar:
    """Decides whether an instant is inside DeepSeek's peak hours."""

    def __init__(
        self,
        *,
        extra_offpeak_dates: Collection[date] = (),
        extra_peak_dates: Collection[date] = (),
        workday: WorkdayFn = chinese_calendar.is_workday,
        on_fallback: Callable[[int], None] | None = None,
    ) -> None:
        both = set(extra_offpeak_dates) & set(extra_peak_dates)
        if both:
            raise ValueError(
                "pricing.extra_offpeak_dates and pricing.extra_peak_dates both list "
                + ", ".join(sorted(d.isoformat() for d in both))
            )
        self._offpeak_dates = frozenset(extra_offpeak_dates)
        self._peak_dates = frozenset(extra_peak_dates)
        self._workday = workday
        self._on_fallback = on_fallback
        self._warned_years: set[int] = set()

    def library_knows(self, day: date) -> bool:
        """Whether the holiday library has data for ``day`` (False outside its range)."""
        try:
            self._workday(day)
        except Exception:  # outside its years the library raises an error
            return False
        return True

    def is_workday(self, day: date) -> bool:
        """Whether ``day`` (a Beijing date) is a peak-eligible working day."""
        if day in self._peak_dates:
            return True
        if day in self._offpeak_dates:
            return False
        try:
            return bool(self._workday(day))
        except Exception:  # outside the library's years: Monday to Friday
            self._note_fallback(day.year)
            return day.weekday() < 5

    def public_holiday(self, day: date) -> str | None:
        """Name of the Chinese statutory holiday on ``day``; ``None`` on every other day.

        Ordinary weekends and compensatory working days have no name, and neither has a date
        outside the years the library knows (there :meth:`is_workday` falls back to
        Monday to Friday).  The routine model (round 04) uses it to tell a holiday from a
        plain weekend.
        """
        try:
            is_holiday, name = chinese_calendar.get_holiday_detail(day)
        except Exception:  # outside the library's years
            return None
        return str(name) if is_holiday and name else None

    def _note_fallback(self, year: int) -> None:
        if year in self._warned_years:
            return
        self._warned_years.add(year)
        log.warning("holiday_calendar_out_of_range", year=year)
        if self._on_fallback is not None:
            self._on_fallback(year)

    def is_peak(self, at: datetime) -> bool:
        moment = ensure_aware(at)
        if not self.is_workday(moment.astimezone(BEIJING).date()):
            return False
        hour = moment.hour + moment.minute / 60.0 + moment.second / 3600.0
        return any(start <= hour < end for start, end in PEAK_HOURS_UTC)

    def _peak_intervals(
        self, first_day: date, last_day: date
    ) -> Iterator[tuple[datetime, datetime]]:
        day = first_day
        while day <= last_day:
            if self.is_workday(day):
                for start_hour, end_hour in PEAK_HOURS_UTC:
                    base = datetime(day.year, day.month, day.day, tzinfo=UTC)
                    yield base + timedelta(hours=start_hour), base + timedelta(hours=end_hour)
            day += timedelta(days=1)

    def next_offpeak_window(self, now: datetime) -> OffPeakWindow:
        """The off-peak interval that contains ``now`` or, during peak hours, the next one.

        The window ends at the next peak start (at most :data:`SEARCH_HORIZON_DAYS` ahead).
        """
        moment = ensure_aware(now)
        first = moment.astimezone(BEIJING).date() - timedelta(days=1)
        last = first + timedelta(days=SEARCH_HORIZON_DAYS)
        start = moment
        end = moment + timedelta(days=SEARCH_HORIZON_DAYS)
        for peak_start, peak_end in self._peak_intervals(first, last):
            if peak_end <= start:
                continue
            if peak_start <= start:  # inside peak hours: the window opens when they end
                start = peak_end
                continue
            end = peak_start
            break
        return OffPeakWindow(start=start, end=end)

    def coverage(self, today: date, *, years_ahead: int = 1) -> CalendarCoverage:
        """Check the library for this year and the next ``years_ahead`` (R-LLM-007)."""
        covered: list[int] = []
        missing: list[int] = []
        for year in range(today.year, today.year + years_ahead + 1):
            ok = self.library_knows(date(year, 1, 1)) and self.library_knows(date(year, 12, 31))
            (covered if ok else missing).append(year)
        return CalendarCoverage(tuple(covered), tuple(missing))


class Pricing:
    """The price table plus the peak calendar: turns usage into USD."""

    def __init__(
        self,
        prices: Mapping[str, ModelPrice],
        *,
        offpeak_multiplier: float,
        calendar: PeakCalendar,
    ) -> None:
        if not 0 < offpeak_multiplier <= 1:
            raise ValueError("offpeak_multiplier must be in (0, 1]")
        self._prices = dict(prices)
        self.offpeak_multiplier = offpeak_multiplier
        self.calendar = calendar

    @classmethod
    def from_settings(
        cls, settings: Settings, *, on_fallback: Callable[[int], None] | None = None
    ) -> Pricing:
        calendar = PeakCalendar(
            extra_offpeak_dates=settings.pricing.extra_offpeak_dates,
            extra_peak_dates=settings.pricing.extra_peak_dates,
            on_fallback=on_fallback,
        )
        return cls(
            settings.pricing_usd_per_mtok,
            offpeak_multiplier=settings.pricing.offpeak_multiplier,
            calendar=calendar,
        )

    def resolve(self, model: str) -> str:
        """Canonical model name: retired aliases are billed as ``deepseek-flash``."""
        return LEGACY_MODEL_ALIASES.get(model, model)

    def price_for(self, model: str) -> ModelPrice:
        try:
            return self._prices[self.resolve(model)]
        except KeyError:
            raise UnknownModelError(model) from None

    def has_model(self, model: str) -> bool:
        return self.resolve(model) in self._prices

    def is_peak(self, at: datetime) -> bool:
        return self.calendar.is_peak(at)

    def multiplier(self, at: datetime) -> float:
        """1.0 at peak hours, ``offpeak_multiplier`` otherwise."""
        return 1.0 if self.is_peak(at) else self.offpeak_multiplier

    def cost(self, usage: Usage, model: str, at: datetime) -> CostBreakdown:
        """Cost of ``usage`` for a call made at ``at`` (cache hit, cache miss and output)."""
        price = self.price_for(model)
        peak = self.is_peak(at)
        factor = 1.0 if peak else self.offpeak_multiplier
        return CostBreakdown(
            cache_hit_usd=usage.cache_hit_tokens * price.cache_hit / _PER_MILLION * factor,
            cache_miss_usd=usage.cache_miss_tokens * price.cache_miss / _PER_MILLION * factor,
            output_usd=usage.completion_tokens * price.output / _PER_MILLION * factor,
            peak=peak,
            multiplier=factor,
        )

    def estimate(
        self,
        model: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cache_hit_ratio: float = 0.0,
        at: datetime | None = None,
    ) -> float:
        """Planning estimate in USD; without ``at`` peak prices apply (the safe upper bound)."""
        hit = round(prompt_tokens * min(1.0, max(0.0, cache_hit_ratio)))
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_hit_tokens=hit,
            cache_miss_tokens=prompt_tokens - hit,
        )
        price = self.price_for(model)
        factor = 1.0 if at is None else self.multiplier(at)
        return (
            (
                usage.cache_hit_tokens * price.cache_hit
                + usage.cache_miss_tokens * price.cache_miss
                + usage.completion_tokens * price.output
            )
            / _PER_MILLION
            * factor
        )


class CalendarOffPeakPolicy:
    """The production :class:`~twin.ops.jobs.OffPeakPolicy` (R-ARCH-003, R-LLM-007).

    ``offpeak_only`` jobs run when the clock is outside peak hours.  When the off-peak
    discount is switched off (``pricing.offpeak_multiplier`` = 1.0) there is nothing to wait
    for and the jobs may run at any time.
    """

    def __init__(self, pricing: Pricing) -> None:
        self._pricing = pricing

    def allows(self, now: datetime) -> bool:
        if self._pricing.offpeak_multiplier >= 1.0:
            return True
        return not self._pricing.is_peak(now)


def install_offpeak_policy(pricing: Pricing) -> OffPeakPolicy:
    """Register :class:`CalendarOffPeakPolicy` with the job queue; returns the previous policy."""
    return set_offpeak_policy(CalendarOffPeakPolicy(pricing))
