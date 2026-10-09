"""Prices, peak hours and the off-peak policy (R-LLM-006, R-LLM-007, R-ARCH-003)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from twin.config.loader import load_settings
from twin.llm import official
from twin.llm.errors import UnknownModelError
from twin.llm.pricing import (
    CalendarOffPeakPolicy,
    PeakCalendar,
    Pricing,
    install_offpeak_policy,
)
from twin.llm.types import Usage
from twin.ops.jobs import DeferredOffPeakPolicy, get_offpeak_policy, set_offpeak_policy


def utc(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def pricing(multiplier: float = 0.5, **kwargs: object) -> Pricing:
    settings = load_settings(None, {"pricing": {"offpeak_multiplier": multiplier, **kwargs}})
    return Pricing.from_settings(settings)


# --------------------------------------------------------------------- calendar


def test_peak_hours_on_an_ordinary_weekday_and_their_utc_boundaries() -> None:
    calendar = PeakCalendar()
    friday = date(2026, 10, 16)
    assert friday.weekday() == 4
    expectations = [
        ((0, 59, 59), False),
        ((1, 0, 0), True),
        ((3, 59, 59), True),
        ((4, 0, 0), False),
        ((5, 59, 59), False),
        ((6, 0, 0), True),
        ((9, 59, 59), True),
        ((10, 0, 0), False),
        ((23, 59, 59), False),
    ]
    for (hour, minute, second), peak in expectations:
        moment = utc(friday.year, friday.month, friday.day, hour, minute, second)
        assert calendar.is_peak(moment) is peak, moment


def test_weekends_are_off_peak_all_day() -> None:
    calendar = PeakCalendar()
    for day in (date(2026, 10, 17), date(2026, 10, 18)):
        for hour in range(24):
            assert not calendar.is_peak(utc(day.year, day.month, day.day, hour))


def test_public_holidays_are_off_peak_and_compensatory_working_days_are_peak() -> None:
    calendar = PeakCalendar()
    # 2026-10-01 is a Thursday and a public holiday
    assert not calendar.is_peak(utc(2026, 10, 1, 2))
    # 2026-10-10 is a Saturday that is a compensatory working day (the SPEC counts it as peak)
    assert date(2026, 10, 10).weekday() == 5
    assert calendar.is_peak(utc(2026, 10, 10, 2))
    assert calendar.is_peak(utc(2026, 10, 12, 7))


def test_the_beijing_date_decides_not_the_utc_date() -> None:
    calendar = PeakCalendar()
    # Friday 18:00 UTC is already Saturday 02:00 in Beijing: off-peak whatever the UTC date says
    assert not calendar.is_peak(utc(2026, 10, 16, 18))
    # Sunday 20:00 UTC is Monday 04:00 in Beijing but outside the UTC windows
    assert not calendar.is_peak(utc(2026, 10, 18, 20))
    # Monday 01:00 UTC is Monday 09:00 in Beijing
    assert calendar.is_peak(utc(2026, 10, 19, 1))


def test_naive_datetimes_are_rejected() -> None:
    with pytest.raises(ValueError, match="naive"):
        PeakCalendar().is_peak(datetime(2026, 10, 16, 2))  # noqa: DTZ001


def test_extra_dates_override_the_library() -> None:
    calendar = PeakCalendar(
        extra_offpeak_dates=[date(2026, 10, 16)], extra_peak_dates=[date(2026, 10, 17)]
    )
    assert not calendar.is_peak(utc(2026, 10, 16, 2))
    assert calendar.is_peak(utc(2026, 10, 17, 2))
    with pytest.raises(ValueError, match="both list"):
        PeakCalendar(
            extra_offpeak_dates=[date(2026, 10, 16)], extra_peak_dates=[date(2026, 10, 16)]
        )


def test_years_outside_the_library_fall_back_to_weekdays_and_warn_once() -> None:
    warned: list[int] = []
    calendar = PeakCalendar(on_fallback=warned.append)
    monday = date(2031, 3, 3)
    assert monday.weekday() == 0
    assert calendar.is_peak(utc(2031, 3, 3, 2))
    assert calendar.is_peak(utc(2031, 3, 4, 2))
    assert not calendar.is_peak(utc(2031, 3, 8, 2))  # Saturday
    assert warned == [2031]
    assert calendar.is_peak(utc(2032, 3, 1, 2))
    assert warned == [2031, 2032]


def test_extra_dates_apply_in_years_the_library_does_not_know() -> None:
    calendar = PeakCalendar(
        extra_offpeak_dates=[date(2031, 3, 3)], extra_peak_dates=[date(2031, 3, 8)]
    )
    assert not calendar.is_peak(utc(2031, 3, 3, 2))
    assert calendar.is_peak(utc(2031, 3, 8, 2))


def test_a_library_that_raises_is_handled_like_an_unknown_year() -> None:
    def broken(day: date) -> bool:
        raise NotImplementedError(f"no data for {day}")

    warned: list[int] = []
    calendar = PeakCalendar(workday=broken, on_fallback=warned.append)
    assert calendar.is_peak(utc(2026, 10, 16, 2))
    assert not calendar.is_peak(utc(2026, 10, 17, 2))
    assert warned == [2026]
    assert not calendar.library_knows(date(2026, 10, 16))


def test_coverage_reports_the_years_the_library_lacks() -> None:
    coverage = PeakCalendar().coverage(date(2026, 10, 9))
    assert coverage.covered_years == (2026,)
    assert coverage.missing_years == (2027,)
    assert not coverage.complete
    assert PeakCalendar(workday=lambda day: True).coverage(date(2026, 10, 9)).complete


def test_next_offpeak_window_inside_and_outside_peak_hours() -> None:
    calendar = PeakCalendar()
    # Friday 02:00 UTC is peak: off-peak opens at 04:00 and closes at the 06:00 peak
    window = calendar.next_offpeak_window(utc(2026, 10, 16, 2))
    assert (window.start, window.end) == (utc(2026, 10, 16, 4), utc(2026, 10, 16, 6))
    # Friday 08:00 UTC is peak: off-peak runs through the weekend until Monday 01:00 UTC
    window = calendar.next_offpeak_window(utc(2026, 10, 16, 8))
    assert (window.start, window.end) == (utc(2026, 10, 16, 10), utc(2026, 10, 19, 1))
    # already off-peak: the window that contains "now"
    window = calendar.next_offpeak_window(utc(2026, 10, 16, 5))
    assert (window.start, window.end) == (utc(2026, 10, 16, 5), utc(2026, 10, 16, 6))
    assert window.contains(utc(2026, 10, 16, 5, 30)) and not window.contains(window.end)


def test_next_offpeak_window_skips_the_national_day_holiday() -> None:
    calendar = PeakCalendar()
    window = calendar.next_offpeak_window(utc(2026, 9, 30, 8))
    # 1-7 October are public holidays; Thursday the 8th is the first working day again
    assert (window.start, window.end) == (utc(2026, 9, 30, 10), utc(2026, 10, 8, 1))


# ------------------------------------------------------------------------ prices


def test_defaults_match_the_official_price_page() -> None:
    settings = load_settings()
    for model, official_prices in official.PEAK_PRICES_USD_PER_MTOK.items():
        configured = settings.pricing_usd_per_mtok[model]
        assert configured.cache_hit == official_prices["cache_hit"]
        assert configured.cache_miss == official_prices["cache_miss"]
        assert configured.output == official_prices["output"]
        for key, value in official.OFFPEAK_PRICES_USD_PER_MTOK[model].items():
            assert value == pytest.approx(official_prices[key] * official.OFFPEAK_RATIO)
    assert settings.pricing.offpeak_multiplier == official.OFFPEAK_RATIO
    assert settings.deepseek.chat_model == official.FLASH_MODEL


def test_cost_has_three_parts_and_the_offpeak_discount() -> None:
    table = pricing(0.5)
    usage = Usage(
        prompt_tokens=3_000_000,
        completion_tokens=1_000_000,
        cache_hit_tokens=2_000_000,
        cache_miss_tokens=1_000_000,
    )
    peak = table.cost(usage, "deepseek-flash", utc(2026, 10, 16, 2))
    assert peak.peak and peak.multiplier == 1.0
    assert peak.cache_hit_usd == pytest.approx(2 * 0.006)
    assert peak.cache_miss_usd == pytest.approx(0.30)
    assert peak.output_usd == pytest.approx(1.20)
    assert peak.total_usd == pytest.approx(0.012 + 0.30 + 1.20)

    off = table.cost(usage, "deepseek-flash", utc(2026, 10, 17, 2))
    assert not off.peak and off.multiplier == 0.5
    assert off.total_usd == pytest.approx(peak.total_usd * 0.5)
    pro = table.cost(usage, "deepseek-v4-pro", utc(2026, 10, 16, 2))
    assert pro.total_usd == pytest.approx(2 * 0.044 + 1.32 + 3.96)


def test_a_multiplier_of_one_removes_the_discount() -> None:
    table = pricing(1.0)
    usage = Usage(prompt_tokens=10, completion_tokens=10, cache_miss_tokens=10)
    weekend = table.cost(usage, "deepseek-flash", utc(2026, 10, 17, 2))
    weekday = table.cost(usage, "deepseek-flash", utc(2026, 10, 16, 2))
    assert weekend.total_usd == weekday.total_usd
    with pytest.raises(ValueError, match="offpeak_multiplier"):
        Pricing({}, offpeak_multiplier=0, calendar=PeakCalendar())


def test_retired_model_names_are_billed_as_flash_and_unknown_models_are_refused() -> None:
    table = pricing()
    usage = Usage(prompt_tokens=100, completion_tokens=100, cache_miss_tokens=100)
    at = utc(2026, 10, 16, 2)
    flash = table.cost(usage, "deepseek-flash", at).total_usd
    assert table.cost(usage, "deepseek-v4-flash", at).total_usd == flash
    assert table.cost(usage, "deepseek-v4-flash-vision-exp", at).total_usd == flash
    assert table.has_model("deepseek-v4-flash") and not table.has_model("gpt-x")
    with pytest.raises(UnknownModelError, match="gpt-x"):
        table.cost(usage, "gpt-x", at)


def test_estimate_defaults_to_peak_prices_and_honours_cache_hits() -> None:
    table = pricing()
    cold = table.estimate("deepseek-flash", prompt_tokens=1_000_000, completion_tokens=0)
    assert cold == pytest.approx(0.30)
    warm = table.estimate(
        "deepseek-flash", prompt_tokens=1_000_000, completion_tokens=0, cache_hit_ratio=1.0
    )
    assert warm == pytest.approx(0.006)
    off = table.estimate(
        "deepseek-flash", prompt_tokens=1_000_000, completion_tokens=0, at=utc(2026, 10, 17, 2)
    )
    assert off == pytest.approx(0.15)


def test_usage_properties() -> None:
    usage = Usage(prompt_tokens=10, completion_tokens=5, cache_hit_tokens=4, cache_miss_tokens=6)
    assert usage.total_tokens == 15 and usage.cache_hit_ratio == pytest.approx(0.4)
    assert Usage().cache_hit_ratio == 0.0


# ------------------------------------------------------------------------ policy


def test_the_policy_waits_for_off_peak_hours_unless_the_discount_is_off() -> None:
    discounted = CalendarOffPeakPolicy(pricing(0.5))
    assert not discounted.allows(utc(2026, 10, 16, 2))
    assert discounted.allows(utc(2026, 10, 16, 5))
    assert discounted.allows(utc(2026, 10, 17, 2))
    flat = CalendarOffPeakPolicy(pricing(1.0))
    assert flat.allows(utc(2026, 10, 16, 2))


def test_installing_the_policy_replaces_the_default_of_round_00() -> None:
    previous = get_offpeak_policy()
    assert isinstance(previous, DeferredOffPeakPolicy)
    try:
        returned = install_offpeak_policy(pricing())
        assert returned is previous
        policy = get_offpeak_policy()
        assert isinstance(policy, CalendarOffPeakPolicy)
        assert policy.allows(utc(2026, 10, 17, 3))
        assert not policy.allows(utc(2026, 10, 16, 3))
    finally:
        set_offpeak_policy(previous)


def test_next_offpeak_window_is_consistent_with_is_peak() -> None:
    calendar = PeakCalendar()
    moment = utc(2026, 10, 12, 0)
    for _ in range(14 * 24):
        window = calendar.next_offpeak_window(moment)
        assert not calendar.is_peak(window.start)
        assert not calendar.is_peak(window.end - timedelta(seconds=1))
        assert window.end > window.start
        moment += timedelta(hours=1)
