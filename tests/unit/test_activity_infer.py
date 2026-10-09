"""Sleep, busy time and the activity curves inferred from known data (R-ACT-001 to R-ACT-004)."""

from __future__ import annotations

from datetime import date, time

import pytest

from tests.support.synth_chat import ChatSpec, build_chat, build_hourly_chat
from twin.config.settings import ActivityConfig, TzRange
from twin.profile.activity_infer import (
    circular_runs,
    find_valley,
    gaussian_smooth,
)
from twin.profile.activity_model import ActivityModel, SleepWindow, sleep_core_in_daytime
from twin.profile.api import load_activity_model, load_profile
from twin.profile.builder import rebuild
from twin.profile.circular import signed_diff
from twin.profile.overrides import RoutineOverrides
from twin.schedule.daytype import DayTypeCalendar
from twin.services import Services

SPEC_HOURS = (
    244, 62, 75, 44, 27, 28, 13, 64, 118, 81, 245, 352,
    14, 4, 14, 137, 94, 0, 7, 59, 129, 180, 144, 78,
)  # fmt: skip


def learn(services: Services, scope: str = "live", **spec: object) -> ActivityModel:
    build_chat(services, ChatSpec(**spec))  # type: ignore[arg-type]
    rebuild(services, scope, reason="test")
    model = load_activity_model(services, scope, apply_overrides=False)
    assert model is not None
    return model


def window(model: ActivityModel, key: str = "all") -> SleepWindow:
    found = model.sleep_profile().windows.get(key)
    assert found is not None, key
    return found


def within(minute: float, hour: float, tolerance: float = 30.0) -> bool:
    return abs(signed_diff(minute, hour * 60)) <= tolerance


# --------------------------------------------------------------------- helpers


def test_smoothing_is_circular_and_conserves_the_total() -> None:
    values = [0.0] * 96
    values[0] = 96.0
    smooth = gaussian_smooth(values, 2.0)
    assert sum(smooth) == pytest.approx(96.0)
    assert smooth[95] == pytest.approx(smooth[1]) and smooth[95] > smooth[6] > 0 and smooth[10] == 0
    assert gaussian_smooth(values, 0) == values
    assert gaussian_smooth([], 2.0) == []


def test_runs_wrap_around_the_end_of_the_day() -> None:
    mask = [False] * 10
    for index in (8, 9, 0, 1):
        mask[index] = True
    mask[5] = True
    assert sorted(circular_runs(mask)) == [(5, 1), (8, 4)]
    assert circular_runs([True] * 4) == [(0, 4)] and circular_runs([False] * 4) == []


def test_the_valley_is_the_longest_low_stretch() -> None:
    curve = [5.0] * 96
    for slot in range(4, 30):
        curve[slot] = 0.1
    for slot in range(60, 66):
        curve[slot] = 0.1
    valley = find_valley(curve, 12, 0.25)
    assert (valley.start, valley.length) == (4, 26)
    assert valley.start_min == 60 and valley.end_min == 30 * 15
    # nothing reaches three hours: the least active window of that length is chosen
    flat = [3.0 + (slot % 7) for slot in range(96)]
    fallback = find_valley(flat, 12, 0.25)
    assert fallback.length == 12 and fallback.ratio == 0.0
    assert find_valley([0.0] * 96, 12, 0.25).length == 12


# ------------------------------------------------------------------ the main case


def test_sleep_busy_time_and_initiations_are_inferred_within_tolerance(
    services: Services,
) -> None:
    model = learn(services)
    profile = model.sleep_profile()
    assert profile.confidence == "high" and profile.valid_days >= 38
    assert not [w for w in profile.warnings if "source_timezone" in w]
    for key in ("all", "workday", "weekend"):
        found = window(model, key)
        assert within(found.onset_min, 1.0) and within(found.wake_min, 8.5), key
        assert found.source == "days" and 7 * 60 <= found.duration_min() <= 8.5 * 60
    # she is silent between 01:00 and 08:30 whatever the day: no slot there has activity
    for slot in range(4 * 2 + 1, 4 * 7):
        assert model.rate_at(slot, "workday") < 0.03, slot  # blur of the Gaussian smoothing
    # workdays 13:00-17:00: slow, rare answers
    (busy,) = model.busy_windows("workday")
    assert abs(busy.start_min - 13 * 60) <= 30 and abs(busy.end_min - 17 * 60) <= 30
    assert busy.latency_ratio >= 3 and busy.stability >= 0.8 and busy.confidence >= 0.8
    assert busy.latency.median() == pytest.approx(1500, rel=0.4)
    assert model.busy_windows("weekend") == () and model.busy_windows("holiday") == ()
    # four conversations a day opened by her (+-15 %)
    assert model.initiations_per_day == pytest.approx(4.0, rel=0.15)
    total = sum(model.initiation_rate_at(slot, "workday") for slot in range(96))
    assert total == pytest.approx(4.0, rel=0.15)


def test_curves_and_latencies_are_queried_by_local_slot(services: Services) -> None:
    model = learn(services)
    morning = model.rate_at(4 * 9 + 1, "workday")
    assert morning > 0.1
    fast = model.latency_distribution(4 * 21).median()
    assert fast < 120
    for hour in (13, 15, 16):  # busy hours with plenty of answers: the slot itself or its hour
        assert model.latency_distribution(4 * hour + 1).median() > 10 * fast
    # an hour with too few answers (14:00 here) and a night hour answer with the all-day curve
    assert model.latency.level_for(4 * 14) == 2
    assert model.latency_distribution(4 * 14) is model.latency.overall
    night = model.latency_distribution(4 * 4)
    assert night.n > 0
    with pytest.raises(ValueError, match="local slot"):
        model.rate_at(96, "workday")
    with pytest.raises(ValueError, match="local slot"):
        model.latency_distribution(-1)
    # a day type with too few days borrows the curve of the broader one
    assert model.days["holiday"] < 4
    assert model.rate_at(40, "holiday") == model.rate_at(40, "weekend")


def test_the_model_survives_storage(services: Services) -> None:
    model = learn(services)
    again = ActivityModel.from_json(model.to_json())
    assert again.sleep_profile().to_json() == model.sleep_profile().to_json()
    assert again.rate_at(40, "workday") == pytest.approx(model.rate_at(40, "workday"), abs=1e-4)
    assert again.busy_windows("workday")[0].label() == model.busy_windows("workday")[0].label()
    assert again.typical_state(time(4, 0), "workday") == "deep_sleep"
    with pytest.raises(ValueError, match="schema"):
        ActivityModel.from_json({**model.to_json(), "schema": 99})


def test_weekends_may_sleep_later_than_workdays(services: Services) -> None:
    model = learn(services, weekend_sleep=(2.0, 10.0), days=56)
    workday, weekend = window(model, "workday"), window(model, "weekend")
    assert within(workday.wake_min, 8.5) and within(weekend.wake_min, 10.0)
    assert within(workday.onset_min, 1.0) and within(weekend.onset_min, 2.0, 45)
    assert model.typical_state(time(9, 30), "weekend") in ("deep_sleep", "sleep_edge")
    assert model.typical_state(time(9, 30), "workday") == "free"


def test_the_night_message_she_answers_in_the_morning_does_not_change_her_sleep(
    services: Services,
) -> None:
    model = learn(services, night_message=True)
    found = window(model)
    assert within(found.onset_min, 1.0) and within(found.wake_min, 8.5)
    assert model.initiations_per_day == pytest.approx(4.0, rel=0.15)
    # her morning answers come after more than the segment gap: they are delayed replies,
    # counted apart from the latency samples
    profile = load_profile(services, "live")
    assert profile is not None
    delayed = profile.metrics.scalar("her", "delayed_reply_rate")
    assert delayed is not None and delayed > 0.1
    latency = profile.metrics.distribution("her", "reply_latency_s")
    assert latency is not None and latency.maximum() < 3600


def test_a_night_that_begins_before_midnight_is_found_and_its_stats_wrap_correctly(
    services: Services,
) -> None:
    model = learn(services, sleep=(23.83, 7.0), days=45)
    profile = model.sleep_profile()
    found = window(model)
    assert profile.confidence == "high"
    assert within(found.onset_min, 23.83, 30) and within(found.wake_min, 7.0, 30)
    # onsets on both sides of midnight were folded into one circular distribution
    assert found.onset.std < 30 and 6 * 60 <= found.duration_min() <= 8 * 60
    assert not [w for w in profile.warnings if "source_timezone" in w]
    # the sleep that ends this morning, and the one that begins tonight
    assert model.typical_state(time(3, 0), "workday") == "deep_sleep"
    assert model.typical_state(time(12, 0), "workday") != "deep_sleep"
    assert (
        model.typical_state(time(23, 59), "workday") != "deep_sleep"
    )  # she is just falling asleep
    assert model.typical_state(time(0, 45), "workday") == "deep_sleep"


def test_the_chinese_calendar_decides_the_day_types_of_data_learnt_in_shanghai(
    services: Services,
) -> None:
    services.settings.time.source_timezone = "Asia/Shanghai"
    start = date(2026, 9, 21)
    model = learn(services, zone="Asia/Shanghai", start=start, days=28)
    calendar = DayTypeCalendar()
    expected = calendar.days_between(start, date(2026, 10, 18), "Asia/Shanghai")
    assert {k: model.days[k] for k in ("workday", "weekend", "holiday")} == expected
    assert expected["holiday"] >= 4  # National Day
    # the working Saturday of 2026-10-10 is a workday for the model, a plain Saturday is not
    assert calendar.day_type(date(2026, 10, 10), "Asia/Shanghai") == "workday"
    assert calendar.day_type(date(2026, 10, 17), "Asia/Shanghai") == "weekend"
    # the same dates in a US zone are ordinary weekdays and weekends
    us = calendar.days_between(start, date(2026, 10, 18), "America/Chicago")
    assert us["holiday"] == 1 and us != expected  # Columbus Day only


# ------------------------------------------------------------------ summer time


@pytest.mark.parametrize(
    ("start", "label"),
    [
        (date(2026, 2, 23), "spring forward 2026-03-08"),
        (date(2026, 10, 19), "fall back 2026-11-01"),
    ],
)
def test_clock_changes_do_not_move_the_routine(services: Services, start: date, label: str) -> None:
    model = learn(services, start=start, days=35)
    found = window(model)
    assert within(found.onset_min, 1.0) and within(found.wake_min, 8.5), label
    assert found.onset.std < 30
    assert model.sleep_profile().confidence == "high"


# ------------------------------------------------------------------- time zones


def test_data_from_another_time_zone_raises_the_plausibility_warning(services: Services) -> None:
    model = learn(services, zone="Asia/Shanghai")  # learnt as America/Chicago
    profile = model.sleep_profile()
    found = window(model)
    assert sleep_core_in_daytime(found, model.edge_minutes)
    texts = [w for w in profile.warnings if "source_timezone" in w]
    assert texts and "America/Chicago" in texts[0] and "10:00–18:00" in texts[0]
    core_start, _ = found.core(model.edge_minutes)
    assert 10 * 60 <= core_start < 18 * 60


def test_learning_in_the_right_zone_gives_no_warning(services: Services) -> None:
    services.settings.time.source_timezone = "Asia/Shanghai"
    model = learn(services, zone="Asia/Shanghai")
    profile = model.sleep_profile()
    assert not [w for w in profile.warnings if "source_timezone" in w]
    found = window(model)
    assert within(found.onset_min, 1.0) and within(found.wake_min, 8.5)
    assert model.zone == "Asia/Shanghai"


def test_zone_ranges_switch_the_clock_by_date_for_the_whole_model(services: Services) -> None:
    start = date(2026, 8, 3)
    zones = {
        date.fromordinal(start.toordinal() + offset): "Asia/Shanghai" for offset in range(22, 56)
    }
    services.settings.time.source_timezone_ranges = [
        TzRange(from_=date(2026, 8, 25), to=date(2026, 9, 27), tz="Asia/Shanghai")
    ]
    model = learn(services, start=start, days=56, zone_by_day=zones)
    found = window(model)
    assert within(found.onset_min, 1.0, 45) and within(found.wake_min, 8.5, 45)
    assert not [w for w in model.sleep_profile().warnings if "source_timezone" in w]
    assert set(model.extra["zones"]) == {"America/Chicago", "Asia/Shanghai"}
    # without the ranges the Shanghai weeks are read on the Chicago clock: the nights land in
    # the daytime, the plausibility warning fires, and the Chicago weeks no longer fit
    services.settings.time.source_timezone_ranges = []
    rebuild(services, "live", reason="test", force=True)
    mixed = load_activity_model(services, "live", apply_overrides=False)
    assert mixed is not None
    assert any("source_timezone" in w for w in mixed.sleep_profile().warnings)
    assert mixed.sleep_profile().valid_days < model.sleep_profile().valid_days


# ------------------------------------------------------- the sample of SPEC section 0


def low_hours(totals: list[float]) -> set[int]:
    """Hours whose count is at most half the average hourly count."""
    mean = sum(totals) / len(totals)
    return {hour for hour, count in enumerate(totals) if count <= 0.5 * mean}


def runs_of_at_least(hours: set[int], length: int) -> list[set[int]]:
    found: list[set[int]] = []
    for start in range(24):
        run = set()
        hour = start
        while hour % 24 in hours and len(run) < 24:
            run.add(hour % 24)
            hour += 1
        if len(run) >= length and (start - 1) % 24 not in hours:
            found.append(run)
    return found


@pytest.mark.parametrize("shift", [0, 5, 9, 14, 20])
def test_the_spec_sample_gives_a_low_activity_interval_and_a_consistent_warning(
    services: Services, shift: int
) -> None:
    totals = [float(SPEC_HOURS[(hour - shift) % 24]) for hour in range(24)]
    build_hourly_chat(services, totals, days=28)
    rebuild(services, "live", reason="test")  # must not raise
    model = load_activity_model(services, "live", apply_overrides=False)
    assert model is not None
    profile = model.sleep_profile()
    found = window(model)
    runs = runs_of_at_least(low_hours(totals), 3)
    assert runs, "the sample has stretches of three low hours"
    # every minute of the inferred interval lies in a low-activity stretch of three hours or more
    length = round(found.duration_min())
    assert length >= 3 * 60
    inside = {int(((found.onset_min + step) % 1440) // 60) for step in range(0, length, 5)}
    assert any(inside <= run for run in runs), (sorted(inside), runs)
    # the warning says "daytime" exactly when most of the core lies between 10:00 and 18:00
    core_start, _ = found.core(model.edge_minutes)
    span = max(0.0, found.duration_min() - 2 * model.edge_minutes)
    daytime = sum(1 for step in range(int(span)) if 10 * 60 <= (core_start + step) % 1440 < 18 * 60)
    expected = span > 0 and daytime / span >= 0.5
    warned = any("source_timezone" in w for w in profile.warnings)
    assert warned == expected
    assert profile.confidence == "low"  # no night of this sample has three silent hours


def test_few_nights_mark_the_result_low_confidence_and_use_the_valley(
    services: Services,
) -> None:
    model = learn(services, days=10)
    profile = model.sleep_profile()
    assert profile.confidence == "low" and profile.valid_days >= 8
    assert any("低置信" in w for w in profile.warnings)
    found = window(model)
    assert found.source == "curve"
    # the valley still lies inside the true silence 01:00-08:30 (give or take the blur)
    start, end = found.onset_min, found.wake_min
    assert within(start, 1.0, 90) and within(end, 8.5, 90)
    assert found.duration_min() >= 3 * 60


def test_a_threshold_that_is_too_strict_widens_until_a_valley_exists() -> None:
    assert ActivityConfig().sleep_rate_ratio == 0.25
    cfg = ActivityConfig(sleep_min_hours=10.0)
    curve = [1.0 + 0.5 * ((slot * 7) % 5) for slot in range(96)]
    valley = find_valley(curve, int(cfg.sleep_min_hours * 4), cfg.sleep_rate_ratio)
    assert valley.length == 40


def test_too_little_data_gives_no_sleep_at_all(services: Services) -> None:
    model = learn(services, days=1)
    profile = model.sleep_profile()
    assert profile.confidence in ("none", "low")
    if profile.confidence == "none":
        assert profile.windows == {} and profile.warnings
        assert model.typical_state(time(3, 0), "workday") == "free"


def test_holidays_set_by_hand_are_learnt_as_holidays(services: Services) -> None:
    build_chat(services, ChatSpec(days=30))
    rebuild(services, "live", reason="test")
    before = load_activity_model(services, "live", apply_overrides=False)
    assert before is not None and before.days["holiday"] <= 1
    overrides = RoutineOverrides(services.db, services.clock)
    overrides.add_holiday(date(2026, 8, 10), date(2026, 8, 14))
    rebuild(services, "live", reason="test")
    after = load_activity_model(services, "live", apply_overrides=False)
    assert after is not None and after.days["holiday"] == before.days["holiday"] + 5
    assert after.days["workday"] == before.days["workday"] - 5
