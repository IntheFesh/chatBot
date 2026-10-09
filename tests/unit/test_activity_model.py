"""``typical_state`` and the manual corrections on the activity model (R-ACT-005, R-ACT-006)."""

from __future__ import annotations

import random
from datetime import UTC, datetime, time

from twin.profile.activity_model import (
    CURVE_KEYS,
    ActivityModel,
    BusyWindow,
    SleepProfile,
    SleepWindow,
    sleep_core_in_daytime,
)
from twin.profile.circular import CircularStats
from twin.profile.distribution import BucketedDistribution, EmpiricalDistribution
from twin.profile.overrides import OverrideView


def clock(hour: float) -> time:
    minutes = round(hour * 60)
    return time(minutes // 60, minutes % 60)


def sleep_window(onset: float, wake: float) -> SleepWindow:
    return SleepWindow(
        CircularStats.from_values([onset * 60]),
        CircularStats.from_values([wake * 60]),
        30,
        "days",
    )


def busy_window(start: float, end: float, weekdays: tuple[int, ...] = ()) -> BusyWindow:
    return BusyWindow(
        start * 60,
        end * 60,
        0.9,
        4.0,
        0.9,
        EmpiricalDistribution.from_counter({1500.0: 20}, discrete=True),
        "override" if weekdays else "inferred",
        weekdays,
    )


def make_model(
    windows: dict[str, SleepWindow],
    busy: dict[str, tuple[BusyWindow, ...]] | None = None,
    edge: int = 30,
    warnings: tuple[str, ...] = (),
) -> ActivityModel:
    curve = tuple(float(i % 5) for i in range(96))
    curves = dict.fromkeys(CURVE_KEYS, curve)
    latency = BucketedDistribution.from_counters({50: {20.0: 40}}, sizes=(1, 4), min_samples=20)
    return ActivityModel(
        scope="live",
        zone="America/Chicago",
        edge_minutes=edge,
        days={"all": 60, "workday": 40, "weekend": 18, "holiday": 2},
        rate=curves,
        rate_raw=curves,
        initiation=curves,
        initiation_raw=curves,
        latency=latency,
        latency_workday=latency,
        sleep=SleepProfile(windows, "high", 30, warnings),
        busy=busy or {},
        her_messages=1000,
        initiations_per_day=4.0,
    )


def states(model: ActivityModel, day_type: str, hours: list[float], **kw: object) -> list[str]:
    return [model.typical_state(clock(h), day_type, **kw) for h in hours]  # type: ignore[arg-type]


def test_state_boundaries_of_a_night_after_midnight() -> None:
    model = make_model({"all": sleep_window(1.0, 8.5)})
    probes = [0.0, 0.98, 1.0, 1.49, 1.5, 4.0, 7.98, 8.0, 8.49, 8.5, 12.0, 23.98]
    assert states(model, "workday", probes) == [
        "free",  # 00:00
        "free",  # 00:58
        "sleep_edge",  # 01:00 - falling asleep
        "sleep_edge",  # 01:29
        "deep_sleep",  # 01:30
        "deep_sleep",  # 04:00
        "deep_sleep",  # 07:58
        "sleep_edge",  # 08:00 - about to wake
        "sleep_edge",  # 08:29
        "free",  # 08:30 - awake
        "free",
        "free",
    ]
    assert model.typical_state(time(23, 59, 59), "weekend") == "free"
    assert model.typical_state(time(0, 0), "holiday") == "free"


def test_state_boundaries_of_a_night_across_midnight() -> None:
    model = make_model({"all": sleep_window(23.0, 7.0)})
    probes = [22.98, 23.0, 23.49, 23.5, 3.0, 6.49, 6.5, 6.98, 7.0]
    assert states(model, "workday", probes) == [
        "free",
        "sleep_edge",
        "sleep_edge",
        "deep_sleep",
        "deep_sleep",
        "deep_sleep",
        "sleep_edge",
        "sleep_edge",
        "free",
    ]
    assert model.typical_state(time(23, 59, 59), "workday") == "deep_sleep"
    assert model.typical_state(time(0, 0), "workday") == "deep_sleep"


def test_the_sleep_that_begins_tonight_is_the_one_of_tomorrows_type() -> None:
    model = make_model(
        {
            "all": sleep_window(23.0, 7.0),
            "workday": sleep_window(23.0, 7.0),
            "weekend": sleep_window(1.0, 10.0),
        }
    )
    # a Friday evening: tonight she sleeps like on a weekend morning (01:00-10:00)
    assert model.typical_state(clock(23.5), "workday", next_day_type="weekend") == "free"
    assert model.typical_state(clock(23.5), "workday", next_day_type="workday") == "deep_sleep"
    assert model.typical_state(clock(23.5), "workday") == "deep_sleep"  # same type by default
    # the morning after a Friday night belongs to Saturday's window
    assert model.typical_state(clock(9.0), "weekend") == "deep_sleep"
    assert model.typical_state(clock(9.0), "workday") == "free"
    # a day type without its own window uses the all-days window
    assert model.typical_state(clock(3.0), "holiday") == "deep_sleep"


def test_a_short_sleep_is_all_edge() -> None:
    model = make_model({"all": sleep_window(1.0, 1.0 + 40 / 60)}, edge=30)
    assert states(model, "workday", [1.0, 1.3, 1.6, 1.7]) == [
        "sleep_edge",
        "sleep_edge",
        "sleep_edge",
        "free",
    ]
    assert (
        make_model({"all": sleep_window(1.0, 8.5)}, edge=0).typical_state(clock(1.0), "workday")
        == "deep_sleep"
    )


def test_busy_windows_apply_to_their_day_type_and_never_to_sleep() -> None:
    model = make_model(
        {"all": sleep_window(1.0, 8.5)}, busy={"workday": (busy_window(13.0, 17.0),)}
    )
    probes = [12.98, 13.0, 16.98, 17.0]
    assert states(model, "workday", probes) == ["free", "busy", "busy", "free"]
    assert states(model, "weekend", probes) == ["free"] * 4
    overlapping = make_model(
        {"all": sleep_window(1.0, 8.5)}, busy={"workday": (busy_window(7.0, 10.0),)}
    )
    assert overlapping.typical_state(clock(7.5), "workday") == "deep_sleep"  # sleep wins
    assert overlapping.typical_state(clock(8.7), "workday") == "busy"  # awake and busy
    assert overlapping.typical_state(clock(10.0), "workday") == "free"


def test_a_model_without_a_sleep_estimate_is_always_awake() -> None:
    model = make_model({}, busy={"workday": (busy_window(13.0, 17.0),)})
    assert model.sleep_profile().window_for("workday") is None
    assert states(model, "workday", [3.0, 14.0]) == ["free", "busy"]


def test_typical_state_is_pure() -> None:
    model = make_model({"all": sleep_window(1.0, 8.5)})
    random.seed(1)
    first = [model.typical_state(clock(h), "workday") for h in (3.0, 9.0)]
    random.seed(99)
    assert first == [model.typical_state(clock(h), "workday") for h in (3.0, 9.0)]


def test_core_and_daytime_plausibility() -> None:
    assert not sleep_core_in_daytime(sleep_window(1.0, 8.5), 30)
    assert not sleep_core_in_daytime(sleep_window(23.0, 7.0), 30)
    assert sleep_core_in_daytime(sleep_window(12.0, 19.5), 30)
    assert sleep_core_in_daytime(sleep_window(10.5, 17.5), 30)
    assert not sleep_core_in_daytime(
        sleep_window(14.0, 17.0 + 8), 30
    )  # core 14:30-00:30: 3.5 of 10h
    window = sleep_window(1.0, 8.5)
    start, end = window.core(30)
    assert (start, end) == (90.0, 480.0) and window.duration_min() == 450.0
    assert not window.crosses_midnight() and sleep_window(23.0, 7.0).crosses_midnight()
    assert sleep_window(1.0, 1.5).core_overlap_with_daytime(30) == 0.0


# --------------------------------------------------------- manual corrections


def override(kind: str, params: dict[str, object], enabled: bool = True) -> OverrideView:
    return OverrideView(
        id=f"{kind}-{len(params)}",
        kind=kind,
        params=params,
        enabled=enabled,
        note=None,
        created_at=datetime(2026, 10, 9, tzinfo=UTC),
    )


def test_a_sleep_correction_replaces_what_the_data_suggests() -> None:
    wrong = (
        "推断的睡眠核心（全部日子）12:17–19:13 落在当地白天，可能是 time.source_timezone 设置不对"
    )
    model = make_model({"all": sleep_window(12.0, 19.5)}, warnings=(wrong, "另一条提示"))
    assert model.typical_state(clock(15.0), "workday") == "deep_sleep"
    fixed = model.with_overrides(
        [override("sleep", {"start": "01:00", "end": "08:30", "day_types": None})]
    )
    assert fixed.typical_state(clock(15.0), "workday") == "free"
    assert fixed.typical_state(clock(3.0), "weekend") == "deep_sleep"
    assert fixed.typical_state(clock(3.0), "holiday") == "deep_sleep"
    assert fixed.sleep_profile().window_for("workday").source == "override"  # type: ignore[union-attr]
    assert fixed.sleep_profile().warnings == ("另一条提示",)  # the zone warning is answered
    assert fixed.applied_overrides == ("sleep-3",)
    assert model.typical_state(clock(15.0), "workday") == "deep_sleep"  # the original is unchanged


def test_a_correction_may_name_day_types() -> None:
    model = make_model({"all": sleep_window(1.0, 8.5)})
    fixed = model.with_overrides(
        [override("sleep", {"start": "02:00", "end": "10:00", "day_types": ["weekend"]})]
    )
    assert fixed.typical_state(clock(9.0), "weekend") == "deep_sleep"
    assert fixed.typical_state(clock(9.0), "workday") == "free"
    assert fixed.typical_state(clock(3.0), "workday") == "deep_sleep"


def test_weekly_busy_corrections_replace_inferred_windows_on_their_weekdays() -> None:
    model = make_model(
        {"all": sleep_window(1.0, 8.5)}, busy={"workday": (busy_window(13.0, 17.0),)}
    )
    fixed = model.with_overrides(
        [override("busy", {"weekdays": [1, 3], "start": "09:00", "end": "11:00"})]
    )
    # Tuesday (1): only the correction; Monday (0): the inferred window; no weekday: inferred
    assert fixed.typical_state(clock(10.0), "workday", weekday=1) == "busy"
    assert fixed.typical_state(clock(14.0), "workday", weekday=1) == "free"
    assert fixed.typical_state(clock(14.0), "workday", weekday=0) == "busy"
    assert fixed.typical_state(clock(10.0), "workday") == "free"
    assert fixed.typical_state(clock(10.0), "workday", weekday=0) == "free"
    (manual,) = fixed.busy_windows("workday", weekday=3)
    assert manual.source == "override" and manual.label() == "09:00–11:00"


def test_disabled_and_unrelated_corrections_change_nothing() -> None:
    model = make_model({"all": sleep_window(1.0, 8.5)})
    same = model.with_overrides(
        [
            override("sleep", {"start": "03:00", "end": "12:00", "day_types": None}, enabled=False),
            override("holiday", {"from": "2026-10-01", "to": "2026-10-07"}),
        ]
    )
    assert same.applied_overrides == ()
    assert same.typical_state(clock(2.0), "workday") == "deep_sleep"
    assert same.sleep_profile().windows.keys() == model.sleep_profile().windows.keys()
