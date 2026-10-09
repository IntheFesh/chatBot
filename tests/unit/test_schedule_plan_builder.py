"""Making a day plan: sleep, busy periods, meals, the proactive quota and the greeting.

R-SCH-004 and R-PRO-002 (the quota), R-SCH-003 (the plans of the two days of the clock change).
The builder is a pure function of its request, so every test says exactly what it is given.
"""

from __future__ import annotations

import random
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from statistics import mean
from zoneinfo import ZoneInfo

import pytest

from tests.support.routine import (
    SPREAD,
    busy_window,
    fixed_model,
    make_model,
    rate_curve,
    sleep_window,
    student_model,
)
from twin.config.settings import ScheduleConfig
from twin.profile.activity_model import ActivityModel
from twin.profile.overrides import OverrideView
from twin.schedule.plan_builder import (
    PlanRequest,
    QuotaRange,
    build_plan,
    draw_count,
    duration_bounds,
    fingerprint,
    model_fingerprint,
    plan_seed,
    solve_rate,
    stream,
)
from twin.schedule.plan_model import DailyPlan, busy_ref, parse_busy_ref
from twin.schedule.wallclock import day_bounds_utc, to_utc

CHICAGO = ZoneInfo("America/Chicago")
SHANGHAI = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 10, 9, 5, 0, tzinfo=UTC)


def local(moment: datetime, zone: ZoneInfo = CHICAGO) -> str:
    return moment.astimezone(zone).strftime("%m-%d %H:%M")


def request(
    day: date,
    model: ActivityModel | None,
    *,
    zone: ZoneInfo = CHICAGO,
    salt: str = "salt",
    day_type: str | None = None,
    next_day_type: str | None = None,
    **fields: object,
) -> PlanRequest:
    def kind(d: date) -> str:
        return "workday" if d.weekday() < 5 else "weekend"

    start, _ = day_bounds_utc(day, zone)
    base = PlanRequest(
        day=day,
        zone=zone,
        day_type=day_type or kind(day),
        next_day_type=next_day_type or kind(day + timedelta(days=1)),
        seed=plan_seed(day, salt),
        model=model,
        effective_from=start,
        go_to_bed_at=None,
        quota=QuotaRange(1, 6),
        last_greeting_at=None,
        previous_day=None,
        same_day=None,
        reason="daily",
        config=ScheduleConfig(),
        inputs_hash="inputs",
        routine_hash=model_fingerprint(model),
        created_at=start,
    )
    return replace(base, **fields)  # type: ignore[arg-type]


def states(plan: DailyPlan, zone: ZoneInfo = CHICAGO) -> list[tuple[str, str, str]]:
    return [(s.kind, local(s.start, zone), local(s.end, zone)) for s in plan.segments]


# ---------------------------------------------------------------------- the day


def test_a_plan_decides_sleep_busy_meals_quota_and_greeting_of_a_workday() -> None:
    day = date(2026, 10, 9)  # a Friday
    plan = build_plan(request(day, fixed_model(), next_day_type="workday"))
    assert (plan.local_date, plan.timezone, plan.day_type) == (day, "America/Chicago", "workday")
    assert plan.morning is not None and plan.night is not None
    assert (local(plan.morning.onset), local(plan.morning.wake)) == ("10-08 23:30", "10-09 07:30")
    assert (local(plan.night.onset), local(plan.night.wake)) == ("10-09 23:30", "10-10 07:30")
    assert [(k, a, b) for k, a, b in states(plan)] == [
        ("deep_sleep", "10-09 00:00", "10-09 07:00"),
        ("sleep_edge", "10-09 07:00", "10-09 07:30"),
        ("free", "10-09 07:30", "10-09 13:00"),
        ("busy", "10-09 13:00", "10-09 17:00"),
        ("free", "10-09 17:00", "10-09 23:30"),
        ("sleep_edge", "10-09 23:30", "10-10 00:00"),
        ("deep_sleep", "10-10 00:00", "10-10 07:00"),
        ("sleep_edge", "10-10 07:00", "10-10 07:30"),
    ]
    assert plan.ends_at == plan.night.wake
    assert plan.effective_from == day_bounds_utc(day, CHICAGO)[0]
    assert plan.quota.enabled and 1 <= plan.quota.total <= 6
    assert plan.greeting.allowed and plan.greeting.reason == "ok"
    assert plan.greeting.earliest == plan.morning.wake + timedelta(minutes=5)
    assert plan.greeting.latest == plan.morning.wake + timedelta(minutes=40)
    assert plan.warnings == ()


def test_the_states_of_a_plan_tile_it_without_a_hole_or_an_overlap() -> None:
    plan = build_plan(request(date(2026, 10, 9), student_model()))
    segments = plan.segments
    assert segments[0].start == plan.effective_from and segments[-1].end == plan.ends_at
    assert all(a.end == b.start for a, b in pairwise(segments))
    assert all(a.kind != b.kind or a.busy != b.busy for a, b in pairwise(segments))
    assert all(s.start < s.end for s in segments)
    for moment in (plan.effective_from, plan.ends_at - timedelta(seconds=1)):
        assert plan.covers(moment)
    assert not plan.covers(plan.ends_at) and not plan.covers(
        plan.effective_from - timedelta(seconds=1)
    )


def test_sleep_beats_busy_beats_free_where_they_meet() -> None:
    """A busy window that runs into the night is cut by the sleep."""
    model = make_model(
        {"all": sleep_window(15.0, 23.0)},  # sleeps 15:00 to 23:00
        {"workday": (busy_window(13, 17),)},
    )
    plan = build_plan(request(date(2026, 10, 9), model, next_day_type="workday"))
    kinds = [(k, a[6:], b[6:]) for k, a, b in states(plan)]
    assert ("busy", "13:00", "15:00") in kinds  # the first two hours of the window
    assert ("sleep_edge", "15:00", "15:30") in kinds and ("deep_sleep", "15:30", "22:30") in kinds
    assert not any(k == "busy" and a == "15:00" for k, a, _ in kinds)


# ------------------------------------------------------------- seed, reproducible


def test_the_same_inputs_draw_the_same_plan_and_the_seed_is_in_it() -> None:
    day = date(2026, 10, 9)
    first = build_plan(request(day, student_model()))
    second = build_plan(request(day, student_model()))
    assert first.to_json() == second.to_json()
    assert first.seed == plan_seed(day, "salt")
    other_salt = build_plan(request(day, student_model(), salt="another install"))
    assert other_salt.seed != first.seed
    next_day = build_plan(request(date(2026, 10, 10), student_model()))
    assert next_day.seed != first.seed


def test_the_seed_is_a_hash_of_the_date_and_the_salt() -> None:
    seeds = {plan_seed(date(2026, 10, d), "s") for d in range(1, 29)}
    assert len(seeds) == 28 and all(0 <= s < 2**62 for s in seeds)
    assert plan_seed(date(2026, 10, 9), "a") != plan_seed(date(2026, 10, 9), "b")
    assert plan_seed(date(2026, 10, 9), "a") == plan_seed(date(2026, 10, 9), "a")


def test_every_part_has_its_own_stream_so_a_changed_range_does_not_move_the_night() -> None:
    day = date(2026, 10, 9)
    narrow = build_plan(request(day, student_model(), quota=QuotaRange(1, 2)))
    wide = build_plan(request(day, student_model(), quota=QuotaRange(1, 6)))
    assert narrow.morning == wide.morning and narrow.night == wide.night
    assert narrow.meals == wide.meals and narrow.busy == wide.busy
    assert stream(1, "sleep:night").random() != stream(1, "quota").random()
    assert stream(1, "quota").random() == stream(1, "quota").random()


def test_nights_vary_from_day_to_day_when_her_history_varies() -> None:
    wakes = {
        build_plan(request(date(2026, 9, 1) + timedelta(days=n), student_model())).night.wake.time()  # type: ignore[union-attr]
        for n in range(30)
    }
    assert len(wakes) > 3
    fixed = {
        build_plan(request(date(2026, 9, 1) + timedelta(days=n), fixed_model())).night.wake.time()  # type: ignore[union-attr]
        for n in range(30)
    }
    assert len(fixed) == 1


# ------------------------------------------------------------------------ sleep


def test_the_length_of_every_night_lies_between_the_5th_and_95th_percentile() -> None:
    window = sleep_window(23.5, 7.5, spread=SPREAD)
    low, high = duration_bounds(window)
    assert 420 < low < 480 < high < 540  # around the median of 8 hours
    model = make_model({"all": window})
    for offset in range(120):  # also the clock change of 2026-11-01: lengths are wall-clock lengths
        plan = build_plan(
            request(date(2026, 9, 1) + timedelta(days=offset), model, next_day_type="workday")
        )
        for episode in plan.episodes:
            onset = episode.onset.astimezone(CHICAGO).replace(tzinfo=None)
            wake = episode.wake.astimezone(CHICAGO).replace(tzinfo=None)
            minutes = (wake - onset).total_seconds() / 60
            assert low <= minutes <= high, (plan.local_date, minutes)
    assert duration_bounds(sleep_window(23.5, 7.5)) == (480.0, 480.0)  # a fixed night


def test_a_manual_sleep_correction_wins_over_the_history() -> None:
    model = student_model().with_overrides(
        [
            OverrideView(
                "o1",
                "sleep",
                {"start": "02:00", "end": "10:00", "day_types": None},
                True,
                None,
                NOW,
            )
        ]
    )
    for offset in range(10):
        plan = build_plan(request(date(2026, 10, 1) + timedelta(days=offset), model))
        assert plan.morning is not None and plan.night is not None
        assert local(plan.morning.wake)[6:] == "10:00" and local(plan.night.onset)[6:] == "02:00"
        assert plan.morning.source == "override" and plan.night.source == "override"
    # the correction can be limited to one day type
    weekend_only = student_model().with_overrides(
        [
            OverrideView(
                "o2",
                "sleep",
                {"start": "03:00", "end": "11:00", "day_types": ["weekend"]},
                True,
                None,
                NOW,
            )
        ]
    )
    friday = build_plan(request(date(2026, 10, 9), weekend_only))
    assert friday.morning is not None and friday.night is not None
    assert friday.morning.source == "days" and local(friday.morning.wake)[6:] != "11:00"
    assert local(friday.night.onset) == "10-10 03:00"  # the night before a weekend day
    assert friday.night.source == "override" and local(friday.night.wake)[6:] == "11:00"
    saturday = build_plan(request(date(2026, 10, 10), weekend_only))
    assert saturday.morning is not None and local(saturday.morning.wake)[6:] == "11:00"
    assert saturday.morning.source == "override"


def test_each_day_type_draws_from_its_own_sleep_window() -> None:
    """A night belongs to the day she wakes on: Friday night ends a Saturday morning."""
    model = student_model()
    friday = build_plan(request(date(2026, 10, 9), model))  # workday, next day is a weekend day
    assert friday.morning is not None and friday.night is not None
    assert local(friday.morning.wake)[6:] in {"07:00", "07:15", "07:30", "07:45", "08:00"}
    assert local(friday.night.wake)[6:] in {"08:30", "08:45", "09:00", "09:15", "09:30"}
    assert friday.night.onset.astimezone(CHICAGO).date() == date(2026, 10, 10)  # after midnight
    holiday = build_plan(
        request(date(2026, 10, 12), model, day_type="holiday", next_day_type="workday")
    )
    assert holiday.morning is not None
    assert local(holiday.morning.wake)[6:] in {"08:30", "08:45", "09:00", "09:15", "09:30"}


def test_the_night_starts_a_working_day_after_the_wake_up() -> None:
    """A late riser's evening cannot start the next night inside the minimum awake time."""
    model = make_model({"all": sleep_window(11.0, 12.0, spread=SPREAD)})  # an hour of sleep at noon
    plan = build_plan(request(date(2026, 10, 9), model))
    assert plan.morning is not None and plan.night is not None
    assert plan.night.onset - plan.morning.wake >= timedelta(hours=ScheduleConfig().min_awake_h)


def test_a_missing_model_or_sleep_window_gives_a_plan_with_a_warning_and_no_sleep() -> None:
    plan = build_plan(request(date(2026, 10, 9), None))
    assert plan.morning is None and plan.night is None and plan.busy == ()
    assert plan.warnings == ("no_routine_model",)
    assert [s.kind for s in plan.segments] == ["free"]
    assert plan.quota.mean_source == "range_midpoint" and plan.quota.mean_target == 3.5
    assert plan.greeting.reason == "no_wake_up_planned" and not plan.greeting.allowed
    low = student_model()
    low = replace(low, sleep=replace(low.sleep, confidence="low"))
    assert "sleep_confidence_low" in build_plan(request(date(2026, 10, 9), low)).warnings
    empty = make_model({})
    assert "no_sleep_window" in build_plan(request(date(2026, 10, 9), empty)).warnings


# ------------------------------------------------------------------------- busy


def test_busy_periods_come_with_the_reference_of_their_latency_distribution() -> None:
    model = fixed_model()
    plan = build_plan(request(date(2026, 10, 9), model))
    (span,) = plan.busy
    assert (local(span.start), local(span.end), span.label) == (
        "10-09 13:00",
        "10-09 17:00",
        "13:00–17:00",
    )
    assert (span.latency_median_s, span.latency_p90_s) == (1500.0, 1500.0)
    assert span.latency_ref == busy_ref("workday", 4, 0) == "workday|4|0"
    day_type, weekday, index = parse_busy_ref(span.latency_ref)
    window = model.busy_windows(day_type, weekday)[index]  # the reference leads to the window
    assert window.latency.median() == 1500.0
    assert parse_busy_ref(busy_ref("holiday", None, 2)) == ("holiday", None, 2)
    weekend = build_plan(request(date(2026, 10, 10), model))
    assert weekend.busy == ()


def test_a_weekly_busy_correction_replaces_the_inferred_windows_of_its_weekday() -> None:
    manual = busy_window(9, 11, weekdays=(1,))  # Tuesdays
    model = fixed_model(manual_busy=(manual,))
    tuesday = build_plan(request(date(2026, 10, 6), model))
    assert [(local(b.start)[6:], local(b.end)[6:], b.source) for b in tuesday.busy] == [
        ("09:00", "11:00", "override")
    ]
    assert tuesday.busy[0].latency_ref == "workday|1|0"
    wednesday = build_plan(request(date(2026, 10, 7), model))
    assert [local(b.start)[6:] for b in wednesday.busy] == ["13:00"]


# ------------------------------------------------------------------------ quota


@pytest.mark.parametrize("target", [3.6, 2.0, 5.0])
def test_the_mean_quota_follows_her_real_rate_and_stays_in_the_range(target: float) -> None:
    """R-PRO-002: over many days the mean is her real daily rate, +-10%, inside [1, 6]."""
    model = fixed_model(initiations=target)
    counts = [
        build_plan(request(date(2024, 1, 1) + timedelta(days=n), model)).quota.total
        for n in range(1500)
    ]
    assert min(counts) >= 1 and max(counts) <= 6
    assert abs(mean(counts) - target) <= 0.1 * target, mean(counts)
    assert len(set(counts)) >= 4  # a distribution, not a constant


def test_a_rate_outside_the_range_is_pulled_to_its_edge_and_a_changed_range_is_obeyed() -> None:
    high = fixed_model(initiations=9.0)
    assert {
        build_plan(request(date(2026, 10, 1) + timedelta(days=n), high)).quota.total
        for n in range(40)
    } == {6}
    low = fixed_model(initiations=0.1)
    assert {
        build_plan(request(date(2026, 10, 1) + timedelta(days=n), low)).quota.total
        for n in range(40)
    } == {1}
    model = fixed_model(initiations=3.6)
    counts = [
        build_plan(
            request(date(2026, 10, 1) + timedelta(days=n), model, quota=QuotaRange(2, 5))
        ).quota
        for n in range(300)
    ]
    assert min(q.total for q in counts) >= 2 and max(q.total for q in counts) <= 5
    assert all((q.minimum, q.maximum) == (2, 5) for q in counts)
    assert abs(mean(q.total for q in counts) - 3.6) <= 0.36


def test_proactive_messages_switched_off_mean_a_quota_of_zero() -> None:
    off = build_plan(
        request(date(2026, 10, 9), fixed_model(), quota=QuotaRange(1, 6, enabled=False))
    )
    assert (off.quota.total, off.quota.for_plan, off.quota.enabled) == (0, 0, False)
    assert off.quota.mean_source == "disabled"
    zero = build_plan(request(date(2026, 10, 9), fixed_model(), quota=QuotaRange(0, 0)))
    assert zero.quota.total == 0 and not zero.quota.enabled


def test_the_poisson_rate_is_solved_for_the_clipped_mean() -> None:
    rng = random.Random(7)
    for target in (1.5, 3.6, 4.4):
        rate = solve_rate(target, 1, 6)
        draws = [draw_count(rng, rate, 1, 6) for _ in range(20000)]
        assert abs(mean(draws) - target) < 0.05
    assert solve_rate(0.5, 1, 6) == 0.0 and draw_count(rng, 0.0, 1, 6) == 1
    assert draw_count(rng, 500.0, 1, 6) == 6
    assert solve_rate(2.0, 3, 3) == 0.0


def test_a_plan_made_in_the_middle_of_the_day_gets_the_share_of_the_awake_time_left() -> None:
    day = date(2026, 10, 9)
    model = fixed_model(initiations=6.0)
    full = build_plan(request(day, model, quota=QuotaRange(6, 6)))
    afternoon = to_utc(day, 15 * 60, CHICAGO)
    late = build_plan(request(day, model, quota=QuotaRange(6, 6), effective_from=afternoon))
    assert full.quota.total == full.quota.for_plan == 6
    assert late.quota.total == 6 and late.quota.for_plan == round(6 * (8.5 / 16.0))  # 15:00-23:30
    assert late.effective_from == afternoon


# ------------------------------------------------------------------------ meals


def test_meals_follow_her_activity_peaks_and_fall_back_to_the_common_hours() -> None:
    model = fixed_model(rate=rate_curve({49: 2.0, 75: 2.0}))  # peaks at 12:15 and 18:45
    plan = build_plan(request(date(2026, 10, 9), model, config=ScheduleConfig(meal_jitter_min=0)))
    meals = {m.kind: m for m in plan.meals}
    assert meals["lunch"].source == "history" and local(meals["lunch"].at)[6:] == "12:22"
    assert meals["dinner"].source == "history" and local(meals["dinner"].at)[6:] == "18:52"
    assert meals["breakfast"].source == "default" and local(meals["breakfast"].at)[6:] == "07:30"
    jittered = build_plan(request(date(2026, 10, 9), model))  # the default spread of 15 minutes
    lunch = next(m for m in jittered.meals if m.kind == "lunch")
    assert abs((lunch.at - meals["lunch"].at).total_seconds()) <= 45 * 60
    flat = fixed_model(rate=rate_curve())
    assert {m.source for m in build_plan(request(date(2026, 10, 9), flat)).meals} == {"default"}
    assert {m.source for m in build_plan(request(date(2026, 10, 9), None)).meals} == {"default"}


def test_a_meal_during_sleep_is_left_out() -> None:
    night_worker = make_model(
        {"all": sleep_window(6.0, 15.0)}
    )  # sleeps through breakfast and lunch
    plan = build_plan(
        request(date(2026, 10, 9), night_worker, config=ScheduleConfig(meal_jitter_min=0))
    )
    assert [m.kind for m in plan.meals] == ["dinner"]


# --------------------------------------------------------------------- greeting


def test_the_greeting_waits_eighteen_hours_after_the_last_one() -> None:
    day = date(2026, 10, 9)
    model = fixed_model()
    wake = to_utc(day, 7 * 60 + 30, CHICAGO)

    def greeting(last: datetime | None, created: datetime | None = None):  # type: ignore[no-untyped-def]
        return build_plan(
            request(
                day, model, last_greeting_at=last, created_at=created or wake - timedelta(hours=8)
            )
        ).greeting

    assert greeting(None).allowed
    assert greeting(wake - timedelta(hours=30)).allowed
    # 17h before the earliest greeting time: too soon
    soon = greeting(wake + timedelta(minutes=5) - timedelta(hours=17, minutes=59))
    assert not soon.allowed and soon.reason == "within_min_gap_of_the_last_greeting"
    edge = greeting(wake + timedelta(minutes=5) - timedelta(hours=18))
    assert edge.allowed  # exactly 18 hours is enough
    sent = greeting(wake + timedelta(minutes=10))
    assert not sent.allowed and sent.reason == "already_sent_after_this_wake_up"
    over = greeting(None, created=wake + timedelta(minutes=41))
    assert not over.allowed and over.reason == "wake_up_already_passed"
    inside = greeting(None, created=wake + timedelta(minutes=20))
    assert inside.allowed and inside.earliest == wake + timedelta(minutes=20)


def test_the_greeting_window_is_configurable() -> None:
    config = ScheduleConfig(greeting_window_min=[10, 20], greeting_min_gap_h=1)
    day = date(2026, 10, 9)
    plan = build_plan(request(day, fixed_model(), config=config))
    wake = to_utc(day, 7 * 60 + 30, CHICAGO)
    assert (plan.greeting.earliest, plan.greeting.latest) == (
        wake + timedelta(minutes=10),
        wake + timedelta(minutes=20),
    )
    with pytest.raises(ValueError, match="earliest < latest"):
        ScheduleConfig(greeting_window_min=[40, 5])
    with pytest.raises(ValueError, match="earliest < latest"):
        ScheduleConfig(greeting_window_min=[5])


# --------------------------------------------------------------- daylight saving


def test_the_plan_of_the_spring_day_has_23_hours_and_no_hole() -> None:
    day = date(2026, 3, 8)  # a Sunday: the clock jumps from 02:00 to 03:00
    plan = build_plan(request(day, fixed_model(), day_type="weekend", next_day_type="workday"))
    assert plan.morning is not None and plan.night is not None
    # she went to bed at 23:30 CST and wakes at 07:30 CDT: seven hours slept, not eight
    assert plan.morning.onset == datetime(2026, 3, 8, 5, 30, tzinfo=UTC)
    assert plan.morning.wake == datetime(2026, 3, 8, 12, 30, tzinfo=UTC)
    assert plan.morning.duration == timedelta(hours=7)
    assert plan.night.onset == datetime(2026, 3, 9, 4, 30, tzinfo=UTC)  # 23:30 CDT
    assert plan.night.wake == datetime(2026, 3, 9, 12, 30, tzinfo=UTC)
    day_start, day_end = day_bounds_utc(day, CHICAGO)
    assert day_end - day_start == timedelta(hours=23)
    inside = [s for s in plan.segments if s.start < day_end]
    assert inside[0].start == day_start
    assert all(a.end == b.start for a, b in pairwise(plan.segments))
    assert plan.extra["dst"] == []  # nothing she does falls into the missing hour


def test_a_time_inside_the_missing_hour_happens_an_hour_later_and_is_recorded() -> None:
    day = date(2026, 3, 8)
    night_owl = make_model({"all": sleep_window(2.5, 10.5)})  # falls asleep at 02:30
    plan = build_plan(request(day, night_owl, day_type="weekend", next_day_type="weekend"))
    assert plan.morning is not None
    assert plan.morning.onset == datetime(2026, 3, 8, 8, 30, tzinfo=UTC)  # 02:30 CST = 03:30 CDT
    assert local(plan.morning.onset) == "03-08 03:30"
    notes = [n for n in plan.extra["dst"] if n["what"] == "onset"]
    assert len(notes) == 1  # the draws that were turned down leave no note
    assert notes[0]["rule"] == "gap" and notes[0]["asked"] == "2026-03-08 02:30"
    assert notes[0]["read_as"] == "2026-03-08 03:30"
    assert plan.morning.wake == datetime(2026, 3, 8, 15, 30, tzinfo=UTC)  # 10:30 CDT


def test_the_plan_of_the_autumn_day_has_25_hours_and_no_overlap() -> None:
    day = date(2026, 11, 1)  # a Sunday: the clock goes back from 02:00 to 01:00
    plan = build_plan(request(day, fixed_model(), day_type="weekend", next_day_type="workday"))
    assert plan.morning is not None and plan.night is not None
    # to bed at 23:30 CDT, up at 07:30 CST: nine hours slept, not eight
    assert plan.morning.onset == datetime(2026, 11, 1, 4, 30, tzinfo=UTC)
    assert plan.morning.wake == datetime(2026, 11, 1, 13, 30, tzinfo=UTC)
    assert plan.morning.duration == timedelta(hours=9)
    assert plan.night.onset == datetime(2026, 11, 2, 5, 30, tzinfo=UTC)  # 23:30 CST
    day_start, day_end = day_bounds_utc(day, CHICAGO)
    assert day_end - day_start == timedelta(hours=25)
    assert all(a.end == b.start for a, b in pairwise(plan.segments))
    instants = [s.start for s in plan.segments]
    assert instants == sorted(set(instants))  # no instant starts two states


def test_a_time_inside_the_repeated_hour_is_its_first_occurrence() -> None:
    day = date(2026, 11, 1)
    night_owl = make_model({"all": sleep_window(1.5, 9.5)})  # falls asleep at 01:30
    plan = build_plan(request(day, night_owl, day_type="weekend", next_day_type="weekend"))
    assert plan.morning is not None
    assert plan.morning.onset == datetime(2026, 11, 1, 6, 30, tzinfo=UTC)  # 01:30 CDT, first pass
    assert plan.morning.wake == datetime(2026, 11, 1, 15, 30, tzinfo=UTC)  # 09:30 CST
    (note,) = [n for n in plan.extra["dst"] if n["what"] == "onset"]
    assert note["rule"] == "repeated" and note["asked"] == "2026-11-01 01:30"
    assert note["read_as"] == "2026-11-01 01:30"
    states_of_day = {s.kind for s in plan.segments}
    assert {"deep_sleep", "sleep_edge", "free"} <= states_of_day


def test_a_zone_without_daylight_saving_has_plain_days() -> None:
    plan = build_plan(request(date(2026, 3, 8), fixed_model(), zone=SHANGHAI, day_type="weekend"))
    assert plan.timezone == "Asia/Shanghai"
    assert plan.morning is not None and plan.morning.duration == timedelta(hours=8)
    assert plan.extra["dst"] == []


# ------------------------------------------------------------------ fingerprints


def test_a_fingerprint_changes_with_anything_that_decides_the_plan() -> None:
    base = model_fingerprint(student_model())
    assert base == model_fingerprint(student_model())
    assert base != model_fingerprint(fixed_model())
    assert base != model_fingerprint(student_model(initiations=4.0))
    assert base != model_fingerprint(
        student_model(manual_busy=(busy_window(9, 10, weekdays=(1,)),))
    )
    assert model_fingerprint(None) == "no-model"
    assert fingerprint("a", 1, [2]) == fingerprint("a", 1, [2]) != fingerprint("a", 1, [3])
