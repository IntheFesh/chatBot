"""The thinned Poisson process over the day: how likely a tick is to bring a message (R-PRO-001)."""

from __future__ import annotations

import math
import random
from datetime import date, datetime, timedelta
from itertools import pairwise
from zoneinfo import ZoneInfo

import pytest

from tests.support.clock import ManualClock
from tests.support.proactive_world import opening_curve, proactive_model
from tests.support.routine import Rig
from twin.schedule.plan_model import DailyPlan
from twin.schedule.proactive.curve import (
    AWAKE_STATES,
    DayGrid,
    build_grid,
    edge_probability,
    slot_of,
    window_share,
)
from twin.services import Services

STEP = timedelta(minutes=5)
SPACING = timedelta(minutes=60)
FRIDAY = date(2026, 10, 9)
SALT = "proactive-tests"


@pytest.fixture
def rig(services: Services, clock: ManualClock) -> Rig:
    return Rig.build(services, clock, proactive_model(), salt=SALT)


def day_plan(rig: Rig, day: date = FRIDAY) -> DailyPlan:
    return rig.planner.ensure(day)


def run_day(grid: DayGrid, needed: int, seed: int, spacing: timedelta = SPACING) -> list[int]:
    """One day of ticks: the steps at which a message fired (a firing needs the spacing)."""
    rng = random.Random(seed)
    fired: list[int] = []
    gap = math.ceil(spacing / grid.step)
    for index in range(grid.size):
        left = needed - len(fired)
        if left <= 0:
            break
        if fired and index - fired[-1] < gap:
            continue
        moment = grid.start + index * grid.step
        if rng.random() < grid.probability(moment, left, spacing):
            fired.append(index)
    return fired


def test_the_grid_gives_no_weight_to_the_steps_she_sleeps(rig: Rig) -> None:
    plan = day_plan(rig)
    grid = build_grid(plan, step=STEP, model=rig.holder["model"], day_type="workday")
    zone = ZoneInfo(plan.timezone)
    assert grid.size == 288
    for index, weight in enumerate(grid.weights):
        moment = grid.start + index * STEP
        segment = plan.segment_at(moment)
        awake = segment is not None and segment.kind in AWAKE_STATES
        assert (weight > 0) == awake, moment.astimezone(zone)
    assert grid.remaining(0) == pytest.approx(sum(grid.weights))
    assert grid.remaining(grid.size) == 0.0


def test_a_day_always_ends_with_the_messages_that_were_owed(rig: Rig) -> None:
    plan = day_plan(rig)
    grid = build_grid(plan, step=STEP, model=rig.holder["model"], day_type="workday")
    for needed in (1, 3, 6):
        for seed in range(60):
            fired = run_day(grid, needed, seed)
            assert len(fired) == needed, (needed, seed)
            gaps = [b - a for a, b in pairwise(fired)]
            assert all(gap >= 12 for gap in gaps)  # an hour apart, 12 steps of five minutes
            assert all(grid.weights[i] > 0 for i in fired)  # only while she is awake


def test_the_expected_number_of_messages_in_the_rest_of_the_day_is_the_number_needed(
    rig: Rig,
) -> None:
    plan = day_plan(rig)
    grid = build_grid(plan, step=STEP, model=rig.holder["model"], day_type="workday")
    no_spacing = timedelta(0)
    needed = 3
    total = 0
    runs = 600
    for seed in range(runs):
        rng = random.Random(seed)
        count = 0
        for index in range(grid.size):
            left = needed - count
            if left <= 0:
                break
            moment = grid.start + index * STEP
            if rng.random() < grid.probability(moment, left, no_spacing):
                count += 1
        total += count
    assert total / runs == pytest.approx(needed, abs=0.02)  # exactly the number owed


def test_the_messages_follow_the_curve_of_her_openings(rig: Rig) -> None:
    peak = {48: 3.0, 49: 3.0}  # 12:00-12:30, far above everything else
    model = proactive_model(opening_curve(base=0.05, peaks=peak))
    rig.use(model)
    plan = day_plan(rig)
    grid = build_grid(plan, step=STEP, model=model, day_type="workday")
    zone = ZoneInfo(plan.timezone)
    inside = other = 0
    for seed in range(300):
        for index in run_day(grid, 1, seed, timedelta(0)):
            local = (grid.start + index * STEP).astimezone(zone)
            if local.hour == 12 and local.minute < 30:
                inside += 1
            else:
                other += 1
    assert inside / (inside + other) > 0.6  # the peak is half an hour of an awake day


def test_the_probability_is_certain_when_no_later_step_can_fit_what_is_owed(rig: Rig) -> None:
    plan = day_plan(rig)
    grid = build_grid(plan, step=STEP, model=rig.holder["model"], day_type="workday")
    last = max(index for index, weight in enumerate(grid.weights) if weight > 0)
    late = grid.start + last * STEP
    assert grid.probability(late, 1, SPACING) == 1.0
    assert grid.probability(late, 2, SPACING) == 1.0  # two owed, and an hour apart: no room
    early = grid.start + (last - 100) * STEP
    assert 0.0 < grid.probability(early, 1, SPACING) < 1.0


def test_nothing_fires_while_nothing_is_owed_or_outside_the_day(rig: Rig) -> None:
    plan = day_plan(rig)
    grid = build_grid(plan, step=STEP, model=rig.holder["model"], day_type="workday")
    noon = grid.start + 150 * STEP
    assert grid.probability(noon, 0, SPACING) == 0.0
    assert grid.probability(noon, -2, SPACING) == 0.0
    assert grid.probability(grid.start - STEP, 1, SPACING) == 0.0
    assert grid.probability(grid.start + grid.size * STEP, 1, SPACING) == 0.0
    sleeping = grid.start + 3 * 12 * STEP  # 03:00
    assert grid.weights[grid.index(sleeping)] == 0.0
    assert grid.probability(sleeping, 2, SPACING) == 0.0


def test_a_day_without_a_routine_model_is_spread_evenly_over_the_waking_hours(
    services: Services, clock: ManualClock
) -> None:
    rig = Rig.build(services, clock, proactive_model(), salt=SALT)
    plan = day_plan(rig)
    grid = build_grid(plan, step=STEP, model=None, day_type="workday")
    positive = {weight for weight in grid.weights if weight > 0}
    assert len(positive) == 1  # every waking step weighs the same


def test_the_grid_of_a_day_with_a_clock_change_has_the_hours_it_really_has(
    services: Services, clock: ManualClock
) -> None:
    rig = Rig.build(services, clock, proactive_model(), zone="America/Chicago", salt=SALT)
    sunday = date(2026, 11, 1)  # the clocks go back: 25 hours
    plan = day_plan(rig, sunday)
    grid = build_grid(plan, step=STEP, model=rig.holder["model"], day_type="weekend")
    assert grid.size == 25 * 12


def test_the_probability_of_an_edge_message_is_her_opening_rate_spread_over_the_ticks() -> None:
    model = proactive_model(opening_curve(base=0.3, peaks={}))
    zone = ZoneInfo("America/Chicago")
    moment = datetime(2026, 10, 10, 4, 30, tzinfo=zone)
    assert edge_probability(model, moment, zone, "workday", STEP) == pytest.approx(0.1)
    assert edge_probability(None, moment, zone, "workday", STEP) == 0.0
    big = proactive_model(opening_curve(base=9.0, peaks={}))
    assert edge_probability(big, moment, zone, "workday", STEP) == 1.0


def test_the_share_of_a_window_is_the_sum_of_the_rates_it_covers() -> None:
    model = proactive_model(opening_curve(base=0.1, peaks={}))
    zone = ZoneInfo("America/Chicago")
    start = datetime(2026, 10, 9, 12, 0, tzinfo=zone)
    assert window_share(
        model, zone, "workday", start, start + timedelta(hours=1), fallback=0.5
    ) == (pytest.approx(0.4))
    half = window_share(model, zone, "workday", start, start + timedelta(minutes=30), fallback=0.5)
    assert half == pytest.approx(0.2)
    assert (
        window_share(None, zone, "workday", start, start + timedelta(hours=1), fallback=0.5) == 0.5
    )
    rich = proactive_model(opening_curve(base=2.0, peaks={}))
    assert (
        window_share(rich, zone, "workday", start, start + timedelta(hours=1), fallback=0.5) == 1.0
    )


def test_the_slot_of_a_moment_is_read_on_her_clock() -> None:
    zone = ZoneInfo("Asia/Shanghai")
    moment = datetime(2026, 10, 9, 4, 20, tzinfo=ZoneInfo("UTC"))  # 12:20 in Shanghai
    assert slot_of(moment, zone) == 49
