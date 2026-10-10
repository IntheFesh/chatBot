"""The fixed moments of her day and the follow-up slot (R-PRO-004, R-MEM-006)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from itertools import pairwise

import pytest

from tests.support.clock import ManualClock
from tests.support.proactive_world import opening_curve, proactive_model
from tests.support.routine import Rig
from twin.config.settings import ProactiveConfig
from twin.memory.records import FollowupRecord
from twin.schedule.plan_builder import QuotaRange
from twin.schedule.plan_model import DailyPlan
from twin.schedule.proactive.slots import followup_slot, keep_apart, routine_slots
from twin.schedule.proactive.store import NewCandidate
from twin.schedule.proactive.types import PRIORITY, TriggerKind
from twin.services import Services

FRIDAY = date(2026, 10, 9)
SALT = "proactive-tests"
CONFIG = ProactiveConfig()


def make(
    services: Services,
    clock: ManualClock,
    *,
    curve: tuple[float, ...] | None = None,
    quota: QuotaRange | None = None,
    day: date = FRIDAY,
) -> tuple[Rig, DailyPlan]:
    rig = Rig.build(
        services, clock, proactive_model(curve), quota=quota or QuotaRange(6, 6), salt=SALT
    )
    clock.set_time(rig.at(day.year, day.month, day.day, 0, 10))  # the plan is made at midnight
    return rig, rig.planner.ensure(day)


def slots_of(rig: Rig, plan: DailyPlan) -> dict[str, object]:
    found = routine_slots(plan, model=rig.holder["model"], day_type="workday", config=CONFIG)
    return {slot.key: slot for slot in found}


def slot(key: str, kind: TriggerKind, minute: int, window: int = 30) -> NewCandidate:
    start = datetime(2026, 10, 9, 12, 0, tzinfo=UTC) + timedelta(minutes=minute)
    return NewCandidate(
        local_date=FRIDAY,
        timezone="America/Chicago",
        key=key,
        kind=kind,
        priority=PRIORITY[kind],
        planned_at=start,
        window_end=start + timedelta(minutes=window),
    )


def test_two_slots_closer_than_the_spacing_are_moved_apart_or_dropped() -> None:
    spacing = timedelta(minutes=60)
    greeting = slot("greeting", TriggerKind.GREETING, 0, 30)
    early_meal = slot("meal:breakfast", TriggerKind.MEAL, -20, 90)  # before the greeting ...
    late_meal = slot("meal:lunch", TriggerKind.MEAL, 20, 30)  # ... and one with no room at all
    far = slot("meal:dinner", TriggerKind.MEAL, 300, 30)
    kept = keep_apart([far, late_meal, greeting, early_meal], spacing)
    assert [s.key for s in kept] == ["greeting", "meal:breakfast", "meal:dinner"]
    assert [int((s.planned_at - greeting.planned_at).total_seconds() // 60) for s in kept] == [
        0,
        60,
        300,
    ]
    assert all(s.planned_at <= s.window_end for s in kept)
    assert keep_apart([], spacing) == []


def test_a_slot_that_still_fits_its_window_is_moved_not_dropped() -> None:
    spacing = timedelta(minutes=60)
    greeting = slot("greeting", TriggerKind.GREETING, 0, 30)
    meal = slot("meal:breakfast", TriggerKind.MEAL, 10, 120)
    kept = keep_apart([meal, greeting], spacing)
    assert [s.key for s in kept] == ["greeting", "meal:breakfast"]
    assert (
        kept[1].planned_at == greeting.planned_at + spacing
        and kept[1].window_end == meal.window_end
    )


def test_a_day_with_busy_mealtimes_gets_a_greeting_first_then_meals_and_goodnight(
    services: Services, clock: ManualClock
) -> None:
    rig, plan = make(services, clock, curve=opening_curve(base=2.0, peaks={}))
    slots = routine_slots(plan, model=rig.holder["model"], day_type="workday", config=CONFIG)
    keys = [slot.key for slot in slots]
    assert keys[0] == "greeting" and {"meal:lunch", "meal:dinner", "bedtime"} <= set(keys)
    assert set(keys) <= {"greeting", "meal:breakfast", "meal:lunch", "meal:dinner", "bedtime"}
    gaps = [b.planned_at - a.planned_at for a, b in pairwise(slots)]
    assert all(gap >= timedelta(minutes=CONFIG.min_spacing_min) for gap in gaps)
    greeting = slots[0]
    assert plan.greeting.earliest is not None and plan.greeting.latest is not None
    assert (
        plan.greeting.earliest <= greeting.planned_at <= greeting.window_end == plan.greeting.latest
    )
    assert greeting.priority == PRIORITY[TriggerKind.GREETING]
    for slot in slots:
        assert slot.local_date == FRIDAY and slot.timezone == plan.timezone
        assert slot.planned_at <= slot.window_end
        assert slot.plan_id == plan.id


def test_a_meal_goes_out_around_the_mealtime_and_never_while_she_sleeps(
    services: Services, clock: ManualClock
) -> None:
    rig, plan = make(services, clock, curve=opening_curve(base=2.0, peaks={}))
    around = timedelta(minutes=CONFIG.meal_window_min)
    for slot in routine_slots(plan, model=rig.holder["model"], day_type="workday", config=CONFIG):
        if slot.kind is not TriggerKind.MEAL:
            continue
        meal = next(m for m in plan.meals if f"meal:{m.kind}" == slot.key)
        assert meal.at - around <= slot.planned_at <= meal.at + around == slot.window_end
        segment = plan.segment_at(slot.planned_at)
        assert segment is not None and segment.kind in ("free", "busy")


def test_goodnight_is_fifteen_to_sixty_minutes_before_she_falls_asleep(
    services: Services, clock: ManualClock
) -> None:
    rig, plan = make(services, clock, curve=opening_curve(base=2.0, peaks={}))
    assert plan.night is not None
    found = slots_of(rig, plan)["bedtime"]
    onset = plan.night.onset
    assert onset - timedelta(minutes=60) <= found.planned_at <= onset - timedelta(minutes=15)  # type: ignore[attr-defined]
    assert found.window_end == onset - timedelta(minutes=15)  # type: ignore[attr-defined]


def test_the_same_day_always_gets_the_same_slots(services: Services, clock: ManualClock) -> None:
    rig, plan = make(services, clock, curve=opening_curve(base=0.6, peaks={}))
    first = routine_slots(plan, model=rig.holder["model"], day_type="workday", config=CONFIG)
    again = routine_slots(plan, model=rig.holder["model"], day_type="workday", config=CONFIG)
    assert first == again


def test_a_meal_happens_with_the_chance_that_she_opened_a_conversation_then(
    services: Services, clock: ManualClock
) -> None:
    """Her history says how likely a meal message is: none where she never opened one."""
    quiet = opening_curve(base=0.0, peaks={})
    rig, plan = make(services, clock, curve=quiet)
    keys = set(slots_of(rig, plan))
    assert keys == {"greeting"}  # nothing in her history at any mealtime or at goodnight
    # and about the chance of the window across many days
    counts = {"meal:lunch": 0}
    days = 80
    curve = opening_curve(base=0.0, peaks={48: 0.2, 49: 0.2})  # 0.4 openings around noon
    rig2 = Rig.build(services, clock, proactive_model(curve), quota=QuotaRange(6, 6), salt=SALT)
    for number in range(days):
        day = date(2026, 3, 2) + timedelta(days=number)
        if day.weekday() >= 5:
            continue
        plan2 = rig2.planner.ensure(day)
        found = routine_slots(plan2, model=rig2.holder["model"], day_type="workday", config=CONFIG)
        counts["meal:lunch"] += any(slot.key == "meal:lunch" for slot in found)
    workdays = sum(1 for n in range(days) if (date(2026, 3, 2) + timedelta(days=n)).weekday() < 5)
    share = counts["meal:lunch"] / workdays
    assert 0.2 < share < 0.6  # the window holds 0.4 openings a day


def test_without_any_history_a_meal_or_goodnight_has_a_fixed_chance(
    services: Services, clock: ManualClock
) -> None:
    _, plan = make(services, clock)
    found = routine_slots(plan, model=None, day_type="workday", config=CONFIG)
    assert {slot.key for slot in found} <= {
        "greeting",
        "meal:breakfast",
        "meal:lunch",
        "meal:dinner",
        "bedtime",
    }
    assert "greeting" in {slot.key for slot in found}


def test_the_slots_count_against_the_days_quota_and_the_greeting_stays(
    services: Services, clock: ManualClock
) -> None:
    rig, plan = make(
        services,
        clock,
        curve=opening_curve(base=2.0, peaks={}),
        quota=QuotaRange(1, 1),
    )
    assert plan.quota.for_plan == 1
    keys = [
        s.key
        for s in routine_slots(plan, model=rig.holder["model"], day_type="workday", config=CONFIG)
    ]
    assert keys == ["greeting"]
    rig3, plan3 = make(
        services,
        clock,
        curve=opening_curve(base=2.0, peaks={}),
        quota=QuotaRange(3, 3),
        day=date(2026, 10, 12),
    )
    kept = routine_slots(plan3, model=rig3.holder["model"], day_type="workday", config=CONFIG)
    assert len(kept) == 3 and kept[0].key == "greeting"


def test_a_day_with_proactive_messages_off_has_no_slots(
    services: Services, clock: ManualClock
) -> None:
    rig, plan = make(
        services,
        clock,
        curve=opening_curve(base=2.0, peaks={}),
        quota=QuotaRange(1, 6, enabled=False),
    )
    assert routine_slots(plan, model=rig.holder["model"], day_type="workday", config=CONFIG) == []


def test_no_greeting_is_made_when_the_plan_does_not_allow_one(
    services: Services, clock: ManualClock
) -> None:
    rig, plan = make(services, clock, curve=opening_curve(base=2.0, peaks={}))
    rig.planner.record_wake_greeting(plan.wake - timedelta(hours=1))  # type: ignore[operator]
    clock.set_time(plan.effective_from)
    rebuilt = rig.planner.refresh("test", force=True).plan
    assert not rebuilt.greeting.allowed
    keys = {
        s.key
        for s in routine_slots(
            rebuilt, model=rig.holder["model"], day_type="workday", config=CONFIG
        )
    }
    assert "greeting" not in keys


def follow(due: datetime, window: int = 240, number: str = "f1") -> FollowupRecord:
    return FollowupRecord(
        id=number,
        text="周五考试",
        due_at=due,
        window_minutes=window,
        source_turn_id=None,
        status="open",
        created_at=due - timedelta(days=1),
        closed_at=None,
        close_reason=None,
        origin="bot_session",
        fact_id=None,
    )


def test_a_followup_goes_out_a_little_after_the_event_inside_its_window(
    services: Services, clock: ManualClock
) -> None:
    _, plan = make(services, clock)
    due = plan.effective_from + timedelta(hours=14)
    record = follow(due)
    slot = followup_slot(record, plan=plan, now=due)
    assert slot.kind is TriggerKind.FOLLOWUP and slot.key == "followup:f1"
    assert due + timedelta(minutes=24) <= slot.planned_at <= due + timedelta(minutes=96)
    assert slot.window_end == record.window_end
    assert slot.detail["followup_id"] == "f1"
    assert followup_slot(record, plan=plan, now=due) == slot  # drawn from its id: the same
    other = followup_slot(follow(due, number="f2"), plan=plan, now=due)
    assert other.planned_at != slot.planned_at


def test_a_followup_that_came_due_long_ago_is_asked_now(
    services: Services, clock: ManualClock
) -> None:
    _, plan = make(services, clock)
    due = plan.effective_from + timedelta(hours=10)
    now = due + timedelta(hours=3)
    slot = followup_slot(follow(due), plan=plan, now=now)
    assert slot.planned_at == now
    assert slot.window_end >= slot.planned_at


@pytest.mark.parametrize("window", [30, 240, 1440])
def test_the_followup_is_planned_inside_windows_of_every_length(
    services: Services, clock: ManualClock, window: int
) -> None:
    _, plan = make(services, clock)
    due = plan.effective_from + timedelta(hours=9)
    slot = followup_slot(follow(due, window), plan=plan, now=due)
    assert due <= slot.planned_at <= slot.window_end
