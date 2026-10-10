"""The wiring: one time service, one calendar, one planner per container (R-SCH-001, R-SCOPE-005).

And the day types as the planner sees them: holidays of the country of the zone, the make-up
working days of China, a year the holiday library does not know, holidays set by hand.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from tests.support.clock import ManualClock
from tests.support.routine import Rig, fixed_model, student_model
from twin.config.runtime import BOT_TIMEZONE
from twin.llm.runtime import build_llm_runtime
from twin.memory.memory import Memory
from twin.profile.overrides import RoutineOverrides
from twin.schedule.service import (
    peak_calendar_for,
    schedule_kit,
    time_service_for,
)
from twin.services import Services

SHANGHAI = "Asia/Shanghai"
SRC = Path(__file__).resolve().parents[2] / "src" / "twin"


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


@pytest.fixture
def rig(services: Services, clock: ManualClock) -> Rig:
    clock.set_time(utc(2026, 10, 9, 12, 0))
    return Rig.build(services, clock, student_model())


# ----------------------------------------------------------------------- wiring


def test_everything_asks_the_same_time_service(services: Services) -> None:
    kit = schedule_kit(services)
    assert schedule_kit(services) is kit and time_service_for(services) is kit.time
    runtime = build_llm_runtime(services)
    assert runtime.time_service is kit.time  # the budget and the ledger
    assert Memory(services).clock.bot_zone() == kit.time.bot_timezone()  # the memory's local days


def test_a_zone_change_reaches_the_budget_the_memory_and_the_plans_alike(
    services: Services, clock: ManualClock
) -> None:
    clock.set_time(
        utc(2026, 10, 9, 20, 0)
    )  # 15:00 on the 9th in Chicago, 04:00 on the 10th in Shanghai
    kit = schedule_kit(services)
    runtime = build_llm_runtime(services)
    memory = Memory(services)
    assert runtime.time_service.local_date() == date(2026, 10, 9)
    assert memory.clock.bot_date(clock.now_utc()) == date(2026, 10, 9)
    services.runtime.set(BOT_TIMEZONE, SHANGHAI, by="test")
    assert kit.time.local_date() == date(2026, 10, 10)  # the zone is read on every call
    assert runtime.time_service.local_date() == date(2026, 10, 10)
    assert memory.clock.bot_date(clock.now_utc()) == date(2026, 10, 10)
    assert memory.clock.bot_bounds(date(2026, 10, 10))[0] == utc(2026, 10, 9, 16, 0)
    assert runtime.budget.compute().day == date(2026, 10, 10)


def test_the_price_table_and_the_day_types_use_one_holiday_calendar(services: Services) -> None:
    runtime = build_llm_runtime(services)
    calendar = peak_calendar_for(services)
    assert runtime.pricing.calendar is calendar and peak_calendar_for(services) is calendar
    kit = schedule_kit(services)
    assert kit.calendar()._china is calendar


def test_a_year_the_holiday_library_does_not_know_raises_one_alert_for_both_users(
    services: Services, clock: ManualClock
) -> None:
    from sqlalchemy import select

    from twin.storage.models import Alert

    services.runtime.set(BOT_TIMEZONE, SHANGHAI, by="test")
    kit = schedule_kit(services)
    kit.drop_caches()
    runtime = build_llm_runtime(services)
    assert kit.time.day_type(date(2099, 6, 1)) == "workday"  # a Monday: the fallback
    assert runtime.pricing.is_peak(utc(2099, 6, 1, 7, 0))  # the same fallback for the price
    with services.db.session() as session:
        categories = [a.category for a in session.scalars(select(Alert))]
    assert categories == ["calendar_out_of_range"]  # one warning for the year, whoever asked first


def test_the_bot_time_zone_setting_is_read_in_one_place_only() -> None:
    """Nobody converts "now" to a local time on their own: the zone comes from the time service."""
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if re.search(r"\bBOT_TIMEZONE\b", text):
            offenders.append(path.relative_to(SRC).as_posix())
    assert offenders == ["config/runtime.py", "schedule/service.py"]


# ----------------------------------------------------------------- day types in plans


def test_the_plan_of_a_us_holiday_uses_the_holiday_routine(rig: Rig) -> None:
    columbus = rig.planner.ensure(date(2026, 10, 12))  # Columbus Day, a Monday
    assert columbus.day_type == "holiday"
    assert columbus.morning is not None
    assert rig.zone().key == "America/Chicago"
    assert columbus.busy == ()  # the busy hours are a workday thing
    ordinary = rig.planner.ensure(date(2026, 10, 13))
    assert ordinary.day_type == "workday" and len(ordinary.busy) == 1
    saturday = rig.planner.ensure(date(2026, 10, 10))
    assert saturday.day_type == "weekend" and saturday.busy == ()


def test_the_weekend_that_is_a_working_day_in_china_gets_the_busy_hours(
    rig: Rig, services: Services, clock: ManualClock
) -> None:
    services.runtime.set(BOT_TIMEZONE, SHANGHAI, by="test")
    rig.kit.drop_caches()
    make_up = rig.planner.ensure(date(2026, 10, 10))  # a Saturday, the make-up day of National Day
    assert make_up.day_type == "workday" and len(make_up.busy) == 1
    sunday = rig.planner.ensure(date(2026, 10, 11))
    assert sunday.day_type == "weekend" and sunday.busy == ()
    national_day = rig.planner.ensure(date(2026, 10, 2))
    assert national_day.day_type == "holiday"


def test_a_year_the_library_does_not_know_still_gets_a_plan(
    rig: Rig, services: Services, clock: ManualClock
) -> None:
    services.runtime.set(BOT_TIMEZONE, SHANGHAI, by="test")
    rig.kit.drop_caches()
    monday = rig.planner.ensure(date(2099, 6, 1))
    saturday = rig.planner.ensure(date(2099, 6, 6))
    assert (monday.day_type, saturday.day_type) == ("workday", "weekend")
    assert len(monday.busy) == 1 and monday.morning is not None


def test_a_holiday_set_by_hand_changes_the_plan_after_the_calendar_is_dropped(
    rig: Rig, services: Services, clock: ManualClock
) -> None:
    wednesday = date(2026, 10, 14)
    assert rig.kit.time.day_type(wednesday) == "workday"
    RoutineOverrides(services.db, clock).add_holiday(wednesday, wednesday + timedelta(days=1))
    assert rig.kit.time.day_type(wednesday) == "workday"  # the calendar is read once ...
    rig.kit.drop_caches()  # ... until the state watcher drops it
    assert rig.kit.time.day_type(wednesday) == "holiday"
    assert rig.kit.time.day_type(wednesday + timedelta(days=2)) == "workday"
    plan = rig.planner.ensure(wednesday)
    assert plan.day_type == "holiday" and plan.busy == ()


def test_the_plan_follows_the_day_type_of_the_zone_across_a_switch(
    services: Services, clock: ManualClock
) -> None:
    clock.set_time(utc(2026, 10, 12, 14, 0))  # Monday 09:00 in Chicago: Columbus Day
    rig = Rig.build(services, clock, fixed_model())
    before = rig.planner.refresh("startup").plan
    assert before.day_type == "holiday"
    clock.set_time(utc(2026, 10, 12, 20, 0))  # 04:00 on Tuesday the 13th in Shanghai
    outcome = rig.planner.switch_timezone(SHANGHAI)
    assert outcome.plan is not None
    assert outcome.plan.local_date == date(2026, 10, 13)
    assert outcome.plan.day_type == "workday"  # there is no Columbus Day in China
