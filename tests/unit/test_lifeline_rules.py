"""The rules a drawn day must obey, and the correction by rule (R-MEM-005).

Times do not overlap, nothing is planned while she sleeps, a stretch in a busy period is marked
busy (and the other way round), a plan made mid-day has no morning.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from tests.support.lifeline import event, good_workday
from tests.support.routine import fixed_model
from twin.config.settings import ScheduleConfig
from twin.memory.lifeline_rules import (
    DayFrame,
    Span,
    check_rules,
    clock_text,
    repair,
)
from twin.memory.schemas import DrawnEvent
from twin.schedule.plan_builder import PlanRequest, QuotaRange, build_plan, plan_seed
from twin.schedule.wallclock import day_bounds_utc, to_utc

CHICAGO = ZoneInfo("America/Chicago")
FRIDAY = date(2026, 10, 9)


def frame(*, from_minute: int | None = None, day: date = FRIDAY) -> DayFrame:
    start, _ = day_bounds_utc(day, CHICAGO)
    effective = start if from_minute is None else to_utc(day, from_minute, CHICAGO)
    model = fixed_model()
    plan = build_plan(
        PlanRequest(
            day=day,
            zone=CHICAGO,
            day_type="workday",
            next_day_type="workday",
            seed=plan_seed(day, "s"),
            model=model,
            effective_from=effective,
            go_to_bed_at=None,
            quota=QuotaRange(1, 6),
            last_greeting_at=None,
            previous_day=None,
            same_day=None,
            reason="daily",
            config=ScheduleConfig(),
            inputs_hash="x",
            routine_hash="r",
            created_at=datetime(2026, 10, 9, 5, tzinfo=UTC),
        )
    )
    return DayFrame.from_plan(plan, CHICAGO)


def drawn(items: list[dict]) -> list[DrawnEvent]:  # type: ignore[type-arg]
    return [DrawnEvent.model_validate(item) for item in items]


def codes(
    events: list[DrawnEvent], frame_: DayFrame, **kwargs: int
) -> list[tuple[str, int | None]]:
    return [(p.code, p.event) for p in check_rules(events, frame_, **kwargs)]


# ------------------------------------------------------------------- the frame


def test_the_frame_is_the_plan_in_the_clock_times_of_the_day() -> None:
    found = frame()
    assert found.wake == 450 and found.bed == 23 * 60 + 30  # 07:30 and 23:30
    assert found.sleep == (Span(0, 450), Span(1410, 1440))
    assert [(s.start, s.end, s.label) for s in found.busy] == [(780, 1020, "13:00–17:00")]
    assert dict(found.meals).keys() == {"breakfast", "lunch", "dinner"}
    assert found.not_before == 0.0 and found.zone_key == "America/Chicago"
    assert found.busy_share(780, 1020) == 1.0 and found.busy_share(600, 660) == 0.0
    assert round(found.busy_share(720, 840), 3) == 0.5 and found.busy_share(5, 5) == 0.0
    assert found.asleep_minutes(0, 600) == 450


def test_a_plan_made_in_the_middle_of_the_day_has_no_morning() -> None:
    found = frame(from_minute=15 * 60)
    assert found.not_before == 900.0 and found.wake is None  # her wake-up was in the old plan
    assert found.bed == 23 * 60 + 30


# ------------------------------------------------------------------ the checks


def test_a_consistent_day_has_nothing_wrong() -> None:
    assert codes(drawn(good_workday()), frame()) == []


def test_unreadable_or_backwards_times_are_found() -> None:
    items = good_workday()
    items[1] = event("八点", "12:00", "在图书馆看文献")
    items[2] = event("13:00", "12:50", "吃午饭")
    items[3] = event("13:00", "24:00", "上课", busy=True)
    assert codes(drawn(items), frame()) == [("bad_time", 2), ("bad_time", 3), ("bad_time", 4)]


def test_overlapping_stretches_are_found() -> None:
    items = good_workday()
    items[1] = event("08:30", "12:30", "在图书馆看文献")  # runs into lunch at 12:10
    problems = codes(drawn(items), frame())
    assert problems == [("overlap", 3)]
    assert "第 3 段与第 2 段" in check_rules(drawn(items), frame())[0].text


def test_a_stretch_in_the_sleep_is_found_unless_it_is_sleep_itself() -> None:
    items = good_workday()
    items.insert(0, event("06:00", "07:30", "写论文"))  # she sleeps until 07:30
    assert ("during_sleep", 1) in codes(drawn(items), frame())
    sleeping = good_workday()
    sleeping.insert(0, event("06:00", "07:30", "赖床"))  # lying in bed is sleep-ish
    assert codes(drawn(sleeping), frame()) == []
    brushing = good_workday()
    brushing[-1] = event("23:20", "23:35", "看书")  # five minutes over the bedtime: tolerated
    assert codes(drawn(brushing), frame()) == []
    late = good_workday()
    late[-1] = event("23:00", "23:59", "刷手机")  # 29 minutes into the night
    assert ("during_sleep", 7) in codes(drawn(late), frame())


def test_busy_periods_and_the_busy_mark_agree() -> None:
    items = good_workday()
    items[3] = event("13:00", "17:00", "在咖啡馆发呆", busy=False)  # in the busy period
    items[1] = event("08:30", "12:00", "在图书馆看文献", busy=True)  # busy but not in one
    assert codes(drawn(items), frame()) == [("busy_mismatch", 2), ("busy_mismatch", 4)]
    half = good_workday()
    half[2] = event("12:20", "13:20", "吃午饭", busy=False)  # a third of it in the period: free
    half[3] = event("13:20", "17:00", "上课", busy=True)
    assert codes(drawn(half), frame()) == []
    half[2] = event("12:30", "13:30", "吃午饭", busy=False)  # half of it counts as busy
    half[3] = event("13:30", "17:00", "上课", busy=True)
    assert codes(drawn(half), frame()) == [("busy_mismatch", 3)]
    half[2] = event("12:30", "13:30", "吃午饭", busy=True)
    assert codes(drawn(half), frame()) == []


def test_a_day_before_the_start_of_the_plan_and_a_day_with_too_few_stretches() -> None:
    found = frame(from_minute=15 * 60)
    items = [event("14:00", "16:00", "上课", busy=True), *good_workday()[4:]]
    assert codes(drawn(items), found) == [("before_plan", 1)]
    assert codes(drawn(good_workday()[:2]), frame()) == [("too_few", None)]
    assert codes(drawn(good_workday()[:2]), frame(), minimum=2) == []


# ------------------------------------------------------------------- the repair


def test_a_good_day_is_left_as_it_is() -> None:
    events = drawn(good_workday())
    assert repair(events, frame()) == events


def test_the_sleeping_part_is_cut_off_and_what_cannot_be_saved_is_dropped() -> None:
    items = [
        event("06:00", "08:00", "写论文"),  # 07:30-08:00 is left
        event("08:00", "08:05", "喝水"),  # too short once it is moved
        event("07:50", "09:00", "吃早饭"),  # starts inside the previous one: moved to 08:00
        event("03:00", "05:00", "写代码"),  # entirely asleep
        event("12:00", "12:00", "吃饭"),  # no length
        event("20:00", "21:00", "看书", busy=True),  # marked busy outside a busy period
        event("14:00", "15:00", "开会", busy=False),  # in a busy period but not marked
    ]
    fixed = repair(drawn(items), frame(), frozenset())
    assert [(e.start, e.end, e.activity, e.busy) for e in fixed] == [
        ("07:30", "08:00", "写论文", False),
        ("08:00", "09:00", "吃早饭", False),
        ("14:00", "15:00", "开会", True),
        ("20:00", "21:00", "看书", False),
    ]
    assert check_rules(fixed, frame(), minimum=0) == []


def test_the_stretches_the_checker_found_contradictory_are_dropped() -> None:
    events = drawn(good_workday())
    fixed = repair(events, frame(), frozenset({2, 4}))
    assert [e.activity for e in fixed] == [
        "吃早饭",
        "吃午饭",
        "吃晚饭",
        "写作业，顺便追剧",
        "洗漱，准备睡觉",
    ]
    assert check_rules(fixed, frame(), minimum=0) == []


def test_a_stretch_that_starts_before_the_plan_is_cut_to_its_start() -> None:
    found = frame(from_minute=15 * 60)
    fixed = repair(drawn([event("14:00", "16:00", "上课", busy=True)]), found)
    assert [(e.start, e.end) for e in fixed] == [("15:00", "16:00")]


def test_a_repaired_day_always_passes_the_rules() -> None:
    messy = [
        event("00:00", "23:59", "在家"),
        event("09:00", "10:00", "跑步"),
        event("09:30", "11:00", "买菜", busy=True),
        event("xx", "yy", "神秘"),
        event("22:00", "23:50", "追剧"),
    ]
    fixed = repair(drawn(messy), frame())
    assert check_rules(fixed, frame(), minimum=0) == []
    assert all(e.start < e.end for e in fixed)


def test_clock_text_never_says_24_00() -> None:
    assert clock_text(0) == "00:00" and clock_text(1440) == "23:59" and clock_text(-5) == "00:00"
    assert clock_text(59.6) == "01:00" and clock_text(450) == "07:30"


def test_the_frame_of_a_day_with_a_shifted_clock_uses_the_clock_of_the_day() -> None:
    """On the day the clocks go back her 23:30 bedtime is 23:30 on the wall, not 22:30."""
    day = date(2026, 11, 1)
    start, end = day_bounds_utc(day, CHICAGO)
    assert end - start == timedelta(hours=25)
    found = frame(day=day)
    assert found.bed == 23 * 60 + 30 and found.wake == 450
