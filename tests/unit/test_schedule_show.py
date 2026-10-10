"""A day plan in words, in the clock times of the place she lives (``twin plan show``, R-SCH-004).

Everything is stored as instants; ``render_plan`` turns them into the wall clock of the plan's
zone - on an ordinary day, on the two days of the clock change (R-SCH-003) and after the bot moved
to another zone (R-SCH-002).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from tests.support.routine import fixed_model
from tests.unit.test_schedule_plan_builder import request
from twin.schedule.plan_builder import build_plan
from twin.schedule.show import SOURCES, STATES, local_text, render_plan

CHICAGO = ZoneInfo("America/Chicago")
SHANGHAI = ZoneInfo("Asia/Shanghai")


def test_a_moment_is_a_clock_time_and_gets_its_date_when_it_is_not_the_day() -> None:
    moment = datetime(2026, 11, 1, 7, 30, tzinfo=UTC)  # 01:30 CST: the second 01:30 of the day
    assert local_text(moment, CHICAGO, date(2026, 11, 1)) == "01:30"
    assert local_text(moment, CHICAGO, date(2026, 10, 31)) == "11-01 01:30"
    assert local_text(moment, SHANGHAI, date(2026, 11, 1)) == "15:30"
    assert local_text(moment, SHANGHAI, date(2026, 11, 2)) == "11-01 15:30"


def test_an_ordinary_day_is_shown_in_the_clock_times_of_its_zone() -> None:
    plan = build_plan(request(date(2026, 10, 9), fixed_model()))
    lines = render_plan(plan)
    assert lines[0] == "2026-10-09（周五，工作日，America/Chicago）"
    assert "睡眠·早晨：10-08 23:30 → 07:30（8小时00分；" in "\n".join(lines)
    assert "睡眠·夜晚：23:30 → 10-10 07:30（8小时00分；" in "\n".join(lines)
    assert any(line.startswith("  13:00 – 17:00  忙碌") for line in lines)  # her workday
    assert any(
        line.startswith("- 饭点：午饭 11:") or line.startswith("- 饭点：午饭 12:") for line in lines
    )
    assert "- 状态时间线：" in lines and lines[-1].startswith("  ")
    assert not any(line.startswith("!!!") for line in lines)
    assert all(state in STATES.values() for state in ("深睡", "空闲")) and SOURCES


def test_the_day_the_clocks_go_back_shows_the_hours_as_they_are_on_the_wall() -> None:
    day = date(2026, 11, 1)  # 25 hours: 01:00 to 02:00 happens twice
    plan = build_plan(request(day, fixed_model(), next_day_type="workday"))
    lines = render_plan(plan)
    assert lines[0].startswith("2026-11-01（周日，周末，America/Chicago）")
    first = plan.segments[0]
    assert first.start == datetime(2026, 11, 1, 5, 0, tzinfo=UTC)  # midnight, still summer time
    assert plan.ends_at > first.start
    covered = lines[1]
    assert "生效 00:00 至 " in covered  # from midnight on the wall clock
    timeline = lines[lines.index("- 状态时间线：") + 1 :]
    assert timeline[0].startswith("  00:00 – ") and "深睡" in timeline[0]
    assert len(timeline) == len(plan.segments)


def test_the_day_the_clocks_go_forward_says_what_was_read_differently() -> None:
    day = date(2026, 3, 8)  # 23 hours: 02:00 to 03:00 does not exist
    plan = build_plan(request(day, fixed_model(), next_day_type="workday"))
    lines = render_plan(plan)
    assert lines[0].startswith("2026-03-08（周日，周末，America/Chicago）")
    notes = [line for line in lines if line.startswith("- 夏令时：")]
    assert len(notes) == len(plan.extra["dst"])
    for line in notes:
        assert "实际读作" in line and ("春季跳过的时刻" in line or "秋季重复的时刻" in line)


def test_a_plan_made_after_a_zone_switch_is_shown_on_the_new_wall_clock() -> None:
    day = date(2026, 10, 9)
    chicago = render_plan(build_plan(request(day, fixed_model(), zone=CHICAGO)))
    shanghai = render_plan(build_plan(request(day, fixed_model(), zone=SHANGHAI)))
    assert chicago[0].endswith("America/Chicago）") and shanghai[0].endswith("Asia/Shanghai）")
    for lines in (chicago, shanghai):  # her routine moves over unchanged: busy from 13 to 17
        assert any(line.startswith("  13:00 – 17:00  忙碌") for line in lines)
    first_chicago = build_plan(request(day, fixed_model(), zone=CHICAGO)).segments[0].start
    first_shanghai = build_plan(request(day, fixed_model(), zone=SHANGHAI)).segments[0].start
    assert (first_chicago - first_shanghai).total_seconds() == 13 * 3600  # 5 vs -8 hours from UTC


def test_a_preview_says_it_is_not_stored_and_a_replaced_plan_says_how_often() -> None:
    plan = build_plan(request(date(2026, 10, 9), fixed_model()))
    preview = render_plan(plan, stored=False)
    assert preview[0].endswith("[预览：尚未保存，生成第一份计划后种子才固定]")
    assert "此前替换过 2 份" in render_plan(plan, replaced=2)[1]
    assert "此前替换过" not in render_plan(plan)[1]
