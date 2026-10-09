"""When the daily summaries are queued: before she wakes, never at peak prices, by two hours after.

R-MEM-002 (the trigger time) with R-LLM-007 (the peak hours of the price table).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta

import pytest

from tests.support.bot_turns import conversation
from tests.support.clock import ManualClock
from tests.support.routine import Rig, fixed_model
from tests.support.synth_chat import MessageWriter
from twin.config.settings import ScheduleConfig
from twin.llm.pricing import PeakCalendar
from twin.memory.recent import register_bot_turn_reader
from twin.schedule.summary import queue_previous_day_summaries, summary_scopes, summary_times
from twin.services import Services

CALENDAR = PeakCalendar()


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


def times(wake: datetime, *, discount: bool = True, config: ScheduleConfig | None = None):  # type: ignore[no-untyped-def]
    return summary_times(
        wake, calendar=CALENDAR, offpeak_discount=discount, config=config or ScheduleConfig()
    )


def test_the_summary_is_queued_an_hour_before_she_wakes_when_that_is_off_peak() -> None:
    wake = utc(2026, 10, 9, 12, 30)  # 07:30 in Chicago, 20:30 in Beijing: off peak
    found = times(wake)
    assert found.lead == utc(2026, 10, 9, 11, 30) and found.trigger_at == found.lead
    assert found.deadline == utc(2026, 10, 9, 14, 30)  # two hours after she woke


def test_at_peak_prices_it_waits_for_the_nearest_off_peak_window() -> None:
    """06:00-10:00 UTC on a Beijing working day is peak; the window opens at 10:00."""
    found = times(utc(2026, 10, 9, 10, 30))  # the hour before is 09:30 UTC: peak
    assert found.lead == utc(2026, 10, 9, 9, 30)
    assert found.trigger_at == utc(2026, 10, 9, 10, 0) and found.trigger_at <= found.deadline
    early = times(utc(2026, 10, 9, 3, 0))  # lead 02:00 UTC: the morning peak of 01:00-04:00
    assert early.trigger_at == utc(2026, 10, 9, 4, 0)


def test_it_is_never_later_than_two_hours_after_she_woke() -> None:
    wake = utc(2026, 10, 9, 7, 0)  # lead 06:00 UTC is peak until 10:00 - too late
    found = times(wake)
    assert found.trigger_at == found.deadline == utc(2026, 10, 9, 9, 0)
    # a shorter limit and an earlier lead are configurable
    config = ScheduleConfig(summary_lead_min=30, summary_latest_after_wake_h=1)
    other = times(utc(2026, 10, 9, 12, 30), config=config)
    assert other.lead == utc(2026, 10, 9, 12, 0) and other.deadline == utc(2026, 10, 9, 13, 30)


def test_a_beijing_day_off_has_no_peak_hours() -> None:
    saturday = utc(2026, 10, 17, 7, 0)  # a plain Saturday: 06:00-10:00 UTC is not peak
    found = times(saturday)
    assert found.trigger_at == found.lead == utc(2026, 10, 17, 6, 0)
    national_day = times(utc(2026, 10, 2, 7, 0))  # a public holiday
    assert national_day.trigger_at == national_day.lead
    make_up = times(utc(2026, 10, 10, 7, 0))  # a Saturday that is a working day: peak
    assert make_up.lead == utc(2026, 10, 10, 6, 0) and make_up.trigger_at == make_up.deadline


def test_without_the_off_peak_discount_there_is_nothing_to_wait_for() -> None:
    found = times(utc(2026, 10, 9, 10, 30), discount=False)
    assert found.trigger_at == found.lead == utc(2026, 10, 9, 9, 30)


# ------------------------------------------------------------------- what is queued


@pytest.fixture
def rig(services: Services, clock: ManualClock) -> Iterator[Rig]:
    clock.set_time(utc(2026, 10, 9, 12, 0))
    yield Rig.build(services, clock, fixed_model())
    register_bot_turn_reader(None)


def test_the_scopes_are_the_real_records_of_the_day_and_the_bots_conversation(
    rig: Rig, services: Services
) -> None:
    day = date(2026, 10, 8)
    assert summary_scopes(services, day) == ()  # nothing imported, no conversation yet
    writer = MessageWriter(services)
    writer.add(utc(2026, 10, 8, 20, 0), True, "text", "晚上好")  # Thursday afternoon in Chicago
    writer.store()
    assert summary_scopes(services, day) == ("real",)
    assert summary_scopes(services, date(2026, 10, 7)) == ()  # another day has no records
    register_bot_turn_reader(lambda _services: conversation(2, start=utc(2026, 10, 8, 20, 30)))
    assert summary_scopes(services, day) == ("real", "bot")
    assert summary_scopes(services, date(2026, 10, 7)) == ()  # no conversation that day either


def test_both_summaries_are_queued_for_the_day_before_and_not_twice(
    rig: Rig, services: Services
) -> None:
    writer = MessageWriter(services)
    writer.add(utc(2026, 10, 8, 20, 0), True, "text", "晚上好")
    writer.store()
    register_bot_turn_reader(lambda _services: conversation(2, start=utc(2026, 10, 8, 20, 30)))
    plan = rig.planner.ensure(date(2026, 10, 9))
    found = times(plan.wake or utc(2026, 10, 9, 12, 30))
    first = queue_previous_day_summaries(services, plan, found)
    assert len(first) == 2
    assert queue_previous_day_summaries(services, plan, found) == []  # already waiting
    from twin.ops.jobs import JobQueue

    queued = JobQueue(services.db, services.clock).list_jobs(job_type="memory_summary")
    assert {(j.payload["date"], j.payload["scope"]) for j in queued} == {
        ("2026-10-08", "real"),
        ("2026-10-08", "bot"),
    }
    assert all(j.offpeak_only and j.deadline == found.deadline for j in queued)
    assert found.deadline - found.lead == timedelta(hours=3)
