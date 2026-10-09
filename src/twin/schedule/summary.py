"""When the daily summaries are queued (R-MEM-002).

The summary of the day before is written by an offline job (:mod:`twin.memory.jobs`).  This module
decides *when it is queued*: 60 minutes before she wakes, but not while DeepSeek is at peak
prices - then at the start of the nearest off-peak window - and never later than two hours after
she woke (the job's own deadline then lets it run at any price).  With the off-peak discount
switched off (``pricing.offpeak_multiplier`` = 1) there is nothing to wait for.

Which summaries: the ``real`` scope when the real records have lines on that day (the day cut in
the zone she was in when the records were made), the ``bot`` scope once the bot's conversation can
be read (round 09).  ``queue_daily_summary`` does not queue a scope twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from twin.config.settings import ScheduleConfig
from twin.llm.pricing import PeakCalendar
from twin.memory.dayload import bot_day_lines, real_day_lines
from twin.memory.jobs import queue_daily_summary
from twin.memory.localdate import BOT, REAL, MemoryClock
from twin.memory.recent import bot_turn_reader
from twin.schedule.plan_model import DailyPlan
from twin.services import Services


@dataclass(frozen=True)
class SummaryTimes:
    """When to queue the summary of the day before, and the time it has to run by."""

    lead: datetime  # the ideal moment: before she wakes
    trigger_at: datetime  # when it is queued: ``lead`` or the start of the off-peak window
    deadline: datetime  # the job runs by now at any price


def summary_times(
    wake: datetime, *, calendar: PeakCalendar, offpeak_discount: bool, config: ScheduleConfig
) -> SummaryTimes:
    """The moments for a wake-up at ``wake`` (see the module description)."""
    lead = wake - timedelta(minutes=config.summary_lead_min)
    deadline = wake + timedelta(hours=config.summary_latest_after_wake_h)
    trigger = lead
    if offpeak_discount and calendar.is_peak(lead):
        trigger = calendar.next_offpeak_window(lead).start
    return SummaryTimes(lead, min(trigger, deadline), deadline)


def summary_scopes(services: Services, day: date) -> tuple[str, ...]:
    """The scopes that have something to summarise on ``day``."""
    scopes: list[str] = []
    clock = MemoryClock.from_services(services)
    if real_day_lines(services, clock, day):
        scopes.append(REAL)
    reader = bot_turn_reader(services)
    if reader is not None and bot_day_lines(reader, clock, day):
        scopes.append(BOT)
    return tuple(scopes)


def queue_previous_day_summaries(
    services: Services, plan: DailyPlan, times: SummaryTimes
) -> list[str]:
    """Queue the summaries of the day before ``plan``'s day; returns the new job ids."""
    day = plan.local_date - timedelta(days=1)
    scopes = summary_scopes(services, day)
    if not scopes:
        return []
    return queue_daily_summary(services, day, scopes, deadline=times.deadline)
