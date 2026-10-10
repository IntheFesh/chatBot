"""Wiring of the schedule: one time service, one planner and one event bus per container.

:func:`schedule_kit` builds, once per :class:`~twin.services.Services`, the pieces that must be
the same object everywhere:

* the zone-name provider, which reads the runtime setting on every call (one primary-key lookup),
  so a ``/时区`` or ``twin timezone set`` - in this process or in another - is seen at once, and
  the day-type calendar, a :class:`~twin.schedule.time_service.Cached` value (it reads the holiday
  ranges set by hand) that the schedule component drops when the state watcher reports a change,
  together with the plans read from the database, within two seconds;
* the :class:`~twin.schedule.time_service.BotTimeService` the budget, the memory and the engine ask;
* the :class:`~twin.schedule.planner.DailyPlanner` behind ``her_state``;
* the :class:`~twin.schedule.events.ScheduleEvents` bus rounds 09 and 10 subscribe to.

The Chinese holiday calendar is the one of the price table's peak hours: :func:`peak_calendar_for`
is the single :class:`~twin.llm.pricing.PeakCalendar` of a container, shared by
:class:`~twin.llm.pricing.Pricing` and by the day types, with the same fallback for years the
library does not know and the same alert.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from twin.config.runtime import (
    BOT_TIMEZONE,
    PROACTIVE_DAILY_MAX,
    PROACTIVE_DAILY_MIN,
    PROACTIVE_ENABLED,
)
from twin.llm.pricing import PeakCalendar
from twin.profile.activity_model import ActivityModel
from twin.profile.api import load_activity_model
from twin.profile.overrides import RoutineOverrides
from twin.schedule.daytype import DayTypeCalendar
from twin.schedule.events import ScheduleEvents
from twin.schedule.plan_builder import QuotaRange
from twin.schedule.planner import DailyPlanner
from twin.schedule.time_service import BotTimeService, Cached

if TYPE_CHECKING:
    from twin.services import Services

KIT_KEY = "schedule_kit"
PEAK_KEY = "peak_calendar"


@dataclass
class ScheduleKit:
    """The shared schedule objects of one container."""

    calendar: Cached[DayTypeCalendar]
    time: BotTimeService
    planner: DailyPlanner
    events: ScheduleEvents

    def drop_caches(self) -> None:
        """Forget what was read from the database (the state watcher calls this on a change)."""
        self.planner.invalidate()


def peak_calendar_for(services: Services) -> PeakCalendar:
    """The one holiday calendar of a container (peak hours of the price table, day types)."""
    found = services.extras.get(PEAK_KEY)
    if isinstance(found, PeakCalendar):
        return found

    def fallback(year: int) -> None:
        services.alerts.raise_alert(
            "calendar_out_of_range",
            f"the holiday calendar does not cover {year}; peak hours assume Monday to Friday",
            severity="warning",
            detail={"year": year},
            dedup_key=f"calendar_out_of_range:{year}",
        )

    pricing = services.settings.pricing
    calendar = PeakCalendar(
        extra_offpeak_dates=pricing.extra_offpeak_dates,
        extra_peak_dates=pricing.extra_peak_dates,
        on_fallback=fallback,
    )
    services.extras[PEAK_KEY] = calendar
    return calendar


def build_kit(
    services: Services,
    *,
    model_source: Callable[[], ActivityModel | None] | None = None,
    quota_source: Callable[[], QuotaRange] | None = None,
) -> ScheduleKit:
    """Create the schedule objects of ``services`` (see :func:`schedule_kit` for the shared set)."""
    runtime = services.runtime
    settings = services.settings

    def day_calendar() -> DayTypeCalendar:
        return DayTypeCalendar(
            zone_countries=settings.safety.timezone_country,
            holiday_ranges=RoutineOverrides(services.db, services.clock).holiday_ranges(),
            peak_calendar=peak_calendar_for(services),
        )

    def proactive_range() -> QuotaRange:
        return QuotaRange(
            runtime.get(PROACTIVE_DAILY_MIN),
            runtime.get(PROACTIVE_DAILY_MAX),
            runtime.get(PROACTIVE_ENABLED),
        )

    def switch(name: str, by: str) -> None:
        runtime.set(BOT_TIMEZONE, name, by=by)

    def zone_name() -> str:
        return runtime.get(BOT_TIMEZONE)

    calendar: Cached[DayTypeCalendar] = Cached(day_calendar)
    time = BotTimeService(services.clock, zone_name, calendar=calendar)
    planner = DailyPlanner(
        db=services.db,
        clock=services.clock,
        time_service=time,
        config=settings.schedule,
        model_source=model_source or (lambda: load_activity_model(services, "live")),
        quota_source=quota_source or proactive_range,
        set_timezone=switch,
        drop_caches=(calendar.invalidate,),
    )
    time.attach_states(planner)
    return ScheduleKit(calendar, time, planner, ScheduleEvents())


def schedule_kit(services: Services) -> ScheduleKit:
    """The container's schedule objects, made on first use."""
    found = services.extras.get(KIT_KEY)
    if isinstance(found, ScheduleKit):
        return found
    kit = build_kit(services)
    services.extras[KIT_KEY] = kit
    return kit


def time_service_for(services: Services) -> BotTimeService:
    """The :class:`TimeService` of the container: what everything asks for the time."""
    return schedule_kit(services).time
