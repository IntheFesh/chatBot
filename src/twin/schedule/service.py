"""Wiring of the schedule: one time service, one planner and one event bus per container.

:func:`schedule_kit` builds, once per :class:`~twin.services.Services`, the pieces that must be
the same object everywhere:

* the zone-name provider, which reads the runtime setting on every call (one primary-key lookup),
  so a ``/时区`` or ``twin timezone set`` - in this process or in another - is seen at once, and
  the day-type calendar, a :class:`~twin.schedule.time_service.Cached` value (it reads the holiday
  ranges set by hand) that the schedule component drops when the state watcher reports a change,
  together with the plans read from the database, within two seconds;
* the :class:`~twin.schedule.time_service.BotTimeService` the budget, the memory and the engine ask;
* the bot's zone timeline (:class:`~twin.schedule.zone_timeline.BotZoneTimeline`, a ``Cached``
  value read from ``timezone_history``), which the memory dates the bot's conversation by (D-732)
  and which is dropped together with the other caches when a switch is made;
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
from typing import TYPE_CHECKING, Any

from twin.config.runtime import (
    BOT_TIMEZONE,
    PROACTIVE_DAILY_MAX,
    PROACTIVE_DAILY_MIN,
    PROACTIVE_ENABLED,
    SettingSpec,
)
from twin.llm.pricing import PeakCalendar
from twin.profile.activity_model import ActivityModel
from twin.profile.api import load_activity_model
from twin.profile.overrides import RoutineOverrides
from twin.schedule.daytype import DayTypeCalendar
from twin.schedule.events import ScheduleEvents
from twin.schedule.plan_builder import QuotaRange
from twin.schedule.planner import DailyPlanner
from twin.schedule.store import TimezoneHistory
from twin.schedule.time_service import BotTimeService, Cached
from twin.schedule.zone_timeline import BotZoneTimeline

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
    zones: Cached[BotZoneTimeline]

    def drop_caches(self) -> None:
        """Forget what was read from the database (the state watcher calls this on a change).

        The planner's own caches go, and with them - it holds the list - the calendar and the bot's
        zone timeline.
        """
        self.planner.invalidate()


def is_bot_timezone(spec: SettingSpec[Any]) -> bool:
    """Whether ``spec`` is the bot's time zone: changing it is a switch, not a plain setting
    (``twin settings set time.bot_timezone`` goes the way of ``twin timezone set``, D-829).
    The setting is named here and in :mod:`twin.config.runtime` only."""
    return spec is BOT_TIMEZONE


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


def day_type_calendar_for(services: Services) -> DayTypeCalendar:
    """The container's day-type calendar: what the time service, the profile and the windows use.

    It is made of the configured zone-to-country table, the holiday ranges set by hand
    (``twin routine add holiday``) and the one holiday calendar of the container
    (:func:`peak_calendar_for`), so ``pricing.extra_peak_dates`` / ``extra_offpeak_dates`` change
    the day type everywhere the same way (D-733).  A new calendar is made on every call and holds
    the holiday ranges as they are now; the kit's own one is the cached one (``kit.calendar``).
    """
    return DayTypeCalendar(
        zone_countries=services.settings.safety.timezone_country,
        holiday_ranges=RoutineOverrides(services.db, services.clock).holiday_ranges(),
        peak_calendar=peak_calendar_for(services),
    )


def build_kit(
    services: Services,
    *,
    model_source: Callable[[], ActivityModel | None] | None = None,
    quota_source: Callable[[], QuotaRange] | None = None,
) -> ScheduleKit:
    """Create the schedule objects of ``services`` (see :func:`schedule_kit` for the shared set)."""
    runtime = services.runtime
    settings = services.settings

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

    calendar: Cached[DayTypeCalendar] = Cached(lambda: day_type_calendar_for(services))
    time = BotTimeService(services.clock, zone_name, calendar=calendar)
    history = TimezoneHistory(services.db, services.clock)
    zones: Cached[BotZoneTimeline] = Cached(
        lambda: BotZoneTimeline.from_history(history, zone_name)
    )
    planner = DailyPlanner(
        db=services.db,
        clock=services.clock,
        time_service=time,
        config=settings.schedule,
        model_source=model_source or (lambda: load_activity_model(services, "live")),
        quota_source=quota_source or proactive_range,
        set_timezone=switch,
        drop_caches=(calendar.invalidate, zones.invalidate),
        history=history,
        min_sleep_minutes=settings.activity.sleep_min_hours * 60.0,
    )
    time.attach_states(planner)
    return ScheduleKit(calendar, time, planner, ScheduleEvents(), zones)


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
