"""The time service: the one place that knows what time it is for her (R-SCH-001, R-SCOPE-005).

Everything that cares about "now", "today", "this month" or "is she asleep" as the user lives it
asks a :class:`TimeService` instead of converting time zones by hand: the budget (the day and
month of the account), the memory (local days of the bot's conversation), the day plan, the
proactive scheduler, the reply engine.  There are two sources of "now" and no more: the injected
:class:`~twin.clock.Clock` (this module) and, for the instant a clock time stands for, the
daylight saving rules of :mod:`twin.schedule.wallclock`.

:class:`BotTimeService` provides, for the bot's *current* time zone (``time.bot_timezone`` in the
runtime settings, default America/Chicago, switched with ``/时区``):

``now_local()`` / ``local_date()``
    the wall clock and the calendar date there;
``day_bounds_utc()`` / ``month_bounds_utc()``
    the instants a local day or month starts and ends (a day is 23 or 25 hours long when the
    clock changes);
``local_to_utc()``
    the instant of a clock time, with the rules for the times that do not exist or exist twice;
``day_type()``
    workday / weekend / holiday of a local date by the calendar of the place (the US federal
    holidays for a US zone, ``chinese_calendar`` with its make-up working days for China, the same
    calendar module as the price table's peak hours; years the library does not know fall back
    to Monday-to-Friday), plus the holiday ranges set by hand;
``her_state()``
    what she is doing: ``deep_sleep`` / ``sleep_edge`` / ``busy`` / ``free``, since when and until
    when, read from the day plan (never drawn again at the call site).

The zone name is asked of a callable on every call, so a change of the time zone - made here or
by another process - is seen by the very next call.  What is expensive to read - the holiday ranges
set by hand, the day plans - is cached by the pieces that hold it and dropped when the state watcher
reports a change (within two seconds), see :mod:`twin.schedule.service`.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

from twin.clock import Clock, ensure_aware
from twin.schedule.daytype import DayType, DayTypeCalendar
from twin.schedule.plan_model import HerState
from twin.schedule.wallclock import day_bounds_utc, to_utc


class PlanUnavailableError(RuntimeError):
    """No plan decides the requested moment and none may be made for it."""


class StateSource(Protocol):
    """Where :meth:`TimeService.her_state` reads the plans from (the planner, round 08)."""

    def state_at(self, moment: datetime) -> HerState: ...


class TimeService(Protocol):
    """Local calendar arithmetic in the bot's time zone and her state."""

    def now_utc(self) -> datetime:
        """The current instant (aware UTC), from the injected clock."""
        ...

    def bot_timezone(self) -> ZoneInfo:
        """The time zone the bot currently lives in."""
        ...

    def now_local(self) -> datetime:
        """The current time on the wall clock of the bot's time zone."""
        ...

    def local_date(self, at: datetime | None = None) -> date:
        """The bot-local calendar date of ``at`` (now if omitted)."""
        ...

    def day_bounds_utc(self, day: date) -> tuple[datetime, datetime]:
        """``[start, end)`` of the local calendar ``day`` as aware UTC datetimes."""
        ...

    def month_bounds_utc(self, day: date) -> tuple[datetime, datetime]:
        """``[start, end)`` of the local calendar month containing ``day``."""
        ...

    def local_to_utc(self, day: date, minute: float) -> datetime:
        """The instant of ``minute`` minutes after local midnight of ``day``."""
        ...

    def day_type(self, day: date | None = None) -> DayType:
        """Workday, weekend or holiday of a local date (today if omitted)."""
        ...

    def her_state(self, at: datetime | None = None) -> HerState:
        """What she is doing at ``at`` (now if omitted), read from the day plan."""
        ...


class Cached[T]:
    """A value loaded on first use and again after :meth:`invalidate` (the watcher calls it)."""

    def __init__(self, load: Callable[[], T]) -> None:
        self._load = load
        self._box: list[T] = []

    def __call__(self) -> T:
        if not self._box:
            self._box.append(self._load())
        return self._box[0]

    def invalidate(self) -> None:
        self._box.clear()


class BotTimeService:
    """:class:`TimeService` driven by a clock, a zone-name provider and the day plans."""

    def __init__(
        self,
        clock: Clock,
        timezone_name: Callable[[], str],
        *,
        calendar: Callable[[], DayTypeCalendar] | None = None,
        states: StateSource | None = None,
    ) -> None:
        self._clock = clock
        self._timezone_name = timezone_name
        self._calendar = calendar or Cached(DayTypeCalendar)
        self._states = states

    def attach_states(self, states: StateSource) -> None:
        """Connect the source of :meth:`her_state` (built after the service, it needs it)."""
        self._states = states

    # ------------------------------------------------------------ now and dates

    def now_utc(self) -> datetime:
        return self._clock.now_utc()

    def bot_timezone(self) -> ZoneInfo:
        return ZoneInfo(self._timezone_name())

    def now_local(self) -> datetime:
        return self.now_utc().astimezone(self.bot_timezone())

    def local_date(self, at: datetime | None = None) -> date:
        moment = ensure_aware(at) if at is not None else self.now_utc()
        return moment.astimezone(self.bot_timezone()).date()

    # ----------------------------------------------------------------- bounds

    def day_bounds_utc(self, day: date) -> tuple[datetime, datetime]:
        return day_bounds_utc(day, self.bot_timezone())

    def month_bounds_utc(self, day: date) -> tuple[datetime, datetime]:
        zone = self.bot_timezone()
        first = day.replace(day=1)
        following = (
            date(first.year + 1, 1, 1)
            if first.month == 12
            else date(first.year, first.month + 1, 1)
        )
        return to_utc(first, 0, zone), to_utc(following, 0, zone)

    def local_to_utc(self, day: date, minute: float) -> datetime:
        return to_utc(day, minute, self.bot_timezone())

    # --------------------------------------------------------- day type, state

    def day_type(self, day: date | None = None) -> DayType:
        target = day if day is not None else self.local_date()
        return self._calendar().day_type(target, self.bot_timezone().key)

    def next_day_type(self, day: date | None = None) -> DayType:
        """The type of the day after ``day`` (a night belongs to the day she wakes on)."""
        target = day if day is not None else self.local_date()
        return self.day_type(target + timedelta(days=1))

    def her_state(self, at: datetime | None = None) -> HerState:
        if self._states is None:
            raise PlanUnavailableError("this time service has no day plans attached")
        moment = ensure_aware(at) if at is not None else self.now_utc()
        return self._states.state_at(moment)
