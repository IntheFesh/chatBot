"""The time service: local dates in the bot's time zone (R-SCOPE-005, R-LLM-008).

Everything that cares about "today" or "this month" as the user lives it (the budget, the
day plan, greetings) asks a :class:`TimeService` instead of converting time zones by hand.
Round 01 needs only the local date and the UTC bounds of a local day or month, provided by
:class:`ConfiguredTimeService`, which reads the *current* bot time zone through a callable so
that a ``/时区`` switch takes effect on the next call.  Round 08 extends the service with the
rest of the time and schedule logic behind the same protocol.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

from twin.clock import Clock, ensure_aware


class TimeService(Protocol):
    """Local calendar arithmetic in the bot's time zone."""

    def bot_timezone(self) -> ZoneInfo:
        """The time zone the bot currently lives in."""
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


class ConfiguredTimeService:
    """:class:`TimeService` driven by a clock and a time zone name that may change."""

    def __init__(self, clock: Clock, timezone_name: Callable[[], str]) -> None:
        self._clock = clock
        self._timezone_name = timezone_name

    def bot_timezone(self) -> ZoneInfo:
        return ZoneInfo(self._timezone_name())

    def local_date(self, at: datetime | None = None) -> date:
        moment = ensure_aware(at) if at is not None else self._clock.now_utc()
        return moment.astimezone(self.bot_timezone()).date()

    def _midnight_utc(self, day: date) -> datetime:
        zone = self.bot_timezone()
        return datetime(day.year, day.month, day.day, tzinfo=zone).astimezone(UTC)

    def day_bounds_utc(self, day: date) -> tuple[datetime, datetime]:
        return self._midnight_utc(day), self._midnight_utc(day + timedelta(days=1))

    def month_bounds_utc(self, day: date) -> tuple[datetime, datetime]:
        first = day.replace(day=1)
        following = (
            date(first.year + 1, 1, 1)
            if first.month == 12
            else date(first.year, first.month + 1, 1)
        )
        return self._midnight_utc(first), self._midnight_utc(following)
