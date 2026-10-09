"""Local calendar days of the memory: hers for the real records, the bot's for its own.

A summary belongs to a *local day* (R-MEM-002).  The real records are dated on the clock of the
place she was in (``time.source_timezone`` and its date ranges, R-ACT-001: :class:`SourceTime`);
the bot's own conversation is dated in the bot's time zone, asked of the
:class:`~twin.schedule.time_service.TimeService` (round 08 completes it; only the protocol is
used here).  :class:`MemoryClock` is the one place that converts between instants and these days,
so the summarizer, the replay, the as-of view and the assembler cut days the same way.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from twin.config.runtime import BOT_TIMEZONE
from twin.ingest.times import SourceTime
from twin.schedule.time_service import ConfiguredTimeService, TimeService

if TYPE_CHECKING:
    from twin.services import Services

REAL = "real"
BOT = "bot"


class MemoryClock:
    """Local dates and day bounds for both scopes of the memory."""

    def __init__(self, source_time: SourceTime, time_service: TimeService) -> None:
        self._source = source_time
        self._time = time_service

    @classmethod
    def from_services(
        cls, services: Services, time_service: TimeService | None = None
    ) -> MemoryClock:
        service = time_service or ConfiguredTimeService(
            services.clock, lambda: services.runtime.get(BOT_TIMEZONE)
        )
        return cls(SourceTime.from_config(services.settings.time), service)

    @property
    def source_time(self) -> SourceTime:
        return self._source

    # ------------------------------------------------------------------- real

    def real_zone(self, moment: datetime) -> ZoneInfo:
        """The zone she was in at ``moment``."""
        return self._source.zone_for(moment)

    def real_date(self, moment: datetime) -> date:
        return self._source.local_date(moment)

    def real_bounds(self, day: date) -> tuple[datetime, datetime]:
        """``[start, end)`` of her local ``day`` (in the zone that applies to that date)."""
        zone = self._source.zone_on(day)
        return _bounds(day, zone)

    # -------------------------------------------------------------------- bot

    def bot_zone(self) -> ZoneInfo:
        return self._time.bot_timezone()

    def bot_date(self, moment: datetime) -> date:
        return self._time.local_date(moment)

    def bot_bounds(self, day: date) -> tuple[datetime, datetime]:
        return self._time.day_bounds_utc(day)

    # ----------------------------------------------------------------- either

    def zone_of(self, scope: str, moment: datetime) -> ZoneInfo:
        return self.real_zone(moment) if scope == REAL else self.bot_zone()

    def date_of(self, scope: str, moment: datetime) -> date:
        return self.real_date(moment) if scope == REAL else self.bot_date(moment)

    def bounds_of(self, scope: str, day: date) -> tuple[datetime, datetime]:
        return self.real_bounds(day) if scope == REAL else self.bot_bounds(day)

    def zone_key_for_day(self, scope: str, day: date) -> str:
        """The IANA name of the zone a summary of ``day`` is cut in."""
        return self._source.zone_on(day).key if scope == REAL else self.bot_zone().key


def _bounds(day: date, zone: ZoneInfo) -> tuple[datetime, datetime]:
    start = datetime(day.year, day.month, day.day, tzinfo=zone).astimezone(UTC)
    following = day + timedelta(days=1)
    end = datetime(following.year, following.month, following.day, tzinfo=zone).astimezone(UTC)
    return start, end
