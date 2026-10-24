"""Local calendar days of the memory: hers for the real records, the bot's for its own.

A summary belongs to a *local day* (R-MEM-002).  The real records are dated on the clock of the
place she was in (``time.source_timezone`` and its date ranges, R-ACT-001: :class:`SourceTime`);
the bot's own conversation is dated in the zone the bot lived in *at that moment*
(:class:`BotZoneTimeline`, built from the history of the ``/时区`` switches; D-732).  Before the
timeline existed every moment was dated in the bot's current zone, so after a switch the evenings
before it moved to the next day.  :class:`MemoryClock` is the one place that converts between
instants and these days, so the summarizer, the replay, the as-of view and the assembler cut days
the same way.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from twin.clock import ensure_aware
from twin.ingest.times import SourceTime
from twin.schedule.service import schedule_kit
from twin.schedule.time_service import TimeService
from twin.schedule.wallclock import day_bounds_utc
from twin.schedule.zone_timeline import BotZoneTimeline, ZoneSwitch

if TYPE_CHECKING:
    from twin.services import Services

REAL = "real"
BOT = "bot"

__all__ = ["BOT", "REAL", "BotZoneTimeline", "MemoryClock", "ZoneSwitch"]


class MemoryClock:
    """Local dates and day bounds for both scopes of the memory.

    ``bot_zones`` gives the bot's timeline (the container's cached one, which the schedule drops
    when a switch is made); without it the bot lives in the zone the time service says for all
    moments - the right answer as long as it was never switched.
    """

    def __init__(
        self,
        source_time: SourceTime,
        time_service: TimeService,
        bot_zones: Callable[[], BotZoneTimeline] | None = None,
    ) -> None:
        self._source = source_time
        self._time = time_service
        if bot_zones is None:
            unswitched = BotZoneTimeline((), lambda: time_service.bot_timezone().key)

            def bot_zones() -> BotZoneTimeline:
                return unswitched

        self._bot_zones = bot_zones

    @classmethod
    def from_services(
        cls, services: Services, time_service: TimeService | None = None
    ) -> MemoryClock:
        kit = schedule_kit(services)
        return cls(
            SourceTime.from_config(services.settings.time), time_service or kit.time, kit.zones
        )

    @property
    def source_time(self) -> SourceTime:
        return self._source

    # ------------------------------------------------------------------- real

    def real_zone(self, moment: datetime) -> ZoneInfo:
        """The zone she was in at ``moment``."""
        return self._source.zone_for(moment)

    def real_date(self, moment: datetime) -> date:
        """Her local day that holds ``moment`` (D-782: the same days :meth:`real_bounds` cuts)."""
        return self._source.local_date(moment)

    def real_bounds(self, day: date) -> tuple[datetime, datetime]:
        """``[start, end)`` of her local ``day``: from its midnight in the zone of its date to the
        next date's midnight in the zone of that one, so the days fit together (D-782)."""
        return self._source.day_bounds(day)

    # -------------------------------------------------------------------- bot

    def bot_zone(self) -> ZoneInfo:
        """The zone the bot lives in now (for a moment in the past: :meth:`zone_of`)."""
        return self._time.bot_timezone()

    def bot_date(self, moment: datetime) -> date:
        """The calendar date of ``moment`` in the zone the bot lived in then."""
        at = ensure_aware(moment)
        return at.astimezone(self._bot_zones().zone_for(at)).date()

    def bot_bounds(self, day: date) -> tuple[datetime, datetime]:
        """``[start, end)`` of the bot's local ``day``, cut in the zone of that day."""
        return day_bounds_utc(day, self._bot_zones().zone_on(day))

    # ----------------------------------------------------------------- either

    def zone_of(self, scope: str, moment: datetime) -> ZoneInfo:
        """The zone ``scope`` was in at ``moment``."""
        return self.real_zone(moment) if scope == REAL else self._bot_zones().zone_for(moment)

    def date_of(self, scope: str, moment: datetime) -> date:
        return self.real_date(moment) if scope == REAL else self.bot_date(moment)

    def bounds_of(self, scope: str, day: date) -> tuple[datetime, datetime]:
        return self.real_bounds(day) if scope == REAL else self.bot_bounds(day)

    def zone_key_for_day(self, scope: str, day: date) -> str:
        """The IANA name of the zone a summary of ``day`` is cut in."""
        return (
            self._source.zone_on(day).key if scope == REAL else self._bot_zones().zone_on(day).key
        )
