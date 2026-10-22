"""The zone the bot lived in at any moment, from the history of its switches (R-SCH-002, D-732).

``/时区 Asia/Shanghai`` changes the bot's time zone from one instant on; ``timezone_history``
records each change.  Everything that dates a moment of the bot's *past* - the daily summaries, the
extractor's idea of "tomorrow", the as-of memory - needs the zone that was in force then, not the
one in force now: the evening of 5 October in Chicago is the 6th in Shanghai, and a bot that was
switched on the 6th must still call that evening the 5th.  :class:`BotZoneTimeline` answers that;
:class:`~twin.memory.localdate.MemoryClock` is what the memory asks (this module is in the schedule
package because the schedule owns the history and the caches, and the memory already depends on the
schedule, not the other way round).

The timeline is read once from the table and kept in the schedule kit as a
:class:`~twin.schedule.time_service.Cached` value, which :meth:`ScheduleKit.drop_caches
<twin.schedule.service.ScheduleKit.drop_caches>` drops together with the planner's other caches
(after a switch made here, and when the state watcher sees that another process made one).
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from twin.clock import ensure_aware

if TYPE_CHECKING:
    from twin.schedule.store import TimezoneHistory

NOON_UTC = 12  # the hour of a date at which the zone of that date is looked up (zone_on)


@dataclass(frozen=True)
class ZoneSwitch:
    """One change of the bot's time zone: from ``at`` on it lives in ``to_zone``."""

    at: datetime
    from_zone: str
    to_zone: str


class BotZoneTimeline:
    """The zone the bot lived in at any moment, from the history of its time zone switches.

    ``switches`` are the rows of ``timezone_history``; ``current`` names the zone the bot lives in
    now (the runtime setting).  Before the first switch the zone is the one the bot was switched
    *from*; between two switches the one it was switched *to*; after the last one - and when it
    was never switched - the current zone, so nothing about the present changes.

    :meth:`zone_for` is for a moment: a switch is in force from its own instant.  :meth:`zone_on`
    is for a calendar day and gives the zone in force at 12:00 UTC of its date, the way a trip of
    hers is a range of dates (R-ACT-001): a day is cut in one zone only.

    **Known limitation** (of the same kind as D-233's same-label one): a day on which the zone was
    switched is cut in one of the two zones, so the bounds of the days on either side of a switch
    do not fit exactly.  When the clock jumps forward (Chicago to Shanghai) the last day of the old
    zone and the first of the new one share thirteen hours, and the lines of those hours are read
    by both days; when it jumps back (Shanghai to Chicago) thirteen hours belong to neither day,
    and their lines are not in any daily summary.  A *moment* is never dated wrongly - it is always
    dated in its own zone.  A summary written before this timeline existed keeps the date it was
    given; ``twin memory summarize --force`` writes it again.
    """

    def __init__(self, switches: Sequence[ZoneSwitch], current: Callable[[], str]) -> None:
        ordered = sorted(switches, key=lambda switch: ensure_aware(switch.at))
        self._switches = tuple(ordered)
        self._instants = [ensure_aware(switch.at) for switch in ordered]
        self._current = current

    @classmethod
    def from_history(cls, history: TimezoneHistory, current: Callable[[], str]) -> BotZoneTimeline:
        """The timeline of the rows of ``timezone_history``, read once."""
        rows = history.after(None)
        return cls(
            [ZoneSwitch(r.changed_at, r.from_timezone, r.to_timezone) for r in rows], current
        )

    @property
    def switches(self) -> tuple[ZoneSwitch, ...]:
        """The recorded switches, oldest first."""
        return self._switches

    def zone_for(self, moment: datetime) -> ZoneInfo:
        """The zone the bot lived in at ``moment`` (a switch takes effect at its own instant)."""
        passed = bisect_right(self._instants, ensure_aware(moment))
        if not self._switches or passed == len(self._switches):
            return ZoneInfo(self._current())
        if passed == 0:
            return ZoneInfo(self._switches[0].from_zone)
        return ZoneInfo(self._switches[passed - 1].to_zone)

    def zone_on(self, day: date) -> ZoneInfo:
        """The zone a summary of the calendar ``day`` is cut in: the one at 12:00 UTC of it."""
        return self.zone_for(datetime(day.year, day.month, day.day, NOON_UTC, tzinfo=UTC))
