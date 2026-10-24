"""Local clock positions of messages: date, minute of day and 15-minute slot (R-ACT-001).

A message time is an absolute instant (UTC).  The routine is learnt on the *clock time of
the place she was in* (``time.source_timezone``, or the zone of the matching entry of
``time.source_timezone_ranges``), so that "wakes at 8" can later be moved unchanged to the
bot's current time zone.  Daylight saving time is handled by ``zoneinfo``: the local clock
time of an instant is whatever the wall clock showed, so the hour that does not exist in
spring is never produced and the repeated hour of autumn maps both instants to the same slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from twin.ingest.times import SourceTime

SLOT_MINUTES = 15
SLOTS_PER_DAY = 24 * 60 // SLOT_MINUTES  # 96
MINUTES_PER_DAY = 24 * 60


def slot_of_minute(minute: float) -> int:
    """The 15-minute slot (0..95) containing the minute of day ``minute``."""
    return min(SLOTS_PER_DAY - 1, max(0, int(minute // SLOT_MINUTES)))


def format_minute(minute: float) -> str:
    """``HH:MM`` of a minute of day (wraps around midnight)."""
    total = round(minute) % MINUTES_PER_DAY
    return f"{total // 60:02d}:{total % 60:02d}"


def parse_clock(text: str) -> int:
    """Minute of day of an ``H:MM`` / ``HH:MM`` text (24:00 is not accepted)."""
    parts = text.strip().split(":")
    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
        raise ValueError(f"not a clock time (use HH:MM): {text!r}")
    hour, minute = int(parts[0]), int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"not a clock time (use HH:MM): {text!r}")
    return hour * 60 + minute


@dataclass(frozen=True, slots=True)
class LocalStamp:
    """Where an instant falls on the local wall clock."""

    day: date
    minute: float  # minutes since local midnight, with seconds as a fraction
    slot: int  # 0..95
    zone: str  # IANA key of the zone that applied

    @property
    def wall(self) -> float:
        """Minutes on a continuous wall-clock axis (days since year 1, times 1440, plus minute)."""
        return self.day.toordinal() * MINUTES_PER_DAY + self.minute


class LocalClock:
    """Converts instants to :class:`LocalStamp` with the source zone rules of the config."""

    def __init__(self, source_time: SourceTime) -> None:
        self._source = source_time

    def stamp(self, moment: datetime) -> LocalStamp:
        zone: ZoneInfo = self._source.zone_for(moment)
        local = moment.astimezone(UTC).astimezone(zone)
        minute = local.hour * 60 + local.minute + local.second / 60.0
        return LocalStamp(local.date(), minute, slot_of_minute(minute), zone.key)
