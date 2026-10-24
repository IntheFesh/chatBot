"""Wall-clock times of a local calendar and the instants they stand for (R-SCH-003).

A plan is written in the clock times of the place she lives ("wakes at 07:30") and stored as
instants (UTC).  Two days a year the conversion is not a one-to-one mapping, so the rules are
fixed here, once, and every part of the schedule goes through :func:`resolve`:

``gap`` - a clock time that does not exist
    In spring the clock jumps forward (Chicago, 2026-03-08: 02:00 becomes 03:00), so 02:30 is
    never shown.  It is read with the offset that was in force *before* the change: 02:30 on
    that day is 08:30 UTC, which the clock then shows as 03:30.  An event planned inside the gap
    therefore happens *later*, by the length of the gap, never earlier; two times inside the gap
    keep their distance, and a wall-clock span that starts before the gap and ends after it is
    one hour shorter in real time.

``repeated`` - a clock time that exists twice
    In autumn the clock goes back (Chicago, 2026-11-01: 02:00 becomes 01:00), so 01:30 is shown
    twice.  The *first* occurrence is used: 01:30 on that day is 06:30 UTC (still daylight
    time).  The second pass through the hour is covered by whatever state is in force; no
    boundary is ever placed in it.

A local midnight that does not exist (some zones change the clock at midnight) is the first
instant of the day, which is the moment of the change.

Because every boundary is an instant and the conversion never moves backwards in time, the
states of a day always tile it exactly: the 23-hour day of spring and the 25-hour day of
autumn have no hole and no overlap.  Wall-clock durations are kept as planned (she gets up when
the alarm rings at 07:30), so the real time slept on the night of the change differs by an hour
from a normal night.

Python's ``zoneinfo`` implements exactly these two rules for ``fold=0`` (PEP 495); this module
names them, tells the caller which case applied and gives the rest of the code a single place
to ask.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo

MINUTES_PER_DAY = 24 * 60


class WallKind(StrEnum):
    """How a clock time relates to the instants of its day."""

    NORMAL = "normal"
    GAP = "gap"  # the clock never shows this time (spring)
    REPEATED = "repeated"  # the clock shows this time twice (autumn)


@dataclass(frozen=True)
class WallTime:
    """A clock time of a local date and the instant it is read as."""

    day: date  # the local date asked for, after carrying minutes beyond 24 hours
    minute: float  # minute of that day asked for
    kind: WallKind
    utc: datetime

    def reading(self, zone: ZoneInfo) -> datetime:
        """What the wall clock shows at :attr:`utc` (differs from the request inside a gap)."""
        return self.utc.astimezone(zone)


def carry(day: date, minute: float) -> tuple[date, float]:
    """``(day, minute)`` with the minute brought into ``[0, 1440)`` by moving the date."""
    shift, rest = divmod(minute, MINUTES_PER_DAY)
    return day + timedelta(days=int(shift)), rest


def resolve(day: date, minute: float, zone: ZoneInfo) -> WallTime:
    """The instant of ``minute`` (minutes after local midnight of ``day``) in ``zone``.

    ``minute`` may be negative or beyond 1440: the date is carried first ("-30" is 23:30 of the
    day before).
    """
    real_day, real_minute = carry(day, minute)
    # adding a timedelta to an aware datetime moves the clock reading, not the instant (fold=0)
    local = datetime(real_day.year, real_day.month, real_day.day, tzinfo=zone) + timedelta(
        minutes=real_minute
    )
    first = local.utcoffset()
    second = local.replace(fold=1).utcoffset()
    if first is None or second is None:
        raise ValueError(f"time zone {zone.key} gave no offset for {local.isoformat()}")
    if first == second:
        kind = WallKind.NORMAL
    elif first > second:
        kind = WallKind.REPEATED
    else:
        kind = WallKind.GAP
    return WallTime(real_day, real_minute, kind, local.astimezone(UTC))


def to_utc(day: date, minute: float, zone: ZoneInfo) -> datetime:
    """The instant of a clock time (see :func:`resolve` for the daylight saving rules)."""
    return resolve(day, minute, zone).utc


def day_start_utc(day: date, zone: ZoneInfo) -> datetime:
    """The first instant of the local calendar ``day``."""
    return to_utc(day, 0, zone)


def day_bounds_utc(day: date, zone: ZoneInfo) -> tuple[datetime, datetime]:
    """``[start, end)`` of the local calendar ``day`` as aware UTC datetimes."""
    return day_start_utc(day, zone), day_start_utc(day + timedelta(days=1), zone)


def day_length(day: date, zone: ZoneInfo) -> timedelta:
    """How long the local ``day`` lasts: 24 hours, or 23 / 25 on the days the clock changes."""
    start, end = day_bounds_utc(day, zone)
    return end - start


def minute_of_day(moment: datetime, zone: ZoneInfo) -> float:
    """Minutes after local midnight shown by the wall clock at ``moment``."""
    local = moment.astimezone(zone)
    return local.hour * 60 + local.minute + local.second / 60.0 + local.microsecond / 6e7


def clock_text(moment: datetime, zone: ZoneInfo) -> str:
    """``HH:MM`` shown by the wall clock at ``moment``."""
    return moment.astimezone(zone).strftime("%H:%M")


@dataclass(frozen=True)
class Transition:
    """A change of the clock: the instant, the zone's name for the time before and after."""

    at: datetime
    before: str  # e.g. "CDT"
    after: str  # e.g. "CST"
    shift: timedelta  # positive when the clock jumps forward


def next_transition(
    zone: ZoneInfo, after: datetime, *, horizon_days: int = 800
) -> Transition | None:
    """The next change of the clock (daylight saving or a new standard offset) after ``after``.

    Looks day by day and then narrows the day down to the minute; ``None`` for a zone whose
    clock does not change within ``horizon_days`` (Asia/Shanghai).
    """
    start = after.astimezone(UTC)
    offset = start.astimezone(zone).utcoffset()
    low = start
    for _ in range(horizon_days):
        high = low + timedelta(days=1)
        if high.astimezone(zone).utcoffset() != offset:
            while high - low > timedelta(minutes=1):
                middle = low + (high - low) / 2
                if middle.astimezone(zone).utcoffset() == offset:
                    low = middle
                else:
                    high = middle
            before, later = low.astimezone(zone), high.astimezone(zone)
            changed = later.utcoffset()
            if offset is None or changed is None:
                return None
            return Transition(
                high.replace(second=0, microsecond=0),
                before.tzname() or "",
                later.tzname() or "",
                changed - offset,
            )
        low = high
    return None
