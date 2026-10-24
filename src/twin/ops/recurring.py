"""When something that repeats was last due, on a time zone's clock (R-OPS-003/005/006).

The backup (every day at ``ops.backup_hour_local``), the integrity check (Sundays 03:30) and the
monthly cost report (the 1st, 09:00) are all "at this local time, every day / week / month".
:func:`latest_due` answers the one question each of them asks: *which was the most recent such
moment that is not in the future?*  A task that remembers when it last ran compares the two; one
that was not running at the moment (the computer was off) therefore still runs once, when the
application is next up, and never twice for the same moment.

Moments are built on the wall clock of the zone and converted to UTC, so a daylight-saving change
moves them in UTC but not on the clock, which is what "every day at 04:00" means.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Daily:
    hour: int
    minute: int = 0


@dataclass(frozen=True)
class Weekly:
    weekday: int  # Monday is 0, Sunday is 6
    hour: int
    minute: int = 0


@dataclass(frozen=True)
class Monthly:
    day: int
    hour: int
    minute: int = 0


Rule = Daily | Weekly | Monthly


def _at(day: date, hour: int, minute: int, zone: ZoneInfo) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=zone)


def _later(candidate: datetime, now: datetime) -> bool:
    """Whether a wall-clock moment is after ``now`` - compared as instants, so that a moment
    inside a daylight-saving gap or fold is not judged by its (shared-zone) wall time."""
    return candidate.astimezone(UTC) > now.astimezone(UTC)


def latest_due(rule: Rule, now: datetime, zone: ZoneInfo) -> datetime:
    """The most recent moment of ``rule`` at or before ``now`` (aware, UTC)."""
    local = now.astimezone(zone)
    today = local.date()
    match rule:
        case Daily(hour, minute):
            candidate = _at(today, hour, minute, zone)
            if _later(candidate, now):
                candidate = _at(today - timedelta(days=1), hour, minute, zone)
        case Weekly(weekday, hour, minute):
            day = today - timedelta(days=(today.weekday() - weekday) % 7)
            candidate = _at(day, hour, minute, zone)
            if _later(candidate, now):
                candidate = _at(day - timedelta(days=7), hour, minute, zone)
        case Monthly(day_of_month, hour, minute):
            candidate = _at(_month_day(today.year, today.month, day_of_month), hour, minute, zone)
            if _later(candidate, now):
                first = today.replace(day=1) - timedelta(days=1)
                candidate = _at(
                    _month_day(first.year, first.month, day_of_month), hour, minute, zone
                )
    return candidate.astimezone(UTC)


def _month_day(year: int, month: int, day: int) -> date:
    """``day`` of the month, or its last day when the month is shorter."""
    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


def is_due(rule: Rule, now: datetime, zone: ZoneInfo, last_run: datetime | None) -> bool:
    """True if the latest moment of ``rule`` has not been served yet (``last_run`` is before it)."""
    return last_run is None or last_run < latest_due(rule, now, zone)
