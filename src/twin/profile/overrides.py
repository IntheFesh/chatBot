"""Manual corrections of the routine (R-ACT-005; the data side of the WeChat commands).

Three kinds of corrections are stored in ``routine_overrides``; each is expressed in the
*local clock* of the routine, not in a time zone:

``sleep``
    ``{"start": "01:00", "end": "08:30", "day_types": ["workday", ...] | null}``: she sleeps
    from ``start`` to ``end`` on days of these types (null = every type);
``busy``
    ``{"weekdays": [0, 1, 2, 3, 4], "start": "13:00", "end": "17:00"}``: she is busy at
    these times on these weekdays (Monday is 0);
``holiday``
    ``{"from": "2026-10-01", "to": "2026-10-07"}``: these dates are holidays.

A correction wins over what the data suggests: :meth:`ActivityModel.with_overrides` replaces
the inferred sleep of the named day types and, for a weekday that has a busy correction,
the inferred busy windows.  The same API serves ``twin routine`` and the chat commands of
round 11.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from sqlalchemy import select

from twin.clock import Clock
from twin.profile.localtime import format_minute, parse_clock
from twin.storage.db import Database
from twin.storage.profile_models import RoutineOverride

DAY_TYPE_NAMES = ("workday", "weekend", "holiday")
WEEKDAY_NAMES = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


class OverrideError(ValueError):
    """A correction was given in a form that cannot be stored."""


@dataclass(frozen=True)
class OverrideView:
    id: str
    kind: str  # sleep | busy | holiday
    params: dict[str, Any]
    enabled: bool
    note: str | None
    created_at: datetime

    def describe(self) -> str:
        p = self.params
        if self.kind == "sleep":
            types = "、".join(p["day_types"]) if p.get("day_types") else "所有日类型"
            return f"睡眠 {p['start']}–{p['end']}（{types}）"
        if self.kind == "busy":
            days = "、".join(WEEKDAY_NAMES[d] for d in p["weekdays"])
            return f"忙碌 {days} {p['start']}–{p['end']}"
        return f"节假日 {p['from']} 至 {p['to']}"


_WEEKDAY_NAMES = {
    **{name: i for i, name in enumerate(("mon", "tue", "wed", "thu", "fri", "sat", "sun"))},
    **{
        name: i
        for i, name in enumerate(
            ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
        )
    },
    **{name: i for i, name in enumerate("一二三四五六日")},
    "天": 6,
    **{str(i + 1): i for i in range(7)},
}
_LIST_SPLIT = re.compile(r"[,，、;；\s]+")
_RANGE_SPLIT = re.compile(r"[-–—~至到]")


def _weekday(token: str) -> int:
    key = token.strip().lower()
    for prefix in ("星期", "周", "礼拜"):
        key = key.removeprefix(prefix)
    if key not in _WEEKDAY_NAMES:
        raise OverrideError(f"not a weekday: {token!r} (use mon..sun, 周一..周日 or 1..7)")
    return _WEEKDAY_NAMES[key]


def parse_weekdays(text: str) -> list[int]:
    """Weekday numbers (Monday = 0) from ``mon-fri``, ``sat,sun``, ``周一至周五`` and the like."""
    days: set[int] = set()
    for part in _LIST_SPLIT.split(text.strip()):
        if not part:
            continue
        ends = [piece for piece in _RANGE_SPLIT.split(part) if piece]
        if len(ends) == 1:
            days.add(_weekday(ends[0]))
        elif len(ends) == 2:
            first, last = _weekday(ends[0]), _weekday(ends[1])
            day = first
            while True:
                days.add(day)
                if day == last:
                    break
                day = (day + 1) % 7
        else:
            raise OverrideError(f"cannot read the weekday range {part!r}")
    if not days:
        raise OverrideError("no weekday given")
    return sorted(days)


def _view(row: RoutineOverride) -> OverrideView:
    return OverrideView(
        id=row.id,
        kind=row.kind,
        params=dict(row.params),
        enabled=row.enabled,
        note=row.note,
        created_at=row.created_at,
    )


def _clock(text: str | int) -> str:
    """``HH:MM`` of a clock time given as text or minute of day."""
    if isinstance(text, int):
        if not 0 <= text < 1440:
            raise OverrideError(f"minute of day out of range: {text}")
        return format_minute(text)
    try:
        return format_minute(parse_clock(text))
    except ValueError as exc:
        raise OverrideError(str(exc)) from exc


class RoutineOverrides:
    """Repository over ``routine_overrides``."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def _add(self, kind: str, params: dict[str, Any], note: str | None) -> OverrideView:
        now = self._clock.now_utc()
        with self._db.transaction() as session:
            row = RoutineOverride(
                kind=kind, params=params, note=note, enabled=True, created_at=now, updated_at=now
            )
            session.add(row)
            session.flush()
            return _view(row)

    def add_sleep(
        self,
        start: str | int,
        end: str | int,
        *,
        day_types: Sequence[str] | None = None,
        note: str | None = None,
    ) -> OverrideView:
        begin, finish = _clock(start), _clock(end)
        if begin == finish:
            raise OverrideError("sleep start and end are the same time")
        types = sorted(set(day_types)) if day_types else None
        if types and not set(types) <= set(DAY_TYPE_NAMES):
            raise OverrideError(f"day types must be among {', '.join(DAY_TYPE_NAMES)}")
        return self._add("sleep", {"start": begin, "end": finish, "day_types": types}, note)

    def add_busy(
        self,
        weekdays: Sequence[int],
        start: str | int,
        end: str | int,
        *,
        note: str | None = None,
    ) -> OverrideView:
        days = sorted(set(weekdays))
        if not days or not all(isinstance(d, int) and 0 <= d <= 6 for d in days):
            raise OverrideError("weekdays are numbers 0 (Monday) to 6 (Sunday)")
        begin, finish = _clock(start), _clock(end)
        if parse_clock(begin) >= parse_clock(finish):
            raise OverrideError("a busy period must start before it ends (within one day)")
        return self._add("busy", {"weekdays": days, "start": begin, "end": finish}, note)

    def add_holiday(self, first: date, last: date, *, note: str | None = None) -> OverrideView:
        if last < first:
            raise OverrideError("the holiday ends before it starts")
        return self._add("holiday", {"from": first.isoformat(), "to": last.isoformat()}, note)

    def remove(self, override_id: str) -> bool:
        with self._db.transaction() as session:
            row = session.get(RoutineOverride, override_id)
            if row is None:
                return False
            session.delete(row)
            return True

    def set_enabled(self, override_id: str, enabled: bool) -> bool:
        with self._db.transaction() as session:
            row = session.get(RoutineOverride, override_id)
            if row is None:
                return False
            row.enabled = enabled
            row.updated_at = self._clock.now_utc()
            return True

    def entries(self, *, include_disabled: bool = True) -> list[OverrideView]:
        stmt = select(RoutineOverride).order_by(RoutineOverride.created_at, RoutineOverride.id)
        if not include_disabled:
            stmt = stmt.where(RoutineOverride.enabled.is_(True))
        with self._db.session() as session:
            return [_view(row) for row in session.scalars(stmt)]

    def holiday_ranges(self) -> list[tuple[date, date]]:
        """The enabled holiday date ranges."""
        return [
            (date.fromisoformat(v.params["from"]), date.fromisoformat(v.params["to"]))
            for v in self.entries(include_disabled=False)
            if v.kind == "holiday"
        ]
