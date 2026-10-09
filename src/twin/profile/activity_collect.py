"""Collecting the raw counts of the routine model in one pass (R-ACT-002).

:class:`ActivityCollector` is fed every message together with the :class:`~twin.profile.units.Step`
the unit tracker produced for it (so bursts, segments and latencies are computed once).  It
keeps, per day type, her messages and her initiations in each 15-minute local slot, the days on
which anything was said, the reply-latency histograms by the slot in which the message she
answers arrived, and the local wall-clock position of each of her messages for the per-day
sleep search.  :class:`ActivityRaw` is what the inference (:mod:`twin.profile.activity_infer`)
reads.
"""

from __future__ import annotations

from array import array
from collections import Counter
from dataclasses import dataclass, field
from datetime import date

from twin.profile.localtime import SLOTS_PER_DAY
from twin.profile.units import Rec, Step

ALL = "all"
WORKDAY = "workday"


@dataclass
class ActivityRaw:
    """Counters and positions gathered from the messages of one scope."""

    zone_names: list[str]
    day_types: dict[date, str]
    counts: dict[str, list[int]]
    inits: dict[str, list[int]]
    latency: dict[int, Counter[float]]
    latency_workday: dict[int, Counter[float]]
    her_wall: array[float]
    her_zone: array[int]
    workday_day_counts: dict[date, list[int]]
    her_messages: int
    initiations: int
    first_day: date | None
    last_day: date | None
    days: dict[str, int] = field(default_factory=dict)

    def primary_zone(self) -> str:
        if not self.zone_names:
            return ""
        used = Counter(self.her_zone)
        return self.zone_names[used.most_common(1)[0][0]] if used else self.zone_names[0]


class ActivityCollector:
    """Accumulates :class:`ActivityRaw` message by message."""

    def __init__(self) -> None:
        self._zones: list[str] = []
        self._zone_index: dict[str, int] = {}
        self._day_types: dict[date, str] = {}
        self._counts: dict[str, list[int]] = {}
        self._inits: dict[str, list[int]] = {}
        self._latency: dict[int, Counter[float]] = {}
        self._latency_workday: dict[int, Counter[float]] = {}
        self._wall: array[float] = array("d")
        self._zone_ids: array[int] = array("b")
        self._workday_counts: dict[date, list[int]] = {}
        self._her_messages = 0
        self._initiations = 0

    def _curve(self, table: dict[str, list[int]], key: str) -> list[int]:
        found = table.get(key)
        if found is None:
            found = [0] * SLOTS_PER_DAY
            table[key] = found
        return found

    def _zone_id(self, name: str) -> int:
        found = self._zone_index.get(name)
        if found is None:
            found = len(self._zones)
            self._zones.append(name)
            self._zone_index[name] = found
        return found

    def feed(self, rec: Rec, step: Step) -> None:
        day = rec.stamp.day
        if day not in self._day_types:
            self._day_types[day] = rec.day_type
        if not rec.her:
            return
        slot = rec.stamp.slot
        self._her_messages += 1
        self._curve(self._counts, rec.day_type)[slot] += 1
        self._curve(self._counts, ALL)[slot] += 1
        self._wall.append(rec.stamp.wall)
        self._zone_ids.append(self._zone_id(rec.stamp.zone))
        if rec.day_type == WORKDAY:
            per_day = self._workday_counts.get(day)
            if per_day is None:
                per_day = [0] * SLOTS_PER_DAY
                self._workday_counts[day] = per_day
            per_day[slot] += 1
        if step.initiation:
            self._initiations += 1
            self._curve(self._inits, rec.day_type)[slot] += 1
            self._curve(self._inits, ALL)[slot] += 1
        if step.latency_s is not None and step.answered is not None:
            seconds = float(round(step.latency_s))
            arrival = step.answered.stamp.slot
            self._latency.setdefault(arrival, Counter())[seconds] += 1
            if step.answered.day_type == WORKDAY:
                self._latency_workday.setdefault(arrival, Counter())[seconds] += 1

    def finish(self) -> ActivityRaw:
        days: dict[str, int] = Counter(self._day_types.values())
        days[ALL] = len(self._day_types)
        for key in (ALL, "workday", "weekend", "holiday"):
            self._curve(self._counts, key)
            self._curve(self._inits, key)
            days.setdefault(key, 0)
        observed = sorted(self._day_types)
        return ActivityRaw(
            zone_names=list(self._zones),
            day_types=dict(self._day_types),
            counts=self._counts,
            inits=self._inits,
            latency=self._latency,
            latency_workday=self._latency_workday,
            her_wall=self._wall,
            her_zone=self._zone_ids,
            workday_day_counts=self._workday_counts,
            her_messages=self._her_messages,
            initiations=self._initiations,
            first_day=observed[0] if observed else None,
            last_day=observed[-1] if observed else None,
            days=dict(days),
        )
