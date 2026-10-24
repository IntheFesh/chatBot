"""The random side of a day: how likely a tick is to bring a message (R-PRO-001).

The day's quota ``N`` (round 08) says how many conversations she opens on her own today.  Her
real habit says *when*: the curve ``lambda(slot)`` is the number of conversations she opens in
each 15-minute slot of the local day (the activity model, ``initiation_rate_at``).  This module
spreads ``N`` over the part of the day in which a message may go out - she is awake, not in deep
sleep - in proportion to that curve:

* the day is laid out on a grid of ``tick_minutes`` steps; a step that falls in a state in which
  she is awake gets the weight of its slot (``lambda`` plus a small floor, so that a stretch where
  her history shows nothing is not impossible; with no history at all every waking step weighs
  the same), every other step weighs nothing;
* a tick that still needs ``r`` more messages fires with the probability
  ``1 - exp(-r * w(now) / W(now))`` where ``W(now)`` is the weight that is left from now on - a
  thinned inhomogeneous Poisson process whose expected number of events in the rest of the day is
  exactly ``r``, and which conditions itself: the less time is left, the more likely a tick is,
  until it is certain.  Every message that goes out (or was planned for another reason) lowers
  ``r``, so the day ends with ``N`` messages, no matter how the dice fell;
* the last start is moved forward by the spacing: ``r`` messages that are an hour apart need
  ``(r - 1)`` hours to themselves, and from that moment on the tick is certain.

The functions here are pure (a grid, numbers, a random generator): the scheduler does the reading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from twin.profile.activity_model import ActivityModel
from twin.schedule.plan_model import DailyPlan
from twin.schedule.wallclock import day_bounds_utc

SLOT_MINUTES = 15
AWAKE_STATES = frozenset({"free", "busy"})
WEIGHT_FLOOR = 0.02  # a share of the mean weight: even a quiet slot can bring a message


@dataclass(frozen=True)
class DayGrid:
    """The weights of the steps of one local day (see the module description)."""

    start: datetime  # the first instant of the local day (UTC)
    step: timedelta
    weights: tuple[float, ...]
    prefix: tuple[float, ...]  # prefix[i] = the weight of the steps before step i

    @classmethod
    def of(cls, start: datetime, step: timedelta, weights: tuple[float, ...]) -> DayGrid:
        total = 0.0
        prefix = [0.0]
        for weight in weights:
            total += weight
            prefix.append(total)
        return cls(start, step, weights, tuple(prefix))

    @property
    def size(self) -> int:
        return len(self.weights)

    def index(self, moment: datetime) -> int:
        """The step that holds ``moment`` (before the day: -1; after it: ``size``)."""
        if moment < self.start:
            return -1
        found = int((moment - self.start) / self.step)
        return min(found, self.size)

    def remaining(self, index: int) -> float:
        """The weight from step ``index`` on (the step itself included)."""
        return self.prefix[-1] - self.prefix[max(0, min(index, self.size))]

    def probability(self, moment: datetime, needed: int, spacing: timedelta) -> float:
        """The chance that the tick at ``moment`` brings a message (see the module description)."""
        index = self.index(moment)
        if needed <= 0 or not 0 <= index < self.size:
            return 0.0
        here = self.weights[index]
        if here <= 0.0:
            return 0.0
        hold_back = max(0, needed - 1) * (spacing / self.step)
        last_start = self.size - 1 - math.ceil(hold_back)
        if index >= last_start:
            return 1.0  # no later step can still fit the messages that are owed
        usable = self.prefix[last_start + 1] - self.prefix[index]
        if usable <= 0.0:
            return 1.0
        return 1.0 - math.exp(-needed * here / usable)


def slot_of(moment: datetime, zone: ZoneInfo) -> int:
    """The 15-minute slot of the local day that holds ``moment``."""
    local = moment.astimezone(zone)
    return (local.hour * 60 + local.minute) // SLOT_MINUTES


def build_grid(
    plan: DailyPlan,
    *,
    step: timedelta,
    model: ActivityModel | None,
    day_type: str,
) -> DayGrid:
    """The grid of ``plan``'s local day: a step is usable where the plan says she is awake."""
    zone = ZoneInfo(plan.timezone)
    start, end = day_bounds_utc(plan.local_date, zone)
    count = max(0, math.ceil((end - start) / step))
    raw: list[float] = []
    usable: list[bool] = []
    for number in range(count):
        moment = start + number * step
        segment = plan.segment_at(moment)
        awake = segment is not None and segment.kind in AWAKE_STATES
        usable.append(awake)
        if model is not None:
            raw.append(model.initiation_rate_at(slot_of(moment, zone), day_type))
        else:
            raw.append(1.0)
    awake_values = [value for value, ok in zip(raw, usable, strict=True) if ok]
    mean = sum(awake_values) / len(awake_values) if awake_values else 0.0
    floor = WEIGHT_FLOOR * mean if mean > 0 else 1.0  # no history at all: even over the day
    weights = tuple((value + floor) if ok else 0.0 for value, ok in zip(raw, usable, strict=True))
    return DayGrid.of(start, step, weights)


def edge_probability(
    model: ActivityModel | None, moment: datetime, zone: ZoneInfo, day_type: str, tick: timedelta
) -> float:
    """The chance that a tick at the edge of her sleep brings a message (R-PRO-005).

    Her real late-night and early-morning openings: the conversations she opened in this slot on
    a day of this kind, per day, spread over the ticks of the slot.  Without a routine model she
    is never awake at the edge.
    """
    if model is None:
        return 0.0
    rate = model.initiation_rate_at(slot_of(moment, zone), day_type)
    return min(1.0, max(0.0, rate * (tick / timedelta(minutes=SLOT_MINUTES))))


def window_share(
    model: ActivityModel | None,
    zone: ZoneInfo,
    day_type: str,
    start: datetime,
    end: datetime,
    *,
    fallback: float,
) -> float:
    """How many conversations she opens in ``[start, end)`` on a day of this kind (at most 1).

    The sum of the opening rates of the slots the window covers, each by the share of it that
    lies inside - the chance, read from her history, that she opened one in that window.  Without
    a model ``fallback`` stands in.
    """
    if model is None:
        return fallback
    total = 0.0
    moment = start
    while moment < end:
        boundary = min(end, moment + timedelta(minutes=1))
        total += model.initiation_rate_at(slot_of(moment, zone), day_type) * (
            (boundary - moment) / timedelta(minutes=SLOT_MINUTES)
        )
        moment = boundary
    return min(1.0, max(0.0, total))
