"""The activity (routine) model and its queries (R-ACT-002 to R-ACT-006).

The model describes when she is awake, busy, asleep, speaks first and how fast she answers,
all on the *local clock of the place she was in* (15-minute slots 0..95, day type workday /
weekend / holiday).  It does not know the current time zone: moving a routine to the bot's
zone ("8 点起床" stays 8 点) is the job of ``TimeService`` (round 08).

Queries (all pure; none samples except where the name says so)::

    rate_at(slot, day_type)              messages per slot per day
    initiation_rate_at(slot, day_type)   conversations she opens in that slot per day
    latency_distribution(slot)           seconds until she answers a message arriving then
    sleep_profile()                      typical onset / wake per day type, confidence, warnings
    busy_windows(day_type, weekday=None) busy periods, with their latency distributions
    typical_state(time, day_type)        deep_sleep | sleep_edge | busy | free

``typical_state`` is the pure function of R-ACT-006: it compares the clock time with the
*median* onset and wake time (circular) and the busy windows; the edge of the sleep is
``activity.edge_minutes`` after falling asleep and before waking.  The training export and the
evaluation sandbox use it to infer her state at a historical moment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import time
from typing import Any, Literal

from twin.profile.circular import DAY_MINUTES, CircularStats, wrap
from twin.profile.distribution import BucketedDistribution, EmpiricalDistribution
from twin.profile.localtime import SLOT_MINUTES, SLOTS_PER_DAY, format_minute, parse_clock
from twin.profile.overrides import OverrideView

SCHEMA = 1
TypicalState = Literal["deep_sleep", "sleep_edge", "busy", "free"]
LOCAL_DAY_TYPES = ("workday", "weekend", "holiday")
CURVE_KEYS = (*LOCAL_DAY_TYPES, "all")
DAY_TYPE_LABELS = {"all": "全部日子", "workday": "工作日", "weekend": "周末", "holiday": "节假日"}
MIN_TYPE_DAYS = 4  # a day type with fewer observed days borrows the curve of a broader one
FALLBACK_CHAIN: Mapping[str, tuple[str, ...]] = {
    "workday": ("workday", "all"),
    "weekend": ("weekend", "all"),
    "holiday": ("holiday", "weekend", "all"),
    "all": ("all",),
}
DAYTIME_FIRST = 10 * 60  # a sleep core inside 10:00-18:00 is implausible (R-ACT-003)
DAYTIME_LAST = 18 * 60


# ---------------------------------------------------------------------- sleep


@dataclass(frozen=True)
class SleepWindow:
    """Typical falling asleep and waking up on one kind of day."""

    onset: CircularStats
    wake: CircularStats
    valid_days: int
    source: Literal["days", "curve", "override"]

    @property
    def onset_min(self) -> float:
        return self.onset.median

    @property
    def wake_min(self) -> float:
        return self.wake.median

    def duration_min(self) -> float:
        return (self.wake.median - self.onset.median) % DAY_MINUTES

    def crosses_midnight(self) -> bool:
        return self.onset.median > self.wake.median

    def core(self, edge_minutes: float) -> tuple[float, float]:
        """Deep-sleep core ``(start, end)`` in minutes of day (``end`` may be < ``start``)."""
        edge = min(edge_minutes, self.duration_min() / 2)
        return wrap(self.onset.median + edge), wrap(self.wake.median - edge)

    def core_overlap_with_daytime(self, edge_minutes: float) -> float:
        """Fraction of the core that lies inside 10:00-18:00."""
        start, _ = self.core(edge_minutes)
        length = max(0.0, self.duration_min() - 2 * min(edge_minutes, self.duration_min() / 2))
        if length <= 0:
            return 0.0
        inside = 0.0
        step = 5.0
        position = 0.0
        while position < length:
            minute = wrap(start + position + step / 2)
            if DAYTIME_FIRST <= minute < DAYTIME_LAST:
                inside += min(step, length - position)
            position += step
        return inside / length

    def to_json(self) -> dict[str, Any]:
        return {
            "onset": self.onset.to_json(),
            "wake": self.wake.to_json(),
            "valid_days": self.valid_days,
            "source": self.source,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> SleepWindow:
        return cls(
            CircularStats.from_json(data["onset"]),
            CircularStats.from_json(data["wake"]),
            int(data["valid_days"]),
            data["source"],
        )

    @classmethod
    def manual(cls, start: float, end: float) -> SleepWindow:
        return cls(CircularStats.fixed(start), CircularStats.fixed(end), 0, "override")


@dataclass(frozen=True)
class SleepProfile:
    """Sleep per day type (``all`` = every day), how sure the model is, and warnings."""

    windows: Mapping[str, SleepWindow]
    confidence: Literal["high", "low", "none"]
    valid_days: int
    warnings: tuple[str, ...] = ()

    def window_for(self, day_type: str) -> SleepWindow | None:
        for key in (*FALLBACK_CHAIN.get(day_type, ("all",)), "all"):
            found = self.windows.get(key)
            if found is not None:
                return found
        return None

    def to_json(self) -> dict[str, Any]:
        return {
            "windows": {key: window.to_json() for key, window in self.windows.items()},
            "confidence": self.confidence,
            "valid_days": self.valid_days,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> SleepProfile:
        return cls(
            {key: SleepWindow.from_json(value) for key, value in data["windows"].items()},
            data["confidence"],
            int(data["valid_days"]),
            tuple(data.get("warnings", ())),
        )


def sleep_core_in_daytime(window: SleepWindow, edge_minutes: float) -> bool:
    """True when most of the deep-sleep core lies in 10:00-18:00 (R-ACT-003 plausibility)."""
    return window.core_overlap_with_daytime(edge_minutes) >= 0.5


def _episode_state(
    minute: float, onset: float, wake: float, edge: float
) -> Literal["deep_sleep", "sleep_edge"] | None:
    duration = (wake - onset) % DAY_MINUTES
    elapsed = (minute - onset) % DAY_MINUTES
    if elapsed >= duration:
        return None
    reach = min(edge, duration / 2)
    if elapsed < reach or duration - elapsed <= reach:
        return "sleep_edge"
    return "deep_sleep"


# ----------------------------------------------------------------------- busy


@dataclass(frozen=True)
class BusyWindow:
    """A period in which she is probably busy: little activity and slow answers."""

    start_min: float
    end_min: float
    confidence: float
    latency_ratio: float
    stability: float
    latency: EmpiricalDistribution
    source: Literal["inferred", "override"] = "inferred"
    weekdays: tuple[int, ...] = ()

    def contains(self, minute: float) -> bool:
        return self.start_min <= minute < self.end_min

    def to_json(self) -> dict[str, Any]:
        return {
            "start": self.start_min,
            "end": self.end_min,
            "confidence": round(self.confidence, 3),
            "latency_ratio": round(self.latency_ratio, 3),
            "stability": round(self.stability, 3),
            "latency": self.latency.to_json(),
            "source": self.source,
            "weekdays": list(self.weekdays),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> BusyWindow:
        return cls(
            float(data["start"]),
            float(data["end"]),
            float(data["confidence"]),
            float(data["latency_ratio"]),
            float(data["stability"]),
            EmpiricalDistribution.from_json(data["latency"]),
            data["source"],
            tuple(int(d) for d in data.get("weekdays", ())),
        )

    def label(self) -> str:
        return f"{format_minute(self.start_min)}–{format_minute(self.end_min)}"


# ---------------------------------------------------------------------- model


@dataclass(frozen=True)
class ActivityModel:
    scope: str
    zone: str
    edge_minutes: int
    days: Mapping[str, int]
    rate: Mapping[str, tuple[float, ...]]
    rate_raw: Mapping[str, tuple[float, ...]]
    initiation: Mapping[str, tuple[float, ...]]
    initiation_raw: Mapping[str, tuple[float, ...]]
    latency: BucketedDistribution
    latency_workday: BucketedDistribution
    sleep: SleepProfile
    busy: Mapping[str, tuple[BusyWindow, ...]]
    her_messages: int
    initiations_per_day: float
    manual_busy: tuple[BusyWindow, ...] = ()
    applied_overrides: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------- curve queries

    def _curve(self, table: Mapping[str, tuple[float, ...]], day_type: str) -> tuple[float, ...]:
        for key in FALLBACK_CHAIN.get(day_type, ("all",)):
            if key in table and (key == "all" or self.days.get(key, 0) >= MIN_TYPE_DAYS):
                return table[key]
        return table["all"]

    @staticmethod
    def _check_slot(slot: int) -> int:
        if not 0 <= slot < SLOTS_PER_DAY:
            raise ValueError(f"local slot must be 0..{SLOTS_PER_DAY - 1}, got {slot}")
        return slot

    def rate_at(self, local_slot: int, day_type: str) -> float:
        """Messages she sends in this 15-minute slot, per day of this type (smoothed)."""
        return self._curve(self.rate, day_type)[self._check_slot(local_slot)]

    def initiation_rate_at(self, local_slot: int, day_type: str) -> float:
        """Conversations she opens in this slot, per day of this type (smoothed)."""
        return self._curve(self.initiation, day_type)[self._check_slot(local_slot)]

    def latency_distribution(self, local_slot: int) -> EmpiricalDistribution:
        """Seconds until she answers a message that arrives in this slot (with fallbacks)."""
        return self.latency.get(self._check_slot(local_slot))

    # ------------------------------------------------------ sleep, busy, state

    def sleep_profile(self) -> SleepProfile:
        return self.sleep

    def busy_windows(self, day_type: str, weekday: int | None = None) -> tuple[BusyWindow, ...]:
        """Busy periods of a day type; manual weekly corrections replace them on their weekdays."""
        if weekday is not None:
            manual = tuple(w for w in self.manual_busy if weekday in w.weekdays)
            if manual:
                return manual
        for key in FALLBACK_CHAIN.get(day_type, ("all",)):
            if key in self.busy:
                return self.busy[key]
        return ()

    def typical_state(
        self,
        local_time: time,
        day_type: str,
        *,
        next_day_type: str | None = None,
        weekday: int | None = None,
    ) -> TypicalState:
        """Her typical state at a local clock time (pure; see the module docstring).

        ``next_day_type`` is the type of the following day: a sleep that begins this evening
        and ends tomorrow is the one tomorrow's type describes.  ``weekday`` (Monday = 0)
        enables the weekly busy corrections.
        """
        minute = local_time.hour * 60 + local_time.minute + local_time.second / 60.0
        edge = float(self.edge_minutes)
        today = self.sleep.window_for(day_type)
        tonight = self.sleep.window_for(next_day_type or day_type)
        if today is not None:
            if today.crosses_midnight():
                if minute < today.wake_min:
                    state = _episode_state(minute, today.onset_min, today.wake_min, edge)
                    if state is not None:
                        return state
            elif today.onset_min <= minute < today.wake_min:
                state = _episode_state(minute, today.onset_min, today.wake_min, edge)
                if state is not None:
                    return state
        if tonight is not None and tonight.crosses_midnight() and minute >= tonight.onset_min:
            state = _episode_state(minute, tonight.onset_min, tonight.wake_min, edge)
            if state is not None:
                return state
        if any(w.contains(minute) for w in self.busy_windows(day_type, weekday)):
            return "busy"
        return "free"

    # ---------------------------------------------------------- corrections

    def with_overrides(self, overrides: Sequence[OverrideView]) -> ActivityModel:
        """The model with the enabled manual corrections applied (they win over inference)."""
        windows = dict(self.sleep.windows)
        manual: list[BusyWindow] = []
        applied: list[str] = []
        warnings = list(self.sleep.warnings)
        confidence = self.sleep.confidence
        for item in overrides:
            if not item.enabled:
                continue
            p = item.params
            if item.kind == "sleep":
                window = SleepWindow.manual(parse_clock(p["start"]), parse_clock(p["end"]))
                for key in p.get("day_types") or CURVE_KEYS:
                    windows[key] = window
                confidence = "high"
                warnings = [w for w in warnings if "source_timezone" not in w]
                applied.append(item.id)
            elif item.kind == "busy":
                manual.append(
                    BusyWindow(
                        float(parse_clock(p["start"])),
                        float(parse_clock(p["end"])),
                        1.0,
                        0.0,
                        1.0,
                        EmpiricalDistribution.empty(discrete=True),
                        "override",
                        tuple(int(d) for d in p["weekdays"]),
                    )
                )
                applied.append(item.id)
        sleep = replace(
            self.sleep, windows=windows, confidence=confidence, warnings=tuple(warnings)
        )
        return replace(
            self,
            sleep=sleep,
            manual_busy=(*self.manual_busy, *manual),
            applied_overrides=(*self.applied_overrides, *applied),
        )

    # ---------------------------------------------------------------- JSON

    def to_json(self) -> dict[str, Any]:
        def curves(table: Mapping[str, tuple[float, ...]]) -> dict[str, list[float]]:
            return {key: [round(v, 5) for v in values] for key, values in table.items()}

        return {
            "schema": SCHEMA,
            "scope": self.scope,
            "zone": self.zone,
            "edge_minutes": self.edge_minutes,
            "slot_minutes": SLOT_MINUTES,
            "days": dict(self.days),
            "rate": curves(self.rate),
            "rate_raw": curves(self.rate_raw),
            "initiation": curves(self.initiation),
            "initiation_raw": curves(self.initiation_raw),
            "latency": self.latency.to_json(),
            "latency_workday": self.latency_workday.to_json(),
            "sleep": self.sleep.to_json(),
            "busy": {key: [w.to_json() for w in windows] for key, windows in self.busy.items()},
            "her_messages": self.her_messages,
            "initiations_per_day": round(self.initiations_per_day, 5),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ActivityModel:
        if data.get("schema") != SCHEMA:
            raise ValueError(f"unsupported activity model schema {data.get('schema')!r}")

        def curves(table: Mapping[str, Sequence[float]]) -> dict[str, tuple[float, ...]]:
            return {key: tuple(float(v) for v in values) for key, values in table.items()}

        return cls(
            scope=str(data["scope"]),
            zone=str(data["zone"]),
            edge_minutes=int(data["edge_minutes"]),
            days={key: int(value) for key, value in data["days"].items()},
            rate=curves(data["rate"]),
            rate_raw=curves(data["rate_raw"]),
            initiation=curves(data["initiation"]),
            initiation_raw=curves(data["initiation_raw"]),
            latency=BucketedDistribution.from_json(data["latency"]),
            latency_workday=BucketedDistribution.from_json(data["latency_workday"]),
            sleep=SleepProfile.from_json(data["sleep"]),
            busy={
                key: tuple(BusyWindow.from_json(w) for w in windows)
                for key, windows in data["busy"].items()
            },
            her_messages=int(data["her_messages"]),
            initiations_per_day=float(data["initiations_per_day"]),
            extra=dict(data.get("extra", {})),
        )
