"""The shapes of a day plan: sleep, busy periods, meals, the proactive quota, her state (R-SCH-004).

A :class:`DailyPlan` is what the scheduler decides about one local day, once, with a seeded random
generator, and then only reads.  Everything in it is an instant (UTC) or a number; the clock
times she lives by are produced when it is shown.  It covers the stretch ``[effective_from,
ends_at)``: a plan made at 00:05 starts at local midnight and runs to the end of the coming night
(her next wake-up), a plan made in the middle of the day because the time zone changed starts at
that moment.

The states of the stretch are derived, not stored: sleep (``sleep_edge`` for the first and the
last ``edge_minutes`` of a night, ``deep_sleep`` between) beats ``busy`` beats ``free``
(:func:`build_segments`).  ``HerState`` is what the rest of the application asks for.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import cached_property
from itertools import pairwise
from typing import Any, Literal

StateKind = Literal["deep_sleep", "sleep_edge", "busy", "free"]
STATE_KINDS: tuple[StateKind, ...] = ("deep_sleep", "sleep_edge", "busy", "free")
SLEEP_KINDS = frozenset({"deep_sleep", "sleep_edge"})
SchemaVersion = 1
MealKind = Literal["breakfast", "lunch", "dinner"]


def _iso(moment: datetime) -> str:
    return moment.isoformat()


def _when(text: str) -> datetime:
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        raise ValueError(f"a stored plan time without a zone: {text!r}")
    return moment


# ----------------------------------------------------------------------- pieces


@dataclass(frozen=True)
class BusySpan:
    """A period in which she is probably busy, with the reference of its latency distribution.

    ``latency_ref`` names the busy window of the activity model the span came from
    (:func:`busy_ref`), so the reply delay of this period is sampled from that window's own
    distribution; the median and the 90th percentile are copied here to show in a plan.
    """

    start: datetime
    end: datetime
    label: str  # the clock times, e.g. "13:00–17:00"
    source: Literal["inferred", "override"]
    latency_ref: str
    latency_median_s: float | None = None
    latency_p90_s: float | None = None

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment < self.end

    def to_json(self) -> dict[str, Any]:
        return {
            "start": _iso(self.start),
            "end": _iso(self.end),
            "label": self.label,
            "source": self.source,
            "latency_ref": self.latency_ref,
            "latency_median_s": self.latency_median_s,
            "latency_p90_s": self.latency_p90_s,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> BusySpan:
        return cls(
            _when(data["start"]),
            _when(data["end"]),
            str(data["label"]),
            data["source"],
            str(data["latency_ref"]),
            data.get("latency_median_s"),
            data.get("latency_p90_s"),
        )


def busy_ref(day_type: str, weekday: int | None, index: int) -> str:
    """The reference of a busy window: the arguments of ``ActivityModel.busy_windows`` + index."""
    return f"{day_type}|{'-' if weekday is None else weekday}|{index}"


def parse_busy_ref(ref: str) -> tuple[str, int | None, int]:
    """``(day_type, weekday or None, index)`` of a reference made by :func:`busy_ref`."""
    day_type, weekday, index = ref.split("|")
    return day_type, None if weekday == "-" else int(weekday), int(index)


@dataclass(frozen=True)
class SleepEpisode:
    """One night: she falls asleep at ``onset`` and wakes at ``wake``.

    ``adopted_from`` names the earlier plan whose night this is (a day's morning continues the
    previous day's night, so the two plans agree); ``clipped`` says that the onset is later than
    the sampled one because she went to bed at the moment the plan was rebuilt (a time zone
    switch that makes it night).
    """

    onset: datetime
    wake: datetime
    edge_minutes: int
    source: Literal["days", "curve", "override"]
    adopted_from: str | None = None
    clipped: bool = False

    @property
    def duration(self) -> timedelta:
        return self.wake - self.onset

    def edge(self) -> timedelta:
        return min(timedelta(minutes=self.edge_minutes), self.duration / 2)

    def intervals(self) -> list[tuple[StateKind, datetime, datetime]]:
        """The night as ``sleep_edge`` / ``deep_sleep`` / ``sleep_edge`` pieces."""
        edge = self.edge()
        if edge <= timedelta(0):
            return [("deep_sleep", self.onset, self.wake)]
        return [
            ("sleep_edge", self.onset, self.onset + edge),
            ("deep_sleep", self.onset + edge, self.wake - edge),
            ("sleep_edge", self.wake - edge, self.wake),
        ]

    def contains(self, moment: datetime) -> bool:
        return self.onset <= moment < self.wake

    def to_json(self) -> dict[str, Any]:
        return {
            "onset": _iso(self.onset),
            "wake": _iso(self.wake),
            "edge_minutes": self.edge_minutes,
            "source": self.source,
            "adopted_from": self.adopted_from,
            "clipped": self.clipped,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> SleepEpisode:
        return cls(
            _when(data["onset"]),
            _when(data["wake"]),
            int(data["edge_minutes"]),
            data["source"],
            data.get("adopted_from"),
            bool(data.get("clipped", False)),
        )


@dataclass(frozen=True)
class Meal:
    kind: MealKind
    at: datetime
    minutes: int
    source: Literal["history", "default"]

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "at": _iso(self.at),
            "minutes": self.minutes,
            "source": self.source,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Meal:
        return cls(data["kind"], _when(data["at"]), int(data["minutes"]), data["source"])


@dataclass(frozen=True)
class Quota:
    """How many conversations the bot may open on its own (R-PRO-002).

    ``total`` is drawn for the whole local day.  ``for_plan`` is the share that belongs to this
    plan: the same as ``total`` for a plan that covers the day, scaled by the awake time that is
    left for a plan made in the middle of it.  Round 10 spends ``for_plan`` against the messages
    sent since the plan took effect.
    """

    total: int
    for_plan: int
    minimum: int
    maximum: int
    mean_target: float | None
    mean_source: Literal["profile", "range_midpoint", "disabled"]
    enabled: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "for_plan": self.for_plan,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean_target": self.mean_target,
            "mean_source": self.mean_source,
            "enabled": self.enabled,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Quota:
        return cls(
            int(data["total"]),
            int(data["for_plan"]),
            int(data["minimum"]),
            int(data["maximum"]),
            data.get("mean_target"),
            data["mean_source"],
            bool(data["enabled"]),
        )


@dataclass(frozen=True)
class GreetingWindow:
    """When the wake-up greeting may be sent, and why not (R-PRO-004, R-SCH-002)."""

    allowed: bool
    reason: str
    earliest: datetime | None = None
    latest: datetime | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "earliest": _iso(self.earliest) if self.earliest else None,
            "latest": _iso(self.latest) if self.latest else None,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> GreetingWindow:
        earliest, latest = data.get("earliest"), data.get("latest")
        return cls(
            bool(data["allowed"]),
            str(data["reason"]),
            _when(earliest) if earliest else None,
            _when(latest) if latest else None,
        )


# --------------------------------------------------------------------- the states


@dataclass(frozen=True)
class Segment:
    """A stretch of one state inside a plan."""

    kind: StateKind
    start: datetime
    end: datetime
    busy: BusySpan | None = None

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment < self.end


@dataclass(frozen=True)
class HerState:
    """What she is doing: the state, since when it lasts, until when, and from which plan."""

    kind: StateKind
    since: datetime
    until: datetime
    plan_id: str | None
    busy: BusySpan | None = None

    @property
    def asleep(self) -> bool:
        return self.kind in SLEEP_KINDS


_RANK = {"deep_sleep": 3, "sleep_edge": 2, "busy": 1}


def build_segments(
    start: datetime,
    end: datetime,
    episodes: Sequence[SleepEpisode],
    busy: Sequence[BusySpan],
) -> tuple[Segment, ...]:
    """Tile ``[start, end)`` with states: sleep beats busy beats free; equal neighbours merge."""
    if end <= start:
        return ()
    pieces: list[Segment] = []
    for episode in episodes:
        for piece_kind, first, last in episode.intervals():
            pieces.append(Segment(piece_kind, first, last))
    for span in busy:
        pieces.append(Segment("busy", span.start, span.end, span))
    cuts = {start, end}
    for piece in pieces:
        for moment in (piece.start, piece.end):
            if start < moment < end:
                cuts.add(moment)
    segments: list[Segment] = []
    for first, last in pairwise(sorted(cuts)):
        winner: Segment | None = None
        for piece in pieces:
            if (
                piece.start <= first
                and last <= piece.end
                and (winner is None or _RANK[piece.kind] > _RANK[winner.kind])
            ):
                winner = piece
        kind: StateKind = winner.kind if winner is not None else "free"
        owner = winner.busy if winner is not None else None
        previous = segments[-1] if segments else None
        if previous is not None and previous.kind == kind and previous.busy == owner:
            segments[-1] = Segment(kind, previous.start, last, owner)
        else:
            segments.append(Segment(kind, first, last, owner))
    return tuple(segments)


# ---------------------------------------------------------------------- the plan


@dataclass(frozen=True)
class DailyPlan:
    """What the scheduler decided about one local day (see the module description)."""

    id: str
    local_date: date
    timezone: str
    day_type: str
    seed: int
    effective_from: datetime
    ends_at: datetime
    created_at: datetime
    reason: str
    morning: SleepEpisode | None
    night: SleepEpisode | None
    busy: tuple[BusySpan, ...]
    meals: tuple[Meal, ...]
    quota: Quota
    greeting: GreetingWindow
    warnings: tuple[str, ...] = ()
    inputs_hash: str = ""
    superseded_by: str | None = None
    superseded_at: datetime | None = None
    lifeline_job_id: str | None = None
    lifeline_done_at: datetime | None = None
    summary_queued_at: datetime | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def episodes(self) -> tuple[SleepEpisode, ...]:
        return tuple(e for e in (self.morning, self.night) if e is not None)

    @property
    def wake(self) -> datetime | None:
        """The wake-up of this local day (the end of the morning's night), if it is planned."""
        return self.morning.wake if self.morning else None

    @cached_property
    def segments(self) -> tuple[Segment, ...]:
        return build_segments(self.effective_from, self.ends_at, self.episodes, self.busy)

    def covers(self, moment: datetime) -> bool:
        """Whether this plan decides the state at ``moment`` (it was in force then)."""
        if not self.effective_from <= moment < self.ends_at:
            return False
        return self.superseded_at is None or moment < self.superseded_at

    def segment_at(self, moment: datetime) -> Segment | None:
        for segment in self.segments:
            if segment.contains(moment):
                return segment
        return None

    def to_json(self) -> dict[str, Any]:
        """The content that is stored sealed in ``daily_plans.plan``."""
        return {
            "schema": SchemaVersion,
            "local_date": self.local_date.isoformat(),
            "timezone": self.timezone,
            "day_type": self.day_type,
            "seed": self.seed,
            "effective_from": _iso(self.effective_from),
            "ends_at": _iso(self.ends_at),
            "reason": self.reason,
            "morning": self.morning.to_json() if self.morning else None,
            "night": self.night.to_json() if self.night else None,
            "busy": [span.to_json() for span in self.busy],
            "meals": [meal.to_json() for meal in self.meals],
            "quota": self.quota.to_json(),
            "greeting": self.greeting.to_json(),
            "warnings": list(self.warnings),
            "inputs_hash": self.inputs_hash,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_json(
        cls,
        data: Mapping[str, Any],
        *,
        id: str,
        created_at: datetime,
        superseded_by: str | None = None,
        superseded_at: datetime | None = None,
        lifeline_job_id: str | None = None,
        lifeline_done_at: datetime | None = None,
        summary_queued_at: datetime | None = None,
    ) -> DailyPlan:
        if data.get("schema") != SchemaVersion:
            raise ValueError(f"unsupported day plan schema {data.get('schema')!r}")
        morning, night = data.get("morning"), data.get("night")
        return cls(
            id=id,
            local_date=date.fromisoformat(data["local_date"]),
            timezone=str(data["timezone"]),
            day_type=str(data["day_type"]),
            seed=int(data["seed"]),
            effective_from=_when(data["effective_from"]),
            ends_at=_when(data["ends_at"]),
            created_at=created_at,
            reason=str(data["reason"]),
            morning=SleepEpisode.from_json(morning) if morning else None,
            night=SleepEpisode.from_json(night) if night else None,
            busy=tuple(BusySpan.from_json(item) for item in data.get("busy", ())),
            meals=tuple(Meal.from_json(item) for item in data.get("meals", ())),
            quota=Quota.from_json(data["quota"]),
            greeting=GreetingWindow.from_json(data["greeting"]),
            warnings=tuple(data.get("warnings", ())),
            inputs_hash=str(data.get("inputs_hash", "")),
            superseded_by=superseded_by,
            superseded_at=superseded_at,
            lifeline_job_id=lifeline_job_id,
            lifeline_done_at=lifeline_done_at,
            summary_queued_at=summary_queued_at,
            extra=dict(data.get("extra", {})),
        )
