"""Synthetic routine models and a schedule rig for the round-08 tests.

Nothing here comes from real chat data: a routine is a few clock times (23:30 to 07:30, busy from
13:00 to 17:00) with a small spread around them, built straight into an
:class:`~twin.profile.activity_model.ActivityModel`.  The :class:`Rig` wires the real schedule kit
(time service, planner, events) to a services container whose model and proactive range the test
can change at will.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from tests.support.clock import ManualClock
from twin.config.runtime import BOT_TIMEZONE
from twin.profile.activity_model import (
    CURVE_KEYS,
    ActivityModel,
    BusyWindow,
    SleepProfile,
    SleepWindow,
)
from twin.profile.circular import CircularStats
from twin.profile.distribution import BucketedDistribution, EmpiricalDistribution
from twin.schedule.plan_builder import QuotaRange
from twin.schedule.planner import DailyPlanner
from twin.schedule.service import ScheduleKit, build_kit
from twin.services import Services

SPREAD = {-30.0: 1, -15.0: 3, 0.0: 8, 15.0: 3, 30.0: 1}  # minutes around the median


def stats(hour: float, spread: Mapping[float, int] | None = None) -> CircularStats:
    """A clock time (hours) with the observed offsets ``spread`` (none: a fixed time)."""
    minute = (hour * 60) % 1440
    if not spread:
        return CircularStats.fixed(minute)
    return CircularStats(
        minute,
        minute,
        15.0,
        sum(spread.values()),
        EmpiricalDistribution.from_counter(dict(spread), discrete=True),
    )


def sleep_window(
    onset: float,
    wake: float,
    *,
    spread: Mapping[float, int] | None = None,
    source: str = "days",
) -> SleepWindow:
    """A night from ``onset`` to ``wake`` (hours); ``spread`` makes the nights vary."""
    return SleepWindow(stats(onset, spread), stats(wake, spread), 30, source)  # type: ignore[arg-type]


def busy_window(
    start: float, end: float, *, weekdays: tuple[int, ...] = (), latency_s: float = 1500.0
) -> BusyWindow:
    return BusyWindow(
        start * 60,
        end * 60,
        0.9,
        4.0,
        0.9,
        EmpiricalDistribution.from_counter({latency_s: 20}, discrete=True),
        "override" if weekdays else "inferred",
        weekdays,
    )


def rate_curve(peaks: Mapping[int, float] | None = None, base: float = 0.2) -> tuple[float, ...]:
    """A 96-slot activity curve with peaks at the given slots."""
    values = [base] * 96
    for slot, value in (peaks or {}).items():
        values[slot] = value
    return tuple(values)


def make_model(
    windows: Mapping[str, SleepWindow],
    busy: Mapping[str, tuple[BusyWindow, ...]] | None = None,
    *,
    edge: int = 30,
    initiations: float = 3.6,
    rate: tuple[float, ...] | None = None,
    manual_busy: tuple[BusyWindow, ...] = (),
    confidence: str = "high",
    days: Mapping[str, int] | None = None,
) -> ActivityModel:
    curve = rate or rate_curve({49: 2.0, 75: 2.0})  # 12:15 and 18:45
    curves = dict.fromkeys(CURVE_KEYS, curve)
    latency = BucketedDistribution.from_counters({50: {20.0: 40}}, sizes=(1, 4), min_samples=20)
    return ActivityModel(
        scope="live",
        zone="America/Chicago",
        edge_minutes=edge,
        days=dict(days or {"all": 60, "workday": 40, "weekend": 18, "holiday": 2}),
        rate=curves,
        rate_raw=curves,
        initiation=curves,
        initiation_raw=curves,
        latency=latency,
        latency_workday=latency,
        sleep=SleepProfile(dict(windows), confidence, 30, ()),  # type: ignore[arg-type]
        busy=dict(busy or {}),
        her_messages=1000,
        initiations_per_day=initiations,
        manual_busy=manual_busy,
    )


def student_model(**kwargs: object) -> ActivityModel:
    """Sleeps 23:30-07:30 on workdays and 00:30-09:00 on weekends and holidays; busy 13-17."""
    windows = {
        "all": sleep_window(23.5, 7.5, spread=SPREAD),
        "workday": sleep_window(23.5, 7.5, spread=SPREAD),
        "weekend": sleep_window(0.5, 9.0, spread=SPREAD),
        "holiday": sleep_window(0.5, 9.0, spread=SPREAD),
    }
    busy = {"workday": (busy_window(13, 17),)}
    return make_model(windows, busy, **kwargs)  # type: ignore[arg-type]


def fixed_model(**kwargs: object) -> ActivityModel:
    """The same night every day (no spread): 23:30 to 07:30, busy 13:00 to 17:00 on workdays."""
    windows = {"all": sleep_window(23.5, 7.5)}
    return make_model(windows, {"workday": (busy_window(13, 17),)}, **kwargs)  # type: ignore[arg-type]


@dataclass
class Rig:
    """The real schedule kit over a services container, with a model the test can swap."""

    services: Services
    clock: ManualClock
    kit: ScheduleKit
    holder: dict[str, ActivityModel | None]
    quota: dict[str, QuotaRange]

    @classmethod
    def build(
        cls,
        services: Services,
        clock: ManualClock,
        model: ActivityModel | None = None,
        *,
        zone: str | None = None,
        quota: QuotaRange | None = None,
    ) -> Rig:
        holder: dict[str, ActivityModel | None] = {"model": model}
        range_holder = {"quota": quota or QuotaRange(1, 6)}
        if zone is not None:
            services.runtime.set(BOT_TIMEZONE, zone, by="test")
        kit = build_kit(
            services,
            model_source=lambda: holder["model"],
            quota_source=lambda: range_holder["quota"],
        )
        return cls(services, clock, kit, holder, range_holder)

    @property
    def planner(self) -> DailyPlanner:
        return self.kit.planner

    def use(self, model: ActivityModel | None) -> None:
        self.holder["model"] = model

    def set_quota(self, quota: QuotaRange) -> None:
        self.quota["quota"] = quota

    def zone(self) -> ZoneInfo:
        return self.kit.time.bot_timezone()

    def at(self, year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
        """The instant of a clock time in the bot's current zone."""
        return self.kit.time.local_to_utc(date(year, month, day), hour * 60 + minute)

    def move_to(self, moment: datetime) -> None:
        self.clock.set_time(moment)

    def advance(self, **kwargs: float) -> None:
        self.clock.tick(timedelta(**kwargs).total_seconds())
