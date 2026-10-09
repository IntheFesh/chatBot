"""From raw counts to the activity model: smoothing, sleep and busy inference (R-ACT-002/003/004).

**Curves.**  For every day type the rate of messages and of initiations per 15-minute local
slot is the count divided by the number of days of that type on which anything was said, then
smoothed circularly (so 23:45 and 00:00 are neighbours) with a Gaussian kernel of
``activity.smoothing_sigma_slots``.

**Sleep.**  Nothing about the hours of a day is written into the algorithm.

1. The overall (all day types) smoothed curve is searched for its *valley*: the longest
   circular run of slots below ``activity.sleep_rate_ratio`` times the mean rate (the ratio is
   widened step by step if no run reaches ``activity.sleep_min_hours``; failing that, the
   window of that length with the smallest total rate).
2. The "activity day" is a 24-hour window with the valley in its *middle*, so every window
   contains one whole night.  (SPEC R-ACT-003 words this as "the lowest point is the boundary
   of the activity day"; a boundary in the middle of the valley would cut the sleep in two, the
   thing the rule exists to avoid — see DECISIONS D-152.)
3. In each activity day the longest silence between two consecutive messages of hers is the
   sleep of that day, if it is at least ``activity.sleep_min_hours`` long and covers the middle
   part of the window.  Onset and wake are the messages around the silence (a lone "good
   night" counts: it is the last thing she did before sleeping).  An odd night — she answered
   at 3 a.m. — moves that night's estimate only; the median over the nights ignores it.  The
   day belongs to the day type of the wake date.
4. Onsets and wakes are summarised with circular statistics per day type
   (:class:`~twin.profile.circular.CircularStats`).  With fewer than ``activity.min_valid_days``
   such days the result is marked low confidence and the sleep is the valley itself.
5. A deep-sleep core (``edge_minutes`` after falling asleep to ``edge_minutes`` before waking)
   that lies mostly in 10:00-18:00 triggers the warning that ``time.source_timezone`` may be
   wrong.

**Busy.**  On workdays, awake slots whose smoothed rate is below ``activity.busy_rate_ratio``
times the awake average *and* whose hour has a median reply latency at least
``activity.busy_latency_ratio`` times the day's median, on enough workdays (stability), are
merged into busy windows with their latency distributions and a confidence.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from typing import Literal

from twin.config.settings import ActivityConfig
from twin.profile.activity_collect import ALL, WORKDAY, ActivityRaw
from twin.profile.activity_model import (
    CURVE_KEYS,
    DAY_TYPE_LABELS,
    LOCAL_DAY_TYPES,
    ActivityModel,
    BusyWindow,
    SleepProfile,
    SleepWindow,
    sleep_core_in_daytime,
)
from twin.profile.circular import DAY_MINUTES, CircularStats, wrap
from twin.profile.distribution import BucketedDistribution, EmpiricalDistribution
from twin.profile.localtime import SLOT_MINUTES, SLOTS_PER_DAY, format_minute
from twin.schedule.daytype import DayTypeCalendar

MIN_MESSAGES = 50
MIN_OBSERVED_DAYS = 3
MIN_DAY_MESSAGES = 4
CENTRE_HALF_WIDTH = 360.0  # the sleep must cover the middle +-6 hours of the activity day
MIN_TYPE_OBSERVATIONS = 5
RATIO_STEPS = (1.0, 1.5, 2.0, 3.0, 4.0)
BUSY_MIN_SAMPLES = 8
BUSY_MIN_SLOTS = 4
BUSY_MAX_GAP = 2
BUSY_MIN_STABILITY = 0.5
LATENCY_MIN_SAMPLES = 20


# ------------------------------------------------------------------- curves


def gaussian_smooth(values: Sequence[float], sigma: float) -> list[float]:
    """Circular Gaussian smoothing (``sigma`` in slots; 0 leaves the values unchanged)."""
    n = len(values)
    if sigma <= 0 or n == 0:
        return list(values)
    radius = max(1, math.ceil(4 * sigma))
    kernel = [math.exp(-0.5 * (i / sigma) ** 2) for i in range(-radius, radius + 1)]
    total = sum(kernel)
    kernel = [k / total for k in kernel]
    return [
        sum(kernel[j] * values[(i + j - radius) % n] for j in range(len(kernel))) for i in range(n)
    ]


def circular_runs(mask: Sequence[bool]) -> list[tuple[int, int]]:
    """Maximal runs of ``True`` on a circle as ``(start index, length)``."""
    n = len(mask)
    if n == 0 or not any(mask):
        return []
    if all(mask):
        return [(0, n)]
    first_false = mask.index(False)
    runs: list[tuple[int, int]] = []
    start = -1
    length = 0
    for k in range(1, n + 1):
        index = (first_false + k) % n
        if mask[index]:
            if length == 0:
                start = index
            length += 1
        elif length:
            runs.append((start, length))
            length = 0
    if length:
        runs.append((start, length))
    return runs


@dataclass(frozen=True)
class Valley:
    """The longest low-activity stretch of the overall curve."""

    start: int  # slot
    length: int  # slots
    ratio: float  # threshold used, as a fraction of the mean rate

    @property
    def start_min(self) -> float:
        return float(self.start * SLOT_MINUTES)

    @property
    def end_min(self) -> float:
        return wrap((self.start + self.length) * SLOT_MINUTES)

    @property
    def middle_min(self) -> float:
        return wrap((self.start + self.length / 2) * SLOT_MINUTES)


def find_valley(curve: Sequence[float], min_slots: int, base_ratio: float) -> Valley:
    n = len(curve)
    mean = sum(curve) / n if n else 0.0
    if mean <= 0:
        return Valley(0, min_slots, 0.0)
    for step in RATIO_STEPS:
        ratio = base_ratio * step
        runs = circular_runs([v < ratio * mean for v in curve])
        if not runs:
            continue
        start, length = max(
            runs, key=lambda r: (r[1], -sum(curve[(r[0] + k) % n] for k in range(r[1])))
        )
        if length >= min_slots:
            return Valley(start, length, ratio)
    best = min(range(n), key=lambda s: sum(curve[(s + k) % n] for k in range(min_slots)))
    return Valley(best, min_slots, 0.0)


# -------------------------------------------------------------------- sleep


@dataclass(frozen=True)
class SleepObservation:
    onset: float
    wake: float
    day_type: str
    wake_day: date


def observe_sleep(
    raw: ActivityRaw, valley: Valley, min_minutes: float, calendar: DayTypeCalendar
) -> list[SleepObservation]:
    """One sleep per activity day that has a clear one (see the module docstring)."""
    cut = wrap(valley.middle_min + DAY_MINUTES / 2)
    windows: dict[int, list[tuple[float, int]]] = {}
    for wall, zone in zip(raw.her_wall, raw.her_zone, strict=True):
        index = math.floor((wall - cut) / DAY_MINUTES)
        windows.setdefault(index, []).append((wall, zone))
    found: list[SleepObservation] = []
    for index, items in sorted(windows.items()):
        if len({zone for _, zone in items}) > 1:
            continue  # the zone changed inside this window: its clock times do not compare
        items.sort()
        times = [wall for wall, _ in items]
        if len(times) < MIN_DAY_MESSAGES:
            continue
        centre = index * DAY_MINUTES + cut + DAY_MINUTES / 2
        best: tuple[float, float, float] | None = None
        for start, end in pairwise(times):
            length = end - start
            if length < min_minutes:
                continue
            if start >= centre + CENTRE_HALF_WIDTH or end <= centre - CENTRE_HALF_WIDTH:
                continue
            if best is None or length > best[0]:
                best = (length, start, end)
        if best is None:
            continue
        _, start, end = best
        wake_day = date.fromordinal(int(end // DAY_MINUTES))
        zone_name = raw.zone_names[items[0][1]]
        found.append(
            SleepObservation(
                wrap(start), wrap(end), calendar.day_type(wake_day, zone_name), wake_day
            )
        )
    return found


def _window_from(observations: Sequence[SleepObservation]) -> SleepWindow:
    return SleepWindow(
        CircularStats.from_values([o.onset for o in observations]),
        CircularStats.from_values([o.wake for o in observations]),
        len(observations),
        "days",
    )


def infer_sleep(
    raw: ActivityRaw,
    curve_all: Sequence[float],
    config: ActivityConfig,
    calendar: DayTypeCalendar,
    zone: str,
) -> tuple[SleepProfile, Valley | None]:
    """The sleep profile and the valley it was searched around."""
    if raw.her_messages < MIN_MESSAGES or raw.days.get(ALL, 0) < MIN_OBSERVED_DAYS:
        warning = (
            f"消息太少（她 {raw.her_messages} 条，{raw.days.get(ALL, 0)} 天），无法推断睡眠时段"
        )
        return SleepProfile({}, "none", 0, (warning,)), None
    min_slots = max(1, math.ceil(config.sleep_min_hours * 60 / SLOT_MINUTES))
    valley = find_valley(curve_all, min_slots, config.sleep_rate_ratio)
    observations = observe_sleep(raw, valley, config.sleep_min_hours * 60, calendar)
    warnings: list[str] = []
    windows: dict[str, SleepWindow] = {}
    confidence: Literal["high", "low"]
    if len(observations) >= config.min_valid_days:
        windows[ALL] = _window_from(observations)
        for day_type in LOCAL_DAY_TYPES:
            chosen = [o for o in observations if o.day_type == day_type]
            if len(chosen) >= MIN_TYPE_OBSERVATIONS:
                windows[day_type] = _window_from(chosen)
        confidence = "high"
    else:
        windows[ALL] = SleepWindow(
            CircularStats.fixed(valley.start_min), CircularStats.fixed(valley.end_min), 0, "curve"
        )
        confidence = "low"
        warnings.append(
            f"可用于推断的夜晚只有 {len(observations)} 个（少于 {config.min_valid_days}）：低置信，"
            "睡眠时段取整体活跃曲线中最长的低谷"
        )
    for key, window in windows.items():
        if sleep_core_in_daytime(window, config.edge_minutes):
            start, end = window.core(config.edge_minutes)
            label = DAY_TYPE_LABELS[key]
            warnings.append(
                f"推断的睡眠核心（{label}）{format_minute(start)}–{format_minute(end)} 落在当地"
                f"白天 10:00–18:00，可能是 time.source_timezone 设置不对（当前学习用 {zone}）"
            )
    profile = SleepProfile(windows, confidence, len(observations), tuple(warnings))
    return profile, valley


# --------------------------------------------------------------------- busy


def _merge_counters(
    per_slot: Mapping[int, Counter[float]], slots: Sequence[int]
) -> dict[float, int]:
    merged: dict[float, int] = {}
    for slot in slots:
        for value, count in per_slot.get(slot, {}).items():
            merged[value] = merged.get(value, 0) + count
    return merged


def infer_busy(
    raw: ActivityRaw,
    rate_workday: Sequence[float],
    sleep: SleepProfile,
    config: ActivityConfig,
) -> tuple[BusyWindow, ...]:
    """Busy windows of workdays (see the module docstring)."""
    workdays = raw.days.get(WORKDAY, 0)
    window = sleep.window_for(WORKDAY)
    if workdays < MIN_OBSERVED_DAYS:
        return ()
    asleep = [False] * SLOTS_PER_DAY
    if window is not None:
        onset, wake = window.onset_min, window.wake_min
        for slot in range(SLOTS_PER_DAY):
            minute = slot * SLOT_MINUTES + SLOT_MINUTES / 2
            asleep[slot] = (minute - onset) % DAY_MINUTES < (wake - onset) % DAY_MINUTES
    awake = [s for s in range(SLOTS_PER_DAY) if not asleep[s]]
    if not awake:
        return ()
    base = sum(rate_workday[s] for s in awake) / len(awake)
    overall = EmpiricalDistribution.from_counter(
        _merge_counters(raw.latency_workday, range(SLOTS_PER_DAY)), discrete=True
    )
    if base <= 0 or overall.n < BUSY_MIN_SAMPLES:
        return ()
    overall_median = max(overall.median(), 1.0)
    slots_per_hour = 60 // SLOT_MINUTES
    slow_hour: dict[int, float] = {}
    for hour in range(24):
        counter = _merge_counters(
            raw.latency_workday, range(hour * slots_per_hour, (hour + 1) * slots_per_hour)
        )
        dist = EmpiricalDistribution.from_counter(counter, discrete=True)
        if dist.n >= BUSY_MIN_SAMPLES:
            slow_hour[hour] = dist.median() / overall_median
    busy = [
        (not asleep[s])
        and rate_workday[s] < config.busy_rate_ratio * base
        and slow_hour.get(s // slots_per_hour, 0.0) >= config.busy_latency_ratio
        for s in range(SLOTS_PER_DAY)
    ]
    # close gaps of up to BUSY_MAX_GAP awake slots between busy ones
    for _ in range(BUSY_MAX_GAP):
        busy = [
            b or (not asleep[s] and 0 < s < SLOTS_PER_DAY - 1 and busy[s - 1] and busy[s + 1])
            for s, b in enumerate(busy)
        ]
    found: list[BusyWindow] = []
    start = None
    for slot in range(SLOTS_PER_DAY + 1):
        active = slot < SLOTS_PER_DAY and busy[slot]
        if active and start is None:
            start = slot
        elif not active and start is not None:
            if slot - start >= BUSY_MIN_SLOTS:
                candidate = _busy_window(
                    raw, rate_workday, start, slot, base, overall_median, config
                )
                if candidate is not None:
                    found.append(candidate)
            start = None
    return tuple(found)


def _busy_window(
    raw: ActivityRaw,
    rate_workday: Sequence[float],
    first: int,
    stop: int,
    base: float,
    overall_median: float,
    config: ActivityConfig,
) -> BusyWindow | None:
    slots = list(range(first, stop))
    latency = EmpiricalDistribution.from_counter(
        _merge_counters(raw.latency_workday, slots), discrete=True
    )
    if latency.n < BUSY_MIN_SAMPLES:
        return None
    allowed = base * len(slots) * config.busy_rate_ratio
    quiet_days = sum(
        1 for counts in raw.workday_day_counts.values() if sum(counts[s] for s in slots) <= allowed
    )
    stability = quiet_days / max(1, raw.days.get(WORKDAY, 0))
    if stability < BUSY_MIN_STABILITY:
        return None
    ratio = latency.median() / overall_median
    strength = min(1.0, ratio / (2 * config.busy_latency_ratio))
    return BusyWindow(
        float(first * SLOT_MINUTES),
        float(stop * SLOT_MINUTES),
        min(1.0, stability * strength),
        ratio,
        stability,
        latency,
    )


# -------------------------------------------------------------------- model


def build_activity_model(
    raw: ActivityRaw,
    config: ActivityConfig,
    calendar: DayTypeCalendar,
    scope: str,
) -> ActivityModel:
    """The activity model of one scope from its raw counts."""
    sigma = config.smoothing_sigma_slots
    rate_raw: dict[str, tuple[float, ...]] = {}
    init_raw: dict[str, tuple[float, ...]] = {}
    for key in CURVE_KEYS:
        days = max(1, raw.days.get(key, 0))
        rate_raw[key] = tuple(c / days for c in raw.counts[key])
        init_raw[key] = tuple(c / days for c in raw.inits[key])
    rate = {key: tuple(gaussian_smooth(values, sigma)) for key, values in rate_raw.items()}
    initiation = {key: tuple(gaussian_smooth(values, sigma)) for key, values in init_raw.items()}
    zone = raw.primary_zone()
    sleep, valley = infer_sleep(raw, rate[ALL], config, calendar, zone)
    busy: dict[str, tuple[BusyWindow, ...]] = {}
    found = infer_busy(raw, rate[WORKDAY], sleep, config)
    if found:
        busy[WORKDAY] = found
    latency = BucketedDistribution.from_counters(
        raw.latency, sizes=(1, 4), min_samples=LATENCY_MIN_SAMPLES, discrete=True
    )
    latency_workday = BucketedDistribution.from_counters(
        raw.latency_workday, sizes=(1, 4), min_samples=LATENCY_MIN_SAMPLES, discrete=True
    )
    days_total = max(1, raw.days.get(ALL, 0))
    extra: dict[str, object] = {
        "first_day": raw.first_day.isoformat() if raw.first_day else None,
        "last_day": raw.last_day.isoformat() if raw.last_day else None,
        "zones": sorted(set(raw.zone_names)),
    }
    if valley is not None:
        extra["valley"] = {
            "start": format_minute(valley.start_min),
            "end": format_minute(valley.end_min),
            "ratio": valley.ratio,
        }
    return ActivityModel(
        scope=scope,
        zone=zone,
        edge_minutes=config.edge_minutes,
        days={key: raw.days.get(key, 0) for key in CURVE_KEYS},
        rate=rate,
        rate_raw=rate_raw,
        initiation=initiation,
        initiation_raw=init_raw,
        latency=latency,
        latency_workday=latency_workday,
        sleep=sleep,
        busy=busy,
        her_messages=raw.her_messages,
        initiations_per_day=raw.initiations / days_total,
        extra=extra,
    )
