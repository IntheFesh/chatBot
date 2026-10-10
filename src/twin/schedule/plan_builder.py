"""Making the plan of a local day: seeded, reproducible, from her routine (R-SCH-004, R-PRO-002).

:func:`build_plan` is a pure function.  Given the routine model (round 04, manual corrections
already applied), the day type, the proactive range and the seed it returns the same plan every
time - a restart, a wake-up or a debugging session sees the plan that was made, and the seed
(a hash of the local date and the installation's random salt) is in the plan to prove it.  Each
part of the plan has its own random stream (``Random(f"{seed}:{part}")``), so changing the proactive
range does not move the night's sleep.

What it decides, in this order:

**Sleep** - the night that ends on the day (the morning) and the one that begins on it (the
night).  A night is drawn from the sleep window of the type of the day she wakes on: her fall-asleep
and wake-up clock times each get an offset from the median, drawn from the offsets observed in her
history (:meth:`CircularStats.offsets`); the pair is accepted only if the length of the sleep lies
between the 5th and the 95th percentile of the sleep lengths those offsets give.  A manual
correction is a fixed window, so it wins by construction.  The morning is not drawn again when the
plan of the previous day (same time zone) already holds that night: the two plans agree.  The
night also starts at least ``schedule.min_awake_h`` after the wake-up.

**Busy periods** - the busy windows of the day type, or the manual ones of the weekday, each with
the reference of its own reply-latency distribution.

**Meals** - the activity peak of her curve inside the usual meal hours; where the curve shows no
peak, the common local meal time, marked as such.

**Quota** - the number of conversations she opens on her own that day (R-PRO-002): a Poisson
count whose rate is solved so that, *after* clipping to ``[daily_min, daily_max]``, the mean is her
real "conversations opened per day" from the profile; 0 when proactive messages are off.

**Greeting** - the window ``wake + 5..40 min``, unless the last greeting was less than 18 hours
ago (R-SCH-002) or the window is already over.

Daylight saving: every clock time goes through :mod:`twin.schedule.wallclock`; where its rules
moved a time, the plan says so in ``extra["dst"]``.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import orjson

from twin.config.settings import ScheduleConfig
from twin.profile.activity_model import ActivityModel, SleepWindow
from twin.profile.circular import CircularStats
from twin.profile.localtime import SLOT_MINUTES, SLOTS_PER_DAY, format_minute
from twin.schedule.plan_model import (
    BusySpan,
    DailyPlan,
    GreetingWindow,
    Meal,
    MealKind,
    Quota,
    SleepEpisode,
    build_segments,
    busy_ref,
)
from twin.schedule.wallclock import WallKind, day_bounds_utc, resolve

DURATION_SAMPLES = 600
ACCEPT_ATTEMPTS = 60
MEAL_SEARCH: dict[MealKind, tuple[int, int]] = {
    "breakfast": (6 * 60 + 30, 9 * 60 + 30),
    "lunch": (11 * 60, 14 * 60),
    "dinner": (17 * 60, 20 * 60 + 30),
}
# the common local meal times, used where her curve shows no peak (marked source="default")
MEAL_DEFAULT: dict[MealKind, int] = {
    "breakfast": 7 * 60 + 30,
    "lunch": 12 * 60,
    "dinner": 18 * 60 + 30,
}
MEAL_MINUTES: dict[MealKind, int] = {"breakfast": 20, "lunch": 40, "dinner": 50}
MEAL_PEAK_RATIO = 1.25  # a meal peak must stand this far above the mean of its search window
QUOTA_ITERATIONS = 80


@dataclass(frozen=True)
class QuotaRange:
    """The proactive range of the moment: ``/主动 2-5`` and ``/主动 关`` (R-PRO-002)."""

    minimum: int
    maximum: int
    enabled: bool = True


@dataclass(frozen=True)
class PlanRequest:
    """Everything a plan is drawn from."""

    day: date
    zone: ZoneInfo
    day_type: str
    next_day_type: str
    seed: int
    model: ActivityModel | None
    effective_from: datetime  # not before the first instant of ``day``
    go_to_bed_at: datetime | None  # a time zone switch: inside a night she goes to bed now
    quota: QuotaRange
    last_greeting_at: datetime | None
    previous_day: (
        DailyPlan | None
    )  # the plan of the day before, same zone: its night continues here
    same_day: DailyPlan | None  # the plan this one replaces (same day, same zone)
    reason: str
    config: ScheduleConfig
    inputs_hash: str  # everything that decides the plan, for "is the plan stale?"
    routine_hash: str  # the routine alone: nights are only taken over between equal routines
    created_at: datetime


# ---------------------------------------------------------------------- seeds


def plan_seed(day: date, salt: str) -> int:
    """The seed of a day: a hash of the local date and the installation's random salt (62 bits)."""
    digest = hashlib.sha256(f"{day.isoformat()}|{salt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 2


def stream(seed: int, part: str) -> random.Random:
    """The random stream of one part of a plan."""
    return random.Random(f"{seed}:{part}")  # noqa: S311 - a plan is drawn, not kept secret


def fingerprint(*parts: object) -> str:
    """A short hash of the inputs that decide a plan (JSON-able values only)."""
    data = orjson.dumps(list(parts), option=orjson.OPT_SORT_KEYS | orjson.OPT_NON_STR_KEYS)
    return hashlib.sha1(data, usedforsecurity=False).hexdigest()


def model_fingerprint(model: ActivityModel | None) -> str:
    """What of the routine model reaches a plan: sleep, busy windows (also manual), curves."""
    if model is None:
        return "no-model"
    return fingerprint(model.to_json(), [w.to_json() for w in model.manual_busy])


# ---------------------------------------------------------------------- sleep


def _offset(stats: CircularStats, rng: random.Random) -> float:
    """A signed offset in minutes from the median, as observed (0 for a fixed time)."""
    return 0.0 if stats.offsets.is_empty else float(stats.offsets.sample(rng))


def duration_bounds(window: SleepWindow) -> tuple[float, float]:
    """The 5th and 95th percentile of the length of her sleep, in minutes.

    The history keeps the fall-asleep and the wake-up offsets separately, so the lengths are the
    median length plus the difference of one offset of each, drawn with a fixed generator (the
    bounds of a window never change between calls).  A fixed window has exactly one length.
    """
    median = window.duration_min()
    if window.onset.offsets.is_empty and window.wake.offsets.is_empty:
        return median, median
    rng = random.Random("sleep-duration-bounds")  # noqa: S311 - a fixed stream, not a secret
    lengths = sorted(
        median + _offset(window.wake, rng) - _offset(window.onset, rng)
        for _ in range(DURATION_SAMPLES)
    )
    return lengths[int(0.05 * (len(lengths) - 1))], lengths[int(0.95 * (len(lengths) - 1))]


class _Notes:
    """Collects where a daylight saving rule moved a clock time."""

    def __init__(self, zone: ZoneInfo) -> None:
        self.zone = zone
        self.items: list[dict[str, str]] = []

    def peek(self, day: date, minute: float) -> datetime:
        """The instant of a clock time, without noting how a daylight saving rule read it."""
        return resolve(day, minute, self.zone).utc

    def instant(self, what: str, day: date, minute: float) -> datetime:
        found = resolve(day, minute, self.zone)
        if found.kind is not WallKind.NORMAL:
            self.items.append(
                {
                    "what": what,
                    "rule": found.kind.value,
                    "asked": f"{found.day.isoformat()} {format_minute(found.minute)}",
                    "read_as": found.reading(self.zone).strftime("%Y-%m-%d %H:%M"),
                }
            )
        return found.utc


def _sample_episode(
    window: SleepWindow,
    wake_day: date,
    rng: random.Random,
    notes: _Notes,
    *,
    edge_minutes: int,
    onset_not_before: datetime | None,
) -> SleepEpisode:
    """Draw the night that ends on ``wake_day`` (see the module description)."""
    median_wake = window.wake.median
    median_length = window.duration_min()
    low, high = duration_bounds(window)
    chosen: tuple[float, float] | None = None
    last: tuple[float, float] = (0.0, 0.0)
    for _ in range(ACCEPT_ATTEMPTS):
        onset_offset, wake_offset = _offset(window.onset, rng), _offset(window.wake, rng)
        length = median_length + wake_offset - onset_offset
        last = (onset_offset, wake_offset)
        if not low <= length <= high:
            continue
        onset = notes.peek(wake_day, median_wake - median_length + onset_offset)
        if onset_not_before is not None and onset < onset_not_before:
            continue
        chosen = last
        break
    if chosen is None:
        # nothing fitted: keep the drawn wake-up, make the night as long as the bounds allow
        # and, if it still begins too early, start it as early as the rest of the day permits
        wake_offset = last[1]
        length = min(max(median_length + wake_offset - last[0], low), high)
        onset = notes.instant("onset", wake_day, median_wake + wake_offset - length)
        wake = notes.instant("wake", wake_day, median_wake + wake_offset)
        if onset_not_before is not None and onset < onset_not_before:
            onset = onset_not_before
            if wake <= onset:
                wake = onset + timedelta(minutes=length)
        return SleepEpisode(onset, wake, edge_minutes, window.source)
    onset_offset, wake_offset = chosen
    onset = notes.instant("onset", wake_day, median_wake - median_length + onset_offset)
    wake = notes.instant("wake", wake_day, median_wake + wake_offset)
    return SleepEpisode(onset, wake, edge_minutes, window.source)


def _clip(episode: SleepEpisode, moment: datetime | None) -> SleepEpisode:
    """She goes to bed now when the zone changed inside a night that should already have begun."""
    if moment is not None and episode.onset < moment < episode.wake:
        return replace(episode, onset=moment, clipped=True)
    return episode


def _adopted(episode: SleepEpisode, plan: DailyPlan) -> SleepEpisode:
    return replace(episode, adopted_from=plan.id or episode.adopted_from)


def _morning(
    request: PlanRequest, window: SleepWindow | None, notes: _Notes, edge: int
) -> SleepEpisode | None:
    zone = request.zone
    same = request.same_day
    if (
        same is not None
        and same.morning is not None
        and same.morning.wake <= request.effective_from
    ):
        return same.morning  # the morning is over: it is history
    previous = request.previous_day
    if (
        previous is not None
        and previous.night is not None
        and previous.timezone == zone.key
        and previous.extra.get("routine_hash") == request.routine_hash
        and previous.night.wake.astimezone(zone).date() == request.day
    ):
        return _adopted(previous.night, previous)
    if window is None:
        return None
    fresh = _sample_episode(
        window,
        request.day,
        stream(request.seed, "sleep:morning"),
        notes,
        edge_minutes=edge,
        onset_not_before=None,
    )
    return _clip(fresh, request.go_to_bed_at)


def _night(
    request: PlanRequest,
    window: SleepWindow | None,
    morning: SleepEpisode | None,
    notes: _Notes,
    edge: int,
) -> SleepEpisode | None:
    old = request.same_day.night if request.same_day is not None else None
    if old is not None and old.onset <= request.effective_from:
        return old  # she is already asleep (or was): a rebuild does not wake her up
    if window is None:
        return None
    not_before = (
        morning.wake + timedelta(hours=request.config.min_awake_h) if morning is not None else None
    )
    fresh = _sample_episode(
        window,
        request.day + timedelta(days=1),
        stream(request.seed, "sleep:night"),
        notes,
        edge_minutes=edge,
        onset_not_before=not_before,
    )
    return _clip(fresh, request.go_to_bed_at)


# ----------------------------------------------------------------------- busy


def _busy_spans(request: PlanRequest, notes: _Notes) -> tuple[BusySpan, ...]:
    model = request.model
    if model is None:
        return ()
    weekday = request.day.weekday()
    spans: list[BusySpan] = []
    for index, window in enumerate(model.busy_windows(request.day_type, weekday)):
        start = notes.instant("busy_start", request.day, window.start_min)
        end = notes.instant("busy_end", request.day, window.end_min)
        if end <= start:
            continue
        latency = window.latency
        spans.append(
            BusySpan(
                start,
                end,
                window.label(),
                window.source,
                busy_ref(request.day_type, weekday, index),
                None if latency.is_empty else round(latency.median(), 1),
                None if latency.is_empty else round(latency.quantile(0.9), 1),
            )
        )
    return tuple(spans)


# ---------------------------------------------------------------------- meals


def _meals(
    request: PlanRequest, notes: _Notes, sleeping: tuple[SleepEpisode, ...]
) -> tuple[Meal, ...]:
    model = request.model
    rng = stream(request.seed, "meals")
    jitter = request.config.meal_jitter_min
    meals: list[Meal] = []
    for kind, (first, last) in MEAL_SEARCH.items():
        source: str = "default"
        centre = float(MEAL_DEFAULT[kind])
        if model is not None:
            slots = range(first // SLOT_MINUTES, min(SLOTS_PER_DAY, last // SLOT_MINUTES + 1))
            rates = [model.rate_at(slot, request.day_type) for slot in slots]
            mean = sum(rates) / len(rates) if rates else 0.0
            if rates and mean > 0 and max(rates) >= MEAL_PEAK_RATIO * mean:
                best = list(slots)[rates.index(max(rates))]
                centre = best * SLOT_MINUTES + SLOT_MINUTES / 2
                source = "history"
        minute = centre + (rng.gauss(0.0, jitter) if jitter > 0 else 0.0)
        minute = min(max(minute, float(first)), float(last))
        at = notes.instant(f"meal:{kind}", request.day, round(minute))
        if any(episode.contains(at) for episode in sleeping):
            continue
        if not request.effective_from <= at:
            continue
        meals.append(
            Meal(kind, at, MEAL_MINUTES[kind], "history" if source == "history" else "default")
        )
    return tuple(meals)


# ----------------------------------------------------------------------- quota


def _clipped_poisson_mean(rate: float, low: int, high: int) -> float:
    """``E[min(max(X, low), high)]`` for ``X ~ Poisson(rate)``."""
    if low >= high:
        return float(low)
    probability = math.exp(-rate)
    below_or_at_low = 0.0
    middle = 0.0
    cumulative = 0.0
    for k in range(high):
        if k > 0:
            probability *= rate / k
        cumulative += probability
        if k <= low:
            below_or_at_low += probability
        else:
            middle += k * probability
    return low * below_or_at_low + middle + high * max(0.0, 1.0 - cumulative)


def solve_rate(target: float, low: int, high: int) -> float:
    """The Poisson rate whose clipped mean is ``target`` (the clipped mean grows with the rate)."""
    target = min(max(target, float(low)), float(high))
    if low >= high or target <= low:
        return 0.0
    left, right = 0.0, float(high) * 4 + 10
    if _clipped_poisson_mean(right, low, high) <= target:
        return right
    for _ in range(QUOTA_ITERATIONS):
        middle = (left + right) / 2
        if _clipped_poisson_mean(middle, low, high) < target:
            left = middle
        else:
            right = middle
    return (left + right) / 2


def draw_count(rng: random.Random, rate: float, low: int, high: int) -> int:
    """One Poisson draw by inversion, clipped to ``[low, high]``."""
    draw = rng.random()
    probability = math.exp(-rate)
    cumulative = probability
    k = 0
    while cumulative < draw and k < high:
        k += 1
        probability *= rate / k
        cumulative += probability
    return min(max(k, low), high)


def _quota(request: PlanRequest, scale: float) -> Quota:
    low = max(0, request.quota.minimum)
    high = max(low, request.quota.maximum)
    if not request.quota.enabled or high == 0:
        return Quota(0, 0, low, high, None, "disabled", False)
    model = request.model
    if model is not None and model.initiations_per_day > 0:
        target, source = model.initiations_per_day, "profile"
    else:
        target, source = (low + high) / 2, "range_midpoint"
    target = min(max(target, float(low)), float(high))
    rate = solve_rate(target, low, high)
    total = draw_count(stream(request.seed, "quota"), rate, low, high)
    share = total if scale >= 1.0 else round(total * max(scale, 0.0))
    return Quota(
        total,
        share,
        low,
        high,
        round(target, 3),
        "profile" if source == "profile" else "range_midpoint",
        True,
    )


# --------------------------------------------------------------------- greeting


def _greeting(request: PlanRequest, morning: SleepEpisode | None) -> GreetingWindow:
    if morning is None:
        return GreetingWindow(False, "no_wake_up_planned")
    first, last = request.config.greeting_window_min
    earliest = morning.wake + timedelta(minutes=first)
    latest = morning.wake + timedelta(minutes=last)
    if latest <= request.created_at:
        return GreetingWindow(False, "wake_up_already_passed", earliest, latest)
    sent = request.last_greeting_at
    if sent is not None and sent >= morning.wake:
        return GreetingWindow(False, "already_sent_after_this_wake_up", earliest, latest)
    gap = timedelta(hours=request.config.greeting_min_gap_h)
    if sent is not None and earliest - sent < gap:
        return GreetingWindow(False, "within_min_gap_of_the_last_greeting", earliest, latest)
    return GreetingWindow(True, "ok", max(earliest, request.created_at), latest)


# ------------------------------------------------------------------------ plan


def build_plan(request: PlanRequest, *, plan_id: str = "") -> DailyPlan:
    """Draw the plan of ``request.day`` (see the module description)."""
    zone = request.zone
    notes = _Notes(zone)
    model = request.model
    warnings: list[str] = []
    morning_window: SleepWindow | None = None
    night_window: SleepWindow | None = None
    edge = 30
    if model is None:
        warnings.append("no_routine_model")
    else:
        edge = model.edge_minutes
        morning_window = model.sleep.window_for(request.day_type)
        night_window = model.sleep.window_for(request.next_day_type)
        if morning_window is None or night_window is None:
            warnings.append("no_sleep_window")
        if model.sleep.confidence != "high":
            warnings.append(f"sleep_confidence_{model.sleep.confidence}")
    morning = _morning(request, morning_window, notes, edge)
    night = _night(request, night_window, morning, notes, edge)
    day_start, next_start = day_bounds_utc(request.day, zone)
    start = max(request.effective_from, day_start)
    ends_at = night.wake if night is not None else next_start
    if ends_at <= start:
        ends_at = next_start if next_start > start else start + timedelta(minutes=1)
    busy_all = _busy_spans(request, notes)
    busy = tuple(
        replace(span, start=max(span.start, start), end=min(span.end, ends_at))
        for span in busy_all
        if min(span.end, ends_at) > max(span.start, start)
    )
    episodes = tuple(e for e in (morning, night) if e is not None)
    meals = _meals(request, notes, episodes)
    whole_day = build_segments(day_start, next_start, episodes, busy_all)
    awake_total = sum(
        (min(s.end, next_start) - max(s.start, day_start)).total_seconds()
        for s in whole_day
        if s.kind in ("busy", "free")
    )
    awake_left = sum(
        (min(s.end, next_start) - max(s.start, start)).total_seconds()
        for s in whole_day
        if s.kind in ("busy", "free") and s.end > start
    )
    scale = 1.0 if start <= day_start else (awake_left / awake_total if awake_total > 0 else 0.0)
    return DailyPlan(
        id=plan_id,
        local_date=request.day,
        timezone=zone.key,
        day_type=request.day_type,
        seed=request.seed,
        effective_from=start,
        ends_at=ends_at,
        created_at=request.created_at,
        reason=request.reason,
        morning=morning,
        night=night,
        busy=busy,
        meals=meals,
        quota=_quota(request, scale),
        greeting=_greeting(request, morning),
        warnings=tuple(warnings),
        inputs_hash=request.inputs_hash,
        extra={
            "dst": notes.items,
            "next_day_type": request.next_day_type,
            "routine_hash": request.routine_hash,
        },
    )
