"""The rules a drawn day has to obey, checked and enforced by code (R-MEM-005).

The model draws the stretches of her day; the day plan (round 08) says when she sleeps, when she
is busy and when she eats.  A drawn day is consistent when

* every stretch has readable clock times and ends after it starts (a stretch does not cross
  midnight);
* no two stretches overlap;
* nothing is planned while she sleeps (a stretch whose activity is sleep itself may lie in it);
* a stretch that lies in a busy period is marked busy and a stretch marked busy lies in one;
* the day starts where the plan does (a plan made in the middle of the day has no morning).

:func:`check_rules` lists what is wrong in words the model can act on, so the second attempt can be
told.  :func:`repair` is what is used when the second attempt is wrong as well: it keeps what can be
kept - cuts the sleeping part off a stretch, moves a stretch that overlaps the previous one, drops
what cannot be saved, sets the busy marks from the plan - so the result always passes the check.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from itertools import pairwise
from zoneinfo import ZoneInfo

from twin.memory.lifeline import SLEEP_WORDS, minutes_of
from twin.memory.schemas import DrawnEvent
from twin.schedule.plan_model import DailyPlan
from twin.schedule.wallclock import day_bounds_utc, minute_of_day

MIN_EVENT_MINUTES = 10
SLEEP_TOLERANCE_MINUTES = 5
BUSY_SHARE = 0.5
MIN_EVENTS = 3
LAST_MINUTE = 24 * 60 - 1


@dataclass(frozen=True)
class Span:
    """A stretch of the local day in minutes after midnight."""

    start: float
    end: float
    label: str = ""

    def overlap(self, start: float, end: float) -> float:
        return max(0.0, min(self.end, end) - max(self.start, start))


@dataclass(frozen=True)
class DayFrame:
    """What the plan says about one local day, in the clock times the day is drawn in."""

    day: date
    zone_key: str
    not_before: float  # the plan starts here (0 for a plan of the whole day)
    wake: float | None
    bed: float | None  # 1440 when she falls asleep after midnight
    sleep: tuple[Span, ...]
    busy: tuple[Span, ...]
    meals: tuple[tuple[str, float], ...]

    @classmethod
    def from_plan(cls, plan: DailyPlan, zone: ZoneInfo) -> DayFrame:
        day_start, next_start = day_bounds_utc(plan.local_date, zone)

        def local(moment: datetime) -> float:
            if moment >= next_start:
                return 1440.0
            if moment <= day_start:
                return 0.0
            return minute_of_day(moment, zone)

        def inside(start: datetime, end: datetime) -> bool:
            return min(end, next_start) > max(start, day_start)

        sleep = tuple(
            Span(local(episode.onset), local(episode.wake))
            for episode in plan.episodes
            if inside(episode.onset, episode.wake)
        )
        busy = tuple(
            Span(local(span.start), local(span.end), span.label)
            for span in plan.busy
            if inside(span.start, span.end)
        )
        morning, night = plan.morning, plan.night
        begins = max(day_start, plan.effective_from)  # a wake-up before this is in the old plan
        wake = local(morning.wake) if morning and begins <= morning.wake < next_start else None
        bed = local(night.onset) if night is not None else None
        meals = tuple(
            (meal.kind, local(meal.at)) for meal in plan.meals if day_start <= meal.at < next_start
        )
        not_before = local(plan.effective_from) if plan.effective_from > day_start else 0.0
        return cls(plan.local_date, zone.key, not_before, wake, bed, sleep, busy, meals)

    def busy_share(self, start: float, end: float) -> float:
        """The part of ``[start, end)`` that lies in busy periods (0 to 1)."""
        length = end - start
        if length <= 0:
            return 0.0
        return min(1.0, sum(span.overlap(start, end) for span in self.busy) / length)

    def asleep_minutes(self, start: float, end: float) -> float:
        return sum(span.overlap(start, end) for span in self.sleep)

    def awake_minutes(self) -> float:
        """How much of the day the plan covers and she is awake (what there is to write about)."""
        return max(0.0, 1440.0 - self.not_before - self.asleep_minutes(self.not_before, 1440.0))

    def expected_events(self) -> int:
        """How many stretches a day like this needs: three, fewer when little of the day is left."""
        awake = self.awake_minutes()
        if awake <= 0:
            return 0
        return MIN_EVENTS if awake >= 4 * 60 else 1


@dataclass(frozen=True)
class RuleProblem:
    code: str  # bad_time | before_plan | overlap | during_sleep | busy_mismatch | too_few
    event: int | None  # 1-based number in the drawn list
    text: str  # what is wrong, worded for the model


def clock_text(minute: float) -> str:
    """``HH:MM`` of a minute of the day, never ``24:00``."""
    whole = min(LAST_MINUTE, max(0, round(minute)))
    return f"{whole // 60:02d}:{whole % 60:02d}"


def _span(event: DrawnEvent) -> tuple[int, int] | None:
    start, end = minutes_of(event.start), minutes_of(event.end)
    if start is None or end is None or end <= start:
        return None
    return start, end


def _says_sleep(event: DrawnEvent) -> bool:
    return any(word in event.activity for word in SLEEP_WORDS)


def check_rules(
    events: list[DrawnEvent], frame: DayFrame, *, minimum: int | None = None
) -> list[RuleProblem]:
    """Everything wrong with a drawn day, in the order of the events.

    ``minimum`` is the least number of stretches (default: what the frame expects of its day).
    """
    minimum = frame.expected_events() if minimum is None else minimum
    problems: list[RuleProblem] = []
    spans: list[tuple[int, int, int]] = []
    for number, event in enumerate(events, 1):
        span = _span(event)
        if span is None:
            problems.append(
                RuleProblem(
                    "bad_time",
                    number,
                    f"第 {number} 段的时间（{event.start}-{event.end}）读不出来，或结束不晚于开始；"
                    "请用同一天内的 HH:MM，结束晚于开始",
                )
            )
            continue
        start, end = span
        spans.append((start, end, number))
        if start < frame.not_before - 1:
            problems.append(
                RuleProblem(
                    "before_plan",
                    number,
                    f"第 {number} 段在 {clock_text(frame.not_before)} 之前开始，"
                    f"这份安排只从 {clock_text(frame.not_before)} 起写",
                )
            )
        asleep = frame.asleep_minutes(start, end)
        if asleep > SLEEP_TOLERANCE_MINUTES and not _says_sleep(event):
            problems.append(
                RuleProblem(
                    "during_sleep",
                    number,
                    f"第 {number} 段（{event.start}-{event.end}）落在她睡觉的时间里，"
                    "睡觉时不要安排活动",
                )
            )
        share = frame.busy_share(start, end)
        if share >= BUSY_SHARE and not event.busy:
            problems.append(
                RuleProblem(
                    "busy_mismatch",
                    number,
                    f"第 {number} 段落在忙碌时段内，要写成她在忙的事，busy 必须是 true",
                )
            )
        elif share < BUSY_SHARE and event.busy:
            problems.append(
                RuleProblem(
                    "busy_mismatch",
                    number,
                    f"第 {number} 段的 busy 是 true，但它不在忙碌时段内；请改成 false",
                )
            )
    spans.sort()
    for (_, previous_end, first), (start, _, second) in pairwise(spans):
        if start < previous_end:
            problems.append(
                RuleProblem(
                    "overlap", second, f"第 {second} 段与第 {first} 段的时间重叠；时间段不能重叠"
                )
            )
    if len(events) < minimum:
        problems.append(
            RuleProblem("too_few", None, f"至少要有 {minimum} 段活动，现在只有 {len(events)} 段")
        )
    return problems


def _awake_pieces(start: float, end: float, frame: DayFrame) -> list[tuple[float, float]]:
    """``[start, end)`` without the time she sleeps."""
    pieces = [(start, end)]
    for sleeping in sorted(frame.sleep, key=lambda s: s.start):
        cut: list[tuple[float, float]] = []
        for first, last in pieces:
            if sleeping.end <= first or sleeping.start >= last:
                cut.append((first, last))
                continue
            if sleeping.start > first:
                cut.append((first, sleeping.start))
            if sleeping.end < last:
                cut.append((sleeping.end, last))
        pieces = cut
    return pieces


def repair(
    events: list[DrawnEvent], frame: DayFrame, drop: frozenset[int] = frozenset()
) -> list[DrawnEvent]:
    """The drawn day, corrected by rule so that :func:`check_rules` finds nothing wrong.

    ``drop`` holds the numbers of the stretches the checker found to contradict what is known.
    """
    kept: list[tuple[float, float, DrawnEvent]] = []
    for number, event in enumerate(events, 1):
        span = _span(event)
        if number in drop or span is None:
            continue
        start, end = float(max(span[0], frame.not_before)), float(span[1])
        if not _says_sleep(event):
            pieces = [p for p in _awake_pieces(start, end, frame) if p[1] - p[0] >= 1]
            if not pieces:
                continue
            start, end = max(pieces, key=lambda piece: piece[1] - piece[0])
        if end - start >= MIN_EVENT_MINUTES:
            kept.append((start, end, event))
    kept.sort(key=lambda item: (item[0], item[1]))
    result: list[DrawnEvent] = []
    cursor = 0.0
    for start, end, event in kept:
        begin = max(start, cursor)
        if end - begin < MIN_EVENT_MINUTES:
            continue
        cursor = end
        result.append(
            event.model_copy(
                update={
                    "start": clock_text(begin),
                    "end": clock_text(end),
                    "busy": frame.busy_share(begin, end) >= BUSY_SHARE,
                }
            )
        )
    return result
