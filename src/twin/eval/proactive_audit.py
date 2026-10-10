"""The audit of the proactive messages (R-EVAL-005): what the log says about the last days.

:func:`audit_days` reads ``proactive_log`` and ``ratings`` for a run of **completed** local days
(the days before today, in the bot's zone) and checks, for every day:

* **deep sleep** - no message went out while she was in deep sleep (the core of the night): zero;
* **the range** - the number of messages that went out is inside the range the day was planned
  with (``[daily_min, daily_max]``; ``[0, 0]`` when proactive messages were off).  A day *below* its
  minimum is excused when the log shows why the platform or the user stopped it: the window was
  over or the count used up, nobody was logged in, a pause, the switch, the budget (the
  messages "suppressed by the window" of R-PRO-003); a day *above* its maximum never is;
* **spacing and chase** - two messages are at least ``min_spacing_min`` apart, and a message
  that nobody answered is followed by at most ``max_chase`` more;
* **the edge of sleep** - at most ``edge_of_sleep_weekly_max`` messages sent at the edge of her
  sleep in any seven days;
* **observed** - the scheduler opened the day (it was running): a day without a mark was not
  watched, so it counts for nothing and ends the streak of observed days.

Besides the verdict it keeps the distribution of the local hours the messages went out, the number
of messages the window suppressed, the reasons of the other refusals and the ratings of the period.
The M3 gate (:mod:`twin.eval.proactive_gate`) is made of exactly these numbers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from twin.clock import Clock
from twin.config.settings import ProactiveConfig
from twin.schedule.proactive.store import (
    LogEntry,
    ProactiveLogStore,
    RatingRow,
    RatingStore,
    average,
)
from twin.schedule.proactive.types import OUTSIDE_REASONS, WINDOW_REASONS
from twin.schedule.time_service import TimeService

WEEK = timedelta(days=7)
OUTSIDE = frozenset(reason.value for reason in OUTSIDE_REASONS)
WINDOW = frozenset(reason.value for reason in WINDOW_REASONS)


@dataclass(frozen=True)
class DayAudit:
    """One completed local day."""

    day: date
    observed: bool
    low: int | None
    high: int | None
    enabled: bool | None
    sent: int
    deep_sleep: int
    edge: int
    spacing_violations: int
    chase_violations: int
    suppressed_window: int
    refused: dict[str, int]
    excused: bool
    hours: dict[int, int]

    @property
    def count_ok(self) -> bool:
        if self.high is None or self.low is None:
            return False
        low, high = (self.low, self.high) if self.enabled is not False else (0, 0)
        if self.sent > high:
            return False
        return self.sent >= low or self.excused

    @property
    def problems(self) -> list[str]:
        found: list[str] = []
        if not self.observed:
            return ["程序没有运行（当天没有记录）"]
        if self.deep_sleep:
            found.append(f"深睡时段发了 {self.deep_sleep} 条")
        if not self.count_ok:
            low = self.low if self.enabled is not False else 0
            high = self.high if self.enabled is not False else 0
            found.append(f"发了 {self.sent} 条，不在 {low}-{high} 范围内")
        if self.spacing_violations:
            found.append(f"{self.spacing_violations} 次间隔不足")
        if self.chase_violations:
            found.append(f"{self.chase_violations} 次追发超限")
        return found

    @property
    def compliant(self) -> bool:
        return self.observed and not self.problems

    def to_json(self) -> dict[str, Any]:
        return {
            "day": self.day.isoformat(),
            "observed": self.observed,
            "low": self.low,
            "high": self.high,
            "enabled": self.enabled,
            "sent": self.sent,
            "deep_sleep": self.deep_sleep,
            "edge": self.edge,
            "spacing_violations": self.spacing_violations,
            "chase_violations": self.chase_violations,
            "suppressed_window": self.suppressed_window,
            "refused": dict(self.refused),
            "excused": self.excused,
            "compliant": self.compliant,
            "hours": {str(hour): count for hour, count in sorted(self.hours.items())},
        }


@dataclass(frozen=True)
class Audit:
    """The audit of a run of completed local days (see the module description)."""

    first_day: date
    last_day: date
    days: tuple[DayAudit, ...]
    streak: int  # observed days in a row, ending with the last day
    edge_max_week: int
    edge_weekly_max: int
    ratings: tuple[RatingRow, ...]
    rating_mean: float | None
    hours: dict[int, int] = field(default_factory=dict)

    @property
    def sent(self) -> int:
        return sum(day.sent for day in self.days)

    @property
    def deep_sleep(self) -> int:
        return sum(day.deep_sleep for day in self.days)

    @property
    def spacing_violations(self) -> int:
        return sum(day.spacing_violations for day in self.days)

    @property
    def chase_violations(self) -> int:
        return sum(day.chase_violations for day in self.days)

    @property
    def suppressed_window(self) -> int:
        return sum(day.suppressed_window for day in self.days)

    @property
    def edge_ok(self) -> bool:
        return self.edge_max_week <= self.edge_weekly_max

    @property
    def count_ok(self) -> bool:
        return all(day.observed and day.count_ok for day in self.days)

    @property
    def compliant(self) -> bool:
        """Every day compliant and the edge allowance kept in every week."""
        return all(day.compliant for day in self.days) and self.edge_ok

    @property
    def complete(self) -> bool:
        """The whole run was watched (a streak as long as the run)."""
        return self.streak >= len(self.days)

    @property
    def missing_days(self) -> int:
        return max(0, len(self.days) - self.streak)

    def to_json(self) -> dict[str, Any]:
        return {
            "first_day": self.first_day.isoformat(),
            "last_day": self.last_day.isoformat(),
            "days": [day.to_json() for day in self.days],
            "streak": self.streak,
            "sent": self.sent,
            "deep_sleep": self.deep_sleep,
            "spacing_violations": self.spacing_violations,
            "chase_violations": self.chase_violations,
            "suppressed_window": self.suppressed_window,
            "edge_max_week": self.edge_max_week,
            "edge_weekly_max": self.edge_weekly_max,
            "compliant": self.compliant,
            "complete": self.complete,
            "rating_count": len(self.ratings),
            "rating_mean": self.rating_mean,
            "hours": {str(hour): count for hour, count in sorted(self.hours.items())},
        }


def _day_range(rows: Sequence[LogEntry]) -> tuple[int | None, int | None, bool | None]:
    """The range the day was planned with: the newest row of the day that carries one."""
    for row in reversed(rows):
        if row.range_min is not None and row.range_max is not None:
            return row.range_min, row.range_max, row.enabled
    return None, None, None


def _edge_peak(sent: Sequence[LogEntry]) -> int:
    """The most edge messages in any seven days."""
    stamps = [row.at for row in sent if row.her_state == "sleep_edge"]
    best = 0
    for index, moment in enumerate(stamps):
        inside = sum(1 for other in stamps[: index + 1] if moment - other < WEEK)
        best = max(best, inside)
    return best


def audit_days(
    log: ProactiveLogStore,
    ratings: RatingStore,
    config: ProactiveConfig,
    *,
    first_day: date,
    last_day: date,
    now: datetime,
    rating_from: datetime | None = None,
) -> Audit:
    """The audit of the local days ``first_day`` to ``last_day`` (both included)."""
    spacing = timedelta(minutes=config.min_spacing_min)
    rows = log.entries(first_day=first_day, last_day=last_day)
    sent_all = [row for row in rows if row.outcome == "sent"]
    previous = log.last_sent_before(sent_all[0].at) if sent_all else None
    by_day: dict[date, list[LogEntry]] = {}
    for row in rows:
        by_day.setdefault(row.local_date, []).append(row)

    spacing_by_day: Counter[date] = Counter()
    last = previous
    for row in sent_all:
        if last is not None and row.at - last.at < spacing:
            spacing_by_day[row.local_date] += 1
        last = row

    days: list[DayAudit] = []
    hours_total: Counter[int] = Counter()
    day = first_day
    while day <= last_day:
        of_day = by_day.get(day, [])
        sent = [row for row in of_day if row.outcome == "sent"]
        refused = Counter(
            row.reason or "unknown"
            for row in of_day
            if row.outcome in ("rejected", "failed", "dropped") and row.reason
        )
        low, high, enabled = _day_range(of_day)
        hours: Counter[int] = Counter()
        for row in sent:
            hours[int(row.local_at[11:13])] += 1
        hours_total.update(hours)
        days.append(
            DayAudit(
                day=day,
                observed=any(row.outcome == "opened" for row in of_day),
                low=low,
                high=high,
                enabled=enabled,
                sent=len(sent),
                deep_sleep=sum(1 for row in sent if row.her_state == "deep_sleep"),
                edge=sum(1 for row in sent if row.her_state == "sleep_edge"),
                spacing_violations=spacing_by_day[day],
                chase_violations=sum(1 for row in sent if row.chase_seq > config.max_chase),
                suppressed_window=sum(count for key, count in refused.items() if key in WINDOW),
                refused=dict(refused),
                excused=any(key in OUTSIDE for key in refused),
                hours=dict(hours),
            )
        )
        day += timedelta(days=1)

    streak = 0
    for item in reversed(days):
        if not item.observed:
            break
        streak += 1
    start = rating_from if rating_from is not None else now - WEEK
    given = tuple(ratings.between(start, now + timedelta(seconds=1)))
    return Audit(
        first_day=first_day,
        last_day=last_day,
        days=tuple(days),
        streak=streak,
        edge_max_week=_edge_peak(sent_all),
        edge_weekly_max=config.edge_of_sleep_weekly_max,
        ratings=given,
        rating_mean=average(given),
        hours=dict(hours_total),
    )


def audit_recent(
    log: ProactiveLogStore,
    ratings: RatingStore,
    config: ProactiveConfig,
    time: TimeService,
    clock: Clock,
    *,
    days: int,
    include_today: bool = False,
) -> Audit:
    """The audit of the last ``days`` completed local days (plus today so far if asked)."""
    if days < 1:
        raise ValueError("an audit covers at least one day")
    now = clock.now_utc()
    today = time.local_date(now)
    last_day = today if include_today else today - timedelta(days=1)
    first_day = last_day - timedelta(days=days - 1)
    start, _ = time.day_bounds_utc(first_day)
    return audit_days(
        log, ratings, config, first_day=first_day, last_day=last_day, now=now, rating_from=start
    )
