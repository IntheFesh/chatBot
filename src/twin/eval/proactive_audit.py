"""The audit of the proactive messages (R-EVAL-005): what the log says about the last days.

:func:`audit_days` reads ``proactive_log`` and ``ratings`` for a run of **completed** local days
(the days before today, in the bot's zone) and checks, for every day:

* **deep sleep** - no message went out while she was in deep sleep (the core of the night): zero;
* **the range** - the number of messages that went out is inside the range the day was planned
  with (``[daily_min, daily_max]``; ``[0, 0]`` when proactive messages were off).  Above the
  maximum is a violation on every day.  Below the minimum it depends on the kind of day (round
  18, user decision, D-850):

  - an **excused day** (免责日) is a day she could write on whose count stayed *below* its minimum
    while the log shows a refusal from outside the schedule - the platform window or count, the
    login, the budget, a running channel probe, a user who does not answer (the chase limit); it
    is neither a violation nor a day of observation, and the reasons are kept for the report;
  - a **day of observation** (观察日) is every other day she could write on (watched, the switch
    on, not paused, the range above ``0``): its count must be strictly inside the range, below the
    minimum included - a day inside its range is one even if something was refused that day;
  - a day the user switched off (``/主动 关``), paused (``/暂停``) or planned for ``0-0``, and a day
    the scheduler did not run, is neither: it shows nothing of "会主动" and is not counted (R17-14,
    D-663; a pause is a user's decision, so it does not excuse - it takes the day out);

* **spacing and chase** - two messages are at least ``min_spacing_min`` apart, and a message
  that nobody answered is followed by at most ``max_chase`` more;
* **the edge of sleep** - at most ``edge_of_sleep_weekly_max`` messages sent at the edge of her
  sleep in any seven days.

Besides the verdict it keeps the distribution of the local hours the messages went out, the number
of messages the window suppressed, the reasons of the other refusals and the ratings of the period.
The M3 gate (:mod:`twin.eval.proactive_gate`) is made of exactly these numbers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
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
from twin.schedule.proactive.types import (
    OUTSIDE_REASONS,
    REASON_LABELS,
    WINDOW_REASONS,
    Reason,
)
from twin.schedule.time_service import TimeService

WEEK = timedelta(days=7)
INACTIVE_NOTE = "当天主动消息关闭、暂停或范围 0-0，不计入观察"
UNOBSERVED_NOTE = "程序没有运行（当天没有记录），不计入观察"
OUTSIDE = frozenset(reason.value for reason in OUTSIDE_REASONS)
WINDOW = frozenset(reason.value for reason in WINDOW_REASONS)
USER_OFF = frozenset({Reason.PAUSED.value, Reason.DISABLED.value})  # /暂停 and /主动 关
EXTERNAL = OUTSIDE - USER_OFF  # what excuses a day below its minimum (D-850)


def describe_reasons(reasons: Mapping[str, int]) -> str:
    """``被窗口抑制 2、平台条数用完 1``: the most frequent first (ties by the code)."""
    ordered = sorted(reasons.items(), key=lambda item: (-item[1], item[0]))
    return "、".join(f"{REASON_LABELS.get(key, key)} {count}" for key, count in ordered)


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
    excused: bool  # the log shows a refusal from outside the schedule (:data:`EXTERNAL`)
    hours: dict[int, int]
    switched_off: bool = False  # a row of the day says the switch was off (``/主动 关``)
    paused: bool = False  # a candidate of the day was refused because of ``/暂停``

    @property
    def effective_range(self) -> tuple[int, int] | None:
        """The range the count of the day is held to: ``0-0`` while the switch was off."""
        if self.high is None or self.low is None:
            return None
        return (0, 0) if self.enabled is False else (self.low, self.high)

    @property
    def user_off(self) -> bool:
        """The user took the day out: proactive messages switched off or paused."""
        return self.switched_off or self.paused

    @property
    def active(self) -> bool:
        """A watched day on which she could write (see the module description)."""
        return self.observed and not self.user_off and self.high is not None and self.high > 0

    @property
    def exempt(self) -> bool:
        """An excused day: below the minimum because of a refusal from outside (D-850)."""
        return self.active and self.low is not None and self.sent < self.low and self.excused

    @property
    def observation(self) -> bool:
        """A day of observation: one she could write on that is not excused (D-850)."""
        return self.active and not self.exempt

    @property
    def exempt_reasons(self) -> dict[str, int]:
        """The refusals from outside that excused the day (empty unless it is excused)."""
        if not self.exempt:
            return {}
        return {key: count for key, count in self.refused.items() if key in EXTERNAL}

    @property
    def count_ok(self) -> bool:
        """Never above the maximum; below the minimum only on a day that is not observed."""
        bounds = self.effective_range
        if bounds is None:
            return True  # no range was planned (the scheduler did not run): nothing to hold to
        low, high = bounds
        if self.sent > high:
            return False
        return self.sent >= low or not self.observation

    @property
    def problems(self) -> list[str]:
        found: list[str] = []
        if self.deep_sleep:
            found.append(f"深睡时段发了 {self.deep_sleep} 条")
        if not self.count_ok:
            bounds = self.effective_range
            span = f"{bounds[0]}-{bounds[1]}" if bounds is not None else "未知"
            found.append(f"发了 {self.sent} 条，不在 {span} 范围内")
        if self.spacing_violations:
            found.append(f"{self.spacing_violations} 次间隔不足")
        if self.chase_violations:
            found.append(f"{self.chase_violations} 次追发超限")
        return found

    @property
    def notes(self) -> list[str]:
        """Why a day that breaks no rule is not a day of observation."""
        if not self.observed:
            return [UNOBSERVED_NOTE]
        if not self.active:
            return [INACTIVE_NOTE]
        if self.exempt:
            return [
                f"免责日：发了 {self.sent} 条，少于下限 {self.low}，当天有外部拒绝"
                f"（{describe_reasons(self.exempt_reasons)}），不计入观察"
            ]
        return []

    @property
    def compliant(self) -> bool:
        """No rule broken (a day that is not observed breaks none by not being observed)."""
        return not self.problems

    def to_json(self) -> dict[str, Any]:
        return {
            "day": self.day.isoformat(),
            "observed": self.observed,
            "active": self.active,
            "user_off": self.user_off,
            "exempt": self.exempt,
            "observation": self.observation,
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
        return all(day.count_ok for day in self.days)

    @property
    def compliant(self) -> bool:
        """Every day compliant and the edge allowance kept in every week."""
        return all(day.compliant for day in self.days) and self.edge_ok

    @property
    def observation_days(self) -> int:
        """Days of observation (D-850): she could write, and the day is not excused."""
        return sum(1 for day in self.days if day.observation)

    @property
    def exempt_days(self) -> int:
        """Excused days: below the minimum because of a refusal from outside (D-850)."""
        return sum(1 for day in self.days if day.exempt)

    @property
    def exempt_reasons(self) -> Counter[str]:
        """The refusals from outside on the excused days, by reason."""
        found: Counter[str] = Counter()
        for day in self.days:
            found.update(day.exempt_reasons)
        return found

    @property
    def inactive_days(self) -> int:
        """Watched days that do not count: switched off, paused, or a range of ``0-0``."""
        return sum(1 for day in self.days if day.observed and not day.active)

    @property
    def unobserved_days(self) -> int:
        """Days the scheduler did not run (no mark): they show nothing either way."""
        return sum(1 for day in self.days if not day.observed)

    def to_json(self) -> dict[str, Any]:
        return {
            "first_day": self.first_day.isoformat(),
            "last_day": self.last_day.isoformat(),
            "days": [day.to_json() for day in self.days],
            "observation_days": self.observation_days,
            "exempt_days": self.exempt_days,
            "exempt_reasons": dict(self.exempt_reasons),
            "inactive_days": self.inactive_days,
            "unobserved_days": self.unobserved_days,
            "sent": self.sent,
            "deep_sleep": self.deep_sleep,
            "spacing_violations": self.spacing_violations,
            "chase_violations": self.chase_violations,
            "suppressed_window": self.suppressed_window,
            "edge_max_week": self.edge_max_week,
            "edge_weekly_max": self.edge_weekly_max,
            "compliant": self.compliant,
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
                excused=any(key in EXTERNAL for key in refused),
                hours=dict(hours),
                switched_off=any(
                    row.enabled is False or row.reason == Reason.DISABLED.value for row in of_day
                ),
                paused=Reason.PAUSED.value in refused,
            )
        )
        day += timedelta(days=1)

    start = rating_from if rating_from is not None else now - WEEK
    given = tuple(ratings.between(start, now + timedelta(seconds=1)))
    return Audit(
        first_day=first_day,
        last_day=last_day,
        days=tuple(days),
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
