"""A simulated month for the proactive scheduler: the clock, the user and the statistics.

:func:`simulate` drives the **production** scheduler (``tests.support.proactive_world.World``) over
whole local days with a manual clock.  The user is a simple model: most evenings he writes a few
messages (which reopens the platform window), he answers a share of her messages after a while, and
on some days he does not write at all.  Every tick is the scheduler's own; the pauses between the
bubbles of a message jump the clock.  :class:`SimResult` collects what the log says, per day and
altogether, and :func:`render` prints it as text tables (``scripts/simulate_proactive.py``).
"""

from __future__ import annotations

import heapq
import random
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from tests.support.proactive_world import World
from twin.schedule.proactive.store import LogEntry

TICK = timedelta(minutes=5)
TICK_OFFSET_S = 40.0


@dataclass(frozen=True)
class Session:
    """A time of day at which he sits down to write: how likely, from when to when, how much."""

    probability: float
    hours: tuple[float, float]
    exchanges: tuple[int, int] = (2, 5)  # messages of one session


EVENING = Session(0.9, (19.5, 22.5))
ALL_DAY = (Session(1.0, (8.0, 9.0)), Session(1.0, (14.0, 15.0)), Session(1.0, (19.5, 21.0)))


@dataclass
class UserBehavior:
    """How the simulated user behaves (all draws come from the simulation's seed)."""

    reply_probability: float = 0.7  # he answers a message of hers ...
    reply_delay_min: tuple[float, float] = (4.0, 120.0)  # ... after this many minutes
    sessions: tuple[Session, ...] = (EVENING,)  # when he writes: by default most evenings
    silent_days: frozenset[int] = frozenset()  # day numbers (0-based) on which he never writes


@dataclass
class DaySummary:
    day: date
    sent: int
    low: int
    high: int
    quota: int
    kinds: Counter[str]
    refused: Counter[str]
    hours: Counter[int]
    states: Counter[str]
    silent: bool


@dataclass
class SimResult:
    days: list[DaySummary]
    entries: list[LogEntry]
    ticks: int
    first_day: date
    last_day: date
    behavior: UserBehavior
    messages: int = 0

    @property
    def sent(self) -> list[LogEntry]:
        return [row for row in self.entries if row.outcome == "sent"]

    @property
    def mean_per_day(self) -> float:
        return sum(day.sent for day in self.days) / len(self.days)

    @property
    def mean_quota(self) -> float:
        return sum(day.quota for day in self.days) / len(self.days)

    def refused(self) -> Counter[str]:
        total: Counter[str] = Counter()
        for day in self.days:
            total.update(day.refused)
        return total

    def kinds(self) -> Counter[str]:
        total: Counter[str] = Counter()
        for day in self.days:
            total.update(day.kinds)
        return total

    def hours(self) -> Counter[int]:
        total: Counter[int] = Counter()
        for day in self.days:
            total.update(day.hours)
        return total

    def states(self) -> Counter[str]:
        total: Counter[str] = Counter()
        for day in self.days:
            total.update(day.states)
        return total


async def simulate(
    world: World,
    *,
    first_day: date,
    days: int,
    behavior: UserBehavior | None = None,
    seed: int = 1,
) -> SimResult:
    """Run the scheduler over ``days`` local days from ``first_day`` (see the module text)."""
    chosen = behavior or UserBehavior()
    rng = random.Random(seed)
    time = world.rig.kit.time
    start, _ = time.day_bounds_utc(first_day)
    last_day = first_day + timedelta(days=days - 1)
    _, end = time.day_bounds_utc(last_day)
    events: list[tuple[datetime, int, str]] = []
    counter = 0

    def schedule(moment: datetime, what: str) -> None:
        nonlocal counter
        counter += 1
        heapq.heappush(events, (moment, counter, what))

    for number in range(days):
        day = first_day + timedelta(days=number)
        if number in chosen.silent_days:
            continue
        for session in chosen.sessions:
            if rng.random() >= session.probability:
                continue
            first = rng.uniform(*session.hours)
            for index in range(rng.randint(*session.exchanges)):
                schedule(time.local_to_utc(day, (first + index * 0.05) * 60), "session")

    world.clock.set_time(start + timedelta(seconds=30))
    ticks = 0
    messages = 0
    next_tick = world.clock.now_utc()
    while next_tick < end:
        world.clock.set_time(max(world.clock.now_utc(), next_tick))
        while events and events[0][0] <= world.clock.now_utc():
            when, _, what = heapq.heappop(events)
            world.user_writes("今天好累" if what == "session" else "好的", at=max(when, start))
            messages += 1
        report = await world.scheduler.tick()
        ticks += 1
        if report.outcome == "sent":
            number = (world.clock.now_utc().astimezone(time.bot_timezone()).date() - first_day).days
            if number not in chosen.silent_days and rng.random() < chosen.reply_probability:
                delay = rng.uniform(*chosen.reply_delay_min)
                schedule(world.clock.now_utc() + timedelta(minutes=delay), "reply")
        next_tick = _align_up(max(next_tick, world.clock.now_utc()) + timedelta(seconds=1))

    entries = world.log.entries(first_day=first_day, last_day=last_day, with_text=False)
    summaries: list[DaySummary] = []
    for number in range(days):
        day = first_day + timedelta(days=number)
        of_day = [row for row in entries if row.local_date == day]
        sent = [row for row in of_day if row.outcome == "sent"]
        opened = [row for row in of_day if row.outcome == "opened"]
        summaries.append(
            DaySummary(
                day=day,
                sent=len(sent),
                low=opened[0].range_min or 0 if opened else 0,
                high=opened[0].range_max or 0 if opened else 0,
                quota=opened[0].quota_total or 0 if opened else 0,
                kinds=Counter(row.kind for row in sent),
                refused=Counter(row.reason or "?" for row in of_day if row.outcome == "rejected"),
                hours=Counter(int(row.local_at[11:13]) for row in sent),
                states=Counter(row.her_state or "?" for row in sent),
                silent=number in chosen.silent_days,
            )
        )
    return SimResult(summaries, entries, ticks, first_day, last_day, chosen, messages)


def _align_up(moment: datetime) -> datetime:
    """The first tick (a multiple of five minutes plus 40 seconds) at or after ``moment``."""
    period = TICK.total_seconds()
    number = -(-(moment.timestamp() - TICK_OFFSET_S) // period)
    return datetime.fromtimestamp(number * period + TICK_OFFSET_S, tz=moment.tzinfo)


def render(result: SimResult) -> str:
    """The statistics as text: per day, then the refusals, the kinds and the hours."""
    lines = ["日期        条数 范围  配额  类型                      被拒"]
    for day in result.days:
        kinds = " ".join(f"{k}:{n}" for k, n in sorted(day.kinds.items()))
        refused = " ".join(f"{k}:{n}" for k, n in sorted(day.refused.items()))
        lines.append(
            f"{day.day}  {day.sent:>3}  {day.low}-{day.high}  {day.quota:>3}   "
            f"{kinds:<26}{refused}" + ("  (用户整天没说话)" if day.silent else "")
        )
    lines.append("")
    lines.append(
        f"日均 {result.mean_per_day:.2f} 条（配额日均 {result.mean_quota:.2f}），"
        f"共 {len(result.sent)} 条，{result.ticks} 个 tick"
    )
    lines.append(
        "拒绝原因：" + (", ".join(f"{k}={n}" for k, n in sorted(result.refused().items())) or "无")
    )
    lines.append("类型：" + ", ".join(f"{k}={n}" for k, n in sorted(result.kinds().items())))
    lines.append("她的状态：" + ", ".join(f"{k}={n}" for k, n in sorted(result.states().items())))
    lines.append("发送时刻（当地小时）：")
    peak = max(result.hours().values(), default=1)
    for hour in range(24):
        count = result.hours().get(hour, 0)
        lines.append(f"  {hour:02d}:00 {count:>3} {'#' * round(24 * count / peak)}")
    return "\n".join(lines)


__all__ = [
    "ALL_DAY",
    "EVENING",
    "DaySummary",
    "Session",
    "SimResult",
    "UserBehavior",
    "render",
    "simulate",
]
