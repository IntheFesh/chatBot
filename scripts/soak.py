"""Soak: ``twin run`` over many days on a fast-forward clock (R-NFR-002, R-ARCH-004).

A development tool, not a ``twin`` command.  It assembles the **production** application with the
very function ``twin run`` uses (``twin.assembly.assemble``) on the local console channel, a
throw-away data directory, a fast-forward clock and DeepSeek replaced by the scripted double of
``tests/support`` - then lets a synthetic user live with it for days: mornings and evenings of
talk, a night message now and then, pictures and stickers, a command or two, days of silence and one
stretch of three days without a word.  It imports ``tests.support``, which is why it lives here and
not in the package; nothing it prints comes from real chat data, and nothing it does touches the
real data directory, the real credentials or the network.

Usage::

    uv run python scripts/soak.py --days 14 --accelerated
    uv run python scripts/soak.py --days 14 --accelerated --seed 5 --output soak.txt
    uv run python scripts/soak.py --days 3 --accelerated --platform ilink

At the end of every simulated day (local midnight) it takes a sample of the process - resident
memory after a garbage collection, the job queue, the database, the tasks and threads of the loop -
and counts the exceptions the application logged or lost.  The verdicts (see :class:`Thresholds`):

* the resident memory stays below the limit of R-NFR-002 (1.5 GB) and does not grow from day to
  day (a leak would show as a slope);
* the job queue has no backlog (pending jobs that do not wait for the user's approval) and nothing
  fails;
* the database grows by what the conversation writes and no more;
* the loop has no more tasks and threads on the last day than on the second;
* no ERROR was logged, no task died with an exception, no ``engine_error`` alert was raised;
* what every story of the scenario tests promises still holds after all the days: no bubble lost
  or doubled, nothing in deep sleep, the platform's count respected, the proactive audit clean, and
  nothing of the bot's own words in her messages, her profile or the library of examples
  (CLAUDE.md rule 7).

The report goes to the standard output, or to ``--output``; ``--json`` writes the numbers.  The exit
code is 1 when a verdict fails.  ``--accelerated`` is required: a soak at the speed of the clock is
``twin run`` itself.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # `tests.support` lives next to `src`

if TYPE_CHECKING:
    from tests.support.life_world import LifeWorld

TRACE_FROM_DAY = 2  # --trace-memory compares the memory after this many days with the end
PLATFORMS = ("console", "ilink")
DEFAULT_FIRST_DAY = "2026-10-26"  # a Monday; 14 days from it cross the end of daylight saving time
SECONDS_PER_DAY = 24 * 3600

# her opening curve: the wake-up, the meals, and the half hour before she goes to bed
PEAKS = {31: 0.5, 32: 0.5, 48: 0.6, 49: 0.6, 73: 0.6, 74: 0.6, 90: 0.5, 91: 0.5, 92: 0.5}
PEAKS |= {93: 0.5, 94: 0.5, 95: 0.5}

# what the synthetic user says (nothing here is anybody's words)
SAYINGS = (
    "今天好累啊", "刚下课", "晚上吃火锅吧", "你在干嘛", "我明天要早起", "周末去逛街吧",
    "下周三有考试", "天气好冷", "刚到家", "今天的课好无聊", "我饿了", "在吗",
    "我室友又在打游戏", "快放假了", "好想睡觉", "你看到我发的了吗", "晚饭吃什么好",
    "今天跑步了", "地铁好挤", "刚开完会", "明天要交作业", "咖啡喝多了睡不着", "笑得停不下来",
    "刚洗完澡", "这周好忙", "我妈刚打电话来", "想吃甜的", "明天下雨", "刚看完一部电影",
)  # fmt: skip
COMMANDS = ("/状态", "/费用 今天", "/记忆", "/帮助")
# what the made-up model answers, one answer per message of the user (a line is a bubble)
REPLIES = (
    "嗯嗯", "好呀", "太好笑了吧", "真的吗", "辛苦啦", "是哦", "那你早点休息", "我也是",
    "吃了吗", "好困啊", "[表情包:开心]", "等一下哦\n马上回你", "别太累了", "怎么啦", "对呀对呀",
    "知道啦", "那就好", "我刚刚在看书\n有点困", "你呢", "笑死我了", "加油呀", "晚安哦",
)  # fmt: skip


# ----------------------------------------------------------------------------- the user


@dataclass(frozen=True)
class Event:
    """One thing the user does: ``at_s`` seconds after the local midnight of the day."""

    at_s: int
    kind: Literal["text", "picture", "sticker", "command"]
    text: str = ""
    seed: int = 0


@dataclass(frozen=True)
class DayPlan:
    """A day of the user."""

    day: date
    mood: Literal["normal", "chatty", "quiet", "silent"]
    events: tuple[Event, ...]


# (earliest, latest) second of the day, probability, and how many messages: the sessions of a day
SESSIONS = (
    ((7 * 3600 + 30 * 60, 9 * 3600 + 30 * 60), 0.6, (2, 6)),  # the morning
    ((12 * 3600, 13 * 3600 + 30 * 60), 0.55, (2, 6)),  # the noon
    ((15 * 3600, 17 * 3600 + 30 * 60), 0.45, (2, 5)),  # the afternoon
    ((19 * 3600, 23 * 3600), 0.95, (6, 20)),  # the evening
    ((1 * 3600, 4 * 3600), 0.15, (1, 2)),  # the night
)
LONGEST_GAP_S = 150


def plan_days(seed: int, first_day: date, days: int) -> list[DayPlan]:
    """The days of the user: the same seed gives the same days (and the same ones for a longer run).

    Every day has its own generator, so that asking for more days does not change the first ones.
    A stretch of three days without a word lies in the middle of any run of eight days or more,
    and now and then a single day is silent.
    """
    if days < 1:
        raise ValueError("a soak needs at least one day")
    long_silence = range(0)
    if days >= 8:
        start = 3 + random.Random(seed).randrange(days - 6)  # noqa: S311
        long_silence = range(start, start + 3)
    plans: list[DayPlan] = []
    for index in range(days):
        rng = random.Random(f"{seed}/{index}")  # noqa: S311
        day = first_day + timedelta(days=index)
        mood = rng.choices(("normal", "chatty", "quiet"), (0.55, 0.2, 0.25))[0]
        if index in long_silence or (index >= 2 and rng.random() < 0.06):
            plans.append(DayPlan(day, "silent", ()))
            continue
        events: list[Event] = []
        weekend = day.weekday() >= 5
        for (first, last), chance, (fewest, most) in SESSIONS:
            if rng.random() >= chance:
                continue
            count = rng.randint(fewest, most)
            count = (
                max(1, count // 2) if mood == "quiet" else count * (2 if mood == "chatty" else 1)
            )
            at = rng.randint(first + (3600 if weekend and last < 12 * 3600 else 0), last)
            at = min(at, SECONDS_PER_DAY - 600 - count * LONGEST_GAP_S)  # the session ends today
            for _ in range(count):
                roll = rng.random()
                kind: Literal["text", "picture", "sticker", "command"] = "text"
                if roll < 0.05:
                    kind = "picture"
                elif roll < 0.10:
                    kind = "sticker"
                elif roll < 0.14:
                    kind = "command"
                text = rng.choice(COMMANDS if kind == "command" else SAYINGS)
                events.append(Event(at, kind, text, rng.randrange(10**6)))
                at += rng.randint(15, LONGEST_GAP_S)
        if not any(event.kind == "text" for event in events):
            events.append(Event(20 * 3600 + 15 * 60, "text", rng.choice(SAYINGS), 0))
        events.sort(key=lambda event: event.at_s)
        plans.append(DayPlan(day, mood, tuple(events)))
    if days >= 4 and not any(e.at_s < 5 * 3600 for plan in plans for e in plan.events):
        # a person who never writes at night is not the one this tool is for: one message, once
        index = next(n for n in range(1, days) if plans[n].events)
        late = Event(2 * 3600 + 10 * 60 + seed % 50 * 60, "text", SAYINGS[seed % len(SAYINGS)], 0)
        plans[index] = replace(plans[index], events=(late, *plans[index].events))
    return plans


# --------------------------------------------------------------------------- the numbers


@dataclass(frozen=True)
class Thresholds:
    """What passes.  The absolute limits are the specification's; the growth limits are ours."""

    rss_limit_mb: float = (
        1536.0  # R-NFR-002: resident memory below 1.5 GB (llama-server not counted)
    )
    rss_growth_mb_per_day: float = 3.0  # least-squares slope after the warm-up (~1 GB a year)
    # The process takes about three weeks to reach its working size (the caches of the database
    # and of the vector index, the first run of every weekly job: a 28-day run grows 43 MB in
    # its first week and 1 MB in its last).  The trend is judged on the days after that, and only
    # when there are enough of them to fit a line.
    warmup_days: int = 21
    trend_days_min: int = 5
    pending_jobs_max: int = 40  # not counting those that wait for the user's approval
    oldest_pending_s_max: float = 6 * 3600.0
    failed_jobs_max: int = 0
    db_growth_mb_per_day: float = 10.0
    extra_tasks_max: int = 3  # tasks of the loop on the last day beyond those of the second
    extra_threads_max: int = 3
    errors_max: int = 0


@dataclass
class Sample:
    """The process and the application at the end of a simulated day."""

    day_index: int
    day: str
    local_end: str
    rss_mb: float
    rss_anon_mb: float  # the heap (0 where the system cannot say)
    rss_file_mb: float  # mapped files: libraries, the database, the vectors
    tasks: int
    threads: int
    jobs_pending: int  # waiting to run (not counting the ones that wait for approval)
    jobs_awaiting_approval: int
    jobs_running: int
    jobs_failed: int
    oldest_pending_s: float
    db_mb: float
    messages_in: int
    bubbles_out: int
    proactive_sent: int
    errors: int
    warnings: int
    clock_steps: int


@dataclass
class Verdict:
    name: str
    ok: bool
    detail: str


@dataclass
class SoakReport:
    """Everything a run found; :func:`evaluate` turns it into verdicts."""

    days: int
    seed: int
    platform: str
    first_day: str
    real_seconds: float = 0.0
    samples: list[Sample] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)  # promises of the stories that broke
    error_events: list[str] = field(
        default_factory=list
    )  # the first few ERROR records (names only)
    task_errors: int = 0
    engine_errors: int = 0
    calls: dict[str, int] = field(default_factory=dict)  # requests to the made-up DeepSeek by kind
    alerts: dict[str, int] = field(default_factory=dict)
    jobs: dict[str, int] = field(default_factory=dict)
    summary: dict[str, int] = field(default_factory=dict)  # messages, bubbles, proactive by kind
    baseline_rss_mb: float = 0.0
    growth_sources: list[str] = field(default_factory=list)  # with --trace-memory
    peak_rss_mb: float = 0.0
    verdicts: list[Verdict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.verdicts) and all(verdict.ok for verdict in self.verdicts)


def slope(points: list[tuple[float, float]]) -> float:
    """The least-squares slope of ``y`` over ``x`` (0 for fewer than two distinct ``x``)."""
    if len(points) < 2:
        return 0.0
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    spread = sum((x - mean_x) ** 2 for x, _ in points)
    if spread == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / spread


def evaluate(report: SoakReport, limits: Thresholds | None = None) -> list[Verdict]:
    """The verdicts of ``report`` against ``limits`` (see the module description)."""
    limits = limits or Thresholds()
    samples = report.samples
    verdicts: list[Verdict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        verdicts.append(Verdict(name, ok, detail))

    if not samples:
        add("samples", False, "no day was completed")
        return verdicts
    peak = max([sample.rss_mb for sample in samples] + [report.peak_rss_mb])
    add(
        "memory limit (R-NFR-002)",
        peak < limits.rss_limit_mb,
        f"highest resident memory {peak:,.0f} MB, limit {limits.rss_limit_mb:,.0f} MB",
    )
    settled = [(float(s.day_index), s.rss_mb) for s in samples if s.day_index >= limits.warmup_days]
    if len(settled) >= limits.trend_days_min:
        trend = slope(settled)
        add(
            "memory growth",
            trend <= limits.rss_growth_mb_per_day,
            f"{trend:+.2f} MB/day over days {settled[0][0] + 1:.0f}-{settled[-1][0] + 1:.0f} "
            f"({settled[0][1]:,.0f} -> {settled[-1][1]:,.0f} MB), limit "
            f"{limits.rss_growth_mb_per_day:+.1f} MB/day",
        )
    else:
        first, last = samples[0].rss_mb, samples[-1].rss_mb
        add(
            "memory growth",
            True,
            f"not judged: the trend needs {limits.warmup_days + limits.trend_days_min} days "
            f"({limits.warmup_days} of warm-up), this run has {len(samples)}; "
            f"{first:,.0f} -> {last:,.0f} MB, only the limit is judged "
            f"(run --days {limits.warmup_days + 7} for the trend)",
        )
    backlog = max(sample.jobs_pending for sample in samples)
    oldest = max(sample.oldest_pending_s for sample in samples)
    failed = samples[-1].jobs_failed
    add(
        "job queue",
        backlog <= limits.pending_jobs_max
        and oldest <= limits.oldest_pending_s_max
        and failed <= limits.failed_jobs_max,
        f"at most {backlog} waiting at a midnight (limit {limits.pending_jobs_max}), the oldest "
        f"{oldest / 60:.0f} min (limit {limits.oldest_pending_s_max / 60:.0f}), "
        f"{failed} failed (limit {limits.failed_jobs_max}); "
        f"{samples[-1].jobs_awaiting_approval} wait for approval",
    )
    days_between = max(1, samples[-1].day_index - samples[0].day_index)
    growth = (samples[-1].db_mb - samples[0].db_mb) / days_between
    add(
        "database size",
        growth <= limits.db_growth_mb_per_day,
        f"{samples[0].db_mb:,.1f} -> {samples[-1].db_mb:,.1f} MB, {growth:+.2f} MB/day, "
        f"limit {limits.db_growth_mb_per_day:+.1f} MB/day",
    )
    reference = samples[1] if len(samples) > 1 else samples[0]
    more_tasks = samples[-1].tasks - reference.tasks
    more_threads = samples[-1].threads - reference.threads
    add(
        "tasks and threads",
        more_tasks <= limits.extra_tasks_max and more_threads <= limits.extra_threads_max,
        f"tasks {reference.tasks} -> {samples[-1].tasks}, threads {reference.threads} -> "
        f"{samples[-1].threads}",
    )
    errors = samples[-1].errors + report.task_errors + report.engine_errors
    add(
        "exceptions",
        errors <= limits.errors_max,
        f"{samples[-1].errors} ERROR records, {report.task_errors} lost task exceptions, "
        f"{report.engine_errors} engine_error alerts (limit {limits.errors_max})"
        + (f"; first: {', '.join(report.error_events[:3])}" if report.error_events else ""),
    )
    add(
        "promises of the scenarios",
        not report.invariants,
        "all hold" if not report.invariants else "; ".join(report.invariants[:4]),
    )
    return verdicts


def render(report: SoakReport) -> str:
    """The report as plain text."""
    lines = [
        f"soak: {report.days} days from {report.first_day}, seed {report.seed}, "
        f"{report.platform} channel, {report.real_seconds:,.0f} s of real time",
        "",
        "day  date        rss MB  heap  files  tasks thr  jobs wait/appr/fail  oldest min  "
        "db MB   in  out  proactive  err",
    ]
    for s in report.samples:
        lines.append(
            f"{s.day_index + 1:>3}  {s.day}  {s.rss_mb:>6,.0f}  {s.rss_anon_mb:>4,.0f}  "
            f"{s.rss_file_mb:>5,.0f}  {s.tasks:>5} {s.threads:>3}  "
            f"{s.jobs_pending:>4}/{s.jobs_awaiting_approval:<4}/{s.jobs_failed:<4}  "
            f"{s.oldest_pending_s / 60:>10,.0f}  {s.db_mb:>5.1f}  {s.messages_in:>3}  "
            f"{s.bubbles_out:>3}  {s.proactive_sent:>9}  {s.errors:>3}"
        )
    lines += [
        "",
        f"memory: {report.baseline_rss_mb:,.0f} MB at the start, "
        f"highest {report.peak_rss_mb:,.0f} MB",
        "totals: " + ", ".join(f"{k} {v}" for k, v in sorted(report.summary.items())),
        "model requests: " + ", ".join(f"{k} {v}" for k, v in sorted(report.calls.items())),
        "alerts: " + (", ".join(f"{k} {v}" for k, v in sorted(report.alerts.items())) or "none"),
        "jobs: " + ", ".join(f"{k} {v}" for k, v in sorted(report.jobs.items())),
        "",
        "verdicts:",
    ]
    lines += [f"  [{'ok' if v.ok else 'FAIL'}] {v.name}: {v.detail}" for v in report.verdicts]
    if report.growth_sources:
        lines += ["", "memory that grew after the warm-up (tracemalloc):"]
        lines += [f"  {line}" for line in report.growth_sources]
    lines.append("")
    lines.append("result: " + ("PASSED" if report.passed else "FAILED"))
    return "\n".join(lines)


def report_json(report: SoakReport) -> str:
    return json.dumps(asdict(report), ensure_ascii=False, indent=2)


# -------------------------------------------------------------------------- the machinery


class LogTally(logging.Handler):
    """Counts what the application logs at WARNING and above (names of events, never content)."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.errors = 0
        self.warnings = 0
        self.events: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.ERROR:
            self.errors += 1
            if len(self.events) < 20:
                self.events.append(f"{record.name}:{record.getMessage()[:60]}")
        else:
            self.warnings += 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--days", type=int, default=14, help="local days to simulate (14)")
    parser.add_argument(
        "--accelerated", action="store_true", help="run on the fast-forward clock (required)"
    )
    parser.add_argument("--seed", type=int, default=1, help="seed of the user and the dice (1)")
    parser.add_argument("--first-day", default=DEFAULT_FIRST_DAY, help="first local day")
    parser.add_argument("--platform", choices=PLATFORMS, default="console", help="the channel")
    parser.add_argument("--output", type=Path, default=None, help="write the text report here")
    parser.add_argument("--json", type=Path, default=None, help="write the numbers here")
    parser.add_argument(
        "--trim-heap",
        action="store_true",
        help="Linux: give freed heap back to the system before each sample (is it a leak?)",
    )
    parser.add_argument(
        "--trace-memory",
        action="store_true",
        help="name the source lines whose memory grew after the warm-up (slows the run down)",
    )
    parser.add_argument("--rss-limit-mb", type=float, default=Thresholds.rss_limit_mb)
    parser.add_argument(
        "--max-rss-growth", type=float, default=Thresholds.rss_growth_mb_per_day,
        help="largest accepted slope of the resident memory, MB per day",
    )  # fmt: skip
    args = parser.parse_args(argv)
    if not args.accelerated:
        parser.error(
            "--accelerated is required: this tool lives on the fast-forward clock "
            "(a soak at the speed of the clock is `twin run` itself)"
        )
    if args.days < 1:
        parser.error("--days must be at least 1")
    try:
        date.fromisoformat(args.first_day)
    except ValueError:
        parser.error("--first-day must be a date, YYYY-MM-DD")
    return args


def trim_heap() -> None:
    """Ask the C library to return freed memory to the system (glibc; elsewhere nothing)."""
    import ctypes

    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        return


async def take_sample(
    world: LifeWorld, tally: LogTally, index: int, day: date, started: float, trim: bool = False
) -> Sample:
    """One sample at the end of a simulated day; the stage trims what only the test doubles hold."""
    import gc
    import threading

    from sqlalchemy import func, select
    from tests.support.process_metrics import current_rss_bytes, megabytes, rss_breakdown_bytes

    from twin.storage.models import Job

    world.deepseek.log.clear()  # the doubles keep every request; the application does not
    world.clock.sleeps.clear()
    world.api.reset()  # (the router and every route keep each request they saw)
    if world.double is not None:
        world.double.requests.clear()
    gc.collect()
    if trim:
        trim_heap()
    rss = megabytes(current_rss_bytes())
    split = rss_breakdown_bytes()
    jobs = world.jobs()
    awaiting = len(world.awaiting_approval())
    with world.services.db.session() as session:
        oldest = session.scalar(
            select(func.min(Job.created_at)).where(
                Job.status == "pending", Job.requires_approval.is_(False)
            )
        )
    oldest_age = max(0.0, (world.now - oldest).total_seconds()) if oldest is not None else 0.0
    paths = world.services.paths
    db_bytes = sum(
        path.stat().st_size
        for path in (paths.db_path, paths.db_path.with_name(paths.db_path.name + "-wal"))
        if path.exists()
    )
    rows = world.rows()
    return Sample(
        day_index=index,
        day=day.isoformat(),
        local_end=world.local_text(world.now),
        rss_mb=rss,
        rss_anon_mb=megabytes(split.get("RssAnon", 0)),
        rss_file_mb=megabytes(split.get("RssFile", 0)),
        tasks=len(asyncio.all_tasks()),
        threads=threading.active_count(),
        jobs_pending=jobs.get("pending", 0) - awaiting,
        jobs_awaiting_approval=awaiting,
        jobs_running=jobs.get("running", 0),
        jobs_failed=jobs.get("failed", 0),
        oldest_pending_s=oldest_age,
        db_mb=megabytes(db_bytes),
        messages_in=sum(1 for row in rows if row.direction == "in"),
        bubbles_out=sum(1 for row in rows if row.direction == "out" and not row.is_command),
        proactive_sent=len(world.proactive_rows(outcomes=["sent"])),
        errors=tally.errors,
        warnings=tally.warnings,
        clock_steps=world.clock.steps,
    )


async def live_through(
    world: LifeWorld,
    plans: list[DayPlan],
    rng: random.Random,
    tally: LogTally,
    report: SoakReport,
    started: float,
    log: Any,
    trace: bool = False,
    trim: bool = False,
) -> None:
    """The user's days, one after the other; a sample at every local midnight."""
    import tracemalloc

    from tests.fixtures.synth_export import make_image_bytes

    book = world.deepseek.book
    for index, plan in enumerate(plans):
        midnight = world.local(0, 0, on=plan.day)
        for event in plan.events:
            moment = midnight + timedelta(seconds=event.at_s)
            if event.at_s >= 3 * 3600:  # (a DST day is 23 or 25 hours long: the hour is local)
                moment = world.local(event.at_s // 3600, event.at_s % 3600 // 60, on=plan.day)
                moment += timedelta(seconds=event.at_s % 60)
            await world.run_until(moment)
            if event.kind == "command":
                await world.say(event.text)
                continue
            book.queue.append(rng.choice(REPLIES))
            if event.kind == "picture":
                await world.say_picture(make_image_bytes(random.Random(event.seed), "PNG"))  # noqa: S311
            elif event.kind == "sticker":
                await world.say_sticker(make_image_bytes(random.Random(event.seed), "PNG"))  # noqa: S311
            else:
                await world.say(event.text)
        await world.run_until(world.local(0, 0, on=plan.day + timedelta(days=1)))
        sample = await take_sample(world, tally, index, plan.day, started, trim)
        report.samples.append(sample)
        report.peak_rss_mb = max(report.peak_rss_mb, sample.rss_mb)
        if trace and index == TRACE_FROM_DAY - 1:
            reference = tracemalloc.take_snapshot()
        log(
            f"day {index + 1}/{len(plans)} {plan.day} ({plan.mood}): {sample.rss_mb:,.0f} MB "
            f"(heap {sample.rss_anon_mb:,.0f}, files {sample.rss_file_mb:,.0f}), "
            f"{sample.jobs_pending} jobs waiting, db {sample.db_mb:.1f} MB, "
            f"{sample.messages_in} in / {sample.bubbles_out} out, {sample.errors} errors"
        )
    if trace and len(plans) > TRACE_FROM_DAY:
        report.growth_sources = growth_sources(reference, tracemalloc.take_snapshot())


def growth_sources(before: Any, after: Any, top: int = 12) -> list[str]:
    """The source lines whose allocations grew most between two ``tracemalloc`` snapshots."""
    found = []
    for stat in after.compare_to(before, "lineno")[:top]:
        frame = stat.traceback[0]
        name = Path(frame.filename).as_posix().rsplit("/src/", 1)[-1].rsplit("/tests/", 1)[-1]
        found.append(
            f"{stat.size_diff / 1024:+,.0f} KB ({stat.count_diff:+,} blocks) {name}:{frame.lineno}"
        )
    return found


async def check_promises(world: LifeWorld, report: SoakReport, embedder: Any, before: Any) -> None:
    """What every story of the scenario tests promises, over all the days at once."""
    from tests.support import life_checks as checks

    last = report.samples[-1].day if report.samples else report.first_day
    first_day = date.fromisoformat(report.first_day)
    calls = [
        ("screen and records agree", lambda: checks.assert_screen_matches_records(world)),
        ("nothing looks like an error", lambda: checks.assert_clean_screen(world)),
        ("silent in deep sleep", lambda: checks.assert_never_in_deep_sleep(world)),
        ("the platform's count", lambda: checks.assert_within_quota(world)),
        (
            "the proactive audit",
            lambda: checks.assert_proactive_rules(
                world, first_day, date.fromisoformat(last) - timedelta(days=0)
            ),
        ),
    ]
    for name, check in calls:
        try:
            check()
        except AssertionError as exc:
            report.invariants.append(f"{name}: {str(exc)[:200]}")
    try:
        await checks.assert_bot_text_stays_out(world, before, embedder)
    except AssertionError as exc:
        report.invariants.append(f"bot text stays out of her data: {str(exc)[:200]}")
    if world.deepseek.unexpected:
        report.invariants.append(f"unknown model requests: {world.deepseek.unexpected[:3]}")


async def run(args: argparse.Namespace, home: Path, log: Any) -> SoakReport:
    import time

    from sqlalchemy import select
    from tests.support.life_checks import snapshot_isolation
    from tests.support.life_env import life_environment
    from tests.support.proactive_world import opening_curve, proactive_model
    from tests.support.process_metrics import current_rss_bytes, megabytes

    from twin.storage.models import Alert

    first_day = date.fromisoformat(args.first_day)
    report = SoakReport(args.days, args.seed, args.platform, args.first_day)
    started = time.monotonic()
    tally = LogTally()
    logging.getLogger().addHandler(tally)
    lost: list[str] = []
    asyncio.get_running_loop().set_exception_handler(
        lambda _loop, context: lost.append(str(context.get("message")))
    )
    try:
        async with life_environment(
            home,
            start=world_start(first_day),
            model=proactive_model(opening_curve(base=0.02, peaks=PEAKS)),
            platform=args.platform,
        ) as env:
            world = env.world
            world.add_her_sticker("开心")
            before = snapshot_isolation(world)
            report.baseline_rss_mb = megabytes(current_rss_bytes())
            plans = plan_days(args.seed, first_day, args.days)
            rng = random.Random(f"{args.seed}/replies")  # noqa: S311
            if args.trace_memory:
                import tracemalloc

                tracemalloc.start(8)
            await live_through(
                world, plans, rng, tally, report, started, log, args.trace_memory, args.trim_heap
            )
            await world.run_for(hours=3)
            await world.run_until_idle()
            await world.drain_jobs(approval_ok=True)
            await check_promises(world, report, env.embedder, before)
            report.calls = dict(world.deepseek.calls)
            report.alerts = dict(Counter(category for category, _ in world.alerts()))
            report.jobs = dict(world.jobs())
            rows = world.rows()
            kinds = Counter(r.kind for r in world.proactive_rows(outcomes=["sent"]))
            report.summary = {
                "messages in": sum(1 for r in rows if r.direction == "in" and not r.is_command),
                "commands": sum(1 for r in rows if r.direction == "in" and r.is_command),
                "bubbles out": sum(1 for r in rows if r.direction == "out" and not r.is_command),
                **{f"proactive {kind}": count for kind, count in sorted(kinds.items())},
            }
            with world.services.db.session() as session:
                report.engine_errors = sum(
                    1 for a in session.scalars(select(Alert)) if a.category == "engine_error"
                )
    finally:
        logging.getLogger().removeHandler(tally)
    report.task_errors = len(lost)
    report.error_events = list(tally.events) + lost[:5]
    report.real_seconds = time.monotonic() - started
    return report


def world_start(day: date) -> datetime:
    """06:00 local time of ``day`` in the bot's zone (America/Chicago), as an instant in UTC."""
    from zoneinfo import ZoneInfo

    local = datetime(day.year, day.month, day.day, 6, tzinfo=ZoneInfo("America/Chicago"))
    return local.astimezone(UTC)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    from tests.support.life_env import isolate

    with tempfile.TemporaryDirectory(prefix="twin-soak-") as folder:
        home = Path(folder)
        isolate(home)
        report = asyncio.run(run(args, home, log))
    limits = Thresholds(rss_limit_mb=args.rss_limit_mb, rss_growth_mb_per_day=args.max_rss_growth)
    report.verdicts = evaluate(report, limits)
    text = render(report)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.json is not None:
        args.json.write_text(report_json(report) + "\n", encoding="utf-8")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
