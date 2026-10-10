"""Replaying the real history into the memory, day by day (R-MEM-010, R-LLM-014).

The real chat records were written long before the memory existed.  The replay walks them in
local-day order and does for each day what the running bot does for a day of its own: writes the
``real`` summary and extracts the facts and follow-ups.  Each fact's ``known_at`` is the time of
the latest message that proves it, so the memory at any past moment - ``memory_view(t)`` - holds
exactly what had been said by then.  The replay covers the whole history, **the hold-out period
included**: the blind test of round 09b needs the memory as it was at the moments it tests.

*Cost and consent* (R-LLM-014).  Replaying everything is a one-time batch: :func:`plan_replay`
prices it with the token estimator and the price table (peak prices: an upper bound), queues the
work as jobs that wait for ``twin jobs approve <batch>``, and the work is recorded on the
``one_time`` account of its batch, never on the daily budget.  Above 120 % of the estimate the
batch is paused (:mod:`twin.llm.onetime`).  A batch never exceeds ``budget.one_time_usd``; a
larger history is split into several batches.  An *incremental* replay - days added by a later
import - below ``memory.replay_auto_approve_ratio`` of that limit is approved on its own.

*Resuming.*  A day is marked done in ``memory_replay_days`` only after its summary, facts and
follow-ups are stored, together with a fingerprint of its messages.  A stopped replay is run
again and skips what is done; a day whose messages changed since is done again.

*Order.*  Days are processed in order inside a job, but jobs may run side by side, so nothing
depends on the order of processing: a fact's ``known_at`` decides which of two equal-ranked facts
is the newer (:mod:`twin.memory.conflict`).
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from twin.ingest.corpus import conversation_timeline, messages_between
from twin.ingest.transcript import message_text
from twin.llm.deepseek import DeepSeekClient
from twin.llm.errors import ApiError, BudgetDeniedError, CircuitOpenError, LlmError
from twin.llm.onetime import BatchItem, BatchStatus
from twin.llm.runtime import LlmRuntime, build_llm_runtime
from twin.llm.tokens import TokenEstimator
from twin.llm.types import LedgerTag
from twin.memory.dayload import LineHasher, lines_hash, real_day_lines
from twin.memory.extract import DialogueLine, FactExtractor
from twin.memory.followups import FollowupStore
from twin.memory.localdate import MemoryClock
from twin.memory.memory import Memory
from twin.memory.records import FollowupRecord
from twin.memory.store import MemoryStore
from twin.memory.summarize import DailySummarizer
from twin.memory.writer import MemoryWriter
from twin.ops.jobs import BatchTooLargeError, JobQueue
from twin.ops.logging import get_logger
from twin.profile.prompt_templates import (
    MEMORY_CONFLICT,
    MEMORY_EXTRACT,
    MEMORY_SUMMARY,
    TemplateStore,
    newest_file_template,
)
from twin.services import Services
from twin.storage.ids import new_id

log = get_logger("twin.memory.replay")

REPLAY_JOB = "memory_replay"
BATCH_PRIORITY = 120
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
FAR_FUTURE = datetime(2200, 1, 1, tzinfo=UTC)
ONE_MICROSECOND = timedelta(microseconds=1)
MESSAGE_OVERHEAD_TOKENS = 8
SUMMARY_COMPLETION_TOKENS = 450
EXTRACT_COMPLETION_TOKENS = 1800
CONFLICT_COMPLETION_TOKENS = 500
CONFLICT_ITEM_TOKENS = 90  # one candidate line of a conflict prompt
CONFLICT_BATCH = 6
FACTS_PER_DAY_BOUND = 12
CANDIDATES_BOUND = 6
OPEN_FOLLOWUPS_SHOWN = 10
CONTEXT_WINDOW_DAYS = 2  # a follow-up due within this many days after the day may be closed by it

STATE_NEW = "new"
STATE_CHANGED = "changed"
STATE_DONE = "done"


@dataclass(frozen=True)
class ReplayRequest:
    """Which days to replay: from/to are her local dates (inclusive); ``force`` redoes done days."""

    start: date | None = None
    end: date | None = None
    force: bool = False


@dataclass(frozen=True)
class DayStat:
    """One local day of the real history, as the planner sees it."""

    day: date
    lines: int
    tokens: int  # estimated tokens of the dialogue text
    input_hash: str
    state: str  # new | changed (messages differ from the replayed ones) | done


@dataclass(frozen=True)
class PlannedBatch:
    batch_id: str
    days: int
    jobs: int
    estimated_usd: float


@dataclass
class ReplayEstimate:
    """The price of replaying some days (nothing is queued by computing it)."""

    days: list[DayStat] = field(default_factory=list)  # the days that would be replayed
    already_done: int = 0
    first_replay: bool = True  # no day has been replayed before
    day_prices: list[float] = field(default_factory=list)  # USD per day in ``days``
    lines: int = 0
    tokens: int = 0

    @property
    def estimated_usd(self) -> float:
        return sum(self.day_prices)


@dataclass
class ReplayPlan:
    """What :func:`plan_replay` queued."""

    estimate: ReplayEstimate
    batches: list[PlannedBatch] = field(default_factory=list)
    approved: bool = False
    already_queued: int = 0  # days left out because a waiting job has them

    @property
    def estimated_usd(self) -> float:
        return self.estimate.estimated_usd

    @property
    def jobs(self) -> int:
        return sum(b.jobs for b in self.batches)


# ------------------------------------------------------------------------ scanning


def scan_days(
    services: Services,
    clock: MemoryClock,
    request: ReplayRequest,
    estimator: TokenEstimator,
) -> list[DayStat]:
    """Every local day with messages in the request, with its size and fingerprint."""
    low = clock.real_bounds(request.start)[0] if request.start else EPOCH
    high = clock.real_bounds(request.end)[1] - ONE_MICROSECOND if request.end else FAR_FUTURE
    replayed = _replayed_days(services)
    stats: list[DayStat] = []
    current: date | None = None
    hasher = LineHasher()
    lines = tokens = 0

    def close() -> None:
        if current is None or not lines:
            return
        digest = hasher.hexdigest()
        done = replayed.get(current)
        state = STATE_NEW if done is None else (STATE_DONE if done == digest else STATE_CHANGED)
        stats.append(DayStat(current, lines, tokens, digest, state))

    with services.db.session() as session:
        stmt = messages_between(low, high).execution_options(yield_per=5000)
        for row in session.scalars(stmt):
            text = message_text(row)
            if not text:
                continue
            day = clock.real_date(row.create_time_utc)
            if day != current:
                close()
                current, hasher, lines, tokens = day, LineHasher(), 0, 0
            hasher.update(
                DialogueLine(row.id, "user" if row.is_sent else "her", text, row.create_time_utc)
            )
            lines += 1
            tokens += estimator.estimate_text(text) + MESSAGE_OVERHEAD_TOKENS
    close()
    return stats


def _replayed_days(services: Services) -> dict[date, str]:
    return {day: rec.input_hash for day, rec in MemoryStore(services).replay_days().items()}


# ------------------------------------------------------------------------ pricing


def price_days(services: Services, runtime: LlmRuntime, days: Sequence[DayStat]) -> list[float]:
    """USD per day at peak prices: the summary, the extraction and the conflict calls.

    An upper bound: the conflict calls are counted for the most facts a day may yield, each with
    the most candidates, and every call is priced without cache hits.
    """
    estimator = runtime.estimator
    model = services.settings.deepseek.offline_model
    summary_base = estimator.estimate_messages(
        newest_file_template(MEMORY_SUMMARY).render(
            scope_note="", date="2026-01-01", weekday="周一", zone="UTC", count=0, dialogue=""
        )
    )
    extract_base = estimator.estimate_messages(
        newest_file_template(MEMORY_EXTRACT).render(
            max_facts=FACTS_PER_DAY_BOUND,
            zone="UTC",
            reference="2026-01-01 00:00",
            open_followups="（没有）" + "字" * OPEN_FOLLOWUPS_SHOWN * 40,
            count=0,
            dialogue="",
        )
    )
    conflict_base = estimator.estimate_messages(
        newest_file_template(MEMORY_CONFLICT).render(items="")
    )
    chunk_lines = services.settings.memory.replay_chunk_lines
    conflict_calls = math.ceil(FACTS_PER_DAY_BOUND / CONFLICT_BATCH)
    conflict_prompt = conflict_base + CONFLICT_BATCH * (1 + CANDIDATES_BOUND) * CONFLICT_ITEM_TOKENS
    prices: list[float] = []
    for day in days:
        chunks = max(1, math.ceil(day.lines / chunk_lines))
        items = [
            BatchItem(model, summary_base + day.tokens // chunks, SUMMARY_COMPLETION_TOKENS)
            for _ in range(chunks)
        ]
        if chunks > 1:  # the pieces are merged by one more call
            items.append(
                BatchItem(
                    model,
                    summary_base + chunks * SUMMARY_COMPLETION_TOKENS,
                    SUMMARY_COMPLETION_TOKENS,
                )
            )
        items += [
            BatchItem(model, extract_base + day.tokens // chunks, EXTRACT_COMPLETION_TOKENS)
            for _ in range(chunks)
        ]
        items += [
            BatchItem(model, conflict_prompt, CONFLICT_COMPLETION_TOKENS)
            for _ in range(conflict_calls * chunks)
        ]
        prices.append(runtime.batches.estimate(items).total_usd)
    return prices


def estimate_replay(
    services: Services,
    request: ReplayRequest | None = None,
    *,
    runtime: LlmRuntime | None = None,
    clock: MemoryClock | None = None,
) -> ReplayEstimate:
    """Which days a replay would do and what it would cost (reads only)."""
    ask = request or ReplayRequest()
    llm = runtime or build_llm_runtime(services)
    calendar = clock or MemoryClock.from_services(services)
    stats = scan_days(services, calendar, ask, llm.estimator)
    pending = [d for d in stats if ask.force or d.state != STATE_DONE]
    estimate = ReplayEstimate(
        days=pending,
        already_done=len(stats) - len(pending),
        first_replay=not _replayed_days(services),
        lines=sum(d.lines for d in pending),
        tokens=sum(d.tokens for d in pending),
    )
    estimate.day_prices = price_days(services, llm, pending)
    return estimate


# ------------------------------------------------------------------------ planning


def queued_days(queue: JobQueue) -> set[date]:
    """Days that wait in a pending or running replay job."""
    found: set[date] = set()
    for status in ("pending", "running"):
        for job in queue.list_jobs(status=status, job_type=REPLAY_JOB, limit=100_000):
            found.update(date.fromisoformat(str(d)) for d in job.payload.get("dates", []))
    return found


def group_jobs(
    days: Sequence[DayStat], prices: Sequence[float], *, per_job: int, limit_usd: float
) -> list[list[int]]:
    """Indices of days grouped into jobs of at most ``per_job`` days and ``limit_usd``."""
    jobs: list[list[int]] = [[]]
    running = 0.0
    for index, price in enumerate(prices):
        if price > limit_usd:
            raise BatchTooLargeError(f"day {days[index].day}", price, limit_usd)
        if jobs[-1] and (len(jobs[-1]) >= per_job or running + price > limit_usd):
            jobs.append([])
            running = 0.0
        jobs[-1].append(index)
        running += price
    return [job for job in jobs if job]


def plan_replay(
    services: Services,
    request: ReplayRequest | None = None,
    *,
    runtime: LlmRuntime | None = None,
    auto_approve: bool = False,
    reason: str = "manual",
) -> ReplayPlan:
    """Queue the replay of the days that need it as one-time batches (R-LLM-014).

    The jobs wait for ``twin jobs approve <batch>``.  With ``auto_approve`` the batches are
    approved at once if the replay is incremental (days were replayed before) and its estimate
    is below ``memory.replay_auto_approve_ratio`` of ``budget.one_time_usd``.
    """
    ask = request or ReplayRequest()
    llm = runtime or build_llm_runtime(services)
    estimate = estimate_replay(services, ask, runtime=llm)
    queue = JobQueue(services.db, services.clock)
    waiting = queued_days(queue)
    keep = [i for i, d in enumerate(estimate.days) if d.day not in waiting]
    plan = ReplayPlan(
        ReplayEstimate(
            days=[estimate.days[i] for i in keep],
            already_done=estimate.already_done,
            first_replay=estimate.first_replay,
            day_prices=[estimate.day_prices[i] for i in keep],
            lines=sum(estimate.days[i].lines for i in keep),
            tokens=sum(estimate.days[i].tokens for i in keep),
        ),
        already_queued=len(estimate.days) - len(keep),
    )
    days, prices = plan.estimate.days, plan.estimate.day_prices
    if not days:
        return plan
    config = services.settings.memory
    limit = llm.batches.limit_usd
    jobs = group_jobs(days, prices, per_job=config.replay_job_days, limit_usd=limit)
    job_prices = [sum(prices[i] for i in job) for job in jobs]
    groups: list[list[int]] = [[]]
    running = 0.0
    for number, price in enumerate(job_prices):
        if groups[-1] and running + price > limit:
            groups.append([])
            running = 0.0
        groups[-1].append(number)
        running += price
    stamp = f"{services.clock.now_utc():%Y%m%d%H%M%S}-{new_id()[-4:].lower()}"
    for number, group in enumerate(groups, start=1):
        batch_id = f"memory-{stamp}-{number}"
        payloads = [
            {
                "dates": [days[i].day.isoformat() for i in jobs[j]],
                "batch_id": batch_id,
                "force": ask.force,
                "reason": reason,
            }
            for j in group
        ]
        llm.batches.enqueue(
            batch_id,
            REPLAY_JOB,
            payloads,
            [job_prices[j] for j in group],
            priority=BATCH_PRIORITY,
            offpeak_only=True,
        )
        plan.batches.append(
            PlannedBatch(
                batch_id,
                sum(len(jobs[j]) for j in group),
                len(group),
                sum(job_prices[j] for j in group),
            )
        )
    small = plan.estimated_usd < config.replay_auto_approve_ratio * limit
    if auto_approve and not estimate.first_replay and small:
        for batch in plan.batches:
            llm.batches.approve(batch.batch_id)
        plan.approved = True
    return plan


# ------------------------------------------------------------------------ running


@dataclass
class DayResult:
    day: date
    status: str  # replayed | skipped | empty
    facts: int = 0
    followups: int = 0
    summarised: bool = False


@dataclass
class ReplayRun:
    """What a run over some days did."""

    results: list[DayResult] = field(default_factory=list)
    failures: list[date] = field(default_factory=list)

    @property
    def replayed(self) -> int:
        return sum(1 for r in self.results if r.status == "replayed")

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == "skipped")

    @property
    def facts(self) -> int:
        return sum(r.facts for r in self.results)


class MemoryReplayer:
    """Replays days of the real history (see the module description)."""

    def __init__(self, memory: Memory, client: DeepSeekClient) -> None:
        self._memory = memory
        self._store = memory.store
        services = memory.services
        templates = TemplateStore(services.db, services.clock)
        self._summarizer = DailySummarizer(memory, client, templates=templates)
        self._extractor = FactExtractor(services, client, memory.clock, templates=templates)
        self._writer = MemoryWriter(memory, client, templates=templates)
        self._followups = FollowupStore(memory)

    async def replay_day(
        self, day: date, tag: LedgerTag, *, force: bool = False, batch_id: str | None = None
    ) -> DayResult:
        services = self._memory.services
        lines = await asyncio.to_thread(real_day_lines, services, self._memory.clock, day)
        digest = lines_hash(lines)
        done = (await asyncio.to_thread(self._store.replay_days)).get(day)
        if done is not None and done.input_hash == digest and not force:
            return DayResult(day, "skipped")
        if not lines:
            await asyncio.to_thread(self._mark, day, digest, 0, 0, 0, None, batch_id)
            return DayResult(day, "empty")
        summary = await self._summarizer.summarize("real", day, lines, tag, force=force)
        facts = followups = 0
        if self._extractor.enabled:
            open_followups = await asyncio.to_thread(self._open_around, day)
            extraction = await self._extractor.extract(
                lines, mode="real", tag=tag, open_followups=open_followups
            )
            report = await self._writer.write(extraction, tag)
            facts, followups = report.facts_added, report.followups_added
        await asyncio.to_thread(self._expire, day)
        version = summary.record.version if summary.record else None
        await asyncio.to_thread(
            self._mark, day, digest, len(lines), facts, followups, version, batch_id
        )
        return DayResult(day, "replayed", facts, followups, summary.record is not None)

    def _mark(
        self,
        day: date,
        digest: str,
        lines: int,
        facts: int,
        followups: int,
        version: int | None,
        batch_id: str | None,
    ) -> None:
        self._store.mark_replayed(
            day,
            input_hash=digest,
            lines=lines,
            facts_added=facts,
            followups_added=followups,
            summary_version=version,
            batch_id=batch_id,
        )

    def _open_around(self, day: date) -> list[FollowupRecord]:
        """Open follow-ups this day's conversation may show to be over."""
        start, end = self._memory.clock.real_bounds(day)
        horizon = end + timedelta(days=CONTEXT_WINDOW_DAYS)
        found = [
            f
            for f in self._followups.open()
            if f.created_at < end and f.window_end >= start and f.due_at <= horizon
        ]
        return found[:OPEN_FOLLOWUPS_SHOWN]

    def _expire(self, day: date) -> None:
        """Follow-ups whose window ended by the end of the day and that nobody raised."""
        self._followups.expire_overdue(self._memory.clock.real_bounds(day)[1])

    async def replay_days(
        self,
        days: Iterable[date],
        tag: LedgerTag,
        *,
        force: bool = False,
        batch_id: str | None = None,
        on_day: Callable[[DayResult], None] | None = None,
    ) -> ReplayRun:
        """Replay days in order.  A day that fails for a reason of its own is noted and the run
        goes on; running out of budget, an open circuit breaker or a paused batch stops it."""
        run = ReplayRun()
        for day in sorted(set(days)):
            try:
                result = await self.replay_day(day, tag, force=force, batch_id=batch_id)
            except (BudgetDeniedError, CircuitOpenError):
                raise
            except ApiError:
                raise
            except LlmError as exc:
                log.warning("replay_day_failed", day=day.isoformat(), error=type(exc).__name__)
                run.failures.append(day)
                continue
            run.results.append(result)
            if on_day is not None:
                on_day(result)
        return run


# ------------------------------------------------------------------------ status


@dataclass
class ReplayStatus:
    """How far the replay is (``twin memory replay status``)."""

    days_with_messages: int
    days_replayed: int
    batches: list[BatchStatus]
    jobs: dict[str, int]
    facts_by_source: dict[str, int]
    summaries: int
    followups_open: int

    @property
    def days_waiting(self) -> int:
        return max(0, self.days_with_messages - self.days_replayed)


def count_message_days(services: Services, clock: MemoryClock) -> set[date]:
    """The local days that have a message other than a system notice."""
    days: set[date] = set()
    with services.db.session() as session:
        rows = session.execute(conversation_timeline().execution_options(yield_per=20_000))
        for moment, _is_sent, kind in rows:
            if kind != "system":
                days.add(clock.real_date(moment))
    return days


def replay_status(
    services: Services, *, runtime: LlmRuntime | None = None, clock: MemoryClock | None = None
) -> ReplayStatus:
    """Progress of the replay: days done, the state of each batch, what the memory holds."""
    llm = runtime or build_llm_runtime(services)
    calendar = clock or MemoryClock.from_services(services)
    memory = Memory(services)
    queue = JobQueue(services.db, services.clock)
    jobs = {"pending": 0, "running": 0, "done": 0, "failed": 0, "cancelled": 0}
    batch_ids: dict[str, None] = {}
    for status in jobs:
        found = queue.list_jobs(status=status, job_type=REPLAY_JOB, limit=100_000)
        jobs[status] = len(found)
        for job in found:
            if job.batch_id:
                batch_ids.setdefault(job.batch_id, None)
    counts = memory.store.counts()
    return ReplayStatus(
        days_with_messages=len(count_message_days(services, calendar)),
        days_replayed=counts["memory_replay_days"],
        batches=[llm.batches.status(batch_id) for batch_id in sorted(batch_ids)],
        jobs=jobs,
        facts_by_source=memory.store.fact_counts_by_source(),
        summaries=len(memory.store.current_summaries()),
        followups_open=len(memory.store.followups(only_open=True)),
    )
