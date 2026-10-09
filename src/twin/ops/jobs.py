"""Persistent job queue and worker (R-ARCH-003).

* :class:`JobQueue` is the repository over the ``jobs`` table (payloads sealed).
* Handlers are registered by job type with ``@job_handler("type")``.  A job whose
  type has no registered handler simply stays ``pending`` (``twin jobs list``
  marks it "no handler"); that is normal queueing, e.g. a job enqueued by a later
  round's code before its handler module is loaded.
* :class:`Worker` runs handlers with bounded concurrency, retries failures with
  exponential backoff, marks a job ``failed`` and raises an alert once its
  attempts are used up, and after a restart returns ``running`` jobs to
  ``pending`` (:meth:`Worker.recover`).
* Off-peak scheduling is decided by an injected :class:`OffPeakPolicy`.  The
  production policy is registered by round 01 (R-LLM-007); until then
  :class:`DeferredOffPeakPolicy` is in force, under which ``offpeak_only`` jobs
  wait for their ``deadline``.  This dependency is recorded in TRACEABILITY.
* One-time batches (R-LLM-014): jobs may carry ``batch_id``,
  ``estimated_cost_usd`` and ``requires_approval``; an unapproved job is never
  claimed.  ``twin jobs approve`` records the approval.  Estimation logic is
  provided by the business rounds.
"""

from __future__ import annotations

import asyncio
import os
import random
import uuid
from collections.abc import Awaitable, Callable, Collection, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol

from sqlalchemy import and_, func, or_, select

from twin.clock import Clock
from twin.llm.redaction import redact_text
from twin.ops.alerts import AlertSink
from twin.ops.logging import get_logger
from twin.storage.db import Database
from twin.storage.models import JOB_STATUSES, Job

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.jobs")

DEFAULT_PRIORITY = 100
MAX_ERROR_CHARS = 500


class JobError(RuntimeError):
    """Base class for job queue errors."""


class JobDeferred(Exception):
    """Raised by a handler to hand its job back unchanged (no attempt is counted).

    Used when the work cannot continue for a reason that is neither a failure nor a
    completion, for example when the job's one-time batch was paused for overspending
    (R-LLM-014) or the budget level forbids the work until the next day (R-LLM-008).
    The job is claimable again after ``retry_in_s`` seconds.
    """

    def __init__(self, reason: str, *, retry_in_s: float = 300.0) -> None:
        super().__init__(reason)
        self.retry_in_s = retry_in_s


class BatchNotFoundError(JobError):
    """No job awaiting approval belongs to the batch."""


class BatchTooLargeError(JobError):
    """The batch estimate exceeds the one-time budget (R-LLM-014)."""

    def __init__(self, batch_id: str, estimate: float, limit: float) -> None:
        self.estimate = estimate
        self.limit = limit
        super().__init__(
            f"batch {batch_id} is estimated at ${estimate:.2f}, above the one-time limit "
            f"${limit:.2f}; split it into smaller batches"
        )


# ----------------------------------------------------------------- policy


class OffPeakPolicy(Protocol):
    """Decides whether ``offpeak_only`` jobs may run now."""

    def allows(self, now: datetime) -> bool: ...


class DeferredOffPeakPolicy:
    """In force until round 01 registers the production policy: off-peak jobs wait."""

    def allows(self, now: datetime) -> bool:
        return False


_offpeak_policy: OffPeakPolicy = DeferredOffPeakPolicy()


def set_offpeak_policy(policy: OffPeakPolicy) -> OffPeakPolicy:
    """Register the off-peak policy; returns the previous one."""
    global _offpeak_policy
    previous = _offpeak_policy
    _offpeak_policy = policy
    return previous


def get_offpeak_policy() -> OffPeakPolicy:
    return _offpeak_policy


# --------------------------------------------------------------- data types


@dataclass(frozen=True)
class JobView:
    """Immutable snapshot of a job row (payload already decrypted)."""

    id: str
    type: str
    payload: Any
    priority: int
    status: str
    attempts: int
    max_attempts: int
    run_after: datetime
    offpeak_only: bool
    deadline: datetime | None
    last_error: str | None
    batch_id: str | None
    estimated_cost_usd: float | None
    requires_approval: bool
    approved_at: datetime | None
    approved_usd: float | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class BatchApproval:
    batch_id: str
    job_count: int
    total_usd: float


@dataclass(frozen=True)
class BatchSummary:
    batch_id: str
    jobs: int
    awaiting_approval: int
    estimated_usd: float
    approved_usd: float


def _view(job: Job) -> JobView:
    return JobView(
        id=job.id,
        type=job.type,
        payload=job.payload,
        priority=job.priority,
        status=job.status,
        attempts=job.attempts,
        max_attempts=job.max_attempts,
        run_after=job.run_after,
        offpeak_only=job.offpeak_only,
        deadline=job.deadline,
        last_error=job.last_error,
        batch_id=job.batch_id,
        estimated_cost_usd=job.estimated_cost_usd,
        requires_approval=job.requires_approval,
        approved_at=job.approved_at,
        approved_usd=job.approved_usd,
        started_at=job.started_at,
        finished_at=job.finished_at,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


# -------------------------------------------------------------------- queue


class JobQueue:
    """Repository over the ``jobs`` table."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def enqueue(
        self,
        job_type: str,
        payload: Any,
        *,
        priority: int = DEFAULT_PRIORITY,
        max_attempts: int = 3,
        run_after: datetime | None = None,
        offpeak_only: bool = False,
        deadline: datetime | None = None,
        batch_id: str | None = None,
        estimated_cost_usd: float | None = None,
        requires_approval: bool = False,
    ) -> str:
        """Add a job and return its id."""
        if not job_type:
            raise ValueError("job type must not be empty")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if requires_approval and not batch_id:
            raise ValueError("a job that requires approval must belong to a batch")
        now = self._clock.now_utc()
        job = Job(
            type=job_type,
            payload=payload,
            priority=priority,
            max_attempts=max_attempts,
            run_after=run_after or now,
            offpeak_only=offpeak_only,
            deadline=deadline,
            batch_id=batch_id,
            estimated_cost_usd=estimated_cost_usd,
            requires_approval=requires_approval,
            created_at=now,
            updated_at=now,
        )
        with self._db.transaction() as session:
            session.add(job)
            job_id = job.id
        return job_id

    def get(self, job_id: str) -> JobView | None:
        with self._db.session() as session:
            job = session.get(Job, job_id)
            return _view(job) if job else None

    def list_jobs(
        self,
        *,
        status: str | None = None,
        job_type: str | None = None,
        batch_id: str | None = None,
        limit: int = 50,
    ) -> list[JobView]:
        stmt = select(Job).order_by(Job.created_at.desc(), Job.id.desc()).limit(limit)
        if status is not None:
            if status not in JOB_STATUSES:
                raise ValueError(f"unknown job status {status!r}")
            stmt = stmt.where(Job.status == status)
        if job_type is not None:
            stmt = stmt.where(Job.type == job_type)
        if batch_id is not None:
            stmt = stmt.where(Job.batch_id == batch_id)
        with self._db.session() as session:
            return [_view(job) for job in session.execute(stmt).scalars()]

    def counts(self) -> dict[str, int]:
        with self._db.session() as session:
            rows = session.execute(select(Job.status, func.count()).group_by(Job.status)).all()
        counts = dict.fromkeys(JOB_STATUSES, 0)
        for status, count in rows:
            counts[status] = count
        return counts

    # -- claiming and completion (used by the worker) ---------------------

    def claim_next(
        self, handled_types: Collection[str], *, offpeak_allowed: bool, worker_id: str
    ) -> JobView | None:
        """Atomically take the most urgent runnable job, or ``None``."""
        if not handled_types:
            return None
        now = self._clock.now_utc()
        with self._db.transaction(bump_state=False) as session:
            stmt = select(Job).where(
                Job.status == "pending",
                Job.run_after <= now,
                Job.type.in_(sorted(handled_types)),
                or_(Job.requires_approval.is_(False), Job.approved_at.is_not(None)),
            )
            if not offpeak_allowed:
                stmt = stmt.where(
                    or_(
                        Job.offpeak_only.is_(False),
                        and_(Job.deadline.is_not(None), Job.deadline <= now),
                    )
                )
            stmt = stmt.order_by(Job.priority, Job.run_after, Job.created_at, Job.id).limit(1)
            job = session.execute(stmt).scalars().first()
            if job is None:
                return None
            job.status = "running"
            job.attempts += 1
            job.started_at = now
            job.locked_by = worker_id
            session.flush()
            return _view(job)

    def _finish(
        self,
        job_id: str,
        worker_id: str,
        *,
        status: str,
        error: str | None,
        retry_at: datetime | None,
    ) -> str | None:
        now = self._clock.now_utc()
        with self._db.transaction(bump_state=False) as session:
            job = session.get(Job, job_id)
            if job is None or job.status != "running" or job.locked_by != worker_id:
                return None  # cancelled or recovered meanwhile: discard this result
            job.status = status
            job.locked_by = None
            if error is not None:
                job.last_error = error
            if retry_at is not None:
                job.run_after = retry_at
            else:
                job.finished_at = now
            return job.status

    def complete(self, job_id: str, worker_id: str) -> bool:
        return self._finish(job_id, worker_id, status="done", error=None, retry_at=None) is not None

    def fail(
        self, job_id: str, worker_id: str, error: str, *, retry_at: datetime | None
    ) -> str | None:
        """Record a failure: retry at ``retry_at`` or, if ``None``, mark ``failed``."""
        status = "pending" if retry_at is not None else "failed"
        return self._finish(job_id, worker_id, status=status, error=error, retry_at=retry_at)

    def release(self, job_id: str, worker_id: str, *, run_after: datetime | None = None) -> bool:
        """Give back a job without counting the attempt (shutdown, or ``JobDeferred``)."""
        with self._db.transaction(bump_state=False) as session:
            job = session.get(Job, job_id)
            if job is None or job.status != "running" or job.locked_by != worker_id:
                return False
            job.status = "pending"
            job.locked_by = None
            job.attempts = max(0, job.attempts - 1)
            if run_after is not None:
                job.run_after = run_after
            return True

    def recover_running(self) -> tuple[int, list[JobView]]:
        """After a restart: ``running`` -> ``pending`` (or ``failed`` if attempts are used up).

        Returns ``(requeued_count, newly_failed_jobs)``.
        """
        requeued = 0
        failed: list[JobView] = []
        now = self._clock.now_utc()
        with self._db.transaction(bump_state=False) as session:
            for job in session.execute(select(Job).where(Job.status == "running")).scalars():
                job.locked_by = None
                if job.attempts >= job.max_attempts:
                    job.status = "failed"
                    job.finished_at = now
                    job.last_error = "process stopped while the job was running; attempts exhausted"
                    failed.append(_view(job))
                else:
                    job.status = "pending"
                    requeued += 1
        return requeued, failed

    # -- user actions ------------------------------------------------------

    def retry(self, job_id: str) -> bool:
        """Put a failed or cancelled job back in the queue with a fresh attempt budget."""
        now = self._clock.now_utc()
        with self._db.transaction() as session:
            job = session.get(Job, job_id)
            if job is None or job.status not in ("failed", "cancelled"):
                return False
            job.status = "pending"
            job.attempts = 0
            job.run_after = now
            job.finished_at = None
            job.last_error = None
            return True

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending or running job (a running handler's result is discarded)."""
        now = self._clock.now_utc()
        with self._db.transaction() as session:
            job = session.get(Job, job_id)
            if job is None or job.status not in ("pending", "running"):
                return False
            job.status = "cancelled"
            job.locked_by = None
            job.finished_at = now
            return True

    def batch_summary(self, batch_id: str) -> BatchSummary:
        with self._db.session() as session:
            jobs = [
                _view(job)
                for job in session.execute(select(Job).where(Job.batch_id == batch_id)).scalars()
            ]
        if not jobs:
            raise BatchNotFoundError(f"no jobs belong to batch {batch_id!r}")
        return BatchSummary(
            batch_id=batch_id,
            jobs=len(jobs),
            awaiting_approval=sum(
                1
                for j in jobs
                if j.requires_approval and j.approved_at is None and j.status == "pending"
            ),
            estimated_usd=sum(j.estimated_cost_usd or 0.0 for j in jobs),
            approved_usd=sum(j.approved_usd or 0.0 for j in jobs),
        )

    def revoke_approval(self, batch_id: str) -> int:
        """Withdraw the approval of the unfinished jobs of a batch (pauses it, R-LLM-014).

        The jobs stay in the queue; they are claimed again only after ``approve_batch``.
        Returns the number of jobs affected.
        """
        with self._db.transaction() as session:
            jobs = list(
                session.execute(
                    select(Job).where(
                        Job.batch_id == batch_id,
                        Job.requires_approval.is_(True),
                        Job.status.in_(("pending", "running")),
                    )
                ).scalars()
            )
            for job in jobs:
                job.approved_at = None
                job.approved_usd = None
            return len(jobs)

    def approve_batch(self, batch_id: str, *, max_usd: float) -> BatchApproval:
        """Approve every job of ``batch_id`` that is waiting (R-LLM-014)."""
        now = self._clock.now_utc()
        with self._db.transaction() as session:
            waiting = list(
                session.execute(
                    select(Job).where(
                        Job.batch_id == batch_id,
                        Job.requires_approval.is_(True),
                        Job.approved_at.is_(None),
                        Job.status == "pending",
                    )
                ).scalars()
            )
            if not waiting:
                raise BatchNotFoundError(f"no job of batch {batch_id!r} is waiting for approval")
            total = sum(job.estimated_cost_usd or 0.0 for job in waiting)
            if total > max_usd:
                raise BatchTooLargeError(batch_id, total, max_usd)
            for job in waiting:
                job.approved_at = now
                job.approved_usd = job.estimated_cost_usd or 0.0
            return BatchApproval(batch_id, len(waiting), total)


# --------------------------------------------------------------- handlers


@dataclass(frozen=True)
class JobContext:
    """What a handler receives."""

    job: JobView
    services: Services | None
    clock: Clock


JobHandler = Callable[[JobContext], Awaitable[None]]


class HandlerRegistry:
    """Maps job types to handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, JobHandler] = {}

    def register(self, job_type: str, handler: JobHandler, *, replace: bool = False) -> JobHandler:
        if job_type in self._handlers and not replace:
            raise ValueError(f"a handler for job type {job_type!r} is already registered")
        self._handlers[job_type] = handler
        return handler

    def get(self, job_type: str) -> JobHandler | None:
        return self._handlers.get(job_type)

    def has(self, job_type: str) -> bool:
        return job_type in self._handlers

    def types(self) -> frozenset[str]:
        return frozenset(self._handlers)


default_registry = HandlerRegistry()

# Modules that register job handlers on import; each business round appends its module.
HANDLER_MODULES: tuple[str, ...] = (
    "twin.ingest.jobs",
    "twin.ingest.captions",
    "twin.stickers.download",
    "twin.profile.jobs",
    "twin.retrieval.jobs",
    "twin.profile.persona.jobs",
    "twin.stickers.tag_jobs",
    "twin.memory.jobs",
    "twin.schedule.jobs",
    "twin.engine.turns",  # registers the reader of the bot's conversation (R-MEM-001)
)


def job_handler(
    job_type: str, *, registry: HandlerRegistry | None = None
) -> Callable[[JobHandler], JobHandler]:
    """Decorator registering ``async def handler(ctx: JobContext)`` for ``job_type``."""

    def decorate(func: JobHandler) -> JobHandler:
        (registry or default_registry).register(job_type, func)
        return func

    return decorate


def load_handlers(modules: Iterable[str] | None = None) -> HandlerRegistry:
    """Import every handler module so that the default registry is complete."""
    import importlib

    for name in modules if modules is not None else HANDLER_MODULES:
        importlib.import_module(name)
    return default_registry


# ------------------------------------------------------------------ worker


@dataclass
class RunSummary:
    done: int = 0
    retried: int = 0
    failed: int = 0
    discarded: int = 0  # finished after being cancelled
    deferred: int = 0  # handed back by the handler (JobDeferred)
    failures: list[str] = field(default_factory=list)


def backoff_delay(attempts: int, base_s: float, cap_s: float) -> float:
    """Exponential backoff: base, 2*base, 4*base ... capped (no jitter)."""
    return float(min(cap_s, base_s * (2 ** max(0, attempts - 1))))


def describe_failure(exc: BaseException) -> str:
    """Redacted, truncated error text safe to store and show (no chat content)."""
    return f"{type(exc).__name__}: {redact_text(str(exc))}"[:MAX_ERROR_CHARS]


class Worker:
    """Executes queued jobs with bounded concurrency."""

    def __init__(
        self,
        queue: JobQueue,
        registry: HandlerRegistry,
        clock: Clock,
        *,
        services: Services | None = None,
        offpeak: OffPeakPolicy | None = None,
        alerts: AlertSink | None = None,
        concurrency: int = 2,
        poll_interval_s: float = 2.0,
        backoff_base_s: float = 30.0,
        backoff_cap_s: float = 3600.0,
        jitter_ratio: float = 0.0,
        rng: random.Random | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self._queue = queue
        self._registry = registry
        self._clock = clock
        self._services = services
        self._offpeak = offpeak
        self._alerts = alerts
        self._concurrency = concurrency
        self._poll_interval_s = poll_interval_s
        self._backoff_base_s = backoff_base_s
        self._backoff_cap_s = backoff_cap_s
        self._jitter_ratio = jitter_ratio
        self._rng = rng or random.Random()  # noqa: S311 - retry jitter, not security
        self._worker_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._running: set[asyncio.Task[None]] = set()
        self._summary = RunSummary()

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def running_count(self) -> int:
        return len(self._running)

    def _policy(self) -> OffPeakPolicy:
        return self._offpeak or get_offpeak_policy()

    def retry_delay(self, attempts: int) -> float:
        delay = backoff_delay(attempts, self._backoff_base_s, self._backoff_cap_s)
        if self._jitter_ratio:
            delay *= 1 + self._jitter_ratio * (self._rng.random() * 2 - 1)
        return max(0.0, delay)

    async def recover(self) -> int:
        """Return jobs left ``running`` by a previous process to ``pending``."""
        requeued, failed = await asyncio.to_thread(self._queue.recover_running)
        for job in failed:
            self._raise_failure_alert(job, job.last_error or "attempts exhausted")
        if requeued or failed:
            log.warning("jobs_recovered", requeued=requeued, failed=len(failed))
        return requeued

    async def _claim(self) -> JobView | None:
        offpeak_allowed = self._policy().allows(self._clock.now_utc())
        return await asyncio.to_thread(
            self._queue.claim_next,
            self._registry.types(),
            offpeak_allowed=offpeak_allowed,
            worker_id=self._worker_id,
        )

    async def _fill_slots(self) -> int:
        started = 0
        self._running -= {task for task in self._running if task.done()}
        while len(self._running) < self._concurrency:
            job = await self._claim()
            if job is None:
                break
            task = asyncio.create_task(self._execute(job), name=f"job-{job.type}-{job.id}")
            self._running.add(task)
            task.add_done_callback(self._running.discard)
            started += 1
        return started

    async def _execute(self, job: JobView) -> None:
        handler = self._registry.get(job.type)
        log.info("job_started", job_id=job.id, job_type=job.type, attempt=job.attempts)
        try:
            if handler is None:  # the registry changed after the claim
                raise JobError(f"no handler registered for job type {job.type!r}")
            await handler(JobContext(job=job, services=self._services, clock=self._clock))
        except asyncio.CancelledError:
            await asyncio.shield(asyncio.to_thread(self._queue.release, job.id, self._worker_id))
            raise
        except JobDeferred as deferred:
            retry_at = self._clock.now_utc() + timedelta(seconds=deferred.retry_in_s)
            await asyncio.to_thread(
                partial(self._queue.release, job.id, self._worker_id, run_after=retry_at)
            )
            self._summary.deferred += 1
            log.info("job_deferred", job_id=job.id, job_type=job.type, reason=str(deferred)[:200])
        except Exception as exc:
            await self._record_failure(job, exc)
        else:
            recorded = await asyncio.to_thread(self._queue.complete, job.id, self._worker_id)
            if recorded:
                self._summary.done += 1
                log.info("job_done", job_id=job.id, job_type=job.type)
            else:
                self._summary.discarded += 1

    async def _record_failure(self, job: JobView, exc: Exception) -> None:
        error = describe_failure(exc)
        if job.attempts < job.max_attempts:
            retry_at = self._clock.now_utc() + timedelta(seconds=self.retry_delay(job.attempts))
            outcome = await asyncio.to_thread(
                self._queue.fail, job.id, self._worker_id, error, retry_at=retry_at
            )
            if outcome is not None:
                self._summary.retried += 1
                log.warning(
                    "job_retry_scheduled",
                    job_id=job.id,
                    job_type=job.type,
                    attempt=job.attempts,
                    retry_at=retry_at.isoformat(),
                    error=error,
                )
            return
        outcome = await asyncio.to_thread(
            self._queue.fail, job.id, self._worker_id, error, retry_at=None
        )
        if outcome is not None:
            self._summary.failed += 1
            self._summary.failures.append(job.id)
            log.error("job_failed", job_id=job.id, job_type=job.type, error=error)
            self._raise_failure_alert(job, error)

    def _raise_failure_alert(self, job: JobView, error: str) -> None:
        if self._alerts is None:
            return
        self._alerts.raise_alert(
            "job_failed",
            f"job {job.type} failed after {job.attempts} attempts",
            severity="warning",
            detail={"job_id": job.id, "job_type": job.type, "error": error},
            dedup_key=f"job_failed:{job.type}",
        )

    async def run_until_idle(self) -> RunSummary:
        """Run until no job is runnable *now* and nothing is still executing."""
        self._summary = RunSummary()
        while True:
            started = await self._fill_slots()
            if not self._running:
                if started == 0:
                    return self._summary
                continue
            await asyncio.wait(set(self._running), return_when=asyncio.FIRST_COMPLETED)

    async def run_forever(self, stop: asyncio.Event) -> None:
        """Poll for work until ``stop`` is set, then wait for running jobs to finish."""
        await self.recover()
        stop_waiter = asyncio.ensure_future(stop.wait())
        try:
            while not stop.is_set():
                await self._fill_slots()
                sleeper = asyncio.ensure_future(self._clock.sleep(self._poll_interval_s))
                waiting: set[asyncio.Future[Any]] = {sleeper, stop_waiter, *self._running}
                try:
                    await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    if not sleeper.done():
                        sleeper.cancel()
        finally:
            stop_waiter.cancel()
            await self.drain()

    async def drain(self, timeout_s: float = 30.0) -> None:
        """Wait for running jobs; cancel (and release) those that outlast ``timeout_s``."""
        if not self._running:
            return
        _, pending = await asyncio.wait(set(self._running), timeout=timeout_s)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
