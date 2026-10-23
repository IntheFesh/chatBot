"""R18-D1 (D-795): a DeepSeek outage on the account side hands background jobs back, it never
fails them.

A rejected key (401/403) or an exhausted balance (402) is not a property of the job: the same job
works again as soon as the user stores a new key or tops the account up.  Before round 18 the
worker counted these like any failure (30 s, 60 s, then ``failed``) and a memory extraction that
met a 402 was gone for good about 90 seconds later, without a notice (``job_failed`` does not
notify, D-445).  The worker now treats every "DeepSeek cannot do this now" error the same way, in
one place: the job goes back to the queue, no attempt is counted, and it is asked again later
(one hour for the account and the budget, five minutes for the circuit breaker).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any

import httpx
import pytest
import respx
from typer.testing import CliRunner

from tests.support.clock import ManualClock
from tests.support.deepseek import API, TEST_KEY, error
from tests.support.embedding import HashingBackend
from tests.support.memory import FactRule, ScriptedMemoryModel
from tests.support.policies import AlwaysOffPeak
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.llm.errors import (
    SERVICE_UNAVAILABLE,
    ApiError,
    AuthenticationFailedError,
    BudgetDeniedError,
    CircuitOpenError,
    InsufficientBalanceError,
    unavailable_retry_s,
)
from twin.llm.runtime import DEEPSEEK_SECRET, close_llm_runtime
from twin.memory.jobs import EXTRACT_JOB, handle_memory_extract, queue_bot_extraction
from twin.memory.memory import Memory
from twin.memory.recent import BotMessage, use_bot_turn_reader
from twin.ops.alerts import AlertService
from twin.ops.jobs import HandlerRegistry, JobContext, JobQueue, Worker, default_registry
from twin.services import Services, build_services
from twin.storage.db import Database

HOUR = 3600.0


def make_worker(
    queue: JobQueue, registry: HandlerRegistry, db: Database, clock: ManualClock
) -> Worker:
    return Worker(queue, registry, clock, offpeak=AlwaysOffPeak(), alerts=AlertService(db, clock))


# ------------------------------------------------------------------ the worker


@pytest.mark.parametrize(
    ("failure", "wait"),
    [
        (InsufficientBalanceError("DeepSeek balance exhausted", status=402), HOUR),
        (AuthenticationFailedError("DeepSeek rejected the API key", status=401), HOUR),
        (BudgetDeniedError("proactive", 3), HOUR),
        (CircuitOpenError(42.0), 300.0),
    ],
    ids=["balance", "key", "budget", "breaker"],
)
async def test_the_worker_hands_a_job_back_when_deepseek_cannot_do_it_now(
    db: Database, clock: ManualClock, failure: Exception, wait: float
) -> None:
    """R18-D1: the handler raises the client's own error; the worker alone decides (rule 8)."""
    queue = JobQueue(db, clock)
    registry = HandlerRegistry()
    calls: list[int] = []

    async def handler(ctx: JobContext) -> None:
        calls.append(ctx.job.attempts)
        raise failure

    registry.register("llm_work", handler)
    job_id = queue.enqueue("llm_work", {}, max_attempts=3)
    worker = make_worker(queue, registry, db, clock)
    summary = await worker.run_until_idle()
    assert (summary.deferred, summary.retried, summary.failed) == (1, 0, 0)
    job = queue.get(job_id)
    assert job is not None and job.status == "pending" and job.attempts == 0
    assert job.run_after == clock.now_utc() + timedelta(seconds=wait)
    assert job.last_error is None  # nothing went wrong with the job itself
    for _ in range(5):  # an outage that lasts: asked again each time, never failed
        clock.tick(wait + 1)
        await worker.run_until_idle()
    job = queue.get(job_id)
    assert job is not None and job.status == "pending" and job.attempts == 0
    assert calls == [1] * 6
    assert unavailable_retry_s(failure) == wait


async def test_an_outage_wrapped_by_the_handler_is_recognised_by_its_cause(
    db: Database, clock: ManualClock
) -> None:
    """R18-D1: ``raise SomethingElse(...) from InsufficientBalanceError`` is the same outage."""
    queue = JobQueue(db, clock)
    registry = HandlerRegistry()

    async def handler(ctx: JobContext) -> None:
        try:
            raise InsufficientBalanceError("DeepSeek balance exhausted", status=402)
        except InsufficientBalanceError as exc:
            raise RuntimeError("the summary could not be written") from exc

    registry.register("wrapped", handler)
    job_id = queue.enqueue("wrapped", {})
    summary = await make_worker(queue, registry, db, clock).run_until_idle()
    assert summary.deferred == 1 and summary.failed == 0
    job = queue.get(job_id)
    assert job is not None and job.status == "pending" and job.attempts == 0


async def test_other_api_errors_still_count_as_failures(db: Database, clock: ManualClock) -> None:
    """R18-D1: only the account, the budget and the breaker defer; a 400 is the job's problem."""
    queue = JobQueue(db, clock)
    registry = HandlerRegistry()

    async def handler(ctx: JobContext) -> None:
        raise ApiError("DeepSeek rejected the request (400)", status=400)

    registry.register("bad_request", handler)
    job_id = queue.enqueue("bad_request", {}, max_attempts=2)
    worker = make_worker(queue, registry, db, clock)
    first = await worker.run_until_idle()
    assert first.retried == 1 and first.deferred == 0
    clock.tick(31)
    second = await worker.run_until_idle()
    assert second.failed == 1
    job = queue.get(job_id)
    assert job is not None and job.status == "failed" and job.attempts == 2
    assert unavailable_retry_s(ApiError("x", status=400)) is None


def test_the_unavailable_errors_are_the_llm_errors_of_the_account_budget_and_breaker() -> None:
    assert set(SERVICE_UNAVAILABLE) == {
        BudgetDeniedError,
        CircuitOpenError,
        AuthenticationFailedError,
        InsufficientBalanceError,
    }


# ---------------------------------------------------------- a real memory job


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture(autouse=True)
def no_bot_reader() -> Iterator[None]:
    with use_bot_turn_reader(None):
        yield


def turns(at: datetime) -> list[BotMessage]:
    return [
        BotMessage("t1", "user", "我养了一只狗", at),
        BotMessage("t2", "bot", "什么品种呀", at + timedelta(seconds=30)),
    ]


@pytest.mark.parametrize(("status", "message"), [(402, "Insufficient Balance"), (401, "bad key")])
async def test_a_memory_extraction_survives_an_account_outage_and_runs_when_it_ends(
    services: Services,
    clock: ManualClock,
    embedder: HashingBackend,
    api: respx.MockRouter,
    status: int,
    message: str,
) -> None:
    """R18-D1: the repro of the audit, reversed.

    Before: ``calls 3 status failed attempts 3`` within 90 seconds.  Now the job waits through
    hours of 402 without spending an attempt and is done once the API answers again.
    """
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    model = ScriptedMemoryModel(
        facts=[
            FactRule(
                "我养了一只狗",
                {
                    "speaker": "user",
                    "subject": "user",
                    "category": "life",
                    "text": "对方养了一只狗",
                },
            )
        ]
    )
    outage = {"on": True}

    def answer(request: httpx.Request) -> httpx.Response:
        return error(status, message) if outage["on"] else model(request)

    route = api.post(API).mock(side_effect=answer)
    assert queue_bot_extraction(services, turns(clock.now_utc()))
    queue = JobQueue(services.db, clock)
    (job,) = queue.list_jobs(job_type=EXTRACT_JOB)
    registry = HandlerRegistry()
    registry.register(EXTRACT_JOB, handle_memory_extract)
    worker = Worker(
        queue, registry, clock, services=services, offpeak=AlwaysOffPeak(), alerts=services.alerts
    )
    try:
        summary = await worker.run_until_idle()
        assert (summary.deferred, summary.retried, summary.failed) == (1, 0, 0)
        for _ in range(3):  # 90 seconds was the end of the job before round 18
            clock.tick(61)
            await worker.run_until_idle()
        assert route.call_count == 1  # it waits an hour, it does not hammer the API
        for _ in range(4):  # the outage lasts four hours
            clock.tick(HOUR + 1)
            await worker.run_until_idle()
        waiting = queue.get(job.id)
        assert waiting is not None and waiting.status == "pending" and waiting.attempts == 0
        assert route.call_count == 5
        outage["on"] = False  # the user topped the account up / stored a new key
        clock.tick(HOUR + 1)
        done = await worker.run_until_idle()
    finally:
        await close_llm_runtime(services)
    assert done.done == 1 and done.failed == 0
    finished = queue.get(job.id)
    assert finished is not None and finished.status == "done"
    assert [f.text for f in Memory(services).store.all_facts()] == ["对方养了一只狗"]


# ------------------------------------------------------------ twin jobs retry --failed

runner = CliRunner()


@pytest.fixture
def cli_data(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    result = runner.invoke(app, ["db", "upgrade"])
    assert result.exit_code == 0, result.output
    return path


def _fail_jobs(kinds: list[str]) -> dict[str, list[str]]:
    """Enqueue one job per kind and fail them all; returns the ids by kind (plus a done one)."""
    settings = load_settings()
    services = build_services(settings, root=resolve_paths(settings).root)
    ids: dict[str, list[str]] = {}
    try:
        queue = JobQueue(services.db, services.clock)
        for kind in kinds:
            job_id = queue.enqueue(kind, {}, max_attempts=1)
            claimed = queue.claim_next({kind}, offpeak_allowed=True, worker_id="w")
            assert claimed is not None
            queue.fail(job_id, "w", "InsufficientBalanceError status=402", retry_at=None)
            ids.setdefault(kind, []).append(job_id)
        done_id = queue.enqueue("memory_summary", {})
        assert queue.claim_next({"memory_summary"}, offpeak_allowed=True, worker_id="w")
        assert queue.complete(done_id, "w")
        ids["done"] = [done_id]
        cancelled = queue.enqueue("memory_extract", {})
        assert queue.cancel(cancelled)
        ids["cancelled"] = [cancelled]
    finally:
        services.close()
    return ids


def _statuses() -> dict[str, tuple[str, int]]:
    settings = load_settings()
    services = build_services(settings, root=resolve_paths(settings).root)
    try:
        queue = JobQueue(services.db, services.clock)
        return {job.id: (job.status, job.attempts) for job in queue.list_jobs(limit=100)}
    finally:
        services.close()


def test_jobs_retry_failed_requeues_every_failed_job_of_a_type(cli_data: Any) -> None:
    """R18-D1: after an outage, one command puts the failed jobs back (``--type`` narrows it)."""
    ids = _fail_jobs(["memory_extract", "memory_extract", "image_caption"])
    narrowed = runner.invoke(app, ["jobs", "retry", "--failed", "--type", "memory_extract"])
    assert narrowed.exit_code == 0, narrowed.output
    assert "2 failed job(s) queued again" in narrowed.output
    states = _statuses()
    assert all(states[i] == ("pending", 0) for i in ids["memory_extract"])
    assert states[ids["image_caption"][0]] == ("failed", 1)
    assert states[ids["done"][0]][0] == "done"
    assert states[ids["cancelled"][0]][0] == "cancelled"  # --failed means failed only
    everything = runner.invoke(app, ["jobs", "retry", "--failed"])
    assert everything.exit_code == 0 and "1 failed job(s) queued again" in everything.output
    assert _statuses()[ids["image_caption"][0]] == ("pending", 0)
    nothing = runner.invoke(app, ["jobs", "retry", "--failed"])
    assert nothing.exit_code == 0 and "no failed job" in nothing.output


def test_jobs_run_says_how_many_jobs_were_handed_back(cli_data: Any) -> None:
    """R18-D1: a foreground run during an outage says ``deferred=1`` instead of all zeros."""

    async def no_balance(ctx: JobContext) -> None:
        raise InsufficientBalanceError("DeepSeek balance exhausted", status=402)

    default_registry.register("outage_demo", no_balance)
    try:
        settings = load_settings()
        services = build_services(settings, root=resolve_paths(settings).root)
        try:
            job_id = JobQueue(services.db, services.clock).enqueue("outage_demo", {})
        finally:
            services.close()
        result = runner.invoke(app, ["jobs", "run", "--until-idle"])
    finally:
        default_registry._handlers.pop("outage_demo", None)
    assert result.exit_code == 0, result.output
    assert "failed=0" in result.output and "deferred=1" in result.output
    assert _statuses()[job_id] == ("pending", 0)


def test_jobs_retry_takes_an_id_or_failed_but_not_both(cli_data: Any) -> None:
    neither = runner.invoke(app, ["jobs", "retry"])
    assert neither.exit_code == 2 and "--failed" in neither.output
    both = runner.invoke(app, ["jobs", "retry", "some-id", "--failed"])
    assert both.exit_code == 2
    typed = runner.invoke(app, ["jobs", "retry", "some-id", "--type", "memory_extract"])
    assert typed.exit_code == 2 and "--type" in typed.output
