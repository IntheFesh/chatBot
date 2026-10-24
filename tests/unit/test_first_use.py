"""The first use, with the application not running (README "第一次使用"; R18-F1, D-826).

What went wrong: every HEAVY command said "the running application executes it" whether or not it
was running, the error hints proposed commands that only queue, ``twin import --foreground``
ran the import alone (by design, D-144) and ``twin jobs run --until-idle`` stopped as soon as
nothing was runnable *now* - while the persona statistics had handed themselves back for 60
seconds because the profile was being recomputed beside them.  So after the steps of the README
neither ``twin profile show`` nor ``twin persona show`` had anything to show.

Now the hint depends on whether the application runs (one implementation,
:func:`twin.ops.process_model.queued_hint` / :func:`~twin.ops.process_model.rerun_hint`), a
foreground run waits for the jobs it handed back while they wait for another job of the queue
(:meth:`twin.ops.jobs.Worker.run_until_idle`, ``wait_s``), and it lists what waits for an approval.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tests.support.clock import InstantClock, ManualClock
from tests.support.embedding import HashingBackend
from tests.support.ingest import make_export
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.llm.errors import BudgetDeniedError
from twin.ops.alerts import AlertService
from twin.ops.instance_lock import LOCK_RUN, InstanceLock
from twin.ops.jobs import (
    FOREGROUND_WAIT_S,
    HandlerRegistry,
    JobContext,
    JobDeferred,
    JobQueue,
    Worker,
)
from twin.ops.process_model import ExitCode, JobSpec, enqueue_heavy
from twin.profile.persona.jobs import REFRESH_JOB
from twin.services import Services, build_services
from twin.storage import migrate
from twin.storage.crypto import KeyRing
from twin.storage.db import Database

runner = CliRunner()


class AlwaysOffPeak:
    def allows(self, now: object) -> bool:
        return True


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    assert runner.invoke(app, ["db", "upgrade"]).exit_code == 0
    return path


def cli(*args: str) -> Any:
    return runner.invoke(app, list(args))


def with_services[T](work: Callable[[Services], T]) -> T:
    settings = load_settings()
    services = build_services(settings, root=resolve_paths(settings).root)
    try:
        return work(services)
    finally:
        services.close()


@pytest.fixture
def application_running(data_dir: Path) -> Iterator[None]:
    """Another process holds the ``run`` lock, as ``twin run`` does."""
    lock = InstanceLock(LOCK_RUN, locks_dir=resolve_paths(load_settings()).locks_dir)
    assert lock.acquire()
    try:
        yield
    finally:
        lock.release()


# ------------------------------------------------------------------ the hints


def test_a_queued_import_says_how_to_run_it_while_the_application_is_stopped(
    data_dir: Path, tmp_path: Path
) -> None:
    """R18-F1: no claim that a running application executes it, and the two ways to run it now."""
    export = make_export(tmp_path, target_messages=30, other_conversations=0)
    result = cli("import", str(export.root), "--yes")
    assert result.exit_code == 0, result.output
    assert "the running application executes it" not in result.output
    assert "the application is not running" in result.output
    assert "`twin import --resume --foreground`" in result.output
    assert "`twin jobs run --until-idle`" in result.output


def test_a_queued_import_says_the_running_application_executes_it(
    application_running: None, tmp_path: Path
) -> None:
    """R18-F1: with the application running the old sentence is the true one."""
    export = make_export(tmp_path, target_messages=30, other_conversations=0)
    result = cli("import", str(export.root), "--yes")
    assert result.exit_code == 0, result.output
    assert "the running application executes it" in result.output
    assert "--foreground" not in result.output


@pytest.mark.parametrize(
    ("args", "foreground"),
    [
        (["profile", "rebuild"], "twin profile rebuild --foreground"),
        (["retrieval", "rebuild"], "twin retrieval rebuild --foreground"),
        (["memory", "reindex"], "twin memory reindex --foreground"),
        (
            ["persona", "regenerate", "--stats-only"],
            "twin persona regenerate --stats-only --foreground",
        ),
    ],
)
def test_every_queueing_command_names_its_foreground_form_while_the_application_is_stopped(
    data_dir: Path, args: list[str], foreground: str
) -> None:
    """R18-F1: the same wording from every HEAVY command (one implementation)."""
    result = cli(*args)
    assert result.exit_code == 0, result.output
    assert "the running application executes it" not in result.output
    assert f"`{foreground}`" in result.output and "`twin jobs run --until-idle`" in result.output


def test_enqueue_heavy_tells_which_of_the_two_is_true(
    services: Services, capsys: pytest.CaptureFixture[str]
) -> None:
    """R18-F1: the generic HEAVY helper checks the ``run`` lock too."""
    enqueue_heavy(services, [JobSpec("heavy_hint_demo", {"n": 1})])
    stopped = capsys.readouterr().out
    assert "the application is not running" in stopped and "twin jobs run --until-idle" in stopped
    lock = InstanceLock(LOCK_RUN, locks_dir=services.paths.locks_dir)
    assert lock.acquire()
    try:
        enqueue_heavy(services, [JobSpec("heavy_hint_demo", {"n": 2})])
    finally:
        lock.release()
    assert "the running application executes it" in capsys.readouterr().out


def test_the_hints_of_a_missing_profile_and_card_name_a_command_that_runs(data_dir: Path) -> None:
    """R18-F1: `profile show` and `persona show` propose the foreground form of the command."""
    profile = cli("profile", "show")
    assert profile.exit_code == 1
    assert "`twin profile rebuild --scope live --foreground`" in profile.output
    phrases = cli("profile", "phrases")
    assert "`twin profile rebuild --scope live --foreground`" in phrases.output
    history = cli("profile", "history")
    assert "`twin profile rebuild --foreground`" in history.output
    persona = cli("persona", "show")
    assert persona.exit_code == 1
    assert "`twin persona regenerate --stats-only --foreground`" in persona.output


def test_the_hints_while_the_application_runs_need_no_foreground(
    application_running: None,
) -> None:
    """R18-F1: the queueing form is right when the application executes the queue."""
    profile = cli("profile", "show")
    assert "`twin profile rebuild --scope live`" in profile.output
    assert "--foreground" not in profile.output
    assert "the running application executes it" in profile.output


def test_chat_local_while_the_application_runs_says_to_stop_it(application_running: None) -> None:
    """R18-F1: the README step "试聊" needs the application stopped; the refusal says how."""
    result = runner.invoke(app, ["chat", "--local"], input="/quit\n")
    assert result.exit_code == int(ExitCode.BUSY), result.output
    assert "twin service stop" in result.output


# ------------------------------------------------------------------ the whole first use


def test_import_in_the_foreground_then_the_queue_gives_the_profile_and_the_card(
    data_dir: Path, tmp_path: Path, embedder: HashingBackend
) -> None:
    """R18-F1: README step 2 then step 3 - `profile show` and `persona show` have something."""
    export = make_export(tmp_path, target_messages=300, other_conversations=0)
    imported = cli("import", str(export.root), "--yes", "--foreground")
    assert imported.exit_code == 0, imported.output
    assert "`twin jobs run --until-idle`" in imported.output  # the hooks' jobs wait in the queue
    ran = cli("jobs", "run", "--until-idle")
    assert ran.exit_code == 0, ran.output
    assert "waiting for an approval" in ran.output and "twin jobs approve persona-" in ran.output
    assert cli("profile", "show").exit_code == 0
    persona = cli("persona", "show")
    assert persona.exit_code == 0, persona.output

    def left(services: Services) -> list[str]:
        queue = JobQueue(services.db, services.clock)
        return [job.type for job in queue.list_jobs(status="pending") if not job.requires_approval]

    assert REFRESH_JOB not in with_services(left)


# ------------------------------------------------------------------ the worker


@pytest.fixture
def instant_db(tmp_path: Path, keyring_ring: KeyRing) -> Iterator[tuple[Database, InstantClock]]:
    clock = InstantClock()
    path = tmp_path / "data" / "jobs.db"
    migrate.upgrade(path)
    database = Database(path, clock=clock)
    yield database, clock
    database.dispose()


def worker(db: Database, clock: Any, registry: HandlerRegistry, concurrency: int = 2) -> Worker:
    return Worker(
        JobQueue(db, clock),
        registry,
        clock,
        offpeak=AlwaysOffPeak(),
        alerts=AlertService(db, clock),
        concurrency=concurrency,
    )


def dependent_pair(queue: JobQueue, registry: HandlerRegistry, done: list[str]) -> None:
    """``refresh`` hands itself back while ``rebuild`` waits or runs (as the persona card does)."""
    refresh_tried = asyncio.Event()

    async def rebuild(ctx: JobContext) -> None:
        await refresh_tried.wait()  # both run at the same time, as with concurrency 2
        done.append("rebuild")

    async def refresh(ctx: JobContext) -> None:
        busy = queue.list_jobs(status="running", job_type="rebuild") + queue.list_jobs(
            status="pending", job_type="rebuild"
        )
        refresh_tried.set()
        if busy:
            raise JobDeferred("the profile is being recomputed", retry_in_s=60.0)
        done.append("refresh")

    registry.register("rebuild", rebuild)
    registry.register("refresh", refresh)
    queue.enqueue("rebuild", {}, priority=60)
    queue.enqueue("refresh", {}, priority=65)


async def test_a_foreground_run_finishes_a_job_that_waited_for_another_one(
    db: Database, clock: ManualClock
) -> None:
    """R18-F1: the handed-back job runs again as soon as nothing else runs - no wall-clock wait."""
    registry, done = HandlerRegistry(), []
    queue = JobQueue(db, clock)
    dependent_pair(queue, registry, done)
    summary = await worker(db, clock, registry).run_until_idle(wait_s=FOREGROUND_WAIT_S)
    assert done == ["rebuild", "refresh"]
    assert (summary.done, summary.deferred) == (2, 1)
    assert clock.sleeps == []  # the manual clock never moved, so nothing slept


async def test_without_a_wait_the_run_stops_as_before(db: Database, clock: ManualClock) -> None:
    """R18-F1: ``twin run`` and the other callers of ``run_until_idle`` are not changed."""
    registry, done = HandlerRegistry(), []
    queue = JobQueue(db, clock)
    dependent_pair(queue, registry, done)
    summary = await worker(db, clock, registry).run_until_idle()
    assert done == ["rebuild"] and summary.deferred == 1


async def test_a_job_waiting_for_something_outside_is_waited_for_within_the_budget(
    instant_db: tuple[Database, InstantClock],
) -> None:
    """R18-F1: retried once at once, then at its time, until the five minutes are spent."""
    db, clock = instant_db
    registry = HandlerRegistry()
    tries: list[int] = []

    async def busy_elsewhere(ctx: JobContext) -> None:
        tries.append(1)
        raise JobDeferred("the index is in use by another process", retry_in_s=120.0)

    registry.register("index", busy_elsewhere)
    JobQueue(db, clock).enqueue("index", {})
    summary = await worker(db, clock, registry).run_until_idle(wait_s=FOREGROUND_WAIT_S)
    assert FOREGROUND_WAIT_S == 300.0
    assert len(tries) == summary.deferred == 4  # first, at once again, after 120 s, after 240 s
    assert sum(clock.sleeps) == pytest.approx(240.0)
    job = JobQueue(db, clock).list_jobs(job_type="index")[0]
    assert job.status == "pending"


async def test_a_job_handed_back_because_deepseek_cannot_do_it_is_not_waited_for(
    instant_db: tuple[Database, InstantClock],
) -> None:
    """R18-F1: an outage (D-795) is not something a foreground run sits out."""
    db, clock = instant_db
    registry = HandlerRegistry()

    async def no_budget(ctx: JobContext) -> None:
        raise BudgetDeniedError("image_caption", 3)

    registry.register("caption", no_budget)
    JobQueue(db, clock).enqueue("caption", {})
    summary = await worker(db, clock, registry).run_until_idle(wait_s=FOREGROUND_WAIT_S)
    assert summary.deferred == 1 and clock.sleeps == []
