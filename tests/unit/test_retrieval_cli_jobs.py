"""The retrieval job, the import hook, the re-split and ``twin retrieval`` (R-IMP-011, R-RET-003,
R-RET-006, R-ARCH-006)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from typer.testing import CliRunner

from tests.support.embedding import H, HashingBackend, U, day, write_dialogue
from tests.support.ingest import make_export, run_import
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.ingest.hooks import HookContext, load_hooks
from twin.ops.instance_lock import LOCK_RUN, InstanceLock
from twin.ops.jobs import HANDLER_MODULES, HandlerRegistry, JobQueue, Worker, load_handlers
from twin.ops.process_model import CommandKind, get_spec, iter_commands
from twin.profile.holdout import LISTENER_MODULES, get_holdout, resplit_holdout
from twin.profile.queue import PROFILE_JOB
from twin.retrieval import embedder as embedder_module
from twin.retrieval.embedder import reset_embedding_services
from twin.retrieval.hook import queue_retrieval
from twin.retrieval.indexer import INDEX_JOB, collect_stats, run_index, window_table
from twin.retrieval.jobs import MISMATCH_ALERT, handle_retrieval_index
from twin.retrieval.queue import queue_retrieval_index
from twin.services import Services, build_services
from twin.storage.models import Alert
from twin.storage.retrieval_models import ExampleWindow

runner = CliRunner()


def configure(services: Services, backend: HashingBackend) -> None:
    services.settings.time.source_timezone = "UTC"
    services.settings.retrieval.model = backend.info.model


def fill(services: Services, count: int = 30, *, start: int = 0) -> None:
    write_dialogue(
        services,
        [(day(start + n), [U(f"问题{n}吃饭"), H(f"回答{n}")]) for n in range(count)],
        append=start > 0,
    )


def worker(services: Services) -> Worker:
    handlers = HandlerRegistry()
    handlers.register(INDEX_JOB, handle_retrieval_index)
    return Worker(
        JobQueue(services.db, services.clock),
        handlers,
        services.clock,
        services=services,
        alerts=services.alerts,
    )


def hook_context(services: Services, *, inserted: int = 10, first: bool = False) -> HookContext:
    return HookContext(
        services=services,
        run_id="run",
        conversation_id="c",
        export_id="e",
        inserted=inserted,
        changed=0,
        first_import=first,
    )


def pending(services: Services) -> list[Any]:
    return JobQueue(services.db, services.clock).list_jobs(status="pending", job_type=INDEX_JOB)


# ----------------------------------------------------------------- the hook


def test_the_hook_is_registered_after_the_profile_with_its_backfill_command() -> None:
    registry = load_hooks()
    names = registry.names()
    hook = next(h for h in registry.hooks() if h.name == "retrieval")
    assert hook.backfill_command == "retrieval rebuild"
    assert names.index("retrieval") > names.index("profile")
    commands = {name for name, _ in iter_commands(app)}
    assert hook.backfill_command in commands


def test_the_hook_queues_one_update_unless_nothing_is_new(
    services: Services, embedder: HashingBackend
) -> None:
    configure(services, embedder)
    quiet = queue_retrieval(hook_context(services, inserted=0))
    assert quiet.status == "skipped" and not pending(services)
    first = queue_retrieval(hook_context(services, inserted=0, first=True))
    assert first.status == "queued" and first.jobs == 1
    again = queue_retrieval(hook_context(services))
    assert again.status == "queued" and again.jobs == 0 and "already waiting" in again.detail
    (job,) = pending(services)
    assert job.payload == {"mode": "update", "full": False, "reason": "import", "run_id": "run"}


def test_the_hook_reports_an_index_made_with_another_model(
    services: Services, embedder: HashingBackend
) -> None:
    configure(services, embedder)
    fill(services)
    run_index(services)
    services.settings.retrieval.model = "BAAI/bge-m3"
    result = queue_retrieval(hook_context(services))
    assert result.status == "failed" and "twin retrieval rebuild" in result.detail
    assert not pending(services)


def test_queueing_is_deduplicated_by_its_parameters(services: Services) -> None:
    a = queue_retrieval_index(services)
    assert (
        queue_retrieval_index(services).already_queued
        and queue_retrieval_index(services).job_id == a.job_id
    )
    assert not queue_retrieval_index(services, full=True).already_queued
    assert not queue_retrieval_index(services, mode="rebuild").already_queued
    with pytest.raises(ValueError, match="mode must be"):
        queue_retrieval_index(services, mode="sometimes")


# ------------------------------------------------------------------ the job


async def test_a_queued_job_builds_the_library_through_the_worker(
    services: Services, embedder: HashingBackend
) -> None:
    configure(services, embedder)
    fill(services)
    queue_retrieval_index(services, reason="import")
    summary = await worker(services).run_until_idle()
    assert summary.done == 1 and summary.failed == 0
    assert window_table(services).count() == 27
    assert collect_stats(services).progress is not None


async def test_the_job_is_registered_with_the_application() -> None:
    assert "twin.retrieval.jobs" in HANDLER_MODULES
    assert load_handlers().has(INDEX_JOB)


async def test_an_index_of_another_model_raises_an_alert_instead_of_retrying(
    services: Services, embedder: HashingBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure(services, embedder)
    fill(services)
    run_index(services)
    # the configuration now names another model: the factory builds that one
    services.settings.retrieval.model = "other/model"
    newer = HashingBackend(model="other/model")
    monkeypatch.setattr(embedder_module, "backend_factory", lambda config, paths: newer)
    reset_embedding_services()
    queue_retrieval_index(services)
    summary = await worker(services).run_until_idle()
    assert summary.done == 1 and summary.retried == 0 and summary.failed == 0
    with services.db.session() as session:
        titles = [
            alert.title
            for alert in session.scalars(select(Alert).where(Alert.category == MISMATCH_ALERT))
        ]
    assert len(titles) == 1 and "twin retrieval rebuild" in titles[0]


async def test_a_busy_index_defers_the_job(services: Services, embedder: HashingBackend) -> None:
    from twin.ops.filelock import FileLock

    configure(services, embedder)
    fill(services)
    queue_retrieval_index(services)
    lock = FileLock(services.paths.locks_dir / "retrieval-index.lock")
    assert lock.acquire()
    try:
        summary = await worker(services).run_until_idle()
    finally:
        lock.release()
    assert summary.done == 0 and summary.deferred == 1
    assert window_table(services).count() == 0


async def test_after_a_real_import_the_hook_queues_the_index_and_a_reimport_queues_nothing(
    services: Services, embedder: HashingBackend, tmp_path: Path
) -> None:
    configure(services, embedder)
    export = make_export(tmp_path, target_messages=300, other_conversations=0)
    first = run_import(services, export)
    assert first.run.hooks["retrieval"]["status"] == "queued"
    assert "retrieval" in first.run.hooks and len(pending(services)) == 1
    summary = await worker(services).run_until_idle()
    assert summary.done == 1 and summary.failed == 0
    stats = collect_stats(services)
    assert stats.vectors > 0 and stats.held_out > 0 and stats.awaiting == 0
    seen = len(embedder.seen)

    again = run_import(services, export)  # the same export: every message is a duplicate
    assert again.run.hooks["retrieval"]["status"] == "skipped"
    assert not pending(services) and len(embedder.seen) == seen


# ------------------------------------------------------------------ re-split


def test_the_resplit_listener_is_loaded_for_any_caller() -> None:
    assert "twin.retrieval.resplit" in LISTENER_MODULES


def test_a_resplit_takes_new_holdout_windows_out_of_the_index_at_once(
    services: Services, embedder: HashingBackend
) -> None:
    configure(services, embedder)
    fill(services)
    run_index(services)
    assert window_table(services).count() == 27
    services.settings.retrieval.holdout_ratio = 0.3
    result = resplit_holdout(services)
    note = next(n for n in result.notes if n.startswith("retrieval:"))
    assert "6 window(s) are now held out" in note and "0 were released" in note
    assert get_holdout(services).held_out_blocks == 9  # type: ignore[union-attr]
    assert window_table(services).count() == 21  # gone from the index immediately
    stats = collect_stats(services)
    assert (stats.held_out, stats.indexed) == (9, 21)
    assert len(pending(services)) == 1


async def test_a_smaller_holdout_releases_windows_that_are_then_encoded(
    services: Services, embedder: HashingBackend
) -> None:
    configure(services, embedder)
    fill(services)
    run_index(services)
    services.settings.retrieval.holdout_ratio = 0.2
    resplit_holdout(services)
    assert window_table(services).count() == 24
    services.settings.retrieval.holdout_ratio = 0.05
    result = resplit_holdout(services)
    note = next(n for n in result.notes if n.startswith("retrieval:"))
    assert "0 window(s) are now held out" in note and "4 were released" in note
    assert collect_stats(services).awaiting == 4
    await worker(services).run_until_idle()
    assert window_table(services).count() == 28
    with services.db.session() as session:
        flags = [row.holdout for row in session.scalars(select(ExampleWindow))]
    assert sum(flags) == 2


def test_a_resplit_before_the_library_exists_only_says_so(
    services: Services, embedder: HashingBackend
) -> None:
    configure(services, embedder)
    fill(services)
    result = resplit_holdout(services)
    note = next(n for n in result.notes if n.startswith("retrieval:"))
    assert "has not been built yet" in note
    assert not pending(services)
    assert PROFILE_JOB  # the profile of the pre-holdout scope is queued by its own listener


# ----------------------------------------------------------------------- CLI


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedder: HashingBackend) -> Path:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    monkeypatch.setenv("TWIN_TIME__SOURCE_TIMEZONE", "UTC")
    monkeypatch.setenv("TWIN_RETRIEVAL__MODEL", embedder.info.model)
    result = runner.invoke(app, ["db", "upgrade"])
    assert result.exit_code == 0, result.output
    return path


def cli(*args: str, input_text: str | None = None) -> Any:
    return runner.invoke(app, list(args), input=input_text)


def with_services[T](work: Callable[[Services], T]) -> T:
    settings = load_settings()
    services = build_services(settings, root=resolve_paths(settings).root)
    try:
        return work(services)
    finally:
        services.close()


def test_every_command_declares_its_process_model() -> None:
    commands = dict(iter_commands(app))
    kinds = {
        name: get_spec(commands[name]).kind  # type: ignore[union-attr]
        for name in ("retrieval rebuild", "retrieval stats", "retrieval resplit")
    }
    assert kinds == {
        "retrieval rebuild": CommandKind.HEAVY,
        "retrieval stats": CommandKind.READ,
        "retrieval resplit": CommandKind.HEAVY,
    }


def test_stats_of_an_empty_library_say_how_to_build_it(data_dir: Path) -> None:
    result = cli("retrieval", "stats")
    assert result.exit_code == 0 and "twin retrieval rebuild" in result.output


def test_rebuild_queues_then_builds_in_the_foreground_and_stats_describe_it(
    data_dir: Path,
) -> None:
    with_services(fill)
    queued = cli("retrieval", "rebuild")
    assert queued.exit_code == 0 and "queued" in queued.output
    assert "twin retrieval stats" in queued.output
    again = cli("retrieval", "rebuild")
    assert "already waiting" in again.output

    done = cli("retrieval", "rebuild", "--foreground")
    assert done.exit_code == 0, done.output
    assert "windows (her reply blocks)" in done.output and "27" in done.output

    stats = cli("retrieval", "stats")
    assert stats.exit_code == 0, stats.output
    for fragment in (
        "windows (her reply blocks)",
        "held out (evaluation set)",
        "hold-out cutoff",
        "windows with a vector",
        "vectors in the index",
        "test/hashing-bigram",
        "last index run",
    ):
        assert fragment in stats.output
    assert "回答" not in stats.output and "问题" not in stats.output  # counts only, no chat text

    full = cli("retrieval", "rebuild", "--full", "--foreground")
    assert full.exit_code == 0, full.output


def test_a_running_application_executes_the_job_itself(data_dir: Path) -> None:
    with_services(fill)
    lock = InstanceLock(LOCK_RUN, locks_dir=resolve_paths(load_settings()).locks_dir)
    assert lock.acquire()
    try:
        result = cli("retrieval", "rebuild", "--foreground")
    finally:
        lock.release()
    assert result.exit_code == 0 and "the application is running" in result.output


def test_stats_name_a_slow_model_and_a_mismatching_one(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with_services(fill)
    assert cli("retrieval", "rebuild", "--foreground").exit_code == 0
    monkeypatch.setenv("TWIN_RETRIEVAL__MODEL", "BAAI/bge-m3")
    result = cli("retrieval", "stats")
    assert result.exit_code == 0
    assert "problem:" in result.output and "twin retrieval rebuild" in result.output
    rebuild = cli("retrieval", "rebuild")
    assert "note:" in rebuild.output and "slower" in rebuild.output  # the CPU warning for bge-m3


def test_resplit_asks_first_and_then_moves_the_cutoff(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with_services(fill)
    assert cli("retrieval", "rebuild", "--foreground").exit_code == 0
    before = with_services(lambda s: get_holdout(s))
    assert before is not None

    refused = cli("retrieval", "resplit", input_text="n\n")
    assert refused.exit_code != 0
    assert with_services(lambda s: get_holdout(s)) == before

    monkeypatch.setenv("TWIN_RETRIEVAL__HOLDOUT_RATIO", "0.2")
    done = cli("retrieval", "resplit", "--yes", "--foreground")
    assert done.exit_code == 0, done.output
    assert "cutoff:" in done.output and "not comparable" in done.output
    assert "retrieval:" in done.output and "held out" in done.output
    after = with_services(lambda s: get_holdout(s))
    assert after is not None and after.cutoff < before.cutoff and after.held_out_blocks == 6
    stats = with_services(collect_stats)
    assert (stats.held_out, stats.indexed) == (6, 24)


def test_resplit_with_too_little_data_is_an_error(data_dir: Path) -> None:
    result = cli("retrieval", "resplit", "--yes")
    assert result.exit_code != 0
    assert "reply blocks" in result.output
