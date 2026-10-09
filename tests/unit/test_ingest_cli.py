"""``twin import``, ``twin images`` and ``twin stickers`` (R-IMP-003, R-IMP-006, R-ARCH-006)."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import respx
from sqlalchemy import func, select, update
from typer.testing import CliRunner

from tests.fixtures.synth_export import SynthExport
from tests.support.ingest import make_export
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.config.runtime import TARGET_USERNAME
from twin.ingest.importer import ImportRunner, prepare_import
from twin.ingest.runs import RunView, latest_run
from twin.ops.instance_lock import LOCK_RUN, InstanceLock
from twin.ops.process_model import ExitCode
from twin.services import Services, build_services
from twin.storage.chat_models import ImportRun, Message, Sticker
from twin.storage.models import Job

runner = CliRunner()


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    result = runner.invoke(app, ["db", "upgrade"])
    assert result.exit_code == 0, result.output
    return path


def cli(*args: str, input: str | None = None) -> Any:
    return runner.invoke(app, list(args), input=input)


def with_services[T](work: Callable[[Services], T]) -> T:
    settings = load_settings()
    services = build_services(settings, root=resolve_paths(settings).root)
    try:
        return work(services)
    finally:
        services.close()


def stored_messages() -> int:
    def count(services: Services) -> int:
        with services.db.session() as session:
            return int(session.scalar(select(func.count()).select_from(Message)) or 0)

    return with_services(count)


def import_jobs(services: Services) -> list[tuple[str, str]]:
    with services.db.session() as session:
        return [
            (job.type, job.status) for job in session.scalars(select(Job).order_by(Job.created_at))
        ]


def export_with(tmp_path: Path, **options: Any) -> SynthExport:
    return make_export(tmp_path, **{"target_messages": 60, "other_conversations": 0, **options})


# ------------------------------------------------------------------------ start


def test_the_first_import_asks_which_conversation_and_remembers_the_answer(
    data_dir: Path, tmp_path: Path
) -> None:
    export = export_with(tmp_path, other_conversations=2)
    result = cli("import", str(export.root), input="1\n")
    assert result.exit_code == 0, result.output
    assert "Number of the conversation" in result.output
    assert export.target_display_name in result.output  # nicknames are shown ...
    assert export.target_username not in result.output  # ... ids are masked
    assert (
        export.group_username is not None and "周末群聊" not in result.output
    )  # groups not offered
    assert "queued import run" in result.output and "twin import status" in result.output

    def check(services: Services) -> tuple[str | None, list[tuple[str, str]]]:
        return services.runtime.get(TARGET_USERNAME), import_jobs(services)

    target, jobs = with_services(check)
    assert target == export.target_username
    assert jobs == [("import", "pending")]
    assert stored_messages() == 0  # queued, not run: the application executes it


def test_a_stored_target_is_not_asked_for_again_and_start_is_the_same_command(
    data_dir: Path, tmp_path: Path
) -> None:
    export = export_with(tmp_path)
    assert cli("import", str(export.root), "--yes").exit_code == 0  # a single candidate
    again = cli("import", "start", str(export.root))
    assert again.exit_code == 0, again.output
    assert "Number of the conversation" not in again.output
    assert "continuing the unfinished run" in again.output


def test_the_target_can_be_given_on_the_command_line(data_dir: Path, tmp_path: Path) -> None:
    export = export_with(tmp_path, other_conversations=2)
    second = cli("import", str(export.root), "--target", "2")
    assert second.exit_code == 0, second.output
    by_id = cli("import", str(export.root), "--target", export.other_usernames[1])
    assert by_id.exit_code == 0, by_id.output
    assert with_services(lambda s: s.runtime.get(TARGET_USERNAME)) is None  # not stored
    wrong = cli("import", str(export.root), "--target", "wxid_" + "nobody0000")
    assert wrong.exit_code == int(ExitCode.USAGE)
    assert "not in this export" in wrong.output and export.target_display_name in wrong.output
    assert export.target_username not in wrong.output


def test_an_out_of_range_number_is_refused(data_dir: Path, tmp_path: Path) -> None:
    export = export_with(tmp_path)
    result = cli("import", str(export.root), input="9\n")
    assert result.exit_code == int(ExitCode.USAGE) and "choose a number" in result.output


def test_the_export_directory_may_come_from_the_configuration(
    data_dir: Path, tmp_path: Path
) -> None:
    export = export_with(tmp_path)
    result = cli("--set", f"paths.export_dir={export.root}", "import", "start", "--yes")
    assert result.exit_code == 0, result.output
    assert cli("import", "start", "--yes").exit_code == int(ExitCode.USAGE)  # nothing configured


def test_bad_exports_are_reported_without_a_traceback(data_dir: Path, tmp_path: Path) -> None:
    missing = cli("import", str(tmp_path / "nowhere"), "--yes")
    assert missing.exit_code == 1 and "not found" in missing.output
    old = export_with(tmp_path / "v2", schema_version=2)
    refused = cli("import", str(old.root), "--yes")
    assert refused.exit_code == 1 and "schemaVersion" in refused.output
    assert "Traceback" not in refused.output


def test_import_with_nothing_to_import_shows_help_and_resume_without_a_run_fails(
    data_dir: Path,
) -> None:
    assert "Usage" in cli("import").output
    result = cli("import", "--resume")
    assert result.exit_code == 1 and "no unfinished import" in result.output


# ----------------------------------------------------------------------- running


def test_a_queued_import_is_executed_by_the_job_runner_when_the_app_is_stopped(
    data_dir: Path, tmp_path: Path
) -> None:
    export = export_with(tmp_path, target_messages=90)
    assert cli("import", str(export.root), "--yes").exit_code == 0
    queued = cli("import", "status")
    assert "queued" in queued.output and "processed 0 / 90" in queued.output
    ran = cli("jobs", "run", "--until-idle")
    assert ran.exit_code == 0 and "done=" in ran.output
    status = cli("import", "status")
    assert "done" in status.output and "processed 90 / 90 (100.0%)" in status.output
    assert stored_messages() == 90


def test_foreground_runs_the_import_here_with_a_report(data_dir: Path, tmp_path: Path) -> None:
    export = export_with(tmp_path, target_messages=75)
    result = cli("import", str(export.root), "--yes", "--foreground")
    assert result.exit_code == 0, result.output
    assert stored_messages() == 75
    assert "# 导入报告" in result.output and "processed 75" in result.output
    assert export.target_username not in result.output
    for sentence in export.texts:
        assert sentence not in result.output
    # the foreground run executes the import job only: the sticker download stays queued
    jobs = with_services(import_jobs)
    assert ("import", "done") in jobs and ("sticker_download", "pending") in jobs


def test_foreground_yields_to_a_running_application(data_dir: Path, tmp_path: Path) -> None:
    export = export_with(tmp_path)
    locks = resolve_paths(load_settings()).locks_dir
    locks.mkdir(parents=True, exist_ok=True)
    lock = InstanceLock(LOCK_RUN, locks_dir=locks)
    assert lock.acquire()
    try:
        result = cli("import", str(export.root), "--yes", "--foreground")
    finally:
        lock.release()
    assert result.exit_code == 0 and "the application is running" in result.output
    assert stored_messages() == 0


def test_resume_continues_an_import_that_stopped_halfway(data_dir: Path, tmp_path: Path) -> None:
    export = export_with(tmp_path, target_messages=120)

    class Stop(BaseException):
        pass

    def crash(services: Services) -> str:
        run_id = prepare_import(
            services, export.root, target_username=export.target_username
        ).run_id

        def die(event: Any) -> None:
            if event.batch_number == 2:
                raise Stop

        with pytest.raises(Stop):
            ImportRunner(services, batch_size=30, on_batch=die).run(run_id)
        return run_id

    run_id = with_services(crash)
    assert stored_messages() == 60
    assert "running" in cli("import", "status").output
    result = cli("--set", "ingest.batch_size=30", "import", "--resume", "--foreground")
    assert result.exit_code == 0, result.output
    assert f"resuming import run {run_id} at 60 messages" in result.output
    assert stored_messages() == 120


def test_a_failed_foreground_import_says_why(data_dir: Path, tmp_path: Path) -> None:
    export = export_with(tmp_path, target_messages=40)
    data = export.messages_path.read_bytes()
    export.messages_path.write_bytes(data[: len(data) // 2])
    result = cli("import", str(export.root), "--yes", "--foreground")
    assert result.exit_code == 1 and "did not finish" in result.output
    assert "cut off" in result.output
    status = cli("import", "status")
    assert "failed" in status.output and "error:" in status.output


# ------------------------------------------------------------------------ status


def test_status_without_an_import_says_what_to_do(data_dir: Path) -> None:
    result = cli("import", "status")
    assert result.exit_code == 1 and "twin import <directory>" in result.output


def test_status_shows_progress_speed_remaining_time_and_hooks(
    data_dir: Path, tmp_path: Path
) -> None:
    export = export_with(tmp_path, target_messages=50)
    cli("import", str(export.root), "--yes", "--foreground")

    def make_it_look_busy(services: Services) -> None:
        with services.db.transaction() as session:
            session.execute(
                update(ImportRun).values(
                    status="running", phase="messages", processed=20, total=50, speed_per_s=10.0
                )
            )

    with_services(make_it_look_busy)
    result = cli("import", "status")
    assert result.exit_code == 0
    assert "running, phase messages" in result.output
    assert "processed 20 / 50 (40.0%)" in result.output
    assert "speed 10 messages/s" in result.output and "remaining 3s" in result.output
    assert "post-import hooks:" in result.output
    assert "image_caption" in result.output and "sticker_download: queued" in result.output
    assert "report:" in result.output
    assert cli("import", "status", "--run", "nope").exit_code == 1


def test_watch_returns_as_soon_as_the_import_has_finished(data_dir: Path, tmp_path: Path) -> None:
    export = export_with(tmp_path, target_messages=30)
    cli("import", str(export.root), "--yes", "--foreground")
    result = cli("import", "status", "--watch", "--interval", "0.01")
    assert result.exit_code == 0 and "processed 30 / 30" in result.output


def test_watch_follows_a_running_import_until_it_ends(data_dir: Path, tmp_path: Path) -> None:
    export = export_with(tmp_path, target_messages=30)
    cli("import", str(export.root), "--yes", "--foreground")

    def set_status(status: str, phase: str) -> None:
        def write(services: Services) -> None:
            with services.db.transaction() as session:
                session.execute(update(ImportRun).values(status=status, phase=phase))

        with_services(write)

    set_status("running", "media")
    finisher = threading.Timer(0.3, set_status, args=("done", "done"))
    finisher.start()
    try:
        result = cli("import", "status", "--watch", "--interval", "0.02")
    finally:
        finisher.cancel()
        finisher.join()
    assert result.exit_code == 0 and ": done, phase done" in result.output


def test_status_of_a_failed_run_shows_the_error(data_dir: Path, tmp_path: Path) -> None:
    export = export_with(tmp_path, target_messages=10)
    cli("import", str(export.root), "--yes")

    def fail(services: Services) -> RunView | None:
        with services.db.transaction() as session:
            session.execute(update(ImportRun).values(status="failed", error="disk full"))
        return latest_run(services.db)

    assert with_services(fail) is not None
    assert "error: disk full" in cli("import", "status").output


# ----------------------------------------------------------------------- inspect


def test_inspect_prints_and_saves_the_structure_without_values(
    data_dir: Path, tmp_path: Path
) -> None:
    export = export_with(tmp_path, target_messages=80, integrity="json")
    result = cli("import", "inspect", str(export.root))
    assert result.exit_code == 0, result.output
    reports = sorted((data_dir / "reports").glob("inspect-*.md"))
    assert len(reports) == 1
    saved = reports[0].read_text(encoding="utf-8")
    assert "structure report written to" in result.output
    for text in (saved, result.output):
        assert "renderType" in text and "offlineMedia" in text and "manifest.json" in text
        for sentence in export.texts:
            assert sentence not in text
        assert export.target_username not in text and export.target_display_name not in text
        assert "wxid_" not in text
    assert cli("import", "inspect").exit_code == 2 or cli("import", "inspect").exit_code == 1


# ------------------------------------------------------------- images and stickers


def test_caption_backfill_queues_a_batch_and_says_how_to_approve_it(
    data_dir: Path, tmp_path: Path
) -> None:
    from datetime import UTC, datetime

    export = make_export(
        tmp_path,
        target_messages=120,
        mix={"image": 60.0, "text": 40.0},
        start=datetime(2026, 10, 1, tzinfo=UTC),
        average_gap_s=600.0,
        other_conversations=0,
        include_group=False,
    )
    # the import hook has queued the pictures already; remove them to plan again by hand
    assert cli("import", str(export.root), "--yes", "--foreground").exit_code == 0

    def drop(services: Services) -> None:
        with services.db.transaction() as session:
            for job in session.scalars(select(Job).where(Job.type == "image_caption")):
                session.delete(job)

    with_services(drop)
    result = cli("images", "caption-backfill", "--days", "36500")
    assert result.exit_code == 0, result.output
    assert "picture(s) queued, estimated $" in result.output
    assert "approve with: twin jobs approve caption-" in result.output
    again = cli("images", "caption-backfill", "--days", "36500")
    assert "no undescribed pictures" in again.output and "already queued" in again.output


def test_caption_backfill_with_no_pictures(data_dir: Path, tmp_path: Path) -> None:
    export = export_with(tmp_path, mix={"text": 100.0})
    cli("import", str(export.root), "--yes", "--foreground")
    result = cli("images", "caption-backfill")
    assert result.exit_code == 0 and "no undescribed pictures in the last 90 days" in result.output


@pytest.fixture
def web() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


def test_stickers_download_queues_a_job_and_runs_it_in_the_foreground(
    data_dir: Path, tmp_path: Path, web: respx.MockRouter
) -> None:
    export = export_with(tmp_path, target_messages=200, local_sticker_ratio=0.0, sticker_kinds=4)
    assert cli("import", str(export.root), "--yes", "--foreground").exit_code == 0
    remote = [info for info in export.stickers.values() if not info.local]
    assert remote
    for info in remote:
        web.get(info.url).respond(200, content=info.data)
    nothing = cli("stickers", "download")
    assert nothing.exit_code == 0 and "already waiting" in nothing.output
    result = cli(
        "--set", "ingest.sticker_download.per_second=50", "stickers", "download", "--foreground"
    )
    assert result.exit_code == 0, result.output

    def statuses(services: Services) -> set[str]:
        with services.db.session() as session:
            return set(session.scalars(select(Sticker.status)))

    assert "available" in result.output
    assert with_services(statuses) <= {"available", "pending"}
    done = cli("stickers", "download")
    assert "no sticker is waiting" in done.output


def test_stickers_download_yields_to_a_running_application(data_dir: Path, tmp_path: Path) -> None:
    export = export_with(tmp_path, target_messages=100, local_sticker_ratio=0.0)
    cli("import", str(export.root), "--yes", "--foreground")
    locks = resolve_paths(load_settings()).locks_dir
    lock = InstanceLock(LOCK_RUN, locks_dir=locks)
    assert lock.acquire()
    try:
        result = cli("stickers", "download", "--foreground")
    finally:
        lock.release()
    assert result.exit_code == 0 and "the application is running" in result.output


def test_target_username_is_masked_in_the_settings_listing(data_dir: Path, tmp_path: Path) -> None:
    export = export_with(tmp_path)
    cli("import", str(export.root), "--yes")
    listing = cli("settings", "list")
    assert "target.username" in listing.output
    assert export.target_username not in listing.output
    history = cli("settings", "history", "target.username")
    assert export.target_username not in history.output
    changed = cli("settings", "set", "target.username", export.target_username)
    assert export.target_username not in changed.output
