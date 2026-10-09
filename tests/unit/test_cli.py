"""The ``twin`` command line: help, doctor, config, settings, secrets, db, jobs, run."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import threading
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from tests.support.synthetic import wxid
from tests.support.waiting import wait_until
from twin import __version__
from twin import cli as cli_module
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.ops.instance_lock import LOCK_RUN, InstanceLock
from twin.ops.jobs import JobContext, JobQueue, default_registry
from twin.ops.process_model import ExitCode
from twin.services import Services, build_services
from twin.storage.db import Database
from twin.storage.migrate import head_revision

runner = CliRunner()
SECRET = "sk-synthetic-not-a-real-key-123456"


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    return path


@pytest.fixture
def initialised(data_dir: Path) -> Path:
    result = runner.invoke(app, ["db", "upgrade"])
    assert result.exit_code == 0, result.output
    return data_dir


def open_services() -> Services:
    return build_services(load_settings(), root=resolve_paths(load_settings()).root)


# ------------------------------------------------------------ help and version


def test_help_lists_every_command_group() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for word in ("run", "doctor", "config", "settings", "secrets", "db", "jobs"):
        assert word in result.output


def test_running_without_arguments_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_version_option() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and __version__ in result.output


@pytest.mark.parametrize("group", ["config", "settings", "secrets", "db", "jobs"])
def test_group_help(group: str) -> None:
    result = runner.invoke(app, [group, "--help"])
    assert result.exit_code == 0 and "Usage" in result.output


# --------------------------------------------------------------------- doctor


def test_doctor_passes_on_a_fresh_machine_and_diagnoses_the_keyring(data_dir: Path) -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    out = result.output
    for check in ("python", "dependencies", "tzdata", "keyring", "data-dir", "database", "power"):
        assert check in out
    assert "encrypted file fallback" in out  # explicit, diagnosable (WARN not FAIL)
    assert "twin db upgrade" in out  # uninitialised database is a hint, not a failure
    assert "FAIL" not in out.upper().replace("FAILED", "")


def test_doctor_reports_an_unwritable_data_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "a-file"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(blocker / "data"))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "not writable" in result.output


def test_doctor_survives_a_broken_configuration(tmp_path: Path) -> None:
    broken = tmp_path / "broken.yaml"
    broken.write_text("surprise: true\n", encoding="utf-8")
    result = runner.invoke(app, ["--config", str(broken), "doctor"])
    assert result.exit_code == 1
    assert (
        "Extra inputs are not permitted" in result.output.replace("\n", " ")
        or "invalid" in result.output
    )


def test_doctor_reports_an_initialised_database(initialised: Path) -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "up to date" in result.output


# ----------------------------------------------------------------------- config


def test_config_show_masks_personal_values_and_honours_overrides() -> None:
    result = runner.invoke(
        app,
        ["--set", f"target.username={wxid()}", "--set", "budget.daily_usd=2.5", "config", "show"],
    )
    assert result.exit_code == 0, result.output
    shown = yaml.safe_load(result.output)
    assert shown["budget"]["daily_usd"] == 2.5
    assert "synthetic123" not in result.output and "wx***23" in result.output
    assert shown["time"]["bot_timezone"] == "America/Chicago"


def test_config_file_option(tmp_path: Path) -> None:
    config = tmp_path / "mine.yaml"
    config.write_text("time: { bot_timezone: Asia/Shanghai }\n", encoding="utf-8")
    result = runner.invoke(app, ["--config", str(config), "config", "show"])
    assert yaml.safe_load(result.output)["time"]["bot_timezone"] == "Asia/Shanghai"


# --------------------------------------------------------------------- db


def test_db_upgrade_and_status(data_dir: Path) -> None:
    before = runner.invoke(app, ["db", "status"])
    assert before.exit_code == 0 and "missing" in before.output
    upgraded = runner.invoke(app, ["db", "upgrade"])
    assert upgraded.exit_code == 0 and f"new -> {head_revision()}" in upgraded.output
    after = runner.invoke(app, ["db", "status"])
    assert f"current (applied {head_revision()}, latest {head_revision()})" in after.output
    again = runner.invoke(app, ["db", "upgrade"])
    assert f"{head_revision()} -> {head_revision()}" in again.output


# ------------------------------------------------------------------- settings


def test_settings_set_list_and_history(initialised: Path) -> None:
    changed = runner.invoke(app, ["settings", "set", "time.bot_timezone", "Asia/Shanghai"])
    assert changed.exit_code == 0 and "Asia/Shanghai" in changed.output
    unchanged = runner.invoke(app, ["settings", "set", "time.bot_timezone", "Asia/Shanghai"])
    assert "unchanged" in unchanged.output
    listing = runner.invoke(app, ["settings", "list"])
    assert (
        listing.exit_code == 0 and "Asia/Shanghai" in listing.output and "paused" in listing.output
    )
    history = runner.invoke(app, ["settings", "history", "time.bot_timezone"])
    assert (
        history.exit_code == 0
        and "cli" in history.output
        and "America/Chicago" not in history.output.split("old")[0]
    )
    flag = runner.invoke(app, ["settings", "set", "paused", "true"])
    assert flag.exit_code == 0


def test_settings_errors(initialised: Path) -> None:
    unknown = runner.invoke(app, ["settings", "set", "nope", "1"])
    assert unknown.exit_code == 1 and "unknown setting" in unknown.output
    invalid = runner.invoke(app, ["settings", "set", "time.bot_timezone", "Mars/Base"])
    assert invalid.exit_code == ExitCode.USAGE and "unknown IANA time zone" in invalid.output
    unparsable = runner.invoke(app, ["settings", "set", "paused", "[unclosed"])
    assert unparsable.exit_code == ExitCode.USAGE


# -------------------------------------------------------------------- secrets


def test_secrets_set_list_check_delete_never_reveal_the_value(initialised: Path) -> None:
    missing = runner.invoke(app, ["secrets", "check", "deepseek_api_key"])
    assert missing.exit_code == 1 and "not set" in missing.output
    stored = runner.invoke(
        app, ["secrets", "set", "deepseek_api_key", "--stdin"], input=SECRET + "\n"
    )
    assert stored.exit_code == 0 and SECRET not in stored.output
    present = runner.invoke(app, ["secrets", "check", "deepseek_api_key"])
    assert present.exit_code == 0 and "set" in present.output and SECRET not in present.output
    listing = runner.invoke(app, ["secrets", "list"])
    assert listing.exit_code == 0 and SECRET not in listing.output
    assert "deepseek_api_key" in listing.output and "yes" in listing.output
    assert "credential store" in listing.output
    deleted = runner.invoke(app, ["secrets", "delete", "deepseek_api_key"])
    assert "deleted" in deleted.output
    assert "was not set" in runner.invoke(app, ["secrets", "delete", "deepseek_api_key"]).output


def test_secrets_set_prompts_with_hidden_input(initialised: Path) -> None:
    result = runner.invoke(app, ["secrets", "set", "smtp_password"], input=f"{SECRET}\n{SECRET}\n")
    assert result.exit_code == 0
    assert SECRET not in result.output  # hidden prompt does not echo
    check = runner.invoke(app, ["secrets", "check", "smtp_password"])
    assert check.exit_code == 0


def test_secrets_rejects_unknown_names_and_empty_values(initialised: Path) -> None:
    unknown = runner.invoke(app, ["secrets", "set", "not_a_secret", "--stdin"], input="x\n")
    assert unknown.exit_code == ExitCode.USAGE and "unknown secret" in unknown.output
    empty = runner.invoke(app, ["secrets", "set", "smtp_password", "--stdin"], input="\n")
    assert empty.exit_code == ExitCode.USAGE and "empty" in empty.output


def test_secrets_set_works_before_the_database_exists(data_dir: Path) -> None:
    result = runner.invoke(
        app, ["secrets", "set", "deepseek_api_key", "--stdin"], input=SECRET + "\n"
    )
    assert result.exit_code == 0, result.output  # no state to bump yet: not an error


def test_secrets_set_bumps_state_version_so_a_running_app_reloads(initialised: Path) -> None:
    from twin.storage.state import read_state_version

    runner.invoke(app, ["secrets", "set", "deepseek_api_key", "--stdin"], input=SECRET + "\n")
    services = open_services()
    try:
        with services.db.session() as session:
            assert read_state_version(session) == 1
    finally:
        services.close()


def test_rotate_db_key_command(initialised: Path) -> None:
    queue_owner = open_services()
    try:
        queue = JobQueue(queue_owner.db, queue_owner.clock)
        for index in range(3):
            queue.enqueue("rotate_demo", {"n": index, "text": "正文"})
        queue_owner.runtime.initialize()
    finally:
        queue_owner.close()
    result = runner.invoke(app, ["secrets", "rotate-db-key"])
    assert result.exit_code == 0, result.output
    assert "new key id 2" in result.output and "retired keys [1]" in result.output
    listing = runner.invoke(app, ["secrets", "list"])
    assert (
        "db-key-1" in listing.output and "retired" in listing.output and "current" in listing.output
    )
    check = open_services()
    try:
        jobs = JobQueue(check.db, check.clock).list_jobs(job_type="rotate_demo")
        assert sorted(job.payload["n"] for job in jobs) == [0, 1, 2]
    finally:
        check.close()


def test_secrets_list_before_any_key_exists(data_dir: Path) -> None:
    result = runner.invoke(app, ["secrets", "list"])
    assert result.exit_code == 0 and "created on first start" in result.output


# ----------------------------------------------------------------------- jobs


@pytest.fixture
def demo_handler() -> None:
    ran: list[int] = []

    async def handler(ctx: JobContext) -> None:
        ran.append(ctx.job.payload["n"])
        if ctx.job.payload.get("fail"):
            raise RuntimeError("synthetic failure")

    default_registry.register("cli_demo", handler)
    yield ran  # type: ignore[misc]
    default_registry._handlers.pop("cli_demo", None)


def enqueue(*payloads: dict[str, object], **options: object) -> list[str]:
    services = open_services()
    try:
        queue = JobQueue(services.db, services.clock)
        return [queue.enqueue("cli_demo", p, **options) for p in payloads]  # type: ignore[arg-type]
    finally:
        services.close()


def test_jobs_list_marks_jobs_without_a_handler(initialised: Path) -> None:
    enqueue({"n": 1})
    services = open_services()
    try:
        JobQueue(services.db, services.clock).enqueue("orphan_type", {"x": 1})
    finally:
        services.close()
    result = runner.invoke(app, ["jobs", "list"])
    assert result.exit_code == 0, result.output
    assert "orphan_type" in result.output and "无处理器" in result.output
    assert "pending=2" in result.output
    filtered = runner.invoke(app, ["jobs", "list", "--type", "orphan_type", "--status", "pending"])
    assert "orphan_type" in filtered.output and "cli_demo" not in filtered.output
    assert runner.invoke(app, ["jobs", "list", "--status", "bogus"]).exit_code == ExitCode.USAGE


def test_jobs_show_hides_the_payload_unless_asked(initialised: Path) -> None:
    (job_id,) = enqueue({"n": 5, "text": "秘密内容"})
    plain = runner.invoke(app, ["jobs", "show", job_id])
    assert plain.exit_code == 0 and "cli_demo" in plain.output and "秘密内容" not in plain.output
    full = runner.invoke(app, ["jobs", "show", job_id, "--payload"])
    assert "秘密内容" in full.output
    missing = runner.invoke(app, ["jobs", "show", "01NOSUCHJOB"])
    assert missing.exit_code == 1 and "no job" in missing.output


def test_jobs_cancel_and_retry(initialised: Path) -> None:
    (job_id,) = enqueue({"n": 1})
    assert runner.invoke(app, ["jobs", "retry", job_id]).exit_code == 1  # still pending
    cancelled = runner.invoke(app, ["jobs", "cancel", job_id])
    assert cancelled.exit_code == 0 and "cancelled" in cancelled.output
    assert runner.invoke(app, ["jobs", "cancel", job_id]).exit_code == 1
    retried = runner.invoke(app, ["jobs", "retry", job_id])
    assert retried.exit_code == 0 and "queued again" in retried.output
    assert runner.invoke(app, ["jobs", "cancel", "missing"]).exit_code == 1


def test_jobs_run_until_idle_executes_in_the_foreground(
    initialised: Path, demo_handler: list[int]
) -> None:
    enqueue({"n": 1}, {"n": 2})
    result = runner.invoke(app, ["jobs", "run", "--until-idle"])
    assert result.exit_code == 0, result.output
    assert "done=2" in result.output and sorted(demo_handler) == [1, 2]
    listing = runner.invoke(app, ["jobs", "list", "--status", "done"])
    assert "done=2" in listing.output


def test_jobs_run_requires_the_flag_and_defers_to_a_running_application(
    initialised: Path, demo_handler: list[int]
) -> None:
    enqueue({"n": 1})
    assert runner.invoke(app, ["jobs", "run"]).exit_code == ExitCode.USAGE
    services = open_services()
    lock = InstanceLock(LOCK_RUN, locks_dir=services.paths.locks_dir)
    assert lock.acquire()
    try:
        refused = runner.invoke(app, ["jobs", "run", "--until-idle"])
        assert refused.exit_code == ExitCode.BUSY and "--force" in refused.output
        forced = runner.invoke(app, ["jobs", "run", "--until-idle", "--force"])
        assert forced.exit_code == 0 and demo_handler == [1]
    finally:
        lock.release()
        services.close()


def test_forced_second_worker_leaves_the_running_applications_jobs_alone(
    initialised: Path, demo_handler: list[int]
) -> None:
    (busy_id,) = enqueue({"n": 1})
    services = open_services()
    queue = JobQueue(services.db, services.clock)
    claimed = queue.claim_next({"cli_demo"}, offpeak_allowed=True, worker_id="the-running-app")
    assert claimed is not None and claimed.id == busy_id
    enqueue({"n": 2})
    lock = InstanceLock(LOCK_RUN, locks_dir=services.paths.locks_dir)
    assert lock.acquire()
    try:
        forced = runner.invoke(app, ["jobs", "run", "--until-idle", "--force"])
    finally:
        lock.release()
    assert forced.exit_code == 0 and demo_handler == [2]
    still_running = queue.get(busy_id)
    services.close()
    assert still_running is not None and still_running.status == "running"


def test_jobs_run_reports_failures(initialised: Path, demo_handler: list[int]) -> None:
    enqueue({"n": 1, "fail": True}, max_attempts=1)
    result = runner.invoke(app, ["jobs", "run", "--until-idle"])
    assert "failed=1" in result.output
    shown = runner.invoke(app, ["jobs", "list", "--status", "failed"])
    assert "failed=1" in shown.output


def test_jobs_approve_flow(initialised: Path, demo_handler: list[int]) -> None:
    enqueue({"n": 1}, {"n": 2}, batch_id="replay-7", estimated_cost_usd=4.0, requires_approval=True)
    listing = runner.invoke(app, ["jobs", "list", "--batch", "replay-7"])
    assert "待批准" in listing.output
    assert runner.invoke(app, ["jobs", "run", "--until-idle"]).output.count("done=0") == 1
    declined = runner.invoke(app, ["jobs", "approve", "replay-7"], input="n\n")
    assert declined.exit_code == 1 and "estimated $8.00" in declined.output
    approved = runner.invoke(app, ["jobs", "approve", "replay-7"], input="y\n")
    assert approved.exit_code == 0 and "approved 2 job(s), $8.00" in approved.output
    assert "done=2" in runner.invoke(app, ["jobs", "run", "--until-idle"]).output
    again = runner.invoke(app, ["jobs", "approve", "replay-7", "--yes"])
    assert again.exit_code == 1 and "waiting" in again.output


def test_jobs_approve_refuses_oversized_and_unknown_batches(initialised: Path) -> None:
    enqueue({"n": 1}, batch_id="huge", estimated_cost_usd=99.0, requires_approval=True)
    too_big = runner.invoke(app, ["jobs", "approve", "huge", "--yes"])
    assert too_big.exit_code == 1 and "split it into smaller batches" in too_big.output
    unknown = runner.invoke(app, ["jobs", "approve", "nothing", "--yes"])
    assert unknown.exit_code == 1 and "no jobs belong to batch" in unknown.output


# ------------------------------------------------------------------------ run


def test_run_command_wires_logging_masked_config_runtime_settings_and_the_lock(
    initialised: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served: list[bool] = []

    async def fake_serve(services: Services) -> None:
        # The lock is a file on POSIX but a named mutex on Windows, so ask the lock itself.
        probe = InstanceLock(LOCK_RUN, locks_dir=services.paths.locks_dir)
        served.append(await asyncio.to_thread(probe.is_held_elsewhere))  # held while serving

    monkeypatch.setattr(cli_module, "_serve", fake_serve)
    result = runner.invoke(app, ["--set", f"target.username={wxid()}", "run"])
    assert result.exit_code == 0, result.output
    assert "effective configuration" in result.output and "synthetic123" not in result.output
    assert served == [True]
    assert not InstanceLock(LOCK_RUN, locks_dir=initialised / "locks").is_held_elsewhere()
    assert (initialised / "logs" / "twin.log").exists()
    services = open_services()
    try:
        assert services.runtime.snapshot()["time.bot_timezone"] == "America/Chicago"
        with services.db.session() as session:
            from twin.storage.settings_store import has_setting

            assert has_setting(session, "paused")  # runtime settings were seeded
    finally:
        services.close()


def test_run_command_refuses_a_second_instance_and_a_missing_database(
    data_dir: Path, initialised: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served: list[bool] = []

    async def fake_serve(services: Services) -> None:
        served.append(True)

    monkeypatch.setattr(cli_module, "_serve", fake_serve)
    lock = InstanceLock(LOCK_RUN, locks_dir=initialised / "locks")
    assert lock.acquire()
    try:
        result = runner.invoke(app, ["run"])
    finally:
        lock.release()
    assert result.exit_code == ExitCode.BUSY and "already running" in result.output
    assert served == []


def test_run_command_reports_a_component_that_fails_to_start(
    initialised: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from twin.app import ComponentStartError

    async def failing_serve(services: Services) -> None:
        raise ComponentStartError("job_worker", RuntimeError("no database"))

    monkeypatch.setattr(cli_module, "_serve", failing_serve)
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 1 and "'job_worker' failed to start" in result.output


@pytest.mark.skipif(
    sys.platform == "win32", reason="os.kill(SIGTERM) terminates a Windows process outright"
)
async def test_serve_runs_the_application_until_a_termination_signal(
    services: Services,
) -> None:
    """The real wiring: components start, SIGTERM stops them gracefully."""
    started = asyncio.create_task(cli_module._serve(services))
    from twin.storage.settings_store import get_setting

    def heartbeat_written() -> bool:
        with services.db.session() as session:
            return get_setting(session, "heartbeat") is not None

    await wait_until(heartbeat_written)
    await asyncio.sleep(0.1)
    timer = threading.Timer(0.0, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.start()
    await asyncio.wait_for(started, timeout=10)
    timer.join()


def test_python_m_twin_entry_point_exists() -> None:
    import importlib

    module = importlib.import_module("twin.__main__")
    assert module.app is app
    Database  # noqa: B018  (imported for type availability in fixtures)
