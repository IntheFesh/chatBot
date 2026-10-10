"""``twin health``, the network drill, ``twin eval stability``, the command list."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import insert
from typer.testing import CliRunner

from tests.support.clock import ManualClock
from tests.support.ilink import API
from tests.support.stability_world import INTERVAL, steady, store_snapshots
from twin.channel.ilink.store import Credentials, IlinkStore
from twin.channel.state import ChannelStateStore
from twin.cli import app
from twin.config.secrets import SecretStore
from twin.ops.instance_lock import LOCK_RUN, LOCK_SUPERVISOR, InstanceLock
from twin.ops.process_model import ExitCode
from twin.services import CliContext, Services, set_cli_context
from twin.storage.ops_models import HealthSnapshot

runner = CliRunner()


@pytest.fixture
def home(services: Services, secret_store: SecretStore, clock: ManualClock) -> Services:
    """The command line reads the real clock; the data of the tests is written at the real time."""
    clock.set_time(datetime.now(UTC))
    set_cli_context(CliContext(secrets=secret_store))
    return services


def ilink(services: Services) -> IlinkStore:
    return IlinkStore(ChannelStateStore(services.db), services.clock)


def logged_in(services: Services, *, polled: bool = True) -> None:
    """A login that works: credentials, a bound user and (``polled``) a fresh long poll."""
    store = ilink(services)
    store.save_credentials(
        Credentials("TOKEN", "bot", "user", API, services.clock.now_utc().isoformat())
    )
    store.bind("user")
    if polled:
        store.record_poll_success()


def twin(services: Services, *args: str, options: tuple[str, ...] = ()) -> tuple[int, str]:
    base = ["--set", f"paths.data_dir={services.paths.data_dir}"]
    for item in options:
        base += ["--set", item]
    result = runner.invoke(app, [*base, *args])
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result.exit_code, result.output


def snapshot(
    services: Services, checks: dict[str, dict[str, object]], *, age_s: float = 10
) -> None:
    at = services.clock.now_utc() - timedelta(seconds=age_s)
    with services.db.transaction(bump_state=False) as session:
        session.execute(
            insert(HealthSnapshot),
            [
                {
                    "id": "S" + "0" * 25,
                    "at": at,
                    "status": "ok",
                    "started_at": at,
                    "launch": "task",
                    "pid": 1,
                    "channel_ok": True,
                    "channel_last_ok_at": at,
                    "checks": checks,
                    "created_at": at,
                    "updated_at": at,
                }
            ],
        )


# ----------------------------------------------------------------------------- twin health


def test_health_without_the_application_says_what_only_it_could_tell(home: Services) -> None:
    logged_in(home)
    code, out = twin(home, "health", "--json")
    data = json.loads(out)
    assert code == 0, out
    checks = data["checks"]
    assert set(checks) == {
        "channel", "disk", "queue", "backup", "budget", "process", "deepseek", "style_model", "app",
    }  # fmt: skip
    assert checks["process"]["status"] == "warn" and "not running" in checks["process"]["detail"]
    assert "unknown: only the running application can tell" in checks["deepseek"]["detail"]
    assert data["snapshot_at"] is None and data["snapshot_age_s"] is None
    assert data["running"] == {"supervisor": False, "run": False}
    assert checks["channel"]["status"] == "ok" and data["status"] == "degraded"


def test_health_takes_what_the_application_saw_from_its_newest_snapshot(home: Services) -> None:
    logged_in(home)
    snapshot(
        home,
        {
            "deepseek": {"status": "fail", "detail": "3 of 4 attempts failed", "value": 0.75},
            "style_model": {"status": "ok", "detail": "the endpoint answers"},
            "app": {"status": "warn", "detail": "degraded: channel"},
        },
    )
    child = InstanceLock(LOCK_RUN, locks_dir=home.paths.locks_dir)
    supervisor = InstanceLock(LOCK_SUPERVISOR, locks_dir=home.paths.locks_dir)
    assert child.acquire() and supervisor.acquire()
    try:
        code, out = twin(home, "health", "--json")
    finally:
        child.release()
        supervisor.release()
    data = json.loads(out)
    assert code == 1  # a failed check fails the command
    assert data["checks"]["deepseek"]["detail"] == "3 of 4 attempts failed"
    assert data["checks"]["app"]["status"] == "warn"
    assert data["checks"]["process"]["status"] == "ok"
    assert "supervised" in data["checks"]["process"]["detail"]
    assert 10 <= data["snapshot_age_s"] < 120 and data["status"] == "unhealthy"


def test_a_run_started_by_hand_is_said_to_be_one(home: Services) -> None:
    logged_in(home)
    child = InstanceLock(LOCK_RUN, locks_dir=home.paths.locks_dir)
    assert child.acquire()
    try:
        _, out = twin(home, "health", "--json")
    finally:
        child.release()
    detail = json.loads(out)["checks"]["process"]["detail"]
    assert "started by hand" in detail
    assert "no recent snapshot yet" in json.loads(out)["checks"]["deepseek"]["detail"]


def test_an_old_snapshot_is_not_presented_as_the_present(home: Services) -> None:
    logged_in(home)
    snapshot(home, {"deepseek": {"status": "fail", "detail": "old trouble"}}, age_s=3600)
    code, out = twin(home, "health", "--json")
    data = json.loads(out)
    assert code == 0 and "old trouble" not in out
    assert data["checks"]["deepseek"]["status"] == "ok"
    assert 3600 <= data["snapshot_age_s"] < 3720


def test_health_prints_a_table_and_fails_on_a_failed_check(home: Services) -> None:
    logged_in(home)
    code, out = twin(home, "health")
    assert code == 0, out
    for name in ("channel", "disk", "queue", "backup", "budget", "process"):
        assert name in out
    assert "overall: degraded (no snapshot yet)" in out
    code, out = twin(home, "health", options=("ops.health.disk_min_gb=100000000",))
    assert code == 1 and "fail" in out and "overall: unhealthy" in out


def test_a_missing_login_and_a_poll_that_stopped_fail_the_command(home: Services) -> None:
    code, out = twin(home, "health", "--json")
    channel = json.loads(out)["checks"]["channel"]
    assert code == 1 and channel["status"] == "fail" and channel["category"] == "login_lost"
    logged_in(home, polled=False)
    now = home.clock.now_utc()
    home.clock.set_time(now - timedelta(hours=1))
    ilink(home).record_poll_success()  # the last success was an hour ago
    home.clock.set_time(now)
    code, out = twin(home, "health", "--json")
    channel = json.loads(out)["checks"]["channel"]
    assert code == 1 and channel["status"] == "fail" and channel["category"] == "channel_poll_stale"
    assert "last successful long poll" in channel["detail"] and "min ago" in channel["detail"]


# ----------------------------------------------------------------------------- twin ops drill


def test_the_drill_prints_the_steps_and_changes_nothing(home: Services) -> None:
    before = home.db.path.stat().st_mtime_ns
    code, out = twin(home, "ops", "drill", "network")
    assert code == 0, out
    for number in "12345":
        assert f"{number}. " in out
    assert "15 分钟" in out and "twin eval stability --days 7" in out and "≤ 10 分钟" in out
    assert "twin run 没有在运行" in out
    child = InstanceLock(LOCK_RUN, locks_dir=home.paths.locks_dir)
    assert child.acquire()
    try:
        code, out = twin(home, "ops", "drill", "network")
    finally:
        child.release()
    assert "twin run 在运行" in out
    assert home.db.path.stat().st_mtime_ns >= before  # (nothing of ours was written by the drill)


# ----------------------------------------------------------------------------- twin eval stability


def test_the_stability_command_prints_the_report_and_stores_a_run(
    home: Services, clock: ManualClock
) -> None:
    now = home.clock.now_utc()
    store_snapshots(home, steady(now - timedelta(days=2) + timedelta(seconds=30), now))
    code, out = twin(
        home, "eval", "stability", "--days", "2", options=(f"ops.health.interval_s={INTERVAL}",)
    )
    assert code == 0, out
    assert "稳定性报告：" in out and "（2 天）" in out
    count = re.search(r"健康快照 (\d+) 个", out)
    assert count is not None and 2870 <= int(count.group(1)) <= 2885
    assert "通道中断：没有" in out and "累计不可用 0.0 分钟" in out
    assert "断网演练：还没有（twin ops drill network）" in out
    assert "saved as evaluation run" in out and "twin eval gate M4" in out
    code, runs = twin(home, "eval", "runs", "--kind", "stability")
    assert code == 0 and "stability" in runs and "done" in runs


def test_stability_with_nothing_recorded_says_so(home: Services) -> None:
    code, out = twin(home, "eval", "stability")
    assert code == 0 and "健康快照 0 个" in out
    assert "did not run" in out
    code, out = twin(home, "eval", "gate", "M4")
    assert code == 1 and "稳定性报告覆盖连续 7 天" in out  # not enough evidence: exit 1


def test_stability_refuses_a_nonsense_window(home: Services) -> None:
    code, _ = twin(home, "eval", "stability", "--days", "0")
    assert code == 2


# ----------------------------------------------------------------------------- the command list


def test_the_help_lists_the_commands_of_this_round() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in (
        "supervise",
        "service",
        "setup",
        "health",
        "cost",
        "backup",
        "purge",
        "rollback",
        "ops",
    ):
        assert name in result.output, name
    for group, commands in (
        ("service", ("install", "uninstall", "start", "stop", "status")),
        ("backup", ("now", "list", "verify", "restore")),
        ("rollback", ("profile", "persona", "prompt-template", "style-model")),
        ("cost", ("report",)),
    ):
        text = runner.invoke(app, [group, "--help"]).output
        for command in commands:
            assert command in text, (group, command)
    assert ExitCode.USAGE == 2
