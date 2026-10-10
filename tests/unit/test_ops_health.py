"""The health check: thresholds, inputs, the monitor, the weekly integrity check (R-OPS-003)."""

from __future__ import annotations

import shutil
from collections import namedtuple
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.waiting import wait_until
from twin.app import ComponentHealth, HealthStatus
from twin.channel.base import AuthState
from twin.channel.ilink.store import Credentials, IlinkStore
from twin.channel.state import ChannelStateStore
from twin.llm.health import LlmHealth, LlmHealthSnapshot, llm_health_of
from twin.ops.backup.archive import Manifest
from twin.ops.backup.ledger import BackupLedger
from twin.ops.health import (
    HealthCheck,
    HealthCollector,
    HealthLevel,
    LiveSources,
    StyleReading,
    check_backup,
    check_budget,
    check_channel,
    check_components,
    check_deepseek,
    check_disk,
    check_queue,
    check_style,
)
from twin.ops.integrity import IntegrityReport, TableCheck
from twin.ops.jobs import JobQueue
from twin.ops.monitor import INTEGRITY_KEY, HealthMonitor, launch_kind
from twin.services import Services
from twin.storage.models import Alert
from twin.storage.ops_models import HealthSnapshot
from twin.storage.settings_store import get_setting

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
GB = 1024**3


def channel(
    last_ok: timedelta | None,
    *,
    started: timedelta = timedelta(hours=1),
    logged_in: bool = True,
    auth: AuthState = AuthState.OK,
    kind: str = "ilink",
) -> HealthCheck:
    return check_channel(
        kind=kind,
        logged_in=logged_in,
        auth=auth,
        last_ok_at=None if last_ok is None else NOW - last_ok,
        now=NOW,
        started_at=NOW - started,
        stale_min=5,
    )


# ----------------------------------------------------------------------- channel


def test_the_console_channel_needs_no_login() -> None:
    assert channel(None, kind="console").level is HealthLevel.OK


def test_not_logged_in_and_an_expired_login_are_critical() -> None:
    for check in (
        channel(None, logged_in=False),
        channel(timedelta(minutes=1), auth=AuthState.NEEDS_RELOGIN),
    ):
        assert (check.level, check.category, check.severity) == (
            HealthLevel.FAIL,
            "login_lost",
            "critical",
        )


@pytest.mark.parametrize(
    ("age_s", "level"),
    [(30, HealthLevel.OK), (299, HealthLevel.OK), (300, HealthLevel.OK), (301, HealthLevel.FAIL)],
)
def test_a_long_poll_older_than_five_minutes_is_a_failure(age_s: int, level: HealthLevel) -> None:
    check = channel(timedelta(seconds=age_s))
    assert check.level is level
    if level is HealthLevel.FAIL:
        assert check.category == "channel_poll_stale" and check.value == age_s


def test_a_fresh_process_is_given_time_to_poll_for_the_first_time() -> None:
    assert channel(None, started=timedelta(minutes=2)).detail == "starting"
    # a success from before the process started says nothing about the new process
    assert channel(timedelta(minutes=30), started=timedelta(minutes=2)).level is HealthLevel.OK
    stuck = channel(None, started=timedelta(minutes=6))
    assert stuck.level is HealthLevel.FAIL and "since the start" in stuck.detail
    assert channel(timedelta(minutes=30), started=timedelta(minutes=6)).level is HealthLevel.FAIL


# ------------------------------------------------------------------------ the rest


def test_deepseek_errors_and_the_breaker() -> None:
    assert check_deepseek(None, error_rate=0.5, min_calls=10).level is HealthLevel.OK
    breaker = check_deepseek(LlmHealthSnapshot(3, 3, True), error_rate=0.5, min_calls=10)
    assert (breaker.level, breaker.category, breaker.severity) == (
        HealthLevel.FAIL,
        "circuit_open",
        "critical",
    )
    few = check_deepseek(LlmHealthSnapshot(9, 9, False), error_rate=0.5, min_calls=10)
    assert few.level is HealthLevel.OK  # nine attempts say nothing
    bad = check_deepseek(LlmHealthSnapshot(10, 5, False), error_rate=0.5, min_calls=10)
    assert bad.level is HealthLevel.FAIL and bad.category == "deepseek_failure"
    fine = check_deepseek(LlmHealthSnapshot(10, 4, False), error_rate=0.5, min_calls=10)
    assert fine.level is HealthLevel.OK


def test_the_style_model_is_checked_only_when_the_backend_needs_it() -> None:
    assert check_style(None).level is HealthLevel.OK
    assert "not needed" in check_style(StyleReading(False, True)).detail
    assert check_style(StyleReading(True, True)).level is HealthLevel.OK
    down = check_style(StyleReading(True, False, "unhealthy"))
    assert (down.level, down.category) == (HealthLevel.FAIL, "style_model_down")


def test_disk_below_five_gigabytes_is_a_failure() -> None:
    assert check_disk(5 * GB, min_gb=5).level is HealthLevel.OK
    low = check_disk(int(4.9 * GB), min_gb=5)
    assert (low.level, low.category, low.severity) == (HealthLevel.FAIL, "disk_low", "warning")
    assert check_disk(2 * GB, min_gb=5).severity == "critical"


def test_the_queue_counts_jobs_and_age() -> None:
    assert check_queue(500, None, max_jobs=500, oldest_h=24).level is HealthLevel.OK
    many = check_queue(501, None, max_jobs=500, oldest_h=24)
    assert (many.level, many.category) == (HealthLevel.FAIL, "queue_backlog")
    assert check_queue(3, timedelta(hours=24), max_jobs=500, oldest_h=24).level is HealthLevel.OK
    old = check_queue(3, timedelta(hours=24, seconds=1), max_jobs=500, oldest_h=24)
    assert old.level is HealthLevel.FAIL and old.category == "queue_backlog"


def test_the_newest_backup_must_be_younger_than_36_hours() -> None:
    day = timedelta(hours=1)
    assert (
        check_backup(36 * day, since_first_run=timedelta(days=9), stale_h=36).level
        is HealthLevel.OK
    )
    old = check_backup(
        timedelta(hours=36, minutes=6), since_first_run=timedelta(days=9), stale_h=36
    )
    assert (old.level, old.category) == (HealthLevel.FAIL, "backup_failed")
    # before the first backup: a warning, then a failure once the grace period is over
    assert check_backup(None, since_first_run=10 * day, stale_h=36).level is HealthLevel.WARN
    never = check_backup(None, since_first_run=37 * day, stale_h=36)
    assert (never.level, never.category) == (HealthLevel.FAIL, "backup_failed")


def test_the_budget_level_is_shown_but_raises_no_alert_of_its_own() -> None:
    assert check_budget(0, 0.2, 0.1).level is HealthLevel.OK
    degraded = check_budget(2, 1.3, 0.4)
    assert degraded.level is HealthLevel.WARN and degraded.category is None


def test_unhealthy_components_are_a_failure() -> None:
    ok = {"a": ComponentHealth(HealthStatus.OK)}
    assert check_components(ok).level is HealthLevel.OK
    assert check_components(None).level is HealthLevel.OK
    weak = {**ok, "b": ComponentHealth(HealthStatus.DEGRADED, "x")}
    assert check_components(weak).level is HealthLevel.WARN
    bad = {**weak, "c": ComponentHealth(HealthStatus.UNHEALTHY, "y")}
    result = check_components(bad)
    assert result.level is HealthLevel.FAIL and "c" in result.detail


def test_a_check_survives_its_json_form() -> None:
    check = check_disk(2 * GB, min_gb=5)
    again = HealthCheck.from_json("disk", check.to_json())
    assert again == check


# ------------------------------------------------------------------- the collector


def log_in(services: Services) -> IlinkStore:
    store = IlinkStore(ChannelStateStore(services.db), services.clock)
    store.save_credentials(
        Credentials(
            "token", "bot", "user", "https://example.invalid", services.clock.now_utc().isoformat()
        )
    )
    return store


async def test_the_collector_reads_the_channel_from_the_stored_state(
    services: Services, clock: ManualClock
) -> None:
    collector = HealthCollector(services, started_at=clock.now_utc() - timedelta(hours=1))
    report = await collector.collect()
    channel_check = report.get("channel")
    assert channel_check is not None and channel_check.category == "login_lost"  # nobody logged in

    store = log_in(services)
    store.record_poll_success()
    report = await collector.collect()
    assert report.get("channel").level is HealthLevel.OK  # type: ignore[union-attr]
    assert report.channel_ok is True and report.channel_last_ok_at == clock.now_utc()
    clock.tick(6 * 60)
    stale = await collector.collect()
    assert stale.get("channel").category == "channel_poll_stale"  # type: ignore[union-attr]
    assert stale.channel_ok is False and stale.status == "unhealthy"
    store.mark_needs_relogin(-14, "expired")
    assert (await collector.collect()).get("channel").category == "login_lost"  # type: ignore[union-attr]


async def test_the_collector_counts_the_ready_jobs(services: Services, clock: ManualClock) -> None:
    services.settings.ops.health.queue_max = 2
    queue = JobQueue(services.db, clock)
    for _ in range(3):
        queue.enqueue("noop", {})
    # a job that waits for the user's approval is not a backlog
    queue.enqueue("batch", {}, batch_id="b", requires_approval=True)
    # one that is not due yet is not either
    queue.enqueue("later", {}, run_after=clock.now_utc() + timedelta(hours=1))
    collector = HealthCollector(services)
    report = await collector.collect()
    check = report.get("queue")
    assert check is not None and check.level is HealthLevel.FAIL and check.value == 3
    clock.tick(25 * 3600)
    services.settings.ops.health.queue_max = 500
    old = (await collector.collect()).get("queue")
    assert old is not None and "h old" in old.detail and old.level is HealthLevel.FAIL


async def test_the_collector_reads_the_age_of_the_newest_backup(
    services: Services, clock: ManualClock
) -> None:
    collector = HealthCollector(services, started_at=clock.now_utc())
    assert (await collector.collect()).get("backup").level is HealthLevel.WARN  # type: ignore[union-attr]
    manifest = Manifest(
        created_at=clock.now_utc().isoformat(),
        local_date="2026-10-09",
        kind="daily",
        schema_revision=None,
        key_id=1,
        key_ids=[1],
        tables={},
        media=[],
        db_sha256="0" * 64,
    )
    BackupLedger(services.db, clock).add_ok(
        manifest,
        kind="daily",
        file_name="twin-2026-10-09.bak.enc",
        size_bytes=1,
        sha256="0" * 64,
        duration_ms=1,
    )
    assert (await collector.collect()).get("backup").level is HealthLevel.OK  # type: ignore[union-attr]
    clock.tick(37 * 3600)
    assert (await collector.collect()).get("backup").level is HealthLevel.FAIL  # type: ignore[union-attr]


async def test_the_disk_check_reads_the_data_directory(
    services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage(100 * GB, 98 * GB, 2 * GB))
    check = (await HealthCollector(services).collect()).get("disk")
    assert check is not None and check.category == "disk_low"


async def test_the_live_checks_come_from_the_process(
    services: Services, clock: ManualClock
) -> None:
    live = LiveSources(
        components=lambda: {"engine": ComponentHealth(HealthStatus.UNHEALTHY, "dead")},
        style=lambda: _style(StyleReading(True, False, "unhealthy")),
    )
    health = llm_health_of(services)
    for _ in range(10):
        health.record(False)
    collector = HealthCollector(services, live=live)
    report = await collector.collect()
    assert report.get("deepseek").category == "deepseek_failure"  # type: ignore[union-attr]
    assert report.get("style_model").category == "style_model_down"  # type: ignore[union-attr]
    assert report.get("app").level is HealthLevel.FAIL  # type: ignore[union-attr]


async def _style(reading: StyleReading) -> StyleReading:
    return reading


async def test_other_components_register_their_own_probe(services: Services) -> None:
    """R-SRV-002: the llama-server of round 14 puts its check into the report this way."""
    collector = HealthCollector(services)

    async def llama() -> HealthCheck:
        return HealthCheck("llama_server", HealthLevel.FAIL, "exited", category="style_model_down")

    async def broken() -> HealthCheck:
        raise RuntimeError("boom")

    collector.register_probe("llama_server", llama)
    collector.register_probe("odd", broken)
    report = await collector.collect()
    assert report.get("llama_server").category == "style_model_down"  # type: ignore[union-attr]
    odd = report.get("odd")
    assert odd is not None and odd.level is HealthLevel.FAIL and "RuntimeError" in odd.detail


def test_llm_health_counts_a_window_of_attempts(clock: ManualClock) -> None:
    health = LlmHealth(clock)
    health.record(True)
    health.record(False)
    clock.tick(600)
    health.record(False)
    assert health.snapshot(900).calls == 3 and health.snapshot(900).failures == 2
    recent = health.snapshot(60)
    assert (recent.calls, recent.failures, recent.error_rate) == (1, 1, 1.0)
    health.breaker_opened()
    assert health.snapshot(60).circuit_open
    health.breaker_closed()
    assert not health.snapshot(60).circuit_open
    clock.tick(120)  # nothing new for two minutes: the one-minute window is empty
    assert health.snapshot(1).calls == 0 and health.snapshot(1).error_rate == 0.0


# --------------------------------------------------------------------- the monitor


def snapshots(services: Services) -> list[HealthSnapshot]:
    with services.db.session() as session:
        rows = list(session.scalars(select(HealthSnapshot).order_by(HealthSnapshot.at)))
        session.expunge_all()  # the rollback at the end of the session would expire them
        return rows


def alerts_of(services: Services) -> list[Alert]:
    with services.db.session() as session:
        rows = list(session.scalars(select(Alert).order_by(Alert.created_at, Alert.id)))
        session.expunge_all()
        return rows


def make_monitor(services: Services, clock: ManualClock, **options: object) -> HealthMonitor:
    started = clock.now_utc() - timedelta(hours=1)
    collector = HealthCollector(services, started_at=started)
    return HealthMonitor(  # type: ignore[arg-type]
        services, collector, launch="task", started_at=started, **options
    )


def clean_integrity(at: datetime = NOW) -> IntegrityReport:
    return IntegrityReport(at, True, ("ok",), ())


async def test_every_look_is_stored_with_how_the_process_was_started(
    services: Services, clock: ManualClock
) -> None:
    monitor = make_monitor(services, clock, integrity=clean_integrity)
    log_in(services).record_poll_success()
    report = await monitor.tick()
    (row,) = snapshots(services)
    assert (row.at, row.launch, row.status) == (clock.now_utc(), "task", report.status)
    assert row.started_at == clock.now_utc() - timedelta(hours=1)
    assert row.channel_ok is True and row.channel_last_ok_at == clock.now_utc()
    assert {"channel", "disk", "queue", "backup", "budget"} <= set(row.checks)
    assert row.checks["channel"]["status"] == "ok"
    with services.db.session() as session:
        assert get_setting(session, "ops.first_run_at") == clock.now_utc().isoformat()


def test_the_launch_kind_comes_from_the_environment() -> None:
    assert launch_kind({"TWIN_LAUNCH": "task"}) == "task"
    assert launch_kind({"TWIN_LAUNCH": "manual"}) == "manual"
    assert launch_kind({"TWIN_LAUNCH": "weird"}) == "unsupervised"
    assert launch_kind({}) == "unsupervised"


async def test_a_failure_is_announced_once_reminded_later_and_closed(
    services: Services, clock: ManualClock
) -> None:
    monitor = make_monitor(services, clock, integrity=clean_integrity)
    store = log_in(services)
    store.record_poll_success()
    await monitor.tick()
    assert [a.category for a in alerts_of(services) if a.kind == "alert"] == []

    clock.tick(6 * 60)  # no poll for six minutes
    await monitor.tick()
    (alert,) = [a for a in alerts_of(services) if a.kind == "alert"]
    assert (alert.category, alert.severity, alert.toast_state) == (
        "channel_poll_stale",
        "warning",
        "pending",
    )
    clock.tick(60)
    await monitor.tick()  # still failing: not announced again
    assert len([a for a in alerts_of(services) if a.kind == "alert"]) == 1

    clock.tick(6 * 3600)  # the reminder after remind_h
    store.state.put("ilink.poll_status", {"consecutive_failures": 0, "last_ok_at": None})
    await monitor.tick()
    assert len([a for a in alerts_of(services) if a.kind == "alert"]) == 2

    store.record_poll_success()  # the poll works again
    clock.tick(60)
    await monitor.tick()
    kinds = [(a.kind, a.resolved_at is not None) for a in alerts_of(services)]
    assert ("recovery", False) in kinds
    assert all(done for kind, done in kinds if kind == "alert")


async def test_a_failure_that_changes_its_nature_is_announced_as_the_new_one(
    services: Services, clock: ManualClock
) -> None:
    monitor = make_monitor(services, clock, integrity=clean_integrity)
    store = log_in(services)
    await monitor.tick()
    clock.tick(6 * 60)
    await monitor.tick()
    store.mark_needs_relogin(-14, "expired")
    await monitor.tick()
    categories = [a.category for a in alerts_of(services) if a.kind == "alert"]
    assert categories == ["channel_poll_stale", "login_lost"]


async def test_old_snapshots_and_alerts_are_removed(services: Services, clock: ManualClock) -> None:
    monitor = make_monitor(services, clock, integrity=clean_integrity)
    log_in(services).record_poll_success()
    await monitor.tick()
    clock.tick(31 * 24 * 3600)
    services.alerts.raise_alert("job_failed", "old")
    clock.tick(60 * 24 * 3600)
    log_in(services)
    await monitor.tick()  # the first look of the later day prunes (an hour has passed)
    assert len(snapshots(services)) == 1  # only the new one
    assert monitor.prune() == 0


async def test_the_integrity_check_runs_on_sundays_at_half_past_three(
    services: Services, clock: ManualClock
) -> None:
    runs: list[datetime] = []

    def integrity() -> IntegrityReport:
        runs.append(clock.now_utc())
        return clean_integrity(clock.now_utc())  # the report says when it was made

    monitor = make_monitor(services, clock, integrity=integrity)
    assert monitor.integrity_due()  # never made
    monitor.run_integrity_now()
    assert len(runs) == 1 and not monitor.integrity_due()
    with services.db.session() as session:
        assert get_setting(session, INTEGRITY_KEY)["ok"] is True
    # Friday noon -> the coming Sunday 03:30 Chicago (08:30 UTC) is what is next due
    clock.set_time(datetime(2026, 10, 11, 8, 29, tzinfo=UTC))
    assert not monitor.integrity_due()
    clock.set_time(datetime(2026, 10, 11, 8, 31, tzinfo=UTC))
    assert monitor.integrity_due()
    monitor.run_integrity_now()
    clock.set_time(datetime(2026, 10, 14, 12, 0, tzinfo=UTC))
    assert not monitor.integrity_due()  # once a week
    clock.set_time(datetime(2026, 10, 18, 9, 0, tzinfo=UTC))
    assert monitor.integrity_due()


async def test_a_damaged_database_or_index_is_a_critical_alert_until_clean(
    services: Services, clock: ManualClock
) -> None:
    bad = IntegrityReport(NOW, False, ("row 3 missing",), (TableCheck("memory_facts", 10, 2),))
    reports = [bad, clean_integrity()]
    monitor = make_monitor(services, clock, integrity=lambda: reports.pop(0))
    report = monitor.run_integrity_now()
    assert not report.ok and "twin memory reindex" in "\n".join(report.problems())
    (alert,) = alerts_of(services)
    assert (alert.category, alert.severity) == ("integrity_failed", "critical")
    assert alert.detail == {"database_ok": False, "orphans": 2}
    monitor.run_integrity_now()
    assert [a.kind for a in alerts_of(services)] == ["alert", "recovery"]


async def test_the_loops_run_under_the_clock(services: Services, clock: ManualClock) -> None:
    monitor = make_monitor(services, clock, integrity=clean_integrity)
    log_in(services).record_poll_success()
    await monitor.start()
    try:
        await wait_until(lambda: monitor.ticks >= 1)
        await clock.advance(60)
        await wait_until(lambda: monitor.ticks >= 2)
    finally:
        await monitor.stop()
    assert monitor.health().status is HealthStatus.OK
    assert len(snapshots(services)) >= 2
