"""The health monitor: a component that looks every minute and says when something is wrong.

:class:`HealthMonitor` runs :class:`~twin.ops.health.HealthCollector` every
``ops.health.interval_s`` (60 s) and

* stores the result in ``health_snapshots`` (together with when the process started and how it
  was launched - ``TWIN_LAUNCH`` is set by ``twin supervise`` to ``task`` or ``manual`` - which is
  what the stability report is made from, R-EVAL-006); rows older than ``ops.health.keep_days``
  are deleted once an hour;
* turns a failing check into an alert the moment it starts failing (the category the check names),
  again every ``ops.health.remind_h`` hours while it goes on, and closes it with a "recovered"
  notice when the check is good again (R-OPS-004);
* every Sunday at 03:30 local time runs the integrity check (:mod:`twin.ops.integrity`) in a
  task of its own - it can take minutes on a large database and must not delay the snapshots -
  keeps the result in the setting ``ops.integrity.last`` and raises ``integrity_failed`` if it is
  not clean.  A check that was missed because the application was not running is made when it
  next starts.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import delete

from twin.app import ComponentHealth, TaskSupervisor
from twin.ops.health import FIRST_RUN_KEY, HealthCollector, HealthLevel, HealthReport
from twin.ops.integrity import IntegrityReport, run_integrity
from twin.ops.logging import get_logger
from twin.ops.recurring import Weekly, latest_due
from twin.retrieval.vector_store import VectorStore
from twin.schedule.service import time_service_for
from twin.services import Services
from twin.storage.ops_models import LAUNCH_KINDS, HealthSnapshot
from twin.storage.settings_store import get_setting, put_setting

log = get_logger("twin.health")

INTEGRITY_KEY = "ops.integrity.last"
INTEGRITY_RULE = Weekly(weekday=6, hour=3, minute=30)  # Sunday 03:30, the bot's local time
INTEGRITY_LOOK_S = 600.0
PRUNE_EVERY_S = 3600.0
LAUNCH_ENV = "TWIN_LAUNCH"


def launch_kind(environ: dict[str, str] | None = None) -> str:
    """How this process was started: ``task``, ``manual`` (``twin supervise``) or none."""
    env = os.environ if environ is None else environ
    value = env.get(LAUNCH_ENV, "")
    return value if value in LAUNCH_KINDS else "unsupervised"


@dataclass
class _Open:
    """A failing check that was announced."""

    category: str
    raised_at: datetime


class HealthMonitor:
    """Component: collect, store, announce (see the module description)."""

    name = "health_monitor"
    depends_on: Sequence[str] = ()

    def __init__(
        self,
        services: Services,
        collector: HealthCollector,
        *,
        started_at: datetime | None = None,
        launch: str | None = None,
        integrity: Callable[[], IntegrityReport] | None = None,
    ) -> None:
        self._services = services
        self._collector = collector
        self._config = services.settings.ops.health
        self._clock = services.clock
        self._started_at = started_at or self._clock.now_utc()
        self._launch = launch or launch_kind()
        self._integrity = integrity or self._real_integrity
        self._open: dict[str, _Open] = {}
        self._pruned_at: float | None = None
        self._supervisor = TaskSupervisor(self.name, self._clock, services.alerts)
        self.ticks = 0
        self.last_report: HealthReport | None = None

    # ------------------------------------------------------------------- one look

    async def tick(self) -> HealthReport:
        """Collect one report, store it, announce what changed."""
        report = await self._collector.collect()
        await asyncio.to_thread(self._store, report)
        await asyncio.to_thread(self._announce, report)
        await self._prune_if_due()
        self.last_report = report
        self.ticks += 1
        return report

    def _store(self, report: HealthReport) -> None:
        with self._services.db.transaction(bump_state=False) as session:
            session.add(
                HealthSnapshot(
                    at=report.at,
                    status=report.status,
                    started_at=self._started_at,
                    launch=self._launch,
                    pid=os.getpid(),
                    channel_ok=report.channel_ok,
                    channel_last_ok_at=report.channel_last_ok_at,
                    checks={check.name: check.to_json() for check in report.checks},
                    created_at=report.at,
                    updated_at=report.at,
                )
            )
            if get_setting(session, FIRST_RUN_KEY) is None:
                put_setting(
                    session,
                    FIRST_RUN_KEY,
                    report.at.isoformat(),
                    clock=self._clock,
                    by="health",
                    record_history=False,
                )

    def _announce(self, report: HealthReport) -> None:
        alerts = self._services.alerts
        remind = timedelta(hours=self._config.remind_h)
        for check in report.checks:
            known = self._open.get(check.name)
            if check.level is HealthLevel.FAIL and check.category is not None:
                if known is not None and known.category != check.category:
                    alerts.recover(known.category, f"{check.name}: now {check.category}")
                    known = None
                if known is None or report.at - known.raised_at >= remind:
                    alerts.raise_alert(
                        check.category,
                        f"{check.name}: {check.detail}"[:200],
                        severity=check.severity,
                        dedup_key=f"health:{check.name}",
                    )
                    self._open[check.name] = _Open(check.category, report.at)
            elif known is not None:
                alerts.recover(known.category, f"{check.name} is back to normal")
                del self._open[check.name]

    async def _prune_if_due(self) -> None:
        now = self._clock.monotonic()
        if self._pruned_at is not None and now - self._pruned_at < PRUNE_EVERY_S:
            return
        self._pruned_at = now
        await asyncio.to_thread(self.prune)

    def prune(self) -> int:
        """Delete snapshots older than ``keep_days`` and alerts older than 90 days."""
        cutoff = self._clock.now_utc() - timedelta(days=self._config.keep_days)
        with self._services.db.transaction(bump_state=False) as session:
            result = session.execute(delete(HealthSnapshot).where(HealthSnapshot.at < cutoff))
            removed = int(getattr(result, "rowcount", 0) or 0)
        self._services.alerts.prune()
        return removed

    # -------------------------------------------------------------------- integrity

    def _real_integrity(self) -> IntegrityReport:
        store = VectorStore(self._services.paths.vectors_dir)
        return run_integrity(self._services.db, store, self._clock.now_utc())

    def integrity_due(self) -> bool:
        zone = time_service_for(self._services).bot_timezone()
        due = latest_due(INTEGRITY_RULE, self._clock.now_utc(), zone)
        with self._services.db.session() as session:
            last = get_setting(session, INTEGRITY_KEY)
        if not isinstance(last, dict) or not isinstance(last.get("at"), str):
            return True
        return datetime.fromisoformat(last["at"]) < due

    def run_integrity_now(self) -> IntegrityReport:
        """Check and keep the result (blocking)."""
        report = self._integrity()
        with self._services.db.transaction(bump_state=False) as session:
            put_setting(
                session,
                INTEGRITY_KEY,
                report.to_json(),
                clock=self._clock,
                by="health",
                record_history=False,
            )
        alerts = self._services.alerts
        if report.ok:
            alerts.recover("integrity_failed", "the integrity check is clean again")
        else:
            problems = report.problems()
            log.error("integrity_check_failed", problems=len(problems))
            alerts.raise_alert(
                "integrity_failed",
                f"integrity check: {len(problems)} problem(s)",
                severity="critical",
                detail={
                    "database_ok": report.database_ok,
                    "orphans": sum(check.orphans for check in report.vectors),
                },
                dedup_key="integrity",
            )
        return report

    async def _integrity_loop(self) -> None:
        while True:
            if await asyncio.to_thread(self.integrity_due):
                await asyncio.to_thread(self.run_integrity_now)
            await self._clock.sleep(INTEGRITY_LOOK_S)

    # ------------------------------------------------------------------ component

    async def _loop(self) -> None:
        while True:
            await self.tick()
            await self._clock.sleep(self._config.interval_s)

    async def start(self) -> None:
        self._supervisor.spawn("watch", self._loop, restart_on_exit=True)
        self._supervisor.spawn("integrity", self._integrity_loop, restart_on_exit=True)

    async def stop(self) -> None:
        await self._supervisor.stop()

    def health(self) -> ComponentHealth:
        return self._supervisor.health()
