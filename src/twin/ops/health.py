"""The health check: what is wrong with the running bot, every minute (R-OPS-003).

Each check is a small pure function from numbers to a :class:`HealthCheck` (``ok`` / ``warn`` /
``fail`` plus a one-line detail, a number and, when a failure is something to announce, the alert
category), so the thresholds are tested without a database:

========  ============================================================================
channel   login state and the last successful long poll (older than five minutes: bad)
deepseek  share of failed attempts over a window, and an open circuit breaker
style     the style model's endpoint, only when the backend needs it
disk      free space of the data directory (below 5 GB: bad)
queue     waiting jobs (more than 500, or one waiting for more than 24 hours: bad)
backup    age of the newest backup (more than 36 hours: bad)
budget    the degradation level (information only: the budget manager raises its own alerts)
app       components of the application that report themselves unhealthy
========  ============================================================================

:class:`HealthCollector` reads the inputs (the database, the disk, and - inside ``twin run`` - the
live objects of the process); :class:`~twin.ops.monitor.HealthMonitor` runs it every
``ops.health.interval_s``, stores a ``health_snapshots`` row and raises or closes the alerts.
``twin health`` shows the same report, made from what a second process can see.  Thresholds are
the ``ops.health`` settings.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select

from twin.app import ComponentHealth, HealthStatus
from twin.channel.base import AuthState
from twin.channel.ilink.store import IlinkStore
from twin.channel.state import ChannelStateStore
from twin.config.settings import HealthConfig
from twin.llm.health import LlmHealthSnapshot, llm_health_of
from twin.storage.models import Job
from twin.storage.ops_models import BackupRecord
from twin.storage.settings_store import get_setting

if TYPE_CHECKING:
    from twin.services import Services

FIRST_RUN_KEY = "ops.first_run_at"


class HealthLevel(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class HealthCheck:
    """The result of one check."""

    name: str
    level: HealthLevel
    detail: str
    value: float | None = None
    category: str | None = None  # the alert a failure raises (R-OPS-004)
    severity: str = "warning"

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.level.value,
            "detail": self.detail,
            "value": self.value,
            "category": self.category,
            "severity": self.severity,
        }

    @classmethod
    def from_json(cls, name: str, data: Mapping[str, Any]) -> HealthCheck:
        return cls(
            name=name,
            level=HealthLevel(data.get("status", "ok")),
            detail=str(data.get("detail", "")),
            value=data.get("value"),
            category=data.get("category"),
            severity=str(data.get("severity", "warning")),
        )


@dataclass(frozen=True)
class HealthReport:
    """All checks at one moment, and the channel facts the stability report needs."""

    at: datetime
    checks: tuple[HealthCheck, ...]
    channel_ok: bool | None = None
    channel_last_ok_at: datetime | None = None

    @property
    def status(self) -> str:
        """``ok``, ``degraded`` (a warning) or ``unhealthy`` (a failure)."""
        if any(check.level is HealthLevel.FAIL for check in self.checks):
            return "unhealthy"
        if any(check.level is HealthLevel.WARN for check in self.checks):
            return "degraded"
        return "ok"

    def get(self, name: str) -> HealthCheck | None:
        return next((check for check in self.checks if check.name == name), None)

    def to_json(self) -> dict[str, Any]:
        return {
            "at": self.at.isoformat(),
            "status": self.status,
            "channel_ok": self.channel_ok,
            "channel_last_ok_at": self.channel_last_ok_at.isoformat()
            if self.channel_last_ok_at
            else None,
            "checks": {check.name: check.to_json() for check in self.checks},
        }


# ------------------------------------------------------------------ the pure checks


def _minutes(delta: timedelta) -> float:
    return delta.total_seconds() / 60.0


def check_channel(
    *,
    kind: str,
    logged_in: bool,
    auth: AuthState,
    last_ok_at: datetime | None,
    now: datetime,
    started_at: datetime,
    stale_min: float,
) -> HealthCheck:
    """Login state first, then the age of the last successful long poll."""
    if kind != "ilink":
        return HealthCheck("channel", HealthLevel.OK, "the console channel needs no login")
    if not logged_in:
        return HealthCheck(
            "channel",
            HealthLevel.FAIL,
            "not logged in to WeChat (run `twin channel login`)",
            category="login_lost",
            severity="critical",
        )
    if auth is AuthState.NEEDS_RELOGIN:
        return HealthCheck(
            "channel",
            HealthLevel.FAIL,
            "the WeChat login expired (scan the QR code again)",
            category="login_lost",
            severity="critical",
        )
    # a success from before this process started says nothing yet: give the poller its time
    fresh = last_ok_at is not None and last_ok_at >= started_at
    reference = last_ok_at if last_ok_at is not None and fresh else started_at
    minutes = _minutes(now - reference)
    if minutes > stale_min:
        what = "last successful long poll" if fresh else "no successful long poll since the start"
        return HealthCheck(
            "channel",
            HealthLevel.FAIL,
            f"{what} {minutes:.0f} min ago (limit {stale_min:g})",
            value=round(minutes * 60.0, 1),
            category="channel_poll_stale",
        )
    return HealthCheck(
        "channel",
        HealthLevel.OK,
        "polling works" if fresh else "starting",
        value=round(minutes * 60.0, 1),
    )


def check_deepseek(
    snapshot: LlmHealthSnapshot | None, *, error_rate: float, min_calls: int
) -> HealthCheck:
    """An open breaker is a failure; so is a share of failed attempts above the limit."""
    if snapshot is None:
        return HealthCheck(
            "deepseek", HealthLevel.OK, "not visible outside the running application"
        )
    if snapshot.circuit_open:
        return HealthCheck(
            "deepseek",
            HealthLevel.FAIL,
            "the circuit breaker is open",
            value=snapshot.error_rate,
            category="circuit_open",
            severity="critical",
        )
    if snapshot.calls >= min_calls and snapshot.error_rate >= error_rate:
        return HealthCheck(
            "deepseek",
            HealthLevel.FAIL,
            f"{snapshot.failures} of {snapshot.calls} recent attempts failed",
            value=snapshot.error_rate,
            category="deepseek_failure",
        )
    return HealthCheck(
        "deepseek",
        HealthLevel.OK,
        f"{snapshot.failures} of {snapshot.calls} recent attempts failed",
        value=snapshot.error_rate,
    )


@dataclass(frozen=True)
class StyleReading:
    """What the style model's endpoint answered (``needed`` false: the backend does not use it)."""

    needed: bool
    ok: bool
    code: str = "ok"


def check_style(reading: StyleReading | None) -> HealthCheck:
    if reading is None:
        return HealthCheck(
            "style_model", HealthLevel.OK, "not visible outside the running application"
        )
    if not reading.needed:
        return HealthCheck(
            "style_model", HealthLevel.OK, "not needed (the DeepSeek backend is used)"
        )
    if reading.ok:
        return HealthCheck("style_model", HealthLevel.OK, "the endpoint answers")
    return HealthCheck(
        "style_model",
        HealthLevel.FAIL,
        f"the endpoint does not answer ({reading.code})",
        category="style_model_down",
    )


def check_disk(free_bytes: int, *, min_gb: float) -> HealthCheck:
    free_gb = free_bytes / 1024**3
    if free_gb < min_gb:
        return HealthCheck(
            "disk",
            HealthLevel.FAIL,
            f"{free_gb:.1f} GB free (limit {min_gb:g})",
            value=round(free_gb, 2),
            category="disk_low",
            severity="critical" if free_gb < min_gb / 2 else "warning",
        )
    return HealthCheck("disk", HealthLevel.OK, f"{free_gb:.0f} GB free", value=round(free_gb, 2))


def check_queue(
    waiting: int, oldest_age: timedelta | None, *, max_jobs: int, oldest_h: float
) -> HealthCheck:
    """Jobs ready to run: too many, or one that has waited too long."""
    hours = oldest_age.total_seconds() / 3600 if oldest_age is not None else 0.0
    if waiting > max_jobs:
        return HealthCheck(
            "queue",
            HealthLevel.FAIL,
            f"{waiting} jobs are waiting (limit {max_jobs})",
            value=float(waiting),
            category="queue_backlog",
        )
    if oldest_age is not None and hours > oldest_h:
        return HealthCheck(
            "queue",
            HealthLevel.FAIL,
            f"the oldest waiting job is {hours:.0f} h old (limit {oldest_h:g})",
            value=round(hours, 1),
            category="queue_backlog",
        )
    return HealthCheck("queue", HealthLevel.OK, f"{waiting} job(s) waiting", value=float(waiting))


def check_backup(
    newest_age: timedelta | None, *, since_first_run: timedelta, stale_h: float
) -> HealthCheck:
    """The newest backup must be younger than ``stale_h``; before the first one, give it time."""
    if newest_age is None:
        if since_first_run > timedelta(hours=stale_h):
            return HealthCheck(
                "backup",
                HealthLevel.FAIL,
                f"no backup was made in the first {stale_h:g} hours",
                category="backup_failed",
            )
        return HealthCheck("backup", HealthLevel.WARN, "no backup yet (the first one is due)")
    hours = newest_age.total_seconds() / 3600
    if hours > stale_h:
        return HealthCheck(
            "backup",
            HealthLevel.FAIL,
            f"the newest backup is {hours:.0f} h old (limit {stale_h:g})",
            value=round(hours, 1),
            category="backup_failed",
        )
    return HealthCheck(
        "backup", HealthLevel.OK, f"the newest backup is {hours:.0f} h old", value=round(hours, 1)
    )


def check_budget(level: int, daily_ratio: float, monthly_ratio: float) -> HealthCheck:
    ratio = max(daily_ratio, monthly_ratio)
    if level > 0:
        return HealthCheck(
            "budget",
            HealthLevel.WARN,
            f"degradation level {level} ({ratio:.0%} of the budget)",
            value=float(level),
        )
    return HealthCheck(
        "budget", HealthLevel.OK, f"{ratio:.0%} of the budget", value=round(ratio, 3)
    )


def check_components(components: Mapping[str, ComponentHealth] | None) -> HealthCheck:
    if components is None:
        return HealthCheck("app", HealthLevel.OK, "not visible outside the running application")
    bad = {n: h for n, h in components.items() if h.status is HealthStatus.UNHEALTHY}
    weak = {n: h for n, h in components.items() if h.status is HealthStatus.DEGRADED}
    if bad:
        return HealthCheck("app", HealthLevel.FAIL, "unhealthy: " + ", ".join(sorted(bad)))
    if weak:
        return HealthCheck("app", HealthLevel.WARN, "degraded: " + ", ".join(sorted(weak)))
    return HealthCheck("app", HealthLevel.OK, f"{len(components)} components running")


# ---------------------------------------------------------------- reading the inputs


@dataclass
class LiveSources:
    """What only the running application can see."""

    components: Callable[[], Mapping[str, ComponentHealth]] | None = None
    style: Callable[[], Awaitable[StyleReading]] | None = None
    llm: bool = True


class HealthCollector:
    """Reads the inputs of the checks and runs them (see the module description)."""

    def __init__(
        self,
        services: Services,
        *,
        started_at: datetime | None = None,
        live: LiveSources | None = None,
    ) -> None:
        self._services = services
        self._config: HealthConfig = services.settings.ops.health
        self._started_at = started_at or services.clock.now_utc()
        self._live = live
        self._budget: Any = None
        self._probes: list[tuple[str, Callable[[], Awaitable[HealthCheck]]]] = []

    # -- database readings (blocking) ------------------------------------------------

    def _channel(self, now: datetime) -> tuple[HealthCheck, bool | None, datetime | None]:
        kind = self._services.settings.channel.kind
        if kind != "ilink":
            return (
                check_channel(
                    kind=kind,
                    logged_in=True,
                    auth=AuthState.OK,
                    last_ok_at=None,
                    now=now,
                    started_at=self._started_at,
                    stale_min=self._config.poll_stale_min,
                ),
                None,
                None,
            )
        store = IlinkStore(ChannelStateStore(self._services.db), self._services.clock)
        credentials = store.credentials()
        auth = store.auth_record()
        poll = store.poll_status()
        check = check_channel(
            kind=kind,
            logged_in=credentials is not None,
            auth=auth.state,
            last_ok_at=poll.last_ok_at,
            now=now,
            started_at=self._started_at,
            stale_min=self._config.poll_stale_min,
        )
        return check, check.level is HealthLevel.OK, poll.last_ok_at

    def _queue(self, now: datetime) -> HealthCheck:
        ready = (
            Job.status == "pending",
            Job.run_after <= now,
            (Job.requires_approval.is_(False)) | (Job.approved_at.is_not(None)),
        )
        with self._services.db.session() as session:
            waiting = int(session.scalar(select(func.count()).select_from(Job).where(*ready)) or 0)
            oldest = session.scalar(select(func.min(Job.run_after)).where(*ready))
        age = now - oldest if oldest is not None else None
        return check_queue(
            waiting,
            age,
            max_jobs=self._config.queue_max,
            oldest_h=self._config.queue_oldest_h,
        )

    def _backup(self, now: datetime) -> HealthCheck:
        with self._services.db.session() as session:
            newest = session.scalar(
                select(func.max(BackupRecord.created_at)).where(BackupRecord.status == "ok")
            )
            first = get_setting(session, FIRST_RUN_KEY)
        first_run = datetime.fromisoformat(first) if isinstance(first, str) else now  # unknown: new
        return check_backup(
            now - newest if newest is not None else None,
            since_first_run=now - first_run,
            stale_h=self._config.backup_stale_h,
        )

    def _budget_check(self) -> HealthCheck:
        from twin.llm.runtime import build_llm_runtime

        if self._budget is None:
            self._budget = build_llm_runtime(self._services).budget
        status = self._budget.compute()
        return check_budget(status.level, status.daily_ratio, status.monthly_ratio)

    def _disk(self) -> HealthCheck:
        directory = self._services.paths.data_dir
        return check_disk(shutil.disk_usage(directory).free, min_gb=self._config.disk_min_gb)

    def stored_checks(self) -> tuple[tuple[HealthCheck, ...], bool | None, datetime | None]:
        """The checks a process other than the application can make."""
        now = self._services.clock.now_utc()
        channel, channel_ok, last_ok = self._channel(now)
        checks = (channel, self._disk(), self._queue(now), self._backup(now), self._budget_check())
        return checks, channel_ok, last_ok

    def register_probe(self, name: str, probe: Callable[[], Awaitable[HealthCheck]]) -> None:
        """Add a check of another component (R-SRV-002: the ``llama-server`` process of round 14
        registers itself here).  The probe answers with a :class:`HealthCheck`; one that raises
        is reported as a failed check of that name."""
        self._probes.append((name, probe))

    # -- the whole report ------------------------------------------------------------

    async def collect(self) -> HealthReport:
        """All checks (the live ones too, when sources are given)."""
        live = self._live
        stored, channel_ok, last_ok = await asyncio.to_thread(self.stored_checks)
        extra: list[HealthCheck] = []
        if live is not None:
            snapshot = None
            if live.llm:
                snapshot = llm_health_of(self._services).snapshot(
                    self._config.llm_window_min * 60.0
                )
            extra.append(
                check_deepseek(
                    snapshot,
                    error_rate=self._config.llm_error_rate,
                    min_calls=self._config.llm_min_calls,
                )
            )
            reading = await live.style() if live.style is not None else None
            extra.append(check_style(reading))
            components = live.components() if live.components is not None else None
            extra.append(check_components(components))
        for name, probe in self._probes:
            try:
                extra.append(await probe())
            except Exception as exc:  # a broken probe is a finding, not a crash of the monitor
                extra.append(
                    HealthCheck(name, HealthLevel.FAIL, f"the probe crashed ({type(exc).__name__})")
                )
        now = self._services.clock.now_utc()
        return HealthReport(now, (*stored, *extra), channel_ok, last_ok)
