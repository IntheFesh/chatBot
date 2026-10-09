"""Application components of round 00: state watcher, heartbeat and job worker.

Later rounds add their components (channel, engine, scheduler, health checks)
to the same :class:`~twin.app.Application`.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence

from twin.app import Application, ComponentHealth, HealthStatus, TaskSupervisor
from twin.clock import Clock
from twin.ops.alerts import AlertSink
from twin.ops.jobs import HandlerRegistry, JobQueue, Worker, default_registry, load_handlers
from twin.ops.logging import get_logger
from twin.ops.state_watch import StateWatcher
from twin.services import Services
from twin.storage.db import Database
from twin.storage.settings_store import put_setting

log = get_logger("twin.components")

HEARTBEAT_KEY = "heartbeat"
HEARTBEAT_INTERVAL_S = 30.0
WORKER_DRAIN_GRACE_S = 35.0


class StateWatcherComponent:
    """Runs the :class:`StateWatcher` loop under supervision."""

    name = "state_watcher"
    depends_on: Sequence[str] = ()

    def __init__(self, watcher: StateWatcher, clock: Clock, alerts: AlertSink | None) -> None:
        self.watcher = watcher
        self._supervisor = TaskSupervisor(self.name, clock, alerts)

    async def start(self) -> None:
        await self.watcher.poll_once()  # baseline: changes after this point are broadcast
        self._supervisor.spawn("poll", self.watcher.run, restart_on_exit=True)

    async def stop(self) -> None:
        await self._supervisor.stop()

    def health(self) -> ComponentHealth:
        return self._supervisor.health()


class HeartbeatComponent:
    """Writes a liveness record to ``settings['heartbeat']`` every 30 seconds."""

    name = "heartbeat"
    depends_on: Sequence[str] = ()

    def __init__(
        self,
        db: Database,
        clock: Clock,
        alerts: AlertSink | None,
        *,
        interval_s: float = HEARTBEAT_INTERVAL_S,
    ) -> None:
        self._db = db
        self._clock = clock
        self._interval_s = interval_s
        self._supervisor = TaskSupervisor(self.name, clock, alerts)
        self._started_at = clock.now_utc()
        self._last_beat: float | None = None

    def beat(self) -> None:
        now = self._clock.now_utc()
        with self._db.transaction(bump_state=False) as session:
            put_setting(
                session,
                HEARTBEAT_KEY,
                {
                    "at": now.isoformat(),
                    "pid": os.getpid(),
                    "started_at": self._started_at.isoformat(),
                },
                clock=self._clock,
                by="heartbeat",
                record_history=False,
            )
        self._last_beat = self._clock.monotonic()

    async def _loop(self) -> None:
        while True:
            await asyncio.to_thread(self.beat)
            await self._clock.sleep(self._interval_s)

    async def start(self) -> None:
        await asyncio.to_thread(self.beat)
        self._supervisor.spawn("beat", self._loop, restart_on_exit=True)

    async def stop(self) -> None:
        await self._supervisor.stop()

    def health(self) -> ComponentHealth:
        supervised = self._supervisor.health()
        if supervised.status is not HealthStatus.OK:
            return supervised
        if self._last_beat is None:
            return ComponentHealth(HealthStatus.DEGRADED, "no heartbeat written yet")
        age = self._clock.monotonic() - self._last_beat
        if age > 3 * self._interval_s:
            return ComponentHealth(HealthStatus.UNHEALTHY, f"last heartbeat {age:.0f}s ago")
        return ComponentHealth(HealthStatus.OK)


class JobWorkerComponent:
    """Runs the job :class:`Worker` for the lifetime of the application."""

    name = "job_worker"
    depends_on: Sequence[str] = ()

    def __init__(self, worker: Worker, clock: Clock, alerts: AlertSink | None) -> None:
        self.worker = worker
        self._stop_event = asyncio.Event()
        self._supervisor = TaskSupervisor(self.name, clock, alerts)

    async def start(self) -> None:
        self._supervisor.spawn("worker", lambda: self.worker.run_forever(self._stop_event))

    async def stop(self) -> None:
        self._stop_event.set()
        await self._supervisor.stop(grace_s=WORKER_DRAIN_GRACE_S)

    def health(self) -> ComponentHealth:
        return self._supervisor.health()


def build_application(
    services: Services, registry: HandlerRegistry | None = None
) -> tuple[Application, StateWatcher]:
    """Assemble the application of round 00 (state watcher, heartbeat, job worker)."""
    load_handlers()
    clock = services.clock
    watcher = StateWatcher(services.db, clock)
    worker = Worker(
        JobQueue(services.db, clock),
        registry or default_registry,
        clock,
        services=services,
        alerts=services.alerts,
        concurrency=services.settings.jobs.concurrency,
    )
    application = Application()
    application.register(StateWatcherComponent(watcher, clock, services.alerts))
    application.register(HeartbeatComponent(services.db, clock, services.alerts))
    application.register(JobWorkerComponent(worker, clock, services.alerts))
    return application, watcher
