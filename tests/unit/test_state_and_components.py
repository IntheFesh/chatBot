"""State watcher (R-ARCH-006.2), heartbeat, job worker component and application assembly."""

from __future__ import annotations

import asyncio

import pytest

from tests.support.clock import ManualClock
from tests.support.waiting import wait_until
from twin.app import HealthStatus
from twin.ops.components import (
    HEARTBEAT_INTERVAL_S,
    HEARTBEAT_KEY,
    HeartbeatComponent,
    JobWorkerComponent,
    StateWatcherComponent,
    build_application,
)
from twin.ops.jobs import HandlerRegistry, JobContext, JobQueue, Worker
from twin.ops.state_watch import POLL_INTERVAL_S, StateWatcher
from twin.services import Services
from twin.storage.db import Database, WritePolicy, use_write_policy
from twin.storage.settings_store import get_setting
from twin.storage.state import read_state_version


def bump_from_another_process(db: Database) -> None:
    """What a LIGHT CLI command does: write, bumping the version in the same transaction."""
    other = Database(db.path, clock=db.clock)
    try:
        with use_write_policy(WritePolicy(bump_state=True)), other.transaction():
            pass
    finally:
        other.dispose()


async def test_first_poll_only_sets_the_baseline(db: Database, clock: ManualClock) -> None:
    watcher = StateWatcher(db, clock)
    events: list[tuple[int, int]] = []
    watcher.subscribe(lambda old, new: events.append((old, new)))
    assert watcher.version is None
    assert await watcher.poll_once() is False
    assert watcher.version == 0 and events == []


async def test_a_change_made_elsewhere_is_broadcast_to_every_subscriber(
    db: Database, clock: ManualClock
) -> None:
    watcher = StateWatcher(db, clock)
    sync_events: list[tuple[int, int]] = []
    async_events: list[tuple[int, int]] = []

    async def async_listener(old: int, new: int) -> None:
        async_events.append((old, new))

    watcher.subscribe(lambda old, new: sync_events.append((old, new)))
    watcher.subscribe(async_listener)
    await watcher.poll_once()
    assert await watcher.poll_once() is False  # nothing changed
    bump_from_another_process(db)
    assert await watcher.poll_once() is True
    assert sync_events == async_events == [(0, 1)]
    bump_from_another_process(db)
    bump_from_another_process(db)
    await watcher.poll_once()
    assert sync_events[-1] == (1, 3)  # several changes between polls arrive as one


async def test_a_failing_or_removed_listener_does_not_affect_the_others(
    db: Database, clock: ManualClock
) -> None:
    watcher = StateWatcher(db, clock)
    seen: list[int] = []

    def broken(old: int, new: int) -> None:
        raise RuntimeError("cache invalidation failed")

    watcher.subscribe(broken)
    unsubscribe = watcher.subscribe(lambda old, new: seen.append(new))
    await watcher.poll_once()
    bump_from_another_process(db)
    await watcher.poll_once()
    assert seen == [1]
    unsubscribe()
    unsubscribe()
    bump_from_another_process(db)
    await watcher.poll_once()
    assert seen == [1]


async def test_the_running_watcher_notices_a_change_within_two_seconds(
    db: Database, clock: ManualClock
) -> None:
    component = StateWatcherComponent(StateWatcher(db, clock), clock, None)
    events: list[tuple[int, int]] = []
    component.watcher.subscribe(lambda old, new: events.append((old, new)))
    await component.start()
    await clock.settle()
    assert POLL_INTERVAL_S == 2.0
    bump_from_another_process(db)
    await clock.advance(1.0)
    await asyncio.sleep(0.05)  # let the polling thread finish
    assert component.health().status is HealthStatus.OK
    await clock.advance(1.0)  # two seconds after the change at the latest
    await wait_until(lambda: events == [(0, 1)])
    await component.stop()


async def test_heartbeat_writes_a_liveness_record_and_reports_health(
    db: Database, clock: ManualClock
) -> None:
    component = HeartbeatComponent(db, clock, None, interval_s=10)
    assert component.health().status is HealthStatus.DEGRADED  # nothing written yet
    await component.start()
    with db.session() as session:
        first = get_setting(session, HEARTBEAT_KEY)
        assert read_state_version(session) == 0  # a heartbeat is not a user-visible change
    assert first["at"] == clock.now_utc().isoformat()
    assert component.health().status is HealthStatus.OK
    await wait_until(lambda: clock.pending_sleepers >= 1)  # the loop is waiting for its next beat
    await clock.advance(10)
    await wait_until(lambda: _heartbeat_at(db) != first["at"])
    clock.tick(31)
    assert component.health().status is HealthStatus.UNHEALTHY
    await component.stop()
    assert HEARTBEAT_INTERVAL_S == 30.0


def _heartbeat_at(db: Database) -> str:
    with db.session() as session:
        return str(get_setting(session, HEARTBEAT_KEY)["at"])


async def test_job_worker_component_runs_jobs_and_stops_gracefully(
    db: Database, clock: ManualClock
) -> None:
    registry = HandlerRegistry()
    done: list[int] = []

    async def handler(ctx: JobContext) -> None:
        done.append(ctx.job.payload["n"])

    registry.register("t", handler)
    queue = JobQueue(db, clock)
    worker = Worker(queue, registry, clock, poll_interval_s=1)
    component = JobWorkerComponent(worker, clock, None)
    await component.start()
    await clock.settle()
    queue.enqueue("t", {"n": 1})
    await clock.advance(1)
    await wait_until(lambda: done == [1])
    assert component.health().status is HealthStatus.OK
    await asyncio.wait_for(component.stop(), timeout=5)
    assert queue.counts()["done"] == 1


async def test_build_application_assembles_the_round_00_components(services: Services) -> None:
    application, watcher = build_application(services)
    assert [c.name for c in application.start_order()] == [
        "state_watcher",
        "heartbeat",
        "job_worker",
    ]
    assert isinstance(watcher, StateWatcher)
    await application.start()
    try:
        health = application.health()
        assert {name: h.status for name, h in health.items()} == {
            "state_watcher": HealthStatus.OK,
            "heartbeat": HealthStatus.OK,
            "job_worker": HealthStatus.OK,
        }
    finally:
        await application.stop()


@pytest.mark.parametrize("component", ["state_watcher", "heartbeat", "job_worker"])
def test_component_names_are_stable(component: str) -> None:
    assert component in {
        StateWatcherComponent.name,
        HeartbeatComponent.name,
        JobWorkerComponent.name,
    }
