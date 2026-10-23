"""R18-D7 (D-801): the stop of ``twin run`` fits in the time ``twin supervise`` gives it.

Before round 18 the job worker's component waited 35 s for its drain while the application gave
every component 30 s: each stop with a long job was cancelled half-way and logged as an ERROR
(``component_stop_failed``), and nothing bounded the stop as a whole below the 40 s after which the
supervisor kills the process.  The budget is now one table of constants (``twin.app``,
``twin.ops.components``, ``twin.ops.jobs``): each component gets at least its own longest wait, all
of them together less than the supervisor's grace.

The behaviour tests run the real components with every duration scaled down by the same factor,
so they check the production ratios without waiting the production seconds.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

import pytest

from tests.support.clock import ManualClock
from tests.support.engine_harness import (
    Harness,
    ScriptedChannel,
    build_harness,
    make_draft,
    run_to_idle,
)
from tests.support.policies import AlwaysOffPeak
from tests.support.waiting import wait_until
from twin import app as app_module
from twin.app import (
    COMPONENT_STOP_TIMEOUT_S,
    CUT_SHORT_S,
    SHUTDOWN_BUDGET_S,
    Application,
    ComponentHealth,
    HealthStatus,
)
from twin.channel.ilink.http import TIMEOUT_QUICK_S
from twin.config.settings import Settings
from twin.engine.machine import STOP_GRACE_S as ENGINE_STOP_GRACE_S
from twin.ops.components import WORKER_DRAIN_GRACE_S, JobWorkerComponent
from twin.ops.jobs import WORKER_DRAIN_S, HandlerRegistry, JobContext, JobQueue, Worker
from twin.ops.power_events import WINDOW_STOP_TIMEOUT_S
from twin.services import Services
from twin.serving.server import STOP_GRACE_S as LLAMA_STOP_GRACE_S

SCALE = 0.2  # every duration of the behaviour test, production ratios kept
SUPERVISOR_GRACE_S = Settings().ops.supervise.stop_grace_s


def test_the_budget_table_fits_inside_the_supervisors_grace() -> None:
    """R18-D7: each component may take its own longest wait; all together stay below 40 s."""
    # the worker's component waits for its drain and the release, and the application waits for it
    assert WORKER_DRAIN_S < WORKER_DRAIN_GRACE_S <= COMPONENT_STOP_TIMEOUT_S
    # every other wait a component has of its own fits in one component's time
    own_waits = (ENGINE_STOP_GRACE_S, LLAMA_STOP_GRACE_S, WINDOW_STOP_TIMEOUT_S, TIMEOUT_QUICK_S)
    for own_wait in own_waits:
        assert own_wait <= COMPONENT_STOP_TIMEOUT_S
    # the two waits for work in flight - a bubble on its way, a running job - fit in the budget
    assert ENGINE_STOP_GRACE_S + WORKER_DRAIN_GRACE_S <= SHUTDOWN_BUDGET_S
    # and the budget, with the moments of the components that have to be cut short, leaves the
    # process time to close its database before the supervisor kills it
    assert COMPONENT_STOP_TIMEOUT_S <= SHUTDOWN_BUDGET_S
    assert SHUTDOWN_BUDGET_S + 4 * CUT_SHORT_S < SUPERVISOR_GRACE_S


class GatedChannel(ScriptedChannel):
    """The platform has the bubble and never answers (the engine waits its whole grace)."""

    def __init__(self, clock: ManualClock, gate_on: str) -> None:
        super().__init__(clock)
        self.gate_on = gate_on
        self.reached = asyncio.Event()

    async def send_text(self, text: str, *args: Any, **kwargs: Any) -> Any:
        result = await super().send_text(text, *args, **kwargs)
        if text == self.gate_on:
            self.reached.set()
            await asyncio.Event().wait()
        return result


class EngineOf:
    """The harness's engine as an application component (what ``EngineComponent`` does)."""

    name = "engine"
    depends_on: Sequence[str] = ()

    def __init__(self, harness: Harness) -> None:
        self.harness = harness

    async def start(self) -> None:
        await self.harness.engine.start()

    async def stop(self) -> None:
        await self.harness.engine.stop()

    def health(self) -> ComponentHealth:
        return ComponentHealth(HealthStatus.OK)


def forever_worker(queue: JobQueue, clock: ManualClock, started: asyncio.Event) -> Worker:
    registry = HandlerRegistry()

    async def forever(ctx: JobContext) -> None:
        started.set()
        await asyncio.Event().wait()  # a job that outlasts any drain

    registry.register("forever", forever)
    return Worker(queue, registry, clock, offpeak=AlwaysOffPeak(), drain_s=WORKER_DRAIN_S * SCALE)


def messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records]


async def test_a_stop_with_a_bubble_on_its_way_and_a_job_that_never_ends_fits_the_budget(
    services: Services, clock: ManualClock, caplog: pytest.LogCaptureFixture
) -> None:
    """R18-D7: the worst usual case - the engine waits for a bubble, the worker for a job.

    The stop ends before the supervisor would kill the process, the job is released without
    an attempt, the bubble that was out is written down, and nothing is logged as failed.
    """
    channel = GatedChannel(clock, gate_on="二")
    harness = build_harness(
        services, clock, channel=channel, stop_grace_s=ENGINE_STOP_GRACE_S * SCALE
    )
    queue = JobQueue(services.db, clock)
    started = asyncio.Event()
    job_id = queue.enqueue("forever", {}, max_attempts=3)
    worker = forever_worker(queue, clock, started)
    application = Application(
        stop_timeout_s=COMPONENT_STOP_TIMEOUT_S * SCALE, stop_budget_s=SHUTDOWN_BUDGET_S * SCALE
    )
    application.register(
        JobWorkerComponent(worker, clock, None, grace_s=WORKER_DRAIN_GRACE_S * SCALE)
    )
    application.register(EngineOf(harness))  # registered later, stopped first (as in twin run)
    await application.start()
    await asyncio.wait_for(started.wait(), 10)
    harness.writer.add(make_draft("一", "二", "三"))
    await harness.message("数数")
    await run_to_idle(harness.engine, clock, until=channel.reached.is_set)

    loop = asyncio.get_running_loop()
    begun = loop.time()
    with caplog.at_level(logging.INFO, logger="twin"):
        await asyncio.wait_for(application.stop(), SUPERVISOR_GRACE_S * SCALE)
    took = loop.time() - begun

    assert took < SHUTDOWN_BUDGET_S * SCALE
    logged = messages(caplog)
    assert "component_stop_failed" not in logged and "component_stop_cut_short" not in logged
    assert logged.count("component_stopped") == 2
    job = queue.get(job_id)
    assert job is not None and job.status == "pending" and job.attempts == 0
    assert channel.texts == ["一", "二"]  # "三" was not started after the stop was asked for
    assert worker.running_count == 0


class StuckStop:
    """A component whose stop never returns by itself (a hung connection, say)."""

    depends_on: Sequence[str] = ()

    def __init__(self, name: str) -> None:
        self.name = name
        self.cancelled = False

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    def health(self) -> ComponentHealth:
        return ComponentHealth(HealthStatus.OK)


async def test_when_the_budget_is_spent_the_rest_is_cut_short_and_nothing_runs_on(
    services: Services,
    clock: ManualClock,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R18-D7: a component that hangs uses up its time; the ones after it get what is left.

    The worker, cut short in the middle of its drain, still cancels and releases its job (no
    attempt lost) instead of running on behind the stop.
    """
    monkeypatch.setattr(app_module, "CUT_SHORT_S", 0.05)
    queue = JobQueue(services.db, clock)
    started = asyncio.Event()
    job_id = queue.enqueue("forever", {}, max_attempts=3)
    worker = forever_worker(queue, clock, started)
    stuck = StuckStop("stuck")
    application = Application(stop_timeout_s=0.3, stop_budget_s=0.4)
    application.register(JobWorkerComponent(worker, clock, None, grace_s=30.0))
    application.register(stuck)
    await application.start()
    await asyncio.wait_for(started.wait(), 10)
    with caplog.at_level(logging.INFO, logger="twin"):
        await asyncio.wait_for(application.stop(), 5)
    assert stuck.cancelled
    cut = [r for r in caplog.records if r.getMessage() == "component_stop_cut_short"]
    failed = [r for r in caplog.records if r.getMessage() == "component_stop_failed"]
    assert len(cut) == 1 and len(failed) == 1  # the worker was cut short; "stuck" failed
    await wait_until(lambda: worker.running_count == 0)

    def released() -> bool:
        job = queue.get(job_id)
        return job is not None and job.status == "pending" and job.attempts == 0

    await wait_until(released)
