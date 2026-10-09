"""Running selected job types in the foreground of a CLI command (R-ARCH-006).

``twin import --foreground`` and ``twin stickers download --foreground`` run their jobs
here when the application is not running.  Only the job types of the given registry are
executed, so a foreground import does not also start every other queued job.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from twin.llm.runtime import activate_offpeak_policy
from twin.ops.jobs import HandlerRegistry, JobQueue, RunSummary, Worker
from twin.services import Services

TICK_SECONDS = 0.5


async def run_jobs_until_idle(
    services: Services,
    registry: HandlerRegistry,
    *,
    on_tick: Callable[[], Awaitable[None]] | None = None,
    tick_seconds: float = TICK_SECONDS,
) -> RunSummary:
    """Run the queued jobs of ``registry``'s types to completion; ``on_tick`` runs periodically."""
    activate_offpeak_policy(services)
    worker = Worker(
        JobQueue(services.db, services.clock),
        registry,
        services.clock,
        services=services,
        alerts=services.alerts,
        concurrency=1,
    )
    await worker.recover()
    task = asyncio.ensure_future(worker.run_until_idle())
    try:
        while not task.done():
            if on_tick is not None:
                await on_tick()
            await asyncio.wait({task}, timeout=tick_seconds)
    except asyncio.CancelledError:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise
    if on_tick is not None:
        await on_tick()
    return task.result()
