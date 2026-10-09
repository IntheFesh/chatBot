"""The application component of the learning: the weekly look (R-LRN-003).

Every ``learning.check_interval_s`` seconds (an hour) it asks
:func:`twin.learning.jobs.queue_if_due` whether the consolidation of the rules is due.  The work
itself is a job (off peak); the component only decides when to queue it, so a machine that was
off for a week queues it once, at start-up.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from twin.app import ComponentHealth, TaskSupervisor
from twin.learning.jobs import queue_if_due
from twin.ops.logging import get_logger
from twin.services import Services

log = get_logger("twin.learning.component")

COMPONENT_NAME = "learning"


class LearningComponent:
    """Queues the weekly consolidation (see the module description)."""

    name = COMPONENT_NAME
    depends_on: Sequence[str] = ()

    def __init__(self, services: Services) -> None:
        self._services = services
        self._supervisor = TaskSupervisor(self.name, services.clock, services.alerts)

    async def look(self) -> str | None:
        """Queue the consolidation if it is due; returns the job id."""
        job_id = await asyncio.to_thread(queue_if_due, self._services)
        if job_id is not None:
            log.info("rules_consolidation_queued")
        return job_id

    async def _loop(self) -> None:
        interval = self._services.settings.learning.check_interval_s
        while True:
            await self.look()
            await self._services.clock.sleep(interval)

    async def start(self) -> None:
        self._supervisor.spawn("weekly", self._loop, restart_on_exit=True)

    async def stop(self) -> None:
        await self._supervisor.stop()

    def health(self) -> ComponentHealth:
        return self._supervisor.health()
