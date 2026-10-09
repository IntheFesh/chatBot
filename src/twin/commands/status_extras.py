"""More lines of ``/状态`` for round 11: imports, the memory replay, the pairs for DPO (R-CMD-002).

Each function makes a source of :attr:`~twin.commands.status.StatusSources.extra`: it reads a
little from the database and returns the line(s), or ``None`` when there is nothing to say.  They
are cheap on purpose - ``/状态`` answers at once - so the replay is read from its jobs, not by
walking the history.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from twin.commands import texts
from twin.ingest.runs import UNFINISHED, latest_run
from twin.learning.pairs import PreferencePairStore, dpo_hint
from twin.memory.replay import REPLAY_JOB
from twin.ops.jobs import JobQueue
from twin.services import Services

StatusLine = Callable[[], Awaitable[str | None]]
NOTHING_RUNNING = "导入与记忆回放：没有进行中的任务"


def import_replay_line(services: Services) -> StatusLine:
    """The progress of the import and of the memory replay, or that nothing is under way."""

    def read() -> str | None:
        lines: list[str] = []
        run = latest_run(services.db)
        if run is not None and run.status in UNFINISHED and run.status != "failed":
            total = f"{run.total:,}" if run.total is not None else "?"
            lines.append(
                texts.STATUS_IMPORT.format(
                    run=run.id,
                    status=run.status,
                    phase=run.phase,
                    processed=f"{run.processed:,}",
                    total=total,
                )
            )
        queue = JobQueue(services.db, services.clock)
        waiting = sum(
            len(queue.list_jobs(status=status, job_type=REPLAY_JOB, limit=1000))
            for status in ("pending", "running")
        )
        if waiting:
            lines.append(texts.STATUS_REPLAY.format(text=f"{waiting} 个回放任务在排队或进行"))
        return "\n".join(lines) if lines else NOTHING_RUNNING

    async def line() -> str | None:
        return await asyncio.to_thread(read)

    return line


def dpo_line(services: Services) -> StatusLine:
    """The reminder that there are enough preference pairs for DPO (R-TRN-012)."""

    def read() -> str | None:
        pairs = PreferencePairStore(services.db, services.clock).count()
        hint = dpo_hint(pairs, services.settings.training.dpo_min_pairs)
        return texts.STATUS_DPO.format(text=hint) if hint else None

    async def line() -> str | None:
        return await asyncio.to_thread(read)

    return line
