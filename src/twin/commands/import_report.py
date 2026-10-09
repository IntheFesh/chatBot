"""Telling the user how an import started from the chat ended (R-ARCH-006).

``/导入`` returns at once; the import runs in the job worker, possibly for a long time.
:class:`ImportNotifyStore` remembers, in the settings table, the runs the chat asked for and has
not heard the end of (it survives a restart of the application).  :class:`ImportReporter` looks at
those runs: when one is over, it makes the system message - the counts of the run (new, duplicate,
updated, kept, invalid messages and the time it took), never a word of the messages - and forgets
the run.  :class:`ImportReportComponent` is the application component that looks every
``commands.import_poll_s`` seconds and hands each message to the engine, which sends it like any
system message.  A run that vanished (the table was reset) is forgotten without a word.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence

from twin.app import ComponentHealth, TaskSupervisor
from twin.commands import texts
from twin.commands.status import format_span
from twin.ingest.runs import RunView, get_run
from twin.ops.logging import get_logger
from twin.services import Services
from twin.storage.settings_store import get_setting, put_setting

log = get_logger("twin.commands.import_report")

NOTIFY_KEY = "commands.import_notify"
TERMINAL = ("done", "failed", "superseded")
MAX_PENDING = 20


class ImportNotifyStore:
    """The runs started from the chat whose end has not been reported yet."""

    def __init__(self, services: Services) -> None:
        self._services = services

    def pending(self) -> list[str]:
        with self._services.db.session() as session:
            value = get_setting(session, NOTIFY_KEY, [])
        return [str(item) for item in value] if isinstance(value, list) else []

    def _write(self, runs: Sequence[str]) -> None:
        with self._services.db.transaction(bump_state=False) as session:
            put_setting(
                session,
                NOTIFY_KEY,
                list(runs)[-MAX_PENDING:],
                clock=self._services.clock,
                by="command",
                record_history=False,
            )

    def add(self, run_id: str) -> None:
        runs = self.pending()
        if run_id not in runs:
            self._write([*runs, run_id])

    def remove(self, run_id: str) -> None:
        runs = self.pending()
        if run_id in runs:
            self._write([r for r in runs if r != run_id])


def render_end(view: RunView) -> str:
    """The message about an import that is over: counts only."""
    if view.status == "done":
        took = (
            format_span(view.finished_at - view.started_at)
            if view.finished_at is not None and view.started_at is not None
            else "-"
        )
        return texts.IMPORT_DONE.format(
            inserted=view.inserted,
            duplicates=view.duplicates,
            updated=view.conflict_updated,
            kept=view.conflict_kept,
            invalid=view.invalid,
            took=took,
        )
    if view.status == "superseded":
        return texts.IMPORT_SUPERSEDED
    total = f"{view.total:,}" if view.total is not None else "?"
    return texts.IMPORT_FAILED.format(
        status=view.status, processed=f"{view.processed:,}", total=total
    )


class ImportReporter:
    """Finds the finished imports the chat is waiting for (see the module description)."""

    def __init__(self, services: Services, notes: ImportNotifyStore | None = None) -> None:
        self._services = services
        self.notes = notes or ImportNotifyStore(services)

    def collect(self) -> list[tuple[str, str]]:
        """``(run id, message)`` of every waited-for import that is over; they are forgotten."""
        found: list[tuple[str, str]] = []
        for run_id in self.notes.pending():
            view = get_run(self._services.db, run_id)
            if view is None:
                self.notes.remove(run_id)
                continue
            if view.status in TERMINAL:
                found.append((run_id, render_end(view)))
        return found

    def forget(self, run_id: str) -> None:
        self.notes.remove(run_id)


Say = Callable[[str], Awaitable[None]]


class ImportReportComponent:
    """Looks for finished imports and says how they ended (a component of the application)."""

    name = "import_report"
    depends_on: Sequence[str] = ()

    def __init__(
        self, services: Services, say: Say, reporter: ImportReporter | None = None
    ) -> None:
        self._services = services
        self._say = say
        self.reporter = reporter or ImportReporter(services)
        self._supervisor = TaskSupervisor(self.name, services.clock, services.alerts)

    async def poll_once(self) -> int:
        """Report what is over now; returns how many messages were sent."""
        sent = 0
        for run_id, message in await asyncio.to_thread(self.reporter.collect):
            await self._say(message)
            await asyncio.to_thread(self.reporter.forget, run_id)
            sent += 1
        return sent

    async def _loop(self) -> None:
        interval = self._services.settings.commands.import_poll_s
        while True:
            await self.poll_once()
            await self._services.clock.sleep(interval)

    async def start(self) -> None:
        self._supervisor.spawn("watch", self._loop, restart_on_exit=True)

    async def stop(self) -> None:
        await self._supervisor.stop()

    def health(self) -> ComponentHealth:
        return self._supervisor.health()
