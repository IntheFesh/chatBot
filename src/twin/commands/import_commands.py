"""``/导入 <路径>``: start the import of a new export from the chat (R-ARCH-006, R-IMP-006).

The command does what ``twin import <directory>`` does and nothing more: it validates the export,
prepares the run and **queues** the ``import`` job (:func:`twin.ingest.jobs.enqueue_import`, the
function the command line calls too); the application's job worker does the work.  The answer comes
at once - "已开始" - and the end of the import is reported later by
:class:`~twin.commands.import_report.ImportReporter` as a system message with the counts of the run
and no message text.  While it runs, ``/状态`` shows its progress.

The target conversation (the one she is imitated from) is chosen once, on the computer
(``twin import``), and stored; the chat cannot choose it for you, so a first import from the chat
is refused with the command to run.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from twin.commands import texts
from twin.commands.import_report import ImportNotifyStore
from twin.commands.registry import CommandCall, UsageError
from twin.config.runtime import TARGET_USERNAME
from twin.ingest.importer import ImportFailure, TargetNotFoundError
from twin.ingest.jobs import QueuedImport, enqueue_import
from twin.ingest.layout import ExportLayout, ExportLayoutError
from twin.ingest.runs import UNFINISHED, latest_run
from twin.ingest.schema import UnsupportedSchema
from twin.ops.logging import get_logger
from twin.services import Services

log = get_logger("twin.commands.import")

QUOTES = "\"'“”‘’「」『』"


def clean_path(text: str) -> str:
    """The path as typed: surrounding quotes and white space removed."""
    return text.strip().strip(QUOTES).strip()


class ImportCommands:
    """The handler of ``/导入``."""

    def __init__(self, services: Services, notes: ImportNotifyStore) -> None:
        self._services = services
        self._notes = notes

    async def start(self, call: CommandCall) -> str:
        text = clean_path(call.args)
        if not text:
            raise UsageError("")
        return await asyncio.to_thread(self._start, text)

    def _start(self, text: str) -> str:
        services = self._services
        directory = Path(text).expanduser()
        if not directory.is_dir():
            return texts.IMPORT_NO_FOLDER.format(path=text)
        running = latest_run(services.db)
        if running is not None and running.status in UNFINISHED and running.status != "failed":
            same = running.source_dir == str(directory)
            if not same and running.status in ("running", "queued"):
                return texts.IMPORT_BUSY.format(run=running.id)
        username = services.runtime.get(TARGET_USERNAME)
        if not username:
            return texts.IMPORT_NEEDS_TARGET.format(path=text)
        try:
            ExportLayout(directory).validate()
            queued = enqueue_import(services, directory, username)
        except TargetNotFoundError as exc:
            return texts.IMPORT_NOT_IN_EXPORT.format(reason=exc)
        except (ExportLayoutError, UnsupportedSchema, ImportFailure) as exc:
            return texts.IMPORT_BAD_EXPORT.format(reason=exc)
        self._notes.add(queued.run_id)
        log.info("import_started_from_chat", run=queued.run_id, resumed=queued.resumed)
        return self._reply(directory, queued)

    @staticmethod
    def _reply(directory: Path, queued: QueuedImport) -> str:
        if queued.resumed:
            return texts.IMPORT_RESUMED.format(run=queued.run_id)
        total = f"{queued.total:,}" if queued.total is not None else "?"
        return texts.IMPORT_STARTED.format(
            name=directory.name or str(directory), run=queued.run_id, total=total
        )
