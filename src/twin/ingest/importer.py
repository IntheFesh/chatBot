"""The chat-record import (R-IMP-003 ... R-IMP-008, R-IMP-010, R-IMP-011).

An import is a *run* (``import_runs``) that moves through phases::

    discover -> messages -> media -> stickers -> finalize -> hooks -> done

* **messages**: ``messages.json`` of the target conversation is streamed with ``ijson``
  (memory does not depend on the file size) in batches of ``ingest.batch_size`` messages;
  each batch is one transaction that also saves the run's progress, so a crash loses
  nothing and a resumed run continues after the last committed batch;
* **media / stickers**: the files the messages refer to are imported (see
  :mod:`twin.ingest.media_import`);
* **finalize / hooks**: conversation totals, the post-import hooks (R-IMP-011) and the
  report (R-IMP-010).

Only the target conversation's ``messages.json`` is ever opened (R-IMP-003).  The runner is
synchronous; the job handler runs it in a worker thread and signals a stop (shutdown, Ctrl+C)
through a :class:`threading.Event` that is checked between batches.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any

import ijson
from pydantic import ValidationError
from sqlalchemy import func, select, update

from twin.config.settings import Settings
from twin.ingest.hooks import HookContext, HookOutcome, load_hooks, run_hooks
from twin.ingest.integrity import (
    FileVerdict,
    IntegrityIndex,
    IntegrityState,
    load_integrity,
    normalize_key,
    verify_file,
)
from twin.ingest.jsonio import first_value, iter_messages, skip_bom
from twin.ingest.layout import (
    MESSAGES_FILE,
    ConversationEntry,
    ExportInfo,
    ExportLayout,
    ExportLayoutError,
    parse_header,
)
from twin.ingest.media_import import MediaImporter
from twin.ingest.normalize import (
    InvalidMessage,
    NormalizeContext,
    NormalizedMessage,
    normalize_message,
)
from twin.ingest.paths import long_path
from twin.ingest.persist import (
    BatchContext,
    ConversationMismatchError,
    RowSealer,
    persist_messages,
)
from twin.ingest.report import build_report, write_report
from twin.ingest.runs import (
    RunView,
    create_run,
    finished_runs_exist,
    get_run,
    unfinished_runs,
    view_of,
)
from twin.ingest.schema import ExportMessage, MessagesHeader, UnsupportedSchema
from twin.ingest.stats import RunStats
from twin.ingest.times import SourceTime, epoch_to_utc
from twin.ops.logging import get_logger
from twin.services import Services
from twin.storage.chat_models import Conversation, ImportRun, Message
from twin.storage.crypto import get_keyring

log = get_logger("twin.ingest")


class ImportFailure(RuntimeError):
    """The import cannot go on; the message is safe to show (no message content)."""


class TargetNotFoundError(ImportFailure):
    """The configured target conversation is not in this export."""


class IntegrityFailure(ImportFailure):
    """A file failed the export's integrity check."""


@dataclass(frozen=True)
class BatchEvent:
    """Reported after every committed batch (tests and progress bars listen to it)."""

    run_id: str
    batch_number: int
    processed: int
    total: int | None
    inserted: int
    speed_per_s: float | None


@dataclass(frozen=True)
class RunOutcome:
    """How :meth:`ImportRunner.run` ended."""

    status: str  # done | interrupted
    run: RunView
    report_path: Path | None = None


# --------------------------------------------------------------------- helpers


def _need[T](value: T | None) -> T:
    """``value``, or an :class:`ImportFailure` if the run vanished from the database."""
    if value is None:
        raise ImportFailure("the import run is no longer in the database")
    return value


def parse_exported_at(value: Any, source_time: SourceTime) -> datetime | None:
    """``exportedAt`` (epoch or ISO text) as an aware UTC datetime."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return epoch_to_utc(value)
    if isinstance(value, str):
        text = value.strip()
        if text.replace(".", "", 1).isdigit():
            return epoch_to_utc(text)
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return source_time.parse_local_text(text)
        if parsed.tzinfo is None:
            return source_time.parse_local_text(text)
        return parsed.astimezone(UTC)
    return None


def read_messages_header(path: Path) -> MessagesHeader:
    """The part of ``messages.json`` that is not the ``messages`` array (schema is checked)."""
    try:
        document = {
            key: first_value(path, key)
            for key in ("schemaVersion", "exportedAt", "account", "conversation", "filters")
        }
    except (ijson.JSONError, UnicodeDecodeError) as exc:
        raise ExportLayoutError("messages.json is not valid JSON") from exc
    return parse_header({key: value for key, value in document.items() if value is not None})


# ------------------------------------------------------------------ preparing


@dataclass(frozen=True)
class PreparedImport:
    run_id: str
    resumed: bool
    total: int | None


def prepare_import(services: Services, export_dir: Path, *, target_username: str) -> PreparedImport:
    """Validate the export and create (or reuse) the run for ``twin import`` to queue.

    An unfinished run of the very same files is reused, so it continues where it stopped;
    unfinished runs of other files are marked ``superseded``.
    """
    layout = ExportLayout(export_dir)
    layout.validate()
    info = layout.report()
    entry = layout.find(target_username)
    if entry is None:
        raise TargetNotFoundError("the target conversation is not part of this export")
    if entry.is_group:
        raise ImportFailure("the target conversation is a group chat; groups are never imported")
    fingerprint = layout.fingerprint(entry, info.export_id)
    reused: str | None = None
    with services.db.transaction(bump_state=False) as session:
        for row in unfinished_runs(session):
            if (
                reused is None
                and row.fingerprint == fingerprint
                and row.target_username == target_username
            ):
                reused = row.id
                row.status = "queued"
                row.error = None
            else:
                row.status = "superseded"
    if reused is not None:
        return PreparedImport(reused, True, entry.meta.messageCount)
    run_id = create_run(
        services.db,
        services.clock,
        source_dir=str(export_dir),
        target_username=target_username,
        export_id=info.export_id,
        fingerprint=fingerprint,
        total=entry.meta.messageCount,
    )
    return PreparedImport(run_id, False, entry.meta.messageCount)


def resumable_run(services: Services) -> RunView | None:
    """The newest unfinished run, for ``twin import --resume``."""
    with services.db.session() as session:
        rows = unfinished_runs(session)
        return view_of(rows[0]) if rows else None


# --------------------------------------------------------------------- runner


class ImportRunner:
    """Executes one import run."""

    def __init__(
        self,
        services: Services,
        *,
        batch_size: int | None = None,
        on_batch: Callable[[BatchEvent], None] | None = None,
    ) -> None:
        self._services = services
        self._db = services.db
        self._clock = services.clock
        self._settings: Settings = services.settings
        self._batch_size = batch_size or self._settings.ingest.batch_size
        self._on_batch = on_batch
        self._source_time = SourceTime.from_config(self._settings.time)

    # ------------------------------------------------------------------ entry

    def run(self, run_id: str, stop: threading.Event | None = None) -> RunOutcome:
        """Run (or continue) ``run_id`` to the end, or until ``stop`` is set."""
        signal = stop or threading.Event()
        try:
            return self._run(run_id, signal)
        except ImportFailure as exc:
            self._mark_failed(run_id, str(exc))
            raise
        except UnsupportedSchema as exc:
            self._mark_failed(run_id, str(exc))
            raise ImportFailure(str(exc)) from exc
        except ExportLayoutError as exc:
            self._mark_failed(run_id, str(exc))
            raise ImportFailure(str(exc)) from exc
        except ConversationMismatchError as exc:
            self._mark_failed(run_id, str(exc))
            raise ImportFailure(str(exc)) from exc
        except ijson.JSONError as exc:
            message = "messages.json is not valid JSON or is cut off"
            self._mark_failed(run_id, message)
            raise ImportFailure(message) from exc
        except Exception as exc:
            self._mark_failed(run_id, f"unexpected {type(exc).__name__}; see the log")
            raise

    def _mark_failed(self, run_id: str, message: str) -> None:
        now = self._clock.now_utc()
        with self._db.transaction(bump_state=False) as session:
            row = session.get(ImportRun, run_id)
            if row is not None and row.status != "done":
                row.status = "failed"
                row.error = message[:500]
                row.finished_at = now
        log.error("import_failed", run_id=run_id, error=message[:200])

    # ------------------------------------------------------------------- flow

    def _run(self, run_id: str, stop: threading.Event) -> RunOutcome:
        view = get_run(self._db, run_id)
        if view is None:
            raise ImportFailure("unknown import run")
        if view.status == "done":
            return RunOutcome("done", view, Path(view.report_path) if view.report_path else None)

        layout = ExportLayout(Path(view.source_dir))
        layout.validate()
        info = layout.report()
        entry = layout.find(view.target_username)
        if entry is None:
            raise TargetNotFoundError("the target conversation is not part of this export")
        if entry.is_group:
            raise ImportFailure("the target conversation is a group chat")

        integrity = load_integrity(layout.root)
        stats = view.stats
        self._start(view, layout.fingerprint(entry, info.export_id), stats)
        view = get_run(self._db, run_id) or view
        stats = view.stats
        stats.target_label = entry.masked_name
        self._check_integrity(layout, entry, integrity, stats)

        header = read_messages_header(entry.messages_path)
        exported_at = parse_exported_at(header.exportedAt, self._source_time)
        conversation_id = self._ensure_conversation(entry, header, info, view)
        sealer = RowSealer(get_keyring())
        missing_ids = frozenset(
            str(item.messageId)
            for item in info.report.missingMedia
            if item.messageId is not None
            and (item.conversation is None or item.conversation == entry.username)
        )

        if view.phase in ("queued", "discover", "messages"):
            finished = self._messages_phase(
                view,
                entry,
                conversation_id,
                info,
                exported_at,
                sealer,
                missing_ids,
                stats,
                stop,
            )
            if not finished:
                return self._interrupted(run_id)

        importer = MediaImporter(
            db=self._db,
            media=self._services.media,
            clock=self._clock,
            layout=layout,
            integrity=integrity,
            stats=stats,
            conversation_id=conversation_id,
            sealer=sealer,
            stop=stop,
            on_progress=lambda: self._save_stats(run_id, stats),
        )
        self._set_phase(run_id, "media", stats)
        importer.plan_avatars(
            entry.meta.avatarPath
            or (header.conversation.avatarPath if header.conversation else None),
            stats.user_avatar_path,
        )
        if not importer.run_assets():
            return self._interrupted(run_id)
        importer.finish_avatars()

        self._set_phase(run_id, "stickers", stats)
        if not importer.run_stickers():
            return self._interrupted(run_id)

        self._set_phase(run_id, "finalize", stats)
        self._finalize(run_id, conversation_id, info, stats)

        self._set_phase(run_id, "hooks", stats)
        self._run_hooks(run_id, conversation_id, info)

        return self._complete(run_id, conversation_id)

    # ---------------------------------------------------------------- bookkeeping

    def _start(self, view: RunView, fingerprint: str, stats: RunStats) -> None:
        now = self._clock.now_utc()
        stats.sessions += 1
        with self._db.transaction(bump_state=False) as session:
            row = session.get(ImportRun, view.id)
            if row is None:
                raise ImportFailure("unknown import run")
            if row.fingerprint != fingerprint:
                if row.processed:
                    stats.note("the export files changed after the run was prepared; started over")
                row.fingerprint = fingerprint
                row.processed = 0
                row.inserted = row.duplicates = row.conflict_updated = 0
                row.conflict_kept = row.invalid = 0
                row.phase = "discover"
                stats = RunStats(notes=stats.notes, sessions=stats.sessions)
            row.status = "running"
            row.error = None
            row.finished_at = None
            if row.started_at is None:
                row.started_at = now
            if row.phase == "queued":
                row.phase = "discover"
            row.stats = stats.to_json()

    def _set_phase(self, run_id: str, phase: str, stats: RunStats) -> None:
        with self._db.transaction(bump_state=False) as session:
            row = session.get(ImportRun, run_id)
            if row is not None:
                row.phase = phase
                row.stats = stats.to_json()

    def _save_stats(self, run_id: str, stats: RunStats) -> None:
        with self._db.transaction(bump_state=False) as session:
            row = session.get(ImportRun, run_id)
            if row is not None:
                row.stats = stats.to_json()

    def _interrupted(self, run_id: str) -> RunOutcome:
        with self._db.transaction(bump_state=False) as session:
            row = session.get(ImportRun, run_id)
            if row is not None and row.status == "running":
                row.status = "interrupted"
        view = _need(get_run(self._db, run_id))
        log.info("import_interrupted", run_id=run_id, processed=view.processed)
        return RunOutcome("interrupted", view)

    # ------------------------------------------------------------------ integrity

    def _check_integrity(
        self,
        layout: ExportLayout,
        entry: ConversationEntry,
        integrity: IntegrityIndex,
        stats: RunStats,
    ) -> None:
        record = stats.integrity
        record["state"] = integrity.state.value
        record["entries"] = len(integrity.entries)
        if integrity.state is IntegrityState.UNRECOGNIZED:
            stats.note(
                "the _integrity folder was found but its format is not recognised; "
                "no file was verified"
            )
            return
        if integrity.state is IntegrityState.ABSENT or record.get("messages_json") == "ok":
            return
        relative = normalize_key(f"conversations/{entry.dir_name}/{MESSAGES_FILE}")
        verdict = verify_file(integrity, relative, entry.messages_path)
        record["messages_json"] = verdict.verdict.value
        if verdict.verdict is FileVerdict.FAILED:
            raise IntegrityFailure(
                f"messages.json of the target conversation failed the integrity check "
                f"({verdict.reason}); nothing was imported"
            )

    # -------------------------------------------------------------- conversation

    def _ensure_conversation(
        self, entry: ConversationEntry, header: MessagesHeader, info: ExportInfo, view: RunView
    ) -> str:
        now = self._clock.now_utc()
        display = entry.meta.displayName or (
            header.conversation.displayName if header.conversation else None
        )
        with self._db.transaction(bump_state=False) as session:
            found: Conversation | None = None
            for candidate in session.scalars(select(Conversation)):
                if candidate.username == entry.username:
                    found = candidate
                    break
            if found is None:
                found = Conversation(
                    username=entry.username,
                    display_name=display,
                    is_group=False,
                    message_count=0,
                    created_at=now,
                    updated_at=now,
                )
                session.add(found)
            elif display and found.display_name != display:
                found.display_name = display
            found.last_export_id = info.export_id
            conversation_id = found.id
            run = session.get(ImportRun, view.id)
            if run is not None:
                run.conversation_id = conversation_id
        return conversation_id

    # ------------------------------------------------------------------ messages

    def _messages_phase(
        self,
        view: RunView,
        entry: ConversationEntry,
        conversation_id: str,
        info: ExportInfo,
        exported_at: datetime | None,
        sealer: RowSealer,
        missing_ids: frozenset[str],
        stats: RunStats,
        stop: threading.Event,
    ) -> bool:
        self._set_phase(view.id, "messages", stats)
        context = NormalizeContext(entry.username, self._source_time)
        started = self._clock.monotonic()
        start_count = view.processed
        processed = view.processed
        batch_number = 0
        with long_path(entry.messages_path).open("rb") as handle:
            skip_bom(handle)
            remaining = iter_messages(handle, processed)
            while True:
                chunk = list(islice(remaining, self._batch_size))
                if not chunk:
                    break
                items, invalid = self._normalize(chunk, context, stats)
                batch_context = BatchContext(
                    conversation_id=conversation_id,
                    export_id=info.export_id,
                    exported_at=exported_at,
                    now=self._clock.now_utc(),
                    sealer=sealer,
                    missing_message_ids=missing_ids,
                )
                processed += len(chunk)
                elapsed = self._clock.monotonic() - started
                speed = (processed - start_count) / elapsed if elapsed > 0 else None
                with self._db.transaction(bump_state=False) as session:
                    result = persist_messages(session, items, batch_context)
                    row = _need(session.get(ImportRun, view.id))
                    row.processed = processed
                    row.inserted += result.inserted
                    row.duplicates += result.duplicates
                    row.conflict_updated += result.conflict_updated
                    row.conflict_kept += result.conflict_kept
                    row.invalid += invalid
                    row.speed_per_s = speed
                    if items:
                        row.last_message_id = items[-1].id
                        if items[-1].sort_seq is not None:
                            row.last_sort_seq = items[-1].sort_seq
                    row.stats = stats.to_json()
                batch_number += 1
                if self._on_batch is not None:
                    self._on_batch(
                        BatchEvent(
                            view.id,
                            batch_number,
                            processed,
                            view.total,
                            result.inserted,
                            speed,
                        )
                    )
                if stop.is_set():
                    return False
        if view.total is not None and processed != view.total:
            stats.note(
                f"meta.json announces {view.total} messages, messages.json holds {processed}"
            )
        return True

    def _normalize(
        self, chunk: list[Any], context: NormalizeContext, stats: RunStats
    ) -> tuple[list[NormalizedMessage], int]:
        items: list[NormalizedMessage] = []
        invalid = 0
        for raw in chunk:
            if not isinstance(raw, dict):
                invalid += 1
                stats.invalid_reasons["not_an_object"] += 1
                continue
            try:
                message = self._validate(raw, stats)
                item = normalize_message(raw, message, context)
            except InvalidMessage as exc:
                invalid += 1
                stats.invalid_reasons[exc.reason] += 1
                continue
            label = f"{item.kind}:{'user' if item.is_sent else 'her'}"
            stats.kinds[label] += 1
            if item.time_mismatch:
                stats.time_mismatches += 1
            if not item.render_type_known:
                stats.unknown_render_types[item.render_type or "-"] += 1
            for name in item.unknown_fields:
                stats.unknown_fields[name] += 1
            if item.is_sent and stats.user_avatar_path is None and item.sender_avatar_path:
                stats.user_avatar_path = item.sender_avatar_path
            items.append(item)
        return items, invalid

    @staticmethod
    def _validate(raw: dict[str, Any], stats: RunStats) -> ExportMessage:
        try:
            return ExportMessage.model_validate(raw)
        except ValidationError as exc:
            bad = {str(error["loc"][0]) for error in exc.errors() if error["loc"]}
            for name in bad:
                stats.schema_errors[name] += 1
            try:
                return ExportMessage.model_validate({k: v for k, v in raw.items() if k not in bad})
            except ValidationError:
                raise InvalidMessage("schema") from None

    # ------------------------------------------------------------------- closing

    def _finalize(
        self, run_id: str, conversation_id: str, info: ExportInfo, stats: RunStats
    ) -> None:
        with self._db.transaction(bump_state=False) as session:
            count, first, last = session.execute(
                select(
                    func.count(Message.id),
                    func.min(Message.create_time_utc),
                    func.max(Message.create_time_utc),
                ).where(Message.conversation_id == conversation_id)
            ).one()
            session.execute(
                update(Conversation)
                .where(Conversation.id == conversation_id)
                .values(
                    message_count=count,
                    first_message_at=first,
                    last_message_at=last,
                    last_export_id=info.export_id,
                    updated_at=self._clock.now_utc(),
                )
            )
            row = session.get(ImportRun, run_id)
            if row is not None:
                row.stats = stats.to_json()

    def _run_hooks(self, run_id: str, conversation_id: str, info: ExportInfo) -> None:
        view = _need(get_run(self._db, run_id))
        registry = load_hooks()
        done = {name for name, entry in view.hooks.items() if entry.get("status") == "done"}
        context = HookContext(
            services=self._services,
            run_id=run_id,
            conversation_id=conversation_id,
            export_id=info.export_id,
            inserted=view.inserted,
            changed=view.conflict_updated,
            first_import=not finished_runs_exist(self._db, conversation_id),
        )

        def remember(outcome: HookOutcome) -> None:
            with self._db.transaction(bump_state=False) as session:
                row = session.get(ImportRun, run_id)
                if row is None:
                    return
                entries = [dict(entry) for entry in row.hooks or []]
                result = {
                    "name": outcome.name,
                    "status": outcome.result.status,
                    "detail": outcome.result.detail,
                    "jobs": outcome.result.jobs,
                    "backfill_command": outcome.backfill_command,
                }
                for position, entry in enumerate(entries):
                    if entry["name"] == outcome.name:
                        entries[position] = result
                        break
                else:
                    entries.append(result)
                row.hooks = entries

        # every registered hook is listed in registration order before it has run
        with self._db.transaction(bump_state=False) as session:
            row = session.get(ImportRun, run_id)
            if row is not None:
                entries = [dict(entry) for entry in row.hooks or []]
                listed = {entry["name"] for entry in entries}
                for hook in registry.hooks():
                    if hook.name not in listed:
                        entries.append(
                            {
                                "name": hook.name,
                                "status": "pending",
                                "detail": "",
                                "jobs": 0,
                                "backfill_command": hook.backfill_command,
                            }
                        )
                row.hooks = entries
        run_hooks(context, registry, skip=done, on_result=remember)

    def _complete(self, run_id: str, conversation_id: str) -> RunOutcome:
        now = self._clock.now_utc()
        view = _need(get_run(self._db, run_id))
        text = build_report(
            self._services, view, conversation_id, self._source_time, now, status="done"
        )
        path = write_report(self._services.paths.reports_dir, text, now)
        with self._db.transaction(bump_state=False) as session:
            row = _need(session.get(ImportRun, run_id))
            row.status = "done"
            row.phase = "done"
            row.finished_at = now
            row.report_path = str(path)
        final = _need(get_run(self._db, run_id))
        log.info(
            "import_done",
            run_id=run_id,
            processed=final.processed,
            inserted=final.inserted,
            duplicates=final.duplicates,
        )
        return RunOutcome("done", final, path)
