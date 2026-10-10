"""Building and updating the vector index of the example windows (R-RET-002, R-RET-006).

One run (:func:`run_index`) does, in order:

1. :func:`~twin.retrieval.windows.sync_windows` - the windows follow the stored messages;
   windows that vanished or entered the hold-out leave the index;
2. the identity check - the index remembers which encoding (model, size, weights hash, text
   scheme) made its vectors.  A different encoding is **refused** for an update (it would mix two
   vector spaces) and answered with "run ``twin retrieval rebuild``"; a rebuild starts from an
   empty index;
3. the encoding of every window that has no vector yet (``embed_version`` NULL, not held out, with
   a context), in chunks of :data:`WRITE_CHUNK`.  A chunk is written to the index first and
   marked in ``example_windows`` second, so an interrupted run loses at most one chunk and the
   next run picks up exactly where this one stopped (the pending set is always derived from the
   database, never from memory).  Progress, speed and the estimated time left are kept in the
   ``retrieval.index_progress`` setting for ``twin retrieval stats`` and the foreground bar.

Before encoding, text is redacted by the embedding service (R-LLM-009); the index receives ids,
vectors, times and the slot/day type only.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.sql.elements import ColumnElement

from twin.clock import to_epoch
from twin.ops.filelock import FileLock
from twin.ops.logging import get_logger
from twin.profile.holdout import HoldoutError, get_holdout
from twin.retrieval.embedder import (
    EmbedderInfo,
    EmbeddingService,
    EncodeKind,
    cpu_speed_warning,
    embedding_service,
    read_manifest,
)
from twin.retrieval.records import load_messages
from twin.retrieval.texts import SCHEME, encoding_text, turn_text
from twin.retrieval.vector_store import IndexMeta, IndexMismatchError, VectorStore, VectorTable
from twin.retrieval.windows import SyncReport, sync_windows
from twin.storage.retrieval_models import ExampleWindow
from twin.storage.settings_store import get_setting, put_setting
from twin.storage.vector_schema import WINDOW_SCHEMA

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.retrieval.indexer")

INDEX_JOB = "retrieval_index"
PROGRESS_KEY = "retrieval.index_progress"
NO_VECTOR = "-"  # embed_version of a window whose context has nothing to encode
WRITE_CHUNK = 256
IndexMode = Literal["update", "rebuild"]


class IndexBusyError(RuntimeError):
    """Another index run holds the lock."""


def encoding_id(info: EmbedderInfo) -> str:
    """The identity of an encoding: what ``embed_version`` holds for an indexed window."""
    return f"{info.model}|{info.dimension}|{info.weights_sha256[:16]}|{SCHEME}"


def window_table(services: Services) -> VectorTable:
    """The LanceDB table of the example windows (``data/vectors/``)."""
    return VectorStore(services.paths.vectors_dir).table(WINDOW_SCHEMA)


# ------------------------------------------------------------------- progress


@dataclass(frozen=True)
class IndexProgress:
    """How far an index run is (stored in the ``retrieval.index_progress`` setting)."""

    state: str  # running | done | stopped
    mode: str
    total: int
    done: int
    started_at: str
    updated_at: str
    seconds: float
    rate_per_s: float | None
    eta_s: float | None
    detail: str = ""

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> IndexProgress:
        return cls(
            state=str(data.get("state", "done")),
            mode=str(data.get("mode", "update")),
            total=int(data.get("total", 0)),
            done=int(data.get("done", 0)),
            started_at=str(data.get("started_at", "")),
            updated_at=str(data.get("updated_at", "")),
            seconds=float(data.get("seconds", 0.0)),
            rate_per_s=_optional_float(data.get("rate_per_s")),
            eta_s=_optional_float(data.get("eta_s")),
            detail=str(data.get("detail", "")),
        )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def get_progress(services: Services) -> IndexProgress | None:
    with services.db.session() as session:
        raw = get_setting(session, PROGRESS_KEY)
    return IndexProgress.from_json(raw) if isinstance(raw, dict) else None


def _put_progress(services: Services, progress: IndexProgress) -> None:
    with services.db.transaction(bump_state=False) as session:
        put_setting(
            session,
            PROGRESS_KEY,
            asdict(progress),
            clock=services.clock,
            by="retrieval",
            record_history=False,
        )


# ---------------------------------------------------------------------- state


def indexable() -> list[ColumnElement[bool]]:
    """Windows that belong in the index: not held out, with a context to encode."""
    return [ExampleWindow.holdout.is_(False), ExampleWindow.context_turns > 0]


def vector_holders() -> list[ColumnElement[bool]]:
    """Indexable windows that are marked as having a vector in the index."""
    return [
        *indexable(),
        ExampleWindow.embed_version.is_not(None),
        ExampleWindow.embed_version != NO_VECTOR,
    ]


def awaiting(current: str | None) -> list[ColumnElement[bool]]:
    """Indexable windows without a vector made by ``current`` (all without one if ``None``)."""
    stale: ColumnElement[bool] = ExampleWindow.embed_version.is_(None)
    if current is not None:
        stale = or_(
            stale,
            and_(ExampleWindow.embed_version != current, ExampleWindow.embed_version != NO_VECTOR),
        )
    return [*indexable(), stale]


def count_windows(services: Services, *where: ColumnElement[bool]) -> int:
    with services.db.session() as session:
        return int(
            session.scalar(select(func.count()).select_from(ExampleWindow).where(*where)) or 0
        )


def _reset_markers(services: Services) -> None:
    """Forget that any window has a vector (the index is empty or was dropped)."""
    with services.db.transaction(bump_state=False) as session:
        session.execute(
            update(ExampleWindow)
            .where(ExampleWindow.embed_version.is_not(None))
            .values(embed_version=None)
        )


def _mark(services: Services, ids: list[str], version: str) -> None:
    now = services.clock.now_utc()
    with services.db.transaction(bump_state=False) as session:
        session.execute(
            update(ExampleWindow)
            .where(ExampleWindow.id.in_(ids))
            .values(embed_version=version, updated_at=now)
        )


# --------------------------------------------------------------------- result


@dataclass(frozen=True)
class IndexResult:
    """What one index run did."""

    note: str
    sync: SyncReport | None = None
    encoded: int = 0
    pending: int = 0
    reset: bool = False
    seconds: float = 0.0
    encoding: str | None = None

    @property
    def per_thousand_s(self) -> float | None:
        """Seconds of encoding per 1,000 windows (``None`` before anything was encoded)."""
        return self.seconds / self.encoded * 1000.0 if self.encoded and self.seconds else None


# ------------------------------------------------------------------ encoding


def _window_texts(services: Services, ids: list[str]) -> dict[str, str]:
    """The encoding text of each window (empty if its context has no words)."""
    with services.db.session() as session:
        rows = session.execute(
            select(ExampleWindow.id, ExampleWindow.context_block_ids).where(
                ExampleWindow.id.in_(ids)
            )
        ).all()
        wanted = [i for _, turns in rows for turn in turns for i in turn]
        messages = load_messages(session, wanted)
    texts: dict[str, str] = {}
    for window_id, turns in rows:
        built = []
        for turn in turns:
            members = [messages[i] for i in turn if i in messages]
            if members:
                built.append(turn_text(members))
        texts[window_id] = encoding_text(built)
    return texts


def _records(services: Services, vectors: dict[str, np.ndarray[Any, Any]]) -> list[dict[str, Any]]:
    with services.db.session() as session:
        rows = session.execute(
            select(
                ExampleWindow.id,
                ExampleWindow.reply_at_utc,
                ExampleWindow.local_slot,
                ExampleWindow.day_type,
            ).where(ExampleWindow.id.in_(list(vectors)))
        ).all()
    return [
        {
            "window_id": row.id,
            "vector": vectors[row.id].tolist(),
            "reply_at_utc": int(to_epoch(row.reply_at_utc)),
            "local_slot": int(row.local_slot),
            "day_type": row.day_type,
        }
        for row in rows
    ]


def _progress_of(
    services: Services, mode: str, total: int, done: int, started: str, t0: float, state: str
) -> IndexProgress:
    elapsed = services.clock.monotonic() - t0
    rate = done / elapsed if elapsed > 0 and done else None
    eta = (total - done) / rate if rate else None
    return IndexProgress(
        state=state,
        mode=mode,
        total=total,
        done=done,
        started_at=started,
        updated_at=services.clock.now_utc().isoformat(),
        seconds=elapsed,
        rate_per_s=rate,
        eta_s=eta if state == "running" else None,
    )


def run_index(
    services: Services,
    *,
    mode: IndexMode = "update",
    full: bool = False,
    embedder: EmbeddingService | None = None,
    stop: threading.Event | None = None,
) -> IndexResult:
    """Sync the windows and encode what is missing (see the module docstring).

    ``mode="update"`` is the incremental run after an import; it refuses an index made with
    another encoding.  ``mode="rebuild"`` replaces such an index; ``full=True`` re-encodes
    everything even if nothing changed.
    """
    lock = FileLock(services.paths.locks_dir / "retrieval-index.lock")
    if not lock.acquire():
        raise IndexBusyError("another index run is active; wait for it to finish")
    try:
        return _run_locked(services, mode, full, embedder, stop)
    finally:
        lock.release()


def _weights_unchanged(services: Services, meta: IndexMeta) -> bool:
    """False if the model files on disk are not the ones the index was made with."""
    record = read_manifest(services.paths.embeddings_dir, services.settings.retrieval.model)
    return record is None or record.get("weights_sha256") == meta.weights_sha256


def _run_locked(
    services: Services,
    mode: IndexMode,
    full: bool,
    embedder: EmbeddingService | None,
    stop: threading.Event | None,
) -> IndexResult:
    config = services.settings.retrieval
    table = window_table(services)
    started_at = services.clock.now_utc().isoformat()
    t0 = services.clock.monotonic()
    try:
        report = sync_windows(services)
    except HoldoutError as exc:
        return IndexResult(note=f"nothing indexed: {exc}")

    table.delete_ids(list(report.removed_ids) + list(report.newly_held_ids))
    table.delete_from(int(to_epoch(report.cutoff)) + 1)

    meta = table.read_meta()
    waiting = count_windows(services, *awaiting(None))
    stamped = count_windows(services, *vector_holders())
    damaged = stamped > 0 and (meta is None or table.count() < stamped)
    up_to_date = (
        not full
        and not damaged
        and waiting == 0
        and meta is not None
        and meta.model == config.model
        and _weights_unchanged(services, meta)
        and count_windows(services, *awaiting(meta.encoding)) == 0
    )
    if up_to_date:
        _put_progress(services, _progress_of(services, mode, 0, 0, started_at, t0, "done"))
        return IndexResult(
            note="the index is up to date", sync=report, encoding=meta.encoding if meta else None
        )

    service = embedder or embedding_service(services)
    info = service.info
    current = encoding_id(info)
    reset = full or damaged
    if meta is not None and meta.encoding != current:
        if mode == "update" and not full:
            raise IndexMismatchError(
                f"the index was made with {meta.model} ({meta.encoding}) but the embedding "
                f"model is now {info.describe()}; run `twin retrieval rebuild`"
            )
        reset = True
    if reset:
        table.drop()
        table.clear_meta()
        _reset_markers(services)
    table.create(info.dimension)
    table.write_meta(
        IndexMeta(
            table=WINDOW_SCHEMA.name,
            encoding=current,
            model=info.model,
            dimension=info.dimension,
            revision=info.revision,
            weights_sha256=info.weights_sha256,
            written_at=services.clock.now_utc().isoformat(),
        )
    )

    with services.db.session() as session:
        pending = list(
            session.scalars(
                select(ExampleWindow.id)
                .where(*awaiting(current))
                .order_by(ExampleWindow.reply_at_utc, ExampleWindow.id)
            )
        )
    total = len(pending)
    warning = cpu_speed_warning(config, info.device, total)
    if warning:
        log.warning("embedding_model_slow_on_cpu", windows=total)
    encoded = done = 0
    stopped = False
    t0 = services.clock.monotonic()  # speed and time left count the encoding only
    for start in range(0, total, WRITE_CHUNK):
        if stop is not None and stop.is_set():
            stopped = True
            break
        chunk = pending[start : start + WRITE_CHUNK]
        texts = _window_texts(services, chunk)
        usable = [i for i in chunk if texts.get(i)]
        empty = [i for i in chunk if not texts.get(i)]
        if usable:
            matrix = service.encode([texts[i] for i in usable], EncodeKind.SYMMETRIC)
            vectors = dict(zip(usable, matrix, strict=True))
            table.upsert(_records(services, vectors), dimension=info.dimension)
            _mark(services, usable, current)
        if empty:
            _mark(services, empty, NO_VECTOR)
        encoded += len(usable)
        done += len(chunk)
        progress = _progress_of(services, mode, total, done, started_at, t0, "running")
        _put_progress(services, progress)
    if encoded:
        table.optimize()
    seconds = services.clock.monotonic() - t0
    state = "stopped" if stopped else "done"
    _put_progress(services, _progress_of(services, mode, total, done, started_at, t0, state))
    finished = not stopped
    note = (
        f"{encoded:,} window(s) encoded"
        if finished
        else f"stopped after {encoded:,} of {total:,} window(s); the next run continues"
    )
    if warning:
        note += f"; {warning}"
    return IndexResult(
        note=note,
        sync=report,
        encoded=encoded,
        pending=total,
        reset=reset,
        seconds=seconds,
        encoding=current,
    )


# ----------------------------------------------------------------- statistics


@dataclass(frozen=True)
class IndexStats:
    """What ``twin retrieval stats`` shows."""

    windows: int
    with_context: int
    without_context: int
    event_only: int
    held_out: int
    indexed: int
    awaiting: int
    no_vector: int
    vectors: int
    cutoff: datetime | None
    first_reply: datetime | None
    last_reply: datetime | None
    meta: IndexMeta | None
    configured_model: str
    progress: IndexProgress | None
    problems: tuple[str, ...]


def collect_stats(services: Services) -> IndexStats:
    config = services.settings.retrieval
    table = window_table(services)
    meta = table.read_meta()
    holdout = get_holdout(services)
    with services.db.session() as session:

        def count(*where: ColumnElement[bool]) -> int:
            query = select(func.count()).select_from(ExampleWindow).where(*where)
            return int(session.scalar(query) or 0)

        span = session.execute(
            select(func.min(ExampleWindow.reply_at_utc), func.max(ExampleWindow.reply_at_utc))
        ).one()
        windows = count()
        stats = {
            "with_context": count(ExampleWindow.context_turns > 0),
            "held_out": count(ExampleWindow.holdout.is_(True)),
            "event_only": count(ExampleWindow.reply_reproducible == 0),
            "indexed": count(
                *indexable(),
                ExampleWindow.embed_version.is_not(None),
                ExampleWindow.embed_version != NO_VECTOR,
            ),
            "awaiting": count(*awaiting(meta.encoding if meta else None)),
            "no_vector": count(ExampleWindow.embed_version == NO_VECTOR),
        }
    vectors = table.count()
    problems: list[str] = []
    if meta is not None and meta.model != config.model:
        problems.append(
            f"the index was made with {meta.model} but retrieval.model is {config.model}: "
            "run `twin retrieval rebuild`"
        )
    if stats["indexed"] > vectors:
        problems.append(
            f"{stats['indexed']:,} windows are marked as indexed but the index holds "
            f"{vectors:,} vectors: run `twin retrieval rebuild --full`"
        )
    if windows and meta is None and stats["indexed"]:
        problems.append("the index metadata is missing: run `twin retrieval rebuild --full`")
    return IndexStats(
        windows=windows,
        with_context=stats["with_context"],
        without_context=windows - stats["with_context"],
        event_only=stats["event_only"],
        held_out=stats["held_out"],
        indexed=stats["indexed"],
        awaiting=stats["awaiting"],
        no_vector=stats["no_vector"],
        vectors=vectors,
        cutoff=holdout.cutoff if holdout else None,
        first_reply=span[0],
        last_reply=span[1],
        meta=meta,
        configured_model=config.model,
        progress=get_progress(services),
        problems=tuple(problems),
    )
