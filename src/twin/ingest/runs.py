"""Reading and writing ``import_runs`` (R-IMP-006, R-ARCH-006).

``RunView`` is an immutable snapshot of a run for status displays and reports;
the importer updates the row inside its own batch transactions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from twin.clock import Clock
from twin.ingest.stats import RunStats
from twin.storage.chat_models import ImportRun
from twin.storage.db import Database

UNFINISHED = ("queued", "running", "interrupted", "failed")
TERMINAL = ("done", "failed", "superseded")


@dataclass(frozen=True)
class RunView:
    id: str
    status: str
    phase: str
    export_id: str | None
    conversation_id: str | None
    total: int | None
    processed: int
    inserted: int
    duplicates: int
    conflict_updated: int
    conflict_kept: int
    invalid: int
    last_sort_seq: int | None
    last_message_id: str | None
    speed_per_s: float | None
    started_at: datetime | None
    updated_at: datetime
    finished_at: datetime | None
    error: str | None
    report_path: str | None
    stats: RunStats
    hooks: dict[str, Any] = field(default_factory=dict)
    source_dir: str = ""
    target_username: str = ""

    @property
    def eta_seconds(self) -> float | None:
        if not self.speed_per_s or self.total is None or self.processed >= self.total:
            return None
        return (self.total - self.processed) / self.speed_per_s

    @property
    def finished(self) -> bool:
        return self.status in ("done", "superseded")


def view_of(row: ImportRun) -> RunView:
    return RunView(
        id=row.id,
        status=row.status,
        phase=row.phase,
        export_id=row.export_id,
        conversation_id=row.conversation_id,
        total=row.total,
        processed=row.processed,
        inserted=row.inserted,
        duplicates=row.duplicates,
        conflict_updated=row.conflict_updated,
        conflict_kept=row.conflict_kept,
        invalid=row.invalid,
        last_sort_seq=row.last_sort_seq,
        last_message_id=row.last_message_id,
        speed_per_s=row.speed_per_s,
        started_at=row.started_at,
        updated_at=row.updated_at,
        finished_at=row.finished_at,
        error=row.error,
        report_path=row.report_path,
        stats=RunStats.from_json(row.stats),
        hooks={entry["name"]: dict(entry) for entry in row.hooks or []},
        source_dir=row.source_dir,
        target_username=row.target_username,
    )


def create_run(
    db: Database,
    clock: Clock,
    *,
    source_dir: str,
    target_username: str,
    export_id: str | None,
    fingerprint: str,
    total: int | None,
) -> str:
    now = clock.now_utc()
    run = ImportRun(
        source_dir=source_dir,
        target_username=target_username,
        export_id=export_id,
        fingerprint=fingerprint,
        total=total,
        status="queued",
        phase="queued",
        processed=0,
        inserted=0,
        duplicates=0,
        conflict_updated=0,
        conflict_kept=0,
        invalid=0,
        stats=RunStats().to_json(),
        created_at=now,
        updated_at=now,
    )
    with db.transaction() as session:
        session.add(run)
        run_id = run.id
    return run_id


def get_run(db: Database, run_id: str) -> RunView | None:
    with db.session() as session:
        row = session.get(ImportRun, run_id)
        return view_of(row) if row is not None else None


def latest_run(db: Database) -> RunView | None:
    with db.session() as session:
        row = session.scalars(
            select(ImportRun).order_by(ImportRun.created_at.desc(), ImportRun.id.desc()).limit(1)
        ).first()
        return view_of(row) if row is not None else None


def unfinished_runs(session: Session) -> list[ImportRun]:
    return list(
        session.scalars(
            select(ImportRun)
            .where(ImportRun.status.in_(UNFINISHED))
            .order_by(ImportRun.created_at.desc(), ImportRun.id.desc())
        )
    )


def finished_runs_exist(db: Database, conversation_id: str) -> bool:
    with db.session() as session:
        return (
            session.scalars(
                select(ImportRun.id)
                .where(ImportRun.conversation_id == conversation_id, ImportRun.status == "done")
                .limit(1)
            ).first()
            is not None
        )
