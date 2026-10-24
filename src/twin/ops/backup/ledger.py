"""The ``backup_records`` table as the backup code uses it (R-OPS-006, R-STO-006)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select

from twin.clock import Clock
from twin.ops.backup.archive import Manifest
from twin.storage.db import Database
from twin.storage.ops_models import BackupRecord


@dataclass(frozen=True)
class BackupView:
    """A row of ``backup_records``."""

    id: str
    kind: str
    status: str
    local_date: str
    file_name: str | None
    size_bytes: int
    sha256: str | None
    duration_ms: int
    key_id: int | None
    key_ids: tuple[int, ...]
    row_count: int
    media_count: int
    schema_revision: str | None
    error: str | None
    created_at: datetime
    mirrored_at: datetime | None
    deleted_at: datetime | None


def _view(row: BackupRecord) -> BackupView:
    return BackupView(
        id=row.id,
        kind=row.kind,
        status=row.status,
        local_date=row.local_date,
        file_name=row.file_name,
        size_bytes=row.size_bytes,
        sha256=row.sha256,
        duration_ms=row.duration_ms,
        key_id=row.key_id,
        key_ids=tuple(row.key_ids or ()),
        row_count=row.row_count,
        media_count=row.media_count,
        schema_revision=row.schema_revision,
        error=row.error,
        created_at=row.created_at,
        mirrored_at=row.mirrored_at,
        deleted_at=row.deleted_at,
    )


class BackupLedger:
    """Reads and writes ``backup_records``."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def add_ok(
        self,
        manifest: Manifest,
        *,
        kind: str,
        file_name: str,
        size_bytes: int,
        sha256: str,
        duration_ms: int,
        created_at: datetime | None = None,
    ) -> BackupView:
        """Record a finished backup; an older record of the same file is marked replaced."""
        now = created_at or self._clock.now_utc()
        with self._db.transaction(bump_state=False) as session:
            for old in session.scalars(
                select(BackupRecord).where(
                    BackupRecord.file_name == file_name, BackupRecord.status == "ok"
                )
            ):
                old.status = "deleted"
                old.deleted_at = now
            row = BackupRecord(
                kind=kind,
                status="ok",
                local_date=manifest.local_date,
                file_name=file_name,
                size_bytes=size_bytes,
                sha256=sha256,
                duration_ms=duration_ms,
                key_id=manifest.key_id,
                key_ids=sorted(manifest.key_ids),
                row_count=manifest.row_count,
                media_count=len(manifest.media),
                schema_revision=manifest.schema_revision,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            session.flush()
            return _view(row)

    def add_failed(self, kind: str, local_date: str, error: str, duration_ms: int) -> BackupView:
        now = self._clock.now_utc()
        with self._db.transaction(bump_state=False) as session:
            row = BackupRecord(
                kind=kind,
                status="failed",
                local_date=local_date,
                size_bytes=0,
                duration_ms=duration_ms,
                key_ids=[],
                row_count=0,
                media_count=0,
                error=error[:160],
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            session.flush()
            return _view(row)

    def usable(self) -> list[BackupView]:
        """Backups whose file should exist, newest first."""
        with self._db.session() as session:
            rows = session.scalars(
                select(BackupRecord)
                .where(BackupRecord.status == "ok")
                .order_by(BackupRecord.created_at.desc(), BackupRecord.id.desc())
            )
            return [_view(row) for row in rows]

    def recent(self, *, limit: int = 30) -> list[BackupView]:
        """All records, newest first (failures and deletions included)."""
        with self._db.session() as session:
            rows = session.scalars(
                select(BackupRecord)
                .order_by(BackupRecord.created_at.desc(), BackupRecord.id.desc())
                .limit(limit)
            )
            return [_view(row) for row in rows]

    def newest_ok(self) -> BackupView | None:
        found = self.usable()
        return found[0] if found else None

    def mark_deleted(self, record_id: str) -> None:
        now = self._clock.now_utc()
        with self._db.transaction(bump_state=False) as session:
            row = session.get(BackupRecord, record_id)
            if row is not None and row.status == "ok":
                row.status = "deleted"
                row.deleted_at = now

    def mark_mirrored(self, record_id: str) -> None:
        now = self._clock.now_utc()
        with self._db.transaction(bump_state=False) as session:
            row = session.get(BackupRecord, record_id)
            if row is not None:
                row.mirrored_at = now

    def kept_key_ids(self) -> set[int]:
        """Every key id a kept backup depends on."""
        keys: set[int] = set()
        for view in self.usable():
            keys.update(view.key_ids)
            if view.key_id is not None:
                keys.add(view.key_id)
        return keys
