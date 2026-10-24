"""Tables of the operation of the program (round 12; R-STO-006, R-OPS-003, R-OPS-006).

``health_snapshots``
    one row per minute while the application runs: when, the overall verdict, when this process
    started and how it was started (the scheduled task, by hand, or without a supervisor), whether
    the chat channel works and when its last long poll succeeded, and one entry per check
    (``checks``: ``{name: {"status", "detail", "value"}}``).  Everything in it is a number, a time
    or a short code - never a message - so the table is not encrypted.  Rows older than
    ``ops.health.keep_days`` are deleted.  The stability report (R-EVAL-006) reads nothing else
    about the past of the process.
``backup_records``
    one row per backup that was made, attempted or deleted: the file name below the backups
    directory, its size and SHA-256, how long it took, the key the archive is sealed with
    (``key_id``) and every key id the archive depends on (``key_ids``: the archive's own plus the
    ones its database rows and media files were sealed with).  The retention policy (14 daily and
    8 weekly backups) and the rule that a retired database key is deleted only when no kept backup
    needs it (R-STO-003) are computed from this table.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, BigInteger, Boolean, CheckConstraint, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from twin.storage.models import Base, TimestampMixin
from twin.storage.types import UTCDateTime

HEALTH_STATUSES = ("ok", "degraded", "unhealthy")
LAUNCH_KINDS = ("task", "manual", "unsupervised")
BACKUP_KINDS = ("daily", "manual", "pre_restore")
BACKUP_STATUSES = ("ok", "failed", "deleted")


def _in_list(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class HealthSnapshot(TimestampMixin, Base):
    """The result of one health check (R-OPS-003)."""

    __tablename__ = "health_snapshots"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    status: Mapped[str] = mapped_column(String(10), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    launch: Mapped[str] = mapped_column(String(12), nullable=False, default="unsupervised")
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    channel_ok: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    channel_last_ok_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    checks: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    __table_args__ = (
        CheckConstraint(_in_list("status", HEALTH_STATUSES), name="status"),
        CheckConstraint(_in_list("launch", LAUNCH_KINDS), name="launch"),
        Index("ix_health_snapshots_at", "at"),
    )


class BackupRecord(TimestampMixin, Base):
    """One backup (R-OPS-006); the archive is ``data/backups/<file_name>``."""

    __tablename__ = "backup_records"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    kind: Mapped[str] = mapped_column(String(12), nullable=False)
    status: Mapped[str] = mapped_column(String(10), nullable=False)
    local_date: Mapped[str] = mapped_column(String(10), nullable=False)
    file_name: Mapped[str | None] = mapped_column(String(160), nullable=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    key_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    key_ids: Mapped[list[int]] = mapped_column(JSON, nullable=False, default=list)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    media_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    schema_revision: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error: Mapped[str | None] = mapped_column(String(160), nullable=True)
    mirrored_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    __table_args__ = (
        CheckConstraint(_in_list("kind", BACKUP_KINDS), name="kind"),
        CheckConstraint(_in_list("status", BACKUP_STATUSES), name="status"),
        Index("ix_backup_records_status_date", "status", "local_date"),
    )
