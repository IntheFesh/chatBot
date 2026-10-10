"""Round 12: health_snapshots, backup_records, alert delivery columns, eval kind ``stability``.

Revision ID: 0015_ops_tables
Revises: 0013_preference_pairs
Create Date: 2026-10-09
"""

import re
from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op

revision: str = "0015_ops_tables"
down_revision: str | None = "0013_preference_pairs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

NEW_KIND = "stability"
_KIND_LIST = re.compile(r"kind\s+IN\s*\(([^)]*)\)", re.IGNORECASE)


def _timestamps() -> list[sa.Column[datetime]]:
    return [
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    ]


def _eval_kinds() -> list[str]:
    """The kinds ``eval_runs`` accepts now, read from its table definition.

    Read instead of written down so that a kind another round added to the same constraint is
    kept when this migration adds its own.
    """
    row = (
        op.get_bind()
        .execute(
            sa.text("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'eval_runs'")
        )
        .fetchone()
    )
    found = _KIND_LIST.search(str(row[0])) if row else None
    if found is None:
        raise RuntimeError("cannot read the kinds of eval_runs from its table definition")
    return re.findall(r"'([^']+)'", found.group(1))


def _set_eval_kinds(kinds: list[str], width: int) -> None:
    listed = ", ".join(repr(kind) for kind in kinds)
    with op.batch_alter_table("eval_runs", recreate="always") as batch:
        batch.drop_constraint(op.f("ck_eval_runs_kind"), type_="check")
        batch.alter_column(
            "kind",
            existing_type=sa.String(),
            type_=sa.String(length=width),
            existing_nullable=False,
        )
        batch.create_check_constraint(op.f("ck_eval_runs_kind"), f"kind IN ({listed})")


def upgrade() -> None:
    op.create_table(
        "health_snapshots",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("at", sa.DateTime(), nullable=False),
        sa.Column("status", sa.String(length=10), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("launch", sa.String(length=12), nullable=False),
        sa.Column("pid", sa.Integer(), nullable=True),
        sa.Column("channel_ok", sa.Boolean(), nullable=True),
        sa.Column("channel_last_ok_at", sa.DateTime(), nullable=True),
        sa.Column("checks", sa.JSON(), nullable=False),
        *_timestamps(),
        sa.CheckConstraint(
            "status IN ('ok', 'degraded', 'unhealthy')", name=op.f("ck_health_snapshots_status")
        ),
        sa.CheckConstraint(
            "launch IN ('task', 'manual', 'unsupervised')",
            name=op.f("ck_health_snapshots_launch"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_health_snapshots")),
    )
    op.create_index("ix_health_snapshots_at", "health_snapshots", ["at"], unique=False)

    op.create_table(
        "backup_records",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("kind", sa.String(length=12), nullable=False),
        sa.Column("status", sa.String(length=10), nullable=False),
        sa.Column("local_date", sa.String(length=10), nullable=False),
        sa.Column("file_name", sa.String(length=160), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("key_id", sa.Integer(), nullable=True),
        sa.Column("key_ids", sa.JSON(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("media_count", sa.Integer(), nullable=False),
        sa.Column("schema_revision", sa.String(length=64), nullable=True),
        sa.Column("error", sa.String(length=160), nullable=True),
        sa.Column("mirrored_at", sa.DateTime(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            "kind IN ('daily', 'manual', 'pre_restore')", name=op.f("ck_backup_records_kind")
        ),
        sa.CheckConstraint(
            "status IN ('ok', 'failed', 'deleted')", name=op.f("ck_backup_records_status")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_backup_records")),
    )
    op.create_index(
        "ix_backup_records_status_date", "backup_records", ["status", "local_date"], unique=False
    )

    with op.batch_alter_table("alerts", recreate="always") as batch:
        batch.add_column(
            sa.Column("kind", sa.String(length=10), nullable=False, server_default="alert")
        )
        batch.add_column(sa.Column("suppressed", sa.Boolean(), nullable=False, server_default="0"))
        batch.add_column(
            sa.Column("toast_state", sa.String(length=8), nullable=False, server_default="none")
        )
        batch.add_column(
            sa.Column("mail_state", sa.String(length=8), nullable=False, server_default="none")
        )
        batch.add_column(sa.Column("toast_at", sa.DateTime(), nullable=True))
        batch.add_column(sa.Column("mail_at", sa.DateTime(), nullable=True))
        batch.add_column(
            sa.Column("mail_attempts", sa.Integer(), nullable=False, server_default="0")
        )
        batch.add_column(sa.Column("mail_next_at", sa.DateTime(), nullable=True))
        batch.add_column(sa.Column("mail_error", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("claimed_at", sa.DateTime(), nullable=True))
        batch.create_check_constraint(op.f("ck_alerts_kind"), "kind IN ('alert', 'recovery')")
        batch.create_check_constraint(
            op.f("ck_alerts_toast_state"), "toast_state IN ('none', 'pending', 'sent', 'failed')"
        )
        batch.create_check_constraint(
            op.f("ck_alerts_mail_state"), "mail_state IN ('none', 'pending', 'sent', 'failed')"
        )

    kinds = _eval_kinds()
    if NEW_KIND not in kinds:
        kinds.append(NEW_KIND)
    _set_eval_kinds(kinds, 12)


def downgrade() -> None:
    op.execute(sa.text("DELETE FROM eval_runs WHERE kind = 'stability'"))
    kinds = [kind for kind in _eval_kinds() if kind != NEW_KIND]
    _set_eval_kinds(kinds, 8 if all(len(kind) <= 8 for kind in kinds) else 12)

    with op.batch_alter_table("alerts", recreate="always") as batch:
        batch.drop_constraint(op.f("ck_alerts_mail_state"), type_="check")
        batch.drop_constraint(op.f("ck_alerts_toast_state"), type_="check")
        batch.drop_constraint(op.f("ck_alerts_kind"), type_="check")
        for column in (
            "claimed_at",
            "mail_error",
            "mail_next_at",
            "mail_attempts",
            "mail_at",
            "toast_at",
            "mail_state",
            "toast_state",
            "suppressed",
            "kind",
        ):
            batch.drop_column(column)

    op.drop_index("ix_backup_records_status_date", table_name="backup_records")
    op.drop_table("backup_records")
    op.drop_index("ix_health_snapshots_at", table_name="health_snapshots")
    op.drop_table("health_snapshots")
