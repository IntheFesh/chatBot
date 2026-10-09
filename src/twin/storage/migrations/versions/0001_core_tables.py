"""Round 00 tables: settings, jobs, cost_ledger, alerts, channel_state.

Revision ID: 0001
Revises:
Create Date: 2026-10-09
"""

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> list[sa.Column[datetime]]:
    return [
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "settings",
        sa.Column("key", sa.String(length=128), nullable=False),
        sa.Column("value", sa.LargeBinary(), nullable=False),
        sa.Column("history", sa.LargeBinary(), nullable=True),
        *_timestamps(),
        sa.PrimaryKeyConstraint("key", name=op.f("pk_settings")),
    )
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("type", sa.String(length=128), nullable=False),
        sa.Column("payload", sa.LargeBinary(), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("run_after", sa.DateTime(), nullable=False),
        sa.Column("offpeak_only", sa.Boolean(), nullable=False),
        sa.Column("deadline", sa.DateTime(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("batch_id", sa.String(length=64), nullable=True),
        sa.Column("estimated_cost_usd", sa.Float(), nullable=True),
        sa.Column("requires_approval", sa.Boolean(), nullable=False),
        sa.Column("approved_at", sa.DateTime(), nullable=True),
        sa.Column("approved_usd", sa.Float(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("locked_by", sa.String(length=64), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'done', 'failed', 'cancelled')",
            name=op.f("ck_jobs_status"),
        ),
        sa.CheckConstraint("attempts >= 0 AND max_attempts >= 1", name=op.f("ck_jobs_attempts")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_jobs")),
    )
    op.create_index("ix_jobs_claim", "jobs", ["status", "priority", "run_after"])
    op.create_index("ix_jobs_type", "jobs", ["type"])
    op.create_index("ix_jobs_batch_id", "jobs", ["batch_id"])
    op.create_table(
        "cost_ledger",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("at", sa.DateTime(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=64), nullable=False),
        sa.Column("purpose", sa.String(length=32), nullable=False),
        sa.Column("cache_hit_tokens", sa.Integer(), nullable=False),
        sa.Column("cache_miss_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("peak", sa.Boolean(), nullable=False),
        sa.Column("thinking", sa.Boolean(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=True),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_cost_ledger")),
    )
    op.create_index("ix_cost_ledger_at", "cost_ledger", ["at"])
    op.create_index("ix_cost_ledger_purpose", "cost_ledger", ["purpose"])
    op.create_table(
        "alerts",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("detail", sa.LargeBinary(), nullable=True),
        sa.Column("dedup_key", sa.String(length=128), nullable=True),
        sa.Column("notified_at", sa.DateTime(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            "severity IN ('info', 'warning', 'critical')", name=op.f("ck_alerts_severity")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_alerts")),
    )
    op.create_index("ix_alerts_category", "alerts", ["category", "created_at"])
    op.create_index("ix_alerts_dedup_key", "alerts", ["dedup_key"])
    op.create_table(
        "channel_state",
        sa.Column("key", sa.String(length=128), nullable=False),
        sa.Column("value", sa.LargeBinary(), nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("key", name=op.f("pk_channel_state")),
    )


def downgrade() -> None:
    op.drop_table("channel_state")
    op.drop_index("ix_alerts_dedup_key", table_name="alerts")
    op.drop_index("ix_alerts_category", table_name="alerts")
    op.drop_table("alerts")
    op.drop_index("ix_cost_ledger_purpose", table_name="cost_ledger")
    op.drop_index("ix_cost_ledger_at", table_name="cost_ledger")
    op.drop_table("cost_ledger")
    op.drop_index("ix_jobs_batch_id", table_name="jobs")
    op.drop_index("ix_jobs_type", table_name="jobs")
    op.drop_index("ix_jobs_claim", table_name="jobs")
    op.drop_table("jobs")
    op.drop_table("settings")
