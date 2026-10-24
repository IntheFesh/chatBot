"""Round 08: daily_plans, timezone_history.

Revision ID: 0008_daily_plans_timezone
Revises: 0007_memory_tables
Create Date: 2026-10-09
"""

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op

revision: str = "0008_daily_plans_timezone"
down_revision: str | None = "0007_memory_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> list[sa.Column[datetime]]:
    return [
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "daily_plans",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("local_date", sa.Date(), nullable=False),
        sa.Column("timezone", sa.String(length=64), nullable=False),
        sa.Column("day_type", sa.String(length=8), nullable=False),
        sa.Column("plan", sa.LargeBinary(), nullable=False),
        sa.Column("seed", sa.BigInteger(), nullable=False),
        sa.Column("effective_from", sa.DateTime(), nullable=False),
        sa.Column("ends_at", sa.DateTime(), nullable=False),
        sa.Column("wake_at", sa.DateTime(), nullable=True),
        sa.Column("reason", sa.String(length=24), nullable=False),
        sa.Column("inputs_hash", sa.String(length=40), nullable=False),
        sa.Column("superseded_by", sa.String(length=26), nullable=True),
        sa.Column("superseded_at", sa.DateTime(), nullable=True),
        sa.Column("lifeline_job_id", sa.String(length=26), nullable=True),
        sa.Column("lifeline_done_at", sa.DateTime(), nullable=True),
        sa.Column("summary_queued_at", sa.DateTime(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            "day_type IN ('workday', 'weekend', 'holiday')", name=op.f("ck_daily_plans_day_type")
        ),
        sa.CheckConstraint("ends_at > effective_from", name=op.f("ck_daily_plans_interval")),
        sa.ForeignKeyConstraint(
            ["superseded_by"],
            ["daily_plans.id"],
            name=op.f("fk_daily_plans_superseded_by_daily_plans"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_daily_plans")),
    )
    op.create_index(
        "ix_daily_plans_date_zone", "daily_plans", ["local_date", "timezone"], unique=False
    )
    op.create_index(
        "ix_daily_plans_interval", "daily_plans", ["effective_from", "ends_at"], unique=False
    )

    op.create_table(
        "timezone_history",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("changed_at", sa.DateTime(), nullable=False),
        sa.Column("from_timezone", sa.String(length=64), nullable=False),
        sa.Column("to_timezone", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=8), nullable=False),
        sa.Column("plan_id", sa.String(length=26), nullable=True),
        sa.Column("last_greeting_at", sa.DateTime(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            "source IN ('cli', 'command', 'app')", name=op.f("ck_timezone_history_source")
        ),
        sa.CheckConstraint(
            "from_timezone <> to_timezone", name=op.f("ck_timezone_history_differs")
        ),
        sa.ForeignKeyConstraint(
            ["plan_id"],
            ["daily_plans.id"],
            name=op.f("fk_timezone_history_plan_id_daily_plans"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_timezone_history")),
    )
    op.create_index(
        "ix_timezone_history_changed_at", "timezone_history", ["changed_at"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_timezone_history_changed_at", table_name="timezone_history")
    op.drop_table("timezone_history")
    op.drop_index("ix_daily_plans_interval", table_name="daily_plans")
    op.drop_index("ix_daily_plans_date_zone", table_name="daily_plans")
    op.drop_table("daily_plans")
