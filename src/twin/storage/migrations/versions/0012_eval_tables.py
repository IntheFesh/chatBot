"""Round 09b: eval_runs, eval_items (blind tests, memory tests, style reports, gate verdicts).

Revision ID: 0012_eval_tables
Revises: 0011_training_plans
Create Date: 2026-10-09
"""

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op

revision: str = "0012_eval_tables"
down_revision: str | None = "0011_training_plans"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> list[sa.Column[datetime]]:
    return [
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "eval_runs",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("kind", sa.String(length=8), nullable=False),
        sa.Column("status", sa.String(length=10), nullable=False),
        sa.Column("mode", sa.String(length=8), nullable=True),
        sa.Column("milestone", sa.String(length=2), nullable=True),
        sa.Column("verdict", sa.String(length=12), nullable=True),
        sa.Column("backends", sa.JSON(), nullable=False),
        sa.Column("batch_ids", sa.JSON(), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            "kind IN ('blind', 'style', 'memory', 'gate')", name=op.f("ck_eval_runs_kind")
        ),
        sa.CheckConstraint(
            "status IN ('planned', 'running', 'done', 'cancelled', 'failed')",
            name=op.f("ck_eval_runs_status"),
        ),
        sa.CheckConstraint(
            "mode IS NULL OR mode IN ('holdout', 'live')", name=op.f("ck_eval_runs_mode")
        ),
        sa.CheckConstraint(
            "verdict IS NULL OR verdict IN ('passed', 'failed', 'insufficient')",
            name=op.f("ck_eval_runs_verdict"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_eval_runs")),
    )
    op.create_index("ix_eval_runs_kind_created", "eval_runs", ["kind", "created_at"], unique=False)
    op.create_table(
        "eval_items",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("run_id", sa.String(length=26), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("sample_key", sa.String(length=64), nullable=False),
        sa.Column("backend", sa.String(length=10), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=True),
        sa.Column("period", sa.String(length=10), nullable=True),
        sa.Column("length_bin", sa.String(length=8), nullable=True),
        sa.Column("at", sa.DateTime(), nullable=False),
        sa.Column("status", sa.String(length=10), nullable=False),
        sa.Column("left_is_bot", sa.Boolean(), nullable=True),
        sa.Column("outcome", sa.String(length=8), nullable=True),
        sa.Column("auto_outcome", sa.String(length=8), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("judged_at", sa.DateTime(), nullable=True),
        sa.Column("payload", sa.LargeBinary(), nullable=False),
        *_timestamps(),
        sa.CheckConstraint(
            "status IN ('pending', 'generated', 'failed', 'judged', 'skipped')",
            name=op.f("ck_eval_items_status"),
        ),
        sa.CheckConstraint(
            "outcome IS NULL OR outcome IN ('correct', 'partial', 'wrong')",
            name=op.f("ck_eval_items_outcome"),
        ),
        sa.CheckConstraint(
            "auto_outcome IS NULL OR auto_outcome IN ('correct', 'partial', 'wrong')",
            name=op.f("ck_eval_items_auto_outcome"),
        ),
        sa.CheckConstraint("seq >= 0", name=op.f("ck_eval_items_seq")),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["eval_runs.id"],
            name=op.f("fk_eval_items_run_id_eval_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_eval_items")),
        sa.UniqueConstraint("run_id", "seq", name="uq_eval_items_run_seq"),
    )
    op.create_index("ix_eval_items_run_status", "eval_items", ["run_id", "status"], unique=False)
    op.create_index("ix_eval_items_sample_key", "eval_items", ["sample_key"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_eval_items_sample_key", table_name="eval_items")
    op.drop_index("ix_eval_items_run_status", table_name="eval_items")
    op.drop_table("eval_items")
    op.drop_index("ix_eval_runs_kind_created", table_name="eval_runs")
    op.drop_table("eval_runs")
