"""Round 13b: training_plans (the plans synthesised for training samples).

Revision ID: 0011_training_plans
Revises: 0010_bot_turns_state_feedback
Create Date: 2026-10-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011_training_plans"
down_revision: str | None = "0010_bot_turns_state_feedback"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "training_plans",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=8), nullable=False),
        sa.Column("batch_id", sa.String(length=64), nullable=True),
        sa.Column("template_ref", sa.String(length=64), nullable=True),
        sa.Column("model", sa.String(length=64), nullable=True),
        sa.Column("cost_usd", sa.Float(), nullable=True),
        sa.Column("input", sa.LargeBinary(), nullable=False),
        sa.Column("plan", sa.LargeBinary(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "status IN ('pending', 'done', 'refused')", name=op.f("ck_training_plans_status")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_training_plans")),
    )
    op.create_index("ix_training_plans_status", "training_plans", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_training_plans_status", table_name="training_plans")
    op.drop_table("training_plans")
