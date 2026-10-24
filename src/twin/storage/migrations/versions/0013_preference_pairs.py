"""Round 11: preference_pairs (what ``/不像 <正确说法>`` leaves for DPO).

Revision ID: 0013_preference_pairs
Revises: 0012_eval_tables
Create Date: 2026-10-09
"""

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op

revision: str = "0013_preference_pairs"
down_revision: str | None = "0012_eval_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> list[sa.Column[datetime]]:
    return [
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "preference_pairs",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("feedback_id", sa.String(length=26), nullable=True),
        sa.Column("reply_id", sa.String(length=26), nullable=False),
        sa.Column("prompt_sample", sa.LargeBinary(), nullable=False),
        sa.Column("chosen", sa.LargeBinary(), nullable=False),
        sa.Column("rejected", sa.LargeBinary(), nullable=False),
        sa.Column("source", sa.String(length=20), nullable=False),
        sa.Column("template_version", sa.String(length=64), nullable=False),
        sa.Column("persona_version", sa.String(length=64), nullable=False),
        *_timestamps(),
        sa.CheckConstraint(
            "source IN ('user_correction')", name=op.f("ck_preference_pairs_source")
        ),
        sa.ForeignKeyConstraint(
            ["feedback_id"],
            ["feedback.id"],
            name=op.f("fk_preference_pairs_feedback_id_feedback"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_preference_pairs")),
    )
    op.create_index("ix_preference_pairs_reply_id", "preference_pairs", ["reply_id"], unique=False)
    op.create_index(
        "ix_preference_pairs_created_at", "preference_pairs", ["created_at"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_preference_pairs_created_at", table_name="preference_pairs")
    op.drop_index("ix_preference_pairs_reply_id", table_name="preference_pairs")
    op.drop_table("preference_pairs")
