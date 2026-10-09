"""Round 05 table: example_windows.

Revision ID: 0005_example_windows
Revises: 0004_profile_activity_tables
Create Date: 2026-10-09
"""

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op

revision: str = "0005_example_windows"
down_revision: str | None = "0004_profile_activity_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> list[sa.Column[datetime]]:
    return [
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "example_windows",
        sa.Column("id", sa.String(length=24), nullable=False),
        sa.Column("conversation_id", sa.String(length=26), nullable=False),
        sa.Column("reply_block_ids", sa.JSON(), nullable=False),
        sa.Column("context_block_ids", sa.JSON(), nullable=False),
        sa.Column("context_turns", sa.Integer(), nullable=False),
        sa.Column("reply_reproducible", sa.Integer(), nullable=False),
        sa.Column("reply_at_utc", sa.DateTime(), nullable=False),
        sa.Column("local_slot", sa.Integer(), nullable=False),
        sa.Column("day_type", sa.String(length=8), nullable=False),
        sa.Column("holdout", sa.Boolean(), nullable=False),
        sa.Column("signature", sa.String(length=40), nullable=False),
        sa.Column("embed_version", sa.String(length=200), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            "local_slot >= 0 AND local_slot < 96", name=op.f("ck_example_windows_local_slot")
        ),
        sa.CheckConstraint(
            "day_type IN ('workday', 'weekend', 'holiday')",
            name=op.f("ck_example_windows_day_type"),
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_example_windows_conversation_id_conversations"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_example_windows")),
    )
    op.create_index(
        "ix_example_windows_reply_at_utc", "example_windows", ["reply_at_utc"], unique=False
    )
    op.create_index(
        "ix_example_windows_holdout_embed_version",
        "example_windows",
        ["holdout", "embed_version"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_table("example_windows")
