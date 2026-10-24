"""Round 09: bot_turns, conversation_state, feedback.

Revision ID: 0010_bot_turns_state_feedback
Revises: 0008_daily_plans_timezone
Create Date: 2026-10-09
"""

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op

revision: str = "0010_bot_turns_state_feedback"
down_revision: str | None = "0009_training_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> list[sa.Column[datetime]]:
    return [
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "bot_turns",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("at", sa.DateTime(), nullable=False),
        sa.Column("direction", sa.String(length=3), nullable=False),
        sa.Column("kind", sa.String(length=10), nullable=False),
        sa.Column("text", sa.LargeBinary(), nullable=False),
        sa.Column("media", sa.LargeBinary(), nullable=True),
        sa.Column("sticker_md5", sa.String(length=32), nullable=True),
        sa.Column("is_command", sa.Boolean(), nullable=False),
        sa.Column("external_id", sa.String(length=128), nullable=True),
        sa.Column("reply_id", sa.String(length=26), nullable=True),
        sa.Column("bubble_index", sa.Integer(), nullable=True),
        sa.Column("backend", sa.String(length=10), nullable=True),
        sa.Column("thinking", sa.Boolean(), nullable=True),
        sa.Column("plan", sa.LargeBinary(), nullable=True),
        sa.Column("cost_usd", sa.Float(), nullable=True),
        sa.Column("timings", sa.LargeBinary(), nullable=True),
        sa.Column("actions", sa.LargeBinary(), nullable=True),
        sa.Column("rejected_at", sa.DateTime(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint("direction IN ('in', 'out')", name=op.f("ck_bot_turns_direction")),
        sa.CheckConstraint(
            "kind IN ('text', 'image', 'voice', 'video', 'file', 'sticker', 'unknown', 'no_reply')",
            name=op.f("ck_bot_turns_kind"),
        ),
        sa.CheckConstraint(
            "backend IS NULL OR backend IN ('deepseek', 'style', 'hybrid', 'fallback', "
            "'command', 'safety')",
            name=op.f("ck_bot_turns_backend"),
        ),
        sa.CheckConstraint(
            "bubble_index IS NULL OR bubble_index >= 0", name=op.f("ck_bot_turns_bubble_index")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_bot_turns")),
    )
    op.create_index("ix_bot_turns_at", "bot_turns", ["at"], unique=False)
    op.create_index("ix_bot_turns_direction_at", "bot_turns", ["direction", "at"], unique=False)
    op.create_index("ix_bot_turns_reply_id", "bot_turns", ["reply_id"], unique=False)
    op.create_index(
        "uq_bot_turns_external",
        "bot_turns",
        ["direction", "external_id"],
        unique=True,
        sqlite_where=sa.text("external_id IS NOT NULL"),
    )

    op.create_table(
        "conversation_state",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("state", sa.String(length=10), nullable=False),
        sa.Column("state_since", sa.DateTime(), nullable=False),
        sa.Column("round_id", sa.String(length=26), nullable=True),
        sa.Column("pending", sa.LargeBinary(), nullable=True),
        sa.Column("sent", sa.LargeBinary(), nullable=True),
        sa.Column("data", sa.LargeBinary(), nullable=True),
        sa.Column("planned_send_at", sa.DateTime(), nullable=True),
        sa.Column("window_start_at", sa.DateTime(), nullable=True),
        sa.Column("last_inbound_at", sa.DateTime(), nullable=True),
        sa.Column("last_outbound_at", sa.DateTime(), nullable=True),
        sa.Column("extracted_through", sa.DateTime(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            "state IN ('IDLE', 'COLLECTING', 'DECIDING', 'GENERATING', 'SENDING')",
            name=op.f("ck_conversation_state_state"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conversation_state")),
    )

    op.create_table(
        "feedback",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("type", sa.String(length=10), nullable=False),
        sa.Column("reply_id", sa.String(length=26), nullable=False),
        sa.Column("bot_turn_id", sa.String(length=26), nullable=True),
        sa.Column("correction", sa.LargeBinary(), nullable=True),
        sa.Column("processed_at", sa.DateTime(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint("type IN ('redo', 'not_like')", name=op.f("ck_feedback_type")),
        sa.ForeignKeyConstraint(
            ["bot_turn_id"],
            ["bot_turns.id"],
            name=op.f("fk_feedback_bot_turn_id_bot_turns"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_feedback")),
    )
    op.create_index("ix_feedback_reply_id", "feedback", ["reply_id"], unique=False)
    op.create_index(
        "ix_feedback_type_processed", "feedback", ["type", "processed_at"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_feedback_type_processed", table_name="feedback")
    op.drop_index("ix_feedback_reply_id", table_name="feedback")
    op.drop_table("feedback")
    op.drop_table("conversation_state")
    op.drop_index("uq_bot_turns_external", table_name="bot_turns")
    op.drop_index("ix_bot_turns_reply_id", table_name="bot_turns")
    op.drop_index("ix_bot_turns_direction_at", table_name="bot_turns")
    op.drop_index("ix_bot_turns_at", table_name="bot_turns")
    op.drop_table("bot_turns")
