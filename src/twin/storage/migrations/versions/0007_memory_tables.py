"""Round 07: facts, daily_summaries, lifeline_events, followups, memory_replay_days.

Revision ID: 0007_memory_tables
Revises: 0006_persona_sticker_tags
Create Date: 2026-10-09
"""

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op

revision: str = "0007_memory_tables"
down_revision: str | None = "0006_persona_sticker_tags"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> list[sa.Column[datetime]]:
    return [
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "facts",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("rev", sa.Integer(), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("subject", sa.String(length=8), nullable=False),
        sa.Column("category", sa.String(length=16), nullable=False),
        sa.Column("text", sa.LargeBinary(), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=8), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("importance", sa.Integer(), nullable=False),
        sa.Column("known_at", sa.DateTime(), nullable=False),
        sa.Column("valid_from", sa.DateTime(), nullable=True),
        sa.Column("valid_to", sa.DateTime(), nullable=True),
        sa.Column("event_date", sa.Date(), nullable=True),
        sa.Column("recurrence", sa.String(length=8), nullable=False),
        sa.Column("superseded_by", sa.String(length=26), nullable=True),
        sa.Column("superseded_at", sa.DateTime(), nullable=True),
        sa.Column("rejected_by", sa.String(length=26), nullable=True),
        sa.Column("evidence", sa.LargeBinary(), nullable=True),
        sa.Column("embedding_id", sa.String(length=64), nullable=True),
        sa.Column("embed_version", sa.String(length=200), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            "subject IN ('her', 'user', 'both', 'other')", name=op.f("ck_facts_subject")
        ),
        sa.CheckConstraint(
            "category IN ('life', 'preference', 'plan', 'anniversary', 'nickname', "
            "'relation', 'work_study', 'other')",
            name=op.f("ck_facts_category"),
        ),
        sa.CheckConstraint(
            "source IN ('real_record', 'user_said', 'bot_invented', 'user_command')",
            name=op.f("ck_facts_source"),
        ),
        sa.CheckConstraint("status IN ('active', 'rejected')", name=op.f("ck_facts_status")),
        sa.CheckConstraint(
            "recurrence IN ('none', 'yearly', 'monthly')", name=op.f("ck_facts_recurrence")
        ),
        sa.CheckConstraint("importance >= 1 AND importance <= 5", name=op.f("ck_facts_importance")),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name=op.f("ck_facts_confidence")),
        sa.ForeignKeyConstraint(
            ["superseded_by"],
            ["facts.id"],
            name=op.f("fk_facts_superseded_by_facts"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["rejected_by"],
            ["facts.id"],
            name=op.f("fk_facts_rejected_by_facts"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_facts")),
        sa.UniqueConstraint("number", name="uq_facts_number"),
    )
    op.create_index("ix_facts_known_at", "facts", ["known_at"], unique=False)
    op.create_index("ix_facts_status_source", "facts", ["status", "source"], unique=False)
    op.create_index("ix_facts_event_date", "facts", ["event_date"], unique=False)

    op.create_table(
        "daily_summaries",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("rev", sa.Integer(), nullable=False),
        sa.Column("scope", sa.String(length=8), nullable=False),
        sa.Column("local_date", sa.Date(), nullable=False),
        sa.Column("timezone", sa.String(length=64), nullable=False),
        sa.Column("utc_start", sa.DateTime(), nullable=False),
        sa.Column("utc_end", sa.DateTime(), nullable=False),
        sa.Column("text", sa.LargeBinary(), nullable=False),
        sa.Column("embedding_id", sa.String(length=64), nullable=True),
        sa.Column("embed_version", sa.String(length=200), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("is_current", sa.Boolean(), nullable=False),
        sa.Column("input_hash", sa.String(length=40), nullable=True),
        sa.Column("template_version", sa.String(length=64), nullable=True),
        *_timestamps(),
        sa.CheckConstraint("scope IN ('real', 'bot')", name=op.f("ck_daily_summaries_scope")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_daily_summaries")),
        sa.UniqueConstraint(
            "scope", "local_date", "version", name="uq_daily_summaries_day_version"
        ),
    )
    op.create_index(
        "ix_daily_summaries_scope_date_current",
        "daily_summaries",
        ["scope", "local_date", "is_current"],
        unique=False,
    )

    op.create_table(
        "lifeline_events",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("rev", sa.Integer(), nullable=False),
        sa.Column("local_date", sa.Date(), nullable=False),
        sa.Column("timezone", sa.String(length=64), nullable=False),
        sa.Column("start_local", sa.String(length=5), nullable=True),
        sa.Column("end_local", sa.String(length=5), nullable=True),
        sa.Column("activity", sa.LargeBinary(), nullable=False),
        sa.Column("place", sa.LargeBinary(), nullable=True),
        sa.Column("mood", sa.LargeBinary(), nullable=True),
        sa.Column("detail", sa.LargeBinary(), nullable=True),
        sa.Column("source", sa.String(length=10), nullable=False),
        sa.Column("status", sa.String(length=12), nullable=False),
        sa.Column("consistency_checked_at", sa.DateTime(), nullable=True),
        sa.Column("invalidated_at", sa.DateTime(), nullable=True),
        sa.Column("invalidated_by", sa.String(length=26), nullable=True),
        sa.Column("fact_id", sa.String(length=26), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            "source IN ('plan', 'improvised')", name=op.f("ck_lifeline_events_source")
        ),
        sa.CheckConstraint(
            "status IN ('active', 'invalidated')", name=op.f("ck_lifeline_events_status")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_lifeline_events")),
    )
    op.create_index(
        "ix_lifeline_events_local_date", "lifeline_events", ["local_date"], unique=False
    )
    op.create_index("ix_lifeline_events_fact_id", "lifeline_events", ["fact_id"], unique=False)

    op.create_table(
        "followups",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("rev", sa.Integer(), nullable=False),
        sa.Column("text", sa.LargeBinary(), nullable=False),
        sa.Column("due_at_utc", sa.DateTime(), nullable=False),
        sa.Column("window_minutes", sa.Integer(), nullable=False),
        sa.Column("source_turn_id", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=10), nullable=False),
        sa.Column("closed_at", sa.DateTime(), nullable=True),
        sa.Column("close_reason", sa.String(length=32), nullable=True),
        sa.Column("origin", sa.String(length=12), nullable=False),
        sa.Column("evidence", sa.LargeBinary(), nullable=True),
        sa.Column("fact_id", sa.String(length=26), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            "status IN ('open', 'done', 'cancelled', 'expired')", name=op.f("ck_followups_status")
        ),
        sa.CheckConstraint(
            "origin IN ('real_record', 'bot_session', 'user_command')",
            name=op.f("ck_followups_origin"),
        ),
        sa.CheckConstraint("window_minutes >= 0", name=op.f("ck_followups_window_minutes")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_followups")),
    )
    op.create_index("ix_followups_status_due", "followups", ["status", "due_at_utc"], unique=False)
    op.create_index("ix_followups_fact_id", "followups", ["fact_id"], unique=False)

    op.create_table(
        "memory_replay_days",
        sa.Column("local_date", sa.Date(), nullable=False),
        sa.Column("input_hash", sa.String(length=40), nullable=False),
        sa.Column("lines", sa.Integer(), nullable=False),
        sa.Column("facts_added", sa.Integer(), nullable=False),
        sa.Column("followups_added", sa.Integer(), nullable=False),
        sa.Column("summary_version", sa.Integer(), nullable=True),
        sa.Column("batch_id", sa.String(length=64), nullable=True),
        sa.Column("replayed_at", sa.DateTime(), nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("local_date", name=op.f("pk_memory_replay_days")),
    )


def downgrade() -> None:
    op.drop_table("memory_replay_days")
    op.drop_index("ix_followups_fact_id", table_name="followups")
    op.drop_index("ix_followups_status_due", table_name="followups")
    op.drop_table("followups")
    op.drop_index("ix_lifeline_events_fact_id", table_name="lifeline_events")
    op.drop_index("ix_lifeline_events_local_date", table_name="lifeline_events")
    op.drop_table("lifeline_events")
    op.drop_index("ix_daily_summaries_scope_date_current", table_name="daily_summaries")
    op.drop_table("daily_summaries")
    op.drop_index("ix_facts_event_date", table_name="facts")
    op.drop_index("ix_facts_status_source", table_name="facts")
    op.drop_index("ix_facts_known_at", table_name="facts")
    op.drop_table("facts")
