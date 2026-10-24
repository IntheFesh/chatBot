"""Round 06: persona_cards, prompt_templates and the tag columns of stickers.

Revision ID: 0006_persona_sticker_tags
Revises: 0005_example_windows
Create Date: 2026-10-09
"""

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op

revision: str = "0006_persona_sticker_tags"
down_revision: str | None = "0005_example_windows"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

STICKER_COLUMNS = (
    "vision_tags",
    "context_tags",
    "manual_tags",
    "tags",
    "description",
    "use_cases",
    "context_note",
    "tag_source",
    "origin",
    "disabled",
    "tagged_at",
    "context_tagged_at",
    "context_cutoff_at",
    "context_uses",
    "desc_vector_id",
    "desc_encoding",
)


def _timestamps() -> list[sa.Column[datetime]]:
    return [
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "persona_cards",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("scope", sa.String(length=16), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("parent_id", sa.String(length=26), nullable=True),
        sa.Column("reason", sa.String(length=32), nullable=False),
        sa.Column("profile_version_id", sa.String(length=26), nullable=True),
        sa.Column("template_version", sa.String(length=128), nullable=True),
        sa.Column("described_her_messages", sa.Integer(), nullable=True),
        sa.Column("described_at", sa.DateTime(), nullable=True),
        sa.Column("content", sa.LargeBinary(), nullable=False),
        sa.Column("provenance", sa.LargeBinary(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint("scope IN ('live', 'pre_holdout')", name=op.f("ck_persona_cards_scope")),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            ["persona_cards.id"],
            name=op.f("fk_persona_cards_parent_id_persona_cards"),
        ),
        sa.ForeignKeyConstraint(
            ["profile_version_id"],
            ["profile_versions.id"],
            name=op.f("fk_persona_cards_profile_version_id_profile_versions"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_persona_cards")),
        sa.UniqueConstraint("scope", "number", name="uq_persona_cards_scope_number"),
    )
    op.create_index(
        "ix_persona_cards_scope_created", "persona_cards", ["scope", "created_at"], unique=False
    )
    op.create_table(
        "prompt_templates",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("content", sa.LargeBinary(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        *_timestamps(),
        sa.CheckConstraint("source IN ('file', 'edited')", name=op.f("ck_prompt_templates_source")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_prompt_templates")),
        sa.UniqueConstraint("name", "version", name="uq_prompt_templates_name_version"),
    )

    for name in ("vision_tags", "context_tags", "manual_tags", "tags"):
        op.add_column("stickers", sa.Column(name, sa.LargeBinary(), nullable=True))
    for name in ("description", "use_cases", "context_note"):
        op.add_column("stickers", sa.Column(name, sa.LargeBinary(), nullable=True))
    op.add_column("stickers", sa.Column("tag_source", sa.String(length=16), nullable=True))
    op.add_column(
        "stickers",
        sa.Column("origin", sa.String(length=16), nullable=False, server_default="import"),
    )
    op.add_column(
        "stickers",
        sa.Column("disabled", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    for name in ("tagged_at", "context_tagged_at", "context_cutoff_at"):
        op.add_column("stickers", sa.Column(name, sa.DateTime(), nullable=True))
    op.add_column(
        "stickers",
        sa.Column("context_uses", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("stickers", sa.Column("desc_vector_id", sa.String(length=64), nullable=True))
    op.add_column("stickers", sa.Column("desc_encoding", sa.String(length=200), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("stickers") as batch:
        for name in reversed(STICKER_COLUMNS):
            batch.drop_column(name)
    op.drop_table("prompt_templates")
    op.drop_index("ix_persona_cards_scope_created", table_name="persona_cards")
    op.drop_table("persona_cards")
