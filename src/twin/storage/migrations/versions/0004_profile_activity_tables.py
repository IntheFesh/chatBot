"""Round 04 tables: profile_versions, activity_models, routine_overrides.

Revision ID: 0004_profile_activity_tables
Revises: 0003_import_tables
Create Date: 2026-10-09
"""

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op

revision: str = "0004_profile_activity_tables"
down_revision: str | None = "0003_import_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCOPES = "'live', 'pre_holdout'"


def _timestamps() -> list[sa.Column[datetime]]:
    return [
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "profile_versions",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("scope", sa.String(length=16), nullable=False),
        sa.Column("parent_id", sa.String(length=26), nullable=True),
        sa.Column("reason", sa.String(length=32), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("her_messages", sa.Integer(), nullable=False),
        sa.Column("data_range", sa.LargeBinary(), nullable=False),
        sa.Column("metrics", sa.LargeBinary(), nullable=False),
        sa.Column("phrases", sa.LargeBinary(), nullable=True),
        sa.Column("summary_rules", sa.LargeBinary(), nullable=False),
        sa.Column("diff", sa.LargeBinary(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(f"scope IN ({SCOPES})", name=op.f("ck_profile_versions_scope")),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            ["profile_versions.id"],
            name=op.f("fk_profile_versions_parent_id_profile_versions"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_profile_versions")),
    )
    op.create_index(
        "ix_profile_versions_scope_created",
        "profile_versions",
        ["scope", "created_at"],
        unique=False,
    )
    op.create_table(
        "activity_models",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("scope", sa.String(length=16), nullable=False),
        sa.Column("parent_id", sa.String(length=26), nullable=True),
        sa.Column("profile_version_id", sa.String(length=26), nullable=True),
        sa.Column("reason", sa.String(length=32), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("her_messages", sa.Integer(), nullable=False),
        sa.Column("data_range", sa.LargeBinary(), nullable=False),
        sa.Column("model", sa.LargeBinary(), nullable=False),
        sa.Column("diff", sa.LargeBinary(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(f"scope IN ({SCOPES})", name=op.f("ck_activity_models_scope")),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            ["activity_models.id"],
            name=op.f("fk_activity_models_parent_id_activity_models"),
        ),
        sa.ForeignKeyConstraint(
            ["profile_version_id"],
            ["profile_versions.id"],
            name=op.f("fk_activity_models_profile_version_id_profile_versions"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_activity_models")),
    )
    op.create_index(
        "ix_activity_models_scope_created",
        "activity_models",
        ["scope", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_activity_models_profile_version_id",
        "activity_models",
        ["profile_version_id"],
        unique=False,
    )
    op.create_table(
        "routine_overrides",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("params", sa.LargeBinary(), nullable=False),
        sa.Column("note", sa.LargeBinary(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        *_timestamps(),
        sa.CheckConstraint(
            "kind IN ('sleep', 'busy', 'holiday')", name=op.f("ck_routine_overrides_kind")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_routine_overrides")),
    )
    op.create_index("ix_routine_overrides_kind", "routine_overrides", ["kind"], unique=False)


def downgrade() -> None:
    op.drop_table("routine_overrides")
    op.drop_table("activity_models")
    op.drop_table("profile_versions")
