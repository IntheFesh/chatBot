"""Round 13: dataset_versions, training_runs, model_registry.

Revision ID: 0008_training_tables
Revises: 0007_memory_tables
Create Date: 2026-10-09
"""

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op

revision: str = "0008_training_tables"
down_revision: str | None = "0007_memory_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUN_STATUSES = (
    "created",
    "uploaded",
    "set_up",
    "trained",
    "dpo_done",
    "evaluated",
    "exported",
    "downloaded",
    "cleaned",
    "failed",
)


def _timestamps() -> list[sa.Column[datetime]]:
    return [
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "dataset_versions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("scope", sa.String(length=16), nullable=False),
        sa.Column("directory", sa.String(length=512), nullable=False),
        sa.Column("holdout_cutoff", sa.DateTime(), nullable=False),
        sa.Column("range_from", sa.DateTime(), nullable=True),
        sa.Column("range_to", sa.DateTime(), nullable=True),
        sa.Column("train_count", sa.Integer(), nullable=False),
        sa.Column("val_count", sa.Integer(), nullable=False),
        sa.Column("test_count", sa.Integer(), nullable=False),
        sa.Column("dpo_count", sa.Integer(), nullable=False),
        sa.Column("plan_ratio", sa.Float(), nullable=False),
        sa.Column("persona_version", sa.String(length=128), nullable=False),
        sa.Column("profile_version", sa.String(length=128), nullable=False),
        sa.Column("template_version", sa.String(length=128), nullable=False),
        sa.Column("files", sa.JSON(), nullable=False),
        sa.Column("stats", sa.JSON(), nullable=False),
        *_timestamps(),
        sa.CheckConstraint(
            "scope IN ('pre_holdout', 'live')", name=op.f("ck_dataset_versions_scope")
        ),
        sa.CheckConstraint(
            "train_count >= 0 AND val_count >= 0 AND test_count >= 0 AND dpo_count >= 0",
            name=op.f("ck_dataset_versions_counts"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_dataset_versions")),
    )
    statuses = ", ".join(repr(status) for status in RUN_STATUSES)
    op.create_table(
        "training_runs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("profile", sa.String(length=32), nullable=False),
        sa.Column("dataset_version", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("bundle_sha256", sa.String(length=64), nullable=True),
        sa.Column("bundle_size", sa.BigInteger(), nullable=True),
        sa.Column("hyperparameters", sa.JSON(), nullable=False),
        sa.Column("steps", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("gpu_model", sa.String(length=128), nullable=True),
        sa.Column("peak_vram_mib", sa.Integer(), nullable=True),
        sa.Column("best_val_loss", sa.Float(), nullable=True),
        sa.Column("best_checkpoint", sa.String(length=64), nullable=True),
        sa.Column("artifacts", sa.JSON(), nullable=False),
        sa.Column("cleaned_at", sa.DateTime(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(f"status IN ({statuses})", name=op.f("ck_training_runs_status")),
        sa.ForeignKeyConstraint(
            ["dataset_version"],
            ["dataset_versions.id"],
            name=op.f("fk_training_runs_dataset_version_dataset_versions"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_training_runs")),
    )
    op.create_index("ix_training_runs_status", "training_runs", ["status"])
    op.create_index("ix_training_runs_dataset_version", "training_runs", ["dataset_version"])
    op.create_table(
        "model_registry",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=8), nullable=False),
        sa.Column("profile", sa.String(length=32), nullable=False),
        sa.Column("base_model", sa.String(length=128), nullable=False),
        sa.Column("quant", sa.String(length=16), nullable=False),
        sa.Column("path", sa.String(length=512), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size", sa.BigInteger(), nullable=False),
        sa.Column("template_version", sa.String(length=128), nullable=False),
        sa.Column("persona_version", sa.String(length=128), nullable=False),
        sa.Column("profile_version", sa.String(length=128), nullable=False),
        sa.Column("dataset_version", sa.String(length=64), nullable=False),
        sa.Column("eval", sa.JSON(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("gate_passed", sa.Boolean(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint("kind IN ('gguf', 'adapter')", name=op.f("ck_model_registry_kind")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_registry")),
        sa.UniqueConstraint("run_id", "quant", name="uq_model_registry_run_quant"),
    )
    op.create_index("ix_model_registry_dataset_version", "model_registry", ["dataset_version"])


def downgrade() -> None:
    op.drop_index("ix_model_registry_dataset_version", table_name="model_registry")
    op.drop_table("model_registry")
    op.drop_index("ix_training_runs_dataset_version", table_name="training_runs")
    op.drop_index("ix_training_runs_status", table_name="training_runs")
    op.drop_table("training_runs")
    op.drop_table("dataset_versions")
