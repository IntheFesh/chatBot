"""Tables of the style model training (round 13; R-STO-006, R-TRN-010, R-SRV-001, R-PRIV-003).

``dataset_versions``
    one row per exported training set: where it lies (a directory below the data directory),
    the cut-off that separates training from testing, the versions of the persona card, the
    profile and the prompt template it was rendered with, the sizes of the splits and the hash of
    every file.  No message text is stored: the data lives in the export directory, which is
    desensitised before it is written.
``training_runs``
    one row per run on a rented machine: profile, dataset version, hyperparameters, how each step
    went (``steps``: started, finished, exit code), the GPU, the peak memory, the best validation
    loss, the files that came back with their sha256, and ``cleaned_at`` - the moment the cleanup
    script finished on the instance.  A run whose data was never cleaned up is reported as long
    as ``cleaned_at`` is empty (R-PRIV-003).
``model_registry``
    one row per file of a trained model that can be used (a GGUF per quantisation, and the LoRA
    adapter for the remote server), with the sha256 and the versions the model was trained with.
    Serving a model renders its prompts with exactly these versions (R-SRV-001).  ``enabled``,
    ``active`` and ``gate_passed`` are set by round 14.

Nothing here is encrypted: the rows hold names, numbers, hashes and versions - no chat content.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from twin.storage.models import Base, TimestampMixin
from twin.storage.types import UTCDateTime

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
MODEL_KINDS = ("gguf", "adapter")


def _in_list(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class DatasetVersion(TimestampMixin, Base):
    """One export of the training set (R-TRN-002, R-TRN-006)."""

    __tablename__ = "dataset_versions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    scope: Mapped[str] = mapped_column(String(16), nullable=False)
    directory: Mapped[str] = mapped_column(String(512), nullable=False)
    holdout_cutoff: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    range_from: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    range_to: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    train_count: Mapped[int] = mapped_column(Integer, nullable=False)
    val_count: Mapped[int] = mapped_column(Integer, nullable=False)
    test_count: Mapped[int] = mapped_column(Integer, nullable=False)
    dpo_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    plan_ratio: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    persona_version: Mapped[str] = mapped_column(String(128), nullable=False)
    profile_version: Mapped[str] = mapped_column(String(128), nullable=False)
    template_version: Mapped[str] = mapped_column(String(128), nullable=False)
    files: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False)
    stats: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    __table_args__ = (
        CheckConstraint("scope IN ('pre_holdout', 'live')", name="scope"),
        CheckConstraint(
            "train_count >= 0 AND val_count >= 0 AND test_count >= 0 AND dpo_count >= 0",
            name="counts",
        ),
    )


class TrainingRun(TimestampMixin, Base):
    """One run on a rented machine, from upload to cleanup (R-TRN-010)."""

    __tablename__ = "training_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    profile: Mapped[str] = mapped_column(String(32), nullable=False)
    dataset_version: Mapped[str] = mapped_column(
        String(64), ForeignKey("dataset_versions.id"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="created")
    bundle_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    bundle_size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    hyperparameters: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    steps: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    gpu_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    peak_vram_mib: Mapped[int | None] = mapped_column(Integer, nullable=True)
    best_val_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_checkpoint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    artifacts: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    cleaned_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint(_in_list("status", RUN_STATUSES), name="status"),
        Index("ix_training_runs_status", "status"),
        Index("ix_training_runs_dataset_version", "dataset_version"),
    )


class ModelRegistryEntry(TimestampMixin, Base):
    """A trained model file that the application can use (R-SRV-001)."""

    __tablename__ = "model_registry"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(8), nullable=False)
    profile: Mapped[str] = mapped_column(String(32), nullable=False)
    base_model: Mapped[str] = mapped_column(String(128), nullable=False)
    quant: Mapped[str] = mapped_column(String(16), nullable=False)
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    template_version: Mapped[str] = mapped_column(String(128), nullable=False)
    persona_version: Mapped[str] = mapped_column(String(128), nullable=False)
    profile_version: Mapped[str] = mapped_column(String(128), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False)
    eval: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    gate_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    __table_args__ = (
        CheckConstraint(_in_list("kind", MODEL_KINDS), name="kind"),
        UniqueConstraint("run_id", "quant", name="uq_model_registry_run_quant"),
        Index("ix_model_registry_dataset_version", "dataset_version"),
    )
