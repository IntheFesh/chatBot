"""The plans synthesised for training samples (round 13b; R-TRN-005, R-LLM-014).

``training_plans``
    one row per sample of the training set that is to carry a plan: the prompt inputs the plan is
    written from (sealed: the turns before her reply, the reply itself and the fact lines of the
    memory block - already desensitised, they are what is sent to DeepSeek), the plan DeepSeek
    wrote (sealed: ``intent``, ``facts_to_use``, ``tone``, ``bubble_hint``) and where the row
    stands.  ``pending`` waits for the plan job of its batch, ``done`` has its plan, ``refused``
    could not get one (the model refused the content or never produced valid JSON) and the sample
    is exported without a plan.  The key is the id of the sample (the example window of her reply
    block), so the plan of a sample is found again by every later export; ``input`` also holds the
    hash of the reply the plan was written for, and a plan whose reply has changed is stale.
"""

from __future__ import annotations

from sqlalchemy import CheckConstraint, Float, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from twin.storage.crypto import SealedBlob
from twin.storage.models import Base, TimestampMixin
from twin.storage.types import encrypted_column, sealed_json

PLAN_STATUSES = ("pending", "done", "refused")


class TrainingPlan(TimestampMixin, Base):
    """The plan of one training sample, and the inputs it is written from (R-TRN-005)."""

    __tablename__ = "training_plans"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(8), nullable=False, default="pending")
    batch_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    template_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    input_ct: Mapped[SealedBlob] = encrypted_column("input", json=True)
    input = sealed_json()
    plan_ct: Mapped[SealedBlob | None] = encrypted_column("plan", json=True, nullable=True)
    plan = sealed_json(optional=True)

    __table_args__ = (
        CheckConstraint("status IN ('pending', 'done', 'refused')", name="status"),
        Index("ix_training_plans_status", "status"),
    )
