"""Tables of the evaluation (round 09b; R-STO-006, R-EVAL-001, R-EVAL-003, R-EVAL-009, R-EVAL-010).

``eval_runs``
    one row per evaluation: a blind test (``blind``), a memory test (``memory``), a style
    report that was kept (``style``) or the verdict of a milestone gate (``gate``).  Plain
    columns only - names, counts, numbers and ids, never message text: the backends compared,
    the batch ids of the one-time batches that generate the replies (R-LLM-014), the parameters
    the run was drawn with (seed, size, hold-out cut-off) and the summary numbers.  A gate row
    carries the milestone, the verdict (``passed`` / ``failed`` / ``insufficient``) and its
    evidence (the runs and the numbers it was decided from).
``eval_items``
    one row per judged unit: a (context, backend) pair of a blind test or a question of the
    memory test.  The sample is identified by ``sample_key`` (the id of her reply block, or of the
    fact), so a context that was used once is never drawn again.  Everything that is text - the
    conversation before the reply, her real reply, the bot's reply, the question and the answer -
    is in the sealed ``payload``.

A blind item moves ``pending`` (drawn) -> ``generated`` (the bot's reply is there) -> ``judged``
(``outcome`` is ``correct`` when the user picked her, ``wrong`` when he picked the bot) or
``skipped`` (not counted); ``failed`` is a reply that could not be produced.  A memory item moves
``pending`` -> ``generated`` (question asked, answered, judged by DeepSeek: ``auto_outcome``)
-> ``judged`` (the user confirmed or changed the verdict: ``outcome`` correct / partial / wrong).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from twin.storage.crypto import SealedBlob
from twin.storage.models import Base, TimestampMixin
from twin.storage.types import UTCDateTime, encrypted_column, sealed_json

RUN_KINDS = ("blind", "style", "memory", "gate")
RUN_STATUSES = ("planned", "running", "done", "cancelled", "failed")
RUN_MODES = ("holdout", "live")
VERDICTS = ("passed", "failed", "insufficient")
ITEM_STATUSES = ("pending", "generated", "failed", "judged", "skipped")
ITEM_OUTCOMES = ("correct", "partial", "wrong")


def _in_list(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class EvalRun(TimestampMixin, Base):
    """One evaluation (a blind test, a memory test, a style report or a gate verdict)."""

    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    kind: Mapped[str] = mapped_column(String(8), nullable=False)
    status: Mapped[str] = mapped_column(String(10), nullable=False, default="planned")
    mode: Mapped[str | None] = mapped_column(String(8), nullable=True)
    milestone: Mapped[str | None] = mapped_column(String(2), nullable=True)
    verdict: Mapped[str | None] = mapped_column(String(12), nullable=True)
    backends: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    batch_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    params: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    __table_args__ = (
        CheckConstraint(_in_list("kind", RUN_KINDS), name="kind"),
        CheckConstraint(_in_list("status", RUN_STATUSES), name="status"),
        CheckConstraint("mode IS NULL OR " + _in_list("mode", RUN_MODES), name="mode"),
        CheckConstraint("verdict IS NULL OR " + _in_list("verdict", VERDICTS), name="verdict"),
        Index("ix_eval_runs_kind_created", "kind", "created_at"),
    )


class EvalItem(TimestampMixin, Base):
    """One (context, backend) pair of a blind test, or one question of the memory test."""

    __tablename__ = "eval_items"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(26), ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    sample_key: Mapped[str] = mapped_column(String(64), nullable=False)
    backend: Mapped[str] = mapped_column(String(10), nullable=False)
    source: Mapped[str | None] = mapped_column(String(16), nullable=True)
    period: Mapped[str | None] = mapped_column(String(10), nullable=True)
    length_bin: Mapped[str | None] = mapped_column(String(8), nullable=True)
    at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    status: Mapped[str] = mapped_column(String(10), nullable=False, default="pending")
    left_is_bot: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(8), nullable=True)
    auto_outcome: Mapped[str | None] = mapped_column(String(8), nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    judged_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    payload_ct: Mapped[SealedBlob] = encrypted_column("payload", json=True)
    payload = sealed_json()

    __table_args__ = (
        CheckConstraint(_in_list("status", ITEM_STATUSES), name="status"),
        CheckConstraint("outcome IS NULL OR " + _in_list("outcome", ITEM_OUTCOMES), name="outcome"),
        CheckConstraint(
            "auto_outcome IS NULL OR " + _in_list("auto_outcome", ITEM_OUTCOMES),
            name="auto_outcome",
        ),
        CheckConstraint("seq >= 0", name="seq"),
        UniqueConstraint("run_id", "seq", name="uq_eval_items_run_seq"),
        Index("ix_eval_items_run_status", "run_id", "status"),
        Index("ix_eval_items_sample_key", "sample_key"),
    )
