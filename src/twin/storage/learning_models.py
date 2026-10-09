"""Tables of the learning from the conversation with the bot (round 11; R-STO-006, R-LRN-002).

``preference_pairs``
    what ``/不像 <正确说法>`` leaves behind: the way the user says she would have put it
    (``chosen``) against what the bot said (``rejected``), for the situation it was said in
    (``prompt_sample``).  The sample is **structured** - the system segment and the conversation
    turns, opening with the user and alternating, made by the same functions that make the
    prompt of the style model (``StylePromptBuilder``) - and is never a rendered prompt string:
    LLaMA-Factory puts the chat template around it once, when the pairs are trained on (DPO,
    R-LRN-002).  Three columns hold text and are sealed: ``prompt_sample``, ``chosen`` and
    ``rejected``.  ``template_version`` and ``persona_version`` name what the system segment was
    made with, so the DPO export can render it again with the versions an adapter is locked to.

The table is only ever read by the DPO export (``twin train export-dpo``): ``rejected`` is the
bot's own text and may be a negative example there and nowhere else (R-LRN-004, CLAUDE.md rule 7).
"""

from __future__ import annotations

from sqlalchemy import CheckConstraint, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from twin.storage.crypto import SealedBlob
from twin.storage.models import Base, TimestampMixin
from twin.storage.types import encrypted_column, sealed_json, sealed_text

PAIR_SOURCES = ("user_correction",)


def _in_list(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class PreferencePair(TimestampMixin, Base):
    """One preference pair: the user's wording against the bot's reply (R-LRN-002)."""

    __tablename__ = "preference_pairs"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    feedback_id: Mapped[str | None] = mapped_column(
        String(26), ForeignKey("feedback.id", ondelete="SET NULL"), nullable=True
    )
    reply_id: Mapped[str] = mapped_column(String(26), nullable=False)
    prompt_sample_ct: Mapped[SealedBlob] = encrypted_column("prompt_sample", json=True)
    prompt_sample = sealed_json()
    chosen_ct: Mapped[SealedBlob] = encrypted_column("chosen")
    chosen = sealed_text()
    rejected_ct: Mapped[SealedBlob] = encrypted_column("rejected")
    rejected = sealed_text()
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="user_correction")
    template_version: Mapped[str] = mapped_column(String(64), nullable=False)
    persona_version: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        CheckConstraint(_in_list("source", PAIR_SOURCES), name="source"),
        Index("ix_preference_pairs_reply_id", "reply_id"),
        Index("ix_preference_pairs_created_at", "created_at"),
    )
