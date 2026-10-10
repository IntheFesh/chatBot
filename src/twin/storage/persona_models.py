"""Tables of the persona card and of the prompt templates (round 06; R-STO-006, R-PERS-005).

``persona_cards``
    one row per version of a persona card, for one scope (``live`` = all data,
    ``pre_holdout`` = data before ``holdout_cutoff()``; R-TRN-013).  ``content`` is the whole
    card as Markdown (sealed: it describes her); ``provenance`` (sealed JSON) records which
    sampled conversation segments, and which of her messages, the automatic description was
    made from and which segment numbers back each statement of it.  ``number`` counts the
    versions of a scope from 1; the version in force is named by the ``persona.active.<scope>``
    setting.  ``described_her_messages`` is the number of her messages when the automatic
    description was last generated: a version that only refreshes the statistics, or only
    carries a hand edit, keeps the value of the version it came from.
``prompt_templates``
    the prompt templates in use, one row per ``(name, version)``, loaded from the files in
    ``twin/profile/templates``.  The text is sealed like every other text (R-STO-002, and the
    database file then holds no readable Chinese at all); its SHA-256 is kept in the clear so
    that an edited template file with an unchanged version number is noticed.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import CheckConstraint, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from twin.storage.crypto import SealedBlob
from twin.storage.models import Base, TimestampMixin
from twin.storage.profile_models import SCOPES
from twin.storage.types import UTCDateTime, encrypted_column, sealed_json, sealed_text

TEMPLATE_SOURCES = ("file", "edited")


def _in_list(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class PersonaCardVersion(TimestampMixin, Base):
    """One version of a persona card (R-PERS-002, R-PERS-003)."""

    __tablename__ = "persona_cards"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    scope: Mapped[str] = mapped_column(String(16), nullable=False)
    number: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_id: Mapped[str | None] = mapped_column(
        String(26), ForeignKey("persona_cards.id"), nullable=True
    )
    reason: Mapped[str] = mapped_column(String(32), nullable=False, default="generate")
    profile_version_id: Mapped[str | None] = mapped_column(
        String(26), ForeignKey("profile_versions.id"), nullable=True
    )
    template_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    described_her_messages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    described_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    content_ct: Mapped[SealedBlob] = encrypted_column("content")
    content = sealed_text()
    provenance_ct: Mapped[SealedBlob | None] = encrypted_column(
        "provenance", json=True, nullable=True
    )
    provenance = sealed_json(optional=True)

    __table_args__ = (
        CheckConstraint(_in_list("scope", SCOPES), name="scope"),
        UniqueConstraint("scope", "number", name="uq_persona_cards_scope_number"),
        Index("ix_persona_cards_scope_created", "scope", "created_at"),
    )


class PromptTemplate(TimestampMixin, Base):
    """A versioned prompt template (R-PERS-005, R-OPS-010)."""

    __tablename__ = "prompt_templates"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    content_ct: Mapped[SealedBlob] = encrypted_column("content")
    content = sealed_text()
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="file")

    __table_args__ = (
        CheckConstraint(_in_list("source", TEMPLATE_SOURCES), name="source"),
        UniqueConstraint("name", "version", name="uq_prompt_templates_name_version"),
    )
