"""Tables of the style profile and the activity model (round 04; R-STO-006).

``profile_versions``
    one row per recomputation of the statistical style profile (R-PROF-004), for one scope
    (``live`` = all data, ``pre_holdout`` = data before ``holdout_cutoff()``; R-TRN-013).
    ``metrics`` holds numbers only; the frequent sentences and n-grams, which are text of the
    real messages, are kept in the separate sealed column ``phrases``.
``activity_models``
    one row per recomputation of the routine model (R-ACT-006), linked to the profile version
    that was computed from the same data.
``routine_overrides``
    the manual corrections of the routine (R-ACT-005): sleep interval, weekly busy time,
    holiday date range.

Every sensitive field is sealed (R-STO-002); scalar columns hold only scope, counts and links.
"""

from __future__ import annotations

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from twin.storage.crypto import SealedBlob
from twin.storage.models import Base, TimestampMixin
from twin.storage.types import (
    encrypted_column,
    sealed_json,
    sealed_optional_text,
    sealed_text,
)

SCOPES = ("live", "pre_holdout")
OVERRIDE_KINDS = ("sleep", "busy", "holiday")


def _in_list(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class ProfileVersion(TimestampMixin, Base):
    """One version of the statistical style profile (R-PROF-004)."""

    __tablename__ = "profile_versions"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    scope: Mapped[str] = mapped_column(String(16), nullable=False)
    parent_id: Mapped[str | None] = mapped_column(
        String(26), ForeignKey("profile_versions.id"), nullable=True
    )
    reason: Mapped[str] = mapped_column(String(32), nullable=False, default="rebuild")
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    her_messages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    data_range_ct: Mapped[SealedBlob] = encrypted_column("data_range", json=True)
    data_range = sealed_json()
    metrics_ct: Mapped[SealedBlob] = encrypted_column("metrics", json=True)
    metrics = sealed_json()
    phrases_ct: Mapped[SealedBlob | None] = encrypted_column("phrases", json=True, nullable=True)
    phrases = sealed_json(optional=True)
    summary_rules_ct: Mapped[SealedBlob] = encrypted_column("summary_rules")
    summary_rules = sealed_text()
    diff_ct: Mapped[SealedBlob | None] = encrypted_column("diff", json=True, nullable=True)
    diff = sealed_json(optional=True)

    __table_args__ = (
        CheckConstraint(_in_list("scope", SCOPES), name="scope"),
        Index("ix_profile_versions_scope_created", "scope", "created_at"),
    )


class ActivityModelVersion(TimestampMixin, Base):
    """One version of the activity (routine) model (R-ACT-006)."""

    __tablename__ = "activity_models"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    scope: Mapped[str] = mapped_column(String(16), nullable=False)
    parent_id: Mapped[str | None] = mapped_column(
        String(26), ForeignKey("activity_models.id"), nullable=True
    )
    profile_version_id: Mapped[str | None] = mapped_column(
        String(26), ForeignKey("profile_versions.id"), nullable=True
    )
    reason: Mapped[str] = mapped_column(String(32), nullable=False, default="rebuild")
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    her_messages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    data_range_ct: Mapped[SealedBlob] = encrypted_column("data_range", json=True)
    data_range = sealed_json()
    model_ct: Mapped[SealedBlob] = encrypted_column("model", json=True)
    model = sealed_json()
    diff_ct: Mapped[SealedBlob | None] = encrypted_column("diff", json=True, nullable=True)
    diff = sealed_json(optional=True)

    __table_args__ = (
        CheckConstraint(_in_list("scope", SCOPES), name="scope"),
        Index("ix_activity_models_scope_created", "scope", "created_at"),
        Index("ix_activity_models_profile_version_id", "profile_version_id"),
    )


class RoutineOverride(TimestampMixin, Base):
    """A manual correction of the routine; it wins over what the data suggests (R-ACT-005)."""

    __tablename__ = "routine_overrides"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    params_ct: Mapped[SealedBlob] = encrypted_column("params", json=True)
    params = sealed_json()
    note_ct: Mapped[SealedBlob | None] = encrypted_column("note", nullable=True)
    note = sealed_optional_text()
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    __table_args__ = (
        CheckConstraint(_in_list("kind", OVERRIDE_KINDS), name="kind"),
        Index("ix_routine_overrides_kind", "kind"),
    )
