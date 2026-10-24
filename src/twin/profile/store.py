"""Reading and switching the stored profile and activity versions (R-PROF-004, R-ACT-006).

Each recomputation writes a new row to ``profile_versions`` and ``activity_models`` and makes
them the *active* version of their scope: the active ids are kept in the ``settings`` table
(``profile.active.<scope>``, ``activity.active.<scope>``).  Online code reads the active
``live`` version, training and evaluation the active ``pre_holdout`` one (R-TRN-013).
Rolling back points the active ids at an older version (and at the activity model computed
together with it); the next recomputation creates a new version whose parent is the one that
was active.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select

from twin.clock import Clock
from twin.profile.activity_model import ActivityModel
from twin.profile.diffing import Change
from twin.profile.overrides import RoutineOverrides
from twin.profile.snapshot import ProfileMetrics
from twin.storage.db import Database
from twin.storage.profile_models import SCOPES, ActivityModelVersion, ProfileVersion
from twin.storage.settings_store import get_setting, put_setting

PROFILE_ACTIVE = "profile.active."
ACTIVITY_ACTIVE = "activity.active."


class VersionError(LookupError):
    """A version could not be found or named unambiguously."""


@dataclass(frozen=True)
class ProfileVersionView:
    id: str
    scope: str
    created_at: datetime
    parent_id: str | None
    reason: str
    her_messages: int
    data_range: dict[str, Any]
    summary_rules: str
    changes: tuple[Change, ...]
    active: bool
    input_hash: str

    @property
    def rule_lines(self) -> list[str]:
        return [line for line in self.summary_rules.splitlines() if line.strip()]


@dataclass(frozen=True)
class ActivityVersionView:
    id: str
    scope: str
    created_at: datetime
    parent_id: str | None
    profile_version_id: str | None
    reason: str
    her_messages: int
    data_range: dict[str, Any]
    active: bool
    input_hash: str


def _changes(raw: Any) -> tuple[Change, ...]:
    return tuple(Change.from_json(item) for item in raw or ())


class VersionStore:
    """Repository over the version tables and the active pointers."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    # ------------------------------------------------------------ pointers

    def active_profile_id(self, scope: str) -> str | None:
        with self._db.session() as session:
            value = get_setting(session, PROFILE_ACTIVE + scope)
        return str(value) if value else None

    def active_activity_id(self, scope: str) -> str | None:
        with self._db.session() as session:
            value = get_setting(session, ACTIVITY_ACTIVE + scope)
        return str(value) if value else None

    # ------------------------------------------------------------ profiles

    def _profile_view(self, row: ProfileVersion, active_id: str | None) -> ProfileVersionView:
        return ProfileVersionView(
            id=row.id,
            scope=row.scope,
            created_at=row.created_at,
            parent_id=row.parent_id,
            reason=row.reason,
            her_messages=row.her_messages,
            data_range=dict(row.data_range),
            summary_rules=row.summary_rules,
            changes=_changes(row.diff),
            active=row.id == active_id,
            input_hash=row.input_hash,
        )

    def latest_profile(self, scope: str) -> ProfileVersionView | None:
        stmt = (
            select(ProfileVersion)
            .where(ProfileVersion.scope == scope)
            .order_by(ProfileVersion.created_at.desc(), ProfileVersion.id.desc())
            .limit(1)
        )
        active = self.active_profile_id(scope)
        with self._db.session() as session:
            row = session.scalars(stmt).first()
            return self._profile_view(row, active) if row else None

    def active_profile(self, scope: str) -> ProfileVersionView | None:
        """The version in force: the active pointer, else the newest of the scope."""
        active = self.active_profile_id(scope)
        if active is not None:
            found = self.get_profile(active)
            if found is not None:
                return found
        return self.latest_profile(scope)

    def get_profile(self, version_id: str) -> ProfileVersionView | None:
        with self._db.session() as session:
            row = session.get(ProfileVersion, version_id)
            if row is None:
                return None
            return self._profile_view(row, self.active_profile_id(row.scope))

    def history(self, scope: str | None = None, limit: int = 30) -> list[ProfileVersionView]:
        stmt = select(ProfileVersion).order_by(
            ProfileVersion.created_at.desc(), ProfileVersion.id.desc()
        )
        if scope is not None:
            stmt = stmt.where(ProfileVersion.scope == scope)
        stmt = stmt.limit(limit)
        active = {s: self.active_profile_id(s) for s in SCOPES}
        with self._db.session() as session:
            return [self._profile_view(row, active[row.scope]) for row in session.scalars(stmt)]

    def metrics(self, version_id: str) -> ProfileMetrics:
        with self._db.session() as session:
            row = session.get(ProfileVersion, version_id)
            if row is None:
                raise VersionError(f"no profile version {version_id}")
            return ProfileMetrics(row.metrics)

    def phrases(self, version_id: str) -> dict[str, Any] | None:
        """The sealed frequent sentences / n-grams / address candidates (local use only)."""
        with self._db.session() as session:
            row = session.get(ProfileVersion, version_id)
            if row is None:
                raise VersionError(f"no profile version {version_id}")
            return dict(row.phrases) if row.phrases else None

    def resolve(self, name: str, scope: str | None = None) -> ProfileVersionView:
        """A version from a full id, a unique id prefix, or ``~N`` (the N-th newest; 0 newest)."""
        text = name.strip()
        if text.startswith("~") and text[1:].isdigit():
            ranked = self.history(scope, limit=int(text[1:]) + 1)
            if len(ranked) <= int(text[1:]):
                raise VersionError(f"there is no version {text} in {scope or 'any scope'}")
            return ranked[int(text[1:])]
        if len(text) < 4 or not text.isalnum():
            raise VersionError("give at least 4 letters or digits of the version id, or ~N")
        with self._db.session() as session:
            stmt = select(ProfileVersion.id).where(ProfileVersion.id.like(text.upper() + "%"))
            if scope is not None:
                stmt = stmt.where(ProfileVersion.scope == scope)
            matches = list(session.scalars(stmt))
        if not matches:
            raise VersionError(f"no profile version starts with {text!r}")
        if len(matches) > 1:
            raise VersionError(f"{text!r} matches {len(matches)} versions; use more characters")
        found = self.get_profile(matches[0])
        if found is None:
            raise VersionError(f"no profile version {matches[0]}")
        return found

    # ------------------------------------------------------------ activity

    def _activity_view(
        self, row: ActivityModelVersion, active_id: str | None
    ) -> ActivityVersionView:
        return ActivityVersionView(
            id=row.id,
            scope=row.scope,
            created_at=row.created_at,
            parent_id=row.parent_id,
            profile_version_id=row.profile_version_id,
            reason=row.reason,
            her_messages=row.her_messages,
            data_range=dict(row.data_range),
            active=row.id == active_id,
            input_hash=row.input_hash,
        )

    def get_activity(self, version_id: str) -> ActivityVersionView | None:
        with self._db.session() as session:
            row = session.get(ActivityModelVersion, version_id)
            if row is None:
                return None
            return self._activity_view(row, self.active_activity_id(row.scope))

    def latest_activity(self, scope: str) -> ActivityVersionView | None:
        stmt = (
            select(ActivityModelVersion)
            .where(ActivityModelVersion.scope == scope)
            .order_by(ActivityModelVersion.created_at.desc(), ActivityModelVersion.id.desc())
            .limit(1)
        )
        active = self.active_activity_id(scope)
        with self._db.session() as session:
            row = session.scalars(stmt).first()
            return self._activity_view(row, active) if row else None

    def active_activity(self, scope: str) -> ActivityVersionView | None:
        active = self.active_activity_id(scope)
        if active is not None:
            found = self.get_activity(active)
            if found is not None:
                return found
        return self.latest_activity(scope)

    def activity_for_profile(self, profile_version_id: str) -> ActivityVersionView | None:
        stmt = select(ActivityModelVersion).where(
            ActivityModelVersion.profile_version_id == profile_version_id
        )
        with self._db.session() as session:
            row = session.scalars(stmt).first()
            return self._activity_view(row, self.active_activity_id(row.scope)) if row else None

    def activity_model(self, version_id: str) -> ActivityModel:
        with self._db.session() as session:
            row = session.get(ActivityModelVersion, version_id)
            if row is None:
                raise VersionError(f"no activity model version {version_id}")
            return ActivityModel.from_json(row.model)

    # ------------------------------------------------------------- rollback

    def rollback(self, version_id: str) -> tuple[ProfileVersionView, ActivityVersionView | None]:
        """Make ``version_id`` (and the activity model computed with it) the active version."""
        target = self.get_profile(version_id)
        if target is None:
            raise VersionError(f"no profile version {version_id}")
        activity = self.activity_for_profile(target.id)
        with self._db.transaction() as session:
            put_setting(
                session,
                PROFILE_ACTIVE + target.scope,
                target.id,
                clock=self._clock,
                by="rollback",
            )
            if activity is not None:
                put_setting(
                    session,
                    ACTIVITY_ACTIVE + activity.scope,
                    activity.id,
                    clock=self._clock,
                    by="rollback",
                )
        return target, activity


def load_activity(
    db: Database,
    clock: Clock,
    scope: str,
    *,
    apply_overrides: bool = True,
) -> ActivityModel | None:
    """The active activity model of ``scope``, with the manual corrections applied."""
    store = VersionStore(db, clock)
    version = store.active_activity(scope)
    if version is None:
        return None
    model = store.activity_model(version.id)
    if apply_overrides:
        model = model.with_overrides(RoutineOverrides(db, clock).entries(include_disabled=False))
    return model
