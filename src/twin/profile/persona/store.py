"""Reading and writing the versions of the persona card (R-PERS-003, R-PERS-005).

Every change writes a new row to ``persona_cards`` and makes it the version in force of its
scope: the id is kept in the ``settings`` table (``persona.active.<scope>``).  Online code reads
the ``live`` card, training and evaluation the ``pre_holdout`` one (R-TRN-013).  Rolling back
points the setting at an older row; the next change becomes a new version whose parent is the
one that was in force.  Writes bump ``settings.state_version`` so that a running application
reloads the card.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from sqlalchemy import func, select

from twin.clock import Clock
from twin.profile.persona.sections import CardText, split_card
from twin.storage.db import Database
from twin.storage.persona_models import PersonaCardVersion
from twin.storage.profile_models import SCOPES
from twin.storage.settings_store import get_setting, put_setting

ACTIVE_KEY = "persona.active."


class PersonaVersionError(LookupError):
    """A card version could not be found or named unambiguously."""


@dataclass(frozen=True)
class PersonaCardView:
    """A stored version of a card."""

    id: str
    scope: str
    number: int
    parent_id: str | None
    reason: str
    created_at: datetime
    profile_version_id: str | None
    template_version: str | None
    described_her_messages: int | None
    described_at: datetime | None
    text: str
    active: bool

    @property
    def sections(self) -> CardText:
        return split_card(self.text)

    @property
    def label(self) -> str:
        return f"v{self.number}"


def _view(row: PersonaCardVersion, active_id: str | None) -> PersonaCardView:
    return PersonaCardView(
        id=row.id,
        scope=row.scope,
        number=row.number,
        parent_id=row.parent_id,
        reason=row.reason,
        created_at=row.created_at,
        profile_version_id=row.profile_version_id,
        template_version=row.template_version,
        described_her_messages=row.described_her_messages,
        described_at=row.described_at,
        text=row.content,
        active=row.id == active_id,
    )


class PersonaStore:
    """Repository over the ``persona_cards`` table and the active pointers."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    # ------------------------------------------------------------ pointers

    def active_id(self, scope: str) -> str | None:
        with self._db.session() as session:
            value = get_setting(session, ACTIVE_KEY + scope)
        return str(value) if value else None

    # -------------------------------------------------------------- reading

    def latest(self, scope: str) -> PersonaCardView | None:
        stmt = (
            select(PersonaCardVersion)
            .where(PersonaCardVersion.scope == scope)
            .order_by(PersonaCardVersion.number.desc())
            .limit(1)
        )
        active = self.active_id(scope)
        with self._db.session() as session:
            row = session.scalars(stmt).first()
            return _view(row, active) if row else None

    def get(self, version_id: str) -> PersonaCardView | None:
        with self._db.session() as session:
            row = session.get(PersonaCardVersion, version_id)
            if row is None:
                return None
            return _view(row, self.active_id(row.scope))

    def active(self, scope: str) -> PersonaCardView | None:
        """The version in force: the active pointer, else the newest of the scope."""
        active = self.active_id(scope)
        if active is not None:
            found = self.get(active)
            if found is not None:
                return found
        return self.latest(scope)

    def history(self, scope: str | None = None, limit: int = 30) -> list[PersonaCardView]:
        stmt = select(PersonaCardVersion).order_by(
            PersonaCardVersion.created_at.desc(), PersonaCardVersion.number.desc()
        )
        if scope is not None:
            stmt = stmt.where(PersonaCardVersion.scope == scope)
        active = {name: self.active_id(name) for name in SCOPES}
        with self._db.session() as session:
            return [_view(row, active[row.scope]) for row in session.scalars(stmt.limit(limit))]

    def provenance(self, version_id: str) -> dict[str, Any] | None:
        """Which segments and messages the automatic text of a version was made from."""
        with self._db.session() as session:
            row = session.get(PersonaCardVersion, version_id)
            if row is None:
                raise PersonaVersionError(f"no persona card version {version_id}")
            found = row.provenance
            return dict(found) if found else None

    def resolve(self, name: str, scope: str) -> PersonaCardView:
        """A version of ``scope`` from ``vN`` (its number), ``~N`` (N-th newest) or an id prefix."""
        text = name.strip()
        if text.lower().startswith("v") and text[1:].isdigit():
            number = int(text[1:])
            with self._db.session() as session:
                row = session.scalars(
                    select(PersonaCardVersion).where(
                        PersonaCardVersion.scope == scope, PersonaCardVersion.number == number
                    )
                ).first()
                active = self.active_id(scope)
                if row is None:
                    raise PersonaVersionError(f"there is no version v{number} in {scope}")
                return _view(row, active)
        if text.startswith("~") and text[1:].isdigit():
            ranked = self.history(scope, limit=int(text[1:]) + 1)
            if len(ranked) <= int(text[1:]):
                raise PersonaVersionError(f"there is no version {text} in {scope}")
            return ranked[int(text[1:])]
        if len(text) < 4 or not text.isalnum():
            raise PersonaVersionError("give vN, ~N or at least 4 letters or digits of the id")
        with self._db.session() as session:
            matches = list(
                session.scalars(
                    select(PersonaCardVersion.id).where(
                        PersonaCardVersion.id.like(text.upper() + "%"),
                        PersonaCardVersion.scope == scope,
                    )
                )
            )
        if not matches:
            raise PersonaVersionError(f"no persona card version of {scope} starts with {text!r}")
        if len(matches) > 1:
            raise PersonaVersionError(f"{text!r} matches {len(matches)} versions; use more")
        found = self.get(matches[0])
        if found is None:
            raise PersonaVersionError(f"no persona card version {matches[0]}")
        return found

    # -------------------------------------------------------------- writing

    def add_version(
        self,
        scope: str,
        text: str,
        *,
        reason: str,
        profile_version_id: str | None = None,
        template_version: str | None = None,
        described_her_messages: int | None = None,
        described_at: datetime | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> PersonaCardView:
        """Store ``text`` as the next version of ``scope`` and make it the one in force."""
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {', '.join(SCOPES)}")
        split_card(text)  # a text that is not a card is never stored
        now = self._clock.now_utc()
        with self._db.transaction(bump_state=True) as session:
            top = session.scalar(
                select(func.max(PersonaCardVersion.number)).where(PersonaCardVersion.scope == scope)
            )
            active = get_setting(session, ACTIVE_KEY + scope)
            parent = session.get(PersonaCardVersion, str(active)) if active else None
            row = PersonaCardVersion(
                scope=scope,
                number=int(top or 0) + 1,
                parent_id=parent.id if parent is not None else None,
                reason=reason,
                profile_version_id=profile_version_id,
                template_version=template_version,
                described_her_messages=described_her_messages,
                described_at=described_at,
                content=text,
                provenance=provenance,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            session.flush()
            put_setting(session, ACTIVE_KEY + scope, row.id, clock=self._clock, by=reason)
            return _view(row, row.id)

    def rollback(self, version_id: str) -> PersonaCardView:
        """Make an older version the one in force (a LIGHT write)."""
        target = self.get(version_id)
        if target is None:
            raise PersonaVersionError(f"no persona card version {version_id}")
        with self._db.transaction(bump_state=True) as session:
            put_setting(
                session, ACTIVE_KEY + target.scope, target.id, clock=self._clock, by="rollback"
            )
        return replace(target, active=True)
