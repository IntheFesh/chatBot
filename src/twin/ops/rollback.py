"""``twin rollback``: one entry to the rollbacks of the versioned things (R-OPS-010, R-PERS-005).

The profile, the persona card, the prompt templates and the style model each have versions and
a rollback of their own (rounds 04, 06, 13); this module calls those and nothing else - there is
no second way of choosing a version - and writes one **audit record** for each rollback into the
setting ``ops.rollback.log`` (when, what, from which version to which; newest last, 100 kept).

========================  ================================================================
``profile <id|~N>``       the statistical profile and its routine model (``--scope``)
``persona <vN|~N|id>``    the persona card (``--scope``)
``prompt-template``       one template, ``<name> <version>``
``style-model <id>``      the registered style model that is in use
========================  ================================================================

All are LIGHT commands: a running application notices the change within two seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from twin.profile.persona.store import PersonaStore, PersonaVersionError
from twin.profile.prompt_templates import TemplateError, TemplateStore
from twin.profile.store import VersionError, VersionStore
from twin.services import Services
from twin.storage.settings_store import get_setting, put_setting
from twin.training.registry import RegistryError, rollback_model

AUDIT_KEY = "ops.rollback.log"
AUDIT_KEEP = 100


class RollbackError(Exception):
    """A rollback was refused; the message is for the person."""


@dataclass(frozen=True)
class RollbackResult:
    """What was rolled back, from what, to what."""

    subject: str
    previous: str | None
    current: str
    detail: str = ""

    def to_json(self, at: str) -> dict[str, Any]:
        return {
            "at": at,
            "subject": self.subject,
            "from": self.previous,
            "to": self.current,
            "detail": self.detail,
        }


def record_audit(services: Services, result: RollbackResult) -> None:
    """Append the audit record of a rollback."""
    entry = result.to_json(services.clock.now_utc().isoformat())
    with services.db.transaction() as session:
        log = list(get_setting(session, AUDIT_KEY) or [])
        log.append(entry)
        put_setting(
            session,
            AUDIT_KEY,
            log[-AUDIT_KEEP:],
            clock=services.clock,
            by="rollback",
            record_history=False,
        )


def audit_entries(services: Services) -> list[dict[str, Any]]:
    """The audit records, oldest first."""
    with services.db.session() as session:
        found = get_setting(session, AUDIT_KEY)
    return [dict(item) for item in found] if isinstance(found, list) else []


def rollback_profile(services: Services, version: str, scope: str | None = None) -> RollbackResult:
    store = VersionStore(services.db, services.clock)
    try:
        target = store.resolve(version, scope)
        before = store.active_profile_id(target.scope)
        chosen, activity = store.rollback(target.id)
    except VersionError as exc:
        raise RollbackError(str(exc)) from exc
    detail = f"scope {chosen.scope}" + (f", routine model {activity.id}" if activity else "")
    return RollbackResult("profile", before, chosen.id, detail)


def rollback_persona(services: Services, version: str, scope: str = "live") -> RollbackResult:
    store = PersonaStore(services.db, services.clock)
    try:
        target = store.resolve(version, scope)
        before = store.active_id(scope)
        chosen = store.rollback(target.id)
    except PersonaVersionError as exc:
        raise RollbackError(str(exc)) from exc
    return RollbackResult("persona", before, chosen.id, f"scope {chosen.scope}, {chosen.label}")


def rollback_prompt_template(
    services: Services, name: str, version: int, directory: Path | None = None
) -> RollbackResult:
    """Make ``name.v<version>`` the template in force (``directory``: the template files)."""
    store = (
        TemplateStore(services.db, services.clock)
        if directory is None
        else TemplateStore(services.db, services.clock, directory)
    )
    try:
        store.sync()
        before = store.active_version(name)
        store.rollback(name, version)
    except TemplateError as exc:
        raise RollbackError(str(exc)) from exc
    return RollbackResult("prompt-template", f"{name}.v{before}", f"{name}.v{version}")


def rollback_style_model(services: Services, reference: str) -> RollbackResult:
    try:
        previous, chosen = rollback_model(services.db, reference)
    except RegistryError as exc:
        raise RollbackError(str(exc)) from exc
    note = "" if chosen.gate_passed else "did not pass the release gate (R-SRV-005)"
    return RollbackResult(
        "style-model", previous.id if previous else None, chosen.id, note or "gate passed"
    )
