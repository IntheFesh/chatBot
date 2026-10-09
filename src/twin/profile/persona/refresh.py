"""Keeping the persona card up to date after an import (R-IMP-011, R-PERS-002).

Three things happen to a card without anyone writing it:

``refresh_stats``
    a new version with a fresh ``[自动-统计规则]`` from the active profile of the scope; every
    other section keeps its exact text.  This costs nothing, so it runs after every import.
``sync_manual_style``
    the pre-holdout card carries the style lines of the live card's ``[手动]`` section (and
    nothing else of it); when the live card is edited the pre-holdout card follows.
``write_description``
    a new version with a new ``[自动-描述]`` (the result of the Map-Reduce run) and, in the same
    version, fresh statistics; ``[手动]`` and ``[不要这样]`` are taken from the version in force
    *at the time of writing*, so a hand edit made while the model was working is not lost.

:func:`description_due` answers whether the automatic text should be written again: never
written, or her messages have grown by ``persona.regen_ratio`` (10 %) since it was.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from twin.ingest.corpus import her_message_count
from twin.profile.holdout import HoldoutError, get_holdout, holdout_cutoff
from twin.profile.persona import compose
from twin.profile.persona.sections import AUTO, DONT, MANUAL, STATS, CardText, split_card
from twin.profile.persona.store import PersonaCardView, PersonaStore
from twin.profile.store import VersionStore
from twin.services import Services

MIN_DESCRIBED_MESSAGES = 30  # fewer messages from her than this are too few to describe her from


@dataclass(frozen=True)
class RefreshResult:
    scope: str
    status: Literal["created", "unchanged", "skipped"]
    note: str
    card: PersonaCardView | None = None


@dataclass(frozen=True)
class Due:
    """Whether the automatic description of a scope should be generated again."""

    scope: str
    due: bool
    reason: str
    her_messages: int
    described_messages: int | None


def persona_store(services: Services) -> PersonaStore:
    return PersonaStore(services.db, services.clock)


def _rules_of(services: Services, scope: str) -> tuple[list[str], str] | None:
    version = VersionStore(services.db, services.clock).active_profile(scope)
    if version is None:
        return None
    return version.rule_lines, version.id


def count_her_messages(services: Services, scope: str, *, split: bool = True) -> int:
    """Her messages in the data of the scope (before the cutoff for ``pre_holdout``).

    ``split=False`` never works out a missing hold-out (that writes the setting, which a
    read-only command may not do); it raises :class:`HoldoutError` instead.
    """
    before: datetime | None = None
    if scope == "pre_holdout":
        if split:
            before = holdout_cutoff(services)
        else:
            stored = get_holdout(services)
            if stored is None:
                raise HoldoutError("the hold-out has not been split yet")
            before = stored.cutoff
    with services.db.session() as session:
        return int(session.scalar(her_message_count(before)) or 0)


def description_due(services: Services, scope: str, *, split: bool = True) -> Due:
    """Is the description of ``scope`` due; ``split=False`` for read-only commands."""
    try:
        current = count_her_messages(services, scope, split=split)
    except HoldoutError as exc:
        return Due(scope, False, f"no hold-out split yet: {exc}", 0, None)
    card = persona_store(services).active(scope)
    described = card.described_her_messages if card is not None else None
    if current < MIN_DESCRIBED_MESSAGES:
        return Due(scope, False, f"only {current} messages from her", current, described)
    if described is None:
        return Due(scope, True, "the description was never generated", current, described)
    ratio = services.settings.persona.regen_ratio
    grown = (current - described) / max(1, described)
    if grown >= ratio:
        return Due(
            scope, True, f"her messages grew {grown:.0%} since the description", current, described
        )
    return Due(
        scope, False, f"her messages grew {grown:.0%} (limit {ratio:.0%})", current, described
    )


def refresh_stats(services: Services, scope: str, *, force: bool = False) -> RefreshResult:
    """A new version with fresh statistics rules, everything else untouched."""
    store = persona_store(services)
    rules = _rules_of(services, scope)
    if rules is None:
        return RefreshResult(
            scope, "skipped", "no profile of this scope yet (`twin profile rebuild`)"
        )
    lines, profile_id = rules
    stats = compose.stats_block(lines)
    card = store.active(scope)
    if card is None:
        manual: list[str] = []
        if scope == "pre_holdout":
            live = store.active("live")
            manual = compose.manual_style_lines(live.sections) if live is not None else []
        text = compose.new_card(scope, stats, compose.empty_auto_block(), manual)
        created = store.add_version(scope, text, reason="stats", profile_version_id=profile_id)
        return RefreshResult(scope, "created", "first version (statistics only)", created)
    sections = card.sections
    if sections.block(STATS) == stats and not force:
        return RefreshResult(scope, "unchanged", "the statistics rules did not change", card)
    text = sections.with_block(STATS, stats).assemble()
    created = store.add_version(
        scope,
        text,
        reason="stats",
        profile_version_id=profile_id,
        template_version=card.template_version,
        described_her_messages=card.described_her_messages,
        described_at=card.described_at,
        provenance=store.provenance(card.id),
    )
    return RefreshResult(scope, "created", "statistics rules refreshed", created)


def sync_manual_style(services: Services) -> RefreshResult:
    """Make the pre-holdout card carry the style lines of the live ``[手动]`` section."""
    store = persona_store(services)
    live = store.active("live")
    target = store.active("pre_holdout")
    if target is None:
        return RefreshResult("pre_holdout", "skipped", "no pre-holdout card yet")
    block = compose.pre_holdout_manual(live.text if live is not None else None)
    sections = target.sections
    if sections.block(MANUAL) == block:
        return RefreshResult("pre_holdout", "unchanged", "the style lines are the same", target)
    created = store.add_version(
        "pre_holdout",
        sections.with_block(MANUAL, block).assemble(),
        reason="manual_sync",
        profile_version_id=target.profile_version_id,
        template_version=target.template_version,
        described_her_messages=target.described_her_messages,
        described_at=target.described_at,
        provenance=store.provenance(target.id),
    )
    return RefreshResult("pre_holdout", "created", "hand-written style lines copied", created)


def refresh_all(
    services: Services, scope: str = "all", *, force: bool = False
) -> list[RefreshResult]:
    """Statistics for the scope(s), then the style lines of the pre-holdout card."""
    scopes = ("live", "pre_holdout") if scope == "all" else (scope,)
    results = [refresh_stats(services, name, force=force) for name in scopes]
    if "pre_holdout" in scopes:
        results.append(sync_manual_style(services))
    return results


def write_description(
    services: Services,
    scope: str,
    description_body: str,
    *,
    provenance: dict[str, Any],
    template_version: str,
    her_messages: int,
    at: datetime,
) -> PersonaCardView:
    """A new version with this ``[自动-描述]`` (and fresh statistics where a profile exists)."""
    store = persona_store(services)
    card = store.active(scope)
    rules = _rules_of(services, scope)
    auto = compose.auto_block(description_body)
    if card is None:
        stats = compose.stats_block(rules[0] if rules else [])
        manual: list[str] = []
        if scope == "pre_holdout":
            live = store.active("live")
            manual = compose.manual_style_lines(live.sections) if live is not None else []
        text = compose.new_card(scope, stats, auto, manual)
    else:
        sections: CardText = card.sections.with_block(AUTO, auto)
        if rules is not None:
            sections = sections.with_block(STATS, compose.stats_block(rules[0]))
        text = sections.assemble()
    return store.add_version(
        scope,
        text,
        reason="generate",
        profile_version_id=rules[1] if rules else (card.profile_version_id if card else None),
        template_version=template_version,
        described_her_messages=her_messages,
        described_at=at,
        provenance=provenance,
    )


# ----------------------------------------------------------------- corrections


def read_corrections(services: Services) -> list[str]:
    """The lines of ``[不要这样]`` of the live card."""
    card = persona_store(services).active("live")
    return compose.correction_lines(card.sections) if card is not None else []


def write_corrections(
    services: Services, lines: Sequence[str], *, reason: str = "corrections"
) -> PersonaCardView:
    """Replace ``[不要这样]`` of the live card with ``lines`` (round 11 writes through this).

    Only that section changes; every other section keeps its exact text.  The caller checks the
    lines (they may only describe a manner of speaking, R-TRN-013).
    """
    store = persona_store(services)
    card = store.active("live")
    cleaned = [" ".join(line.split()) for line in lines if line.strip()]
    block = compose.dont_block(list(dict.fromkeys(cleaned)))
    if card is None:
        stats = compose.stats_block([])
        fresh = compose.new_card("live", stats, compose.empty_auto_block())
        text = split_card(fresh).with_block(DONT, block).assemble()
    else:
        if card.sections.block(DONT) == block:
            return card
        text = card.sections.with_block(DONT, block).assemble()
    return store.add_version(
        "live",
        text,
        reason=reason,
        profile_version_id=card.profile_version_id if card else None,
        template_version=card.template_version if card else None,
        described_her_messages=card.described_her_messages if card else None,
        described_at=card.described_at if card else None,
        provenance=store.provenance(card.id) if card else None,
    )
