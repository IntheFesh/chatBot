"""Storing what the extractor found: conflicts decided, vectors kept, follow-ups linked.

:class:`MemoryWriter` takes an :class:`~twin.memory.extract.Extraction` (or the facts and
follow-ups of one) and writes it:

1. facts that say the same thing twice in one extraction are merged;
2. for each fact the older facts it could clash with are recalled; the model judges the pairs in
   batches (:class:`~twin.memory.conflict.ConflictResolver`) except for an exact repeat, which
   needs no judge;
3. :func:`~twin.memory.conflict.decide` turns the relations into one of: *insert* (replacing
   older, lower-ranked facts; invalidating life line entries a real fact contradicts), *merge*
   into an existing fact that already says it, or *reject* (a higher source vetoed it: kept as a
   rejected candidate, never shown);
4. a fact the bot invented about herself also becomes an ``improvised`` life line entry;
5. new facts are encoded into the vector table;
6. follow-ups are stored (a repeat of an open one is not stored twice), linked to the fact that
   shares their evidence so that forgetting the fact forgets them, and follow-ups the
   conversation shows to be over are closed at the time of the message that shows it.

The text of a fact is never logged; reports carry counts and ids.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

from twin.llm.deepseek import DeepSeekClient
from twin.llm.types import LedgerTag
from twin.memory.conflict import (
    JUDGE_BATCH,
    ConflictResolver,
    Decision,
    Recall,
    decide,
    normalised,
)
from twin.memory.extract import ClosedDraft, Extraction, FactDraft, FollowupDraft
from twin.memory.followups import FollowupStore
from twin.memory.lifeline import LifelineStore, PlannedEvent
from twin.memory.memory import Memory
from twin.memory.records import FactRecord
from twin.memory.store import NewFact
from twin.ops.logging import get_logger
from twin.profile.prompt_templates import TemplateStore
from twin.storage.memory_models import BOT_CONVERSATION_SOURCES

log = get_logger("twin.memory.writer")

LIFELINE_CATEGORIES = ("life", "plan", "work_study")
LIFELINE_ACTIVITY_CHARS = 80


@dataclass
class WriteReport:
    """What a write did (counts and ids only)."""

    inserted: list[str] = field(default_factory=list)  # ids of the new current facts
    merged: int = 0
    rejected: int = 0
    superseded: int = 0
    events_invalidated: int = 0
    improvised_events: int = 0
    judge_calls: int = 0
    followups_added: int = 0
    followups_repeated: int = 0
    followups_closed: int = 0

    def add(self, other: WriteReport) -> None:
        self.inserted.extend(other.inserted)
        for name in (
            "merged",
            "rejected",
            "superseded",
            "events_invalidated",
            "improvised_events",
            "judge_calls",
            "followups_added",
            "followups_repeated",
            "followups_closed",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    @property
    def facts_added(self) -> int:
        return len(self.inserted)


def merge_drafts(drafts: Sequence[FactDraft]) -> list[FactDraft]:
    """Facts of one extraction that say the same thing become one (evidence joined)."""
    merged: dict[tuple[str, str, str], FactDraft] = {}
    for draft in drafts:
        key = (draft.subject, draft.source, normalised(draft.text))
        first = merged.get(key)
        if first is None:
            merged[key] = draft
            continue
        refs = tuple(dict.fromkeys([*first.evidence_refs, *draft.evidence_refs]))
        merged[key] = replace(
            first,
            evidence_refs=refs,
            known_at=min(first.known_at, draft.known_at),
            importance=max(first.importance, draft.importance),
            confidence=max(first.confidence, draft.confidence),
        )
    return list(merged.values())


class MemoryWriter:
    """Writes drafts into the memory (see the module description)."""

    def __init__(
        self,
        memory: Memory,
        client: DeepSeekClient | None,
        *,
        templates: TemplateStore | None = None,
    ) -> None:
        self._memory = memory
        self._store = memory.store
        self._resolver = ConflictResolver(memory, client, templates=templates)
        self._lifeline = LifelineStore(memory)
        self._followups = FollowupStore(memory)

    # --------------------------------------------------------------------- entry

    async def write(self, extraction: Extraction, tag: LedgerTag) -> WriteReport:
        """Store facts, then follow-ups and the follow-ups the conversation closed."""
        report = await self.write_facts(extraction.facts, tag)
        linked = await asyncio.to_thread(self._link_by_evidence, report.inserted)
        report.add(
            await asyncio.to_thread(
                self.write_followups, extraction.followups, extraction.closed, linked
            )
        )
        return report

    async def write_facts(self, drafts: Sequence[FactDraft], tag: LedgerTag) -> WriteReport:
        report = WriteReport()
        if not drafts:
            return report
        await asyncio.to_thread(self._memory.refresh)
        plans = await asyncio.to_thread(self._recall_all, merge_drafts(drafts))
        relations: list[dict[str, str]] = [{} for _ in plans]
        needs: list[int] = []
        for index, (draft, recall) in enumerate(plans):
            if recall.empty:
                continue
            repeat = self._resolver.exact_repeat(draft, recall)
            if repeat is not None:
                relations[index] = {repeat: "same"}
            else:
                needs.append(index)
        if needs:
            judged = await self._resolver.judge([plans[i] for i in needs], tag)
            for index, mapping in zip(needs, judged, strict=True):
                relations[index] = mapping
            report.judge_calls = math.ceil(len(needs) / JUDGE_BATCH)
        applied = await asyncio.to_thread(self._apply_all, plans, relations)
        report.add(applied)
        if applied.inserted:
            new = self._store.facts(applied.inserted)
            await asyncio.to_thread(self._memory.sync_vectors, facts=new)
        return report

    # -------------------------------------------------------------------- facts

    def _recall_all(self, drafts: Sequence[FactDraft]) -> list[tuple[FactDraft, Recall]]:
        return [(draft, self._resolver.recall(draft)) for draft in drafts]

    def _apply_all(
        self, plans: Sequence[tuple[FactDraft, Recall]], relations: Sequence[dict[str, str]]
    ) -> WriteReport:
        report = WriteReport()
        for (draft, recall), mapping in zip(plans, relations, strict=True):
            self._apply(draft, recall, decide(draft, recall, mapping), report)
        self._memory.refresh()
        return report

    def _apply(
        self, draft: FactDraft, recall: Recall, decision: Decision, report: WriteReport
    ) -> None:
        store = self._store
        if draft.source in BOT_CONVERSATION_SOURCES:
            store.mark_bot_online(draft.known_at)
        if decision.action == "merge" and decision.target_id is not None:
            target = store.fact(decision.target_id)
            if target is not None:
                store.update_fact(
                    target.id,
                    evidence=_joined_evidence(target, draft),
                    known_at=min(target.known_at, draft.known_at),
                    confidence=max(target.confidence, draft.confidence),
                    importance=max(target.importance, draft.importance),
                )
                report.merged += 1
                return
        if decision.action == "reject":
            store.add_fact(_new_fact(draft, status="rejected", rejected_by=decision.target_id))
            report.rejected += 1
            return
        newer = store.fact(decision.superseded_by) if decision.superseded_by else None
        record = store.add_fact(
            _new_fact(
                draft,
                superseded_by=newer.id if newer else None,
                superseded_at=newer.known_at if newer else None,
                valid_to=newer.known_at
                if newer and (draft.valid_to is None or draft.valid_to > newer.known_at)
                else draft.valid_to,
            )
        )
        report.inserted.append(record.id)
        known = {c.fact.id: c.fact for c in recall.facts}
        for old_id in decision.supersede:
            old = known.get(old_id)
            at = max(draft.known_at, old.known_at) if old else draft.known_at
            store.supersede_fact(old_id, record.id, at=at)
            report.superseded += 1
        now = self._memory.services.clock.now_utc()
        for event_id in decision.invalidate_events:
            if store.invalidate_event(event_id, by=record.id, at=now):
                report.events_invalidated += 1
        if self._improvise(draft, record):
            report.improvised_events += 1

    def _improvise(self, draft: FactDraft, record: FactRecord) -> bool:
        """A detail the bot invented about herself also goes on her life line (R-MEM-005)."""
        if draft.source != "bot_invented" or draft.subject not in ("her", "both"):
            return False
        hint = draft.lifeline
        if hint is None and draft.category not in LIFELINE_CATEGORIES:
            return False
        day: date = draft.event_date or self._memory.clock.bot_date(draft.known_at)
        event = (
            PlannedEvent(hint.activity, hint.start, hint.end, hint.place, hint.mood, draft.text)
            if hint is not None
            else PlannedEvent(draft.text[:LIFELINE_ACTIVITY_CHARS], detail=draft.text)
        )
        self._lifeline.add_improvised(day, event, fact_id=record.id)
        return True

    # ---------------------------------------------------------------- follow-ups

    def _link_by_evidence(self, fact_ids: Sequence[str]) -> dict[str, str]:
        """``{evidence ref: fact id}`` of the facts just stored (to tie follow-ups to them)."""
        linked: dict[str, str] = {}
        for fact in self._store.facts(list(fact_ids)):
            for ref in (fact.evidence or {}).get("ids", []):
                linked.setdefault(str(ref), fact.id)
        return linked

    def write_followups(
        self,
        followups: Sequence[FollowupDraft],
        closed: Sequence[ClosedDraft],
        linked: dict[str, str],
    ) -> WriteReport:
        report = WriteReport()
        for draft in followups:
            fact_id = next((linked[r] for r in draft.evidence_refs if r in linked), None)
            _, created = self._followups.add(
                draft.text,
                draft.due_at,
                window_minutes=draft.window_minutes,
                origin=draft.origin,
                source_turn_id=draft.source_turn_id,
                created_at=draft.created_at,
                fact_id=fact_id,
                evidence=draft.evidence(),
            )
            if created:
                report.followups_added += 1
            else:
                report.followups_repeated += 1
        for item in closed:
            if self._followups.close(
                item.followup_id, status=item.reason, reason="mentioned", at=item.at
            ):
                report.followups_closed += 1
        return report


def _new_fact(draft: FactDraft, **overrides: Any) -> NewFact:
    base = NewFact(
        subject=draft.subject,
        category=draft.category,
        text=draft.text,
        source=draft.source,
        known_at=draft.known_at,
        confidence=draft.confidence,
        importance=draft.importance,
        valid_from=draft.valid_from,
        valid_to=draft.valid_to,
        event_date=draft.event_date,
        recurrence=draft.recurrence,
        evidence=draft.evidence(),
    )
    return replace(base, **overrides)


def _joined_evidence(target: FactRecord, draft: FactDraft) -> dict[str, object]:
    old = target.evidence or {}
    ids = [*old.get("ids", []), *draft.evidence_refs]
    return {"kind": old.get("kind", draft.evidence_kind), "ids": list(dict.fromkeys(ids))}
