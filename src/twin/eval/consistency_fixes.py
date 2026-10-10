"""Corrections of the memory that a confirmed contradiction suggests (R-EVAL-004).

When the user has confirmed that two records really contradict each other, :func:`propose_fixes`
works out which of them has to give way and what to do about it - and **only proposes**.  The
memory is changed by :class:`FixApplier`, one fix at a time, when the user says yes to that fix;
a fix he does not accept changes nothing.

Who gives way is :func:`~twin.eval.consistency_model.losers_of`: the lower source loses (R-MEM-004),
between equals the model's ``keep`` decides, and what the bot has already *said* stands if the
other side can be changed.  Only what the bot made up may be touched (R-MEM-011): an entry of its
life line, a fact whose source is ``bot_invented`` - never a fact from the real chat, one the user
said or one he wrote with ``/记住``.  A reply that was sent cannot be unsent; if it is the side
that gives way, the facts that were taken from it are what the proposal is about.

======================  ===========================================================================
``invalidate_fact``     the fact is kept as ``rejected`` (never shown, no longer current); the life
                        line entries and follow-ups that were made from it go with it
``rewrite_fact``        a new ``bot_invented`` fact with the corrected wording replaces it (the old
                        one keeps its row, ``superseded_by`` names the new one); the entries made
                        from the old one are replaced by entries of the new wording
``invalidate_lifeline``  the entry is marked invalidated (shown no more, still in the table)
``rewrite_lifeline``     the entry is invalidated and an ``improvised`` entry of the corrected
                        wording takes its time span, place and mood
======================  ===========================================================================

A fix whose record is not what it was when proposed - gone, replaced, invalidated, its wording
changed - is ``stale`` and does nothing.  Every change leaves an audit record with ids and counts,
never the text (the same ``memory_audit`` events the memory commands write).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from twin.eval.consistency_model import Statement, losers_of
from twin.eval.consistency_store import ConsistencyStore, FindingView, FixView, NewFix
from twin.memory.api import LIFELINE_ACTIVITY_CHARS, LifelineStore, Memory, NewFact, PlannedEvent
from twin.memory.records import FactRecord, LifelineRecord
from twin.ops.logging import get_logger
from twin.retrieval.embedder import EmbedderError

audit = get_logger("twin.memory.audit")
log = get_logger("twin.eval.consistency")

MAX_PROPOSALS = 3  # per contradiction: enough to cover both sides and what was made from them
INVALIDATE_FACT = "invalidate_fact"
REWRITE_FACT = "rewrite_fact"
INVALIDATE_LIFELINE = "invalidate_lifeline"
REWRITE_LIFELINE = "rewrite_lifeline"
ACTION_LABELS = {
    INVALIDATE_FACT: "把这条事实标记失效",
    REWRITE_FACT: "把这条事实改写",
    INVALIDATE_LIFELINE: "把这条生活安排标记失效",
    REWRITE_LIFELINE: "把这条生活安排改写",
}


class StaleFixError(Exception):
    """The record a fix is about is not what it was when the fix was proposed."""


@dataclass(frozen=True)
class FixOutcome:
    """What applying a fix came to: ``applied``, or ``stale`` (nothing was changed)."""

    fix: FixView
    status: str


def _fact_line(fact: FactRecord) -> str:
    return fact.text


def _event_line(event: LifelineRecord) -> str:
    return event.line()


def _fixes_for_loser(
    loser: Statement, memory: Memory, *, rewrite: str | None, reason: str
) -> list[NewFix]:
    corpus = memory.corpus
    if loser.kind == "lifeline":
        event = corpus.events.get(loser.item_id)
        if event is None or not event.active:
            return []
        action = REWRITE_LIFELINE if rewrite else INVALIDATE_LIFELINE
        return [NewFix(action, event.id, _event_line(event), rewrite, reason)]
    if loser.kind == "fact":
        fact = corpus.facts.get(loser.item_id)
        if fact is None or not fact.current or fact.source != "bot_invented":
            return []
        action = REWRITE_FACT if rewrite else INVALIDATE_FACT
        return [NewFix(action, fact.id, _fact_line(fact), rewrite, reason)]
    # a reply that was sent: what can be corrected is what the bot took from it
    said = set(loser.message_ids)
    derived = [
        fact
        for fact in corpus.facts.values()
        if fact.source == "bot_invented"
        and fact.current
        and said & {str(i) for i in (fact.evidence or {}).get("ids", [])}
    ]
    derived.sort(key=lambda fact: fact.number)
    return [NewFix(INVALIDATE_FACT, fact.id, _fact_line(fact), None, reason) for fact in derived]


def propose_fixes(finding: FindingView, memory: Memory) -> list[NewFix]:
    """The corrections a confirmed contradiction suggests (nothing is changed or stored here)."""
    memory.refresh()
    losers = losers_of(finding.first, finding.second, finding.keep)
    rewrite = finding.rewrite if len(losers) == 1 else None
    proposals: list[NewFix] = []
    seen: set[str] = set()
    for loser in losers:
        for fix in _fixes_for_loser(loser, memory, rewrite=rewrite, reason=finding.reason):
            if fix.target_id not in seen:
                seen.add(fix.target_id)
                proposals.append(fix)
    return proposals[:MAX_PROPOSALS]


class FixApplier:
    """Applies a proposed fix to the live memory - the only place the audit does (see above)."""

    def __init__(self, memory: Memory, store: ConsistencyStore) -> None:
        self._memory = memory
        self._store = store
        self._lifeline = LifelineStore(memory)

    def apply(self, fix_id: str) -> FixOutcome:
        """Make the change of a proposed fix; one that is not proposed any more is left as it is."""
        fix = self._store.fix(fix_id)
        if fix.status != "proposed":
            return FixOutcome(fix, fix.status)
        self._memory.refresh()
        try:
            new_id = self._change(fix)
        except StaleFixError:
            return FixOutcome(self._store.set_fix(fix_id, status="stale"), "stale")
        done = self._store.set_fix(fix_id, status="applied", new_id=new_id)
        self._memory.refresh()
        return FixOutcome(done, "applied")

    def decline(self, fix_id: str) -> FixOutcome:
        """The user said no: nothing is changed."""
        fix = self._store.fix(fix_id)
        if fix.status != "proposed":
            return FixOutcome(fix, fix.status)
        return FixOutcome(self._store.set_fix(fix_id, status="declined"), "declined")

    # ------------------------------------------------------------------------ the changes

    def _change(self, fix: FixView) -> str | None:
        """Make the change; returns the id of the record it made, if it made one."""
        if fix.action == INVALIDATE_FACT:
            self._invalidate_fact(fix)
            return None
        if fix.action == REWRITE_FACT:
            return self._rewrite_fact(fix)
        if fix.action == INVALIDATE_LIFELINE:
            self._invalidate_event(fix)
            return None
        if fix.action == REWRITE_LIFELINE:
            return self._rewrite_event(fix)
        raise StaleFixError(fix.action)

    def _fact_to_change(self, fix: FixView) -> FactRecord:
        fact = self._memory.corpus.facts.get(fix.target_id)
        if (
            fact is None
            or not fact.current
            or fact.source != "bot_invented"
            or _fact_line(fact) != fix.old_text
        ):
            raise StaleFixError(fix.target_id)
        return fact

    def _event_to_change(self, fix: FixView) -> LifelineRecord:
        event = self._memory.corpus.events.get(fix.target_id)
        if event is None or not event.active or _event_line(event) != fix.old_text:
            raise StaleFixError(fix.target_id)
        return event

    def _now(self) -> datetime:
        return self._memory.services.clock.now_utc()

    def _invalidate_fact(self, fix: FixView) -> None:
        fact = self._fact_to_change(fix)
        store = self._memory.store
        now = self._now()
        store.update_fact(fact.id, status="rejected")
        events = [e for e in store.events_of_fact([fact.id]) if e.active]
        for event in events:
            store.invalidate_event(event.id, by=fact.id, at=now)
        followups = [f for f in store.followups_of_fact([fact.id]) if f.is_open]
        for followup in followups:
            store.close_followup(followup.id, status="cancelled", at=now, reason="consistency")
        audit.info(
            "memory_audit",
            action="consistency_invalidate_fact",
            fact_ids=[fact.id],
            event_ids=[e.id for e in events],
            followup_ids=[f.id for f in followups],
        )

    def _rewrite_fact(self, fix: FixView) -> str:
        old = self._fact_to_change(fix)
        wording = fix.new_text
        if not wording:
            raise StaleFixError(fix.target_id)
        store = self._memory.store
        now = self._now()
        new = store.add_fact(
            NewFact(
                subject=old.subject,
                category=old.category,
                text=wording,
                source="bot_invented",
                known_at=now,
                confidence=old.confidence,
                importance=old.importance,
                valid_from=old.valid_from,
                valid_to=old.valid_to if old.valid_to is not None and old.valid_to > now else None,
                event_date=old.event_date,
                recurrence=old.recurrence,
                evidence={
                    **(old.evidence or {}),
                    "replaces": old.id,
                    "consistency_finding": fix.finding_id,
                },
            )
        )
        store.supersede_fact(old.id, new.id, at=now)
        events = [e for e in store.events_of_fact([old.id]) if e.active]
        for event in events:
            store.invalidate_event(event.id, by=new.id, at=now)
            self._lifeline.add_improvised(
                event.local_date,
                PlannedEvent(
                    wording[:LIFELINE_ACTIVITY_CHARS],
                    event.start_local,
                    event.end_local,
                    event.place,
                    event.mood,
                    wording,
                ),
                fact_id=new.id,
            )
        try:
            self._memory.sync_vectors(facts=[new])
        except EmbedderError:  # the next sync encodes every fact that has no vector yet
            log.warning("consistency_fix_without_vector")
        audit.info(
            "memory_audit",
            action="consistency_rewrite_fact",
            fact_ids=[old.id, new.id],
            event_ids=[e.id for e in events],
        )
        return new.id

    def _invalidate_event(self, fix: FixView) -> None:
        event = self._event_to_change(fix)
        self._lifeline.invalidate(event.id)
        audit.info("memory_audit", action="consistency_invalidate_lifeline", event_ids=[event.id])

    def _rewrite_event(self, fix: FixView) -> str:
        old = self._event_to_change(fix)
        wording = fix.new_text
        if not wording:
            raise StaleFixError(fix.target_id)
        self._lifeline.invalidate(old.id)
        new = self._lifeline.add_improvised(
            old.local_date,
            PlannedEvent(
                wording[:LIFELINE_ACTIVITY_CHARS],
                old.start_local,
                old.end_local,
                old.place,
                old.mood,
                None,
            ),
        )
        audit.info(
            "memory_audit", action="consistency_rewrite_lifeline", event_ids=[old.id, new.id]
        )
        return new.id
