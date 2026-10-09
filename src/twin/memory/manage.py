"""The user's commands on the memory: remember, forget, list (R-MEM-009).

Round 11 turns ``/记住 <内容>``, ``/忘掉 <内容或编号>`` and ``/记忆 [页码|关键词]`` into calls of
:class:`MemoryManager`; ``twin memory`` offers the same on the command line.

``remember(text)``
    stores ``text`` as a fact with source ``user_command``, the highest confidence and a rank
    above every other source (R-MEM-004).  The text is run through the extractor first, so the
    fact gets its subject, category and, for "她下周三生日", its date; a follow-up in it is kept.
    If no model is at hand, or it finds nothing, the text is stored as it is - the command never
    loses what the user asked to remember.
``forget(query_or_id)``
    deletes for good the fact named by its number, its id or by words that match exactly one
    fact, together with everything **derived only from it**: the follow-ups and life line entries
    made from it, its vector.  Facts it had replaced become current again.  Words that match
    several facts delete nothing and return the candidates, unless ``all_matches`` is set.
``list_items(page, keyword)``
    the current facts, newest first, ten to a page, optionally those that contain a word.

Every command writes an audit record - the action, ids and counts, **never the text**.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import date, datetime

from twin.config.secrets import SecretStoreError
from twin.llm.deepseek import DeepSeekClient
from twin.llm.errors import LlmError
from twin.llm.types import DAILY
from twin.memory.extract import DialogueLine, Extraction, FactDraft, FactExtractor
from twin.memory.memory import Memory
from twin.memory.records import FactRecord, FollowupRecord
from twin.memory.writer import MemoryWriter
from twin.ops.logging import get_logger
from twin.profile.prompt_templates import TemplateStore
from twin.storage.ids import is_valid_id, new_id
from twin.storage.vector_schema import VectorKind

audit = get_logger("twin.memory.audit")
log = get_logger("twin.memory.manage")

PAGE_SIZE = 10
MIN_KEYWORD_SCORE = 0.6
COMMAND_IMPORTANCE = 4
MAX_COMMAND_CHARS = 400
MAX_CANDIDATES = 8


@dataclass(frozen=True)
class MemoryItem:
    """A fact as listed to the user."""

    number: int
    id: str
    subject: str
    category: str
    text: str
    source: str
    known_at: datetime
    importance: int
    event_date: date | None

    @classmethod
    def of(cls, fact: FactRecord) -> MemoryItem:
        return cls(
            fact.number,
            fact.id,
            fact.subject,
            fact.category,
            fact.text,
            fact.source,
            fact.known_at,
            fact.importance,
            fact.event_date,
        )


@dataclass(frozen=True)
class MemoryPage:
    items: tuple[MemoryItem, ...]
    page: int
    pages: int
    total: int
    keyword: str | None = None
    followups: tuple[str, ...] = ()  # open follow-ups, listed on the first page


@dataclass(frozen=True)
class RememberResult:
    facts: tuple[MemoryItem, ...]
    followups: int
    enriched: bool  # the extractor understood the text (False: stored as it was written)
    replaced: int = 0  # older facts the new one replaced


@dataclass(frozen=True)
class DeletedItem:
    """One thing a forget removed (the text is returned to the caller, never logged)."""

    kind: str  # fact | followup | lifeline
    id: str
    number: int | None
    text: str


@dataclass(frozen=True)
class ForgetResult:
    deleted: tuple[DeletedItem, ...] = ()
    restored: tuple[int, ...] = ()  # numbers of facts that became current again
    ambiguous: tuple[MemoryItem, ...] = ()  # several matches; nothing was deleted
    followup_matches: tuple[str, ...] = field(default_factory=tuple)

    @property
    def found(self) -> bool:
        return bool(self.deleted or self.ambiguous)


class MemoryManager:
    """``remember`` / ``forget`` / ``list_items`` (see the module description)."""

    def __init__(self, memory: Memory, client: DeepSeekClient | None = None) -> None:
        self._memory = memory
        self._client = client
        services = memory.services
        self._templates = TemplateStore(services.db, services.clock)

    # ------------------------------------------------------------------ remember

    async def remember(self, text: str, *, now: datetime | None = None) -> RememberResult:
        """Store ``text`` as a fact of the highest rank (see the module description)."""
        wording = " ".join(text.split())[:MAX_COMMAND_CHARS]
        if not wording:
            raise ValueError("there is nothing to remember")
        moment = now or self._memory.services.clock.now_utc()
        ref = f"cmd-{new_id()}"
        extraction = await self._understand(wording, ref, moment)
        enriched = bool(extraction.facts)
        if not enriched:
            extraction.facts.append(
                FactDraft(
                    subject="both",
                    category="other",
                    text=wording,
                    importance=COMMAND_IMPORTANCE,
                    confidence=1.0,
                    source="user_command",
                    known_at=moment,
                    evidence_kind="command",
                    evidence_refs=(ref,),
                )
            )
        writer = MemoryWriter(self._memory, self._client, templates=self._templates)
        report = await writer.write(extraction, DAILY)
        self._memory.store.mark_bot_online(moment)
        facts = tuple(MemoryItem.of(f) for f in self._memory.store.facts(report.inserted))
        audit.info(
            "memory_audit",
            action="remember",
            fact_ids=report.inserted,
            replaced=report.superseded,
            followups=report.followups_added,
            enriched=enriched,
        )
        return RememberResult(facts, report.followups_added, enriched, report.superseded)

    async def _understand(self, text: str, ref: str, moment: datetime) -> Extraction:
        services = self._memory.services
        if self._client is None or not services.settings.memory.fact_extraction:
            return Extraction()
        extractor = FactExtractor(
            services, self._client, self._memory.clock, templates=self._templates
        )
        try:
            return await extractor.extract(
                [DialogueLine(ref, "user", text, moment)], mode="command", tag=DAILY
            )
        except (LlmError, SecretStoreError) as exc:
            log.warning("remember_not_understood", error=type(exc).__name__)
            return Extraction()

    # ------------------------------------------------------------------- forget

    def forget(self, query_or_id: str, *, all_matches: bool = False) -> ForgetResult:
        """Delete the fact named by number, id or words, and what was derived only from it."""
        query = query_or_id.strip()
        if not query:
            raise ValueError("say which memory to forget: a number or some words")
        self._memory.refresh()
        matches = self._match(query)
        followups = [] if matches else self._matching_followups(query)
        if not matches and followups:
            return self._forget_followups(followups, all_matches)
        if not matches:
            audit.info("memory_audit", action="forget", matched=0)
            return ForgetResult()
        if len(matches) > 1 and not all_matches:
            audit.info("memory_audit", action="forget", matched=len(matches), deleted=0)
            return ForgetResult(ambiguous=tuple(MemoryItem.of(f) for f in matches[:MAX_CANDIDATES]))
        return self._delete(matches)

    def _match(self, query: str) -> list[FactRecord]:
        corpus = self._memory.corpus
        if query.isdigit():
            number = int(query)
            return [f for f in corpus.facts.values() if f.number == number]
        if is_valid_id(query) and query in corpus.facts:
            return [corpus.facts[query]]
        current = [f for f in corpus.facts.values() if f.current]
        exact = sorted((f for f in current if query in f.text), key=lambda f: f.number)
        if exact:
            return exact
        hits = corpus.fact_index.search(query, MAX_CANDIDATES)
        return [
            corpus.facts[h.doc_id]
            for h in hits
            if h.score >= MIN_KEYWORD_SCORE and corpus.facts[h.doc_id].current
        ]

    def _matching_followups(self, query: str) -> list[FollowupRecord]:
        found = [f for f in self._memory.corpus.followups.values() if f.is_open and query in f.text]
        return sorted(found, key=lambda f: f.due_at)

    def _forget_followups(self, followups: list[FollowupRecord], all_matches: bool) -> ForgetResult:
        if len(followups) > 1 and not all_matches:
            audit.info("memory_audit", action="forget", matched=len(followups), deleted=0)
            return ForgetResult(followup_matches=tuple(f.text for f in followups[:MAX_CANDIDATES]))
        self._memory.store.delete_followups([f.id for f in followups])
        self._memory.refresh()
        audit.info(
            "memory_audit", action="forget", kind="followup", followup_ids=[f.id for f in followups]
        )
        return ForgetResult(
            deleted=tuple(DeletedItem("followup", f.id, None, f.text) for f in followups)
        )

    def _delete(self, facts: list[FactRecord]) -> ForgetResult:
        store = self._memory.store
        ids = [f.id for f in facts]
        followups = store.followups_of_fact(ids)
        events = store.events_of_fact(ids)
        deleted: list[DeletedItem] = [DeletedItem("fact", f.id, f.number, f.text) for f in facts]
        deleted += [DeletedItem("followup", f.id, None, f.text) for f in followups]
        deleted += [DeletedItem("lifeline", e.id, None, e.activity) for e in events]
        store.delete_followups([f.id for f in followups])
        store.delete_events([e.id for e in events])
        restored_ids = store.delete_facts(ids)
        self._memory.vectors.remove(VectorKind.FACT, ids)
        self._memory.refresh()
        restored = tuple(sorted(f.number for f in store.facts(restored_ids)))
        audit.info(
            "memory_audit",
            action="forget",
            fact_ids=ids,
            followup_ids=[f.id for f in followups],
            event_ids=[e.id for e in events],
            restored=len(restored_ids),
        )
        return ForgetResult(tuple(deleted), restored)

    async def aforget(self, query_or_id: str, *, all_matches: bool = False) -> ForgetResult:
        return await asyncio.to_thread(self.forget, query_or_id, all_matches=all_matches)

    # ---------------------------------------------------------------------- list

    def list_items(
        self, page: int = 1, keyword: str | None = None, *, page_size: int = PAGE_SIZE
    ) -> MemoryPage:
        """The current facts, newest first, ``page_size`` to a page; ``keyword`` narrows them."""
        self._memory.refresh()
        corpus = self._memory.corpus
        facts = [f for f in corpus.facts.values() if f.current]
        word = (keyword or "").strip() or None
        if word:
            facts = [f for f in facts if word in f.text]
        facts.sort(key=lambda f: -f.number)
        pages = max(1, math.ceil(len(facts) / page_size))
        chosen = min(max(1, page), pages)
        window = facts[(chosen - 1) * page_size : chosen * page_size]
        open_followups = (
            tuple(
                f.text
                for f in sorted(corpus.followups.values(), key=lambda f: f.due_at)
                if f.is_open
            )
            if chosen == 1 and word is None
            else ()
        )
        return MemoryPage(
            tuple(MemoryItem.of(f) for f in window), chosen, pages, len(facts), word, open_followups
        )
