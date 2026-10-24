"""``memory_view(as_of=t)``: the memory as it was known at a moment (R-MEM-010, R-TRN-013).

A :class:`MemoryView` answers what the bot could have known at ``t``: it holds the corpus and the
moment, and every answer passes :mod:`twin.memory.visible`.  There is no method that returns
anything unfiltered, and the moment cannot be changed after the view is made.

* :meth:`MemoryView.facts` - facts with ``known_at < t`` that are valid and not yet replaced at
  ``t``, shown as they stood then;
* :meth:`MemoryView.summaries` - summaries of local days earlier than the local day of ``t``;
* :meth:`MemoryView.followups` - follow-ups created before ``t`` and still open at ``t`` (shown
  open, whatever became of them later);
* :meth:`MemoryView.lifeline` - the life line, empty before the bot's conversation began;
* :meth:`MemoryView.render` - the memory block the model would have been given at ``t``.

The live memory block is built from the view of *now* (the assembler does not have a second
set of rules), so what a training sample shows and what the running bot shows are produced by the
same code.

One thing differs, on purpose (D-731): a vector search that fails.  The running bot goes on with the
keyword results and with everything that needs no vectors - the follow-ups, the facts of today and
tomorrow, the life line, the recent days - and notes that the view is ``vectors_degraded``
(``degrade_vectors``, which only the live assembler turns on).  A view made for the training export
or the evaluation does not degrade: it raises, because a block that silently lacks what the vectors
would have found is not the block of that moment (iron rule 13).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import floor
from typing import Protocol
from zoneinfo import ZoneInfo

from twin.clock import Clock, ensure_aware, to_epoch
from twin.memory.blocks import MemoryBlock, MemoryQuery
from twin.memory.memory import Memory
from twin.memory.records import FactRecord, FollowupRecord, LifelineRecord, SummaryRecord
from twin.memory.visible import (
    BotEra,
    fact_visible,
    followup_visible,
    lifeline_visible,
    project_fact,
    project_followup,
    summary_visible,
)
from twin.ops.logging import get_logger
from twin.retrieval.vector_store import VectorHit
from twin.storage.vector_schema import VectorKind

log = get_logger("twin.memory.view")

VECTOR_OVERFETCH = 3  # vector hits asked for per wanted hit, as some are not visible
MIN_KEYWORD_SCORE = 0.2  # a keyword hit must hold this share of the query's weight
VECTOR_RETRY_AFTER_S = 300.0  # a vector backend that failed is not asked again for this long


@dataclass(frozen=True)
class FactHit:
    """A visible fact found for a topic: its closeness by meaning and by keyword."""

    fact: FactRecord
    similarity: float  # cosine similarity of the vectors (0 if only the keywords found it)
    keyword: float  # share of the query's keyword weight the fact has (0 if only the vector did)

    @property
    def relevance(self) -> float:
        return max(self.similarity, self.keyword)


@dataclass(frozen=True)
class SummaryHit:
    summary: SummaryRecord
    similarity: float
    keyword: float

    @property
    def relevance(self) -> float:
        return max(self.similarity, self.keyword)


class VectorOutage:
    """What the live memory remembers of a vector backend that failed (D-731).

    A model that cannot be loaded or a table that cannot be read fails the same way on the next
    reply, and a failing model load is slow: after one failure the backend is not asked again for
    ``retry_after_s`` (five minutes), the views of that time are degraded at once, and the first
    request after it tries the backend again.  One instance lives with the live assembler; the
    views of the training export and the evaluation never get one.
    """

    def __init__(self, clock: Clock, retry_after_s: float = VECTOR_RETRY_AFTER_S) -> None:
        self._clock = clock
        self._retry_after_s = retry_after_s
        self._failed_at: float | None = None

    def blocked(self) -> bool:
        failed_at = self._failed_at
        return failed_at is not None and self._clock.monotonic() - failed_at < self._retry_after_s

    def note_failure(self) -> None:
        self._failed_at = self._clock.monotonic()

    def note_success(self) -> None:
        self._failed_at = None


class ViewRenderer(Protocol):
    """Builds the memory block of a view (the assembler; injected by :func:`memory_view`)."""

    def __call__(
        self, view: MemoryView, query: MemoryQuery, budget_tokens: int | None
    ) -> MemoryBlock: ...


class MemoryView:
    """The memory at one moment (see the module description)."""

    def __init__(
        self,
        memory: Memory,
        as_of: datetime,
        *,
        include_lifeline: bool = True,
        refresh: bool = True,
        renderer: ViewRenderer | None = None,
        degrade_vectors: bool = False,
        outage: VectorOutage | None = None,
    ) -> None:
        self._memory = memory
        self._as_of = ensure_aware(as_of)
        self._include_lifeline = include_lifeline
        self._renderer = renderer
        self._degrade_vectors = degrade_vectors
        self._outage = outage
        self._vectors_degraded = False
        if refresh:
            memory.refresh()
        self._era: BotEra = memory.era()

    # ------------------------------------------------------------------ the moment

    @property
    def as_of(self) -> datetime:
        return self._as_of

    @property
    def bot_era(self) -> bool:
        """True if the bot's conversation had begun before this moment."""
        return self._era.reached(self._as_of)

    @property
    def scope(self) -> str:
        """``bot`` when the moment is in the bot's era, else ``real`` (whose calendar dates it)."""
        return "bot" if self.bot_era else "real"

    def today(self) -> date:
        """The local date of the moment (hers before the bot's era, the bot's after)."""
        return self._memory.clock.date_of(self.scope, self._as_of)

    def zone(self) -> ZoneInfo:
        return self._memory.clock.zone_of(self.scope, self._as_of)

    @property
    def vectors_degraded(self) -> bool:
        """True if a vector search of this view failed: its answers lack the vector hits."""
        return self._vectors_degraded

    @property
    def vector_bound(self) -> int:
        """Largest epoch second a vector row may carry to be visible (``at_or_before``)."""
        return floor(to_epoch(self._as_of))

    # --------------------------------------------------------------------- facts

    def fact_visible(self, fact_id: str) -> bool:
        fact = self._memory.corpus.facts.get(fact_id)
        return fact is not None and fact_visible(fact, self._as_of, self._era)

    def fact(self, fact_id: str) -> FactRecord | None:
        """The fact as it stood at the moment, or ``None`` if it was not (yet) there."""
        fact = self._memory.corpus.facts.get(fact_id)
        if fact is None or not fact_visible(fact, self._as_of, self._era):
            return None
        return project_fact(fact, self._as_of)

    def facts(self) -> tuple[FactRecord, ...]:
        found = [
            project_fact(fact, self._as_of)
            for fact in self._memory.corpus.facts.values()
            if fact_visible(fact, self._as_of, self._era)
        ]
        found.sort(key=lambda f: (f.known_at, f.number))
        return tuple(found)

    def dated_facts(self) -> tuple[FactRecord, ...]:
        """Visible facts that name a calendar day."""
        return tuple(
            project_fact(fact, self._as_of)
            for fact in self._memory.corpus.dated_facts()
            if fact_visible(fact, self._as_of, self._era)
        )

    def core_facts(self) -> tuple[FactRecord, ...]:
        """Visible facts of the highest importance."""
        return tuple(
            project_fact(fact, self._as_of)
            for fact in self._memory.corpus.core_facts()
            if fact_visible(fact, self._as_of, self._era)
        )

    # ----------------------------------------------------------------- summaries

    def summary_visible(self, summary: SummaryRecord) -> bool:
        return summary_visible(summary, self._as_of, self._era, self._memory.clock)

    def summary_for(self, scope: str, day: date) -> SummaryRecord | None:
        summary = self._memory.corpus.summary_of(scope, day)
        return summary if summary is not None and self.summary_visible(summary) else None

    def _summary_known(self, summary_id: str) -> bool:
        return self.summary(summary_id) is not None

    def summary(self, summary_id: str) -> SummaryRecord | None:
        summary = self._memory.corpus.summaries.get(summary_id)
        return summary if summary is not None and self.summary_visible(summary) else None

    def summaries(self) -> tuple[SummaryRecord, ...]:
        found = [s for s in self._memory.corpus.summaries.values() if self.summary_visible(s)]
        found.sort(key=lambda s: (s.local_date, s.scope))
        return tuple(found)

    # ----------------------------------------------------------------- follow-ups

    def followups(self) -> tuple[FollowupRecord, ...]:
        found = [
            project_followup(f, self._as_of)
            for f in self._memory.corpus.followups.values()
            if followup_visible(f, self._as_of, self._era)
        ]
        found.sort(key=lambda f: (f.due_at, f.created_at))
        return tuple(found)

    # ------------------------------------------------------------------ life line

    def lifeline(self) -> tuple[LifelineRecord, ...]:
        if not self._include_lifeline:
            return ()
        found = [
            e
            for e in self._memory.corpus.events.values()
            if lifeline_visible(e, self._as_of, self._era)
        ]
        found.sort(key=lambda e: (e.local_date, e.start_local or "", e.created_at))
        return tuple(found)

    # ------------------------------------------------------------------ rendering

    def render(self, query: MemoryQuery, budget_tokens: int | None = None) -> MemoryBlock:
        """The memory block built from this view only (R-MEM-008)."""
        if self._renderer is None:
            raise RuntimeError("this view was made without a renderer; use memory_view()")
        return self._renderer(self, query, budget_tokens)

    # -------------------------------------------------------------------- searching

    def _vector_hits(self, kind: VectorKind, text: str, limit: int) -> list[VectorHit]:
        """The vector search, bounded by this view; only this call is guarded (D-731).

        A view made with ``degrade_vectors`` turns a failure of the backend into "no vector hits"
        (the keyword search that follows still answers) and marks itself degraded; any other view
        lets the error through.  Only the search itself is covered: a bug in what is done with the
        hits is not hidden.
        """
        vectors = self._memory.vectors
        wanted = limit * VECTOR_OVERFETCH
        if not self._degrade_vectors:
            return vectors.search(kind, text, wanted, at_or_before=self.vector_bound)
        outage = self._outage
        if outage is not None and outage.blocked():
            self._vectors_degraded = True
            return []
        try:
            hits = vectors.search(kind, text, wanted, at_or_before=self.vector_bound)
        except Exception as exc:  # the backend may fail in any way (model, disk, library)
            self._vectors_degraded = True
            if outage is not None:
                outage.note_failure()
            log.warning("memory_vectors_unavailable", kind=kind.value, reason=type(exc).__name__)
            return []
        if outage is not None:
            outage.note_success()
        return hits

    def search_facts(self, text: str, limit: int) -> list[FactHit]:
        """Visible facts that fit ``text`` by meaning or by keyword, best first.

        The two searches are bounded by this view *before* their limits apply, so a fact of the
        future never takes a place a visible fact should have.
        """
        if limit <= 0 or not text.strip():
            return []
        memory = self._memory
        floor = memory.services.settings.memory.recall_min_similarity
        found: dict[str, tuple[float, float]] = {}
        for hit in self._vector_hits(VectorKind.FACT, text, limit):
            if hit.similarity >= floor and self.fact_visible(hit.id):
                found[hit.id] = (hit.similarity, 0.0)
        for kw in memory.corpus.fact_index.search(text, limit, allowed=self.fact_visible):
            if kw.score >= MIN_KEYWORD_SCORE:
                similarity, _ = found.get(kw.doc_id, (0.0, 0.0))
                found[kw.doc_id] = (similarity, kw.score)
        hits = [
            FactHit(record, similarity, keyword)
            for fact_id, (similarity, keyword) in found.items()
            if (record := self.fact(fact_id)) is not None
        ]
        hits.sort(key=lambda h: (-h.relevance, h.fact.number))
        return hits[:limit]

    def search_summaries(self, text: str, limit: int) -> list[SummaryHit]:
        """Visible summaries that fit ``text``, best first."""
        if limit <= 0 or not text.strip():
            return []
        memory = self._memory
        floor = memory.services.settings.memory.recall_min_similarity
        found: dict[str, tuple[float, float]] = {}
        for hit in self._vector_hits(VectorKind.SUMMARY, text, limit):
            if hit.similarity >= floor and self.summary(hit.id) is not None:
                found[hit.id] = (hit.similarity, 0.0)
        for kw in memory.corpus.summary_index.search(text, limit, allowed=self._summary_known):
            if kw.score >= MIN_KEYWORD_SCORE:
                similarity, _ = found.get(kw.doc_id, (0.0, 0.0))
                found[kw.doc_id] = (similarity, kw.score)
        hits = [
            SummaryHit(record, similarity, keyword)
            for summary_id, (similarity, keyword) in found.items()
            if (record := self.summary(summary_id)) is not None
        ]
        hits.sort(key=lambda h: (-h.relevance, h.summary.local_date))
        return hits[:limit]
