"""The memory as one object: store, snapshot, vectors and the calendar (round 07).

``Memory(services)`` is what the modules above the store hold on to.  It wires the pieces that
belong together - :class:`~twin.memory.store.MemoryStore` (the tables),
:class:`~twin.memory.corpus.MemoryCorpus` (the snapshot with the keyword index),
:class:`~twin.memory.vectors.MemoryVectors` (the two vector tables) and
:class:`~twin.memory.localdate.MemoryClock` (local days) - and keeps them consistent after a
write: :meth:`Memory.sync_vectors` encodes what is new and, when the embedding model has changed
since, everything that was made with the old one.

One ``Memory`` per process and caller is plenty; building it reads nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from twin.memory.corpus import MemoryCorpus
from twin.memory.localdate import MemoryClock
from twin.memory.records import FactRecord, SummaryRecord
from twin.memory.store import MemoryStore
from twin.memory.vectors import MemoryVectors, VectorSync
from twin.memory.visible import BotEra
from twin.retrieval.embedder import EmbeddingService
from twin.schedule.time_service import TimeService
from twin.services import Services
from twin.storage.vector_schema import VectorKind


class Memory:
    """Store, corpus, vectors and calendar of the memory (see the module description)."""

    def __init__(
        self,
        services: Services,
        *,
        embedder: EmbeddingService | None = None,
        time_service: TimeService | None = None,
    ) -> None:
        self.services = services
        self.store = MemoryStore(services)
        self.corpus = MemoryCorpus(self.store)
        self.vectors = MemoryVectors(services, self.store, embedder)
        self.clock = MemoryClock.from_services(services, time_service)

    def refresh(self) -> bool:
        """Bring the snapshot up to date with the tables (cheap when nothing changed)."""
        return self.corpus.refresh()

    def era(self) -> BotEra:
        return self.corpus.era()

    def note_bot_activity(self, at: datetime) -> None:
        """Remember that the bot's conversation exists from this moment on (R-MEM-010)."""
        self.store.mark_bot_online(at)

    def sync_vectors(
        self,
        *,
        facts: Sequence[FactRecord] = (),
        summaries: Sequence[SummaryRecord] = (),
    ) -> VectorSync:
        """Encode the given records and every record whose vector is not in the current table.

        It is called after something new was stored, so it loads the embedding model.  Besides
        the given records it encodes every record without a vector of the configured model's
        encoding (``embed_version``) - and, when a table is missing or was made with another
        model, every record of that kind: the first sync after a model switch drops the old
        table and the old records must not be left without a vector (D-781).
        :meth:`reindex` redoes everything on purpose.
        """
        self.refresh()
        vectors = self.vectors
        encoding = vectors.encoding()
        wanted_facts = {f.id: f for f in facts}
        wanted_summaries = {s.id: s for s in summaries}
        for fact in vectors.pending_facts(
            list(self.corpus.facts.values()),
            encoding,
            table_current=vectors.made_with(VectorKind.FACT, encoding),
        ):
            wanted_facts.setdefault(fact.id, fact)
        for summary in vectors.pending_summaries(
            list(self.corpus.summaries.values()),
            encoding,
            table_current=vectors.made_with(VectorKind.SUMMARY, encoding),
        ):
            wanted_summaries.setdefault(summary.id, summary)
        first = self.vectors.sync_facts(list(wanted_facts.values()))
        second = self.vectors.sync_summaries(list(wanted_summaries.values()))
        if first.encoded or second.encoded:
            self.refresh()
        return VectorSync(first.encoded + second.encoded, 0, first.reset or second.reset)

    def reindex(self) -> VectorSync:
        """Drop both vector tables and encode every fact and summary again."""
        self.vectors.rebuild_reset()
        self.refresh()
        first = self.vectors.sync_facts(list(self.corpus.facts.values()))
        second = self.vectors.sync_summaries(list(self.corpus.summaries.values()))
        self.refresh()
        return VectorSync(first.encoded + second.encoded, 0, True)
