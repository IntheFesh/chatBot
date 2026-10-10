"""Vectors of the facts and of the summaries, in tables of their own (R-MEM-008, R-STO-005).

``memory_facts`` holds one vector per fact and ``memory_summaries`` one per current daily
summary (:data:`~twin.storage.vector_schema.FACT_SCHEMA`,
:data:`~twin.storage.vector_schema.SUMMARY_SCHEMA`, below ``data/vectors/``).  A row is the id of
the fact or summary, its vector, the moment it became known and its kind: no text and no other
metadata.  The text stays in the encrypted database and is looked up by id.

A table is tied to one encoding (model, size, weights, scheme).  A record remembers the encoding
of its vector (``embed_version``); a record whose vector was made with another encoding is
encoded again, and a table made with another model is dropped first, so vectors of two models
are never mixed (R-RET-006).
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass

from twin.clock import to_epoch
from twin.memory.records import FactRecord, SummaryRecord
from twin.memory.store import MemoryStore
from twin.ops.logging import get_logger
from twin.retrieval.embedder import EmbedderInfo, EmbeddingService, EncodeKind, embedding_service
from twin.retrieval.vector_store import IndexMeta, VectorHit, VectorStore, VectorTable
from twin.services import Services
from twin.storage.vector_schema import FACT_SCHEMA, SUMMARY_SCHEMA, VectorKind, VectorTableSchema

log = get_logger("twin.memory.vectors")

SCHEME = "memory-v1"
WRITE_CHUNK = 128
# Jobs run side by side in one process (the replay of two weeks next to an extraction); creating
# a table and merging rows into it is not safe to do from two threads at once.
_WRITE_LOCK = threading.RLock()


def encoding_of(info: EmbedderInfo) -> str:
    """The identity of an encoding of memory texts (stored on each record)."""
    return f"{info.model}|{info.dimension}|{info.weights_sha256[:16]}|{SCHEME}"


@dataclass(frozen=True)
class VectorSync:
    encoded: int
    removed: int
    reset: bool


class MemoryVectors:
    """Keeps the two vector tables of the memory up to date and searches them."""

    def __init__(
        self,
        services: Services,
        store: MemoryStore,
        embedder: EmbeddingService | None = None,
    ) -> None:
        self._services = services
        self._store = store
        self._embedder = embedder

    def embedder(self) -> EmbeddingService:
        if self._embedder is None:
            self._embedder = embedding_service(self._services)
        return self._embedder

    def table(self, kind: VectorKind) -> VectorTable:
        return VectorStore(self._services.paths.vectors_dir).table(_schema_of(kind))

    # ---------------------------------------------------------------- syncing

    def pending_facts(self, facts: Sequence[FactRecord], encoding: str) -> list[FactRecord]:
        return [f for f in facts if f.status == "active" and f.embed_version != encoding]

    def pending_summaries(
        self, summaries: Sequence[SummaryRecord], encoding: str
    ) -> list[SummaryRecord]:
        return [s for s in summaries if s.is_current and s.embed_version != encoding]

    def _prepare(self, kind: VectorKind) -> tuple[VectorTable, EmbedderInfo, str, bool]:
        """The table for ``kind`` made ready for the configured model (dropped if it differs)."""
        service = self.embedder()
        info = service.info
        current = encoding_of(info)
        table = self.table(kind)
        meta = table.read_meta()
        reset = meta is not None and meta.encoding != current
        if reset:
            table.drop()
            table.clear_meta()
            log.warning("memory_vectors_reset", kind=kind.value)
        table.create(info.dimension)
        if table.read_meta() is None:
            table.write_meta(
                IndexMeta(
                    table=table.schema.name,
                    encoding=current,
                    model=info.model,
                    dimension=info.dimension,
                    revision=info.revision,
                    weights_sha256=info.weights_sha256,
                    written_at=self._services.clock.now_utc().isoformat(),
                )
            )
        return table, info, current, reset

    def sync_facts(self, facts: Sequence[FactRecord]) -> VectorSync:
        """Encode these facts (those that are shown by the model's current vectors are kept)."""
        if not facts:
            return VectorSync(0, 0, False)
        with _WRITE_LOCK:
            return self._sync_facts(facts)

    def _sync_facts(self, facts: Sequence[FactRecord]) -> VectorSync:
        table, info, current, reset = self._prepare(VectorKind.FACT)
        service = self.embedder()
        encoded = 0
        for start in range(0, len(facts), WRITE_CHUNK):
            chunk = facts[start : start + WRITE_CHUNK]
            vectors = service.encode([fact.text for fact in chunk], EncodeKind.PASSAGE)
            rows = [
                {
                    "id": fact.id,
                    "vector": [float(x) for x in vector],
                    "at": int(to_epoch(fact.known_at)),
                    "kind": VectorKind.FACT.value,
                }
                for fact, vector in zip(chunk, vectors, strict=True)
            ]
            table.upsert(rows, dimension=info.dimension)
            self._store.set_fact_embedding({fact.id: (fact.id, current) for fact in chunk})
            encoded += len(chunk)
        return VectorSync(encoded, 0, reset)

    def sync_summaries(self, summaries: Sequence[SummaryRecord]) -> VectorSync:
        if not summaries:
            return VectorSync(0, 0, False)
        with _WRITE_LOCK:
            return self._sync_summaries(summaries)

    def _sync_summaries(self, summaries: Sequence[SummaryRecord]) -> VectorSync:
        table, info, current, reset = self._prepare(VectorKind.SUMMARY)
        service = self.embedder()
        encoded = 0
        for start in range(0, len(summaries), WRITE_CHUNK):
            chunk = summaries[start : start + WRITE_CHUNK]
            vectors = service.encode([s.text for s in chunk], EncodeKind.PASSAGE)
            rows = [
                {
                    "id": summary.id,
                    "vector": [float(x) for x in vector],
                    "at": int(to_epoch(summary.utc_end)),
                    "kind": VectorKind.SUMMARY.value,
                }
                for summary, vector in zip(chunk, vectors, strict=True)
            ]
            table.upsert(rows, dimension=info.dimension)
            self._store.set_summary_embedding({s.id: (s.id, current) for s in chunk})
            encoded += len(chunk)
        return VectorSync(encoded, 0, reset)

    def remove(self, kind: VectorKind, ids: Sequence[str]) -> None:
        """Drop the vectors of deleted facts or of summaries that are no longer current."""
        if ids:
            with _WRITE_LOCK:
                self.table(kind).delete_ids(list(ids))

    # ---------------------------------------------------------------- reading

    def usable(self, kind: VectorKind) -> bool:
        """True if the table exists and was made with the configured model."""
        table = self.table(kind)
        if not table.exists() or table.count() == 0:
            return False
        meta = table.read_meta()
        return meta is not None and meta.encoding == encoding_of(self.embedder().info)

    def search(
        self,
        kind: VectorKind,
        text: str,
        limit: int,
        *,
        at_or_before: int | None = None,
        encode_as: EncodeKind = EncodeKind.QUERY,
    ) -> list[VectorHit]:
        """The ``limit`` records nearest to ``text`` (cosine), oldest-bound by ``at_or_before``."""
        if limit <= 0 or not text.strip() or not self.usable(kind):
            return []
        vector = self.embedder().encode_one(text, encode_as)
        return self.table(kind).search(vector, limit, at_or_before=at_or_before)

    def rebuild_reset(self) -> None:
        """Forget every vector (the next sync encodes everything again)."""
        with _WRITE_LOCK:
            for kind in (VectorKind.FACT, VectorKind.SUMMARY):
                table = self.table(kind)
                table.drop()
                table.clear_meta()


def _schema_of(kind: VectorKind) -> VectorTableSchema:
    if kind is VectorKind.FACT:
        return FACT_SCHEMA
    if kind is VectorKind.SUMMARY:
        return SUMMARY_SCHEMA
    raise ValueError(f"the memory has no vector table for {kind.value}")
