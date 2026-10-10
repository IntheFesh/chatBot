"""Description vectors of the stickers, in a table of their own (R-STK-003, R-STO-005).

Each sticker of hers that has a description gets one vector made from its tags, its description,
its use cases and what the context correction found.  The vectors live in the LanceDB table
``sticker_descriptions`` (``data/vectors/``); the row id is the sticker's MD5, nothing else
about the sticker is stored there.  The selector compares the vector of the current context
with these (R-STK-004).

The table is tied to one encoding (model, size, weights): a sticker whose ``desc_encoding`` is
not the current one is encoded again, and a table made with another model is dropped and
rebuilt - vectors of two models are never mixed (R-RET-006).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import select

from twin.clock import to_epoch
from twin.ops.logging import get_logger
from twin.retrieval.embedder import EmbedderInfo, EmbeddingService, EncodeKind, embedding_service
from twin.retrieval.vector_store import IndexMeta, VectorStore, VectorTable
from twin.services import Services
from twin.stickers.catalog import StickerRecord, record_of
from twin.storage.chat_models import Sticker
from twin.storage.vector_schema import STICKER_SCHEMA, VectorKind

log = get_logger("twin.stickers.vectors")

SCHEME = "sticker-v1"
WRITE_CHUNK = 128


def encoding_of(info: EmbedderInfo) -> str:
    """The identity of an encoding of descriptions (stored on each sticker)."""
    return f"{info.model}|{info.dimension}|{info.weights_sha256[:16]}|{SCHEME}"


def sticker_table(services: Services) -> VectorTable:
    return VectorStore(services.paths.vectors_dir).table(STICKER_SCHEMA)


def description_text(record: StickerRecord) -> str:
    """What is encoded for a sticker: tags, description, use cases, her use of it."""
    parts = [
        "、".join(record.tags),
        record.description or "",
        record.use_cases or "",
        record.context_note or "",
    ]
    return "。".join(part.strip("。 ") for part in parts if part.strip())


@dataclass(frozen=True)
class VectorSync:
    encoded: int
    removed: int
    reset: bool


class StickerVectors:
    """Keeps the description vectors of the library up to date."""

    def __init__(self, services: Services, embedder: EmbeddingService | None = None) -> None:
        self._services = services
        self._embedder = embedder

    def embedder(self) -> EmbeddingService:
        if self._embedder is None:
            self._embedder = embedding_service(self._services)
        return self._embedder

    def table(self) -> VectorTable:
        return sticker_table(self._services)

    # --------------------------------------------------------------- syncing

    def _candidates(self, md5s: Sequence[str] | None) -> list[StickerRecord]:
        stmt = select(Sticker).where(Sticker.her_uses > 0, Sticker.description_ct.is_not(None))
        if md5s is not None:
            stmt = stmt.where(Sticker.md5.in_(list(md5s)))
        with self._services.db.session() as session:
            return [record_of(row) for row in session.scalars(stmt.order_by(Sticker.md5))]

    def sync(self, md5s: Sequence[str] | None = None) -> VectorSync:
        """Encode the stickers whose vector is missing or made with another encoding.

        The embedding model is only loaded when there is something to encode or the table
        was made with another model than the configured one.
        """
        table = self.table()
        meta = table.read_meta()
        if meta is not None and meta.model == self._services.settings.retrieval.model:
            pending = [r for r in self._candidates(md5s) if r.desc_encoding is None]
            if not pending:
                return VectorSync(0, self._drop_stale(table, md5s), False)
        service = self.embedder()
        info = service.info
        current = encoding_of(info)
        reset = meta is not None and meta.encoding != current
        if reset:
            table.drop()
            table.clear_meta()
            with self._services.db.transaction(bump_state=False) as session:
                for row in session.scalars(
                    select(Sticker).where(Sticker.desc_encoding.is_not(None))
                ):
                    row.desc_encoding = None
                    row.desc_vector_id = None
        records = [r for r in self._candidates(md5s) if r.desc_encoding != current]
        removed = self._drop_stale(table, md5s)
        if not records:
            return VectorSync(0, removed, reset)
        table.create(info.dimension)
        table.write_meta(
            IndexMeta(
                table=STICKER_SCHEMA.name,
                encoding=current,
                model=info.model,
                dimension=info.dimension,
                revision=info.revision,
                weights_sha256=info.weights_sha256,
                written_at=self._services.clock.now_utc().isoformat(),
            )
        )
        encoded = 0
        for start in range(0, len(records), WRITE_CHUNK):
            chunk = records[start : start + WRITE_CHUNK]
            vectors = service.encode([description_text(r) for r in chunk], EncodeKind.PASSAGE)
            rows = [
                {
                    "id": record.md5,
                    "vector": [float(x) for x in vector],
                    "at": int(to_epoch(record.first_used_at or self._services.clock.now_utc())),
                    "kind": VectorKind.STICKER.value,
                }
                for record, vector in zip(chunk, vectors, strict=True)
            ]
            table.upsert(rows, dimension=info.dimension)
            with self._services.db.transaction(bump_state=False) as session:
                for record in chunk:
                    found = session.get(Sticker, record.md5)
                    if found is not None:
                        found.desc_vector_id = record.md5
                        found.desc_encoding = current
            encoded += len(chunk)
        return VectorSync(encoded, removed, reset)

    def _drop_stale(self, table: VectorTable, md5s: Sequence[str] | None) -> int:
        """Remove the vectors of stickers that no longer have a description."""
        if md5s is not None or not table.exists():
            return 0
        with self._services.db.session() as session:
            keep = set(
                session.scalars(
                    select(Sticker.md5).where(
                        Sticker.her_uses > 0, Sticker.description_ct.is_not(None)
                    )
                )
            )
        stale = [row["id"] for row in table.rows() if row["id"] not in keep]
        table.delete_ids(stale)
        return len(stale)

    # --------------------------------------------------------------- reading

    def similarities(
        self, context_text: str, md5s: Sequence[str], *, at_or_before: int | None = None
    ) -> dict[str, float]:
        """Cosine similarity of the context with the description of each sticker that has one."""
        table = self.table()
        if not md5s or not table.exists() or table.count() == 0:
            return {}
        meta = table.read_meta()
        if meta is None or meta.encoding != encoding_of(self.embedder().info):
            log.warning(
                "sticker_vectors_made_with_another_model"
            )  # `twin stickers tag-all` redoes them
            return {}
        query: NDArray[np.float32] = self.embedder().encode_one(context_text, EncodeKind.QUERY)
        wanted = set(md5s)
        hits = table.search(query, table.count(), at_or_before=at_or_before)
        return {hit.id: hit.similarity for hit in hits if hit.id in wanted}
