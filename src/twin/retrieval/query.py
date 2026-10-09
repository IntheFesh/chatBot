"""Asking the library: "how did she answer in a situation like this?" (R-RET-005, R-TRN-013).

The query is the last few turns of the conversation - in the bot's conversation the user's
messages and the bot's replies both count as context - the current clock time and day type, and
optionally ``before``.  The answer is a list of :class:`~twin.retrieval.examples.Example`
objects built **only from her real windows**.

Ranking:

1. the query is encoded exactly like a window context (:mod:`twin.retrieval.texts`) and the
   ``retrieval.candidates`` (50) nearest windows are taken from the index;
2. each gets a **score** = cosine similarity + a *time-of-day bonus* (a bell curve on the circular
   distance between the window's local slot and now, ``retrieval.slot_weight`` wide
   ``slot_sigma_slots``; half as much when the day type differs) + a *recency bonus*
   (``retrieval.recency_weight`` halving every ``recency_half_life_days``); a window whose
   reply holds nothing the bot could have written (only a call, a photo, ...) is multiplied by
   ``retrieval.event_only_factor``;
3. **MMR** (``retrieval.mmr_lambda``, 0.7) picks ``k`` of them, trading the score against the
   similarity to what is already picked;
4. a candidate whose reply text is more than ``retrieval.dedup_similarity`` (0.9) alike to a
   picked reply is dropped (character-level ratio): eight copies of "好的" teach nothing.

``before`` (R-TRN-013): only windows with ``reply_at_utc < before`` are considered, so the
evaluation sandbox and the training export (``AsOfView(t)``) see the library as it was at ``t``.
Whatever ``before`` is, windows at or after the hold-out cutoff are never returned, even if a
stale index still holds their vectors.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import select

from twin.clock import ensure_aware, to_epoch
from twin.ops.logging import get_logger
from twin.profile.circular import signed_diff
from twin.profile.holdout import get_holdout
from twin.retrieval.embedder import EmbeddingService, EncodeKind, embedding_service
from twin.retrieval.examples import Example, ExampleBuilder, StickerLabeler
from twin.retrieval.indexer import encoding_id, window_table
from twin.retrieval.records import MessageData, load_messages
from twin.retrieval.texts import TurnText, encoding_text, one_line
from twin.retrieval.vector_store import IndexMismatchError, VectorHit
from twin.retrieval.windows import WindowRecord
from twin.storage.retrieval_models import ExampleWindow

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.retrieval.query")

SLOT_MINUTES = 15.0
DAY_TYPE_MISMATCH_FACTOR = 0.5
SECONDS_PER_DAY = 86_400.0
Vector = NDArray[np.float32]


@dataclass(frozen=True)
class QueryTurn:
    """One turn of the conversation the query is made from."""

    her: bool  # True: the bot speaking as her (or her own words); False: the user
    text: str


@dataclass(frozen=True)
class ExampleQuery:
    """What to look up."""

    turns: Sequence[QueryTurn]  # oldest first; only the last ``retrieval.context_turns`` count
    local_minute: float  # minutes after local midnight now
    day_type: str  # workday | weekend | holiday
    before: datetime | None = None  # only windows with reply_at_utc < before (R-TRN-013)
    k: int | None = None  # examples wanted; default ``engine.examples_k`` (budget level: R-LLM-008)


@dataclass(frozen=True)
class Ranked:
    """A window with its similarity and its final score."""

    window: WindowRecord
    similarity: float
    score: float
    vector: Vector
    reply_text: str


def time_bonus(
    local_minute: float,
    day_type: str,
    slot: int,
    slot_day_type: str,
    *,
    weight: float,
    sigma_slots: float,
) -> float:
    """The bonus for a window from about the same time of day (circular distance)."""
    centre = slot * SLOT_MINUTES + SLOT_MINUTES / 2
    distance_slots = abs(signed_diff(local_minute, centre)) / SLOT_MINUTES
    bell = math.exp(-0.5 * (distance_slots / sigma_slots) ** 2)
    return weight * bell * (1.0 if slot_day_type == day_type else DAY_TYPE_MISMATCH_FACTOR)


def recency_bonus(age_days: float, *, weight: float, half_life_days: float) -> float:
    return weight * math.pow(0.5, max(0.0, age_days) / half_life_days)


def reply_signature(messages: Sequence[MessageData]) -> str:
    """The text of a reply that near-duplicate detection compares (no event lines)."""
    parts = []
    for message in messages:
        if message.kind == "sticker":
            parts.append(f"[表情包:{message.sticker_md5}]")
        elif message.kind in ("text", "quote"):
            parts.append(one_line(message.text))
    return "\n".join(part for part in parts if part)


def mmr_select(
    ranked: Sequence[Ranked], k: int, *, mmr_lambda: float, dedup_similarity: float
) -> list[Ranked]:
    """Pick ``k`` windows: score against redundancy, without near-identical replies."""
    pool = sorted(ranked, key=lambda r: -r.score)
    chosen: list[Ranked] = []
    while pool and len(chosen) < k:
        best_index = -1
        best_value = -math.inf
        for index, candidate in enumerate(pool):
            redundancy = max((float(candidate.vector @ c.vector) for c in chosen), default=0.0)
            value = mmr_lambda * candidate.score - (1.0 - mmr_lambda) * redundancy
            if value > best_value:
                best_index, best_value = index, value
        picked = pool.pop(best_index)
        if any(
            SequenceMatcher(None, picked.reply_text, c.reply_text).ratio() > dedup_similarity
            for c in chosen
            if picked.reply_text and c.reply_text
        ):
            continue
        chosen.append(picked)
    return chosen


class ExampleRetriever:
    """The query side of the library (one per process is plenty)."""

    def __init__(
        self,
        services: Services,
        embedder: EmbeddingService | None = None,
        *,
        builder: ExampleBuilder | None = None,
        sticker_label: StickerLabeler | None = None,
    ) -> None:
        self._services = services
        self._embedder = embedder
        self._builder = builder or ExampleBuilder(services, sticker_label=sticker_label)

    def _service(self) -> EmbeddingService:
        if self._embedder is None:
            self._embedder = embedding_service(self._services)
        return self._embedder

    async def aclose(self) -> None:
        await self._builder.aclose()

    # ------------------------------------------------------------------- ranking

    def rank(self, query: ExampleQuery) -> list[Ranked]:
        """The ``k`` chosen windows with their scores (no picture descriptions yet)."""
        services = self._services
        config = services.settings.retrieval
        k = query.k if query.k is not None else services.settings.engine.examples_k
        turns = [TurnText(t.her, one_line(t.text)) for t in query.turns][-config.context_turns :]
        text = encoding_text(turns)
        holdout = get_holdout(services)
        table = window_table(services)
        meta = table.read_meta()
        if k <= 0 or not text or holdout is None or meta is None or table.count() == 0:
            return []

        embedder = self._service()
        vector = embedder.encode_one(text, EncodeKind.SYMMETRIC)
        current = encoding_id(embedder.info)
        if current != meta.encoding:
            raise IndexMismatchError(
                f"the index was made with {meta.model} ({meta.encoding}) but the embedding "
                f"model is now {embedder.info.describe()}; run `twin retrieval rebuild`"
            )

        before = None if query.before is None else ensure_aware(query.before)
        bound = holdout.cutoff if before is None else min(holdout.cutoff, before)
        hits = table.search(vector, config.candidates, at_or_before=math.floor(to_epoch(bound)))
        if not hits:
            return []
        reference = before or services.clock.now_utc()
        records, replies = self._load(hits, bound)
        ranked: list[Ranked] = []
        for hit in hits:
            record = records.get(hit.id)
            if record is None:
                continue  # a vector without a window (removed since): nothing to show
            score = hit.similarity
            score += time_bonus(
                query.local_minute,
                query.day_type,
                record.local_slot,
                record.day_type,
                weight=config.slot_weight,
                sigma_slots=config.slot_sigma_slots,
            )
            age = (reference - record.reply_at_utc).total_seconds() / SECONDS_PER_DAY
            score += recency_bonus(
                age, weight=config.recency_weight, half_life_days=config.recency_half_life_days
            )
            if record.reply_reproducible == 0:
                score *= config.event_only_factor
            ranked.append(Ranked(record, hit.similarity, score, hit.vector, replies[record.id]))
        return mmr_select(
            ranked, k, mmr_lambda=config.mmr_lambda, dedup_similarity=config.dedup_similarity
        )

    def _load(
        self, hits: Sequence[VectorHit], bound: datetime
    ) -> tuple[dict[str, WindowRecord], dict[str, str]]:
        """Window rows of the hits (exactly before ``bound``, not held out) and reply texts."""
        ids = [hit.id for hit in hits]
        with self._services.db.session() as session:
            rows = session.scalars(select(ExampleWindow).where(ExampleWindow.id.in_(ids)))
            records = {
                row.id: WindowRecord.from_row(row)
                for row in rows
                if not row.holdout and row.reply_at_utc < bound
            }
            messages = load_messages(
                session, [i for record in records.values() for i in record.reply_block_ids]
            )
        replies = {
            record.id: reply_signature(
                [messages[i] for i in record.reply_block_ids if i in messages]
            )
            for record in records.values()
        }
        return records, replies

    # --------------------------------------------------------------------- query

    async def query(self, query: ExampleQuery) -> list[Example]:
        """Examples for the situation, best first (R-RET-005)."""
        chosen = await asyncio.to_thread(self.rank, query)
        if not chosen:
            return []
        scores = {r.window.id: (r.similarity, r.score) for r in chosen}
        return await self._builder.build([r.window for r in chosen], scores)
