"""Her real openings: how she started a conversation, for the proactive messages (R-PRO-006).

The library of round 05 answers "how did she reply in a situation like this?" - it is searched by
the conversation before her reply.  A proactive message has no such conversation: she speaks
first.  The windows that stand for exactly that are the ones **without context**: her reply block
opened a conversation segment (the first message after a silence of ``profile.segment_gap_min``),
see :mod:`twin.retrieval.windows`.  They have no vector (nothing to encode), so they are not
found by the semantic query; :class:`OpenerExamples` reads them by **the time of day** instead:

* only windows of the live library - not held out, earlier than the hold-out cut-off, with
  something in the reply the bot could have written - and, if ``before`` is given, earlier than it;
* ranked by how close their local time of day is to now (the same bell curve as the semantic query,
  half as much for another kind of day) plus the recency bonus of the query;
* a reply that is nearly the same as one already picked is left out, and the ``k`` examples are
  drawn from the best ``3 * k`` so that two messages at the same hour do not get the same
  examples every time.

The examples are built by the same :class:`~twin.retrieval.examples.ExampleBuilder` as every other
example, so a line the bot could not have written (a picture, a call) is shown as a note, never as
something to imitate (R-SAFE-006).  Her real messages are read the way the rest of the library
reads them: by id, from ``messages``, through :mod:`twin.retrieval.records`.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Sequence
from datetime import datetime
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from sqlalchemy import select

from twin.ops.logging import get_logger
from twin.profile.holdout import get_holdout
from twin.retrieval.examples import Example, ExampleBuilder, StickerLabeler
from twin.retrieval.query import SECONDS_PER_DAY, recency_bonus, reply_signature, time_bonus
from twin.retrieval.records import load_messages
from twin.retrieval.windows import WindowRecord
from twin.storage.retrieval_models import ExampleWindow

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.retrieval.openers")

POOL_FACTOR = 3  # the examples are drawn from this many times as many of the best windows
MAX_CANDIDATES = 4000  # windows read per question (the library holds a few thousand openings)


class OpenerExamples:
    """Examples of how she opened a conversation around a time of day (see the module text)."""

    def __init__(
        self,
        services: Services,
        *,
        builder: ExampleBuilder | None = None,
        sticker_label: StickerLabeler | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._services = services
        self._builder = builder or ExampleBuilder(services, sticker_label=sticker_label)
        self._rng = rng or random.Random()  # noqa: S311 - a draw among examples, not security

    async def aclose(self) -> None:
        await self._builder.aclose()

    # ------------------------------------------------------------------ ranking

    def rank(
        self,
        *,
        local_minute: float,
        day_type: str,
        k: int,
        now: datetime,
        before: datetime | None = None,
    ) -> list[tuple[WindowRecord, float]]:
        """The windows to show, with their scores (no messages are read yet)."""
        services = self._services
        config = services.settings.retrieval
        holdout = get_holdout(services)
        if k <= 0 or holdout is None:
            return []
        bound = holdout.cutoff if before is None else min(holdout.cutoff, before)
        stmt = (
            select(ExampleWindow)
            .where(
                ExampleWindow.context_turns == 0,
                ExampleWindow.holdout.is_(False),
                ExampleWindow.reply_reproducible > 0,
                ExampleWindow.reply_at_utc < bound,
            )
            .order_by(ExampleWindow.reply_at_utc.desc())
            .limit(MAX_CANDIDATES)
        )
        with services.db.session() as session:
            records = [WindowRecord.from_row(row) for row in session.scalars(stmt)]
        scored: list[tuple[WindowRecord, float]] = []
        for record in records:
            score = time_bonus(
                local_minute,
                day_type,
                record.local_slot,
                record.day_type,
                weight=1.0,
                sigma_slots=config.slot_sigma_slots,
            )
            age = (now - record.reply_at_utc).total_seconds() / SECONDS_PER_DAY
            score += recency_bonus(
                age, weight=config.recency_weight, half_life_days=config.recency_half_life_days
            )
            scored.append((record, score))
        scored.sort(key=lambda pair: -pair[1])
        pool = scored[: k * POOL_FACTOR]
        self._rng.shuffle(pool)
        return self._distinct(pool, k)

    def _distinct(
        self, pool: Sequence[tuple[WindowRecord, float]], k: int
    ) -> list[tuple[WindowRecord, float]]:
        """Up to ``k`` windows whose replies are not nearly the same text."""
        ids = [i for record, _ in pool for i in record.reply_block_ids]
        with self._services.db.session() as session:
            messages = load_messages(session, ids)
        threshold = self._services.settings.retrieval.dedup_similarity
        chosen: list[tuple[WindowRecord, float]] = []
        texts: list[str] = []
        for record, score in pool:
            text = reply_signature([messages[i] for i in record.reply_block_ids if i in messages])
            if text and any(
                SequenceMatcher(None, text, other).ratio() > threshold for other in texts
            ):
                continue
            chosen.append((record, score))
            texts.append(text)
            if len(chosen) >= k:
                break
        chosen.sort(key=lambda pair: -pair[1])
        return chosen

    # -------------------------------------------------------------------- query

    async def query(
        self,
        *,
        local_minute: float,
        day_type: str,
        now: datetime,
        k: int | None = None,
        before: datetime | None = None,
    ) -> list[Example]:
        """Examples of her openings around this time of day, best first."""
        wanted = k if k is not None else self._services.settings.engine.examples_k
        chosen = await asyncio.to_thread(
            lambda: self.rank(
                local_minute=local_minute, day_type=day_type, k=wanted, now=now, before=before
            )
        )
        if not chosen:
            return []
        scores = {record.id: (0.0, score) for record, score in chosen}
        return await self._builder.build([record for record, _ in chosen], scores)
