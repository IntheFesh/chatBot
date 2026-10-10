"""Choosing a sticker for a tag (R-STK-004, R-TRN-013).

The model writes a line ``[表情包:<标签>]``; :meth:`StickerSelector.choose` turns the tag into
one of her stickers:

* **candidates** - her stickers that carry the tag (the tags a sticker is *selected by*, see
  :mod:`twin.stickers.tags`), that are available, not switched off, and that she used at least
  once in the data the view allows;
* **score** - her use count with add-one smoothing (``(uses + 1) / (total + candidates)``) times
  the cosine similarity of the description vector with the vector of the current context times a
  recency factor that halves every ``stickers.recency_half_life_days`` since she last used it;
* **no repeats** - a sticker that is among the last ``stickers.no_repeat_window`` (10) bubbles is
  not chosen again, unless her own repeat rate (how often she sends a sticker she sent within the
  last 10 bubbles) is above ``stickers.repeat_rate_threshold``;
* **fallback** - no candidate for the tag: the nearby tags of the neighbour table, nearest
  first; still none: ``None``, and the line is deleted.

The pick is a draw in proportion to the score, so a favourite is chosen often and the rest
sometimes, as she does; :meth:`StickerSelector.rank` shows the scores themselves.

**Data views (R-TRN-013).**  The live view uses all of her stickers and all her uses.  An as-of
view (:func:`as_of_view`, built by ``AsOfView(t)`` of rounds 07 and 13) sees only the stickers
she had used *before* ``t`` and counts uses and recency as of ``t``, so the sticker she used for
the first time after ``t`` can never be chosen; the repeat rate comes from the pre-holdout data.
"""

from __future__ import annotations

import math
import random
from bisect import bisect_left
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from sqlalchemy import select

from twin.ingest.corpus import her_bubble_skeleton
from twin.profile.holdout import HoldoutError, holdout_cutoff
from twin.services import Services
from twin.stickers.catalog import StickerCatalog, StickerRecord
from twin.stickers.vectors import StickerVectors
from twin.storage.chat_models import StickerUse

NEUTRAL_SIMILARITY = 0.5  # for a sticker that has no description vector (yet)
MIN_SIMILARITY = 0.01
RECENCY_FLOOR = 0.1  # a sticker she has not used for ages still counts a little
SECONDS_PER_DAY = 86_400.0

Scope = Literal["live", "pre_holdout"]


@dataclass(frozen=True)
class StickerView:
    """Which data the selector may see: ``as_of`` limits her uses to those before that moment."""

    scope: Scope = "live"
    as_of: datetime | None = None


LIVE_VIEW = StickerView()


def as_of_view(moment: datetime) -> StickerView:
    """The view of the training export and the evaluation sandbox at ``moment`` (pre-holdout)."""
    if moment.tzinfo is None:
        raise ValueError("an as-of moment needs a time zone")
    return StickerView("pre_holdout", moment)


@dataclass(frozen=True)
class Usage:
    """Her use of one sticker, as of a moment."""

    uses: int
    last: datetime


class UsageTable:
    """Her uses of every sticker, loaded once; any as-of moment is answered by bisection."""

    def __init__(self, times: dict[str, list[datetime]]) -> None:
        self._times = times
        self._stamps = {md5: [t.timestamp() for t in found] for md5, found in times.items()}

    @classmethod
    def load(cls, services: Services) -> UsageTable:
        stmt = (
            select(StickerUse.sticker_md5, StickerUse.used_at)
            .where(StickerUse.by_her.is_(True))
            .order_by(StickerUse.used_at)
        )
        times: dict[str, list[datetime]] = {}
        with services.db.session() as session:
            for md5, used_at in session.execute(stmt):
                times.setdefault(md5, []).append(used_at)
        return cls(times)

    def as_of(self, moment: datetime | None) -> dict[str, Usage]:
        """Uses before ``moment`` (all uses if ``None``) of the stickers used at all by then."""
        found: dict[str, Usage] = {}
        for md5, times in self._times.items():
            count = (
                len(times) if moment is None else bisect_left(self._stamps[md5], moment.timestamp())
            )
            if count:
                found[md5] = Usage(count, times[count - 1])
        return found


@dataclass(frozen=True)
class Candidate:
    """A sticker with the three factors of its score."""

    record: StickerRecord
    uses: int
    frequency: float
    similarity: float
    recency: float

    @property
    def score(self) -> float:
        return self.frequency * self.similarity * self.recency


def her_repeat_rate(services: Services, window: int, *, before: datetime | None) -> float:
    """How often a sticker of hers repeats one she sent within the previous ``window`` bubbles."""
    recent: deque[str | None] = deque(maxlen=max(1, window))
    stickers = repeats = 0
    with services.db.session() as session:
        for row in session.scalars(her_bubble_skeleton(before).execution_options(yield_per=5000)):
            md5 = row.sticker_md5 if row.kind == "sticker" else None
            if md5:
                stickers += 1
                repeats += int(md5 in recent)
            recent.append(md5)
    return repeats / stickers if stickers else 0.0


class StickerSelector:
    """Chooses one of her stickers for a tag (see the module description)."""

    def __init__(
        self,
        services: Services,
        *,
        view: StickerView = LIVE_VIEW,
        catalog: StickerCatalog | None = None,
        vectors: StickerVectors | None = None,
        usage: UsageTable | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._services = services
        self._view = view
        self._catalog = catalog or StickerCatalog(services)
        self._vectors = vectors or StickerVectors(services)
        self._usage = usage
        self._records: list[StickerRecord] | None = None
        self._rng = rng or random.Random()  # noqa: S311 - a draw among stickers, not security
        self._repeat_rate: float | None = None

    @property
    def view(self) -> StickerView:
        return self._view

    def with_view(self, view: StickerView) -> StickerSelector:
        """The same selector for another view (shares the loaded data)."""
        other = StickerSelector(
            self._services,
            view=view,
            catalog=self._catalog,
            vectors=self._vectors,
            usage=self._usage_table(),
            rng=self._rng,
        )
        other._repeat_rate = self._repeat_rate if view.scope == self._view.scope else None
        other._records = self._records
        return other

    def reload(self) -> None:
        """Forget the loaded uses (after an import)."""
        self._usage = None
        self._records = None
        self._repeat_rate = None

    def _usage_table(self) -> UsageTable:
        if self._usage is None:
            self._usage = UsageTable.load(self._services)
        return self._usage

    def _bound(self) -> datetime | None:
        """The moment the data of the view ends: its as-of moment or, pre-holdout, the cutoff."""
        if self._view.as_of is not None:
            return self._view.as_of
        if self._view.scope == "pre_holdout":
            try:
                return holdout_cutoff(self._services)
            except HoldoutError:
                return None
        return None

    def repeat_rate(self) -> float:
        """Her rate of repeating a sticker within the window, in the data of the view's scope."""
        if self._repeat_rate is None:
            before = None
            if self._view.scope == "pre_holdout":
                try:
                    before = holdout_cutoff(self._services)
                except HoldoutError:
                    before = None
            config = self._services.settings.stickers
            self._repeat_rate = her_repeat_rate(
                self._services, config.no_repeat_window, before=before
            )
        return self._repeat_rate

    def repeats_allowed(self) -> bool:
        return self.repeat_rate() > self._services.settings.stickers.repeat_rate_threshold

    # ------------------------------------------------------------------ ranking

    def _recency(self, last: datetime, now: datetime) -> float:
        half_life = self._services.settings.stickers.recency_half_life_days
        age_days = max(0.0, (now - last).total_seconds() / SECONDS_PER_DAY)
        decay = math.pow(0.5, age_days / half_life)
        return RECENCY_FLOOR + (1.0 - RECENCY_FLOOR) * decay

    def _pool(self, tag: str, usage: dict[str, Usage]) -> list[StickerRecord]:
        if self._records is None:
            self._records = self._catalog.records(status="available", tagged=True)
        return [
            record
            for record in self._records
            if record.usable and tag in record.tags and record.md5 in usage
        ]

    def rank(
        self, tag: str, context_text: str, recent_bubbles: Sequence[str | None] = ()
    ) -> list[Candidate]:
        """The candidates for exactly this tag, best first (no fallback to nearby tags)."""
        config = self._services.settings.stickers
        bound = self._bound()
        usage = self._usage_table().as_of(bound)
        pool = self._pool(tag, usage)
        if not self.repeats_allowed() and config.no_repeat_window:
            blocked = {m for m in recent_bubbles[-config.no_repeat_window :] if m}
            pool = [record for record in pool if record.md5 not in blocked]
        if not pool:
            return []
        total = sum(usage[r.md5].uses for r in pool)
        similarity = (
            self._vectors.similarities(context_text, [r.md5 for r in pool])
            if context_text.strip()
            else {}
        )
        now = bound or self._services.clock.now_utc()
        ranked = [
            Candidate(
                record,
                usage[record.md5].uses,
                (usage[record.md5].uses + 1) / (total + len(pool)),
                max(MIN_SIMILARITY, min(1.0, similarity.get(record.md5, NEUTRAL_SIMILARITY))),
                self._recency(usage[record.md5].last, now),
            )
            for record in pool
        ]
        return sorted(ranked, key=lambda c: (-c.score, c.record.md5))

    def choose(
        self, tag: str, context_text: str, recent_bubbles: Sequence[str | None] = ()
    ) -> StickerRecord | None:
        """One of her stickers for ``tag`` (or a nearby tag); ``None`` if there is none.

        ``recent_bubbles`` lists the last bubbles of the conversation, oldest first: the MD5 of
        the sticker where a bubble was a sticker, ``None`` otherwise.
        """
        for candidate_tag in (tag, *self._catalog.vocabulary.near(tag)):
            ranked = self.rank(candidate_tag, context_text, recent_bubbles)
            if ranked:
                weights = [c.score for c in ranked]
                return self._rng.choices(ranked, weights=weights, k=1)[0].record
        return None
