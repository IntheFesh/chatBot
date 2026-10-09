"""Drawing the contexts of a blind test from the hold-out (R-EVAL-001, R-SAFE-006, R-TRN-013).

A sample is one of her reply blocks - a burst of hers that answers the user - with the
conversation before it.  Where the samples come from is fixed by rules, so the test measures the
bot and not the choice of examples:

* **only the hold-out window**: the blocks whose first message is at or after
  :func:`~twin.profile.holdout.holdout_cutoff`; nothing the bot was built from;
* **never a context an earlier blind test used** (the keys the store hands in);
* **only replies the bot could have written**: every message of the reply is text, a sticker or a
  quote (``is_reproducible``), no line of it is event text or a media marker (R-SAFE-006 4),
  and the conversation before it ends with the user's turn;
* **stratified**: the sample is spread over the four periods of her local day (night 0-6,
  morning 6-12, afternoon 12-18, evening 18-24) and three lengths of the conversation segment
  before the reply (1-2, 3-5 and 6-8 turns; the retrieval windows hold at most eight), in
  proportion to how many blocks each stratum has, with at least one from every stratum that has
  any.  Inside a stratum the draw is random but seeded, so a run can be reproduced.

The messages are read through :mod:`twin.ingest.corpus` and only before the moment of the reply
(:mod:`twin.training.export_blocks`, the training export's own door), never from the bot's
conversation.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict, deque
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from twin.eval.render import (
    Candidate,
    candidate_from_messages,
    has_event_text,
    message_lines,
    render_candidate,
)
from twin.memory.recent import BotMessage, Role, Turn, merge_turns
from twin.profile.holdout import holdout_cutoff
from twin.retrieval.windows import LocalPlace
from twin.stickers.catalog import StickerCatalog
from twin.training.export_blocks import (
    BlockRef,
    LoadedBlock,
    LoadedMessage,
    MessageLoader,
    iter_block_refs,
)
from twin.training.export_text import context_text

if TYPE_CHECKING:
    from twin.services import Services

PERIODS = ("night", "morning", "afternoon", "evening")
PERIOD_LABELS = {
    "night": "凌晨 0-6 点",
    "morning": "上午 6-12 点",
    "afternoon": "下午 12-18 点",
    "evening": "晚上 18-24 点",
}
LENGTH_BINS = ("short", "medium", "long")
LENGTH_LABELS = {"short": "1-2 轮", "medium": "3-5 轮", "long": "6-8 轮"}
BATCH = 64

REASON_NOT_REPRODUCIBLE = "reply_not_reproducible"
REASON_NO_CONTEXT = "no_context"
REASON_USED = "context_used_before"
REASON_NO_USER_TURN = "context_does_not_end_with_the_user"
REASON_EVENT_TEXT = "reply_has_event_text"
REASON_EMPTY = "reply_empty"
REASON_MISSING = "message_missing"


def period_of(hour: int) -> str:
    """The period of her local day an hour belongs to."""
    if not 0 <= hour <= 23:
        raise ValueError("an hour is 0 to 23")
    return PERIODS[hour // 6]


def length_bin_of(context_turns: int) -> str:
    """The length class of the conversation segment before the reply (turns, 1 to 8)."""
    if context_turns <= 2:
        return "short"
    return "medium" if context_turns <= 5 else "long"


Stratum = tuple[str, str]  # (period, length bin)


def allocate[K: Hashable](
    sizes: Mapping[K, int], n: int, *, at_least_one: bool = True
) -> dict[K, int]:
    """How many to draw from each group: proportional to its size, one at least from each.

    The counts add up to ``min(n, total)``; with fewer draws than groups the largest groups win.
    Ties are broken by the groups' order, so the result depends on the sizes only.
    """
    groups = sorted((k for k, size in sizes.items() if size > 0), key=lambda k: (-sizes[k], str(k)))
    total = sum(sizes[k] for k in groups)
    n = min(n, total)
    if n <= 0:
        return {}
    if n < len(groups):
        return dict.fromkeys(groups[:n], 1)
    quota = {k: n * sizes[k] / total for k in groups}
    counts = {k: min(sizes[k], max(1 if at_least_one else 0, math.floor(quota[k]))) for k in groups}
    while sum(counts.values()) < n:
        room = [k for k in groups if counts[k] < sizes[k]]
        counts[max(room, key=lambda k: (quota[k] - counts[k], sizes[k]))] += 1
    while sum(counts.values()) > n:
        floor = 1 if at_least_one else 0
        room = [k for k in groups if counts[k] > floor]
        counts[max(room, key=lambda k: (counts[k] - quota[k], counts[k]))] -= 1
    return {k: c for k, c in counts.items() if c > 0}


@dataclass(frozen=True)
class DrawnSample:
    """A context drawn for the test, with everything the generation and the screen need."""

    sample_key: str
    at: datetime  # the first message of her reply: ``t`` of every as-of read
    period: str
    length_bin: str
    inbound: tuple[dict[str, Any], ...]  # the user's last turn, message by message
    history: tuple[dict[str, Any], ...]  # the merged turns before it (at most seven)
    shown: tuple[dict[str, Any], ...]  # the context for the screen: who and lines
    real: Candidate

    def payload(self) -> dict[str, Any]:
        return {
            "inbound": list(self.inbound),
            "history": list(self.history),
            "shown": list(self.shown),
            "real": self.real.to_json(),
            "t": self.at.isoformat(),
        }


@dataclass
class DrawResult:
    """The samples, and what the draw had to leave out."""

    samples: list[DrawnSample] = field(default_factory=list)
    available: int = 0  # blocks of the hold-out that passed the first checks
    holdout_blocks: int = 0
    excluded: Counter[str] = field(default_factory=Counter)
    strata: dict[str, int] = field(default_factory=dict)  # drawn per "period/length"


class StickerDescriber:
    """``md5 -> description`` of the sticker library (what both sides of a test are shown)."""

    def __init__(self, catalog: StickerCatalog) -> None:
        self._catalog = catalog
        self._known: dict[str, str | None] = {}

    def __call__(self, md5: str) -> str | None:
        if md5 not in self._known:
            record = self._catalog.get(md5)
            described = None
            if record is not None:
                described = record.description or "、".join(record.tags[:3]) or None
            self._known[md5] = described
        return self._known[md5]


def _turn_payload(turn: Turn) -> dict[str, Any]:
    return {
        "role": turn.role,
        "text": turn.text,
        "at": turn.at.isoformat(),
        "last_at": turn.last_at.isoformat(),
        "ids": list(turn.message_ids),
    }


class SampleDrawer:
    """Draws the contexts of a blind test (see the module description)."""

    def __init__(self, services: Services) -> None:
        self._services = services
        self._loader = MessageLoader(services)
        self._place = LocalPlace.create(services)
        catalog = StickerCatalog(services)
        self._tag = catalog.tag_lookup()
        self.describe = StickerDescriber(catalog)

    # ------------------------------------------------------------------ strata

    def stratum_of(self, ref: BlockRef) -> Stratum:
        hour = int(self._place.stamp(ref.reply_at).minute // 60)
        return period_of(min(23, hour)), length_bin_of(len(ref.context_ids))

    def candidates(self, exclude: set[str]) -> tuple[dict[Stratum, list[BlockRef]], DrawResult]:
        """The blocks of the hold-out that pass the checks that need no text, by stratum."""
        cutoff = holdout_cutoff(self._services)
        result = DrawResult()
        groups: dict[Stratum, list[BlockRef]] = defaultdict(list)
        for ref in iter_block_refs(self._services, since=cutoff):
            result.holdout_blocks += 1
            if ref.reproducible != len(ref.reply_ids):
                result.excluded[REASON_NOT_REPRODUCIBLE] += 1
            elif not ref.context_ids:
                result.excluded[REASON_NO_CONTEXT] += 1
            elif ref.sample_id in exclude:
                result.excluded[REASON_USED] += 1
            else:
                groups[self.stratum_of(ref)].append(ref)
        result.available = sum(len(refs) for refs in groups.values())
        return groups, result

    # ------------------------------------------------------------------ drawing

    def draw(self, n: int, *, seed: int, exclude: set[str] | None = None) -> DrawResult:
        """Up to ``n`` samples (fewer when the hold-out has not enough usable blocks)."""
        groups, result = self.candidates(exclude or set())
        rng = random.Random(seed)  # noqa: S311 - a reproducible draw, not security
        queues: dict[Stratum, deque[BlockRef]] = {}
        for stratum in sorted(groups):
            refs = list(groups[stratum])
            rng.shuffle(refs)
            queues[stratum] = deque(refs)
        accepted: dict[Stratum, list[DrawnSample]] = defaultdict(list)
        need = allocate({s: len(q) for s, q in queues.items()}, n)
        while need:
            taken: list[tuple[Stratum, BlockRef]] = []
            for stratum, count in need.items():
                for _ in range(min(count, len(queues[stratum]))):
                    taken.append((stratum, queues[stratum].popleft()))
            if not taken:
                break
            for stratum, block, ref in self._load(taken, result):
                sample = self._sample(stratum, block, ref, result)
                if sample is not None:
                    accepted[stratum].append(sample)
            drawn = sum(len(v) for v in accepted.values())
            remaining = {s: len(q) for s, q in queues.items()}
            need = allocate(remaining, n - drawn, at_least_one=False)
        result.samples = sorted(
            (s for group in accepted.values() for s in group), key=lambda s: (s.at, s.sample_key)
        )
        rng.shuffle(result.samples)  # the order the user meets them in
        result.strata = {
            f"{period}/{length}": len(found) for (period, length), found in sorted(accepted.items())
        }
        return result

    def _load(
        self, taken: Sequence[tuple[Stratum, BlockRef]], result: DrawResult
    ) -> list[tuple[Stratum, LoadedBlock, BlockRef]]:
        loaded: list[tuple[Stratum, LoadedBlock, BlockRef]] = []
        for start in range(0, len(taken), BATCH):
            chunk = taken[start : start + BATCH]
            try:
                blocks = self._loader.load(ref for _, ref in chunk)
            except LookupError:
                result.excluded[REASON_MISSING] += len(chunk)
                continue
            loaded.extend((s, b, r) for (s, r), b in zip(chunk, blocks, strict=True))
        return loaded

    # ----------------------------------------------------------------- one sample

    def _sample(
        self, stratum: Stratum, block: LoadedBlock, ref: BlockRef, result: DrawResult
    ) -> DrawnSample | None:
        real = candidate_from_messages([m.data for m in block.reply])
        if real.empty:
            result.excluded[REASON_EMPTY] += 1
            return None
        if has_event_text(real):
            result.excluded[REASON_EVENT_TEXT] += 1
            return None
        flat = [m for turn in block.context for m in turn]
        messages: list[BotMessage] = []
        for message in flat:
            text = context_text([message.data], self._tag)
            if text:
                role: Role = "user" if message.data.is_sent else "bot"
                messages.append(BotMessage(message.data.id, role, text, message.at))
        turns = merge_turns(messages)
        if not turns or turns[-1].role != "user":
            result.excluded[REASON_NO_USER_TURN] += 1
            return None
        last = turns[-1]
        by_id = {m.data.id: m for m in flat}
        inbound = tuple(self._inbound(by_id[i]) for i in last.message_ids if i in by_id)
        shown = tuple(self._shown(turn, by_id) for turn in turns[-8:])
        return DrawnSample(
            sample_key=block.sample_id,
            at=block.at,
            period=stratum[0],
            length_bin=stratum[1],
            inbound=inbound,
            history=tuple(_turn_payload(t) for t in turns[:-1][-7:]),
            shown=shown,
            real=real,
        )

    def _inbound(self, message: LoadedMessage) -> dict[str, Any]:
        return {
            "id": message.data.id,
            "at": message.at.isoformat(),
            "kind": "text" if message.data.kind == "quote" else message.data.kind,
            "text": context_text([message.data], self._tag),
        }

    def _shown(self, turn: Turn, by_id: Mapping[str, LoadedMessage]) -> dict[str, Any]:
        lines: list[str] = []
        for message_id in turn.message_ids:
            found = by_id.get(message_id)
            if found is not None:
                lines.extend(message_lines(found.data, self.describe))
        return {"who": "me" if turn.role == "user" else "her", "lines": lines}

    def render_real(self, sample: DrawnSample) -> str:
        return render_candidate(sample.real, self.describe)
