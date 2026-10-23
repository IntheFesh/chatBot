"""The hold-out split point: the only place that cuts the data 90 / 10 (R-RET-003, R-TRN-013).

``holdout_cutoff(services)`` answers one question for the whole system: from which moment on
do her messages belong to the held-out evaluation set?  Retrieval (round 05), the persona
card and sticker context (round 06), the evaluation (round 09b) and the training export
(round 13) all call it, and a scan test (``tests/unit/test_holdout_unique.py``) fails if any
other code computes a ratio split of the data.

Definition.  Her *reply blocks* are her bursts (:class:`~twin.profile.units.BurstSegmenter`,
messages of every kind except system notices) that contain at least one message the bot could
write itself (text, sticker, quote).  They are ordered by the time of their first message;
the last ``retrieval.holdout_ratio`` of them (10 %, at least one) are the hold-out set, and the
cutoff is the start time of the earliest one.  Messages before the cutoff are *pre-holdout*.

Persistence.  The first call computes the cutoff and stores it in the ``retrieval.holdout``
setting; later calls return the stored value, so importing new records never moves it (that
would silently change which part of the data the evaluation has seen).  :func:`resplit_holdout`
recomputes it on purpose, records the change, and tells every registered listener — the profile
rebuild of the pre-holdout scope is queued automatically; later rounds register theirs with
:func:`on_holdout_change`.

Drift (R18-C3, D-818).  Because the cut-off stays, everything imported later is held out: kept
from the example library and from the training split.  :func:`holdout_share` counts, without
storing anything, the share of her reply blocks at or after the stored cut-off *now*;
:func:`drift_warning` turns a share clearly above ``retrieval.holdout_ratio`` (more than twice the
ratio and more than ten points above it) into the one sentence that ``twin profile show``, the
retraining check, the export and the import report print: run ``twin retrieval resplit``.
Moving the cut-off by itself is not done here (the user decides).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from twin.ingest.corpus import conversation_timeline
from twin.ingest.events import REPRODUCIBLE_KINDS
from twin.profile.queue import queue_profile_rebuild
from twin.profile.units import BurstSegmenter
from twin.storage.settings_store import get_setting, put_setting

if TYPE_CHECKING:
    from twin.services import Services

HOLDOUT_KEY = "retrieval.holdout"
MIN_BLOCKS = 2
DRIFT_FACTOR: Final = 2.0
DRIFT_MARGIN: Final = 0.10
"""The share after the cut-off has drifted when it is above both ``ratio * DRIFT_FACTOR`` and
``ratio + DRIFT_MARGIN`` (with the default 10 %: above 20 %; D-818)."""


class HoldoutError(RuntimeError):
    """The hold-out cannot be determined (too little data)."""


@dataclass(frozen=True)
class Holdout:
    cutoff: datetime
    ratio: float
    blocks: int  # her reply blocks in the data
    held_out_blocks: int  # reply blocks at or after the cutoff
    computed_at: datetime

    def to_json(self) -> dict[str, Any]:
        return {
            "cutoff": self.cutoff.isoformat(),
            "ratio": self.ratio,
            "blocks": self.blocks,
            "held_out_blocks": self.held_out_blocks,
            "computed_at": self.computed_at.isoformat(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Holdout:
        return cls(
            datetime.fromisoformat(data["cutoff"]),
            float(data["ratio"]),
            int(data["blocks"]),
            int(data["held_out_blocks"]),
            datetime.fromisoformat(data["computed_at"]),
        )


@dataclass(frozen=True)
class HoldoutShare:
    """Where the stored cut-off stands in the data now (counts only; nothing is stored)."""

    cutoff: datetime
    ratio: float  # retrieval.holdout_ratio, the share the cut-off was meant to hold out
    blocks: int  # her reply blocks now
    held_out_blocks: int  # of them at or after the stored cut-off

    @property
    def share(self) -> float:
        return self.held_out_blocks / self.blocks if self.blocks else 0.0

    @property
    def drifted(self) -> bool:
        return self.share > max(self.ratio * DRIFT_FACTOR, self.ratio + DRIFT_MARGIN)


@dataclass(frozen=True)
class ResplitResult:
    previous: Holdout | None
    current: Holdout
    notes: tuple[str, ...]


HoldoutListener = Callable[["Services", Holdout | None, Holdout], str]
_listeners: dict[str, HoldoutListener] = {}

# Modules that register a re-split listener on import; each round with data derived from the
# pre-holdout part appends its module, so that a re-split started from anywhere reaches it.
LISTENER_MODULES: tuple[str, ...] = (
    "twin.retrieval.resplit",
    "twin.profile.persona.hook",
    "twin.stickers.hook",
)


def on_holdout_change(name: str) -> Callable[[HoldoutListener], HoldoutListener]:
    """Register a function to run after an explicit re-split (it returns a one-line note)."""

    def decorate(func: HoldoutListener) -> HoldoutListener:
        _listeners[name] = func
        return func

    return decorate


# ------------------------------------------------------------------- computing


def _reply_block_starts(services: Services) -> list[float]:
    """Start times (epoch seconds) of her reply blocks, oldest first."""
    settings = services.settings.profile
    segmenter = BurstSegmenter(settings.burst_gap_s, settings.segment_gap_min * 60.0)
    starts: list[float] = []
    current_start: float | None = None
    current_ok = False

    def close() -> None:
        if current_start is not None and current_ok:
            starts.append(current_start)

    with services.db.session() as session:
        rows = session.execute(conversation_timeline().execution_options(yield_per=20_000))
        for moment, is_sent, kind in rows:
            if kind == "system":
                continue
            her = not is_sent
            boundary = segmenter.feed(moment.timestamp(), her)
            if boundary.new_block:
                close()
                current_start = moment.timestamp() if her else None
                current_ok = False
            if her and kind in REPRODUCIBLE_KINDS:
                current_ok = True
    close()
    return starts


def compute_holdout(services: Services) -> Holdout:
    """Work out the split point from the data (without storing it)."""
    ratio = services.settings.retrieval.holdout_ratio
    starts = _reply_block_starts(services)
    count = len(starts)
    if count < MIN_BLOCKS:
        raise HoldoutError(
            f"the hold-out needs at least {MIN_BLOCKS} of her reply blocks, found {count}; "
            "import her messages first"
        )
    held = min(count - 1, max(1, int(count * ratio + 0.5)))
    cutoff = datetime.fromtimestamp(starts[count - held], UTC)
    return Holdout(cutoff, ratio, count, held, services.clock.now_utc())


# ------------------------------------------------------------------ persistence


def get_holdout(services: Services) -> Holdout | None:
    """The stored hold-out, or ``None`` before it was first needed (a pure read)."""
    with services.db.session() as session:
        raw = get_setting(session, HOLDOUT_KEY)
    return Holdout.from_json(raw) if raw else None


def _store(services: Services, holdout: Holdout, *, by: str) -> None:
    with services.db.transaction(bump_state=True) as session:
        put_setting(session, HOLDOUT_KEY, holdout.to_json(), clock=services.clock, by=by)


def holdout_cutoff(services: Services) -> datetime:
    """The first moment of the held-out period (computed once, then persisted)."""
    stored = get_holdout(services)
    if stored is not None:
        return stored.cutoff
    computed = compute_holdout(services)
    _store(services, computed, by="holdout_cutoff")
    return computed.cutoff


def holdout_share(services: Services) -> HoldoutShare | None:
    """The share of her reply blocks at or after the stored cut-off now (a pure read).

    ``None`` before the cut-off was first needed.  The blocks are the ones of the cut-off itself
    (:func:`_reply_block_starts`), so the numbers agree with what it was computed from.
    """
    stored = get_holdout(services)
    if stored is None:
        return None
    starts = _reply_block_starts(services)
    edge = stored.cutoff.timestamp()
    held = sum(1 for start in starts if start >= edge)
    return HoldoutShare(stored.cutoff, services.settings.retrieval.holdout_ratio, len(starts), held)


def drift_warning(share: HoldoutShare) -> str | None:
    """The sentence about a cut-off that has fallen behind the data; ``None`` while it has not."""
    if not share.drifted:
        return None
    return (
        f"留出集已占她回复块的 {share.share:.0%}（{share.held_out_blocks:,} / {share.blocks:,}；"
        f"配置的留出比例 {share.ratio:.0%}，"
        f"切分点 {share.cutoff:%Y-%m-%d %H:%M} UTC 不会自动移动）："
        "切分点之后导入的聊天都在留出集里，既不进检索库也不进训练集，重训也学不到它们。"
        "先运行 `twin retrieval resplit` 把切分点移到现在数据的最新部分"
        "（之前的评估结果与之后不可比）"
    )


def resplit_holdout(services: Services, *, by: str = "user") -> ResplitResult:
    """Recompute the split point from the data now and tell the dependants.

    Everything derived from the pre-holdout part of the data is out of date afterwards, and
    evaluation results from before the re-split are not comparable with later ones.
    """
    for module in LISTENER_MODULES:
        importlib.import_module(module)
    previous = get_holdout(services)
    current = compute_holdout(services)
    _store(services, current, by=by)
    notes = [
        "evaluation results from before this re-split are not comparable with later ones",
    ]
    for name, listener in sorted(_listeners.items()):
        notes.append(f"{name}: {listener(services, previous, current)}")
    return ResplitResult(previous, current, tuple(notes))


@on_holdout_change("profile")
def _requeue_pre_holdout_profile(
    services: Services, previous: Holdout | None, current: Holdout
) -> str:
    queued = queue_profile_rebuild(services, scope="pre_holdout", reason="resplit", force=True)
    return "pre-holdout profile and routine rebuild " + (
        "already queued" if queued.already_queued else "queued"
    )
