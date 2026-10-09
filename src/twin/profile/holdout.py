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
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from twin.ingest.corpus import conversation_timeline
from twin.ingest.events import REPRODUCIBLE_KINDS
from twin.profile.queue import queue_profile_rebuild
from twin.profile.units import BurstSegmenter
from twin.storage.settings_store import get_setting, put_setting

if TYPE_CHECKING:
    from twin.services import Services

HOLDOUT_KEY = "retrieval.holdout"
MIN_BLOCKS = 2


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
class ResplitResult:
    previous: Holdout | None
    current: Holdout
    notes: tuple[str, ...]


HoldoutListener = Callable[["Services", Holdout | None, Holdout], str]
_listeners: dict[str, HoldoutListener] = {}


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


def resplit_holdout(services: Services, *, by: str = "user") -> ResplitResult:
    """Recompute the split point from the data now and tell the dependants.

    Everything derived from the pre-holdout part of the data is out of date afterwards, and
    evaluation results from before the re-split are not comparable with later ones.
    """
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
