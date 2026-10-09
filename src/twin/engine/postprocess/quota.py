"""Fitting a reply into what the platform quota leaves (R-ENG-009, R-CH-008).

The platform lets the bot send a limited number of messages before the user writes again, and
part of that is kept for proactive messages.  The engine asks the channel for the remaining quota
at the moment of sending, subtracts ``channel.proactive_reserve`` (at least one bubble remains)
and calls :func:`fit_bubbles_to_quota` with the number it may use.  A reply with more bubbles is
shortened without losing what it says, as long as possible:

1. neighbouring text bubbles are **merged**, joined by a space - the pair with the shortest
   common text first, so the long bubbles stay as they are;
2. only when no two text bubbles touch any more, **sticker bubbles are deleted**, the last one
   first - and then neighbouring text bubbles touch, and step 1 goes on.

Each merge and each deleted sticker is returned as an action for ``bot_turns``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from twin.engine.types import Bubble, PostAction


def _merge_once(bubbles: list[Bubble]) -> bool:
    """Merge the neighbouring text pair with the shortest joint text; ``False`` if none touch."""
    best: int | None = None
    best_length = 0
    for index in range(len(bubbles) - 1):
        left, right = bubbles[index], bubbles[index + 1]
        if left.is_sticker or right.is_sticker:
            continue
        joint = len(left.text) + len(right.text) + 1
        if best is None or joint < best_length:
            best, best_length = index, joint
    if best is None:
        return False
    left, right = bubbles[best], bubbles[best + 1]
    bubbles[best : best + 2] = [replace(left, text=f"{left.text} {right.text}")]
    return True


def fit_bubbles_to_quota(
    bubbles: Sequence[Bubble], available: int
) -> tuple[list[Bubble], list[PostAction]]:
    """The bubbles that fit into ``available`` (at least one), and what was done to get there."""
    limit = max(1, available)
    fitted = list(bubbles)
    merged = deleted = 0
    while len(fitted) > limit:
        if _merge_once(fitted):
            merged += 1
            continue
        # no two text bubbles touch, so with more than one bubble left there is a sticker
        stickers = [i for i, bubble in enumerate(fitted) if bubble.is_sticker]
        del fitted[stickers[-1]]
        deleted += 1
    actions: list[PostAction] = []
    if merged:
        actions.append(PostAction("quota_merge", merged))
    if deleted:
        actions.append(PostAction("quota_sticker_drop", deleted))
    return fitted, actions
