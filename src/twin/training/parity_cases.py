"""Choosing the samples the template check runs on (R-TRN-011).

The check on the instance (``twin.training.parity_check``) takes a few dozen samples of the
exported set - not all of them: it tokenises each one twice - and they should be the awkward ones:
a reply of several lines, a sticker or an emoji code in it, a plan, a context that opened with her,
a single turn and a deep context, a conversation that was trimmed to fit, a sample near the length
limit, one with a desensitised entity.  :class:`ParitySelector` keeps the first few samples of each
kind while the export runs and the longest of all.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final

from twin.training.dataset_dir import ParityCase

TAG_PLAN: Final = "plan"
TAG_PRELUDE: Final = "prelude"
TAG_TRIMMED: Final = "trimmed"
TAG_MULTILINE: Final = "multiline"
TAG_STICKER: Final = "sticker"
TAG_CODE: Final = "code"
TAG_QUOTE: Final = "quote"
TAG_SINGLE: Final = "single_turn"
TAG_DEEP: Final = "deep_context"
TAG_ENTITY: Final = "entity"
TAG_TEST: Final = "test_split"
TAG_LONG: Final = "longest"

PER_TAG: Final = 3
MAX_CASES: Final = 60
DEEP_TURNS: Final = 7
MULTILINE_LINES: Final = 3


@dataclass
class ParitySelector:
    """Keeps up to :data:`PER_TAG` samples of every kind and the longest one."""

    per_tag: int = PER_TAG
    limit: int = MAX_CASES
    _cases: dict[str, ParityCase] = field(default_factory=dict)
    _taken: dict[str, int] = field(default_factory=dict)
    _longest: tuple[int, ParityCase] | None = None

    def wants(self, tags: Sequence[str], tokens: int) -> bool:
        """Is there room for a sample with these tags (so that it is worth building the case)?"""
        longer = self._longest is None or tokens > self._longest[0]
        if longer:
            return True
        if len(self._cases) >= self.limit:
            return False
        return any(self._taken.get(tag, 0) < self.per_tag for tag in tags)

    def offer(
        self, make: Callable[[tuple[str, ...]], ParityCase], tags: Sequence[str], tokens: int
    ) -> None:
        """Offer a sample; ``make`` builds its case only if it is going to be kept."""
        if not self.wants(tags, tokens):
            return
        if self._longest is None or tokens > self._longest[0]:
            self._longest = (tokens, make((*tags, TAG_LONG)))
        room = [tag for tag in tags if self._taken.get(tag, 0) < self.per_tag]
        if not room or len(self._cases) >= self.limit:
            return
        case = make(tuple(tags))
        if case.id in self._cases:
            return
        self._cases[case.id] = case
        for tag in tags:
            self._taken[tag] = self._taken.get(tag, 0) + 1

    def cases(self) -> list[ParityCase]:
        """The kept cases, the longest sample included (once)."""
        chosen = dict(self._cases)
        if self._longest is not None:
            chosen.setdefault(self._longest[1].id, self._longest[1])
        return list(chosen.values())
