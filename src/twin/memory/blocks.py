"""The query and the result of building a memory block (R-MEM-008).

Plain data, shared by the assembler (:mod:`twin.memory.assemble`), the as-of views
(:mod:`twin.memory.view`) and the callers that print or test a block.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MemoryQuery:
    """What the memory is asked about.

    ``text`` is the current topic: the last turns of the conversation and the user's message,
    one string.  ``subjects`` limits facts to those about these subjects (``her``, ``user``,
    ``both``, ``other``); ``None`` means all.
    """

    text: str
    subjects: frozenset[str] | None = None


@dataclass(frozen=True)
class BlockItem:
    """One line of a block and where it came from (ids only, no text)."""

    kind: str  # fact | summary | followup | lifeline
    item_id: str
    section: str
    score: float
    mandatory: bool
    tokens: int


@dataclass(frozen=True)
class MemoryBlock:
    """The memory block for one reply: the text, and what went into it."""

    text: str
    items: tuple[BlockItem, ...] = ()
    tokens: int = 0
    budget_tokens: int = 0  # the budget actually applied (after the budget level's factor)
    dropped: int = 0  # candidates that did not fit
    sections: tuple[str, ...] = field(default_factory=tuple)

    def ids(self, kind: str) -> list[str]:
        return [item.item_id for item in self.items if item.kind == kind]

    @property
    def empty(self) -> bool:
        return not self.items
