"""The two renderings of a persona card (R-PERS-004, R-TRN-013).

``render_full``
    every section, for the DeepSeek backend and the hybrid planner; at most
    ``persona.full_max_tokens`` (1,500).  When the card is too long the lines go in the order
    of R-PERS-004, the last ones first: statistics rules, corrections (``[不要这样]``), forms of
    address and catchphrases, what she says in each mood, her hand-written facts, basic
    information, topics, everything else.
``render_compact``
    style content only - ``[自动-统计规则]``, the style part of ``[自动-描述]``, the style
    part of ``[手动]`` - at most ``persona.compact_max_tokens`` (400).  Facts and corrections are
    never part of it, so a training prompt made with it cannot contain a fact from the future
    (R-TRN-013); the cut order is statistics rules, address and catchphrases, moods, the rest.
    The backend of the style model and the training set both use it.

Both return a :class:`RenderedPersona` that carries the version and the scope of the card it was
made from.  Token counts use the uncalibrated estimator so the same card always renders to the
same text, whatever the process has learnt about the tokenizer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from twin.llm.tokens import TokenEstimator
from twin.profile.persona.sections import (
    AUTO,
    BASICS,
    DONT,
    FACTS,
    MANUAL,
    STATS,
    STYLE,
    CardLine,
    Tier,
    card_lines,
    split_card,
)

Kind = Literal["full", "compact"]

# (heading in the rendering, the card sections and parts that feed it)
GROUPS: tuple[tuple[str, str, str], ...] = (
    ("数字规则", STATS, ""),
    ("风格", AUTO, STYLE),
    ("基本情况", AUTO, BASICS),
    ("补充（风格）", MANUAL, STYLE),
    ("补充（事实）", MANUAL, FACTS),
    ("不要这样", DONT, ""),
)
COMPACT_GROUPS = frozenset({"数字规则", "风格", "补充（风格）"})


@dataclass(frozen=True)
class RenderedPersona:
    """A rendering with the card it came from."""

    text: str
    kind: Kind
    scope: str
    version_id: str
    number: int
    tokens: int
    budget: int
    dropped: int  # lines that did not fit

    def __str__(self) -> str:
        return self.text

    @property
    def fits(self) -> bool:
        return self.tokens <= self.budget


@dataclass(frozen=True)
class _Item:
    group: str
    tier: Tier
    text: str
    order: int


def count_tokens(text: str) -> int:
    """Uncalibrated token estimate of a text (stable between processes)."""
    return math.ceil(TokenEstimator.raw_text(text)) if text else 0


def _group_of(line: CardLine) -> str | None:
    for title, section, part in GROUPS:
        if line.section == section and (not part or line.part == part):
            return title
    return None  # a line of the automatic text outside its two known subsections


def _assemble(items: Sequence[_Item]) -> str:
    blocks: list[str] = []
    for title, _, _ in GROUPS:
        lines = [
            f"- {item.text}" for item in sorted(items, key=lambda i: i.order) if item.group == title
        ]
        if lines:
            blocks.append(f"## {title}\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def _fit(items: list[_Item], budget: int) -> tuple[str, int]:
    """The text of the highest-priority prefix of ``items`` that fits, and the lines left out."""
    ranked = sorted(items, key=lambda item: (item.tier, item.order))
    low, high = 0, len(ranked)
    while low < high:  # the largest prefix whose text fits the budget (tokens grow with it)
        middle = (low + high + 1) // 2
        if count_tokens(_assemble(ranked[:middle])) <= budget:
            low = middle
        else:
            high = middle - 1
    return _assemble(ranked[:low]), len(ranked) - low


def _items(text: str, *, compact: bool, extra_dont: Sequence[str]) -> list[_Item]:
    card = split_card(text)
    items: list[_Item] = []
    seen: set[tuple[str, str]] = set()
    for line in card_lines(card):
        group = _group_of(line)
        if group is None:
            continue
        if compact and group not in COMPACT_GROUPS:
            continue
        key = (group, line.text)
        if key in seen:
            continue
        seen.add(key)
        items.append(_Item(group, line.tier, line.text, len(items)))
    if not compact:
        for extra in extra_dont:
            cleaned = extra.strip()
            if cleaned and ("不要这样", cleaned) not in seen:
                seen.add(("不要这样", cleaned))
                items.append(_Item("不要这样", Tier.DONT, cleaned, len(items)))
    return items


def render_card(
    text: str,
    *,
    kind: Kind,
    scope: str,
    version_id: str,
    number: int,
    budget: int,
    extra_dont: Sequence[str] = (),
) -> RenderedPersona:
    """Render the Markdown of a card (the functions below name the budget and the kind)."""
    items = _items(text, compact=kind == "compact", extra_dont=extra_dont)
    rendered, dropped = _fit(items, budget)
    return RenderedPersona(
        rendered, kind, scope, version_id, number, count_tokens(rendered), budget, dropped
    )
