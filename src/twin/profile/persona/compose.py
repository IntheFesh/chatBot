"""Building the text of a card from its parts (R-PERS-002).

The program writes two sections, ``[自动-统计规则]`` and ``[自动-描述]``.  A new card starts with
empty ``[手动]`` (style and facts) and, for the live scope, an empty ``[不要这样]``; the
pre-holdout card has only the style part of ``[手动]`` and no ``[不要这样]`` (R-PERS-002,
R-TRN-013): hand-written facts and corrections belong to the time after the bot went live.
"""

from __future__ import annotations

from collections.abc import Sequence

from twin.profile.persona.sections import (
    AUTO,
    DONT,
    FACTS,
    MANUAL,
    STATS,
    STYLE,
    CardText,
    block_text,
    content_lines,
    split_card,
    split_subsections,
    subsection,
)


def stats_block(rule_lines: Sequence[str]) -> str:
    """``[自动-统计规则]`` with one bullet per rule."""
    return block_text(STATS, "\n".join(f"- {line}" for line in rule_lines))


def auto_block(description_body: str) -> str:
    """``[自动-描述]`` from the ``### 风格`` / ``### 基本情况`` text of the description."""
    return block_text(AUTO, description_body)


def empty_auto_block() -> str:
    return block_text(AUTO, "")


def manual_block(style_lines: Sequence[str] = (), fact_lines: Sequence[str] | None = ()) -> str:
    """``[手动]``; ``fact_lines=None`` leaves out the ``### 事实`` part (pre-holdout card)."""
    body = subsection(STYLE, list(style_lines))
    if fact_lines is not None:
        body += subsection(FACTS, list(fact_lines))
    return block_text(MANUAL, body)


def dont_block(lines: Sequence[str] = ()) -> str:
    return block_text(DONT, "\n".join(f"- {line}" for line in lines))


def new_card(scope: str, stats: str, auto: str, manual_style: Sequence[str] = ()) -> str:
    """A first version of a card of ``scope``."""
    if scope == "pre_holdout":
        return stats + auto + manual_block(manual_style, None)
    return stats + auto + manual_block(manual_style) + dont_block()


def manual_style_lines(card: CardText) -> list[str]:
    """The lines under ``### 风格`` of ``[手动]``, exactly as written (blank lines dropped)."""
    parts = split_subsections(card.body(MANUAL))
    return [raw.rstrip("\r") for raw in parts.get(STYLE, []) if raw.strip()]


def pre_holdout_manual(live_text: str | None) -> str:
    """The ``[手动]`` block of the pre-holdout card: the live card's style lines only."""
    if live_text is None:
        return manual_block((), None)
    return manual_block(manual_style_lines(split_card(live_text)), None)


def correction_lines(card: CardText) -> list[str]:
    """The corrections written in ``[不要这样]`` (round 11), as plain lines."""
    return content_lines(card.body(DONT).splitlines())
