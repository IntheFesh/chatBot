"""The Markdown layout of a persona card and how its parts are told apart (R-PERS-002).

A card is four sections, each introduced by a ``## [name]`` heading::

    ## [自动-统计规则]   numeric style rules from the profile (R-PROF-005); program rewrites it
    ## [自动-描述]       written by DeepSeek from real conversations; program rewrites it
       ### 风格 / ### 基本情况
    ## [手动]           written by the user; the program never touches it
       ### 风格 / ### 事实
    ## [不要这样]       corrections (round 11); the program never touches it

:func:`split_card` cuts a text into its sections *without changing a byte*: every section is
the exact text from its heading to the next heading, so a rewrite that keeps the ``[手动]`` and
``[不要这样]`` blocks keeps their bytes (the test compares them).  Inside a section the lines
are bullets ``- 标签：内容``; :func:`classify_line` maps a line to a priority tier from its section
and label, which is what the budget trimming of the renderers (:mod:`twin.profile.persona.render`)
needs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum

STATS = "[自动-统计规则]"
AUTO = "[自动-描述]"
MANUAL = "[手动]"
DONT = "[不要这样]"
SECTION_NAMES = (STATS, AUTO, MANUAL, DONT)

STYLE = "风格"
BASICS = "基本情况"
FACTS = "事实"

EMOTIONS = ("开心", "生气", "撒娇", "难过", "拒绝", "道歉")
LABEL_TONE = "语气"
LABEL_CATCHPHRASE = "口头禅"
LABEL_ADDRESS = "称呼"
LABEL_TABOO = "说话禁忌"
LABEL_FACT = "事实"
LABEL_TOPIC = "话题"
LABEL_ATTITUDE = "对用户的态度"

_HEADING = re.compile(r"^## (\[[^\]\r\n]*\])[ \t]*\r?$", re.MULTILINE)
_SUBHEADING = re.compile(r"^### +(.+?)[ \t]*\r?$")
_BULLET = re.compile(r"^\s*(?:[-*•]\s+)?(.*?)\s*$")
_SEPARATORS = ("：", ":")


class CardFormatError(ValueError):
    """The text is not a persona card (unknown or repeated section headings)."""


class Tier(IntEnum):
    """Priority of a line when a rendering must be cut down (smaller is kept longer).

    The order is the one in R-PERS-004: statistics rules, corrections, forms of address and
    catchphrases, what she says in each mood, the user's hand-written facts, the basic
    information, topics, everything else.
    """

    STATS = 0
    DONT = 1
    ADDRESS = 2
    EMOTIONS = 3
    MANUAL_FACTS = 4
    BASICS = 5
    TOPICS = 6
    OTHER = 7


@dataclass(frozen=True)
class CardText:
    """A card cut into its sections, byte for byte."""

    preamble: str
    blocks: tuple[tuple[str, str], ...]  # (section name, text from its heading to the next)

    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.blocks)

    def block(self, name: str) -> str | None:
        return next((text for found, text in self.blocks if found == name), None)

    def body(self, name: str) -> str:
        """The section without its heading line (empty if the section is missing)."""
        block = self.block(name)
        if block is None:
            return ""
        _, _, rest = block.partition("\n")
        return rest

    def assemble(self) -> str:
        return self.preamble + "".join(text for _, text in self.blocks)

    def with_block(self, name: str, text: str) -> CardText:
        """The same card with the block ``name`` replaced; a missing block is put in its place.

        Every other block keeps its exact text.  ``text`` is the block from its heading line on.
        """
        if name not in SECTION_NAMES:
            raise CardFormatError(f"unknown section {name}")
        if self.block(name) is not None:
            return CardText(
                self.preamble,
                tuple((found, text if found == name else old) for found, old in self.blocks),
            )
        rank = SECTION_NAMES.index(name)
        blocks = list(self.blocks)
        position = len(blocks)
        for index, (found, _) in enumerate(blocks):
            if SECTION_NAMES.index(found) > rank:
                position = index
                break
        blocks.insert(position, (name, text))
        return CardText(self.preamble, tuple(blocks))


def split_card(text: str) -> CardText:
    """Cut ``text`` into sections; unknown or repeated headings are an error."""
    matches = list(_HEADING.finditer(text))
    names = [m.group(1) for m in matches]
    unknown = [name for name in names if name not in SECTION_NAMES]
    if unknown:
        raise CardFormatError(f"unknown section heading {unknown[0]}")
    if len(set(names)) != len(names):
        raise CardFormatError("a section appears more than once")
    if not matches:
        raise CardFormatError("the text has no section heading")
    blocks = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append((match.group(1), text[match.start() : end]))
    return CardText(text[: matches[0].start()], tuple(blocks))


def heading(name: str) -> str:
    return f"## {name}\n"


def block_text(name: str, body: str) -> str:
    """A freshly written section: heading, the body, one blank line after it."""
    body = body.strip("\n")
    return heading(name) + (body + "\n\n" if body else "\n")


def subsection(title: str, lines: list[str]) -> str:
    """``### title`` with its lines (blank line after, none when there are no lines)."""
    if not lines:
        return f"### {title}\n\n"
    return f"### {title}\n" + "\n".join(lines) + "\n\n"


def split_subsections(body: str) -> dict[str, list[str]]:
    """The lines under each ``### title`` of a section body (lines before the first are ``""``)."""
    found: dict[str, list[str]] = {}
    current = ""
    for raw in body.splitlines():
        match = _SUBHEADING.match(raw)
        if match:
            current = match.group(1).strip()
            found.setdefault(current, [])
            continue
        found.setdefault(current, []).append(raw)
    return found


def clean_line(raw: str) -> str:
    """A bullet line as text: no bullet mark, no surrounding space; comments become empty."""
    match = _BULLET.match(raw)
    text = match.group(1) if match else raw.strip()
    if text.startswith("<!--") and text.endswith("-->"):
        return ""
    return text


def content_lines(lines: list[str]) -> list[str]:
    """The non-empty lines of a part of a section, as text."""
    return [text for text in (clean_line(raw) for raw in lines) if text]


def split_label(text: str) -> tuple[str | None, str]:
    """``(label, rest)`` of ``标签：内容``; the label is ``None`` when there is no colon."""
    positions = [text.find(sep) for sep in _SEPARATORS if text.find(sep) > 0]
    if not positions:
        return None, text
    cut = min(positions)
    return text[:cut].strip(), text[cut + 1 :].strip()


_ADDRESS_LABELS = frozenset({LABEL_ADDRESS, LABEL_CATCHPHRASE, "对用户的称呼"})
_EMOTION_LABELS = frozenset({*EMOTIONS, *(f"{e}时" for e in EMOTIONS)})
_TOPIC_LABELS = frozenset({LABEL_TOPIC, "常聊话题", "话题"})
_BASIC_LABELS = frozenset({LABEL_FACT, LABEL_ATTITUDE, "态度"})


def classify_line(section: str, part: str, text: str) -> Tier:
    """The tier of one line; ``section`` is the card section, ``part`` its ``###`` subsection."""
    if section == STATS:
        return Tier.STATS
    if section == DONT:
        return Tier.DONT
    label, _ = split_label(text)
    if section == MANUAL and part == FACTS:
        return Tier.MANUAL_FACTS
    if label in _ADDRESS_LABELS:
        return Tier.ADDRESS
    if label in _EMOTION_LABELS:
        return Tier.EMOTIONS
    if section == AUTO and part == BASICS:
        if label in _TOPIC_LABELS:
            return Tier.TOPICS
        return Tier.BASICS
    return Tier.OTHER


@dataclass(frozen=True)
class CardLine:
    """One line of a card with the place it came from."""

    section: str
    part: str
    text: str
    tier: Tier
    order: int = field(default=0, compare=False)


def card_lines(card: CardText) -> list[CardLine]:
    """Every content line of a card, in card order."""
    out: list[CardLine] = []
    for name in (STATS, AUTO, MANUAL, DONT):
        body = card.body(name)
        if not body.strip():
            continue
        parts = split_subsections(body)
        for part, raw_lines in parts.items():
            for text in content_lines(raw_lines):
                out.append(CardLine(name, part, text, classify_line(name, part, text), len(out)))
    return out
