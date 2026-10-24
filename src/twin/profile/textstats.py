"""What one text message contains: punctuation, emoji, laughter, particles (R-PROF-002).

All functions are pure and read a single message text.  The results are counts and closed
vocabularies (punctuation groups, WeChat emoji codes, emoji characters, sentence-final
particles); no function returns a piece of the message itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# bracket codes such as "[拥抱]" (R-PROF-002); the table of known codes is config data
EMOJI_CODE_RE = re.compile(r"\[[一-龥A-Za-z]{1,6}\]")

PUNCT_GROUPS: tuple[str, ...] = (
    "comma",
    "period",
    "question",
    "exclaim",
    "tilde",
    "ellipsis",
    "pause",
    "space",
)
END_CLASSES: tuple[str, ...] = (*PUNCT_GROUPS[:-1], "emoji_code", "none")

_ELLIPSIS_RE = re.compile(r"…|\.{3,}|。{3,}")
_END_ELLIPSIS_RE = re.compile(r"(?:…|\.{3,}|。{3,})\Z")
_END_CODE_RE = re.compile(r"\[[一-龥A-Za-z]{1,6}\]\Z")
_PERIOD_RE = re.compile(r"。|(?<![\d.])\.(?![\d.])")
_COMMA_CHARS = ("，", ",")
_QUESTION_CHARS = ("？", "?")
_EXCLAIM_CHARS = ("！", "!")
_TILDE_CHARS = ("～", "~", "〜")
_END_CHAR_CLASS = {
    "。": "period",
    ".": "period",
    "，": "comma",
    ",": "comma",
    "？": "question",
    "?": "question",
    "！": "exclaim",
    "!": "exclaim",
    "～": "tilde",
    "~": "tilde",
    "〜": "tilde",
    "…": "ellipsis",
    "、": "pause",
}
_TRAILING_PUNCT = "，。！？～~…、,.!?;；:：\"'“”‘’ \t\r\n　"
# single characters that carry the mood at the end of a sentence (a closed set, not text)
PARTICLES = frozenset("啊呀哇啦吧嘛呢哦噢喔哈嘿嘞咯呐哒耶哟诶欸嗯哼喽么吗呗咧捏嘻哩噻滴勒")
_LAUGH_RE = re.compile(r"哈{2,}")
_LATIN_LAUGH_RE = re.compile(r"[hH]{3,}")

# --------------------------------------------------------------- emoji characters

_EMOJI_ELEMENT = (
    r"(?:[\U0001F300-\U0001FAFF☀-➿⭐⭕⌚⌛⏩-⏳⏸-⏺"
    r"▪▫▶◀◻-◾⤴⤵⬅-⬇⬛⬜〰〽"
    r"㊗㊙\U0001F004\U0001F0CF\U0001F170\U0001F171\U0001F17E\U0001F17F\U0001F18E"
    r"\U0001F191-\U0001F19A\U0001F201\U0001F202\U0001F21A\U0001F22F\U0001F232-\U0001F23A"
    r"\U0001F250\U0001F251]"
    r"|[©®‼⁉™ℹ↔-↙↩↪⌨⏏Ⓜ]️)"
    r"[️\U0001F3FB-\U0001F3FF]*"
)
EMOJI_RE = re.compile(
    r"[\U0001F1E6-\U0001F1FF]{2}"  # flags
    r"|[0-9#*]️?⃣"  # keycaps
    rf"|{_EMOJI_ELEMENT}(?:‍{_EMOJI_ELEMENT})*"  # pictographs, skin tones, ZWJ sequences
)


def find_unicode_emoji(text: str) -> list[str]:
    """Emoji characters (and sequences: flags, keycaps, skin tones, ZWJ) in ``text``."""
    return EMOJI_RE.findall(text)


def find_emoji_codes(text: str) -> list[str]:
    """Bracket codes such as ``[拥抱]`` in order of appearance."""
    return EMOJI_CODE_RE.findall(text)


def emoji_code_runs(text: str) -> list[int]:
    """Lengths of the runs of directly adjacent bracket codes (``[A][B][C]`` is one run of 3)."""
    runs: list[int] = []
    end = -1
    current = 0
    for match in EMOJI_CODE_RE.finditer(text):
        if match.start() == end:
            current += 1
        else:
            if current:
                runs.append(current)
            current = 1
        end = match.end()
    if current:
        runs.append(current)
    return runs


# -------------------------------------------------------------------- punctuation


def punctuation_groups(text: str) -> frozenset[str]:
    """Which punctuation groups occur in ``text`` (a group counts once per message)."""
    found: set[str] = set()
    if any(c in text for c in _COMMA_CHARS):
        found.add("comma")
    if _PERIOD_RE.search(text):
        found.add("period")
    if any(c in text for c in _QUESTION_CHARS):
        found.add("question")
    if any(c in text for c in _EXCLAIM_CHARS):
        found.add("exclaim")
    if any(c in text for c in _TILDE_CHARS):
        found.add("tilde")
    if _ELLIPSIS_RE.search(text):
        found.add("ellipsis")
    if "、" in text:
        found.add("pause")
    if " " in text.strip() or "　" in text.strip():
        found.add("space")
    return frozenset(found)


def ending_class(text: str) -> str:
    """How the message ends: a punctuation group, ``emoji_code`` or ``none``."""
    stripped = text.rstrip()
    if not stripped:
        return "none"
    if _END_CODE_RE.search(stripped):
        return "emoji_code"
    if _END_ELLIPSIS_RE.search(stripped):
        return "ellipsis"
    return _END_CHAR_CLASS.get(stripped[-1], "none")


def strip_trailing(text: str) -> str:
    """``text`` without trailing punctuation, spaces and bracket emoji codes."""
    current = text
    while True:
        trimmed = _END_CODE_RE.sub("", current.rstrip(_TRAILING_PUNCT))
        if trimmed == current:
            return trimmed
        current = trimmed


def final_particle(text: str) -> str | None:
    """The sentence-final particle of ``text`` (from the closed set), if it ends with one."""
    body = strip_trailing(text)
    if not body:
        return None
    last = body[-1]
    return last if last in PARTICLES else None


def laugh_runs(text: str) -> list[int]:
    """Lengths of the runs of "哈" (two or more in a row) in ``text``."""
    return [len(m.group(0)) for m in _LAUGH_RE.finditer(text)]


def has_latin_laugh(text: str) -> bool:
    """Whether the text laughs with a run of ``h`` (as in "hhh")."""
    return _LATIN_LAUGH_RE.search(text) is not None


def is_question(text: str) -> bool:
    """A question mark anywhere, or a trailing 吗."""
    if any(c in text for c in _QUESTION_CHARS):
        return True
    return strip_trailing(text).endswith("吗")


@dataclass(frozen=True, slots=True)
class TextFacts:
    """Everything the profile reads from one text, computed once."""

    length: int
    groups: frozenset[str]
    ending: str
    codes: list[str]
    code_runs: list[int]
    emoji: list[str]
    particle: str | None
    laughs: list[int]
    latin_laugh: bool
    question: bool


def analyse_text(text: str) -> TextFacts:
    """Facts of one message text."""
    body = text.strip()
    return TextFacts(
        length=len(body),
        groups=punctuation_groups(body),
        ending=ending_class(body),
        codes=find_emoji_codes(body),
        code_runs=emoji_code_runs(body),
        emoji=find_unicode_emoji(body),
        particle=final_particle(body),
        laughs=laugh_runs(body),
        latin_laugh=has_latin_laugh(body),
        question=is_question(body),
    )
