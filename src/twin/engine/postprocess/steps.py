"""The post-processing steps, one function each (R-ENG-008, R-ENG-012, R-SAFE-002, R-SAFE-006).

Every step takes the :class:`~twin.engine.postprocess.model.Working` reply and the
:class:`~twin.engine.postprocess.model.PostContext`, changes the lines and leaves two kinds of
trace: **actions** (what was done - removed, split, merged - with counts, never with text) and
**violations** (a hard rule was broken; the pipeline asks the model again).  The order of
:data:`STEPS` is the order of R-ENG-008; :func:`remove_think` runs first on the raw text because a
thought can span lines.

====================  ==============================================================
``ai_tone``           Markdown and list marks out; customer-service wording: the bubble
                      goes; the bot calling itself an AI: a violation (unless she was
                      sincerely asked, or the line only uses a word the user used)
``token_leak``        a redaction token such as ``[手机号]`` in the output: a violation
``event_text``        ``[图片]``, ``[语音 5 秒]`` ...: the line goes; nothing left: violation
``commitments``       "我给你打电话": a violation (on the last attempt the bubble goes)
``language``          not Chinese chat at all: a violation
``punctuation``       her punctuation: no final stops, comma sentences become bubbles
``length``            bubbles longer than 1.5 x her 95th percentile are split, never cut
``bubble_cap``        at most 1.5 x her 90th percentile bubbles (and ``engine.max_bubbles``)
``dedupe``            the same bubble twice in a row: once
``emoji_codes``       only her codes, in her runs, at her rate
``stickers``          a tag becomes one of her stickers, or the line goes (see ``stickers``)
``quote``             a quote the channel cannot show is dropped
``no_reply``          ``[不回]`` only when it is allowed and alone
``emptiness``         nothing left to send: a violation
====================  ==============================================================
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable

from twin.engine.postprocess.model import Line, PostContext, Working
from twin.engine.postprocess.stickers import resolve_stickers
from twin.engine.postprocess.text import (
    clamp_marks,
    split_semantic,
    squeeze_spaces,
    strip_list_marker,
    strip_trailing_stops,
)
from twin.llm.redaction import find_tokens
from twin.profile.textstats import emoji_code_runs

Step = Callable[[Working, PostContext], None]

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = "</think>"
_SINGLE_STOP = re.compile(r"(?<!。)。(?!。)|(?<![\d.])\.(?=\s|\Z)")
_COMMA = re.compile(r"[，,]")
_QUESTION_BREAK = re.compile(r"(?<=[？?！!])(?=[^\s？?！!]{2,})")
_CODE = re.compile(r"\[[一-龥A-Za-z]{1,6}\]")
_LATIN = re.compile(r"[A-Za-z]")
_CJK = re.compile(r"[一-鿿]")
MIN_PIECE_CHARS = 2
LANGUAGE_MIN_LETTERS = 6
LANGUAGE_MIN_CJK_SHARE = 0.3


def remove_think(raw: str) -> tuple[str, int]:
    """``raw`` without ``<think>...</think>`` (and an unclosed or half-open one); the count."""
    text, closed = _THINK_BLOCK.subn("", raw)
    text, opened = _THINK_OPEN.subn("", text)
    stray = 0
    if _THINK_CLOSE in text.lower():
        stray = 1
        text = re.split(_THINK_CLOSE, text, flags=re.IGNORECASE)[-1]
    return text.strip(), closed + opened + stray


# ----------------------------------------------------------------------- tone and leaks


def _topic_only(hits: tuple[str, ...], user_text: str) -> bool:
    """Every self-identification phrase of the line is a word the user used in this round.

    Talking about "人工智能" with someone who brought it up is not calling herself one.  The
    line "我是人工智能" matches two phrases, "我是人工智能" and "人工智能"; the user said only
    the second, so it stays a slip.
    """
    said = user_text.lower()
    return bool(said) and all(hit in said for hit in hits)


def ai_tone(work: Working, ctx: PostContext) -> None:
    kept: list[Line] = []
    markup = lists = dropped = 0
    for line in work.lines:
        if line.kind != "text":
            kept.append(line)
            continue
        text = line.text
        if ctx.ai_phrases.has_markup(text):
            text = ctx.ai_phrases.strip_markup(text)
            markup += 1
        listed = strip_list_marker(text)
        if listed != text:
            lists += 1
            text = listed
        text = text.strip()
        if not text:
            continue
        hits = ctx.ai_phrases.self_reference_hits(text)
        if hits:
            if ctx.allow_ai_admission:
                work.act("ai_admission_kept")
            elif _topic_only(hits, ctx.user_text):
                work.act("ai_topic_kept")
            else:
                work.violate("ai_self_reference")
        elif ctx.ai_phrases.find_register(text):
            dropped += 1
            continue
        line.text = text
        kept.append(line)
    work.lines = kept
    work.act("markup_removed", markup)
    work.act("list_marker_removed", lists)
    work.act("ai_phrase_removed", dropped)


def token_leak(work: Working, ctx: PostContext) -> None:
    found: set[str] = set()
    for line in work.text_lines():
        found.update(find_tokens(line.text))
    if work.quote:
        found.update(find_tokens(work.quote))
    if found:
        work.violate("token_leak", ",".join(sorted(found)))


def event_text(work: Working, ctx: PostContext) -> None:
    kept = [
        line
        for line in work.lines
        if not (line.kind == "text" and ctx.detector.is_event_text(line.text))
    ]
    removed = len(work.lines) - len(kept)
    work.lines = kept
    work.act("event_text_removed", removed)
    if removed and not kept:
        work.violate("event_text_only")


def commitments(work: Working, ctx: PostContext) -> None:
    if ctx.commitments is None:
        return
    first: int | None = None
    flagged = 0
    for line in work.text_lines():
        found = ctx.commitments.find(line.text)
        if found is not None:
            line.promises = True
            flagged += 1
            first = found if first is None else first
    if not flagged:
        return
    if ctx.last_attempt:
        work.lines = [line for line in work.lines if not line.promises]
        work.act("commitment_removed", flagged)
    else:
        work.violate("commitment", f"pattern#{first}")


def language(work: Working, ctx: PostContext) -> None:
    body = _CODE.sub("", "".join(line.text for line in work.text_lines()))
    latin = len(_LATIN.findall(body))
    cjk = len(_CJK.findall(body))
    letters = latin + cjk
    if letters >= LANGUAGE_MIN_LETTERS and cjk / letters < LANGUAGE_MIN_CJK_SHARE:
        work.violate("not_chinese")


# ------------------------------------------------------------------------- punctuation


def _split_commas(piece: str, threshold: int) -> list[str]:
    """Cut a comma sentence into bubbles (pieces under two characters stay glued)."""
    if len(piece) <= threshold or not _COMMA.search(piece):
        return [piece]
    out: list[str] = []
    for part in _COMMA.split(piece):
        part = part.strip()
        if not part:
            continue
        if out and len(part) < MIN_PIECE_CHARS:
            out[-1] += part
        else:
            out.append(part)
    return out or [piece]


def normalise_line(text: str, ctx: PostContext) -> list[str]:
    """One text line written the way she writes (see :func:`punctuation`)."""
    style = ctx.style
    text = clamp_marks(squeeze_spaces(text))
    pieces = [text]
    if style.strips_periods:
        pieces = [p for piece in pieces for p in _SINGLE_STOP.split(piece)]
    if style.splits_commas:
        threshold = style.comma_split_threshold()
        pieces = [p for piece in pieces for p in _split_commas(piece.strip(), threshold)]
        pieces = [p for piece in pieces for p in _QUESTION_BREAK.split(piece)]
    cleaned = [strip_trailing_stops(piece).strip() for piece in pieces]
    return [piece for piece in cleaned if piece]


def punctuation(work: Working, ctx: PostContext) -> None:
    out: list[Line] = []
    changed = 0
    for line in work.lines:
        if line.kind != "text":
            out.append(line)
            continue
        pieces = normalise_line(line.text, ctx)
        if pieces != [line.text]:
            changed += 1
        out.extend(Line("text", piece, promises=line.promises) for piece in pieces)
    work.lines = out
    work.act("punctuation_normalised", changed)


# ------------------------------------------------------------------- length and number


def length(work: Working, ctx: PostContext) -> None:
    out: list[Line] = []
    split = 0
    limit = ctx.style.max_chars
    for line in work.lines:
        if line.kind != "text" or len(line.text) <= limit:
            out.append(line)
            continue
        pieces = split_semantic(line.text, limit)
        if len(pieces) > 1:
            split += 1
        out.extend(Line("text", piece, promises=line.promises) for piece in pieces)
    work.lines = out
    work.act("long_bubble_split", split)


def bubble_cap(work: Working, ctx: PostContext) -> None:
    cap = ctx.style.max_bubbles
    dropped = 0
    while len(work.lines) > cap:
        stickers = [i for i, line in enumerate(work.lines) if line.kind == "sticker"]
        del work.lines[stickers[-1] if stickers else -1]
        dropped += 1
    work.act("bubble_cap", dropped)


def dedupe(work: Working, ctx: PostContext) -> None:
    out: list[Line] = []
    removed = 0
    for line in work.lines:
        if out and out[-1].kind == line.kind and out[-1].text.strip() == line.text.strip():
            removed += 1
            continue
        out.append(line)
    work.lines = out
    work.act("duplicate_removed", removed)


# --------------------------------------------------------------------------- emoji codes


def _trim_runs(text: str, longest: int) -> str:
    """Keep at most ``longest`` bracket codes in a run of adjacent codes."""
    out: list[str] = []
    position = 0
    run = 0
    previous_end = -1
    for match in _CODE.finditer(text):
        run = run + 1 if match.start() == previous_end else 1
        out.append(text[position : match.start()])
        if run <= longest:
            out.append(match.group(0))
        position = match.end()
        previous_end = match.end()
    out.append(text[position:])
    return "".join(out)


def emoji_codes(work: Working, ctx: PostContext) -> None:
    policy = ctx.emoji
    removed = trimmed = 0
    kept: list[Line] = []
    for line in work.lines:
        if line.kind != "text":
            kept.append(line)
            continue
        original = line.text
        if policy is None:
            text = _CODE.sub("", original).strip()
        else:
            text = policy.strip_disallowed(original).strip()
        if text != original:
            removed += 1
        longest = ctx.style.max_code_run
        if longest is not None and any(run > longest for run in emoji_code_runs(text)):
            text = _trim_runs(text, longest).strip()
            trimmed += 1
        if text:
            line.text = text
            kept.append(line)
    work.lines = kept
    work.act("emoji_code_removed", removed)
    work.act("emoji_run_trimmed", trimmed)
    # her rate: at most as many bubbles with a code as her share of them asks for
    texts = [line for line in work.lines if line.kind == "text"]
    rate = policy.expected_rate() if policy is not None else 0.0
    allowed = math.ceil(rate * len(texts)) if rate > 0 else 0
    with_codes = [line for line in texts if _CODE.search(line.text)]
    excess = with_codes[allowed:] if allowed < len(with_codes) else []
    emptied: set[int] = set()
    for line in reversed(excess):
        stripped = _CODE.sub("", line.text).strip()
        if stripped:
            line.text = stripped
        else:
            emptied.add(id(line))
    if excess:
        work.lines = [line for line in work.lines if id(line) not in emptied]
        work.act("emoji_rate_trimmed", len(excess))


# ------------------------------------------------------------------------ the rest


def stickers(work: Working, ctx: PostContext) -> None:
    resolve_stickers(work, ctx)


def quote(work: Working, ctx: PostContext) -> None:
    if work.quote is None:
        return
    if not ctx.supports_quote:
        work.quote = None
        work.act("quote_removed", detail="channel")
    elif not work.text_lines():
        work.quote = None
        work.act("quote_removed", detail="no_text")


def no_reply(work: Working, ctx: PostContext) -> None:
    silent = [line for line in work.lines if line.kind == "no_reply"]
    if not silent:
        return
    others = [line for line in work.lines if line.kind != "no_reply"]
    if ctx.no_reply_allowed and not others:
        work.no_reply = True
        work.lines = []
        work.quote = None
        work.act("no_reply")
        return
    work.lines = others
    work.act("no_reply_ignored", detail="mixed" if others else "not_allowed")
    if not others:
        work.violate("no_reply_not_allowed")


def emptiness(work: Working, ctx: PostContext) -> None:
    if not work.lines and not work.no_reply and not work.violations:
        work.violate("empty")


STEPS: tuple[tuple[str, Step], ...] = (
    ("ai_tone", ai_tone),
    ("token_leak", token_leak),
    ("event_text", event_text),
    ("commitments", commitments),
    ("language", language),
    ("punctuation", punctuation),
    ("length", length),
    ("bubble_cap", bubble_cap),
    ("dedupe", dedupe),
    ("emoji_codes", emoji_codes),
    ("stickers", stickers),
    ("quote", quote),
    ("no_reply", no_reply),
    ("emptiness", emptiness),
)

__all__ = ["STEPS", "Step", "remove_think"]
