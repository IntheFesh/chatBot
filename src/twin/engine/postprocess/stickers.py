"""Turning a ``[表情包:<标签>]`` line into one of her stickers (R-STK-004, R-STK-005, R-ENG-008).

The model only names a tag.  Here the tag is checked against the vocabulary, the selector picks a
sticker of hers for it (her use, how well it fits the conversation, how recently she used it, no
repeat of the last ten bubbles), and the share controller decides whether one more sticker would
push the share of stickers above hers (plus 20 %).  A line that gets no sticker - an unknown tag,
no candidate even among the neighbouring tags, the share too high - is deleted; nothing is ever
invented and nothing is forced in (R-STK-005).
"""

from __future__ import annotations

from collections.abc import Sequence

from twin.engine.postprocess.model import Line, PostContext, Working


def canonical_tag(raw: str, known: Sequence[str]) -> str | None:
    """The vocabulary tag ``raw`` stands for, or ``None``.

    An empty vocabulary means "not checked" and any non-empty tag stands for itself; otherwise the
    tag must be in the vocabulary, or contain one of its tags ("开心的" -> "开心").
    """
    tag = raw.strip().strip("\"'“”‘’「」")
    if not tag:
        return None
    if not known:
        return tag
    if tag in known:
        return tag
    inside = [candidate for candidate in known if candidate in tag]
    return max(inside, key=len) if inside else None


def resolve_stickers(work: Working, ctx: PostContext) -> None:
    """Replace the tags of the sticker lines by stickers; delete the lines that get none."""
    if not any(line.kind == "sticker" for line in work.lines):
        return
    kept: list[Line] = []
    unknown = unmatched = limited = 0
    for line in work.lines:
        if line.kind != "sticker":
            kept.append(line)
            continue
        tag = canonical_tag(line.text, ctx.known_tags)
        if tag is None:
            unknown += 1
            continue
        recent = [*ctx.recent_stickers, *((k.md5 if k.kind == "sticker" else None) for k in kept)]
        record = ctx.chooser(tag, ctx.sticker_context, recent) if ctx.chooser else None
        if record is None:
            unmatched += 1
            continue
        pending = [k.kind == "sticker" for k in kept]
        if ctx.rate is not None and ctx.rate.should_drop_after(pending):
            limited += 1
            continue
        kept.append(Line("sticker", tag, md5=record.md5))
    work.lines = kept
    work.act("sticker_unknown_tag", unknown)
    work.act("sticker_no_match", unmatched)
    work.act("sticker_rate_drop", limited)
