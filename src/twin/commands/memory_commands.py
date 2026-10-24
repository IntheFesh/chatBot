"""``/记住``, ``/忘掉``, ``/记忆``: the memory under the user's control (R-MEM-009, round 11).

The three commands are the chat side of :class:`~twin.memory.manage.MemoryManager`, which is also
what ``twin memory`` calls:

``/记住 <内容>``        stores the text as a fact of source ``user_command`` (the highest rank); the
                      reply names its number and shows what was understood;
``/忘掉 <内容或编号>``   deletes the fact named by its number or by words that match exactly one
                      fact, **and** what was made only from it (follow-ups, life line entries);
                      the reply lists what was deleted - number and a short summary of each.  Words
                      that match several facts delete nothing and list the candidates with their
                      numbers;
``/记忆 [页码|关键词]``  the current facts, newest first, ten to a page.

A command message is not conversation (R-CMD-001): the engine marks its row ``is_command``, so
the extractor that reads the bot's conversation never sees ``/记住 ...`` - the text is stored once,
by this command, with the rank the user's own order has.
"""

from __future__ import annotations

import asyncio

from twin.commands import texts
from twin.commands.registry import CommandCall, UsageError
from twin.memory.manage import ForgetResult, MemoryItem, MemoryManager, MemoryPage, RememberResult

MAX_SHOWN_CANDIDATES = 8


def snippet(text: str, limit: int = texts.MEMORY_SNIPPET_CHARS) -> str:
    """``text`` on one line, cut to ``limit`` characters."""
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 1] + "…"


class MemoryCommands:
    """The handlers of this module (see the module description)."""

    def __init__(self, manager: MemoryManager) -> None:
        self._manager = manager

    # --------------------------------------------------------------------------- /记住

    async def remember(self, call: CommandCall) -> str:
        if not call.args:
            raise UsageError("")
        try:
            result = await self._manager.remember(call.args)
        except ValueError:
            raise UsageError("") from None
        return render_remembered(result)

    # --------------------------------------------------------------------------- /忘掉

    async def forget(self, call: CommandCall) -> str:
        if not call.args:
            raise UsageError("")
        result = await self._manager.aforget(call.args)
        return render_forgotten(result)

    # --------------------------------------------------------------------------- /记忆

    async def memory(self, call: CommandCall) -> str:
        page, keyword = 1, None
        if call.args.isdigit():
            page = max(1, int(call.args))
        elif call.args:
            keyword = call.args
        listing = await asyncio.to_thread(self._manager.list_items, page, keyword)
        return render_listing(listing)


def render_remembered(result: RememberResult) -> str:
    """The answer to ``/记住``."""
    if not result.facts:
        return texts.REMEMBER_AS_WRITTEN
    first = result.facts[0]
    reply = texts.REMEMBER_DONE.format(number=first.number, text=snippet(first.text, 80))
    if result.followups:
        reply += texts.REMEMBER_FOLLOWUPS.format(count=result.followups)
    if result.replaced:
        reply += texts.REMEMBER_REPLACED.format(count=result.replaced)
    if not result.enriched:
        reply += texts.REMEMBER_AS_WRITTEN
    return reply


def _numbered(items: tuple[MemoryItem, ...]) -> list[str]:
    return [
        texts.FORGET_FACT.format(number=item.number, text=snippet(item.text))
        for item in items[:MAX_SHOWN_CANDIDATES]
    ]


def render_forgotten(result: ForgetResult) -> str:
    """The answer to ``/忘掉``: what was deleted, or why nothing was."""
    if result.ambiguous:
        example = result.ambiguous[0].number
        head = texts.FORGET_AMBIGUOUS.format(count=len(result.ambiguous), example=example)
        return "\n".join([head, *_numbered(result.ambiguous)])
    if result.followup_matches:
        lines = [snippet(text) for text in result.followup_matches[:MAX_SHOWN_CANDIDATES]]
        return "\n".join([texts.FORGET_FOLLOWUPS_AMBIGUOUS, *lines])
    if not result.deleted:
        return texts.FORGET_NONE
    lines = [texts.FORGET_DONE]
    for gone in result.deleted:
        if gone.kind == "fact" and gone.number is not None:
            lines.append(texts.FORGET_FACT.format(number=gone.number, text=snippet(gone.text)))
        elif gone.kind == "followup":
            lines.append(texts.FORGET_FOLLOWUP.format(text=snippet(gone.text)))
        else:
            lines.append(texts.FORGET_LIFELINE.format(text=snippet(gone.text)))
    if result.restored:
        numbers = "、".join(str(n) for n in result.restored)
        lines.append(texts.FORGET_RESTORED.format(numbers=numbers))
    return "\n".join(lines)


def render_listing(listing: MemoryPage) -> str:
    """The answer to ``/记忆``."""
    if not listing.items and not listing.followups:
        if listing.keyword:
            return texts.MEMORY_NO_MATCH.format(word=listing.keyword)
        return texts.MEMORY_EMPTY
    lines: list[str] = []
    for item in listing.items:
        when = texts.MEMORY_WHEN.format(date=item.event_date) if item.event_date else ""
        lines.append(
            texts.MEMORY_LINE.format(number=item.number, text=snippet(item.text), when=when)
        )
    lines.extend(texts.MEMORY_FOLLOWUP.format(text=snippet(text)) for text in listing.followups)
    footer = texts.MEMORY_FOOTER.format(page=listing.page, pages=listing.pages, total=listing.total)
    if listing.page < listing.pages:
        footer += texts.MEMORY_NEXT.format(page=listing.page + 1)
    lines.append(footer)
    return "\n".join(lines)
