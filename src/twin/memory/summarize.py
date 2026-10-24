"""Daily summaries: one per local day and scope (R-MEM-002).

``real`` is the summary of what the real records say about the day, ``bot`` the summary of the
bot's own conversation.  The prompt asks for at most 300 characters that keep the people, the
events, the mood, the agreements and what is still open; the schema holds it to that (a longer
reply is sent back once, R-LLM-003).  A day too long for one call (more than
``memory.replay_chunk_lines`` lines) is summarised in pieces and the pieces are merged.

A day is summarised again only when its lines changed (the same input gives the same summary and
is skipped) or when asked to; a recomputed summary is stored as the next *version* of the day
(:meth:`~twin.memory.store.MemoryStore.add_summary`).  The summary is encoded into the summaries
vector table.  Which days are summarised, and when, is for the scheduler (round 08: before she
wakes, off peak); this module is "summarise this day" and its job handler (:mod:`twin.memory.jobs`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from twin.llm.deepseek import DeepSeekClient
from twin.llm.types import LedgerTag, Purpose
from twin.memory.dayload import lines_hash
from twin.memory.extract import DialogueLine
from twin.memory.localdate import BOT, REAL
from twin.memory.memory import Memory
from twin.memory.records import SummaryRecord
from twin.memory.render import WEEKDAY_NAMES
from twin.memory.schemas import SummaryOut
from twin.ops.logging import get_logger
from twin.profile.prompt_templates import MEMORY_SUMMARY, MEMORY_SUMMARY_MERGE, TemplateStore

log = get_logger("twin.memory.summarize")

SCOPE_NOTES = {
    REAL: "这是她和对方之间的真实聊天记录。",
    BOT: (
        "这是对方和一个模仿她的聊天机器人之间的对话，“她”指机器人扮演的角色；"
        "机器人说的关于她自己的细节也要记，但要注明是机器人说的。"
    ),
}
LABELS = {"her": "她", "bot": "她", "user": "对方"}


@dataclass(frozen=True)
class SummaryResult:
    """What summarising a day did."""

    record: SummaryRecord | None
    skipped: str | None = None  # why no summary was made: disabled | no_lines | unchanged
    calls: int = 0
    cost_usd: float = 0.0


class DailySummarizer:
    """Writes the summary of one day (see the module description)."""

    def __init__(
        self,
        memory: Memory,
        client: DeepSeekClient,
        *,
        templates: TemplateStore | None = None,
    ) -> None:
        self._memory = memory
        self._client = client
        services = memory.services
        self._templates = templates or TemplateStore(services.db, services.clock)

    @property
    def enabled(self) -> bool:
        """``memory.daily_summary``: with it off no summary is written."""
        return self._memory.services.settings.memory.daily_summary

    async def summarize(
        self,
        scope: str,
        day: date,
        lines: Sequence[DialogueLine],
        tag: LedgerTag,
        *,
        force: bool = False,
    ) -> SummaryResult:
        """Summarise ``lines`` as the ``scope`` summary of ``day``; store and encode it."""
        if scope not in (REAL, BOT):
            raise ValueError(f"unknown summary scope {scope!r}")
        if not self.enabled:
            return SummaryResult(None, "disabled")
        if not lines:
            return SummaryResult(None, "no_lines")
        digest = lines_hash(lines)
        known = await asyncio.to_thread(self._memory.store.current_summary, scope, day)
        if known is not None and known.input_hash == digest and not force:
            return SummaryResult(known, "unchanged")
        text, calls, cost = await self._text(scope, day, lines, tag)
        record = await asyncio.to_thread(self._store, scope, day, lines, text, digest)
        return SummaryResult(record, None, calls, cost)

    def _store(
        self,
        scope: str,
        day: date,
        lines: Sequence[DialogueLine],
        text: str,
        digest: str,
    ) -> SummaryRecord:
        clock = self._memory.clock
        start, end = clock.bounds_of(scope, day)
        record = self._memory.store.add_summary(
            scope=scope,
            local_date=day,
            timezone=clock.zone_key_for_day(scope, day),
            utc_start=start,
            utc_end=end,
            text=text,
            input_hash=digest,
            template_version=self._templates.active(MEMORY_SUMMARY).ref,
        )
        if scope == BOT:
            self._memory.store.mark_bot_online(min(line.at for line in lines))
        self._memory.sync_vectors(summaries=[record])
        log.info("daily_summary_written", scope=scope, day=day.isoformat(), version=record.version)
        return record

    # ------------------------------------------------------------------ the model

    async def _text(
        self, scope: str, day: date, lines: Sequence[DialogueLine], tag: LedgerTag
    ) -> tuple[str, int, float]:
        size = self._memory.services.settings.memory.replay_chunk_lines
        zone = self._memory.clock.zone_key_for_day(scope, day)
        chunks = [lines[start : start + size] for start in range(0, len(lines), size)]
        parts: list[str] = []
        calls, cost = 0, 0.0
        for chunk in chunks:
            reply = await self._client.chat_json(
                self._templates.active(MEMORY_SUMMARY).render(
                    scope_note=SCOPE_NOTES[scope],
                    date=day.isoformat(),
                    weekday=WEEKDAY_NAMES[day.weekday()],
                    zone=zone,
                    count=len(chunk),
                    dialogue=self._dialogue(scope, chunk),
                ),
                SummaryOut,
                purpose=Purpose.SUMMARY,
                tag=tag,
            )
            parts.append(reply.value.summary)
            calls += reply.attempts
            cost += reply.total_cost_usd
        if len(parts) == 1:
            return parts[0], calls, cost
        merged = await self._client.chat_json(
            self._templates.active(MEMORY_SUMMARY_MERGE).render(
                date=day.isoformat(),
                weekday=WEEKDAY_NAMES[day.weekday()],
                count=len(parts),
                parts="\n".join(f"第{number}段：{text}" for number, text in enumerate(parts, 1)),
            ),
            SummaryOut,
            purpose=Purpose.SUMMARY,
            tag=tag,
        )
        return merged.value.summary, calls + merged.attempts, cost + merged.total_cost_usd

    def _dialogue(self, scope: str, lines: Sequence[DialogueLine]) -> str:
        """The lines with their local clock time (hers, or the bot's, as the scope says)."""
        clock = self._memory.clock
        return "\n".join(
            f"[{line.at.astimezone(clock.zone_of(scope, line.at)):%H:%M}] "
            f"{LABELS[line.speaker]}：{line.text}"
            for line in lines
        )
