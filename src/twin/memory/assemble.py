"""Building the memory block of a reply: recall, score, fit into the budget (R-MEM-008).

``MemoryAssembler.build(query, now, budget_tokens)`` answers "what should she remember right
now?" in four steps:

1. **recall** - from the memory *as it is at ``now``* (:func:`twin.memory.view.MemoryView`; the
   live block uses exactly the rules that the training export uses for a past moment):
   facts that fit the topic by meaning (vector) and by keyword, optionally limited to some
   subjects; facts that name a day near today (a birthday, an anniversary, an exam: tomorrow
   or today); the few facts of top importance; the summaries of the last days and of the days
   that fit the topic; today's life line; the follow-ups that are open and near.
2. **score** - ``similarity``, ``importance``, ``recency``, ``source`` priority and ``date``
   relevance, weighted by ``memory.weights`` (R-CFG-004).
3. **fit** - by priority (what must be said today, then facts by score, the life line, the
   recent days, the earlier days) into the token budget: ``memory.block_tokens`` unless the
   caller gives one, multiplied by the budget level's factor (R-LLM-008: halved from level 2,
   a quarter at level 4).  A follow-up that is due today and a fact whose day is today or
   tomorrow are mandatory: they are in the block whatever the budget (at most
   :data:`MAX_MANDATORY` of each kind, so a tiny budget cannot be overrun without bound).
4. **write** - sections with small headings (:mod:`twin.memory.render`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from twin.config.settings import MemoryConfig
from twin.llm.budget import BudgetLimits
from twin.llm.tokens import TokenEstimator
from twin.memory.blocks import BlockItem, MemoryBlock, MemoryQuery
from twin.memory.memory import Memory
from twin.memory.records import FactRecord, FollowupRecord, LifelineRecord, SummaryRecord
from twin.memory.render import (
    SECTION_EARLIER,
    SECTION_FACTS,
    SECTION_LIFELINE,
    SECTION_ORDER,
    SECTION_RECENT,
    SECTION_TODAY,
    fact_line,
    followup_line,
    heading,
    summary_line,
)
from twin.memory.view import MemoryView
from twin.memory.visible import date_score, occurrence_offset

MAX_MANDATORY = 5
CORE_FACTS_TAKEN = 5
PRIORITY = (SECTION_TODAY, SECTION_FACTS, SECTION_LIFELINE, SECTION_RECENT, SECTION_EARLIER)
HEADING_OVERHEAD = 2  # newline and separator tokens around a heading


@dataclass(frozen=True)
class _Entry:
    kind: str
    item_id: str
    section: str
    text: str
    score: float
    mandatory: bool
    sort_key: tuple[float, ...]


def _minutes_of(clock: str | None) -> float:
    """Minutes after midnight of an ``HH:MM`` text; an event without a start sorts last."""
    if not clock or ":" not in clock:
        return float(24 * 60 + 1)
    hours, _, minutes = clock.partition(":")
    return (
        float(int(hours) * 60 + int(minutes)) if hours.isdigit() and minutes.isdigit() else 1441.0
    )


class MemoryAssembler:
    """Recalls, scores and fits the memory into a block (see the module description)."""

    def __init__(
        self,
        memory: Memory,
        *,
        estimator: TokenEstimator | None = None,
        limits: Callable[[], BudgetLimits] | None = None,
    ) -> None:
        self._memory = memory
        self._estimator = estimator or TokenEstimator()
        self._limits = limits

    # ------------------------------------------------------------------ entry points

    def view(
        self, moment: datetime, *, include_lifeline: bool = True, refresh: bool = True
    ) -> MemoryView:
        """The memory at ``moment`` whose :meth:`~MemoryView.render` is this assembler."""
        return MemoryView(
            self._memory,
            moment,
            include_lifeline=include_lifeline,
            refresh=refresh,
            renderer=self.render_view,
        )

    def build(
        self, query_context: MemoryQuery, now: datetime, budget_tokens: int | None = None
    ) -> MemoryBlock:
        """The memory block for a reply at ``now`` (blocking: the vector search reads disk)."""
        return self.render_view(self.view(now), query_context, budget_tokens)

    async def abuild(
        self, query_context: MemoryQuery, now: datetime, budget_tokens: int | None = None
    ) -> MemoryBlock:
        return await asyncio.to_thread(self.build, query_context, now, budget_tokens)

    def effective_budget(self, budget_tokens: int | None) -> int:
        """The budget in force: the asked-for one times the factor of the budget level."""
        base = budget_tokens if budget_tokens is not None else self._memory_config().block_tokens
        factor = self._limits().memory_budget_factor if self._limits is not None else 1.0
        return max(0, int(base * factor))

    # ----------------------------------------------------------------------- rendering

    def render_view(
        self, view: MemoryView, query: MemoryQuery, budget_tokens: int | None = None
    ) -> MemoryBlock:
        budget = self.effective_budget(budget_tokens)
        entries = self._recall(view, query)
        return self._fit(entries, budget)

    def _memory_config(self) -> MemoryConfig:
        return self._memory.services.settings.memory

    # ------------------------------------------------------------------------ recall

    def _recall(self, view: MemoryView, query: MemoryQuery) -> list[_Entry]:
        config = self._memory_config()
        today = view.today()
        now = view.as_of
        entries: dict[tuple[str, str], _Entry] = {}

        def put(entry: _Entry) -> None:
            key = (entry.kind, entry.item_id)
            old = entries.get(key)
            if old is None or (entry.mandatory, entry.score) > (old.mandatory, old.score):
                entries[key] = entry

        # facts: by meaning and keyword, the always-relevant ones, the ones with a near date
        relevance: dict[str, float] = {}
        records: dict[str, FactRecord] = {}
        for hit in view.search_facts(query.text, config.recall_facts):
            if query.subjects is not None and hit.fact.subject not in query.subjects:
                continue
            records[hit.fact.id] = hit.fact
            relevance[hit.fact.id] = min(1.0, max(0.0, hit.relevance))
        core = sorted(view.core_facts(), key=lambda f: (-f.importance, -f.number))
        for fact in core[:CORE_FACTS_TAKEN]:
            if query.subjects is None or fact.subject in query.subjects:
                records.setdefault(fact.id, fact)
        offsets: dict[str, int] = {}
        for fact in view.dated_facts():
            if fact.event_date is None:
                continue
            distance = occurrence_offset(fact.event_date, fact.recurrence, today)
            if date_score(distance) > 0:
                records[fact.id] = fact
                offsets[fact.id] = distance
        mandatory_facts = sorted(
            (f for f in records.values() if offsets.get(f.id) in (0, 1)),
            key=lambda f: (-f.importance, f.number),
        )[:MAX_MANDATORY]
        mandatory_ids = {f.id for f in mandatory_facts}
        for fact in records.values():
            offset = offsets.get(fact.id)
            score = self._score(fact, relevance.get(fact.id, 0.0), offset, now)
            must = fact.id in mandatory_ids
            put(
                _Entry(
                    "fact",
                    fact.id,
                    SECTION_TODAY if must else SECTION_FACTS,
                    fact_line(fact, offset if offset is not None and date_score(offset) else None),
                    score,
                    must,
                    (-score,),
                )
            )

        # follow-ups: due today (mandatory) or open and near
        zone = view.zone()
        lookahead = timedelta(hours=config.followup_lookahead_h)
        due_today: list[FollowupRecord] = []
        near: list[FollowupRecord] = []
        for followup in view.followups():
            if followup.window_end < now:
                continue  # its time has passed for good
            if followup.due_at.astimezone(zone).date() == today:
                due_today.append(followup)
            elif followup.due_at <= now + lookahead:
                near.append(followup)
        for must, group in ((True, due_today[:MAX_MANDATORY]), (False, near)):
            for followup in group:
                put(
                    _Entry(
                        "followup",
                        followup.id,
                        SECTION_TODAY,
                        followup_line(followup, zone),
                        1.0 if must else 0.8,
                        must,
                        (followup.due_at.timestamp(),),
                    )
                )

        # life line of today
        for event in view.lifeline():
            if event.local_date == today:
                put(self._lifeline_entry(event))

        # summaries: the last days, then the days that fit the topic
        recent_ids: set[str] = set()
        for back in range(1, config.recent_summary_days + 1):
            day = today - timedelta(days=back)
            for scope in ("bot", "real"):
                summary = view.summary_for(scope, day)
                if summary is not None:
                    recent_ids.add(summary.id)
                    put(self._summary_entry(summary, SECTION_RECENT, 1.0 - 0.1 * back, (back,)))
        if config.recall_summaries:
            taken = 0
            for summary_hit in view.search_summaries(
                query.text, config.recall_summaries + len(recent_ids)
            ):
                if summary_hit.summary.id in recent_ids or taken >= config.recall_summaries:
                    continue
                taken += 1
                put(
                    self._summary_entry(
                        summary_hit.summary,
                        SECTION_EARLIER,
                        summary_hit.relevance,
                        (-summary_hit.relevance,),
                    )
                )
        return list(entries.values())

    def _score(
        self, fact: FactRecord, relevance: float, offset: int | None, now: datetime
    ) -> float:
        config = self._memory_config()
        weights = config.weights
        age_days = max(0.0, (now - fact.known_at).total_seconds() / 86_400.0)
        recency = float(0.5 ** (age_days / config.recency_half_life_days))
        return (
            weights.similarity * relevance
            + weights.importance * fact.importance / 5.0
            + weights.recency * recency
            + weights.source * fact.priority / 4.0
            + weights.date * (date_score(offset) if offset is not None else 0.0)
        )

    @staticmethod
    def _lifeline_entry(event: LifelineRecord) -> _Entry:
        return _Entry(
            "lifeline",
            event.id,
            SECTION_LIFELINE,
            event.line(),
            0.5,
            False,
            (_minutes_of(event.start_local),),
        )

    @staticmethod
    def _summary_entry(
        summary: SummaryRecord, section: str, score: float, key: tuple[float, ...]
    ) -> _Entry:
        return _Entry("summary", summary.id, section, summary_line(summary), score, False, key)

    # --------------------------------------------------------------------------- fit

    def _fit(self, entries: list[_Entry], budget: int) -> MemoryBlock:
        rank = {section: index for index, section in enumerate(PRIORITY)}
        ordered = sorted(
            entries,
            key=lambda e: (not e.mandatory, rank[e.section], e.sort_key, e.item_id),
        )
        used = 0
        opened: set[str] = set()
        chosen: list[tuple[_Entry, int]] = []
        dropped = 0
        for entry in ordered:
            cost = self._estimator.estimate_text(entry.text) + 1
            header = (
                0
                if entry.section in opened
                else self._estimator.estimate_text(heading(entry.section)) + HEADING_OVERHEAD
            )
            if not entry.mandatory and used + cost + header > budget:
                dropped += 1
                continue
            used += cost + header
            opened.add(entry.section)
            chosen.append((entry, cost))
        if not chosen:
            return MemoryBlock("", (), 0, budget, dropped, ())
        lines: list[str] = []
        sections: list[str] = []
        for section in SECTION_ORDER:
            group = sorted(
                (pair for pair in chosen if pair[0].section == section),
                key=lambda pair: (not pair[0].mandatory, pair[0].sort_key, pair[0].item_id),
            )
            if not group:
                continue
            sections.append(section)
            lines.append(heading(section))
            lines.extend(f"- {entry.text}" for entry, _ in group)
        items = tuple(
            BlockItem(e.kind, e.item_id, e.section, e.score, e.mandatory, cost)
            for e, cost in chosen
        )
        return MemoryBlock("\n".join(lines), items, used, budget, dropped, tuple(sections))
