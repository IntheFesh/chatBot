"""Deciding what a new fact does to the old ones (R-MEM-004, R-MEM-011).

Before a fact is stored, :class:`ConflictResolver` recalls the older facts it could clash with
(close by meaning - the vector search - or by keyword, and about the same subject) and, for a new
fact that outranks the bot's own inventions, the life line entries of the days it speaks of.
DeepSeek judges each pair: *same*, *update*, *conflict* or *unrelated*.  The judgement is only the
question "what is the relation"; **what happens because of it is** :func:`decide`, a pure
function of the new fact, its candidates and their relations, so the rules can be tested without
a model:

* sources rank real record (3) > what the user said (2) > what the bot invented (1); the user's
  own ``/记住`` command (4) outranks everything;
* a **lower** source never replaces a **higher** one: the new fact is kept as a *rejected
  candidate* (``rejected_by`` names the fact that vetoed it) and never shown;
* the **same** source: the newer fact (by ``known_at``, which is the time of the evidence, not of
  the processing - a replay may meet an older day last) replaces the older; the older one gets
  ``superseded_by``, ``superseded_at`` and ``valid_to``;
* a **higher** source replaces lower ones, which is how a real record that arrives later ends
  what the bot made up (R-MEM-011); the same holds for life line entries, which are bot
  inventions: the ones a higher source contradicts are marked invalid;
* **same** with an equal or higher existing fact adds nothing: the evidence and the earliest
  known time are merged into the existing fact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

from twin.llm.deepseek import DeepSeekClient
from twin.llm.types import LedgerTag, Purpose
from twin.memory.extract import FactDraft
from twin.memory.memory import Memory
from twin.memory.records import FactRecord, LifelineRecord
from twin.memory.schemas import ConflictOut
from twin.ops.logging import get_logger
from twin.profile.prompt_templates import MEMORY_CONFLICT, TemplateStore
from twin.retrieval.embedder import EmbedderError, EncodeKind
from twin.storage.memory_models import SOURCE_PRIORITY
from twin.storage.vector_schema import VectorKind

log = get_logger("twin.memory.conflict")

Action = Literal["insert", "merge", "reject"]
JUDGE_BATCH = 6  # new facts judged in one call
VECTOR_OVERFETCH = 3
MAX_LIFELINE_CANDIDATES = 8
MAX_VALID_SPAN_DAYS = 14
SOURCE_NAMES = {
    "real_record": "真实聊天记录",
    "user_said": "对方在和机器人聊天时说的",
    "bot_invented": "机器人自己说的",
    "user_command": "对方用记住指令写的",
}
MIN_KEYWORD_SCORE = 0.2


@dataclass(frozen=True)
class FactCandidate:
    """An older fact a new fact is compared with (``c1``, ``c2``, ...)."""

    key: str
    fact: FactRecord


@dataclass(frozen=True)
class EventCandidate:
    """A life line entry a new fact is compared with (``l1``, ``l2``, ...)."""

    key: str
    event: LifelineRecord


@dataclass(frozen=True)
class Recall:
    """Everything a new fact is compared with."""

    facts: tuple[FactCandidate, ...] = ()
    events: tuple[EventCandidate, ...] = ()

    @property
    def empty(self) -> bool:
        return not self.facts and not self.events


@dataclass(frozen=True)
class Decision:
    """What storing a new fact does (see the module description)."""

    action: Action
    target_id: str | None = None  # merge: the fact merged into; reject: the fact that vetoed
    supersede: tuple[str, ...] = ()  # older facts the new one replaces
    superseded_by: str | None = None  # a newer, equal-ranked fact that already replaces the new one
    invalidate_events: tuple[str, ...] = ()  # life line entries the new fact contradicts
    relation_counts: dict[str, int] = field(default_factory=dict)


def decide(new: FactDraft, recall: Recall, relations: Mapping[str, str]) -> Decision:
    """The rules of R-MEM-004 and R-MEM-011 applied to one new fact."""
    priority = SOURCE_PRIORITY[new.source]
    counts: dict[str, int] = {}
    for key in [c.key for c in recall.facts] + [c.key for c in recall.events]:
        relation = relations.get(key, "unrelated")
        counts[relation] = counts.get(relation, 0) + 1
    same = [c for c in recall.facts if relations.get(c.key) == "same"]
    changed = [c for c in recall.facts if relations.get(c.key) in ("update", "conflict")]
    if same:
        best = max(same, key=_rank)
        if best.fact.priority >= priority:
            return Decision("merge", best.fact.id, relation_counts=counts)
        changed = [*changed, *same]  # the new fact is the better source of the same thing
    vetoers = [c for c in changed if c.fact.priority > priority]
    if vetoers:
        return Decision("reject", max(vetoers, key=_rank).fact.id, relation_counts=counts)
    newer = [c for c in changed if c.fact.priority == priority and c.fact.known_at > new.known_at]
    replaced = tuple(c.fact.id for c in changed if c not in newer)
    invalid: tuple[str, ...] = ()
    if priority > SOURCE_PRIORITY["bot_invented"]:
        invalid = tuple(c.event.id for c in recall.events if relations.get(c.key) == "conflict")
    return Decision(
        "insert",
        None,
        replaced,
        max(newer, key=_rank).fact.id if newer else None,
        invalid,
        counts,
    )


def _rank(candidate: FactCandidate) -> tuple[int, float]:
    return candidate.fact.priority, candidate.fact.known_at.timestamp()


def normalised(text: str) -> str:
    """The text without spaces and punctuation, for telling an exact repeat from a variant."""
    return "".join(ch for ch in text if ch.isalnum()).lower()


class ConflictResolver:
    """Recalls the candidates of a new fact and has DeepSeek judge them."""

    def __init__(
        self,
        memory: Memory,
        client: DeepSeekClient | None,
        *,
        templates: TemplateStore | None = None,
    ) -> None:
        self._memory = memory
        self._client = client
        services = memory.services
        self._templates = templates or TemplateStore(services.db, services.clock)

    # ------------------------------------------------------------------- recall

    def recall(self, draft: FactDraft) -> Recall:
        """Older facts (``c1``..) and life line entries (``l1``..) the draft could clash with."""
        memory = self._memory
        config = memory.services.settings.memory
        found: dict[str, tuple[float, FactRecord]] = {}

        def consider(fact: FactRecord | None, score: float) -> None:
            if fact is None or not fact.current:
                return
            if fact.subject != draft.subject and "both" not in (fact.subject, draft.subject):
                return
            old = found.get(fact.id)
            if old is None or score > old[0]:
                found[fact.id] = (score, fact)

        try:
            hits = memory.vectors.search(
                VectorKind.FACT,
                draft.text,
                config.conflict_candidates * VECTOR_OVERFETCH,
                encode_as=EncodeKind.SYMMETRIC,
            )
        except EmbedderError:
            log.warning("conflict_recall_without_vectors")
            hits = []
        for hit in hits:
            if hit.similarity >= config.conflict_min_similarity:
                consider(memory.corpus.facts.get(hit.id), hit.similarity)
        for kw in memory.corpus.fact_index.search(draft.text, config.conflict_candidates):
            if kw.score >= MIN_KEYWORD_SCORE:
                fact = memory.corpus.facts.get(kw.doc_id)
                if fact is not None and fact.subject == draft.subject:
                    consider(fact, kw.score)
        ranked = sorted(found.values(), key=lambda pair: (-pair[0], pair[1].number))
        facts = tuple(
            FactCandidate(f"c{number}", fact)
            for number, (_, fact) in enumerate(ranked[: config.conflict_candidates], start=1)
        )
        events: tuple[EventCandidate, ...] = ()
        if SOURCE_PRIORITY[draft.source] > SOURCE_PRIORITY["bot_invented"]:
            events = tuple(
                EventCandidate(f"l{number}", event)
                for number, event in enumerate(self._events_near(draft), start=1)
            )
        return Recall(facts, events)

    def _events_near(self, draft: FactDraft) -> list[LifelineRecord]:
        """Active life line entries of the days the fact speaks of."""
        clock = self._memory.clock
        days: set[date] = {clock.real_date(draft.known_at)}
        if draft.event_date is not None:
            days = {draft.event_date}
        if draft.valid_from is not None and draft.valid_to is not None:
            start, end = clock.real_date(draft.valid_from), clock.real_date(draft.valid_to)
            if 0 <= (end - start).days <= MAX_VALID_SPAN_DAYS:
                days |= {start + timedelta(days=n) for n in range((end - start).days + 1)}
        events = [
            event
            for event in self._memory.corpus.events.values()
            if event.active and event.local_date in days
        ]
        events.sort(key=lambda e: (e.local_date, e.start_local or "", e.id))
        return events[:MAX_LIFELINE_CANDIDATES]

    def exact_repeat(self, draft: FactDraft, recall: Recall) -> str | None:
        """The key of a candidate that says exactly the same (no model is needed to know)."""
        wanted = normalised(draft.text)
        for candidate in recall.facts:
            if normalised(candidate.fact.text) == wanted:
                return candidate.key
        return None

    # ------------------------------------------------------------------ judging

    async def judge(
        self,
        batch: Sequence[tuple[FactDraft, Recall]],
        tag: LedgerTag,
    ) -> list[dict[str, str]]:
        """The relation of every candidate to its new fact, one mapping per draft."""
        relations: list[dict[str, str]] = [{} for _ in batch]
        for start in range(0, len(batch), JUDGE_BATCH):
            part = batch[start : start + JUDGE_BATCH]
            answers = await self._ask(part, tag)
            for offset, mapping in enumerate(answers):
                relations[start + offset] = mapping
        return relations

    async def _ask(
        self, part: Sequence[tuple[FactDraft, Recall]], tag: LedgerTag
    ) -> list[dict[str, str]]:
        if self._client is None:
            # without a model nothing can be judged: the facts are stored as unrelated
            log.warning("conflicts_not_judged", facts=len(part))
            return [{} for _ in part]
        template = self._templates.active(MEMORY_CONFLICT)
        messages = template.render(items=self._render(part))
        reply = await self._client.chat_json(
            messages, ConflictOut, purpose=Purpose.EXTRACT, tag=tag
        )
        answers: list[dict[str, str]] = [{} for _ in part]
        for judgement in reply.value.judgements:
            key = judgement.new.strip().lower().removeprefix("n")
            if not key.isdigit() or not 1 <= int(key) <= len(part):
                continue
            recall = part[int(key) - 1][1]
            allowed = {c.key for c in recall.facts} | {c.key for c in recall.events}
            for verdict in judgement.verdicts:
                if verdict.candidate in allowed:
                    answers[int(key) - 1][verdict.candidate] = verdict.relation
        return answers

    def _render(self, part: Sequence[tuple[FactDraft, Recall]]) -> str:
        clock = self._memory.clock
        blocks: list[str] = []
        for number, (draft, recall) in enumerate(part, start=1):
            day = clock.real_date(draft.known_at).isoformat()
            lines = [
                f"新信息 n{number}（来源：{SOURCE_NAMES[draft.source]}；日期：{day}）：{draft.text}"
            ]
            for candidate in recall.facts:
                fact = candidate.fact
                lines.append(
                    f"  旧条目 {candidate.key}（来源：{SOURCE_NAMES[fact.source]}；"
                    f"日期：{clock.real_date(fact.known_at).isoformat()}）：{fact.text}"
                )
            for item in recall.events:
                event = item.event
                day_text = event.local_date.isoformat()
                lines.append(f"  旧条目 {item.key}（{day_text} 的生活安排）：{event.line()}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
