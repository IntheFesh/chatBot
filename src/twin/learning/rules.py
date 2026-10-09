"""The weekly consolidation of ``[不要这样]`` (R-LRN-003).

What the user said about the bot's replies - ``/重来`` (thrown away), ``/不像`` (not like her), a
correction in plain words that was confirmed - piles up in ``feedback``.  Once a week, off peak,
:class:`RuleConsolidator` turns the new feedback into rules for the persona card:

1. the unprocessed feedback is shown to DeepSeek together with the rules already in the card
   (``correction_rules`` template), which writes the new, merged list: similar rules combined,
   contradictions settled for the newer, at most ``learning.rules_max`` (30) rules, each one short
   sentence about a **manner of speaking**;
2. every rule of the answer is checked.  Locally (:mod:`twin.learning.rulecheck`: no number, date,
   time, amount, event, name or place; one short line) and by a second judgement of DeepSeek
   (``correction_rule_check`` template: is it only about style?).  A rule that fails either is
   dropped.  Corrections that are about a *fact* do not belong here - they go to the memory
   (R-LRN-003): when rules were dropped for being facts, an alert says so and points to ``/记住``;
3. rules that say the same thing (one contains the other, or nearly all of their characters agree)
   are one rule, the older wording wins; the list is cut to the limit;
4. the result is written through :func:`~twin.profile.persona.api.write_corrections` as a new
   version of the live card: only ``[不要这样]`` changes, ``[手动]``, the statistics and the
   description keep their exact text.  An answer without any rule never empties a card that has
   rules, and a failed call changes nothing and leaves the feedback unprocessed (the job retries).

The feedback is marked processed after the card is written.  Nothing here reads or writes the style
samples, the retrieval library or the training set (R-LRN-004).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from twin.engine.feedback import FeedbackRecord, FeedbackStore
from twin.engine.turns import BotTurnStore
from twin.learning.pairs import reply_text
from twin.learning.rulecheck import check_rule, clean_rule
from twin.llm.deepseek import DeepSeekClient
from twin.llm.types import DAILY, Purpose
from twin.ops.logging import get_logger
from twin.profile.persona.api import read_corrections, write_corrections
from twin.profile.prompt_templates import CORRECTION_RULE_CHECK, CORRECTION_RULES, TemplateStore
from twin.services import Services

log = get_logger("twin.learning.rules")

MAX_FEEDBACK_ITEMS = 60  # the newest this many are looked at in one run; the rest wait a week
MAX_REPLY_CHARS = 160
SIMILARITY = 0.75
MIN_CONTAINED_CHARS = 6

Status = Literal["written", "unchanged", "nothing_new"]


class RulesOut(BaseModel):
    """The answer of the ``correction_rules`` prompt."""

    model_config = ConfigDict(extra="ignore")

    rules: list[str] = Field(default_factory=list)


class RuleJudgement(BaseModel):
    """One verdict of the ``correction_rule_check`` prompt."""

    model_config = ConfigDict(extra="ignore")

    index: int
    ok: bool
    kind: str = "other"


class RuleJudgements(BaseModel):
    model_config = ConfigDict(extra="ignore")

    verdicts: list[RuleJudgement] = Field(default_factory=list)


@dataclass
class ConsolidationReport:
    """What a run did (counts only; the rules themselves are in the card)."""

    status: Status
    feedback: int = 0
    rules: int = 0
    added: int = 0
    kept: int = 0
    dropped_local: int = 0
    dropped_judged: int = 0
    dropped_facts: int = 0
    duplicates: int = 0
    cut: int = 0
    version: str | None = None
    detail: dict[str, int] = field(default_factory=dict)


def _grams(text: str) -> set[str]:
    plain = "".join(text.split())
    return {plain[i : i + 2] for i in range(len(plain) - 1)} or {plain}


def same_rule(first: str, second: str) -> bool:
    """Do the two rules say the same?  One holds the other, or their character pairs agree."""
    a, b = "".join(first.split()), "".join(second.split())
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= MIN_CONTAINED_CHARS and shorter in longer:
        return True
    left, right = _grams(a), _grams(b)
    return len(left & right) / max(1, len(left | right)) >= SIMILARITY


def merge_rules(existing: Sequence[str], candidates: Sequence[str]) -> tuple[list[str], int]:
    """``candidates`` without repeats; a rule equal to an existing one keeps its old wording.

    Returns the list and how many candidates were repeats.
    """
    result: list[str] = []
    repeats = 0
    for rule in candidates:
        twin = next((old for old in existing if same_rule(old, rule)), None)
        text = twin if twin is not None else rule
        if any(same_rule(text, kept) for kept in result):
            repeats += 1
            continue
        result.append(text)
    return result, repeats


class RuleConsolidator:
    """Turns new feedback into the rules of the persona card (see the module description)."""

    def __init__(
        self,
        services: Services,
        client: DeepSeekClient,
        feedback: FeedbackStore,
        turns: BotTurnStore,
        *,
        templates: TemplateStore | None = None,
    ) -> None:
        self._services = services
        self._client = client
        self._feedback = feedback
        self._turns = turns
        self._templates = templates or TemplateStore(services.db, services.clock)
        self._config = services.settings.learning

    # --------------------------------------------------------------------------- the run

    async def run(self) -> ConsolidationReport:
        """One consolidation; raises what the model layer raises (the job queue retries)."""
        pending = await asyncio.to_thread(self._feedback.unprocessed)
        if not pending:
            return ConsolidationReport("nothing_new")
        used = pending[-MAX_FEEDBACK_ITEMS:]
        lines = await asyncio.to_thread(self._describe, used)
        existing = await asyncio.to_thread(read_corrections, self._services)
        report = ConsolidationReport("unchanged", feedback=len(used))
        if lines:
            candidates = await self._propose(existing, lines)
            final = await self._vet(existing, candidates, report)
        else:  # the replies are gone (the rows were removed): nothing to learn from
            final = list(existing)
        report.rules = len(final)
        if final != existing:
            card = await asyncio.to_thread(
                write_corrections, self._services, final, reason="corrections"
            )
            report.status = "written"
            report.version = card.id
        await asyncio.to_thread(self._feedback.mark_processed, [item.id for item in used])
        if report.dropped_facts:
            self._services.alerts.raise_alert(
                "learning_fact_corrections",
                f"{report.dropped_facts} correction rule(s) were about facts, not about the way "
                "of speaking, and were not written to the persona card; use /记住 for facts",
                severity="info",
                detail={"dropped": report.dropped_facts},
            )
        log.info(
            "rules_consolidated",
            status=report.status,
            feedback=report.feedback,
            rules=report.rules,
            added=report.added,
            dropped_local=report.dropped_local,
            dropped_judged=report.dropped_judged,
            duplicates=report.duplicates,
        )
        return report

    def _describe(self, items: Sequence[FeedbackRecord]) -> list[str]:
        """The feedback as lines for the prompt: what she said and, if given, what she would say."""
        lines: list[str] = []
        seen: set[tuple[str, str]] = set()
        for item in items:
            text = reply_text(self._turns.reply(item.reply_id))
            if not text:
                continue
            wording = (item.correction or "").strip()
            if (text, wording) in seen:
                continue
            seen.add((text, wording))
            kind = "撤销重来" if item.type == "redo" else "不像她"
            line = f"[{kind}] 机器人说：{' / '.join(text.split(chr(10)))[:MAX_REPLY_CHARS]}"
            if wording:
                line += f"；用户认为她会说：{' / '.join(wording.split(chr(10)))[:MAX_REPLY_CHARS]}"
            lines.append(f"{len(lines) + 1}. {line}")
        return lines

    async def _propose(self, existing: Sequence[str], lines: Sequence[str]) -> list[str]:
        template = await asyncio.to_thread(self._templates.active, CORRECTION_RULES)
        shown = [f"{number}. {rule}" for number, rule in enumerate(existing, start=1)]
        messages = template.render(
            max_chars=self._config.rule_max_chars,
            max_rules=self._config.rules_max,
            existing_count=len(existing),
            existing="\n".join(shown) if shown else "（还没有）",
            feedback_count=len(lines),
            feedback="\n".join(lines),
        )
        answer = await self._client.chat_json(
            messages, RulesOut, purpose=Purpose.PERSONA, tag=DAILY
        )
        return [clean_rule(rule) for rule in answer.value.rules if rule.strip()]

    # ------------------------------------------------------------------------- the checks

    async def _vet(
        self, existing: Sequence[str], candidates: Sequence[str], report: ConsolidationReport
    ) -> list[str]:
        """The rules that survive both checks, merged with the old ones and cut to the limit."""
        if not candidates:
            return list(existing)  # an empty answer never wipes the card
        survivors: list[str] = []
        fact_kinds = 0
        for rule in candidates:
            verdict = check_rule(rule, max_chars=self._config.rule_max_chars)
            if verdict.ok:
                survivors.append(rule)
                continue
            report.dropped_local += 1
            fact_kinds += verdict.kind == "fact"
        known = {"".join(old.split()) for old in existing}
        fresh = [rule for rule in survivors if "".join(rule.split()) not in known]
        judgements = await self._judge(fresh)
        approved: list[str] = []
        for rule in survivors:
            if rule in fresh:
                judged = judgements.get(rule)
                if judged is None or not judged.ok:  # a rule the model said nothing about fails
                    report.dropped_judged += 1
                    fact_kinds += 1 if judged is not None and judged.kind == "fact" else 0
                    continue
            approved.append(rule)
        merged, repeats = merge_rules(existing, approved)
        report.duplicates = repeats
        limit = self._config.rules_max
        report.cut = max(0, len(merged) - limit)
        final = merged[:limit]
        report.kept = sum(1 for rule in final if rule in existing)
        report.added = len(final) - report.kept
        report.dropped_facts = fact_kinds
        return final if final else list(existing)

    async def _judge(self, rules: Sequence[str]) -> dict[str, RuleJudgement]:
        """DeepSeek's second judgement of the new rules, by rule."""
        if not rules:
            return {}
        template = await asyncio.to_thread(self._templates.active, CORRECTION_RULE_CHECK)
        numbered = "\n".join(f"{number}. {rule}" for number, rule in enumerate(rules, start=1))
        answer = await self._client.chat_json(
            template.render(rules=numbered), RuleJudgements, purpose=Purpose.PERSONA, tag=DAILY
        )
        return {
            rules[verdict.index - 1]: verdict
            for verdict in answer.value.verdicts
            if 1 <= verdict.index <= len(rules)
        }
