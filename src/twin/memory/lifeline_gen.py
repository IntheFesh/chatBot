"""The daily life line: drawn when she wakes, checked, then stored (R-MEM-005).

When she gets up the scheduler (round 08) queues the ``lifeline_generate`` job
(:mod:`twin.schedule.jobs`), which calls :meth:`LifelineGenerator.generate` with the day plan.  The
input is what the day has to fit:

* the plan: when she wakes and goes to bed, the busy periods, the meals, the day type;
* the last three days of her life line;
* the real facts that tell what her life is like (school, work, home, habits): only facts whose
  source is a real record, never what the bot made up, plus the facts that name this very day.

The model draws 5 to 10 stretches (time, activity, place, mood, detail, busy).  Then the draw is
checked twice, differently:

1. by code (:func:`~twin.memory.lifeline_rules.check_rules`): readable times, no overlap, nothing
   planned while she sleeps, a stretch in a busy period is marked busy and the other way round;
2. by the model, only if the code finds nothing: does the day contradict a real fact or the
   recent days?

A day that fails is drawn **once more**, with the reasons written into the prompt.  If the second
draw fails as well the day is **corrected by rule** (:func:`~twin.memory.lifeline_rules.repair`:
the sleeping part cut off, overlaps moved, the stretches the model found contradictory dropped,
the busy marks set from the plan) and an alert says so.  The result is written with
``LifelineStore.replace_plan`` (what the bot let slip in conversation stays) and checked once more
by the store's own consistency check, which stamps the entries.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from zoneinfo import ZoneInfo

from twin.llm.deepseek import DeepSeekClient
from twin.llm.types import DAILY, LedgerTag, Purpose
from twin.memory.lifeline import LifelineContext, LifelineStore, PlannedEvent
from twin.memory.lifeline_rules import DayFrame, RuleProblem, check_rules, clock_text, repair
from twin.memory.memory import Memory
from twin.memory.records import FactRecord
from twin.memory.render import WEEKDAY_NAMES, day_label, fact_line
from twin.memory.schemas import Contradiction, DayCheckOut, DrawnDay, DrawnEvent
from twin.ops.logging import get_logger
from twin.profile.prompt_templates import LIFELINE_CHECK, LIFELINE_GENERATE, TemplateStore
from twin.schedule.plan_model import DailyPlan
from twin.schedule.time_service import TimeService

log = get_logger("twin.memory.lifeline_gen")

BASE_FACT_CATEGORIES = frozenset({"life", "work_study", "preference"})
BASE_FACT_LIMIT = 14
RECENT_DAYS = 3
DAY_TYPE_NAMES = {"workday": "工作日", "weekend": "周末", "holiday": "节假日"}
MEAL_NAMES = {"breakfast": "早饭", "lunch": "午饭", "dinner": "晚饭"}
NOTHING = "（没有）"


@dataclass(frozen=True)
class Review:
    """What the two checks found in one draw."""

    problems: tuple[RuleProblem, ...] = ()
    contradictions: tuple[Contradiction, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.problems and not self.contradictions

    def reasons(self, events: Sequence[DrawnEvent]) -> list[str]:
        """The findings as sentences for the next prompt."""
        lines = [problem.text for problem in self.problems]
        for item in self.contradictions:
            what = events[item.event - 1].activity if item.event <= len(events) else ""
            side = "与已知的事矛盾" if item.against == "fact" else "与最近几天的安排衔接不上"
            lines.append(f"第 {item.event} 段（{what}）{side}：{item.reason}")
        return lines


@dataclass(frozen=True)
class LifelineResult:
    """What generating one day did."""

    day: date
    plan_id: str
    events: int
    drafts: int  # 1 when the first draw passed, 2 otherwise
    corrected: bool  # the rule-corrected version was written
    left: tuple[str, ...]  # codes of the problems that remained after the second draw
    calls: int
    cost_usd: float


class LifelineGenerator:
    """Draws, checks and stores the life line of a day (see the module description)."""

    def __init__(
        self,
        memory: Memory,
        client: DeepSeekClient,
        time_service: TimeService,
        *,
        templates: TemplateStore | None = None,
    ) -> None:
        self._memory = memory
        self._client = client
        self._time = time_service
        services = memory.services
        self._templates = templates or TemplateStore(services.db, services.clock)
        self._store = LifelineStore(memory, time_service=time_service)
        self._calls = 0
        self._cost = 0.0

    # -------------------------------------------------------------------- facts

    def real_facts(self, day: date, context: LifelineContext) -> list[FactRecord]:
        """The real facts the day has to fit: her life as it is, and what is dated today."""
        self._memory.refresh()
        start, end = self._memory.clock.bot_bounds(day)
        base = [
            fact
            for fact in self._memory.corpus.facts.values()
            if fact.current
            and fact.source == "real_record"
            and fact.subject in ("her", "both")
            and fact.category in BASE_FACT_CATEGORIES
            and (fact.valid_to is None or fact.valid_to >= start)
            and (fact.valid_from is None or fact.valid_from < end)
        ]
        base.sort(key=lambda f: (-f.importance, -f.known_at.timestamp(), f.id))
        chosen = base[:BASE_FACT_LIMIT]
        known = {fact.id for fact in chosen}
        dated = [
            fact
            for fact in context.facts_for_the_day
            if fact.source == "real_record" and fact.id not in known
        ]
        log.debug("lifeline_facts", day=day.isoformat(), count=len(chosen) + len(dated))
        return [*chosen, *dated]

    @staticmethod
    def _fact_text(facts: Sequence[FactRecord]) -> str:
        return "\n".join(f"- {fact_line(fact, None)}" for fact in facts) or NOTHING

    @staticmethod
    def _recent_text(context: LifelineContext) -> str:
        if not context.previous_days:
            return NOTHING
        parts = []
        for when, lines in context.previous_days:
            body = "\n".join(f"  {line}" for line in lines)
            parts.append(f"{day_label(when)}：\n{body}")
        return "\n".join(parts)

    # ------------------------------------------------------------------ prompts

    @staticmethod
    def _frame_text(plan: DailyPlan, frame: DayFrame) -> dict[str, str]:
        if frame.wake is not None:
            wake = clock_text(frame.wake)
        elif frame.not_before > 0:
            wake = f"（已经醒着，这份安排从 {clock_text(frame.not_before)} 开始写）"
        else:
            wake = "（凌晨之前就醒着）"
        bed = (
            clock_text(frame.bed)
            if frame.bed is not None and frame.bed < 1440
            else "今天一直醒到午夜之后"
        )
        busy = "\n".join(f"- {span.label}" for span in frame.busy) or NOTHING
        meals = (
            "\n".join(
                f"- {MEAL_NAMES.get(kind, kind)}约 {clock_text(minute)}"
                for kind, minute in frame.meals
            )
            or NOTHING
        )
        return {"wake": wake, "bed": bed, "busy": busy, "meals": meals}

    async def _draw(
        self,
        plan: DailyPlan,
        frame: DayFrame,
        facts_text: str,
        recent_text: str,
        feedback: Sequence[str],
        tag: LedgerTag,
    ) -> list[DrawnEvent]:
        day = plan.local_date
        text = (
            "\n上一版被退回了，请改正下面这些问题后重新写整天的安排：\n"
            + "\n".join(f"- {line}" for line in feedback)
            if feedback
            else ""
        )
        messages = self._templates.active(LIFELINE_GENERATE).render(
            date=day.isoformat(),
            weekday=WEEKDAY_NAMES[day.weekday()],
            day_type=DAY_TYPE_NAMES.get(plan.day_type, plan.day_type),
            zone=plan.timezone,
            facts=facts_text,
            recent=recent_text,
            feedback=text,
            **self._frame_text(plan, frame),
        )
        reply = await self._client.chat_json(messages, DrawnDay, purpose=Purpose.PLAN, tag=tag)
        self._calls += reply.attempts
        self._cost += reply.total_cost_usd
        return reply.value.events

    async def _contradictions(
        self,
        plan: DailyPlan,
        events: Sequence[DrawnEvent],
        facts_text: str,
        recent_text: str,
        tag: LedgerTag,
    ) -> tuple[Contradiction, ...]:
        day = plan.local_date
        listed = "\n".join(
            f"{number}. {event.start}-{event.end} {event.activity}"
            + (f"（{event.place}）" if event.place else "")
            for number, event in enumerate(events, 1)
        )
        messages = self._templates.active(LIFELINE_CHECK).render(
            date=day.isoformat(),
            weekday=WEEKDAY_NAMES[day.weekday()],
            facts=facts_text,
            recent=recent_text,
            events=listed,
        )
        reply = await self._client.chat_json(messages, DayCheckOut, purpose=Purpose.PLAN, tag=tag)
        self._calls += reply.attempts
        self._cost += reply.total_cost_usd
        return tuple(c for c in reply.value.contradictions if c.event <= len(events))

    async def _review(
        self,
        plan: DailyPlan,
        frame: DayFrame,
        events: list[DrawnEvent],
        facts_text: str,
        recent_text: str,
        tag: LedgerTag,
    ) -> Review:
        problems = tuple(check_rules(events, frame))
        if problems:
            return Review(problems)
        return Review((), await self._contradictions(plan, events, facts_text, recent_text, tag))

    # ----------------------------------------------------------------- generating

    async def generate(self, plan: DailyPlan, tag: LedgerTag = DAILY) -> LifelineResult:
        """Draw, check and store the life line of ``plan.local_date``."""
        self._calls, self._cost = 0, 0.0
        day = plan.local_date
        frame = DayFrame.from_plan(plan, ZoneInfo(plan.timezone))
        context = await asyncio.to_thread(self._store.context_for, day, days=RECENT_DAYS)
        facts = await asyncio.to_thread(self.real_facts, day, context)
        facts_text, recent_text = self._fact_text(facts), self._recent_text(context)

        first = await self._draw(plan, frame, facts_text, recent_text, (), tag)
        review = await self._review(plan, frame, first, facts_text, recent_text, tag)
        drafts, corrected, final = 1, False, first
        left: tuple[str, ...] = ()
        if not review.ok:
            second = await self._draw(
                plan, frame, facts_text, recent_text, review.reasons(first), tag
            )
            drafts = 2
            review = await self._review(plan, frame, second, facts_text, recent_text, tag)
            final = second
            if not review.ok:
                corrected = True
                codes = {problem.code for problem in review.problems}
                if review.contradictions:
                    codes.add("contradiction")
                left = tuple(sorted(codes))
                final = repair(second, frame, frozenset(c.event for c in review.contradictions))
                self._warn(plan, left, len(second), len(final))
        planned = [
            PlannedEvent(e.activity, e.start, e.end, e.place, e.mood, e.detail) for e in final
        ]
        await asyncio.to_thread(self._write, day, planned)
        log.info(
            "lifeline_generated",
            day=day.isoformat(),
            events=len(planned),
            drafts=drafts,
            corrected=corrected,
        )
        return LifelineResult(
            day, plan.id, len(planned), drafts, corrected, left, self._calls, self._cost
        )

    def _write(self, day: date, planned: list[PlannedEvent]) -> None:
        self._store.replace_plan(day, planned)
        report = self._store.check_consistency(day)
        if not report.ok:
            log.warning(
                "lifeline_inconsistent_after_write",
                day=day.isoformat(),
                issues=sorted({issue.kind for issue in report.issues}),
            )

    def _warn(self, plan: DailyPlan, codes: tuple[str, ...], drawn: int, kept: int) -> None:
        """The second draw failed too: say that the day was corrected by rule."""
        self._memory.services.alerts.raise_alert(
            "lifeline_corrected",
            "the life line of a day failed its checks twice and was corrected by rule",
            severity="warning",
            detail={
                "day": plan.local_date.isoformat(),
                "problems": list(codes),
                "drawn": drawn,
                "kept": kept,
            },
            dedup_key=f"lifeline_corrected:{plan.local_date.isoformat()}",
        )
