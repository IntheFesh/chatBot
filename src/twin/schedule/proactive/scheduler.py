"""The proactive scheduler: one look at the clock every few minutes (R-PRO-001 to R-PRO-008).

``ProactiveScheduler.tick()`` is what happens at each tick; the component of
:mod:`twin.schedule.proactive.component` calls it every ``proactive.tick_minutes``.  A tick:

1. reads the facts of the moment in one pass (:class:`Facts`): the day plan, her state, the
   settings, how many messages went out today and since the plan took effect, how many are
   unanswered, what the platform window allows;
2. lays out the day the first time it sees a plan: the fixed moments (greeting, meals, goodnight,
   :mod:`~twin.schedule.proactive.slots`) become ``proactive_candidates``, follow-ups that came due
   become candidates, windows that passed become ``expired``;
3. does nothing more while the user is talking to her (the engine is not idle, or a message was
   exchanged within ``proactive.user_active_min`` minutes) - this tick produces no candidate;
4. takes the candidates that are due, or, if none is, draws one: while she is awake a thinned
   Poisson process over the day's curve (:mod:`~twin.schedule.proactive.curve`) brings a silence
   or sharing message, at the edge of her sleep her real late-night openings bring the occasional
   "can't sleep" message; of several candidates only the one with the highest priority is
   considered (follow-up, then the fixed moments, then silence, then sharing);
5. **checks the hard constraints** (:func:`~twin.schedule.proactive.rules.check`) - a refusal is
   written to ``proactive_log`` with its reason; a fixed moment or a follow-up that is refused
   stays and is tried again at the next tick (logged once per reason), a drawn candidate is given
   up and the draw rests (two hours, or until the constraint lets go);
6. asks the decider (:mod:`~twin.schedule.proactive.decide`) - DeepSeek plans, the style model
   writes if the backend is ``style`` or ``hybrid``, the text goes through the reply pipeline's
   post-processing;
7. **checks the constraints again** with fresh facts - the planner took its time - and cuts the
   bubbles to what the platform has left, then sends them at her pace and writes the log, the
   conversation and the life line.

Every instant is read from the injected clock; the schedule events (a restart, a wake-up from
sleep, a time zone switch) void the candidates they cover and are never made up for (R-SCH-005).
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

from twin.channel.base import SessionState
from twin.clock import Clock
from twin.config.runtime import RuntimeSettings
from twin.config.settings import ProactiveConfig
from twin.engine.pacing import PacingModel
from twin.engine.postprocess import fit_bubbles_to_quota
from twin.engine.sender import SentBubble, StopReason
from twin.engine.state_store import ConversationSnapshot
from twin.engine.turns import ReplyMeta
from twin.engine.types import Bubble, PostAction
from twin.memory.followups import FollowupStore
from twin.memory.lifeline import SHARED_STEP, LifelineStore
from twin.ops.logging import get_logger
from twin.profile.activity_model import ActivityModel
from twin.profile.distribution import EmpiricalDistribution
from twin.schedule.events import CandidatesExpired
from twin.schedule.plan_model import DailyPlan
from twin.schedule.proactive.curve import DayGrid, build_grid, edge_probability
from twin.schedule.proactive.decide import Brief, Draft
from twin.schedule.proactive.rules import Situation, check, window_refusal
from twin.schedule.proactive.send import ProactiveSender, SendOutcome
from twin.schedule.proactive.settings import daily_range, is_enabled, is_paused
from twin.schedule.proactive.slots import followup_slot, routine_slots
from twin.schedule.proactive.store import (
    CandidateRow,
    CandidateStore,
    NewLog,
    ProactiveLogStore,
)
from twin.schedule.proactive.types import Candidate, Reason, Refusal, TriggerKind
from twin.schedule.service import ScheduleKit
from twin.schedule.time_service import PlanUnavailableError

log = get_logger("twin.schedule.proactive.scheduler")

SILENCE_FALLBACK = timedelta(hours=3)  # the silence that brings a message, with no history of hers
SILENCE_FLOOR = timedelta(hours=1)  # ... and never shorter than a conversation segment
HARD_REST_MIN = 120  # a drawn message that a constraint refused is not drawn again for this long
SOFT_REST_MIN = 30  # ... or this long after the planner said no
FOLLOWUP_EVERY = timedelta(minutes=10)  # how often the open follow-ups are looked at
RESCHEDULE_MIN_S = 600.0  # a routine message the planner declined is tried again after this ...
RESCHEDULE_MAX_S = 2400.0  # ... to this long, once
MAX_ATTEMPTS = 2  # a routine message is planned this many times before it is given up
EDGE_WEEK = timedelta(days=7)
UNANSWERED_LOOKBACK = timedelta(days=30)


class Conversation(Protocol):
    """What the scheduler asks of the reply engine (``ConversationEngine`` is one)."""

    def snapshot(self) -> ConversationSnapshot: ...

    @property
    def arrivals(self) -> int: ...

    async def wait_for_arrival(self, since: int, seconds: float) -> bool: ...


class Decider(Protocol):
    """Decides what to say (:class:`~twin.schedule.proactive.decide.ProactiveDecider`)."""

    async def decide(self, candidate: Candidate, brief: Brief) -> Draft: ...


class Turns(Protocol):
    """The question the scheduler asks of the bot's conversation."""

    def last_message_times(self) -> tuple[datetime | None, datetime | None]:
        """The newest message of the user and the newest of either side."""
        ...


@dataclass(frozen=True)
class Facts:
    """What a tick reads of the world (see the module description)."""

    now: datetime
    today: date
    zone: ZoneInfo
    plan: DailyPlan
    day_type: str
    state: str | None
    enabled: bool
    low: int
    high: int
    allowance: int
    spent: int
    sent_today: int
    unanswered: int
    last_unanswered_at: datetime | None
    last_sent_at: datetime | None
    last_interaction: datetime | None
    last_inbound: datetime | None
    edge_week: int
    session: SessionState
    situation: Situation
    paused: bool = False

    @property
    def local_at(self) -> str:
        return f"{self.now.astimezone(self.zone):%Y-%m-%d %H:%M}"


@dataclass(frozen=True)
class _Day:
    """What is worked out once per local day and zone: the plan, the kind of day."""

    zone: ZoneInfo
    today: date
    plan: DailyPlan
    day_type: str


@dataclass
class TickReport:
    """What one tick did (for the tests and the simulator)."""

    at: datetime
    skipped: str | None = None
    candidate: Candidate | None = None
    outcome: str | None = None
    reason: str | None = None
    sent: int = 0
    drawn: bool = False


@dataclass
class _Rest:
    """The draw of a kind rests until ``until`` after a refusal (it is not logged every tick).

    A rest after a hard constraint ends early when that constraint no longer refuses (the user
    wrote and the window reopened); a rest after the planner's answer lasts its time.
    """

    until: datetime
    reason: Reason
    hard: bool = False


class ProactiveScheduler:
    """Decides, at every tick, whether she writes first (see the module description)."""

    def __init__(
        self,
        *,
        clock: Clock,
        config: ProactiveConfig,
        schedule: ScheduleKit,
        runtime: RuntimeSettings,
        channel_state: Callable[[], SessionState],
        conversation: Conversation,
        turns: Turns,
        candidates: CandidateStore,
        log_store: ProactiveLogStore,
        followups: FollowupStore,
        lifeline: LifelineStore,
        decider: Decider,
        sender: ProactiveSender,
        pacing: Callable[[datetime], PacingModel],
        model_source: Callable[[], ActivityModel | None],
        silence_source: Callable[[], EmpiricalDistribution | None],
        budget_allowed: Callable[[], bool],
        rng: random.Random | None = None,
    ) -> None:
        self._clock = clock
        self._config = config
        self._kit = schedule
        self._runtime = runtime
        self._channel_state = channel_state
        self._conversation = conversation
        self._turns = turns
        self.candidates = candidates
        self.log = log_store
        self._followups = followups
        self._lifeline = lifeline
        self._decider = decider
        self._sender = sender
        self._pacing = pacing
        self._model_source = model_source
        self._silence_source = silence_source
        self._budget_allowed = budget_allowed
        self._rng = rng or random.Random()  # noqa: S311 - a draw, not security
        self._lock = asyncio.Lock()
        self._epoch = 0
        self._laid_out: tuple[str, ...] = ()
        self._today: _Day | None = None
        self._opened: tuple[date, str] | None = None
        self._followups_at: datetime | None = None
        self._grid: tuple[str, DayGrid] | None = None
        self._model: tuple[str, ActivityModel | None] | None = None
        self._silence: tuple[str, EmpiricalDistribution | None] | None = None
        self._rest: dict[str, _Rest] = {}

    # ------------------------------------------------------------------- events

    @property
    def epoch(self) -> int:
        """Moves whenever the candidates are void (restart, wake-up, zone switch)."""
        return self._epoch

    async def on_expired(self, event: CandidatesExpired) -> None:
        """Void the candidates a schedule event covers and say so in the log (R-SCH-005)."""
        self._epoch += 1
        reason = Reason.TIMEZONE_SWITCH if event.reason == "timezone_switch" else Reason.INTERRUPTED
        rows = await asyncio.to_thread(self.candidates.expire_covered, event)
        self.forget_day()
        if not rows:
            return
        await asyncio.to_thread(self._log_voided, rows, reason, event.at)
        log.info("proactive_candidates_voided", count=len(rows), reason=reason.value)

    def forget_day(self) -> None:
        """Drop what was worked out for the day (the plan or the routine changed under it)."""
        self._laid_out = ()
        self._today = None
        self._opened = None
        self._followups_at = None
        self._grid = None
        self._model = None
        self._silence = None
        self._rest.clear()

    def _log_voided(self, rows: Sequence[CandidateRow], reason: Reason, at: datetime) -> None:
        zone = self._kit.time.bot_timezone()
        for row in rows:
            candidate = row.candidate()
            self.log.add(
                NewLog(
                    at=at,
                    candidate_at=candidate.planned_at,
                    local_date=at.astimezone(zone).date(),
                    local_at=f"{at.astimezone(zone):%Y-%m-%d %H:%M}",
                    timezone=zone.key,
                    kind=candidate.kind.value,
                    outcome="expired",
                    reason=reason.value,
                    candidate_id=candidate.id,
                    followup_id=candidate.followup_id,
                )
            )

    # --------------------------------------------------------------------- facts

    def _day(self, now: datetime) -> _Day:
        """The plan and the kind of day, worked out once per local day and zone."""
        time = self._kit.time
        zone = time.bot_timezone()
        today = now.astimezone(zone).date()
        found = self._today
        if found is not None and found.zone.key == zone.key and found.today == today:
            return found
        plan = self._kit.planner.ensure(today)
        found = _Day(zone, today, plan, time.day_type(today))
        self._today = found
        return found

    def _gather(self, now: datetime) -> Facts:
        """Everything a tick reads, in a few reads (runs in a worker thread)."""
        day = self._day(now)
        plan, today = day.plan, day.today
        try:
            state: str | None = str(self._kit.time.her_state(now).kind)
        except PlanUnavailableError:
            state = None
        low, high = daily_range(self._runtime, self._config)
        switch = is_enabled(self._runtime)
        paused = is_paused(self._runtime, now)
        last_inbound, last_any = self._turns.last_message_times()
        week_ago = now - EDGE_WEEK
        rows = self.log.sent_since(min(week_ago, last_inbound or now - UNANSWERED_LOOKBACK))
        unanswered = [row for row in rows if last_inbound is None or row.at > last_inbound]
        session = self._channel_state()
        refusal = window_refusal(session)
        config = self._config
        situation = Situation(
            now=now,
            state=state,
            enabled=switch,
            paused=paused,
            budget_allowed=self._budget_allowed,
            window=refusal[0] if refusal else None,
            window_detail=refusal[1] if refusal else None,
            last_sent_at=rows[-1].at if rows else None,
            unanswered=len(unanswered),
            last_unanswered_at=unanswered[-1].at if unanswered else None,
            sent_today=sum(1 for row in rows if row.local_date == today),
            daily_max=high,
            spent_since_plan=sum(1 for row in rows if row.at >= plan.effective_from),
            allowance=plan.quota.for_plan,
            edge_sent_week=sum(
                1 for row in rows if row.at >= week_ago and row.her_state == "sleep_edge"
            ),
            edge_weekly_max=config.edge_of_sleep_weekly_max,
            min_spacing=timedelta(minutes=config.min_spacing_min),
            max_chase=config.max_chase,
            unanswered_after=timedelta(minutes=config.unanswered_after_min),
        )
        return Facts(
            now=now,
            today=today,
            zone=day.zone,
            plan=plan,
            day_type=day.day_type,
            state=state,
            enabled=switch and plan.quota.enabled,
            low=low,
            high=high,
            allowance=plan.quota.for_plan,
            spent=situation.spent_since_plan,
            sent_today=situation.sent_today,
            unanswered=situation.unanswered,
            last_unanswered_at=situation.last_unanswered_at,
            last_sent_at=situation.last_sent_at,
            last_interaction=last_any,
            last_inbound=last_inbound,
            edge_week=situation.edge_sent_week,
            session=session,
            situation=situation,
            paused=paused,
        )

    def _activity(self, plan: DailyPlan) -> ActivityModel | None:
        if self._model is None or self._model[0] != plan.id:
            self._model = (plan.id, self._model_source())
        return self._model[1]

    # ------------------------------------------------------------------- the tick

    async def tick(self) -> TickReport:
        """One look at the clock (see the module description)."""
        async with self._lock:
            now = self._clock.now_utc()
            report = TickReport(at=now)
            facts = await asyncio.to_thread(self._gather, now)
            if await self._user_active(facts):
                await asyncio.to_thread(self._prepare_day, facts)
                report.skipped = "user_active"
                return report
            candidate, drawn = await asyncio.to_thread(self._work, facts)
            if candidate is None:
                return report
            report.candidate, report.drawn = candidate, drawn
            await self._consider(candidate, facts, drawn, report)
            return report

    async def _user_active(self, facts: Facts) -> bool:
        """The user is talking to her: the engine is not idle, or an exchange is just over."""
        snap = await asyncio.to_thread(self._conversation.snapshot)
        if snap.state != "IDLE" or snap.pending:
            return True
        last = facts.last_interaction
        window = timedelta(minutes=self._config.user_active_min)
        return last is not None and facts.now - last < window

    # ------------------------------------------------------------- laying out the day

    def _work(self, facts: Facts) -> tuple[Candidate | None, bool]:
        """Lay out the day, then choose the candidate of this tick (worker thread)."""
        pending = self._prepare_day(facts)
        return self._pick(facts, pending)

    def _prepare_day(self, facts: Facts) -> list[CandidateRow]:
        """Open the day, lay out its fixed moments, let the late ones lapse (worker thread).

        Returns the candidates that still wait.
        """
        plan, now = facts.plan, facts.now
        mark = (facts.today, facts.zone.key)
        if self._opened != mark:
            if self.log.opened(facts.today, facts.zone.key) is None:
                self.log.add(
                    NewLog(
                        at=now,
                        candidate_at=now,
                        local_date=facts.today,
                        local_at=facts.local_at,
                        timezone=facts.zone.key,
                        kind="day",
                        outcome="opened",
                        her_state=facts.state,
                        quota_total=plan.quota.total,
                        range_min=facts.low,
                        range_max=facts.high,
                        enabled=facts.enabled,
                    )
                )
            self._opened = mark
        signature = (plan.id, facts.zone.key, str(facts.today))
        if self._laid_out != signature:
            self._lay_out(facts)
            self._laid_out = signature
        self._take_followups(facts)
        pending = self.candidates.pending()
        if any(row.window_end < now for row in pending):
            for row in self.candidates.expire(before=now, at=now):
                self._log_candidate(facts, row.candidate(), "expired", Reason.WINDOW_OVER)
            pending = [row for row in pending if row.window_end >= now]
        return pending

    def _lay_out(self, facts: Facts) -> None:
        slots = routine_slots(
            facts.plan,
            model=self._activity(facts.plan),
            day_type=facts.day_type,
            config=self._config,
        )
        for slot in slots:
            row = self.candidates.add(slot)
            if row is None:
                continue
            if slot.planned_at < facts.now:  # its time went by while nobody was looking
                self.candidates.mark(row.id, "expired", at=facts.now, reason=Reason.INTERRUPTED)
                reason = Reason.WINDOW_OVER if slot.window_end < facts.now else Reason.INTERRUPTED
                self._log_candidate(facts, row.candidate(), "expired", reason)

    def _take_followups(self, facts: Facts) -> None:
        """Follow-ups that came due become candidates (R-MEM-006); old ones are closed.

        Looked at every ``FOLLOWUP_EVERY`` (their windows are hours long), and at once after
        the day was forgotten.
        """
        last = self._followups_at
        if last is not None and facts.now - last < FOLLOWUP_EVERY:
            return
        self._followups_at = facts.now
        waiting = self._followups.open()
        if any(f.window_end < facts.now for f in waiting):
            self._followups.expire_overdue(facts.now)
        for followup in waiting:
            if not followup.due_at <= facts.now <= followup.window_end:
                continue
            if not self.candidates.has_key(f"followup:{followup.id}"):
                self.candidates.add(followup_slot(followup, plan=facts.plan, now=facts.now))

    # ------------------------------------------------------------ choosing a candidate

    def _pick(self, facts: Facts, pending: Sequence[CandidateRow]) -> tuple[Candidate | None, bool]:
        """The candidate to consider this tick and whether it was drawn (worker thread)."""
        now = facts.now
        due = [
            row.candidate()
            for row in pending
            if row.planned_at <= now <= row.window_end and not self._resting(row.kind.value, facts)
        ]
        if due:
            due.sort(key=lambda c: (c.priority, c.planned_at))
            return due[0], False
        drawn = self._draw(facts, pending)
        return drawn, drawn is not None

    def _resting(self, kind: str, facts: Facts) -> bool:
        """Does the draw of ``kind`` rest after a refusal that still holds?"""
        rest = self._rest.get(kind)
        if rest is None or facts.now >= rest.until:
            return False
        if rest.hard:
            refusal = check(TriggerKind(kind), facts.situation)
            if refusal is None or refusal.reason is not rest.reason:
                del self._rest[kind]
                return False
        return True

    def _rest_after(self, kind: str, facts: Facts, reason: Reason, *, hard: bool) -> None:
        minutes = HARD_REST_MIN if hard else SOFT_REST_MIN
        self._rest[kind] = _Rest(facts.now + timedelta(minutes=minutes), reason, hard)

    def _draw(self, facts: Facts, pending: Sequence[CandidateRow]) -> Candidate | None:
        """The random side of the day (R-PRO-001, R-PRO-005): a silence/share or an edge message.

        The fixed moments that wait count against the day's draw; and nothing is drawn just before
        one of them - a message at that time would take the place of the moment (the spacing).
        """
        now = facts.now
        pending_expected = len(pending)
        if not facts.enabled and facts.state != "sleep_edge":
            return None
        model = self._activity(facts.plan)
        tick = timedelta(minutes=self._config.tick_minutes)
        if facts.state == "sleep_edge":
            if self._resting(TriggerKind.EDGE.value, facts):
                return None
            chance = edge_probability(model, now, facts.zone, facts.day_type, tick)
            if self._rng.random() >= chance:
                return None
            return Candidate(
                TriggerKind.EDGE,
                f"edge:{now.isoformat()}",
                now,
                now + tick,
                detail={"side": self._edge_side(facts)},
            )
        if facts.state not in ("free", "busy"):
            return None
        spacing = timedelta(minutes=self._config.min_spacing_min)
        if facts.last_sent_at is not None and now - facts.last_sent_at < spacing:
            return None  # too soon after the last one: no draw, nothing to refuse
        if any(now < row.planned_at < now + spacing for row in pending):
            return None  # a fixed moment is about to come
        kind = self._drawn_kind(facts)
        if self._resting(kind.value, facts):
            return None
        needed = min(
            facts.allowance - facts.spent - pending_expected,
            facts.high - facts.sent_today - pending_expected,
        )
        grid = self._day_grid(facts, model)
        chance = grid.probability(now, needed, spacing)
        if self._rng.random() >= chance:
            return None
        return Candidate(kind, f"{kind.value}:{now.isoformat()}", now, now + tick)

    def _day_grid(self, facts: Facts, model: ActivityModel | None) -> DayGrid:
        key = f"{facts.plan.id}:{facts.today}"
        if self._grid is None or self._grid[0] != key:
            grid = build_grid(
                facts.plan,
                step=timedelta(minutes=self._config.tick_minutes),
                model=model,
                day_type=facts.day_type,
            )
            self._grid = (key, grid)
        return self._grid[1]

    def _drawn_kind(self, facts: Facts) -> TriggerKind:
        """Silence once she has been quiet as long as she would have been; else sharing."""
        last = facts.last_interaction
        if last is None:
            return TriggerKind.SILENCE
        threshold = self._silence_threshold(facts.plan, last)
        return TriggerKind.SILENCE if facts.now - last >= threshold else TriggerKind.SHARE

    def _silence_threshold(self, plan: DailyPlan, last: datetime) -> timedelta:
        """How long a silence has to be before she breaks it, drawn once per silence."""
        if self._silence is None or self._silence[0] != plan.id:
            self._silence = (plan.id, self._silence_source())
        found = self._silence[1]
        if found is None:
            return SILENCE_FALLBACK
        draw = random.Random(f"proactive:silence:{last.isoformat()}")  # noqa: S311 - a draw
        seconds = found.sample(draw)
        return max(SILENCE_FLOOR, timedelta(seconds=seconds))

    @staticmethod
    def _edge_side(facts: Facts) -> str:
        """Falling asleep or just awake: which edge of a night the plan has her in."""
        for episode in facts.plan.episodes:
            for kind, start, end in episode.intervals():
                if kind == "sleep_edge" and start <= facts.now < end:
                    return "falling" if start == episode.onset else "waking"
        return "falling"

    # ---------------------------------------------------------------- considering one

    async def _consider(
        self, candidate: Candidate, facts: Facts, drawn: bool, report: TickReport
    ) -> None:
        refusal = check(candidate.kind, facts.situation)
        if refusal is None and candidate.kind is TriggerKind.GREETING:
            decision = await asyncio.to_thread(self._kit.planner.greeting_decision, facts.now)
            if not decision.allowed:
                refusal = Refusal(Reason.GREETING_GAP, decision.reason)
        if refusal is not None:
            await asyncio.to_thread(self._refuse, candidate, refusal, facts, drawn)
            report.outcome, report.reason = "rejected", refusal.reason.value
            return
        arrivals = self._conversation.arrivals
        previous = (
            await asyncio.to_thread(self._unanswered_texts, facts) if facts.unanswered else ()
        )
        brief = Brief(
            now=facts.now,
            unanswered=facts.unanswered,
            max_bubbles=self._available(facts, facts.session),
            previous=previous,
        )
        draft = await self._decider.decide(candidate, brief)
        if not draft.usable:
            await asyncio.to_thread(self._decline, candidate, draft, facts, drawn)
            report.outcome = "declined" if draft.failure is None else "failed"
            report.reason = (draft.failure or Reason.PLANNER_DECLINED).value
            return
        await self._deliver(candidate, draft, drawn, arrivals, report)

    def _unanswered_texts(self, facts: Facts) -> tuple[str, ...]:
        """The bubbles of the proactive messages the user has not answered (worker thread)."""
        floor = facts.last_inbound or facts.now - UNANSWERED_LOOKBACK
        texts: list[str] = []
        for row in self.log.entries(outcomes=["sent"], since=floor, with_text=True):
            if row.at > floor and row.content:
                texts.extend(str(text) for text in row.content.get("bubbles", ()))
        return tuple(texts)

    def _available(self, facts: Facts, session: SessionState) -> int:
        """The bubbles one message may use: what is left, keeping one for a chase to come."""
        chase_left = facts.unanswered + 1 <= self._config.max_chase
        return max(1, session.remaining_quota - (1 if chase_left else 0))

    async def _deliver(
        self,
        candidate: Candidate,
        draft: Draft,
        drawn: bool,
        arrivals: int,
        report: TickReport,
    ) -> None:
        """The second check, the cut to the quota, the sending and the books (module text)."""
        now = self._clock.now_utc()
        facts = await asyncio.to_thread(self._gather, now)
        if self._conversation.arrivals != arrivals or await self._user_active(facts):
            await asyncio.to_thread(self._drop, candidate, facts, Reason.USER_ACTIVE, draft, drawn)
            report.outcome, report.reason = "dropped", Reason.USER_ACTIVE.value
            return
        refusal = check(candidate.kind, facts.situation)
        if refusal is None and candidate.kind is TriggerKind.GREETING:
            decision = await asyncio.to_thread(self._kit.planner.greeting_decision, now)
            if not decision.allowed:
                refusal = Refusal(Reason.GREETING_GAP, decision.reason)
        if refusal is not None:
            await asyncio.to_thread(self._refuse, candidate, refusal, facts, drawn, draft)
            report.outcome, report.reason = "rejected", refusal.reason.value
            return
        bubbles, fit_actions = fit_bubbles_to_quota(
            draft.bubbles, self._available(facts, facts.session)
        )
        pacing = await asyncio.to_thread(self._pacing, now)
        meta = ReplyMeta(
            backend=draft.backend,
            thinking=draft.thinking,
            plan=draft.plan,
            cost_usd=draft.cost_usd,
            timings_ms={"generate": draft.latency_ms},
            actions=self._actions(candidate, draft, facts, fit_actions),
        )
        state: dict[str, str | None] = {"log": None}

        async def first(sent: SentBubble, reply_id: str) -> None:
            state["log"] = await asyncio.to_thread(
                self._after_first_bubble, candidate, draft, facts, sent, reply_id, bubbles
            )

        outcome = await self._sender.send(
            bubbles, meta=meta, pacing=pacing, since_arrivals=arrivals, on_first=first
        )
        report.sent = outcome.sent
        if not outcome.started:
            report.outcome = await asyncio.to_thread(
                self._unsent, candidate, outcome, facts, draft, drawn
            )
            report.reason = _unsent_reason(outcome).value
            return
        await asyncio.to_thread(
            self._after_send, state["log"], outcome, bubbles, fit_actions, draft
        )
        report.outcome = "sent"

    # ------------------------------------------------------------------- the books

    def _base_log(
        self,
        facts: Facts,
        kind: str,
        outcome: str,
        *,
        reason: Reason | None = None,
        candidate: Candidate | None = None,
        draft: Draft | None = None,
        at: datetime | None = None,
    ) -> NewLog:
        moment = at or facts.now
        return NewLog(
            at=moment,
            candidate_at=candidate.planned_at if candidate else moment,
            local_date=moment.astimezone(facts.zone).date(),
            local_at=f"{moment.astimezone(facts.zone):%Y-%m-%d %H:%M}",
            timezone=facts.zone.key,
            kind=kind,
            outcome=outcome,
            reason=reason.value if reason else None,
            her_state=facts.state,
            chase_seq=facts.unanswered,
            quota_total=facts.plan.quota.total,
            range_min=facts.low,
            range_max=facts.high,
            enabled=facts.enabled,
            candidate_id=candidate.id if candidate else None,
            followup_id=candidate.followup_id if candidate else None,
            backend=draft.backend if draft else None,
            cost_usd=draft.cost_usd if draft and draft.cost_usd else None,
            plan_reason=draft.reason if draft and draft.reason else None,
        )

    def _log_candidate(
        self, facts: Facts, candidate: Candidate, outcome: str, reason: Reason
    ) -> None:
        self.log.add(
            self._base_log(facts, candidate.kind.value, outcome, reason=reason, candidate=candidate)
        )

    def _refuse(
        self,
        candidate: Candidate,
        refusal: Refusal,
        facts: Facts,
        drawn: bool,
        draft: Draft | None = None,
    ) -> None:
        """A hard constraint said no: write it down (once per reason for a waiting candidate)."""
        reason = refusal.reason
        if candidate.id is not None:
            row = self.candidates.get(candidate.id)
            if row is not None and row.last_reason == reason.value:
                return
            self.candidates.note_reason(candidate.id, reason)
        entry = self._base_log(
            facts, candidate.kind.value, "rejected", reason=reason, candidate=candidate, draft=draft
        )
        self.log.add(replace(entry, result={"detail": refusal.detail} if refusal.detail else None))
        if drawn:
            self._rest_after(candidate.kind.value, facts, reason, hard=True)
        log.info("proactive_refused", kind=candidate.kind.value, reason=reason.value)

    def _decline(self, candidate: Candidate, draft: Draft, facts: Facts, drawn: bool) -> None:
        """The planner said no (or could not say yes): write it down and decide what is left."""
        failure = draft.failure
        reason = failure if failure is not None else Reason.PLANNER_DECLINED
        outcome = "declined" if failure is None else "failed"
        entry = self._base_log(
            facts, candidate.kind.value, outcome, reason=reason, candidate=candidate, draft=draft
        )
        result = {"detail": draft.failure_detail} if draft.failure_detail else None
        self.log.add(replace(entry, result=result, content=None))
        if candidate.followup_id and draft.followup_done:
            self._followups.close(candidate.followup_id, status="done", reason="mentioned_before")
            if candidate.id is not None:
                self.candidates.mark(
                    candidate.id, "dropped", at=facts.now, reason=Reason.FOLLOWUP_CLOSED
                )
            return
        if drawn or candidate.id is None:
            self._rest_after(candidate.kind.value, facts, reason, hard=False)
            return
        self._retry_or_give_up(candidate, facts, reason)

    def _retry_or_give_up(self, candidate: Candidate, facts: Facts, reason: Reason) -> None:
        """A waiting candidate is planned again once or twice later in its window, then dropped."""
        if candidate.id is None:
            return
        if candidate.attempts + 1 < MAX_ATTEMPTS:
            later = facts.now + timedelta(
                seconds=self._rng.uniform(RESCHEDULE_MIN_S, RESCHEDULE_MAX_S)
            )
            if later < candidate.window_end:
                self.candidates.reschedule(candidate.id, later)
                return
        self.candidates.mark(candidate.id, "declined", at=facts.now, reason=reason)

    def _drop(
        self, candidate: Candidate, facts: Facts, reason: Reason, draft: Draft, drawn: bool
    ) -> None:
        """The message was made but not sent: the user started to talk, or the plan changed."""
        entry = self._base_log(
            facts, candidate.kind.value, "dropped", reason=reason, candidate=candidate, draft=draft
        )
        self.log.add(entry)
        if candidate.id is not None:
            self.candidates.mark(candidate.id, "dropped", at=facts.now, reason=reason)
        elif drawn:
            self._rest_after(candidate.kind.value, facts, reason, hard=False)

    def _unsent(
        self, candidate: Candidate, outcome: SendOutcome, facts: Facts, draft: Draft, drawn: bool
    ) -> str:
        """Not one bubble went out: the channel refused, or the user (or sleep) came first.

        Returns the outcome written to the log (``dropped``, ``rejected`` or ``failed``).
        """
        reason = _unsent_reason(outcome)
        label = _unsent_outcome(outcome, reason)
        entry = self._base_log(
            facts, candidate.kind.value, label, reason=reason, candidate=candidate, draft=draft
        )
        self.log.add(replace(entry, result={"stop": outcome.stop.value if outcome.stop else None}))
        if candidate.id is not None:
            self.candidates.mark(candidate.id, "dropped", at=facts.now, reason=reason)
        elif drawn:
            self._rest_after(candidate.kind.value, facts, reason, hard=False)
        return label

    def _actions(
        self, candidate: Candidate, draft: Draft, facts: Facts, fitted: Sequence[PostAction]
    ) -> tuple[dict[str, object], ...]:
        actions: list[dict[str, object]] = [action.to_json() for action in draft.actions]
        actions.extend(action.to_json() for action in fitted)
        actions.append({"step": "proactive", "count": 1, "detail": candidate.kind.value})
        if facts.unanswered:
            actions.append({"step": "chase", "count": facts.unanswered})
        if draft.shared_ids:
            actions.append(
                {"step": SHARED_STEP, "count": len(draft.shared_ids), "ids": list(draft.shared_ids)}
            )
        return tuple(actions)

    def _after_first_bubble(
        self,
        candidate: Candidate,
        draft: Draft,
        facts: Facts,
        sent: SentBubble,
        reply_id: str,
        bubbles: Sequence[Bubble],
    ) -> str:
        """The first bubble is out: this is when the message exists (worker thread)."""
        moment = sent.at
        state = self._state_at(moment) or facts.state
        entry = self._base_log(
            facts, candidate.kind.value, "sent", candidate=candidate, draft=draft, at=moment
        )
        stored = self.log.add(
            replace(
                entry,
                her_state=state,
                bubbles_sent=1,
                reply_id=reply_id,
                content={"bubbles": [b.text for b in bubbles], "plan": draft.plan},
            )
        )
        if candidate.id is not None:
            self.candidates.mark(candidate.id, "sent", at=moment)
        if candidate.kind is TriggerKind.GREETING:
            self._kit.planner.record_wake_greeting(moment)
        if candidate.followup_id:
            self._followups.close(candidate.followup_id, status="done", reason="asked", at=moment)
        self._tell_the_day(draft, facts, moment, reply_id)
        return stored.id

    def _state_at(self, moment: datetime) -> str | None:
        try:
            return str(self._kit.time.her_state(moment).kind)
        except PlanUnavailableError:
            return None

    def _tell_the_day(self, draft: Draft, facts: Facts, moment: datetime, reply_id: str) -> None:
        """What she told is marked as told; a new detail joins the life line (R-PRO-006)."""
        ids = list(draft.shared_ids)
        if draft.new_detail is not None:
            created = self._lifeline.add_improvised(facts.today, draft.new_detail)
            ids.append(created.id)
        if ids:
            self._lifeline.mark_shared(ids, at=moment, reply_id=reply_id)

    def _after_send(
        self,
        log_id: str | None,
        outcome: SendOutcome,
        bubbles: Sequence[Bubble],
        fitted: Sequence[PostAction],
        draft: Draft,
    ) -> None:
        """The message is over: complete its row with what really went out."""
        if log_id is None:
            return
        result: dict[str, object] = {
            "sent": outcome.sent,
            "planned": len(bubbles),
            "skipped": outcome.skipped,
        }
        if outcome.stop is not None:
            result["stop"] = outcome.stop.value
        if outcome.interrupted_by is not None:
            result["interrupted_by"] = outcome.interrupted_by
        if fitted:
            result["fitted"] = [action.to_json() for action in fitted]
        self.log.update(
            log_id,
            bubbles_sent=outcome.sent,
            result=result,
            content={"bubbles": outcome.texts, "plan": draft.plan},
        )


def _unsent_outcome(outcome: SendOutcome, reason: Reason) -> str:
    """How the log calls a message of which nothing went out."""
    if outcome.interrupted_by:
        return "dropped"
    if reason in (Reason.WINDOW_CLOSED, Reason.QUOTA_EXHAUSTED, Reason.CHANNEL_UNAVAILABLE):
        return "rejected"
    return "failed"


def _unsent_reason(outcome: SendOutcome) -> Reason:
    if outcome.interrupted_by == "user":
        return Reason.USER_ACTIVE
    if outcome.interrupted_by == "sleep":
        return Reason.DEEP_SLEEP
    if outcome.interrupted_by == "resume":
        return Reason.INTERRUPTED
    stop = outcome.stop
    if stop is StopReason.EXPIRED:
        return Reason.WINDOW_CLOSED
    if stop is StopReason.QUOTA:
        return Reason.QUOTA_EXHAUSTED
    if stop is StopReason.UNBOUND:
        return Reason.CHANNEL_UNAVAILABLE
    return Reason.SEND_FAILED


__all__ = ["Conversation", "Decider", "Facts", "ProactiveScheduler", "TickReport", "Turns"]
