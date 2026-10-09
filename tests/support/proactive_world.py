"""A synthetic world for the proactive tests and the simulator (round 10).

Nothing here comes from real chat data.  The world wires the **production** scheduler
(:func:`twin.schedule.proactive.component.build_scheduler`) over:

* a services container on the migrated test database, with the real schedule kit of
  ``tests.support.routine`` (a student who sleeps 23:30-07:30 and is busy 13:00-17:00) in place of
  the container's own, so the day plans, her state and the quota are the real ones;
* a real engine (``build_engine``) on a channel that keeps the platform window and message count
  like WeChat does (:class:`WindowedChannel`) - the engine is never started, only its parts are
  used, exactly as ``twin run`` shares them;
* DeepSeek as a ``respx`` side effect (:class:`ProactiveScript`): it answers the planner's JSON
  from the kind of message it is asked about, with knobs to decline, fail or answer by hand.

``Talk`` stands in for the engine as the scheduler's view of the conversation (idle or not, how
many messages the user has sent); the user is simulated with :meth:`World.user_writes`.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from tests.support.clock import ManualClock
from tests.support.deepseek import TEST_KEY, completion, error
from tests.support.engine_harness import Out
from tests.support.routine import Rig, student_model
from twin.channel.base import (
    AuthState,
    Channel,
    ChannelCapabilities,
    InboundMessage,
    OutboundKind,
    OutboundResult,
    QuoteTarget,
    SendBypass,
    SessionState,
)
from twin.channel.window import SessionWindow
from twin.engine.component import EngineComponent, build_engine
from twin.engine.machine import ConversationEngine
from twin.engine.state_store import ConversationSnapshot, ConversationStateStore
from twin.engine.turns import BotTurnStore, OutboundBubble, ReplyMeta
from twin.llm.ledger import LedgerRecord
from twin.llm.runtime import DEEPSEEK_SECRET
from twin.llm.types import CostBreakdown, LedgerTag, Usage
from twin.memory.followups import FollowupStore
from twin.memory.lifeline import LifelineStore
from twin.profile.activity_model import ActivityModel
from twin.profile.distribution import EmpiricalDistribution
from twin.schedule.events import CandidatesExpired, PlanRebuilt, ScheduleEvent, TimezoneSwitched
from twin.schedule.plan_builder import QuotaRange
from twin.schedule.proactive.component import build_scheduler
from twin.schedule.proactive.scheduler import ProactiveScheduler, TickReport
from twin.schedule.proactive.store import (
    CandidateRow,
    CandidateStore,
    NewCandidate,
    NewLog,
    ProactiveLogStore,
    RatingStore,
)
from twin.schedule.proactive.types import PRIORITY, TriggerKind
from twin.schedule.service import KIT_KEY
from twin.services import Services

SLOTS = 96


def opening_curve(
    *, base: float = 0.02, peaks: Mapping[int, float] | None = None
) -> tuple[float, ...]:
    """Conversations she opens per 15-minute slot: quiet most of the day, up at the meals."""
    values = [base] * SLOTS
    for slot, value in (MEAL_PEAKS if peaks is None else peaks).items():
        values[slot] = value
    return tuple(values)


MEAL_PEAKS: Mapping[int, float] = {
    31: 0.5,  # 07:45
    32: 0.5,  # 08:00
    48: 0.6,  # 12:00
    49: 0.6,  # 12:15
    73: 0.6,  # 18:15
    74: 0.6,  # 18:30
    87: 0.4,  # 21:45
    88: 0.4,  # 22:00
}


def proactive_model(
    curve: tuple[float, ...] | None = None, *, initiations: float = 3.6
) -> ActivityModel:
    """The routine of the world: ``tests.support.routine.student_model`` with an opening curve."""
    return student_model(rate=curve or opening_curve(), initiations=initiations)


class Always(random.Random):
    """Every draw comes out lowest: any chance above zero is taken (a test makes her write)."""

    def random(self) -> float:
        return 0.0


class Never(random.Random):
    """Every draw comes out highest: no chance below one is taken (a test draws nothing)."""

    def random(self) -> float:
        return 0.9999999


# ------------------------------------------------------------------------------ channel


class WindowedChannel(Channel):
    """A channel with the platform's window and count (:class:`SessionWindow`), recording sends."""

    def __init__(
        self,
        clock: ManualClock,
        *,
        window_h: float = 22.0,
        quota: int = 8,
        supports_typing: bool = True,
    ) -> None:
        self.clock = clock
        self.window = SessionWindow(window_h=window_h, quota=quota)
        self.supports_typing = supports_typing
        self.out: list[Out] = []
        self.refused: list[OutboundResult] = []
        self.bound = True
        self.auth = AuthState.OK
        self.fail_with: OutboundResult | None = None
        self._numbers = 0
        self.on_send: Callable[[Out], None] | None = None

    # the user ----------------------------------------------------------------------

    def user_writes(self, at: datetime | None = None) -> None:
        """A message of the user reached the platform: the window and the count start afresh."""
        self.window.on_inbound(at or self.clock.now_utc())

    # Channel -----------------------------------------------------------------------

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def incoming(self) -> AsyncIterator[InboundMessage]:
        for message in ():  # nobody writes through this channel: the tests call the scheduler
            yield message

    def _gate(self) -> OutboundResult | None:
        if self.fail_with is not None:
            return self.fail_with
        now = self.clock.now_utc()
        reason = self.window.refusal_reason(now)
        if reason is not None:
            return OutboundResult.failure(OutboundKind.WINDOW_REJECTED, reason)
        return None

    def _record(self, item: Out) -> None:
        self.out.append(item)
        if self.on_send is not None:
            self.on_send(item)

    async def send_text(
        self,
        text: str,
        quote: QuoteTarget | None = None,
        *,
        recipient: str | None = None,
        bypass: SendBypass | None = None,
    ) -> OutboundResult:
        refused = self._gate()
        if refused is not None:
            self.refused.append(refused)
            return refused
        self.window.on_outbound(1)
        self._numbers += 1
        self._record(Out("text", text, self.clock.now_utc(), quote))
        return OutboundResult.success(message_id=f"w{self._numbers}")

    async def send_image(
        self,
        data: bytes | Path,
        mime: str,
        *,
        recipient: str | None = None,
        bypass: SendBypass | None = None,
    ) -> OutboundResult:
        refused = self._gate()
        if refused is not None:
            self.refused.append(refused)
            return refused
        self.window.on_outbound(1)
        self._numbers += 1
        payload = (
            data if isinstance(data, bytes) else await asyncio.to_thread(Path(data).read_bytes)
        )
        self._record(Out("image", None, self.clock.now_utc(), data=payload))
        return OutboundResult.success(message_id=f"w{self._numbers}")

    async def send_typing(self, active: bool, *, recipient: str | None = None) -> None:
        self._record(Out("typing", None, self.clock.now_utc(), active=active))

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            supports_quote=False,
            supports_typing=self.supports_typing,
            proactive_window_h=self.window.window_h,
            outbound_quota=self.window.quota,
            max_text_chars=4000,
        )

    def session_state(self) -> SessionState:
        now = self.clock.now_utc()
        return SessionState(
            auth=self.auth,
            bound=self.bound,
            last_inbound_at=self.window.last_inbound_at,
            outbound_since_inbound=self.window.outbound_since_inbound,
            expired=self.window.expired,
            remaining_quota=self.window.remaining_quota(),
            window_remaining=self.window.window_remaining(now),
            has_context_token=True,
        )

    @property
    def texts(self) -> list[str]:
        return [item.text for item in self.out if item.kind == "text" and item.text is not None]


# ---------------------------------------------------------------------- conversation


class Talk:
    """The scheduler's view of the conversation: idle unless told otherwise (a test double)."""

    def __init__(self, services: Services, clock: ManualClock) -> None:
        self._state = ConversationStateStore(services.db, services.clock)
        self._clock = clock
        self.count = 0
        self._signal = asyncio.Event()

    def snapshot(self) -> ConversationSnapshot:
        return self._state.load()

    @property
    def arrivals(self) -> int:
        return self.count

    def user_wrote(self) -> None:
        self.count += 1
        self._signal.set()

    def busy(self) -> None:
        """The engine is in the middle of a round (a message waits for its answer)."""
        self._state.transition("COLLECTING", pending=("m1",))

    def idle(self) -> None:
        self._state.transition("IDLE", pending=())

    async def wait_for_arrival(self, since: int, seconds: float) -> bool:
        end = self._clock.monotonic() + seconds
        while True:
            if self.count != since:
                return True
            remaining = end - self._clock.monotonic()
            if remaining <= 0:
                return False
            self._signal.clear()
            waker = asyncio.ensure_future(self._signal.wait())
            sleeper = asyncio.ensure_future(self._clock.sleep(remaining))
            await asyncio.wait({waker, sleeper}, return_when=asyncio.FIRST_COMPLETED)
            for task in (waker, sleeper):
                task.cancel()
            await asyncio.gather(waker, sleeper, return_exceptions=True)


# ----------------------------------------------------------------------------- DeepSeek

DEFAULT_MESSAGES: Mapping[str, tuple[str, ...]] = {
    "greeting": ("早啊", "刚醒"),
    "meal": ("吃饭了吗",),
    "bedtime": ("我先睡啦", "晚安"),
    "followup": ("考试怎么样了",),
    "silence": ("在干嘛呀",),
    "share": ("刚刚在图书馆看文献", "有点困"),
    "edge": ("睡不着",),
}
KIND = re.compile(r"类型：([a-z]+)（")


@dataclass
class ProactiveScript:
    """DeepSeek for the planner: a ``respx`` side effect that answers by the kind asked about."""

    messages: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(DEFAULT_MESSAGES))
    decline: set[str] = field(default_factory=set)
    invalid_first: int = 0  # answer this many requests with text that is not JSON
    finish: str = "stop"  # the finish reason of the answers ("content_filter": refused)
    unavailable: bool = False  # the account is out of balance (an error that is not retried)
    refuse_with: str | None = None  # answer 400 with this message (the request was refused)
    by_hand: Callable[[str, dict[str, Any]], dict[str, Any] | str] | None = None
    requests: list[dict[str, Any]] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    on_request: Callable[[str], None] | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        last = str(body["messages"][-1]["content"])
        found = KIND.search(last)
        kind = found.group(1) if found else "unknown"
        self.kinds.append(kind)
        if self.on_request is not None:
            self.on_request(kind)
        if self.unavailable:
            return error(402, "insufficient balance")
        if self.refuse_with is not None:
            return error(400, self.refuse_with)
        if len(self.requests) <= self.invalid_first:
            return httpx.Response(200, json=completion("not json at all", request_id="bad"))
        plan: dict[str, Any] | str
        if self.by_hand is not None:
            plan = self.by_hand(kind, body)
        elif kind in self.decline:
            plan = {"send": False, "kind": kind, "messages": [], "reason": "现在发不合适"}
        else:
            plan = {
                "send": True,
                "kind": kind,
                "messages": list(self.messages.get(kind, ("嗯",))),
                "sticker_hint": "",
                "reason": f"{kind} 的理由",
                "intent": "找他聊聊",
                "tone": "随意",
            }
        text = plan if isinstance(plan, str) else json.dumps(plan, ensure_ascii=False)
        return httpx.Response(
            200,
            json=completion(
                text,
                prompt=400,
                completion_tokens=30,
                finish=self.finish,
                request_id=f"r{len(self.requests)}",
            ),
        )


# ------------------------------------------------------------------------------ the world


@dataclass
class World:
    """Everything a proactive test touches."""

    services: Services
    clock: ManualClock
    rig: Rig
    channel: WindowedChannel
    engine: ConversationEngine
    talk: Talk
    scheduler: ProactiveScheduler
    script: ProactiveScript
    turns: BotTurnStore
    log: ProactiveLogStore
    candidates: CandidateStore
    ratings: RatingStore
    followups: FollowupStore
    lifeline: LifelineStore
    waited: list[float] = field(default_factory=list)

    # time ---------------------------------------------------------------------------

    def at(self, hour: int, minute: int = 0, *, day: int = 9, month: int = 10) -> datetime:
        """The instant of a clock time in the bot's zone (default: Friday 2026-10-09)."""
        return self.rig.at(2026, month, day, hour, minute)

    def go_to(self, moment: datetime) -> None:
        self.clock.set_time(moment)

    async def tick_at(self, moment: datetime) -> TickReport:
        self.clock.set_time(moment)
        return await self.scheduler.tick()

    # the user -------------------------------------------------------------------------

    def user_writes(
        self, text: str = "在吗", *, answered: str | None = "嗯", at: datetime | None = None
    ) -> None:
        """The user writes (the window restarts) and, optionally, she answers: both are stored."""
        moment = at or self.clock.now_utc()
        self.channel.user_writes(moment)
        self.talk.user_wrote()
        self.turns.add_inbound(at=moment, kind="text", text=text, external_id=f"u{self.talk.count}")
        if answered is not None:
            self.turns.add_reply(
                [OutboundBubble(answered, moment + timedelta(seconds=30))], ReplyMeta("deepseek")
            )
            self.channel.window.on_outbound(1)

    def sent_rows(self) -> list[Any]:
        return self.log.entries(outcomes=["sent"], with_text=True)

    def add_log(
        self,
        at: datetime,
        *,
        kind: str = "share",
        outcome: str = "sent",
        reason: str | None = None,
        state: str | None = "free",
        chase: int = 0,
        bubbles: int | None = None,
    ) -> Any:
        """Write a row of the log by hand (what an earlier run of the scheduler left)."""
        zone = self.rig.kit.time.bot_timezone()
        local = at.astimezone(zone)
        return self.log.add(
            NewLog(
                at=at,
                candidate_at=at,
                local_date=local.date(),
                local_at=f"{local:%Y-%m-%d %H:%M}",
                timezone=zone.key,
                kind=kind,
                outcome=outcome,
                reason=reason,
                her_state=state,
                chase_seq=chase,
                bubbles_sent=(1 if outcome == "sent" else 0) if bubbles is None else bubbles,
            )
        )

    def rejected(self) -> list[Any]:
        return self.log.entries(outcomes=["rejected"])

    def reasons(self, outcome: str = "rejected") -> list[str | None]:
        return [row.reason for row in self.log.entries(outcomes=[outcome])]

    # the schedule ----------------------------------------------------------------------

    def put(
        self,
        kind: TriggerKind,
        planned_at: datetime,
        *,
        minutes: int = 60,
        key: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> CandidateRow:
        """Put a candidate into the books by hand (a slot the day plan did not lay out)."""
        zone = self.rig.kit.time.bot_timezone()
        row = self.candidates.add(
            NewCandidate(
                local_date=planned_at.astimezone(zone).date(),
                timezone=zone.key,
                key=key or f"{kind.value}:{planned_at.isoformat()}",
                kind=kind,
                priority=PRIORITY[kind],
                planned_at=planned_at,
                window_end=planned_at + timedelta(minutes=minutes),
                detail=detail or {},
            )
        )
        assert row is not None
        return row

    async def publish(self, events: tuple[ScheduleEvent, ...]) -> None:
        """Tell the scheduler what the schedule decided (as the schedule component does)."""
        for event in events:
            await self.rig.kit.events.publish(event)

    def exhaust_budget(self) -> None:
        """Spend today's whole budget twice over: level 3, where proactive messages stop."""
        llm = self.engine.kit.llm  # type: ignore[union-attr]
        llm.ledger.record(
            LedgerRecord(
                provider="deepseek",
                model="deepseek-flash",
                purpose="reply",
                usage=Usage(prompt_tokens=1, completion_tokens=1, cache_miss_tokens=1),
                cost=CostBreakdown(
                    2 * self.services.settings.budget.daily_usd, 0.0, 0.0, True, 1.0
                ),
                thinking=False,
                latency_ms=1,
                at=self.clock.now_utc(),
                tag=LedgerTag(),
            )
        )
        llm.budget.status(force=True)

    async def aclose(self) -> None:
        await self.engine.kit.llm.client.aclose()  # type: ignore[union-attr]
        await self.engine.kit.data.aclose()  # type: ignore[union-attr]


def build_world(
    services: Services,
    clock: ManualClock,
    *,
    model: ActivityModel | None = None,
    quota: QuotaRange | None = None,
    zone: str | None = None,
    window_h: float = 22.0,
    bubbles: int = 8,
    seed: int = 11,
    script: ProactiveScript | None = None,
    instant: bool = True,
    silence: EmpiricalDistribution | None = None,
    rng: random.Random | None = None,
    salt: str = "proactive-world",
) -> World:
    """Wire the production scheduler over the synthetic world (see the module description).

    ``instant`` replaces the pauses between bubbles by a jump of the manual clock (the user can
    still interrupt: a message that arrives makes the pause return ``True``).
    """
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    rig = Rig.build(
        services,
        clock,
        model if model is not None else proactive_model(),
        zone=zone,
        quota=quota,
        salt=salt,
    )
    services.extras[KIT_KEY] = rig.kit
    channel = WindowedChannel(clock, window_h=window_h, quota=bubbles)
    engine = build_engine(services, channel, rng=random.Random(seed))
    component = EngineComponent(engine, channel, services)
    talk = Talk(services, clock)
    waited: list[float] = []

    async def jump(since: int, seconds: float) -> bool:
        waited.append(seconds)
        clock.tick(seconds)
        return talk.count != since

    scheduler = build_scheduler(
        services,
        component,
        conversation=talk,
        model_source=lambda: rig.holder["model"],
        silence_source=lambda: silence,
        pause=jump if instant else None,
        rng=rng or random.Random(seed),
    )
    kit = rig.kit

    async def on_plan_change(_event: PlanRebuilt | TimezoneSwitched) -> None:
        scheduler.forget_day()

    kit.events.subscribe(CandidatesExpired, scheduler.on_expired)
    kit.events.subscribe(PlanRebuilt, on_plan_change)
    kit.events.subscribe(TimezoneSwitched, on_plan_change)
    chosen = script or ProactiveScript()
    return World(
        services=services,
        clock=clock,
        rig=rig,
        channel=channel,
        engine=engine,
        talk=talk,
        scheduler=scheduler,
        script=chosen,
        turns=BotTurnStore(services.db, services.clock),
        log=ProactiveLogStore(services.db, services.clock),
        candidates=CandidateStore(services.db, services.clock),
        ratings=RatingStore(services.db, services.clock),
        followups=FollowupStore(engine.kit.data.memory),  # type: ignore[union-attr]
        lifeline=LifelineStore(engine.kit.data.memory, time_service=rig.kit.time),  # type: ignore[union-attr]
        waited=waited,
    )
