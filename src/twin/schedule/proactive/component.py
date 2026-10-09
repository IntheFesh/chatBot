"""The proactive scheduler as an application component, and how it is wired (R-ARCH-001, R-PRO-001).

``ProactiveComponent`` calls :meth:`~twin.schedule.proactive.scheduler.ProactiveScheduler.tick`
every ``proactive.tick_minutes``.  The ticks are aligned to the wall clock - a multiple of the
tick length plus a fixed offset of the local day (drawn from the installation's salt), so that
they do not coincide with the other jobs of the application that run on the full five minutes -
and the first one is made at once when the component starts (it opens the day).

:func:`register_proactive` builds the scheduler from the pieces the running engine is made of
(``engine.kit``: the same reply pipeline, data source, sender, style model and DeepSeek client),
subscribes it to the schedule's events (a restart, a wake-up from sleep or a time zone switch voids
its candidates, R-SCH-005) and adds the component after the engine and the schedule.
:func:`proactive_status_for` makes the source of the proactive line of ``/状态`` - it needs nothing
of the engine, so it can be passed to :func:`~twin.engine.component.register_engine` before the
scheduler exists - and :func:`register_rating_command` adds ``/评分`` to the command table.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from collections.abc import Awaitable, Callable, Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING

from twin.app import Application, ComponentHealth, TaskSupervisor
from twin.channel.base import SessionState
from twin.clock import Clock
from twin.commands.rating import rating_command
from twin.commands.status import ProactiveStatus
from twin.engine.component import EngineComponent
from twin.engine.pacing import PacingModel
from twin.engine.pipeline import ReplyPipeline
from twin.engine.sender import BubbleSender
from twin.memory.followups import FollowupStore
from twin.memory.lifeline import LifelineStore
from twin.memory.recent import Turn, bot_turn_reader, merge_turns
from twin.ops.logging import get_logger
from twin.profile.activity_model import ActivityModel
from twin.profile.api import load_activity_model, load_profile
from twin.profile.distribution import EmpiricalDistribution
from twin.profile.prompt_templates import PROACTIVE_PLAN, TemplateStore
from twin.retrieval.openers import OpenerExamples
from twin.schedule.events import CandidatesExpired, PlanRebuilt, TimezoneSwitched
from twin.schedule.proactive.decide import (
    HISTORY_MESSAGES,
    LiveMaterial,
    ProactiveDecider,
)
from twin.schedule.proactive.scheduler import Conversation, ProactiveScheduler
from twin.schedule.proactive.send import ProactiveSender
from twin.schedule.proactive.status import ProactiveStatusSource
from twin.schedule.proactive.store import CandidateStore, ProactiveLogStore, RatingStore
from twin.schedule.service import ScheduleKit, schedule_kit
from twin.schedule.time_service import PlanUnavailableError
from twin.stickers.tags import load_vocabulary

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.schedule.proactive.component")

COMPONENT_NAME = "proactive"
OFFSET_MARGIN_S = 10.0  # a tick is never closer than this to the full tick
EXAMPLES_CAP = 5  # her real openings shown to the planner (fewer than a reply's examples)


def tick_offset_s(salt: str, day: date, tick_s: float) -> float:
    """The fixed offset of the ticks on ``day``: after the full tick, by a draw of the day."""
    digest = hashlib.sha256(f"{salt}|{day.isoformat()}|proactive-tick".encode()).digest()
    span = max(1.0, tick_s - 2 * OFFSET_MARGIN_S)
    return OFFSET_MARGIN_S + (int.from_bytes(digest[:4], "big") / 2**32) * span


def next_tick_at(now: datetime, tick_s: float, offset_s: float) -> datetime:
    """The first aligned tick strictly after ``now``: a multiple of ``tick_s`` plus the offset."""
    seconds = now.timestamp()
    number = int((seconds - offset_s) // tick_s) + 1
    return datetime.fromtimestamp(number * tick_s + offset_s, tz=now.tzinfo)


class ProactiveComponent:
    """Runs :meth:`ProactiveScheduler.tick` on the aligned ticks."""

    name = COMPONENT_NAME
    depends_on: Sequence[str] = ("schedule", "engine")

    def __init__(
        self,
        scheduler: ProactiveScheduler,
        services: Services,
        kit: ScheduleKit,
        *,
        first_tick: bool = True,
    ) -> None:
        self.scheduler = scheduler
        self._services = services
        self._kit = kit
        self._first_tick = first_tick
        self._supervisor = TaskSupervisor(self.name, services.clock, services.alerts)
        self._tick_s = services.settings.proactive.tick_minutes * 60.0
        self._offsets: dict[date, float] = {}
        self.ticks = 0

    async def start(self) -> None:
        if self._first_tick:
            await self._tick()
        self._supervisor.spawn("tick", self._loop, restart_on_exit=True)

    async def stop(self) -> None:
        await self._supervisor.stop()

    def health(self) -> ComponentHealth:
        return self._supervisor.health()

    async def _tick(self) -> None:
        try:
            await self.scheduler.tick()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # one bad tick must not end the schedule of proactive messages
            log.exception("proactive_tick_failed", error=type(exc).__name__)
        self.ticks += 1

    def _offset(self, now: datetime) -> float:
        day = self._kit.time.local_date(now)
        if day not in self._offsets:
            self._offsets = {day: tick_offset_s(self._kit.planner.salt.get(), day, self._tick_s)}
        return self._offsets[day]

    async def _loop(self) -> None:
        clock = self._services.clock
        while True:
            now = clock.now_utc()
            offset = await asyncio.to_thread(self._offset, now)
            wait = (next_tick_at(now, self._tick_s, offset) - now).total_seconds()
            await clock.sleep(max(1.0, wait))
            await self._tick()


# ------------------------------------------------------------------------------ wiring


def proactive_status_for(
    services: Services, channel_state: Callable[[], SessionState]
) -> ProactiveStatusSource:
    """The source of the proactive line of ``/状态`` (it reads, it never writes)."""
    return ProactiveStatusSource(
        clock=services.clock,
        schedule=schedule_kit(services),
        runtime=services.runtime,
        config=services.settings.proactive,
        log_store=ProactiveLogStore(services.db, services.clock),
        candidates=CandidateStore(services.db, services.clock),
        channel_state=channel_state,
    )


def build_scheduler(
    services: Services,
    engine_component: EngineComponent,
    *,
    conversation: Conversation | None = None,
    model_source: Callable[[], ActivityModel | None] | None = None,
    silence_source: Callable[[], EmpiricalDistribution | None] | None = None,
    pause: Callable[[int, float], Awaitable[bool]] | None = None,
    rng: random.Random | None = None,
) -> ProactiveScheduler:
    """The scheduler of this process, made of the parts the running engine is made of.

    The keyword arguments replace one piece each (the tests and the simulator put a routine
    model, a conversation or a clock-driven pause of their own in its place); the running
    application passes none.
    """
    engine = engine_component.engine
    parts = engine.kit
    if parts is None or not isinstance(parts.pipeline, ReplyPipeline):
        raise RuntimeError(
            "the proactive scheduler needs an engine built by build_engine with the reply pipeline"
        )
    settings = services.settings
    clock: Clock = services.clock
    kit = schedule_kit(services)
    memory = parts.data.memory
    lifeline = LifelineStore(memory, time_service=kit.time)
    followups = FollowupStore(memory)
    reader = bot_turn_reader(services)
    rng = rng or random.Random()  # noqa: S311 - the draws of the schedule, not security
    openers = OpenerExamples(services, rng=rng)
    budget = parts.llm.budget

    def read_turns() -> Sequence[Turn]:
        if reader is None:
            return ()
        return merge_turns(reader.messages_since(None, HISTORY_MESSAGES))

    def examples_k() -> int:
        return min(EXAMPLES_CAP, budget.limits().examples_k)

    material = LiveMaterial(
        data_view=parts.data.view,
        lifeline=lifeline,
        followups=followups,
        openers=openers,
        read_turns=read_turns,
        local_date_of=kit.time.local_date,
        examples_k=examples_k,
        memory_tokens=settings.memory.block_tokens,
        last_interaction=lambda: parts.store.last_message_at(),
    )
    decider = ProactiveDecider(
        client=parts.llm.client,
        template=TemplateStore(services.db, clock).active(PROACTIVE_PLAN),
        sticker_tags=load_vocabulary(settings, services.paths.root).tags,
        material=material,
        pipeline=parts.pipeline,
        runtime=services.runtime,
        planner_thinking_allowed=lambda: budget.limits().planner_thinking_allowed,
        backends=parts.style.selector,
        style_writer=parts.style.hybrid.writer,
        auto_rules=settings.thinking.auto_rules,
        rng=rng,
    )
    talk: Conversation = conversation or engine
    scheduler_box: list[ProactiveScheduler] = []

    def asleep(moment: datetime) -> bool:
        try:
            return str(kit.time.her_state(moment).kind) == "deep_sleep"
        except PlanUnavailableError:
            return False

    sender = ProactiveSender(
        sender=BubbleSender(
            parts.channel, parts.stickers, parts.lookup, clock, services.alerts, rng
        ),
        store=parts.store,
        clock=clock,
        arrivals=lambda: talk.arrivals,
        wait_for_arrival=pause or talk.wait_for_arrival,
        asleep=asleep,
        epoch=lambda: scheduler_box[0].epoch if scheduler_box else 0,
    )

    def silence() -> EmpiricalDistribution | None:
        profile = load_profile(services, "live")
        return profile.metrics.distribution("her", "initiation_silence_s") if profile else None

    scheduler = ProactiveScheduler(
        clock=clock,
        config=settings.proactive,
        schedule=kit,
        runtime=services.runtime,
        channel_state=parts.channel.session_state,
        conversation=talk,
        turns=parts.store,
        candidates=CandidateStore(services.db, clock),
        log_store=ProactiveLogStore(services.db, clock),
        followups=followups,
        lifeline=lifeline,
        decider=decider,
        sender=sender,
        pacing=lambda moment: PacingModel.from_view(parts.data.view(moment)),
        model_source=model_source or (lambda: load_activity_model(services, "live")),
        silence_source=silence_source or silence,
        budget_allowed=lambda: budget.limits().proactive_allowed,
        rng=rng,
    )
    scheduler_box.append(scheduler)
    return scheduler


def register_proactive(
    application: Application, services: Services, engine_component: EngineComponent
) -> ProactiveComponent:
    """Build the scheduler, subscribe it to the schedule's events and add the component."""
    kit = schedule_kit(services)
    scheduler = build_scheduler(services, engine_component)

    async def on_plan_change(_event: PlanRebuilt | TimezoneSwitched) -> None:
        scheduler.forget_day()

    kit.events.subscribe(CandidatesExpired, scheduler.on_expired)
    kit.events.subscribe(PlanRebuilt, on_plan_change)
    kit.events.subscribe(TimezoneSwitched, on_plan_change)
    router = engine_component.router
    if router is not None:  # the commands of this round (the full table is round 11's)
        router.register(
            rating_command(RatingStore(services.db, services.clock), services.clock, kit.time)
        )
    component = ProactiveComponent(scheduler, services, kit)
    application.register(component)
    return component


__all__ = [
    "COMPONENT_NAME",
    "ProactiveComponent",
    "ProactiveStatus",
    "build_scheduler",
    "next_tick_at",
    "proactive_status_for",
    "register_proactive",
    "tick_offset_s",
]
