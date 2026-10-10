"""The engine in the running application (R-ARCH-001, R-ARCH-004, round 09 section J).

:func:`build_engine` wires a :class:`~twin.engine.machine.ConversationEngine` from the services of
the process and a channel: the live data source and the reply pipeline, the stores of the
conversation, the crisis handler, the sticker sender, her pacing from the profile.  The engine is
the same for the WeChat channel and the terminal channel; only the channel differs.

:class:`EngineComponent` is the application component around it: it starts the engine and feeds it
the messages of ``channel.incoming()`` one by one - the next message is asked for only after the
previous one is stored, which is what lets the channel hand every message over at least once
(R-CH-004) without losing one to a crash.  :func:`register_engine` adds the component to an
:class:`~twin.app.Application` and subscribes the engine to what it must hear: the schedule's
``Resumed`` (start-up and waking from sleep, R-SCH-005) and the state watcher (a changed setting).

Two more things are wired here, and only here:

* the style model (:class:`~twin.engine.style_runtime.StyleRuntime`): its backends join the
  reply pipeline, its :class:`~twin.engine.backend_select.BackendSelector` chooses the backend of
  every reply and hears how it went (the fallback to DeepSeek and the way back, R-SRV-004), the
  budget manager asks it for the last degradation level (R-LLM-008), and
  :class:`~twin.engine.backend_select.BackendMonitorComponent` keeps looking at the model while
  nobody talks;
* the command router of round 09-2 (:class:`~twin.commands.router.CommandRouter`) in
  :func:`command_port_for` - the one place that names it.  Its ``/状态`` carries the suggested
  quiet window of the engine (R-ENG-002); later rounds add their commands with
  ``router.register`` (``engine.commands`` is the router in use).  Round 11 puts its commands in
  the same router (``/时区 /暂停 /主动 /作息 /记住 /忘掉 /记忆 /不像 /费用 /导入``) and gives the
  engine the service for corrections in plain words
  (:class:`~twin.learning.corrections.CorrectionService`, ``engine.corrections``).

:func:`register_engine` also adds the two small components of round 11: the one that reports how
an import started from the chat ended (:class:`~twin.commands.import_report.ImportReportComponent`)
and the one that queues the weekly consolidation of the correction rules
(:class:`~twin.learning.component.LearningComponent`; ``learning=False`` leaves it out - the
interactive terminal chat is a short session, the weekly look belongs to ``twin run``, and an hour
long sleep would also make a virtual test clock jump an hour at a time).
"""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING

from twin.app import Application, ComponentHealth, TaskSupervisor
from twin.channel.base import Channel
from twin.commands import texts
from twin.commands.import_report import ImportReportComponent
from twin.commands.router import CommandRouter
from twin.commands.routine_commands import ComponentSchedule, ScheduleControl
from twin.commands.status import ProactiveStatus
from twin.engine.command_port import CommandPort
from twin.engine.dataview import LiveDataSource
from twin.engine.fallback import ShortAnswers
from twin.engine.feedback import FeedbackStore
from twin.engine.history import HistoryLoader
from twin.engine.inbound import InboundRenderer
from twin.engine.kit import EngineKit
from twin.engine.machine import ConversationEngine, DraftWriter
from twin.engine.pipeline import ReplyPipeline
from twin.engine.rounds import RoundStore
from twin.engine.safety.crisis import CrisisHandler
from twin.engine.safety.notifier import EmergencyNotifier
from twin.engine.state_store import ConversationStateStore
from twin.engine.sticker_sender import StickerSender
from twin.engine.style_runtime import StyleRuntime
from twin.engine.turns import BotTurnMessages, BotTurnStore
from twin.learning.component import LearningComponent
from twin.learning.corrections import CorrectionService
from twin.learning.dislike import NotLikeRecorder
from twin.learning.pairs import PreferencePairStore
from twin.learning.sample import SampleBuilder
from twin.llm.runtime import DEEPSEEK_SECRET, LlmRuntime, build_llm_runtime
from twin.memory.jobs import queue_bot_extraction
from twin.memory.memory import Memory
from twin.memory.recent import BotMessage, HistoryWindow
from twin.ops.logging import get_logger
from twin.ops.state_watch import StateWatcher
from twin.ops.wiring import build_emergency_notifier
from twin.profile.api import load_profile
from twin.schedule.component import ScheduleComponent
from twin.schedule.events import Resumed
from twin.schedule.service import schedule_kit, time_service_for
from twin.stickers.catalog import StickerCatalog

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.engine.component")

COMPONENT_NAME = "engine"


def quiet_window_line(engine: ConversationEngine) -> Callable[[], Awaitable[str | None]]:
    """The line of ``/状态`` that shows the quiet window of the engine and the suggested one."""

    async def line() -> str | None:
        window = await engine.quiet_window()
        return texts.STATUS_QUIET_WINDOW.format(
            configured=f"{window.configured_s:g}",
            suggested=f"{window.suggested_s:.0f}",
            adaptive="开" if window.adaptive else "关",
        )

    return line


def command_port_for(
    services: Services,
    engine: ConversationEngine,
    *,
    llm: LlmRuntime,
    style: StyleRuntime,
    channel: Channel,
    memory: Memory,
    data: LiveDataSource | None = None,
    schedule: ScheduleControl | None = None,
    recorder: NotLikeRecorder | None = None,
    proactive: Callable[[], ProactiveStatus | None] | None = None,
) -> CommandRouter:
    """The command router of the application (the one place that builds it).

    ``/状态`` reads the channel's session for the platform window and the engine for the
    suggested quiet window; ``/重来`` deletes what the thrown-away reply made up from ``memory``.
    ``schedule`` is how ``/时区`` and ``/作息`` reach the running schedule component (without
    it they use the schedule kit directly), ``data`` and ``recorder`` serve ``/不像``.
    """
    return CommandRouter.from_services(
        services,
        llm,
        selector=style.selector,
        turns=BotTurnStore(services.db, services.clock),
        feedback=FeedbackStore(services.db, services.clock),
        memory=memory,
        session_state=channel.session_state,
        proactive=proactive,
        extra_status=(quiet_window_line(engine),),
        schedule=schedule,
        data=data,
        recorder=recorder,
    )


def build_engine(
    services: Services,
    channel: Channel,
    *,
    runtime: LlmRuntime | None = None,
    style: StyleRuntime | None = None,
    pipeline: DraftWriter | None = None,
    commands: CommandPort | None = None,
    notifier: EmergencyNotifier | None = None,
    rng: random.Random | None = None,
    schedule: ScheduleComponent | None = None,
    proactive: Callable[[], ProactiveStatus | None] | None = None,
) -> ConversationEngine:
    """The engine of this process on ``channel`` (see the module description).

    Needs the DeepSeek key: without it the engine could never answer, so the missing key is
    reported here, with the command that sets it, instead of at the first message.  ``style``
    is the style model's runtime (made from the settings when not given); ``commands`` replaces
    the command router (tests); ``schedule`` is the running schedule component (``/时区`` and
    ``/作息`` go through it).
    """
    services.secrets.require(DEEPSEEK_SECRET)
    settings = services.settings
    notifier = notifier or build_emergency_notifier(services)  # R-SAFE-001: the one fixed mail
    llm = runtime or build_llm_runtime(services)
    styled = style or StyleRuntime.from_services(services, llm)
    llm.budget.set_style_status(styled.selector)  # R-LLM-008: the last level hands over to it
    clock = services.clock
    time = time_service_for(services)
    chance = rng or random.Random()  # noqa: S311 - her pacing and word choice, not security
    source = LiveDataSource(services, time_service=time, rng=chance)
    writer = pipeline or ReplyPipeline.from_services(
        services, llm, extra_backends=styled.backends, rng=chance
    )
    state = ConversationStateStore(services.db, clock)
    reader = BotTurnMessages(services.db)
    window = HistoryWindow(settings.engine.history_turns_min, settings.engine.history_turns_max)
    catalog = StickerCatalog(services)

    def short_answers() -> ShortAnswers:
        profile = load_profile(services, "live")
        return ShortAnswers.from_phrases(profile.phrases() if profile is not None else None)

    def queue(turns: Sequence[BotMessage]) -> str | None:
        return queue_bot_extraction(services, turns)

    turn_store = BotTurnStore(services.db, clock)
    engine_stickers = StickerSender(channel, services.media)
    engine = ConversationEngine(
        channel=channel,
        store=turn_store,
        rounds=RoundStore(services.db),
        state=state,
        history=HistoryLoader(reader, window, state),
        reader=reader,
        writer=writer,
        data=source,
        time=time,
        renderer=InboundRenderer(services, client=llm.client),
        crisis=CrisisHandler.from_services(
            services, llm.client, time_service=time, notifier=notifier
        ),
        stickers=engine_stickers,
        lookup=catalog.get,
        short_answers=short_answers,
        runtime=services.runtime,
        alerts=services.alerts,
        clock=clock,
        settings=settings,
        rng=chance,
        commands=commands,
        backends=styled.selector,
        queue_extraction=queue,
    )
    engine.kit = EngineKit(
        llm=llm,
        style=styled,
        data=source,
        pipeline=writer,
        channel=channel,
        stickers=engine_stickers,
        lookup=catalog.get,
        store=turn_store,
        rng=chance,
    )
    recorder = NotLikeRecorder(
        turn_store,
        FeedbackStore(services.db, clock),
        PreferencePairStore(services.db, clock),
        SampleBuilder(services, reader, source),
    )
    if commands is None:
        control = ComponentSchedule(schedule, schedule_kit(services)) if schedule else None
        engine.attach_commands(
            command_port_for(
                services,
                engine,
                llm=llm,
                style=styled,
                channel=channel,
                memory=source.memory,
                data=source,
                schedule=control,
                recorder=recorder,
                proactive=proactive,
            )
        )
    engine.attach_corrections(
        CorrectionService(services, llm.client, recorder, turn_store, RoundStore(services.db))
    )
    return engine


class EngineComponent:
    """Runs the engine and hands it the messages of the channel."""

    name = COMPONENT_NAME

    def __init__(
        self,
        engine: ConversationEngine,
        channel: Channel,
        services: Services,
        *,
        depends_on: Sequence[str] = (),
        on_finished: Callable[[], None] | None = None,
        drain_on_end: bool = False,
        restart_dispatch: bool = True,
        style: StyleRuntime | None = None,
    ) -> None:
        self.engine = engine
        self.channel = channel
        self._style = style
        self.depends_on: Sequence[str] = tuple(depends_on)
        self._on_finished = on_finished
        self._drain_on_end = drain_on_end
        self._restart = restart_dispatch
        self._supervisor = TaskSupervisor(self.name, services.clock, services.alerts)
        self.handled = 0

    @property
    def style(self) -> StyleRuntime | None:
        """The style model's runtime (the health check asks its selector)."""
        return self._style

    @property
    def router(self) -> CommandRouter | None:
        """The command router in use (``None`` when the engine was given another command port).

        Later rounds add their commands with ``component.router.register(spec)``.
        """
        port = self.engine.commands
        return port if isinstance(port, CommandRouter) else None

    async def start(self) -> None:
        await self.engine.start()
        self._supervisor.spawn("dispatch", self._dispatch, restart_on_exit=self._restart)

    async def _dispatch(self) -> None:
        """Feed the channel's messages to the engine, one at a time (R-CH-004, R-ARCH-004)."""
        async for message in self.channel.incoming():
            try:
                await self.engine.handle_message(message)
            except Exception as exc:  # one bad message must not end the conversation
                log.exception("message_failed", error=type(exc).__name__)
            self.handled += 1
        if self._on_finished is not None:  # the channel has no more messages (end of the input)
            if self._drain_on_end:
                await self.engine.drain()
            self._on_finished()

    async def stop(self) -> None:
        await self._supervisor.stop()
        await self.engine.stop()
        if self._style is not None:
            await self._style.aclose()  # the style model's connection, if one was ever opened

    def health(self) -> ComponentHealth:
        return self._supervisor.health()


def register_engine(
    application: Application,
    services: Services,
    channel: Channel,
    *,
    watcher: StateWatcher | None = None,
    after: Sequence[str] = (),
    commands: CommandPort | None = None,
    runtime: LlmRuntime | None = None,
    style: StyleRuntime | None = None,
    pipeline: DraftWriter | None = None,
    rng: random.Random | None = None,
    on_finished: Callable[[], None] | None = None,
    drain_on_end: bool = False,
    restart_dispatch: bool = True,
    schedule: ScheduleComponent | None = None,
    learning: bool = True,
    proactive: Callable[[], ProactiveStatus | None] | None = None,
) -> EngineComponent:
    """Build the engine, add it to ``application`` and subscribe it to what it must hear.

    ``after`` names the components that must be running first (the channel; the schedule, whose
    ``Resumed`` the engine hears).  ``on_finished`` is called when the channel has no more
    messages (the terminal channel at the end of its input).  The style model's monitor joins
    the application, so a model that dies - or comes back - is noticed while nobody talks.
    """
    llm = runtime or build_llm_runtime(services)
    styled = style or StyleRuntime.from_services(services, llm)
    engine = build_engine(
        services,
        channel,
        runtime=llm,
        style=styled,
        pipeline=pipeline,
        commands=commands,
        rng=rng,
        schedule=schedule,
        proactive=proactive,
    )
    schedule_kit(services).events.subscribe(Resumed, engine.on_resumed)
    if watcher is not None:
        watcher.subscribe(engine.on_state_change)
    application.register(styled.monitor(services))
    application.register(ImportReportComponent(services, engine.notify))
    if learning:
        application.register(LearningComponent(services))
    component = EngineComponent(
        engine,
        channel,
        services,
        depends_on=after,
        on_finished=on_finished,
        drain_on_end=drain_on_end,
        restart_dispatch=restart_dispatch,
        style=styled,
    )
    application.register(component)
    return component
