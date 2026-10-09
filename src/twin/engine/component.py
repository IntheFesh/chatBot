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

The command router of round 09-2 (:class:`~twin.engine.command_port.CommandPort`) is attached in
:func:`command_port_for` - the one place that names it.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from twin.app import Application, ComponentHealth, TaskSupervisor
from twin.channel.base import Channel
from twin.engine.command_port import CommandPort
from twin.engine.dataview import LiveDataSource
from twin.engine.fallback import ShortAnswers
from twin.engine.history import HistoryLoader
from twin.engine.inbound import InboundRenderer
from twin.engine.machine import ConversationEngine, DraftWriter
from twin.engine.pipeline import ReplyPipeline
from twin.engine.rounds import RoundStore
from twin.engine.safety.crisis import CrisisHandler
from twin.engine.safety.notifier import EmergencyNotifier
from twin.engine.state_store import ConversationStateStore
from twin.engine.sticker_sender import StickerSender
from twin.engine.turns import BotTurnMessages, BotTurnStore
from twin.llm.runtime import DEEPSEEK_SECRET, LlmRuntime, build_llm_runtime
from twin.memory.jobs import queue_bot_extraction
from twin.memory.recent import BotMessage, HistoryWindow
from twin.ops.logging import get_logger
from twin.ops.state_watch import StateWatcher
from twin.profile.api import load_profile
from twin.schedule.events import Resumed
from twin.schedule.service import schedule_kit, time_service_for
from twin.stickers.catalog import StickerCatalog

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.engine.component")

COMPONENT_NAME = "engine"


def command_port_for(services: Services, engine: ConversationEngine) -> CommandPort | None:
    """The command router of the application, or ``None`` while no router is attached.

    Round 09-2's ``twin.commands`` provides the router; it is named here and nowhere else.
    Without one, a message that starts with ``/`` is an ordinary message of the conversation.
    """
    return None


def build_engine(
    services: Services,
    channel: Channel,
    *,
    runtime: LlmRuntime | None = None,
    pipeline: DraftWriter | None = None,
    commands: CommandPort | None = None,
    notifier: EmergencyNotifier | None = None,
    rng: random.Random | None = None,
) -> ConversationEngine:
    """The engine of this process on ``channel`` (see the module description).

    Needs the DeepSeek key: without it the engine could never answer, so the missing key is
    reported here, with the command that sets it, instead of at the first message.
    """
    services.secrets.require(DEEPSEEK_SECRET)
    settings = services.settings
    llm = runtime or build_llm_runtime(services)
    clock = services.clock
    time = time_service_for(services)
    chance = rng or random.Random()  # noqa: S311 - her pacing and word choice, not security
    source = LiveDataSource(services, time_service=time, rng=chance)
    writer = pipeline or ReplyPipeline.from_services(services, llm, rng=chance)
    state = ConversationStateStore(services.db, clock)
    reader = BotTurnMessages(services.db)
    window = HistoryWindow(settings.engine.history_turns_min, settings.engine.history_turns_max)
    catalog = StickerCatalog(services)

    def short_answers() -> ShortAnswers:
        profile = load_profile(services, "live")
        return ShortAnswers.from_phrases(profile.phrases() if profile is not None else None)

    def queue(turns: Sequence[BotMessage]) -> str | None:
        return queue_bot_extraction(services, turns)

    return ConversationEngine(
        channel=channel,
        store=BotTurnStore(services.db, clock),
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
        stickers=StickerSender(channel, services.media),
        lookup=catalog.get,
        short_answers=short_answers,
        runtime=services.runtime,
        alerts=services.alerts,
        clock=clock,
        settings=settings,
        rng=chance,
        commands=commands,
        queue_extraction=queue,
    )


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
    ) -> None:
        self.engine = engine
        self.channel = channel
        self.depends_on: Sequence[str] = tuple(depends_on)
        self._on_finished = on_finished
        self._drain_on_end = drain_on_end
        self._restart = restart_dispatch
        self._supervisor = TaskSupervisor(self.name, services.clock, services.alerts)
        self.handled = 0

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
    pipeline: DraftWriter | None = None,
    rng: random.Random | None = None,
    on_finished: Callable[[], None] | None = None,
    drain_on_end: bool = False,
    restart_dispatch: bool = True,
) -> EngineComponent:
    """Build the engine, add it to ``application`` and subscribe it to what it must hear.

    ``after`` names the components that must be running first (the channel; the schedule, whose
    ``Resumed`` the engine hears).  ``on_finished`` is called when the channel has no more
    messages (the terminal channel at the end of its input).
    """
    engine = build_engine(
        services, channel, runtime=runtime, pipeline=pipeline, commands=commands, rng=rng
    )
    if commands is None:
        port = command_port_for(services, engine)
        if port is not None:
            engine.attach_commands(port)
    schedule_kit(services).events.subscribe(Resumed, engine.on_resumed)
    if watcher is not None:
        watcher.subscribe(engine.on_state_change)
    component = EngineComponent(
        engine,
        channel,
        services,
        depends_on=after,
        on_finished=on_finished,
        drain_on_end=drain_on_end,
        restart_dispatch=restart_dispatch,
    )
    application.register(component)
    return component
