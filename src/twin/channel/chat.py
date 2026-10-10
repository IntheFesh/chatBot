"""``twin chat --local``: the whole application with the terminal as the chat (R-CH-011).

``twin chat --local`` runs the application the way ``twin run`` does - state watcher, heartbeat,
job worker and the reply engine - with a :class:`LocalConsoleChannel` in place of WeChat (the
schedule's daily jobs are ``twin run``'s; her state in the day plan is made on demand).  What you
type is a message of the user, and she answers in the terminal at her pace (R-SCOPE-006: there is
no switch that makes her answer at once).  The conversation is the real one: it is stored in
``bot_turns`` and resumed by the next run.

:func:`run_local_chat` also takes a *handler* instead of the engine; that is how the echo diagnostic
(:mod:`twin.channel.echo`, ``twin channel echo-test --local``) checks the channel itself.  A handler
is never a reply path of the product.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Sequence
from typing import Protocol

from twin.app import ComponentHealth, HealthStatus, ShutdownSignals, TaskSupervisor
from twin.channel.base import Channel, InboundMessage
from twin.channel.local import LocalConsoleChannel, TextInput, TextOutput
from twin.clock import Clock
from twin.engine.command_port import CommandPort
from twin.engine.component import register_engine
from twin.engine.machine import DraftWriter
from twin.ops.alerts import AlertSink
from twin.ops.components import build_application
from twin.ops.logging import get_logger
from twin.services import Services

log = get_logger("twin.channel.chat")

COMPONENT_NAME = "local_chat"
CHANNEL_COMPONENT_NAME = "local_channel"


class MessageHandler(Protocol):
    """What happens with each message from the user (a diagnostic; the product uses the engine)."""

    async def __call__(self, message: InboundMessage, channel: Channel) -> None: ...


class LocalChannelComponent:
    """Starts and stops the terminal channel with the application."""

    name = CHANNEL_COMPONENT_NAME
    depends_on: Sequence[str] = ()

    def __init__(self, channel: LocalConsoleChannel) -> None:
        self.channel = channel

    async def start(self) -> None:
        await self.channel.start()

    async def stop(self) -> None:
        await self.channel.stop()

    def health(self) -> ComponentHealth:
        return ComponentHealth(HealthStatus.OK)


class LocalChatComponent:
    """Runs a channel and feeds its messages to a handler until the input ends."""

    name = COMPONENT_NAME
    depends_on: Sequence[str] = ()

    def __init__(
        self,
        channel: Channel,
        handler: MessageHandler,
        clock: Clock,
        alerts: AlertSink | None,
        *,
        on_finished: Callable[[], None],
        limit: int = 0,
    ) -> None:
        self.channel = channel
        self._handler = handler
        self._on_finished = on_finished
        self._limit = limit
        self._supervisor = TaskSupervisor(self.name, clock, alerts)
        self.handled = 0

    async def start(self) -> None:
        await self.channel.start()
        self._supervisor.spawn("dispatch", self._dispatch)

    async def _dispatch(self) -> None:
        try:
            async for message in self.channel.incoming():
                try:
                    await self._handler(message, self.channel)
                except Exception:  # one bad message must not end the conversation (R-ARCH-004)
                    log.exception("handler_failed")
                self.handled += 1
                if self._limit and self.handled >= self._limit:
                    break
        finally:
            self._on_finished()

    async def stop(self) -> None:
        await self._supervisor.stop()
        await self.channel.stop()

    def health(self) -> ComponentHealth:
        return self._supervisor.health()


async def run_local_chat(
    services: Services,
    *,
    input: TextInput,
    output: TextOutput,
    handler: MessageHandler | None = None,
    window_h: float | None = None,
    quota: int | None = None,
    limit: int = 0,
    signals: bool = True,
    pipeline: DraftWriter | None = None,
    commands: CommandPort | None = None,
    rng: random.Random | None = None,
    drain: bool = False,
) -> int:
    """Run the application with the terminal channel; returns how many messages were handled.

    Without a ``handler`` the reply engine answers (``pipeline``, ``commands`` and ``rng`` replace
    its parts, for tests); ``drain`` makes the end of the input wait until she has answered
    everything - the interactive chat leaves at once instead and the next run resumes.
    """
    application, watcher = build_application(services)
    channel = LocalConsoleChannel.from_services(
        services, input=input, output=output, window_h=window_h, quota=quota
    )
    stop = asyncio.Event()
    counted: Callable[[], int]
    if handler is not None:
        chat = LocalChatComponent(
            channel, handler, services.clock, services.alerts, on_finished=stop.set, limit=limit
        )
        application.register(chat)
        counted = lambda: chat.handled  # noqa: E731
    else:
        application.register(LocalChannelComponent(channel))
        engine = register_engine(
            application,
            services,
            channel,
            watcher=watcher,
            after=(CHANNEL_COMPONENT_NAME,),
            commands=commands,
            pipeline=pipeline,
            rng=rng,
            on_finished=stop.set,
            drain_on_end=drain,
            restart_dispatch=False,
            learning=False,
        )
        counted = lambda: engine.handled  # noqa: E731
    output.write_line("type /help for the commands; /quit leaves")
    shutdown = ShutdownSignals(asyncio.get_running_loop(), stop) if signals else None
    await application.run(stop, signals=shutdown)
    return counted()
