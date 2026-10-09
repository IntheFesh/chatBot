"""``twin chat --local``: the application skeleton with only the terminal channel (R-CH-011).

The persona engine is connected in round 09.  Until then the chat command starts the real
application (state watcher, heartbeat, job worker) with a :class:`LocalConsoleChannel` and a
:class:`MessageHandler` that tells the user, truthfully, that nobody is answering yet and what
the channel received.  When the engine exists it is passed as the handler; nothing else in this
module changes.  The echo diagnostic (:mod:`twin.channel.echo`) is another handler and is never
a reply path of the product.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Protocol

from twin.app import ComponentHealth, ShutdownSignals, TaskSupervisor
from twin.channel.base import Channel, InboundMessage, MessageKind
from twin.channel.local import LocalConsoleChannel, TextInput, TextOutput
from twin.clock import Clock
from twin.ops.alerts import AlertSink
from twin.ops.components import build_application
from twin.ops.logging import get_logger
from twin.services import Services

log = get_logger("twin.channel.chat")

COMPONENT_NAME = "local_chat"
ENGINE_NOT_CONNECTED = (
    "The persona engine is not connected yet (it arrives in round 09): the channel receives "
    "what you type, but nobody answers. `twin channel echo-test --local` checks the channel."
)


class MessageHandler(Protocol):
    """What happens with each message from the user (the engine, from round 09)."""

    async def __call__(self, message: InboundMessage, channel: Channel) -> None: ...


class EngineNotConnected:
    """The handler until the engine exists: it says what arrived and that no one replies."""

    def __init__(self, output: TextOutput) -> None:
        self._output = output

    async def __call__(self, message: InboundMessage, channel: Channel) -> None:
        if message.kind is MessageKind.IMAGE and message.media_ref is not None:
            what = f"a picture ({message.media_ref.size} bytes) stored encrypted"
        else:
            what = f"{len(message.text or '')} character(s) of text"
        self._output.write_line(f"  (received {what}; the engine is not connected, no reply)")


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
) -> int:
    """Run the application with the terminal channel; returns how many messages were handled."""
    application, _watcher = build_application(services)
    channel = LocalConsoleChannel.from_services(
        services, input=input, output=output, window_h=window_h, quota=quota
    )
    stop = asyncio.Event()
    component = LocalChatComponent(
        channel,
        handler or EngineNotConnected(output),
        services.clock,
        services.alerts,
        on_finished=stop.set,
        limit=limit,
    )
    application.register(component)
    if handler is None:
        output.write_line(ENGINE_NOT_CONNECTED)
    output.write_line("type /help for the commands; /quit leaves")
    shutdown = ShutdownSignals(asyncio.get_running_loop(), stop) if signals else None
    await application.run(stop, signals=shutdown)
    return component.handled
