"""The channel as an application component (R-ARCH-004, R-CH-002).

``twin run`` starts the WeChat channel next to the job worker and the state watcher: it polls
for the bound user's messages, keeps the window and the login state current and, once the
engine exists (round 09), hands messages over through ``incoming()``.  Nothing is sent without
the engine or the diagnostics asking for it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from twin.app import Application, ComponentHealth
from twin.channel.ilink.channel import IlinkChannel

if TYPE_CHECKING:
    from twin.services import Services

COMPONENT_NAME = "channel"


class ChannelComponent:
    """Start and stop an :class:`IlinkChannel` with the application."""

    name = COMPONENT_NAME
    depends_on: Sequence[str] = ()

    def __init__(self, channel: IlinkChannel) -> None:
        self.channel = channel

    async def start(self) -> None:
        await self.channel.start()

    async def stop(self) -> None:
        await self.channel.stop()

    async def reconnect(self) -> None:
        """Connect again after the machine woke up (the schedule calls this)."""
        await self.channel.reconnect()

    def health(self) -> ComponentHealth:
        return self.channel.health()


def register_channel(application: Application, services: Services) -> ChannelComponent | None:
    """Add the channel to ``application`` when ``channel.kind`` selects the WeChat channel."""
    if services.settings.channel.kind != "ilink":
        return None
    component = ChannelComponent(IlinkChannel.from_services(services))
    application.register(component)
    return component
