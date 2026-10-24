"""``ChannelProbe``: the application component that carries the probe plan out (R-CH-009).

``twin channel probe start`` only writes the plan; the running application (``twin run``)
executes it, because the probe has to keep going for more than a day, survive restarts and
use the live channel (its long poll is what notices the user's messages).  With no plan the
component idles; with a plan it asks for a fresh message, waits, sends, asks, and continues
after any restart from the stored state.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from twin.app import Application, ComponentHealth, TaskSupervisor
from twin.channel.component import COMPONENT_NAME as CHANNEL_COMPONENT
from twin.channel.component import ChannelComponent
from twin.channel.console import AlertBanner, StderrBanner
from twin.channel.ilink.channel import IlinkChannel
from twin.channel.probe.adapter import IlinkProbeChannel
from twin.channel.probe.policy import ProbeSendPolicy
from twin.channel.probe.runner import ProbeRunner
from twin.channel.probe.store import ProbeStore
from twin.clock import Clock
from twin.ops.alerts import AlertSink

if TYPE_CHECKING:
    from twin.services import Services

COMPONENT_NAME = "channel_probe"


def build_runner(
    services: Services, channel: IlinkChannel, *, banner: AlertBanner | None = None
) -> ProbeRunner:
    """A runner on the live channel, with the probe's own send policy."""
    store = ProbeStore(services.db, services.clock)
    return ProbeRunner(
        store=store,
        channel=IlinkProbeChannel(channel),
        policy=ProbeSendPolicy(store),
        clock=services.clock,
        alerts=services.alerts,
        banner=banner or StderrBanner(),
    )


class ChannelProbe:
    """Runs :class:`ProbeRunner` under supervision for the lifetime of the application."""

    name = COMPONENT_NAME
    depends_on: Sequence[str] = (CHANNEL_COMPONENT,)

    def __init__(self, runner: ProbeRunner, clock: Clock, alerts: AlertSink | None) -> None:
        self.runner = runner
        self._supervisor = TaskSupervisor(self.name, clock, alerts)

    async def start(self) -> None:
        self._supervisor.spawn("run", self.runner.run_forever, restart_on_exit=True)

    async def stop(self) -> None:
        await self._supervisor.stop()

    def health(self) -> ComponentHealth:
        return self._supervisor.health()


def register_probe(application: Application, services: Services) -> ChannelProbe | None:
    """Add the probe to ``application`` when the WeChat channel component is there."""
    channel_component = application.components.get(CHANNEL_COMPONENT)
    if not isinstance(channel_component, ChannelComponent):
        return None
    component = ChannelProbe(
        build_runner(services, channel_component.channel), services.clock, services.alerts
    )
    application.register(component)
    return component
