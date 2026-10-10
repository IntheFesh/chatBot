"""The one place where ``twin run`` puts its components together (R-ARCH-001, R-ARCH-004).

:func:`assemble` builds the :class:`~twin.app.Application` of a process from a services container:
the state watcher, the heartbeat and the job worker, the channel (the WeChat conversation, or the
terminal when ``channel.kind`` is ``console``), the schedule, the style model's server, the reply
engine, the proactive scheduler and the operations components.  It does not start anything and it
does not install signal handlers or the sleep guard - that is :func:`twin.cli._serve`, which runs
the result.  The end-to-end tests and the long-run script (``scripts/soak.py``) call the same
function with their own terminal streams, so what they exercise is what ``twin run`` runs.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

from twin.app import Application
from twin.channel.base import Channel
from twin.channel.chat import CHANNEL_COMPONENT_NAME, LocalChannelComponent
from twin.channel.component import ChannelComponent, register_channel
from twin.channel.local import LocalConsoleChannel, TextInput, TextOutput
from twin.channel.probe.component import register_probe
from twin.engine.component import EngineComponent, register_engine
from twin.engine.style_runtime import StyleRuntime
from twin.llm.runtime import LlmRuntime, build_llm_runtime
from twin.ops.components import build_application
from twin.ops.power_events import PowerEventMonitor
from twin.ops.state_watch import StateWatcher
from twin.ops.wiring import OpsKit, register_ops
from twin.schedule.component import ScheduleComponent, register_schedule
from twin.schedule.proactive.component import (
    ProactiveComponent,
    proactive_status_for,
    register_proactive,
)
from twin.services import Services
from twin.serving.component import StyleServingComponent, register_serving


class ConsoleStreamsMissing(ValueError):
    """``channel.kind`` is ``console`` but no terminal streams were given."""


@dataclass
class Assembly:
    """The application of a process and the parts of it that callers reach for."""

    application: Application
    stop: asyncio.Event
    watcher: StateWatcher
    llm: LlmRuntime
    style: StyleRuntime
    channel: Channel
    schedule: ScheduleComponent
    power_monitor: PowerEventMonitor
    serving: StyleServingComponent
    engine: EngineComponent
    proactive: ProactiveComponent
    ops: OpsKit
    channel_component: ChannelComponent | None = None


def assemble(
    services: Services,
    *,
    console_input: TextInput | None = None,
    console_output: TextOutput | None = None,
    rng: random.Random | None = None,
) -> Assembly:
    """Build the application of ``twin run`` (see the module description).

    ``console_input`` / ``console_output`` are the terminal's streams, needed only when
    ``channel.kind`` is ``console``.  With that kind the end of the input stops the application
    (the stop event is set), as it does for ``twin chat``.  ``rng`` is the generator of her pacing
    and word choices and of the proactive scheduler's draws; ``twin run`` passes none (an unseeded
    one is made), a test that needs the same day twice passes a seeded one.
    """
    application, watcher = build_application(services)
    stop = asyncio.Event()
    channel_component = register_channel(application, services)
    register_probe(application, services)
    schedule_component, power_monitor = register_schedule(
        application,
        services,
        watcher,
        reconnect=channel_component.reconnect if channel_component is not None else None,
    )
    engine_component: EngineComponent
    # the style model: its server is up (or loading) before the engine asks it anything
    llm = build_llm_runtime(services)
    style = StyleRuntime.from_services(services, llm)
    serving = register_serving(application, services, style=style, say=None, watcher=watcher)
    channel: Channel
    if channel_component is not None:  # channel.kind "ilink": the user's WeChat conversation
        channel = channel_component.channel
        engine_component = register_engine(
            application,
            services,
            channel,
            watcher=watcher,
            after=(channel_component.name, "schedule", serving.name),
            schedule=schedule_component,
            proactive=proactive_status_for(services, channel.session_state),
            runtime=llm,
            style=style,
            rng=rng,
        )
    else:  # channel.kind "console": the terminal in place of WeChat
        if console_input is None or console_output is None:
            raise ConsoleStreamsMissing("the console channel needs an input and an output stream")
        console = LocalConsoleChannel.from_services(
            services, input=console_input, output=console_output
        )
        channel = console
        application.register(LocalChannelComponent(console))
        engine_component = register_engine(
            application,
            services,
            console,
            watcher=watcher,
            after=(CHANNEL_COMPONENT_NAME, "schedule", serving.name),
            on_finished=stop.set,
            restart_dispatch=False,
            schedule=schedule_component,
            proactive=proactive_status_for(services, console.session_state),
            runtime=llm,
            style=style,
            rng=rng,
        )
    # the scheduler draws from a generator of its own, so that what the engine draws (which depends
    # on how many messages the user writes) cannot change when a proactive message is sent
    proactive_rng = random.Random(rng.getrandbits(64)) if rng is not None else None  # noqa: S311
    proactive = register_proactive(application, services, engine_component, rng=proactive_rng)
    serving.set_say(engine_component.engine.notify)
    ops = register_ops(
        application,
        services,
        style=style.selector,
        schedule=schedule_component,
        probes=(("style_serving", serving.health_check),),
    )
    return Assembly(
        application=application,
        stop=stop,
        watcher=watcher,
        llm=llm,
        style=style,
        channel=channel,
        schedule=schedule_component,
        power_monitor=power_monitor,
        serving=serving,
        engine=engine_component,
        proactive=proactive,
        ops=ops,
        channel_component=channel_component,
    )
