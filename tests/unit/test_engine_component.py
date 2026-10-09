"""The engine in the application: wiring, the component, what it listens to (round 09 section J)."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator

import pytest

from tests.support.clock import ManualClock
from tests.support.engine_extras import ScriptedCommands
from tests.support.engine_harness import (
    ScriptedChannel,
    ScriptedWriter,
    make_draft,
    run_to_idle,
)
from tests.support.style_models import ScriptedStyleClient
from tests.support.waiting import wait_until
from twin.app import Application, HealthStatus
from twin.commands.router import CommandRouter
from twin.config.runtime import SHOW_THINKING
from twin.config.secrets import SecretStoreError
from twin.engine.backend_select import BackendMonitorComponent, BackendSelector
from twin.engine.command_port import CommandOutcome
from twin.engine.component import EngineComponent, build_engine, register_engine
from twin.engine.style_runtime import StyleRuntime
from twin.llm.runtime import DEEPSEEK_SECRET, build_llm_runtime
from twin.ops.state_watch import StateWatcher
from twin.schedule.events import Resumed
from twin.schedule.service import schedule_kit
from twin.services import Services


@pytest.fixture
def keyed(services: Services) -> Services:
    services.secrets.set(DEEPSEEK_SECRET, "synthetic-test-key-0001")
    return services


async def test_the_engine_cannot_be_built_without_the_deepseek_key(
    services: Services, clock: ManualClock
) -> None:
    with pytest.raises(SecretStoreError, match="twin secrets set deepseek_api_key"):
        build_engine(services, ScriptedChannel(clock))


async def test_the_engine_is_built_from_the_services_of_the_process(
    keyed: Services, clock: ManualClock
) -> None:
    channel = ScriptedChannel(clock)
    engine = build_engine(
        keyed, channel, pipeline=ScriptedWriter(make_draft("好")), rng=random.Random(1)
    )
    window = await engine.quiet_window()
    assert (window.configured_s, window.adaptive) == (15.0, False)
    assert window.suggested_s == 33.0  # the SPEC reference number: no profile has been built yet
    assert engine.snapshot().state == "IDLE"


Running = tuple[Application, EngineComponent, ScriptedChannel, ScriptedWriter]


@pytest.fixture
async def running(keyed: Services, clock: ManualClock) -> AsyncIterator[Running]:
    channel = ScriptedChannel(clock)
    writer = ScriptedWriter()
    application = Application()
    component = register_engine(application, keyed, channel, pipeline=writer, rng=random.Random(2))
    await component.start()
    yield application, component, channel, writer
    await component.stop()


async def test_the_component_hands_the_channels_messages_to_the_engine(
    running: Running, clock: ManualClock
) -> None:
    _application, component, channel, writer = running
    writer.add(make_draft("好呀"))
    channel.push("在吗")
    await wait_until(lambda: component.handled == 1)
    await run_to_idle(component.engine, clock)
    assert channel.texts == ["好呀"] and writer.contexts[0].user_text == "在吗"
    assert component.health().status is HealthStatus.OK
    assert component.name == "engine" and component.depends_on == ()


async def test_one_message_that_breaks_does_not_stop_the_component(
    running: Running,
    clock: ManualClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _application, component, channel, writer = running
    original = component.engine.handle_message
    calls: list[str] = []

    async def flaky(message):  # type: ignore[no-untyped-def]
        calls.append(message.text)
        if message.text == "boom":
            raise RuntimeError("the message broke")
        await original(message)

    monkeypatch.setattr(component.engine, "handle_message", flaky)
    channel.push("boom")
    channel.push("在吗")
    await wait_until(lambda: component.handled == 2)
    await run_to_idle(component.engine, clock)
    assert calls == ["boom", "在吗"] and channel.texts == ["好的"]
    assert writer.calls == 1


async def test_the_end_of_the_input_finishes_after_the_reply_when_asked_to_drain(
    keyed: Services, clock: ManualClock
) -> None:
    channel = ScriptedChannel(clock)
    finished: list[bool] = []
    application = Application()
    component = register_engine(
        application,
        keyed,
        channel,
        pipeline=ScriptedWriter(make_draft("好呀")),
        on_finished=lambda: finished.append(True),
        drain_on_end=True,
        restart_dispatch=False,
    )
    await component.start()
    try:
        channel.push("在吗")
        channel.end_input()
        await wait_until(lambda: component.handled == 1)
        assert finished == []  # she has not answered yet
        await run_to_idle(component.engine, clock)
        await wait_until(lambda: finished == [True])
        assert channel.texts == ["好呀"]
    finally:
        await component.stop()


async def test_without_draining_the_end_of_the_input_finishes_at_once(
    keyed: Services, clock: ManualClock
) -> None:
    channel = ScriptedChannel(clock)
    finished: list[bool] = []
    application = Application()
    component = register_engine(
        application,
        keyed,
        channel,
        pipeline=ScriptedWriter(make_draft("好呀")),
        on_finished=lambda: finished.append(True),
        restart_dispatch=False,
    )
    await component.start()
    try:
        channel.push("在吗")
        channel.end_input()
        await wait_until(lambda: finished == [True])
        assert (
            component.engine.snapshot().pending != () or component.engine.snapshot().state != "IDLE"
        )
        # the unfinished round is on disk: the next run resumes it
    finally:
        await component.stop()


async def test_the_engine_listens_to_the_schedule_and_to_the_state_watcher(
    keyed: Services, clock: ManualClock
) -> None:
    channel = ScriptedChannel(clock)
    watcher = StateWatcher(keyed.db, clock)
    application = Application()
    component = register_engine(
        application,
        keyed,
        channel,
        pipeline=ScriptedWriter(),
        watcher=watcher,
        after=("channel", "schedule"),
    )
    engine = component.engine
    assert component.depends_on == ("channel", "schedule")
    await schedule_kit(keyed).events.publish(Resumed(clock.now_utc(), "wake", 12.0, None))
    assert engine._resume_pending is True  # type: ignore[attr-defined]
    engine._pacing = object()  # type: ignore[assignment,attr-defined]
    await watcher.poll_once()
    keyed.runtime.set(SHOW_THINKING, True)
    await watcher.poll_once()
    assert engine._pacing is None  # type: ignore[attr-defined]  # a changed setting: read again


async def test_the_real_command_router_is_attached_and_can_be_replaced_after_the_build(
    keyed: Services, clock: ManualClock
) -> None:
    channel = ScriptedChannel(clock)
    application = Application()
    component = register_engine(application, keyed, channel, pipeline=ScriptedWriter())
    assert isinstance(component.router, CommandRouter)  # attached by default (round 09-4)
    assert component.engine.commands is component.router
    component.engine.attach_commands(ScriptedCommands(帮助=CommandOutcome("⚙️ 帮助")))
    assert component.router is None  # another port is in use now
    await component.start()
    try:
        channel.push("/帮助")
        await wait_until(lambda: channel.texts == ["⚙️ 帮助"])
    finally:
        await component.stop()


async def test_a_router_given_at_registration_is_used_as_it_is(
    keyed: Services, clock: ManualClock
) -> None:
    channel = ScriptedChannel(clock)
    application = Application()
    router = ScriptedCommands(状态=CommandOutcome("⚙️ 正常"))
    component = register_engine(
        application, keyed, channel, pipeline=ScriptedWriter(), commands=router
    )
    await component.start()
    try:
        channel.push("/状态")
        await wait_until(lambda: channel.texts == ["⚙️ 正常"])
        assert asyncio.get_running_loop() is not None
    finally:
        await component.stop()


async def test_the_application_gets_the_style_models_monitor_and_the_budget_hears_the_selector(
    keyed: Services, clock: ManualClock
) -> None:
    channel = ScriptedChannel(clock)
    application = Application()
    llm = build_llm_runtime(keyed)
    style = StyleRuntime.from_services(keyed, llm, client=ScriptedStyleClient())
    try:
        component = register_engine(
            application, keyed, channel, runtime=llm, style=style, pipeline=ScriptedWriter()
        )
        assert isinstance(application.components["backend_monitor"], BackendMonitorComponent)
        assert llm.budget._style is style.selector  # type: ignore[attr-defined]
        assert component.engine._backends is style.selector  # type: ignore[attr-defined]
    finally:
        await llm.client.aclose()


async def test_an_engine_built_without_a_style_runtime_makes_one_from_the_settings(
    keyed: Services, clock: ManualClock
) -> None:
    engine = build_engine(keyed, ScriptedChannel(clock), pipeline=ScriptedWriter())
    selector = engine._backends  # type: ignore[attr-defined]
    assert isinstance(selector, BackendSelector)
    assert (await selector.choose()).name == "deepseek"  # nothing is registered: no style model
    assert isinstance(engine.commands, CommandRouter)
