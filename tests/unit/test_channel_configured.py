"""What the configured channel can do, read without starting it (R-EVAL-002, R18-E2, D-851)."""

from __future__ import annotations

from tests.support.console import RecordingOutput, ScriptedInput
from twin.channel.configured import abilities_of, configured_capabilities
from twin.channel.ilink.channel import IlinkChannel
from twin.channel.local import LocalConsoleChannel
from twin.eval.channel import InMemoryChannel
from twin.services import Services


async def test_r18_e2_the_ilink_channel_is_reported_as_a_running_one_reports_itself(
    services: Services,
) -> None:
    """The style report of the live conversation asks this instead of the running channel."""
    assert services.settings.channel.kind == "ilink"
    running = IlinkChannel.from_services(services, poll=False).capabilities()
    assert configured_capabilities(services) == running
    assert abilities_of(running) == {"supports_quote": False, "supports_typing": True}


async def test_r18_e2_the_console_channel_is_reported_with_its_simulated_limits(
    services: Services,
) -> None:
    services.settings.channel.kind = "console"
    console = LocalConsoleChannel.from_services(
        services, input=ScriptedInput(), output=RecordingOutput()
    )
    assert configured_capabilities(services) == console.capabilities()
    assert abilities_of(configured_capabilities(services))["supports_quote"] is False


def test_r18_e2_the_abilities_are_the_yes_or_no_capabilities_of_any_channel(
    services: Services,
) -> None:
    """The record a generated pair keeps: only the ``supports_*`` flags, by name."""
    quoting = InMemoryChannel(clock=services.clock, supports_quote=True)
    assert abilities_of(quoting.capabilities())["supports_quote"] is True
    sandbox = InMemoryChannel(clock=services.clock)
    assert set(abilities_of(sandbox.capabilities())) == {"supports_quote", "supports_typing"}
    assert abilities_of(sandbox.capabilities())["supports_quote"] is False
