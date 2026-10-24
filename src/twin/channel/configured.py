"""What the configured channel can do, without starting it (R-EVAL-002, round 18, D-851).

The style metrics of the bot's real conversation (``twin eval style --source live``) judge a
metric only where the channel the conversation went through can produce it: the quote rate needs
a channel that can quote.  The report runs in a command of its own, not inside the application,
so it cannot ask the running channel; it asks the class that ``channel.kind`` selects, with the
values the application would build it with (the M0 probe's measurements, the safe thresholds).
The classes keep the one definition (``capabilities_for``); their ``capabilities()`` use it too.
"""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

from twin.channel.base import ChannelCapabilities
from twin.channel.ilink.channel import IlinkChannel
from twin.channel.local import LocalConsoleChannel
from twin.channel.probe.summary import load_channel_probe_summary, measured_capabilities

if TYPE_CHECKING:
    from twin.services import Services


def configured_capabilities(services: Services) -> ChannelCapabilities:
    """The capabilities the channel of ``channel.kind`` reports when the application runs."""
    config = services.settings.channel
    if config.kind == "ilink":
        with services.db.session() as session:
            measured = measured_capabilities(load_channel_probe_summary(session))
        return IlinkChannel.capabilities_for(measured)
    return LocalConsoleChannel.capabilities_for(
        config.proactive_window_safe_h, config.outbound_quota_safe
    )


def abilities_of(capabilities: ChannelCapabilities) -> dict[str, bool]:
    """The yes-or-no abilities of a channel (``supports_*``), by name - what a record keeps."""
    return {
        field.name: bool(getattr(capabilities, field.name))
        for field in fields(capabilities)
        if field.name.startswith("supports_")
    }
