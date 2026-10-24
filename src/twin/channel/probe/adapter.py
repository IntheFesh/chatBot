"""The WeChat channel seen by the probe: :class:`IlinkProbeChannel`.

The runner talks to a small :class:`~twin.channel.probe.runner.ProbeChannel` interface; this
adapter provides it on top of a running :class:`~twin.channel.ilink.channel.IlinkChannel`.  It
adds nothing the channel does not already do: sends go through the channel's own ``send_*``
methods (so the recipient guard, the media allow list and the login checks all apply), and the
"did the user write?" question is answered from the window state the long poller keeps.
"""

from __future__ import annotations

import hashlib

from twin.channel.base import OutboundResult, SendBypass
from twin.channel.ilink.channel import IlinkChannel
from twin.channel.policy import ProbeImageManifest
from twin.channel.probe.runner import InboundMarker


class IlinkProbeChannel:
    """Gives the probe runner access to the WeChat channel."""

    def __init__(self, channel: IlinkChannel) -> None:
        self._channel = channel
        self._manifest = ProbeImageManifest(channel.state)

    def inbound_marker(self) -> InboundMarker:
        store = self._channel.store
        window = store.window_state()
        context = store.context_token()
        fingerprint = (
            hashlib.sha256(context.token.encode("utf-8")).hexdigest()[:8] if context else None
        )
        return InboundMarker(
            last_inbound_at=window.last_inbound_at,
            context_fingerprint=fingerprint,
            auth=store.auth_record().state,
            bound=store.bound_user() is not None,
        )

    def register_image(self, data: bytes) -> str:
        return self._manifest.register_bytes(data)

    async def send_text(self, text: str, *, bypass: SendBypass) -> OutboundResult:
        return await self._channel.send_text(text, bypass=bypass)

    async def send_image(self, data: bytes, mime: str, *, bypass: SendBypass) -> OutboundResult:
        return await self._channel.send_image(data, mime, bypass=bypass)

    async def send_typing(self, active: bool) -> None:
        await self._channel.send_typing(active)
