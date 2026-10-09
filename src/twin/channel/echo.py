"""``twin channel echo-test``: say back what the channel received (a diagnostic only).

It checks the channel itself: a message goes in, the same words go back out through the same
guards (bound recipient only, safe window and count, ``[测试]`` prefix).  It is **not** a reply
path: the product never answers with an echo, and ``tests/unit/test_channel_local.py`` keeps
this module out of everything but the diagnostic command.
"""

from __future__ import annotations

from twin.channel.base import TEST_PREFIX, Channel, InboundMessage, MessageKind
from twin.channel.local import TextOutput


class EchoHandler:
    """A message handler that sends the received words back and reports a refusal."""

    def __init__(self, output: TextOutput) -> None:
        self._output = output

    async def __call__(self, message: InboundMessage, channel: Channel) -> None:
        if message.kind is MessageKind.TEXT and message.text:
            body = f"{TEST_PREFIX}回显:{message.text}"
        else:
            body = f"{TEST_PREFIX}回显:收到一条{message.kind.value}消息"
        result = await channel.send_text(body)
        if not result.ok:
            detail = f"{result.kind.value}: {result.reason}"
            if result.code is not None:
                detail += f", code {result.code}"
            self._output.write_line(f"  (the echo was not sent - {detail})")
