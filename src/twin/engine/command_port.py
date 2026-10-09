"""The engine's view of the command handler (R-CMD-001); the router itself lives in twin.commands."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class CommandContext:
    """What a command may need to know about the message that carried it."""

    at: datetime  # UTC, timezone-aware: when the message arrived
    inbound_id: str  # id of the inbound row in bot_turns


@dataclass(frozen=True)
class CommandOutcome:
    """The result of one command message."""

    reply: str  # system-voice text, already starting with the "⚙️ " prefix
    redo: bool = False  # /重来: send ``reply``, then regenerate the previous reply


class CommandPort(Protocol):
    """``None`` means the message is not a command and goes through the normal chat path."""

    async def handle(self, text: str, context: CommandContext) -> CommandOutcome | None: ...
