"""The engine's view of the service for corrections in plain words (R-LRN-002).

A correction in plain words ("她不会这么说", "你说话不像她") is not a command, so the command port
never sees it.  The engine asks this port at two places:

``propose(answered)``
    after a reply to the messages ``answered`` is out, the service looks at what the user wrote
    and, when it was a correction of the reply before, returns the system message that asks
    whether to record it ("回复‘是’确认"), or ``None``.  The persona has already answered
    normally by then - the question comes after, never instead;
``is_confirmation(text, at)`` / ``confirm(context)``
    a message of the user that is the "yes" to a question that is still open (within the window,
    and no further reply of the persona since) is taken by the service: the engine marks it
    ``is_command`` like a command message, so it never reaches the conversation, the memory or the
    learning, and answers with the outcome the service returns.

The service itself is :class:`twin.learning.corrections.CorrectionService`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from twin.engine.command_port import CommandContext, CommandOutcome


class CorrectionPort(Protocol):
    """What the engine asks of the correction service (see the module description)."""

    async def is_confirmation(self, text: str, at: datetime) -> bool:
        """Is ``text`` the "yes" of a question that is open at ``at``?"""
        ...

    async def confirm(self, context: CommandContext) -> CommandOutcome | None:
        """Record the correction the open question was about; ``None`` if it is no longer open."""
        ...

    async def propose(self, answered: Sequence[str]) -> str | None:
        """The question to ask after a reply to the rows ``answered`` (``None``: nothing to ask)."""
        ...
