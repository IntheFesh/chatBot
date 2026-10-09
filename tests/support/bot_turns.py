"""A conversation kept in memory, for tests that do not need the ``bot_turns`` table (round 07).

``ListBotTurnReader`` implements :class:`twin.memory.recent.BotTurnReader` over a list.  The real
reader (``twin.engine.turns.BotTurnMessages``, round 09) reads the table; tests of the memory and
of the schedule register this one with ``register_bot_turn_reader`` when they want a conversation
of their own.  The memory modules never import this file.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from twin.memory.recent import BotMessage, Role

DEFAULT_START = datetime.fromisoformat("2026-03-05T15:00:00+00:00")


class ListBotTurnReader:
    """Read access to a list of messages (sorted by time on every call)."""

    def __init__(self, messages: Sequence[BotMessage] = ()) -> None:
        self.messages = list(messages)

    def add(self, role: Role, text: str, at: datetime) -> BotMessage:
        message = BotMessage(f"b{len(self.messages) + 1}", role, text, at)
        self.messages.append(message)
        return message

    def _sorted(self) -> list[BotMessage]:
        return sorted(self.messages, key=lambda m: (m.at, m.id))

    def messages_between(self, start: datetime, end: datetime) -> Sequence[BotMessage]:
        return [m for m in self._sorted() if start <= m.at < end]

    def messages_since(
        self, since: datetime | None, limit: int | None = None
    ) -> Sequence[BotMessage]:
        found = self._sorted()
        if since is not None:
            return [m for m in found if m.at >= since]
        return found[-limit:] if limit else found


def conversation(
    turns: int, *, start: datetime = DEFAULT_START, per_turn: int = 1
) -> ListBotTurnReader:
    """``turns`` alternating turns (the user first), ``per_turn`` messages each, a minute apart."""
    reader = ListBotTurnReader()
    moment = start
    for number in range(turns):
        role: Role = "user" if number % 2 == 0 else "bot"
        for part in range(per_turn):
            reader.add(role, f"第{number + 1}轮第{part + 1}句", moment)
            moment += timedelta(seconds=20)
        moment += timedelta(minutes=1)
    return reader
