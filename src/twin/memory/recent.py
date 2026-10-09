"""The bot's recent conversation: merged turns in a batched window (R-MEM-001, R-LLM-010).

The last 30 to 40 turns of the bot's conversation go into the prompt as they were said, so the
model sees the thread.  A **turn** is a merged block - consecutive messages of one side - and the
window the prompt shows starts at a fixed turn and only moves when the conversation outgrows it:
when more than ``engine.history_turns_max`` (40) turns are in the window it moves forward in one
step of ``history_turns_max - history_turns_min`` (10) turns, back to about 30.  Between two moves
every request starts with the same turns as the one before, so the provider's prompt cache keeps
hitting (R-LLM-010).

This round defines the *shapes*: what a row of the bot's conversation looks like to the memory
(:class:`BotMessage`), how it is read (:class:`BotTurnReader`, which round 09 implements on the
``bot_turns`` table - that table is created by round 09), how rows become turns
(:class:`RecentTurns`) and how the window start moves (:class:`HistoryWindow`).  Round 09
stores the window start (the time of its first turn) in ``conversation_state`` and calls
:meth:`RecentTurns.window` for every reply.  Nothing here writes.

A reader is registered with :func:`register_bot_turn_reader` (round 09 does this when its tables
exist); the memory jobs that need the bot's conversation - the bot's daily summary - ask
:func:`bot_turn_reader` and do nothing while there is none.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from twin.services import Services

Role = Literal["user", "bot"]
NEWEST_MESSAGES_PER_TURN = 20  # messages read to find the newest turns of a new conversation


@dataclass(frozen=True)
class BotMessage:
    """One message of the bot's conversation: what the user wrote or what the bot sent."""

    id: str
    role: Role
    text: str
    at: datetime


class BotTurnReader(Protocol):
    """Read access to the bot's conversation (implemented on ``bot_turns`` by round 09)."""

    def messages_between(self, start: datetime, end: datetime) -> Sequence[BotMessage]:
        """Messages with ``start <= at < end``, oldest first."""
        ...

    def messages_since(
        self, since: datetime | None, limit: int | None = None
    ) -> Sequence[BotMessage]:
        """Messages at or after ``since``, oldest first.

        With ``since`` ``None`` the newest ``limit`` messages (all of them without a limit).
        """
        ...


@dataclass(frozen=True)
class Turn:
    """A merged block: consecutive messages of one side."""

    role: Role
    text: str  # the messages joined by a newline
    at: datetime  # time of the first message
    last_at: datetime  # time of the last message
    message_ids: tuple[str, ...]

    @property
    def id(self) -> str:
        """The id of the first message: a turn is named by where it begins."""
        return self.message_ids[0]


def merge_turns(messages: Sequence[BotMessage]) -> list[Turn]:
    """Merge consecutive messages of one side into turns (input oldest first)."""
    turns: list[Turn] = []
    block: list[BotMessage] = []

    def close() -> None:
        if block:
            turns.append(
                Turn(
                    block[0].role,
                    "\n".join(m.text for m in block),
                    block[0].at,
                    block[-1].at,
                    tuple(m.id for m in block),
                )
            )

    for message in messages:
        if block and block[-1].role != message.role:
            close()
            block = []
        block.append(message)
    close()
    return turns


@dataclass(frozen=True)
class WindowResult:
    """The turns the prompt shows, and where the window starts now."""

    turns: tuple[Turn, ...]
    start_at: datetime | None  # time of the first turn of the window; store this
    shifted: bool  # the window moved on this call (the cache prefix changed)


@dataclass(frozen=True)
class HistoryWindow:
    """When the window start moves (R-MEM-001): a pure rule, no state of its own."""

    minimum: int = 30
    maximum: int = 40

    def __post_init__(self) -> None:
        if not 1 <= self.minimum < self.maximum:
            raise ValueError("the window needs 1 <= minimum < maximum turns")

    @property
    def step(self) -> int:
        """How many turns the start moves at a time."""
        return self.maximum - self.minimum

    def start_index(self, turns: Sequence[Turn], start_at: datetime | None) -> int:
        """Index of the first turn of the window given the stored start (``None``: new)."""
        if start_at is None:
            return max(0, len(turns) - self.minimum)
        for index, turn in enumerate(turns):
            if turn.at >= start_at:
                return index
        return max(0, len(turns) - self.minimum)

    def advance(self, turns: Sequence[Turn], start_at: datetime | None) -> WindowResult:
        """The window over ``turns`` (all turns from the stored start on, oldest first)."""
        index = self.start_index(turns, start_at)
        shifted = False
        while len(turns) - index > self.maximum:
            index += self.step
            shifted = True
        shown = tuple(turns[index:])
        return WindowResult(shown, shown[0].at if shown else None, shifted)


class RecentTurns:
    """The recent turns of the bot's conversation, read through a :class:`BotTurnReader`."""

    def __init__(self, reader: BotTurnReader, window: HistoryWindow | None = None) -> None:
        self._reader = reader
        self._window = window or HistoryWindow()

    @classmethod
    def from_settings(cls, reader: BotTurnReader, services: Services) -> RecentTurns:
        engine = services.settings.engine
        return cls(reader, HistoryWindow(engine.history_turns_min, engine.history_turns_max))

    @property
    def policy(self) -> HistoryWindow:
        return self._window

    def window(self, start_at: datetime | None) -> WindowResult:
        """The turns to show now, given the stored window start (``None`` on a new conversation).

        The caller stores :attr:`WindowResult.start_at` and passes it next time.
        """
        if start_at is None:
            # a conversation without a stored start: read the newest messages only
            messages = self._reader.messages_since(
                None, self._window.maximum * NEWEST_MESSAGES_PER_TURN
            )
        else:
            messages = self._reader.messages_since(start_at)
        return self._window.advance(merge_turns(messages), start_at)


# ------------------------------------------------------------------------ registry

ReaderFactory = Callable[["Services"], BotTurnReader]
_factory: ReaderFactory | None = None


def register_bot_turn_reader(factory: ReaderFactory | None) -> None:
    """Round 09 calls this with a factory that reads ``bot_turns`` (``None`` unregisters)."""
    global _factory
    _factory = factory


def bot_turn_reader(services: Services) -> BotTurnReader | None:
    """A reader of the bot's conversation, or ``None`` while round 09 has not registered one."""
    return _factory(services) if _factory is not None else None
