"""The recent conversation for the prompt, with its window start kept on disk (R-MEM-001).

:class:`HistoryLoader` reads the bot's conversation through the memory's
:class:`~twin.memory.recent.BotTurnReader`, leaves out the messages of the round that is being
answered (they are the last user message of the prompt, not history), applies the batch rule of
:class:`~twin.memory.recent.HistoryWindow` and keeps the window start in ``conversation_state``:

* the start only moves when the window outgrows ``engine.history_turns_max`` (40), then in one
  step of ten turns back to about 30, so consecutive prompts share their beginning (R-LLM-010);
* a new conversation starts at its newest 30 turns;
* the start is stored whenever it changed, so a restart keeps showing the same window.

The class decides nothing about the content of the turns; ``merge_turns`` and ``HistoryWindow`` do.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime

from twin.engine.state_store import ConversationStateStore
from twin.memory.recent import (
    NEWEST_MESSAGES_PER_TURN,
    BotTurnReader,
    HistoryWindow,
    WindowResult,
    merge_turns,
)


class HistoryLoader:
    """Loads the window of the recent conversation and remembers where it starts."""

    def __init__(
        self,
        reader: BotTurnReader,
        window: HistoryWindow,
        state: ConversationStateStore,
    ) -> None:
        self._reader = reader
        self._window = window
        self._state = state

    @property
    def policy(self) -> HistoryWindow:
        return self._window

    def load(self, *, exclude_ids: Collection[str] = ()) -> WindowResult:
        """The turns to show now.  ``exclude_ids`` are the messages of the current round."""
        start_at: datetime | None = self._state.load().window_start_at
        if start_at is None:
            messages = self._reader.messages_since(
                None, (self._window.maximum + len(exclude_ids)) * NEWEST_MESSAGES_PER_TURN
            )
        else:
            messages = self._reader.messages_since(start_at)
        excluded = set(exclude_ids)
        kept = [message for message in messages if message.id not in excluded]
        result = self._window.advance(merge_turns(kept), start_at)
        if result.start_at != start_at and result.start_at is not None:
            self._state.set_window_start(result.start_at)
        return result
