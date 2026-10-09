"""The structured prompt sample of a reply the user did not like (R-LRN-002).

:class:`SampleBuilder` rebuilds the situation a reply was written in: the conversation up to the
user's turn that it answered (read through the bot's conversation window, so commands and thrown
away replies are not part of it) and the data view of that moment.  The system segment and the
turns are made by :meth:`~twin.engine.style_prompt.StylePromptBuilder.compose` - the builder of the
running style model, with the same ``normalize_context`` and the same token budget - and kept as
they are: structured, not rendered (see :mod:`twin.learning.pairs`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Protocol

from twin.engine.dataview import ReplyDataView
from twin.engine.style_backend import DEFAULT_BUDGET
from twin.engine.style_prompt import (
    STYLE_CONTEXT_TURNS,
    StylePromptBuilder,
    StylePromptError,
    normalize_context,
    style_turns,
)
from twin.engine.turns import TurnRecord
from twin.learning.pairs import NO_PERSONA, PairError, PromptSample
from twin.memory.recent import BotTurnReader

if TYPE_CHECKING:
    from twin.services import Services

LOOKBACK = timedelta(days=3)
"""How far before the reply the conversation is read; the sample keeps the newest 8 turns."""


class ViewSource(Protocol):
    """What makes the data view of a moment (``LiveDataSource`` in the running bot)."""

    def view(self, at: datetime | None = None) -> ReplyDataView: ...


class LazyViewSource:
    """A :class:`ViewSource` that makes the real one (memory, retrieval) on first use."""

    def __init__(self, factory: Callable[[], ViewSource]) -> None:
        self._factory = factory
        self._source: ViewSource | None = None

    def view(self, at: datetime | None = None) -> ReplyDataView:
        if self._source is None:
            self._source = self._factory()
        return self._source.view(at)


@dataclass(frozen=True)
class BuiltSample:
    """A prompt sample and the versions it was made with."""

    sample: PromptSample
    template_version: str
    persona_version: str


class SampleBuilder:
    """Builds the prompt sample of a reply (see the module description)."""

    def __init__(self, services: Services, reader: BotTurnReader, data: ViewSource) -> None:
        self._services = services
        self._reader = reader
        self._data = data

    def build(self, reply: list[TurnRecord]) -> BuiltSample:
        """The sample of ``reply`` (its rows); :class:`PairError` if it answered no one."""
        if not reply:
            raise PairError("there is no reply to describe")
        moment = reply[0].at
        messages = self._reader.messages_between(moment - LOOKBACK, moment)
        turns = style_turns(messages)
        if not turns or turns[-1].role != "user":
            raise PairError("the reply did not answer a message of the user")
        builder = StylePromptBuilder.from_services(self._services)
        try:
            parts = builder.compose(self._data.view(moment), turns, budget=DEFAULT_BUDGET)
        except StylePromptError as exc:
            raise PairError(str(exc)) from None
        meta = parts.meta
        persona = (
            f"{meta.persona_scope}:{meta.persona_version}"
            if meta.persona_version is not None
            else NO_PERSONA
        )
        prelude = normalize_context(turns, max_turns=STYLE_CONTEXT_TURNS).prelude
        sample = PromptSample(parts.system, parts.turns, moment, prelude)
        return BuiltSample(sample, meta.template_version, persona)
