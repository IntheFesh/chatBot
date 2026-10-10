"""What the reply pipeline may know: the data view (R-TRN-013, R-ENG-005).

:class:`ReplyDataView` is everything the pipeline reads about the world for one round: the clock
and her state, the persona card, the style profile, the memory, her real replies for examples,
the sticker library and the emoji vocabulary.  It is a protocol on purpose.  The live engine runs
the pipeline with a :class:`LiveDataView` (the present: the live profile, the whole memory, the
life line of today).  The evaluation sandbox (round 09b) and the training export run **the same
pipeline** with a view of a past moment - ``twin.memory.asof.AsOfView(t)`` has every member of
this protocol, with the same names and shapes - so a reply is produced and checked by one piece of
code wherever it is made, and a past moment can only see what existed then.

Members, with what ``AsOfView`` answers where it differs from the live view:

====================  ===============================================================
``at``, ``local``     the instant and the same instant on her wall clock (zone, weekday,
                      day type, 15-minute slot)
``persona_full()``    the card for the DeepSeek backend; ``persona_compact()`` the
                      style-only card of the style backend (pre-holdout card for a past
                      moment, the live ``[不要这样]`` lines included in the full one)
``profile``           the style profile (pre-holdout: the profile of the data before the
                      hold-out); ``activity`` the routine model
``her_state()``       what she is doing: from the day plan now, the typical state of the
                      routine for a past moment
``memory_block()``    the memory block (a past moment: what was known then)
``examples()``        her real replies in similar situations (a past moment: only earlier
                      ones, the hold-out never)
``sticker_*``,        the sticker library, the selector, the share controller and the
``emoji_codes``       emoji-code vocabulary of the scope
``lifeline``          the entry of her life line for this moment (past: always empty)
``bot_turns``         the bot's own conversation as the view knows it (past: always empty;
                      the pipeline reads the history from the context, not from here)
====================  ===============================================================
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from twin.llm.tokens import TokenEstimator
from twin.memory.api import lifeline_at
from twin.memory.asof import LocalMoment
from twin.memory.assemble import MemoryAssembler
from twin.memory.blocks import MemoryBlock, MemoryQuery
from twin.memory.memory import Memory
from twin.memory.recent import BotMessage, bot_turn_reader
from twin.memory.records import LifelineRecord
from twin.profile.activity_model import ActivityModel
from twin.profile.api import ProfileSnapshot, load_activity_model, load_profile
from twin.profile.persona.api import RenderedPersona, render_compact, render_full
from twin.retrieval.examples import Example
from twin.retrieval.query import ExampleQuery, ExampleRetriever, QueryTurn
from twin.schedule.service import time_service_for
from twin.schedule.time_service import PlanUnavailableError, TimeService
from twin.stickers.emoji_codes import EmojiCodePolicy
from twin.stickers.rate import BubbleHistory, StickerRateController
from twin.stickers.selector import LIVE_VIEW, StickerSelector, StickerView

if TYPE_CHECKING:
    from twin.services import Services

SLOT_MINUTES = 15
RECENT_MESSAGES_SHOWN = 80  # how many of the newest messages ``bot_turns`` hands out


@runtime_checkable
class ReplyDataView(Protocol):
    """Everything the pipeline reads about the world for one round (see the module text)."""

    @property
    def at(self) -> datetime: ...

    @property
    def local(self) -> LocalMoment: ...

    def persona_full(self) -> RenderedPersona | None: ...

    def persona_compact(self) -> RenderedPersona | None: ...

    @property
    def profile(self) -> ProfileSnapshot | None: ...

    @property
    def activity(self) -> ActivityModel | None: ...

    def her_state(self) -> str | None: ...

    def memory_block(self, query: MemoryQuery, budget_tokens: int | None = None) -> MemoryBlock: ...

    async def examples(
        self, turns: Sequence[QueryTurn], *, k: int | None = None
    ) -> list[Example]: ...

    @property
    def sticker_view(self) -> StickerView: ...

    def sticker_selector(self, rng: random.Random | None = None) -> StickerSelector: ...

    @property
    def emoji_codes(self) -> EmojiCodePolicy | None: ...

    def sticker_rate(self, history: BubbleHistory | None = None) -> StickerRateController: ...

    @property
    def lifeline(self) -> Sequence[LifelineRecord]: ...

    @property
    def bot_turns(self) -> Sequence[BotMessage]: ...


class LiveDataSource:
    """The long-lived pieces of the live view: memory, retrieval, stickers (one per process).

    Building it reads nothing; :meth:`view` makes the cheap per-round view.
    """

    def __init__(
        self,
        services: Services,
        *,
        memory: Memory | None = None,
        retriever: ExampleRetriever | None = None,
        selector: StickerSelector | None = None,
        time_service: TimeService | None = None,
        estimator: TokenEstimator | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.services = services
        self.time = time_service or time_service_for(services)
        self.memory = memory or Memory(services, time_service=self.time)
        self.retriever = retriever or ExampleRetriever(services)
        self.rng = rng or random.Random()  # noqa: S311 - a draw among stickers, not security
        self.selector = selector or StickerSelector(services, view=LIVE_VIEW, rng=self.rng)
        self.assembler = MemoryAssembler(self.memory, estimator=estimator)

    def view(self, at: datetime | None = None) -> LiveDataView:
        """The view of the present (or of ``at``, a moment of the present round)."""
        return LiveDataView(self, at if at is not None else self.time.now_utc())

    async def aclose(self) -> None:
        await self.retriever.aclose()


class _Once[T]:
    """A value read on first use and kept for the rest of the round."""

    def __init__(self, load: Callable[[], T]) -> None:
        self._load = load
        self._box: list[T] = []

    def get(self) -> T:
        if not self._box:
            self._box.append(self._load())
        return self._box[0]


class LiveDataView:
    """The data view of the running bot: the live profile, the whole memory, today's life line."""

    def __init__(self, source: LiveDataSource, at: datetime) -> None:
        self._source = source
        self._services = source.services
        self._at = at
        self._local: LocalMoment | None = None
        self._profile = _Once(lambda: load_profile(self._services, "live"))
        self._activity = _Once(lambda: load_activity_model(self._services, "live"))
        self._emoji = _Once(lambda: EmojiCodePolicy.from_profile(self._services, "live"))

    # ---------------------------------------------------------------- the moment

    @property
    def at(self) -> datetime:
        return self._at

    @property
    def local(self) -> LocalMoment:
        if self._local is None:
            time = self._source.time
            zone = time.bot_timezone()
            local = self._at.astimezone(zone)
            day = local.date()
            self._local = LocalMoment(
                at=self._at,
                local=local,
                zone=zone.key,
                day=day,
                weekday=day.weekday(),
                day_type=time.day_type(day),
                slot=(local.hour * 60 + local.minute) // SLOT_MINUTES,
            )
        return self._local

    # ------------------------------------------------------- persona, profile, routine

    def persona_full(self) -> RenderedPersona | None:
        return render_full(self._services, "live")

    def persona_compact(self) -> RenderedPersona | None:
        return render_compact(self._services, "live")

    @property
    def profile(self) -> ProfileSnapshot | None:
        return self._profile.get()

    @property
    def activity(self) -> ActivityModel | None:
        return self._activity.get()

    def her_state(self) -> str | None:
        """What she is doing now, from the day plan; ``None`` if there is no plan yet."""
        try:
            return str(self._source.time.her_state(self._at).kind)
        except PlanUnavailableError:
            return None

    # -------------------------------------------------------------------- memory

    def memory_block(self, query: MemoryQuery, budget_tokens: int | None = None) -> MemoryBlock:
        return self._source.assembler.build(query, self._at, budget_tokens)

    @property
    def lifeline(self) -> Sequence[LifelineRecord]:
        """The entry of today's life line that covers this moment (nothing if none does)."""
        found = lifeline_at(self._services, self._at, memory=self._source.memory)
        return (found,) if found is not None else ()

    @property
    def bot_turns(self) -> Sequence[BotMessage]:
        reader = bot_turn_reader(self._services)
        return tuple(reader.messages_since(None, RECENT_MESSAGES_SHOWN)) if reader else ()

    # --------------------------------------------------------------- her real replies

    async def examples(self, turns: Sequence[QueryTurn], *, k: int | None = None) -> list[Example]:
        local = self.local
        query = ExampleQuery(
            turns=turns, local_minute=local.minute, day_type=local.day_type, before=None, k=k
        )
        return await self._source.retriever.query(query)

    # ------------------------------------------------------- stickers and emoji codes

    @property
    def sticker_view(self) -> StickerView:
        return LIVE_VIEW

    def sticker_selector(self, rng: random.Random | None = None) -> StickerSelector:
        if rng is None:
            return self._source.selector
        return StickerSelector(self._services, view=LIVE_VIEW, rng=rng)

    @property
    def emoji_codes(self) -> EmojiCodePolicy | None:
        return self._emoji.get()

    def sticker_rate(self, history: BubbleHistory | None = None) -> StickerRateController:
        return StickerRateController.from_services(self._services, scope="live", history=history)
