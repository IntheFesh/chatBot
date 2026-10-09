"""``AsOfView(t)``: everything the bot could have known at a past moment (R-TRN-013).

The training export (round 13, R-TRN-002) and the evaluation sandbox (round 09b, R-EVAL-009)
build the context of a sample - *her reply block at time t, what would the bot have been given?* -
**only** through this class.  It is the one door to the data for a past moment, so the rule
"nothing from the future" is kept in one place:

======================  ==================================================================
``memory``              :class:`~twin.memory.view.MemoryView` of ``t`` (``memory_view(t)``):
                        facts with ``known_at < t``, summaries of earlier local days,
                        follow-ups open at ``t``; no life line.  ``memory_block()`` renders it.
``persona_full()``      the **pre_holdout** card, full version, with the live card's
                        ``[不要这样]`` lines added (they only describe a manner of speaking);
``persona_compact()``   the pre_holdout card, style only (no facts, no corrections);
``profile``             the pre_holdout style profile; ``activity`` the pre_holdout routine;
``her_state()``         ``typical_state`` of the pre_holdout routine for the local clock time
                        and day type of ``t`` in the zone she was in (R-ACT-001);
``examples()``          her real replies from before ``t`` (the retrieval library with
                        ``before=t``; the hold-out is never returned);
``sticker_view`` ...    the sticker library as of ``t`` (only stickers she had used before
                        ``t``; counts and recency as of ``t``), the emoji-code vocabulary and
                        the sticker share of the pre_holdout profile;
``lifeline``            always empty - the bot's life line did not exist then;
``bot_turns``           always empty - nor did its conversation.
======================  ==================================================================

There is no method that reads all of the data: the class holds a private source and no public
attribute leads to the tables, and :data:`PUBLIC_API` lists every public name (a test pins it).

Typical use for a whole export::

    source = AsOfSource(services)            # loads the pre_holdout data once
    view = source.at(sample_time)            # cheap: no database access of its own
    system = view.persona_compact().text
    memory = view.memory_block(MemoryQuery(text=recent_turns_text)).text
    state = view.her_state()
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from twin.clock import ensure_aware
from twin.memory.assemble import MemoryAssembler
from twin.memory.blocks import MemoryBlock, MemoryQuery
from twin.memory.memory import Memory
from twin.memory.view import MemoryView
from twin.profile.activity_model import ActivityModel, TypicalState
from twin.profile.api import ProfileSnapshot, load_activity_model, load_profile
from twin.profile.persona.api import RenderedPersona, render_compact, render_full
from twin.retrieval.examples import Example
from twin.retrieval.query import ExampleQuery, ExampleRetriever, QueryTurn
from twin.retrieval.windows import LocalPlace
from twin.stickers.emoji_codes import EmojiCodePolicy
from twin.stickers.rate import BubbleHistory, MemoryBubbleHistory, StickerRateController
from twin.stickers.selector import StickerSelector, StickerView, as_of_view

if TYPE_CHECKING:
    from twin.services import Services

SCOPE = "pre_holdout"
WEEKDAY_NAMES = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")

PUBLIC_API = frozenset(
    {
        "at",
        "local",
        "memory",
        "memory_block",
        "persona_full",
        "persona_compact",
        "profile",
        "activity",
        "her_state",
        "examples",
        "sticker_view",
        "sticker_selector",
        "emoji_codes",
        "sticker_rate",
        "lifeline",
        "bot_turns",
    }
)


@dataclass(frozen=True)
class LocalMoment:
    """The moment on the clock of the place she was in (R-ACT-001)."""

    at: datetime  # the instant (UTC)
    local: datetime  # the same instant on her wall clock
    zone: str
    day: date
    weekday: int  # Monday = 0
    day_type: str  # workday | weekend | holiday
    slot: int  # 15-minute slot of the local day, 0..95

    @property
    def weekday_name(self) -> str:
        return WEEKDAY_NAMES[self.weekday]

    @property
    def minute(self) -> float:
        return self.local.hour * 60 + self.local.minute + self.local.second / 60.0


class AsOfSource:
    """The data shared by the views of many moments, loaded once (see the module description)."""

    def __init__(
        self,
        services: Services,
        *,
        memory: Memory | None = None,
        retriever: ExampleRetriever | None = None,
    ) -> None:
        self._services = services
        self._memory = memory or Memory(services)
        self._assembler = MemoryAssembler(self._memory)
        self._memory.refresh()
        self._place = LocalPlace.create(services)
        self._retriever = retriever or ExampleRetriever(services)
        self._profile: ProfileSnapshot | None = load_profile(services, SCOPE)
        self._activity: ActivityModel | None = load_activity_model(services, SCOPE)
        self._full = render_full(services, SCOPE, with_live_corrections=True)
        self._compact = render_compact(services, SCOPE)
        self._emoji = EmojiCodePolicy.from_profile(services, SCOPE)
        self._selector: StickerSelector | None = None

    def at(self, moment: datetime) -> AsOfView:
        """The view of ``moment`` (it must carry a time zone)."""
        return AsOfView(self, moment)

    # -- used by the views of this source (not part of the views' public surface) --

    def local_moment(self, moment: datetime) -> LocalMoment:
        stamp = self._place.stamp(moment)
        day_type = self._place.day_type(stamp.day, stamp.zone)
        return LocalMoment(
            at=moment,
            local=moment.astimezone(ZoneInfo(stamp.zone)),
            zone=stamp.zone,
            day=stamp.day,
            weekday=stamp.day.weekday(),
            day_type=day_type,
            slot=stamp.slot,
        )

    def next_day_type(self, local: LocalMoment) -> str:
        return self._place.day_type(local.day + timedelta(days=1), local.zone)

    def memory_view(self, moment: datetime) -> MemoryView:
        return self._assembler.view(moment, include_lifeline=False, refresh=False)

    def render_memory(
        self, view: MemoryView, query: MemoryQuery, budget_tokens: int | None
    ) -> MemoryBlock:
        return self._assembler.render_view(view, query, budget_tokens)

    @property
    def profile(self) -> ProfileSnapshot | None:
        return self._profile

    @property
    def activity(self) -> ActivityModel | None:
        return self._activity

    @property
    def full(self) -> RenderedPersona | None:
        return self._full

    @property
    def compact(self) -> RenderedPersona | None:
        return self._compact

    @property
    def emoji(self) -> EmojiCodePolicy | None:
        return self._emoji

    @property
    def services(self) -> Services:
        return self._services

    @property
    def retriever(self) -> ExampleRetriever:
        return self._retriever

    def selector(self) -> StickerSelector:
        if self._selector is None:
            self._selector = StickerSelector(self._services)
        return self._selector


class AsOfView:
    """What the bot could have known at ``moment`` (see the module description)."""

    def __init__(self, source: AsOfSource | Services, moment: datetime) -> None:
        # `AsOfView(services, t)` builds a source of its own; for many moments share one
        self._source: AsOfSource = source if isinstance(source, AsOfSource) else AsOfSource(source)
        self._at = ensure_aware(moment)
        self._local: LocalMoment | None = None
        self._memory: MemoryView | None = None

    # ------------------------------------------------------------------ the moment

    @property
    def at(self) -> datetime:
        return self._at

    @property
    def local(self) -> LocalMoment:
        """The moment on the wall clock of the place she was in, with its day type."""
        if self._local is None:
            self._local = self._source.local_moment(self._at)
        return self._local

    # ------------------------------------------------------------------- memory

    @property
    def memory(self) -> MemoryView:
        """``memory_view(t)``: facts, summaries and follow-ups as they were known at ``t``."""
        if self._memory is None:
            self._memory = self._source.memory_view(self._at)
        return self._memory

    def memory_block(self, query: MemoryQuery, budget_tokens: int | None = None) -> MemoryBlock:
        """The memory block of ``t`` (R-MEM-008), built from :attr:`memory` only."""
        return self._source.render_memory(self.memory, query, budget_tokens)

    @property
    def lifeline(self) -> tuple[()]:
        """Always empty: the bot's life line did not exist at the time of the real records."""
        return ()

    @property
    def bot_turns(self) -> tuple[()]:
        """Always empty: nor did the conversation with the bot."""
        return ()

    # ------------------------------------------------------- persona, profile, routine

    def persona_full(self) -> RenderedPersona | None:
        """The pre_holdout card, full version, with the live ``[不要这样]`` lines added."""
        return self._source.full

    def persona_compact(self) -> RenderedPersona | None:
        """The pre_holdout card, style only: no facts and no corrections (R-PERS-004)."""
        return self._source.compact

    @property
    def profile(self) -> ProfileSnapshot | None:
        return self._source.profile

    @property
    def activity(self) -> ActivityModel | None:
        return self._source.activity

    def her_state(self) -> TypicalState | None:
        """Her typical state at ``t``: the pre_holdout routine for the local clock and day type.

        ``None`` when no routine has been computed for the pre_holdout scope.
        """
        model = self._source.activity
        if model is None:
            return None
        local = self.local
        return model.typical_state(
            time(local.local.hour, local.local.minute, local.local.second),
            local.day_type,
            next_day_type=self._source.next_day_type(local),
            weekday=local.weekday,
        )

    # --------------------------------------------------------------- her real replies

    async def examples(self, turns: Sequence[QueryTurn], *, k: int | None = None) -> list[Example]:
        """Her real replies to situations like ``turns``, from before ``t`` only (R-RET-005)."""
        local = self.local
        query = ExampleQuery(
            turns=turns,
            local_minute=local.minute,
            day_type=local.day_type,
            before=self._at,
            k=k,
        )
        return await self._source.retriever.query(query)

    # ------------------------------------------------------- stickers and emoji codes

    @property
    def sticker_view(self) -> StickerView:
        """The sticker library as of ``t`` (pre_holdout scope; uses before ``t`` only)."""
        return as_of_view(self._at)

    def sticker_selector(self, rng: random.Random | None = None) -> StickerSelector:
        """A selector for the library as of ``t`` (shares the loaded data of the source)."""
        if rng is None:
            return self._source.selector().with_view(self.sticker_view)
        return StickerSelector(self._source.services, view=self.sticker_view, rng=rng)

    @property
    def emoji_codes(self) -> EmojiCodePolicy | None:
        """Her emoji-code vocabulary, rate and runs from the pre_holdout profile."""
        return self._source.emoji

    def sticker_rate(self, history: BubbleHistory | None = None) -> StickerRateController:
        """The sticker-share controller with the pre_holdout share (R-STK-005).

        ``history`` is the recent bubbles of the sample's own context; by default there are none.
        """
        services = self._source.services
        window = services.settings.stickers.rate_window
        return StickerRateController.from_services(
            services, scope=SCOPE, history=history or MemoryBubbleHistory(window)
        )
