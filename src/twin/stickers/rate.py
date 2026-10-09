"""Keeping the bot's share of stickers near hers (R-STK-005).

Over a rolling window of the last ``stickers.rate_window`` (200) bubbles the share of stickers is
kept within ``stickers.rate_tolerance`` (20 %) of her share (``sticker_share`` of her profile,
about 10.5 % in the sample): at most her share times 1.2.  :meth:`StickerRateController.should_drop`
answers whether one more sticker would push the share over that upper limit - then the
post-processing deletes the sticker line.  The controller never asks for a sticker to be added:
when the share is low nothing is forced (R-STK-005, "不足时不强行插入").

A window that is not full yet is counted as if the missing bubbles had been sent at her share, so
the first sticker of a new conversation is not judged against an empty window.

Where the window comes from.  Round 09 writes every bubble the bot sends into ``bot_turns``; this
module only asks for a :class:`BubbleHistory` - the last bubbles as flags (``True`` for a sticker),
oldest first.  :class:`BotTurnBubbleHistory` reads them from ``bot_turns`` (the bubbles of the
conversation, commands and thrown-away replies left out), so a restart forgets nothing and nothing
has to be recorded twice; :class:`MemoryBubbleHistory` keeps them in memory (the evaluation sandbox
and the tests) and can be filled with :meth:`MemoryBubbleHistory.record`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from sqlalchemy import select

from twin.profile.api import load_profile
from twin.services import Services
from twin.storage.db import Database
from twin.storage.engine_models import BotTurn


class BubbleHistory(Protocol):
    """The last bubbles the bot sent, as flags (``True`` = a sticker), oldest first."""

    def flags(self) -> list[bool]: ...


@runtime_checkable
class RecordingHistory(BubbleHistory, Protocol):
    """A history the caller fills (the ``bot_turns`` one fills itself when a bubble is stored)."""

    def record(self, is_sticker: bool) -> None: ...


class MemoryBubbleHistory:
    """The window in memory."""

    def __init__(self, window: int, initial: Iterable[bool] = ()) -> None:
        self._flags: deque[bool] = deque(initial, maxlen=window)

    def flags(self) -> list[bool]:
        return list(self._flags)

    def record(self, is_sticker: bool) -> None:
        self._flags.append(bool(is_sticker))


class BotTurnBubbleHistory:
    """The window read from ``bot_turns``: the last bubbles the bot sent (R-STK-005, R-ENG-011).

    Only bubbles of the conversation count - not the replies to commands and not replies the user
    threw away - and a bubble is in the window the moment the engine has stored it.
    """

    def __init__(self, db: Database, window: int) -> None:
        if window < 1:
            raise ValueError("the window must hold at least one bubble")
        self._db = db
        self._window = window

    def flags(self) -> list[bool]:
        stmt = (
            select(BotTurn.kind)
            .where(
                BotTurn.direction == "out",
                BotTurn.is_command.is_(False),
                BotTurn.rejected_at.is_(None),
                BotTurn.kind.in_(("text", "sticker")),
            )
            .order_by(BotTurn.at.desc(), BotTurn.id.desc())
            .limit(self._window)
        )
        with self._db.session() as session:
            kinds = list(session.scalars(stmt))
        return [kind == "sticker" for kind in reversed(kinds)]


@dataclass(frozen=True)
class RateStatus:
    """Where the share stands."""

    share: float
    her_share: float | None
    lower: float | None
    upper: float | None
    state: Literal["low", "ok", "high", "unknown"]
    bubbles: int


class StickerRateController:
    """Decides whether one more sticker is allowed (see the module description)."""

    def __init__(
        self,
        her_share: float | None,
        history: BubbleHistory,
        *,
        window: int,
        tolerance: float,
    ) -> None:
        if window < 1:
            raise ValueError("the window must hold at least one bubble")
        if not 0 <= tolerance < 1:
            raise ValueError("the tolerance must be in [0, 1)")
        self._share = her_share
        self._history = history
        self._window = window
        self._tolerance = tolerance

    @classmethod
    def from_services(
        cls, services: Services, *, scope: str = "live", history: BubbleHistory | None = None
    ) -> StickerRateController:
        """A controller for her share in the profile of ``scope`` (live, or pre-holdout)."""
        config = services.settings.stickers
        profile = load_profile(services, scope)
        share = profile.metrics.scalar("her", "sticker_share") if profile is not None else None
        return cls(
            share,
            history or BotTurnBubbleHistory(services.db, config.rate_window),
            window=config.rate_window,
            tolerance=config.rate_tolerance,
        )

    @property
    def her_share(self) -> float | None:
        return self._share

    @property
    def upper(self) -> float | None:
        return None if self._share is None else self._share * (1 + self._tolerance)

    @property
    def lower(self) -> float | None:
        return None if self._share is None else self._share * (1 - self._tolerance)

    def _share_of(self, flags: list[bool]) -> float:
        """The share in the window; missing bubbles count as sent at her share."""
        recent = flags[-self._window :]
        prior = (self._share or 0.0) * (self._window - len(recent))
        return (sum(recent) + prior) / self._window

    def share(self) -> float:
        return self._share_of(self._history.flags())

    def share_if_added(self) -> float:
        """The share after one more sticker bubble."""
        return self._share_of([*self._history.flags(), True])

    def should_drop(self) -> bool:
        """True if one more sticker would take the share above her share plus the tolerance."""
        return self.should_drop_after(())

    def should_drop_after(self, pending: Sequence[bool]) -> bool:
        """The same question once the bubbles of ``pending`` (flags, oldest first) are in.

        A reply with several bubbles is judged bubble by bubble: the stickers already chosen for
        it count, although the engine stores them only when it sends them.
        """
        upper = self.upper
        if upper is None:
            return False  # without her profile there is nothing to compare with
        return self._share_of([*self._history.flags(), *pending, True]) > upper + 1e-12

    def status(self) -> RateStatus:
        flags = self._history.flags()
        share = self._share_of(flags)
        upper, lower = self.upper, self.lower
        state: Literal["low", "ok", "high", "unknown"]
        if upper is None or lower is None:
            state = "unknown"
        elif share > upper + 1e-12:
            state = "high"
        elif share < lower - 1e-12:
            state = "low"
        else:
            state = "ok"
        return RateStatus(share, self._share, lower, upper, state, min(len(flags), self._window))

    def record(self, is_sticker: bool) -> None:
        """Note one bubble that was sent (histories that fill themselves refuse)."""
        if not isinstance(self._history, RecordingHistory):
            raise TypeError("this window is read from bot_turns; store the bubble there instead")
        self._history.record(is_sticker)
