"""A data view with fixed answers, for the tests of the reply pipeline (round 09).

``StaticDataView`` implements :class:`twin.engine.dataview.ReplyDataView` over plain values: the
persona text, a profile built from the numbers a test cares about, the memory text, the examples,
a sticker chooser.  It never touches the database, so the post-processing, the prompt and the
pipeline can be tested for every rule without building a conversation first.  The live data view
and ``AsOfView`` are tested against real services elsewhere.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from twin.memory.asof import LocalMoment
from twin.memory.blocks import BlockItem, MemoryBlock, MemoryQuery
from twin.memory.recent import BotMessage
from twin.memory.records import LifelineRecord
from twin.profile.activity_model import ActivityModel
from twin.profile.distribution import EmpiricalDistribution
from twin.profile.persona.render import RenderedPersona
from twin.profile.snapshot import ProfileMetrics, assemble_metrics
from twin.profile.values import Dist, Leaf, Rates, Scalar
from twin.retrieval.examples import Example
from twin.retrieval.query import QueryTurn
from twin.stickers.catalog import StickerRecord
from twin.stickers.emoji_codes import EmojiCodePolicy
from twin.stickers.rate import BubbleHistory, MemoryBubbleHistory, StickerRateController
from twin.stickers.selector import LIVE_VIEW, StickerView

NOON = datetime(2026, 10, 9, 18, 30, tzinfo=UTC)  # 13:30 in Chicago on a Friday


def local_moment(at: datetime = NOON, zone: str = "America/Chicago") -> LocalMoment:
    local = at.astimezone(ZoneInfo(zone))
    day: date = local.date()
    return LocalMoment(
        at=at,
        local=local,
        zone=zone,
        day=day,
        weekday=day.weekday(),
        day_type="workday" if day.weekday() < 5 else "weekend",
        slot=(local.hour * 60 + local.minute) // 15,
    )


class StaticProfile:
    """What the pipeline reads of ``ProfileSnapshot``: the metrics."""

    def __init__(self, metrics: ProfileMetrics) -> None:
        self.metrics = metrics


def dist(values: dict[float, int]) -> Dist:
    return Dist(EmpiricalDistribution.from_counter(Counter(values), discrete=True))


def make_profile(
    *,
    lengths: dict[float, int] | None = None,
    bursts: dict[float, int] | None = None,
    comma_rate: float = 0.03,
    period_end_rate: float = 0.002,
    code_rate: float = 0.035,
    code_runs: dict[float, int] | None = None,
    closing_no_reply: float | None = 0.4,
    sticker_share: float = 0.105,
    texts: int = 500,
) -> StaticProfile:
    """A profile of her with the numbers a test sets (defaults follow SPEC section 0)."""
    leaves: dict[str, Leaf] = {
        "text_length": dist(lengths or {3.0: 20, 5.0: 40, 8.0: 20, 10.0: 15, 14.0: 5}),
        "burst_size": dist(bursts or {1.0: 40, 2.0: 30, 3.0: 15, 4.0: 10, 6.0: 5}),
        "punct_rate": Rates({"comma": comma_rate, "question": 0.1, "exclaim": 0.05}, texts),
        "end_rate": Rates({"period": period_end_rate, "none": 0.8}, texts),
        "emoji_code_rate": Scalar(code_rate, texts),
        "emoji_code_run_length": dist(code_runs or {1.0: 9, 2.0: 1}),
        "sticker_share": Scalar(sticker_share, 1000),
    }
    if closing_no_reply is not None:
        leaves["closing_no_reply_rate"] = Scalar(closing_no_reply, 60)
    data = assemble_metrics(
        scope="live",
        config={},
        weight=0.6,
        full={"her": leaves, "user": {}},
        recent=None,
        full_info={},
        recent_info={},
    )
    return StaticProfile(ProfileMetrics(data))


class StaticSelector:
    """A sticker chooser with a fixed table ``tag -> record`` and a log of the questions asked."""

    def __init__(self, table: dict[str, StickerRecord]) -> None:
        self.table = table
        self.asked: list[tuple[str, str, tuple[str | None, ...]]] = []

    def choose(
        self, tag: str, context_text: str, recent_bubbles: Sequence[str | None] = ()
    ) -> StickerRecord | None:
        self.asked.append((tag, context_text, tuple(recent_bubbles)))
        return self.table.get(tag)


def sticker_record(md5: str, *tags: str) -> StickerRecord:
    return StickerRecord(
        md5=md5,
        status="available",
        mime="image/png",
        width=64,
        height=64,
        her_uses=3,
        user_uses=0,
        first_used_at=None,
        last_used_at=None,
        tags=tags,
        vision_tags=tags,
        context_tags=(),
        manual_tags=(),
        tag_source="vision",
        description="一只猫",
        use_cases=None,
        context_note=None,
        origin="export",
        disabled=False,
        tagged_at=None,
        context_tagged_at=None,
        context_cutoff_at=None,
        context_uses=0,
        desc_vector_id=None,
        desc_encoding=None,
        sha256="a" * 64,
    )


@dataclass
class StaticDataView:
    """A :class:`~twin.engine.dataview.ReplyDataView` with fixed answers (see the module text)."""

    at: datetime = NOON
    zone: str = "America/Chicago"
    persona_text: str | None = "她说话很短，爱用“哈哈”。"
    profile: StaticProfile | None = field(default_factory=make_profile)  # type: ignore[assignment]
    state: str | None = "free"
    memory_text: str = ""
    example_list: list[Example] = field(default_factory=list)
    lifeline_records: tuple[LifelineRecord, ...] = ()
    selector: StaticSelector = field(default_factory=lambda: StaticSelector({}))
    emoji: EmojiCodePolicy | None = field(
        default_factory=lambda: EmojiCodePolicy({"拥抱": 0.6, "亲亲": 0.4}, rate=0.5)
    )
    history_flags: list[bool] = field(default_factory=list)
    fail_memory: bool = False
    fail_examples: bool = False
    memory_queries: list[tuple[MemoryQuery, int | None]] = field(default_factory=list)
    example_queries: list[tuple[list[QueryTurn], int | None]] = field(default_factory=list)
    activity: ActivityModel | None = None
    bot_turns: Sequence[BotMessage] = ()
    rngs: list[random.Random | None] = field(default_factory=list)

    @property
    def local(self) -> LocalMoment:
        return local_moment(self.at, self.zone)

    def persona_full(self) -> RenderedPersona | None:
        if self.persona_text is None:
            return None
        return RenderedPersona(self.persona_text, "full", "live", "p1", 3, 20, 1500, 0)

    def persona_compact(self) -> RenderedPersona | None:
        if self.persona_text is None:
            return None
        return RenderedPersona(self.persona_text, "compact", "live", "p1", 3, 20, 400, 0)

    def her_state(self) -> str | None:
        return self.state

    def memory_block(self, query: MemoryQuery, budget_tokens: int | None = None) -> MemoryBlock:
        self.memory_queries.append((query, budget_tokens))
        if self.fail_memory:
            raise RuntimeError("the memory is not there")
        if not self.memory_text:
            return MemoryBlock("")
        item = BlockItem("fact", "f1", "相关的事", 1.0, False, 10)
        return MemoryBlock(self.memory_text, (item,), 10, budget_tokens or 0)

    async def examples(self, turns: Sequence[QueryTurn], *, k: int | None = None) -> list[Example]:
        self.example_queries.append((list(turns), k))
        if self.fail_examples:
            raise RuntimeError("the library is not there")
        return list(self.example_list)

    @property
    def sticker_view(self) -> StickerView:
        return LIVE_VIEW

    def sticker_selector(self, rng: random.Random | None = None) -> StaticSelector:  # type: ignore[override]
        self.rngs.append(rng)
        return self.selector

    @property
    def emoji_codes(self) -> EmojiCodePolicy | None:
        return self.emoji

    def sticker_rate(self, history: BubbleHistory | None = None) -> StickerRateController:
        share = self.profile.metrics.scalar("her", "sticker_share") if self.profile else None
        return StickerRateController(
            share,
            history or MemoryBubbleHistory(200, self.history_flags),
            window=200,
            tolerance=0.2,
        )

    @property
    def lifeline(self) -> Sequence[LifelineRecord]:
        return self.lifeline_records


Chooser = Callable[[str, str, Sequence[str | None]], StickerRecord | None]
