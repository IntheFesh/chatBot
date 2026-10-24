"""The data that flows through the reply pipeline (R-ENG-005 to R-ENG-008, R-ENG-011).

``ReplyContext``
    one stretch of conversation to answer: the user's new messages, the window of recent turns
    before them, and the facts about the round (thinking mode, whether she has just woken up, what
    was already said).  Nothing in it is read from the database - the caller builds it, which is
    what lets the live engine and the evaluation sandbox feed the same pipeline.
``ReplyMaterial``
    what the pipeline gathered from the data view for this round (time, her state, memory,
    examples, persona); every backend builds its prompt from it.
``ReplyDraft``
    the pipeline's answer: the bubbles (text or a resolved sticker), the audit trail of the
    post-processing, the violations seen on the way, the cost, the stage timings and - when
    nothing usable could be produced - the signal that the engine must fall back (R-ENG-010).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from twin.config.runtime import ThinkingMode
from twin.engine.turns import ReplyMeta
from twin.memory.asof import LocalMoment
from twin.memory.recent import Turn
from twin.retrieval.examples import Example

BubbleKind = Literal["text", "sticker"]
FallbackReason = Literal["refused", "violations", "backend_error", "empty"]
STATE_KINDS = ("deep_sleep", "sleep_edge", "busy", "free")


@dataclass(frozen=True)
class InboundItem:
    """A message of the user as it stands in the conversation.

    ``text`` is the stable rendering - the user's words, or for a picture, a voice message or a
    sticker its description - produced once when the message arrives and stored as it is, so the
    message reads the same in every later prompt (R-LLM-010).  ``media`` is what ``bot_turns``
    keeps of the attachment (reference, quoted message, flags).  ``id`` is the channel's message
    id, ``turn_id`` the id of the stored row - the one :meth:`HistoryLoader.load` is told to leave
    out of the history.
    """

    id: str
    at: datetime
    kind: str
    text: str
    media: Mapping[str, Any] | None = None
    turn_id: str | None = None  # its row in ``bot_turns`` once stored (the history leaves it out)


@dataclass(frozen=True)
class SendLimits:
    """What the channel allows this reply (R-CH-008, R-ENG-009)."""

    supports_quote: bool = True
    max_bubbles: int | None = None  # the most bubbles the platform quota leaves; None: no limit


@dataclass(frozen=True)
class ReplyContext:
    """One round to answer."""

    inbound: tuple[InboundItem, ...]
    history: tuple[Turn, ...] = ()
    thinking_mode: ThinkingMode = "off"
    backend: str = "deepseek"
    limits: SendLimits = field(default_factory=SendLimits)
    woke_up: bool = False  # "你刚醒，看到了这些消息" (R-ENG-003)
    already_said: tuple[str, ...] = ()  # bubbles of this round that are already out (R-ENG-009)
    recent_stickers: tuple[str | None, ...] = ()  # last bubbles, oldest first: sticker MD5 or None
    seed: int | None = None  # makes the random choices of one run repeatable

    def __post_init__(self) -> None:
        if not self.inbound:
            raise ValueError("a reply needs at least one message of the user to answer")

    @property
    def user_text(self) -> str:
        """The user's messages of this round, one per line."""
        return "\n".join(item.text for item in self.inbound)

    @property
    def last_inbound(self) -> InboundItem:
        return self.inbound[-1]


@dataclass(frozen=True)
class ClosingHint:
    """The user's last message is a short reply that asks nothing; how often she stays silent."""

    no_reply_rate: float  # 0 when the profile has no data: ``[不回]`` is then not allowed


@dataclass(frozen=True)
class ReplyMaterial:
    """What the pipeline gathered from the data view for one round (shared by all attempts)."""

    local: LocalMoment
    state: str | None
    lifeline: str | None
    persona: str
    persona_ref: str | None  # "live v3": which card the persona text came from
    memory_text: str
    memory_item_ids: tuple[str, ...]
    examples: tuple[Example, ...]
    examples_text: str
    closing: ClosingHint | None
    asks_if_ai: bool


@dataclass(frozen=True)
class Bubble:
    """One bubble of the reply: a line of text, or a sticker resolved from a tag."""

    kind: BubbleKind
    text: str  # the line as sent; for a sticker ``[表情包:<标签>]``
    sticker_md5: str | None = None
    sticker_tag: str | None = None  # the tag that was asked for
    quote: str | None = None  # a quoted fragment that goes with this (the first) bubble

    @property
    def is_sticker(self) -> bool:
        return self.kind == "sticker"


@dataclass(frozen=True)
class PostAction:
    """One thing the post-processing did.  Never carries text of the reply (R-ENG-011)."""

    step: str
    count: int = 1
    detail: str | None = None

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"step": self.step, "count": self.count}
        if self.detail is not None:
            data["detail"] = self.detail
        return data


@dataclass(frozen=True)
class Violation:
    """A rule the output broke.  ``kind`` is a closed code; ``detail`` never has reply text."""

    kind: str
    detail: str | None = None

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind}
        if self.detail is not None:
            data["detail"] = self.detail
        return data


@dataclass(frozen=True)
class UsageSummary:
    """Tokens of all the model calls that made a reply."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0

    @property
    def cache_hit_ratio(self) -> float:
        prompt = self.cache_hit_tokens + self.cache_miss_tokens
        return self.cache_hit_tokens / prompt if prompt else 0.0

    def plus(self, other: UsageSummary) -> UsageSummary:
        return UsageSummary(
            self.calls + other.calls,
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.cache_hit_tokens + other.cache_hit_tokens,
            self.cache_miss_tokens + other.cache_miss_tokens,
        )


@dataclass(frozen=True)
class ReplyDraft:
    """The pipeline's answer for one round (see the module description)."""

    bubbles: tuple[Bubble, ...]
    quote: str | None  # the fragment of the user's message to quote with the first text bubble
    no_reply: bool
    needs_fallback: bool
    fallback_reason: FallbackReason | None
    backend: str  # the backend that produced the final output ("deepseek", "style", "hybrid")
    thinking: bool
    reasoning: str | None  # shown only when the user switched ``/显示思考`` on (R-LLM-002)
    plan: dict[str, Any] | None
    cost_usd: float
    usage: UsageSummary
    timings_ms: dict[str, int]
    actions: tuple[PostAction, ...]
    violations: tuple[Violation, ...]
    attempts: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """Something can be sent (or deliberately nothing is): no fallback is needed."""
        return not self.needs_fallback

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(bubble.text for bubble in self.bubbles)

    def actions_json(self) -> tuple[dict[str, Any], ...]:
        return tuple(action.to_json() for action in self.actions)

    def to_meta(self, backend: str | None = None) -> ReplyMeta:
        """The reply-level numbers as ``bot_turns`` stores them (R-ENG-011).

        ``backend`` overrides the backend name (the engine writes ``fallback`` for the short
        natural answer it sends when the pipeline asked for the fallback).
        """
        return ReplyMeta(
            backend=backend or self.backend,
            thinking=self.thinking,
            plan=self.plan,
            cost_usd=self.cost_usd,
            timings_ms=dict(self.timings_ms),
            actions=self.actions_json(),
        )
