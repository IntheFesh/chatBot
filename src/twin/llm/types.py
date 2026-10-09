"""Shared types of the LLM layer: purposes, messages, usage and results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal, NotRequired, TypedDict


class Purpose(StrEnum):
    """Why a call is made (the ``purpose`` label of ``cost_ledger``, R-LLM-006).

    ``probe`` is added to the SPEC list for the M0 probe (R-LLM-013), whose cost is recorded
    on the one-time account.
    """

    REPLY = "reply"
    PLAN = "plan"
    PROACTIVE = "proactive"
    EXTRACT = "extract"
    SUMMARY = "summary"
    PERSONA = "persona"
    CAPTION = "caption"
    STICKER_TAG = "sticker_tag"
    EVAL = "eval"
    TRAIN_PLAN = "train_plan"
    PROBE = "probe"


# purposes that run the model offline on the "offline_model" (R-LLM-001, SPEC deepseek.*)
OFFLINE_PURPOSES = frozenset(
    {
        Purpose.EXTRACT,
        Purpose.SUMMARY,
        Purpose.PERSONA,
        Purpose.STICKER_TAG,
        Purpose.EVAL,
        Purpose.TRAIN_PLAN,
    }
)
# purposes that belong to the live conversation path
LIVE_PURPOSES = frozenset({Purpose.REPLY, Purpose.PLAN, Purpose.PROACTIVE})

Account = Literal["daily", "one_time"]
Role = Literal["system", "user", "assistant"]


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class ImageUrl(TypedDict):
    url: str
    detail: NotRequired[str]


class ImagePart(TypedDict):
    type: Literal["image_url"]
    image_url: ImageUrl


ContentPart = TextPart | ImagePart


class ChatMessage(TypedDict):
    """A message as sent to the API.  There is deliberately no ``reasoning_content`` field."""

    role: Role
    content: str | list[ContentPart]


@dataclass(frozen=True)
class Usage:
    """Token counts of one call (``completion_tokens`` includes reasoning tokens)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cache_hit_ratio(self) -> float:
        """Share of the prompt that came from the cache (0 when there was no prompt)."""
        prompt = self.cache_hit_tokens + self.cache_miss_tokens
        return self.cache_hit_tokens / prompt if prompt else 0.0


@dataclass(frozen=True)
class CostBreakdown:
    """Cost of one call in USD, split like the price table."""

    cache_hit_usd: float
    cache_miss_usd: float
    output_usd: float
    peak: bool
    multiplier: float

    @property
    def total_usd(self) -> float:
        return self.cache_hit_usd + self.cache_miss_usd + self.output_usd


@dataclass(frozen=True)
class LedgerTag:
    """Which account a call counts against (R-LLM-014)."""

    account: Account = "daily"
    batch_id: str | None = None

    def __post_init__(self) -> None:
        if self.account == "one_time" and not self.batch_id:
            raise ValueError("a one_time call needs a batch id")
        if self.account == "daily" and self.batch_id is not None:
            raise ValueError("only one_time calls carry a batch id")


DAILY = LedgerTag()


@dataclass(frozen=True)
class ChatResult:
    """What :meth:`DeepSeekClient.chat` returns (R-LLM-001)."""

    content: str
    reasoning_content: str | None
    usage: Usage
    cost: CostBreakdown
    latency_ms: int
    model: str
    purpose: Purpose
    thinking: bool
    at: datetime
    request_id: str | None = None
    finish_reason: str | None = None
    attempts: int = 1
    ledger_id: str | None = None
    image_count: int = 0
    reasoning_returned: bool = False  # the API sent reasoning content, whatever was asked for

    @property
    def cost_usd(self) -> float:
        return self.cost.total_usd


@dataclass(frozen=True)
class JsonResult[T]:
    """A validated structured reply (R-LLM-003)."""

    value: T
    chat: ChatResult  # the call that produced ``value``
    attempts: int = 1
    total_cost_usd: float = 0.0  # all attempts together
