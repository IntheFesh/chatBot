"""The ``hybrid`` backend: DeepSeek plans the reply, the style model says it (R-ENG-006, R-TRN-005).

Step one - **the plan**.  DeepSeek reads what the DeepSeek backend reads (the full persona card, the
recent conversation, the context of the moment) and answers with a JSON plan, validated with
pydantic (R-LLM-003: a reply that does not validate is sent back once with the error)::

    {"reply": true, "intent": "...", "facts_to_use": ["..."], "tone": "...",
     "bubble_hint": "...", "sticker_hint": "..."}

Thinking, when the round asks for it (``/思考``), happens in this step and nowhere else; the budget
may switch it off.  The planner is part of replying to the user, so its calls are booked as
``reply`` - they are never refused by the budget (R-LLM-008) and a lowered budget level switches
their thinking off like it does for the DeepSeek backend.

Step two - **the reply**.  The plan goes into the ``【规划】`` field of the style model's prompt
(:class:`~twin.engine.style_prompt.StylePromptBuilder`, the same field the training set fills for
its planned samples) and the style model writes the bubbles in her voice
(:class:`~twin.engine.style_backend.StyleWriter`).

``reply: false`` means "do not answer this": the backend returns ``skip_reply`` and the pipeline
sends nothing.  Silence is only allowed where the output convention allows it (R-ENG-007): the
message is a closing one and her history has such silences.  A plan that says ``false`` anywhere
else is ignored - a bot that ghosts a question is worse than one that answers a closing "好的" -
and the backend records that in ``meta["plan_no_reply_ignored"]``.

A planner that cannot give a valid plan (twice) does not stop the conversation: the style model
answers without a plan.  A planner that is unreachable raises; the pipeline then asks DeepSeek for
the whole reply.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from twin.engine.backend import BackendRequest, BackendResult
from twin.engine.deepseek_backend import usage_of
from twin.engine.prompt import NO_PERSONA, lay_out, render_context
from twin.engine.refusal import FILTER_FINISHES
from twin.engine.style_backend import StyleWriter
from twin.engine.style_prompt import PlanFields
from twin.engine.thinking import resolve_thinking
from twin.engine.types import ReplyContext, ReplyMaterial, UsageSummary
from twin.llm.deepseek import DeepSeekClient
from twin.llm.errors import InvalidRequestError, StructuredOutputError
from twin.llm.layout import PromptLayout
from twin.llm.types import Purpose
from twin.ops.logging import get_logger
from twin.profile.prompt_templates import PromptText, TemplateStore

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.engine.hybrid_backend")

TEMPLATE = "reply_plan"
PLAN_TEMPERATURE = 0.7
PLAN_MAX_TOKENS = 600
PLAN_MAX_TOKENS_THINKING = 4000
NO_REPLY_PLANNED = "no_reply_planned"


def _text(value: object) -> str:
    """A field of the plan as text: what the model wrote, or nothing."""
    if value is None:
        return ""
    return str(value).strip()


class HybridPlan(BaseModel):
    """The planner's JSON (R-ENG-006); fields other than ``reply`` may be missing or loose."""

    model_config = ConfigDict(extra="ignore")

    reply: bool
    intent: str = ""
    facts_to_use: list[str] = Field(default_factory=list)
    tone: str = ""
    bubble_hint: str = ""
    sticker_hint: str = ""

    @field_validator("intent", "tone", "bubble_hint", "sticker_hint", mode="before")
    @classmethod
    def _as_text(cls, value: object) -> str:
        return _text(value)

    @field_validator("facts_to_use", mode="before")
    @classmethod
    def _as_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        items = value if isinstance(value, list | tuple) else [value]
        return [text for text in (_text(item) for item in items) if text]

    def fields(self) -> PlanFields:
        """The fields the style model's prompt carries."""
        return PlanFields(
            intent=self.intent,
            facts_to_use=tuple(self.facts_to_use),
            tone=self.tone,
            bubble_hint=self.bubble_hint,
            sticker_hint=self.sticker_hint,
        )


class PlanPromptBuilder:
    """The request of the planner: the DeepSeek prompt layout with the planning rules."""

    def __init__(self, template: PromptText, *, sticker_tags: Sequence[str]) -> None:
        self._template = template
        self._tags = tuple(sticker_tags)

    @classmethod
    def from_services(cls, services: Services, sticker_tags: Sequence[str]) -> PlanPromptBuilder:
        return cls(
            TemplateStore(services.db, services.clock).active(TEMPLATE), sticker_tags=sticker_tags
        )

    @property
    def template(self) -> PromptText:
        return self._template

    def build(
        self, context: ReplyContext, material: ReplyMaterial, notes: Sequence[str] = ()
    ) -> PromptLayout:
        rendered = self._template.render(
            persona=material.persona.strip() or NO_PERSONA,
            sticker_tags="、".join(self._tags) or "（没有可用的表情包标签，留空）",
            context=render_context(context, material, notes),
            message=context.user_text,
        )
        return lay_out(rendered, context)


class HybridBackend:
    """DeepSeek plans, the style model writes (see the module description)."""

    name = "hybrid"

    def __init__(
        self,
        writer: StyleWriter,
        client: DeepSeekClient,
        planner: PlanPromptBuilder,
        *,
        thinking_allowed: Callable[[], bool] = lambda: True,
        auto_rules: bool = True,
    ) -> None:
        self._writer = writer
        self._client = client
        self._planner = planner
        self._thinking_allowed = thinking_allowed
        self._auto_rules = auto_rules

    @property
    def writer(self) -> StyleWriter:
        """The style model's writer, for what writes from a plan of its own (round 10)."""
        return self._writer

    async def generate(self, request: BackendRequest) -> BackendResult:
        context, material = request.context, request.material
        wanted = (
            not request.non_thinking
            and self._thinking_allowed()
            and resolve_thinking(
                context.thinking_mode, context.user_text, auto_rules=self._auto_rules
            )
        )
        layout = self._planner.build(context, material, request.notes)
        try:
            planned = await self._client.chat_json(
                layout,
                HybridPlan,
                purpose=Purpose.REPLY,
                thinking=wanted,
                temperature=None if wanted else PLAN_TEMPERATURE,
                max_tokens=PLAN_MAX_TOKENS_THINKING if wanted else PLAN_MAX_TOKENS,
            )
        except StructuredOutputError:
            log.warning("plan_invalid", attempts=2)
            return await self._write(
                request, None, 0.0, UsageSummary(), 0, False, None, {"plan_failed": True}
            )
        except InvalidRequestError as exc:
            if "risk" not in str(exc).lower():
                raise
            log.warning("generation_refused", kind="content_risk")
            return BackendResult("", self.name, wanted, 0.0, UsageSummary(), 0, refused=True)
        chat = planned.chat
        if chat.finish_reason in FILTER_FINISHES:
            log.warning("generation_refused", kind=chat.finish_reason)
            return BackendResult(
                "",
                self.name,
                chat.thinking,
                planned.total_cost_usd,
                usage_of(chat),
                chat.latency_ms,
                refused=True,
            )
        plan = planned.value
        usage = dataclasses.replace(usage_of(chat), calls=planned.attempts)
        meta: dict[str, Any] = {
            "planner_model": chat.model,
            "planner_template": self._planner.template.ref,
            "planner_attempts": planned.attempts,
            "cache_hit_ratio": round(chat.usage.cache_hit_ratio, 3),
            "requested_thinking": wanted,
        }
        closing = material.closing
        silence_allowed = closing is not None and closing.no_reply_rate > 0
        if not plan.reply:
            if silence_allowed:
                return BackendResult(
                    text="",
                    backend=self.name,
                    thinking=chat.thinking,
                    cost_usd=planned.total_cost_usd,
                    usage=usage,
                    latency_ms=chat.latency_ms,
                    reasoning=chat.reasoning_content,
                    plan=plan.model_dump(),
                    skip_reply=True,
                    skip_reason=NO_REPLY_PLANNED,
                    meta=meta,
                )
            meta["plan_no_reply_ignored"] = True
        return await self._write(
            request,
            plan,
            planned.total_cost_usd,
            usage,
            chat.latency_ms,
            chat.thinking,
            chat.reasoning_content,
            meta,
        )

    async def _write(
        self,
        request: BackendRequest,
        plan: HybridPlan | None,
        planner_cost: float,
        planner_usage: UsageSummary,
        planner_ms: int,
        thinking: bool,
        reasoning: str | None,
        meta: dict[str, Any],
    ) -> BackendResult:
        written = await self._writer.write(request, plan.fields() if plan is not None else None)
        return BackendResult(
            text=written.text,
            backend=self.name,
            thinking=thinking,
            cost_usd=planner_cost,
            usage=planner_usage.plus(written.usage),
            latency_ms=planner_ms + written.latency_ms,
            reasoning=reasoning,
            plan=plan.model_dump() if plan is not None else None,
            meta={**written.meta, **meta},
        )
