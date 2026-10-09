"""The ``style`` backend: the fine-tuned model writes her reply (R-ENG-006, R-LLM-011, R-TRN-011).

:class:`StyleWriter` is the part every style-model call shares: it reads the data view and the
conversation of the round, renders the prompt with
:class:`~twin.engine.style_prompt.StylePromptBuilder` (the registered model's locked card and
template, R-SRV-001), sends **only that string** to the
:class:`~twin.llm.style_client.StyleModelClient` and hands back the raw text.  The ``style``
backend calls it directly; the ``hybrid`` backend calls it with the plan its DeepSeek planner made.

Thinking.  The style model does not think; the thinking mode of the round (``/思考``) decides
whether a planning step comes first.  When it asks for thinking, the ``style`` backend hands the
round to the hybrid backend - the thinking then happens in the DeepSeek planner (R-ENG-006) - unless
the budget has switched chat thinking off.  The result then says ``hybrid``.

A ``<think>`` block in the output is a hard violation (R-TRN-011.6, R-ENG-008).  The backend does
not remove it itself: the text goes on to the post-processing of the pipeline, whose first step
deletes the block and records the violation (``think_tag``) - one implementation for every backend,
and the same step that makes the pipeline write again with a note.  The backend only counts the
blocks in ``meta["think_blocks"]``.

Everything that goes wrong - no active model, a model that is bound to a template this program
cannot render, a server that is down, slow or returns nonsense - is raised as
:class:`~twin.llm.errors.StyleModelError`; the pipeline then answers the round with DeepSeek and
the backend selector sees the failure (:mod:`twin.engine.backend_select`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from twin.config.settings import StyleModelConfig
from twin.engine.backend import BackendRequest, BackendResult, ReplyBackend
from twin.engine.style_models import StyleModels
from twin.engine.style_prompt import (
    LockedVersionError,
    PlanFields,
    PromptParts,
    StylePromptBuilder,
    StylePromptError,
    StyleTurn,
    TokenBudget,
    style_turns,
)
from twin.engine.thinking import resolve_thinking
from twin.engine.types import ReplyContext, UsageSummary
from twin.llm.errors import StyleModelError
from twin.llm.style_client import StyleModelClient, StyleParams
from twin.memory.blocks import MemoryBlock
from twin.ops.logging import get_logger
from twin.training.profiles import CUTOFF_LEN
from twin.training.registry import LockedVersions

log = get_logger("twin.engine.style_backend")

REPLY_RESERVE_TOKENS = 256
PROMPT_TOKEN_BUDGET = CUTOFF_LEN - REPLY_RESERVE_TOKENS
"""What a prompt may use: the training cut-off (2048) less room for her reply, so that the prompt
the bot sends is one the model could have been trained on (the export uses the same cut-off)."""
DEFAULT_BUDGET = TokenBudget(PROMPT_TOKEN_BUDGET)
SEED_STEP = 7919
SEED_MODULUS = 2**31


@dataclass(frozen=True)
class StyleSampling:
    """The sampling settings of one generation (``style_model.*``)."""

    n_predict: int = 200
    temperature: float = 0.7
    top_p: float = 0.9

    @classmethod
    def from_config(cls, config: StyleModelConfig) -> StyleSampling:
        return cls(config.n_predict, config.temperature, config.top_p)


@dataclass(frozen=True)
class StyleWrite:
    """What one call of the style model produced."""

    text: str
    usage: UsageSummary
    latency_ms: int
    meta: dict[str, Any]


BuilderFactory = Callable[[LockedVersions], StylePromptBuilder]


def conversation_of(context: ReplyContext) -> list[StyleTurn]:
    """The conversation of the round: the history, then the messages being answered."""
    turns = style_turns(context.history)
    turns.extend(StyleTurn("user", item.text) for item in context.inbound)
    return turns


class StyleWriter:
    """Renders the prompt of a round and asks the style model (see the module description)."""

    def __init__(
        self,
        *,
        client: StyleModelClient,
        models: StyleModels,
        builders: BuilderFactory,
        sampling: StyleSampling | None = None,
        budget: TokenBudget | None = DEFAULT_BUDGET,
    ) -> None:
        self._client = client
        self._models = models
        self._builders = builders
        self._sampling = sampling or StyleSampling()
        self._budget = budget

    async def write(self, request: BackendRequest, plan: PlanFields | None = None) -> StyleWrite:
        """One reply of the style model; raises :class:`StyleModelError` when it cannot."""
        active = await asyncio.to_thread(self._models.active)
        if active is None:
            raise StyleModelError("no style model is active", kind="unavailable")
        try:
            builder = self._builders(active.versions)
        except LockedVersionError as exc:
            raise StyleModelError(str(exc), kind="locked") from exc
        context = request.context

        def render() -> tuple[PromptParts, bool]:
            turns = conversation_of(context)
            memory = MemoryBlock("")
            memory_failed = False
            try:
                memory = builder.memory_for(request.data, turns)
            except Exception as exc:  # the memory is optional: the reply goes on without it
                log.warning("memory_unavailable", reason=type(exc).__name__)
                memory_failed = True
            parts = builder.compose(
                request.data,
                turns,
                plan=plan,
                woke_up=context.woke_up,
                budget=self._budget,
                memory=memory,
            )
            return parts, memory_failed

        try:
            parts, memory_failed = await asyncio.to_thread(render)
        except LockedVersionError as exc:
            raise StyleModelError(str(exc), kind="locked") from exc
        except StylePromptError as exc:
            raise StyleModelError(str(exc), kind="prompt") from exc
        prompt = parts.render()
        seed = (
            (context.seed * SEED_STEP + request.attempt) % SEED_MODULUS
            if context.seed is not None
            else None
        )
        params = StyleParams(
            n_predict=self._sampling.n_predict,
            temperature=self._sampling.temperature,
            top_p=self._sampling.top_p,
            stop=prompt.stop,
            seed=seed,
        )
        output = await self._client.generate(prompt, params)
        text = output.text
        if output.truncated and "\n" in text.strip():
            text = text.rstrip().rsplit("\n", 1)[0]  # the last line was cut off mid-sentence
        meta = prompt.meta
        info: dict[str, Any] = {
            "template": prompt.template,
            "model": active.id,
            "persona": f"{meta.persona_scope} {meta.persona_version}"
            if meta is not None and meta.persona_version
            else None,
            "locked": meta.locked if meta is not None else False,
            "gate_passed": active.passed_gate,
            "context_turns": meta.context_turns if meta is not None else 0,
            "trimmed_turns": meta.trimmed_turns if meta is not None else 0,
            "over_budget": meta.over_budget if meta is not None else False,
            "stop_reason": output.stop_reason,
            "truncated": output.truncated,
            "think_blocks": text.lower().count("<think>"),
        }
        if memory_failed:
            info["memory_unavailable"] = True
        usage = UsageSummary(1, output.prompt_tokens or 0, output.completion_tokens or 0)
        return StyleWrite(text, usage, output.latency_ms, info)


class StyleBackend:
    """Generates the reply with the style model (see the module description)."""

    name = "style"

    def __init__(
        self,
        writer: StyleWriter,
        *,
        planner: ReplyBackend | None = None,
        thinking_allowed: Callable[[], bool] = lambda: True,
        auto_rules: bool = True,
    ) -> None:
        self._writer = writer
        self._planner = planner
        self._thinking_allowed = thinking_allowed
        self._auto_rules = auto_rules

    async def generate(self, request: BackendRequest) -> BackendResult:
        context = request.context
        wants_thinking = not request.non_thinking and resolve_thinking(
            context.thinking_mode, context.user_text, auto_rules=self._auto_rules
        )
        if wants_thinking and self._planner is not None and self._thinking_allowed():
            return await self._planner.generate(request)
        written = await self._writer.write(request)
        return BackendResult(
            text=written.text,
            backend=self.name,
            thinking=False,
            cost_usd=0.0,
            usage=written.usage,
            latency_ms=written.latency_ms,
            meta=written.meta,
        )
