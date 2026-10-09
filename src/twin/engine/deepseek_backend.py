"""The DeepSeek backend: the model writes the reply directly (R-ENG-006, R-LLM-002, R-LLM-008).

Prompt: :class:`~twin.engine.prompt.PromptBuilder` (fixed rules and persona, the window of recent
turns, one last message that carries everything that varies).  Thinking follows the runtime
setting ``thinking.chat``: ``on`` / ``off``, or ``auto`` by the rules of
:mod:`twin.engine.thinking`; the budget may switch it off (level 1 and above) and what was really
used is reported.  ``reasoning_content`` is returned to the caller for ``/显示思考`` and nowhere
else - the client drops it from every later request.

A model that refuses (finish reason ``content_filter``, HTTP 400 "Content Exists Risk", or a
reply that is only the refusal) is reported as ``refused`` without keeping the text: the pipeline
then falls back and the user never reads the model's refusal (R-SAFE-005).
"""

from __future__ import annotations

from twin.engine.backend import BackendRequest, BackendResult
from twin.engine.prompt import BuiltPrompt, PromptBuilder
from twin.engine.refusal import FILTER_FINISHES, looks_like_refusal
from twin.engine.thinking import resolve_thinking
from twin.engine.types import UsageSummary
from twin.llm.deepseek import DeepSeekClient
from twin.llm.errors import InvalidRequestError
from twin.llm.types import ChatMessage, ChatResult, Purpose
from twin.ops.logging import get_logger

log = get_logger("twin.engine.deepseek_backend")

TEMPERATURE = 1.0
MAX_TOKENS = 600
MAX_TOKENS_THINKING = 4000


def usage_of(result: ChatResult) -> UsageSummary:
    usage = result.usage
    return UsageSummary(
        1,
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.cache_hit_tokens,
        usage.cache_miss_tokens,
    )


class DeepSeekBackend:
    """Generates the reply with the DeepSeek chat model."""

    name = "deepseek"

    def __init__(
        self, client: DeepSeekClient, builder: PromptBuilder, *, auto_rules: bool = True
    ) -> None:
        self._client = client
        self._builder = builder
        self._auto_rules = auto_rules

    def _build(self, request: BackendRequest) -> BuiltPrompt:
        emoji = request.data.emoji_codes
        return self._builder.build(
            request.context,
            request.material,
            emoji_codes=emoji.codes if emoji is not None else (),
            notes=request.notes,
        )

    def preview(self, request: BackendRequest) -> list[ChatMessage]:
        """The messages :meth:`generate` would send for ``request``; nothing is sent.

        The evaluation prices a batch from it (R-LLM-014) with the very prompt the run will use.
        """
        return self._build(request).messages

    async def generate(self, request: BackendRequest) -> BackendResult:
        context = request.context
        built = self._build(request)
        wanted = not request.non_thinking and resolve_thinking(
            context.thinking_mode, context.user_text, auto_rules=self._auto_rules
        )
        try:
            result = await self._client.chat(
                built.layout,
                purpose=Purpose.REPLY,
                thinking=wanted,
                temperature=None if wanted else TEMPERATURE,
                max_tokens=MAX_TOKENS_THINKING if wanted else MAX_TOKENS,
            )
        except InvalidRequestError as exc:
            if "risk" not in str(exc).lower():
                raise
            log.warning("generation_refused", kind="content_risk")
            return BackendResult("", self.name, wanted, 0.0, UsageSummary(), 0, refused=True)
        refused = (
            result.finish_reason in FILTER_FINISHES
            or not result.content.strip()
            or looks_like_refusal(result.content)
        )
        if refused:
            log.warning("generation_refused", kind=result.finish_reason or "empty")
        meta = {
            **built.meta(),
            "model": result.model,
            "finish": result.finish_reason,
            "cache_hit_ratio": round(result.usage.cache_hit_ratio, 3),
            "requested_thinking": wanted,
        }
        return BackendResult(
            text="" if refused else result.content,
            backend=self.name,
            thinking=result.thinking,
            cost_usd=result.cost_usd,
            usage=usage_of(result),
            latency_ms=result.latency_ms,
            reasoning=result.reasoning_content,
            refused=refused,
            meta=meta,
        )
