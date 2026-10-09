"""The reply pipeline: one round in, one :class:`ReplyDraft` out (R-ENG-005 to R-ENG-013).

::

    draft = await pipeline.run(context, data_view)

``run`` is the whole of "producing a reply" and nothing else: it gathers the material from the
data view, asks a backend for the raw text, post-processes it and resolves the stickers.  It does
not wait, send, write ``bot_turns`` or touch the conversation state - the live engine
(round 09 step 3) does that around it - and it reads the world only through the
:class:`~twin.engine.dataview.ReplyDataView` it is given.  The evaluation sandbox (round 09b) calls
the **same** function with a data view of a past moment; that is what makes a reply measured there
the reply the bot would have given.

The attempts, when the output breaks a hard rule (R-ENG-008):

1. the chosen backend writes the reply;
2. a violation (calling herself an AI, a redaction token, only event text, a promise she cannot
   keep, a ``<think>`` block, not Chinese, nothing at all) -> the same backend again, told what
   was wrong;
3. still a violation -> DeepSeek without thinking, told the same (skipped when attempt 2 already
   was exactly that);
4. still a violation, or the backend failed, or the model refused -> ``needs_fallback``: the engine
   sends a late, short, natural answer and raises an alert (R-ENG-010).  A refusal is never shown
   and never retried at once (R-SAFE-005).

The last attempt cuts a promise out of the reply instead of failing on it, when other bubbles
remain.  A backend that raises ends the sequence at once (timeouts are retried later, minutes
apart, by the engine - not in a tight loop here); only a backend other than DeepSeek is replaced by
DeepSeek for the next attempt, so a style model that is down does not stop the conversation.
"""

from __future__ import annotations

import asyncio
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from twin.clock import Clock
from twin.config.lists import locate_list_file
from twin.engine.backend import BackendRequest, BackendResult, ReplyBackend
from twin.engine.dataview import ReplyDataView
from twin.engine.deepseek_backend import DeepSeekBackend
from twin.engine.postprocess import PostContext, PostProcessor, PostResult, StyleLimits
from twin.engine.postprocess.phrases import AiPhrases, load_ai_phrases
from twin.engine.prompt import VIOLATION_NOTES, PromptBuilder
from twin.engine.safety.commitments import CommitmentDetector
from twin.engine.safety.identity import asks_if_ai
from twin.engine.types import (
    ClosingHint,
    FallbackReason,
    PostAction,
    ReplyContext,
    ReplyDraft,
    ReplyMaterial,
    UsageSummary,
    Violation,
)
from twin.llm.budget import BudgetLimits
from twin.llm.types import ChatMessage
from twin.memory.asof import LocalMoment
from twin.memory.blocks import MemoryBlock, MemoryQuery
from twin.memory.records import LifelineRecord
from twin.ops.logging import get_logger
from twin.profile.api import ProfileSnapshot
from twin.profile.closing import is_closing_message
from twin.profile.persona.render import RenderedPersona
from twin.retrieval.examples import Example, render_examples
from twin.retrieval.query import QueryTurn
from twin.stickers.emoji_codes import EmojiCodePolicy
from twin.stickers.tags import TagVocabulary, load_vocabulary

if TYPE_CHECKING:
    from twin.llm.runtime import LlmRuntime
    from twin.services import Services

log = get_logger("twin.engine.pipeline")

FALLBACK_BACKEND = "deepseek"
TOPIC_TURNS = 2  # turns before the user's messages that the memory search also reads
STICKER_CONTEXT_TURNS = 3
_QUOTE_PREFIX = re.compile(r"^(?:\[引用:[^\n]*\]\n)+")


def _own_words(text: str) -> str:
    """The user's messages without the quote lines in front."""
    return _QUOTE_PREFIX.sub("", text)


@dataclass(frozen=True)
class _Read:
    """The plain answers of a data view for one round."""

    persona: RenderedPersona | None
    lifeline: tuple[LifelineRecord, ...]
    state: str | None
    local: LocalMoment
    profile: ProfileSnapshot | None
    emoji: EmojiCodePolicy | None


@dataclass
class _Run:
    """The bookkeeping of one :meth:`ReplyPipeline.run`."""

    cost: float = 0.0
    usage: UsageSummary = field(default_factory=UsageSummary)
    violations: list[Violation] = field(default_factory=list)
    attempt_kinds: list[list[str]] = field(default_factory=list)
    actions: list[PostAction] = field(default_factory=list)
    generate_ms: int = 0
    post_ms: int = 0
    attempts: int = 0
    fallback_used: bool = False

    def note_violations(self, found: Sequence[Violation]) -> None:
        """Remember what went wrong, each kind once (the per-attempt lists keep the history)."""
        known = {violation.kind for violation in self.violations}
        self.violations.extend(v for v in found if v.kind not in known)
        self.attempt_kinds.append([violation.kind for violation in found])

    def add(self, result: BackendResult) -> None:
        self.cost += result.cost_usd
        self.usage = self.usage.plus(result.usage)
        self.generate_ms += result.latency_ms


class ReplyPipeline:
    """Produces the reply to one round from a data view (see the module description)."""

    def __init__(
        self,
        *,
        backends: Mapping[str, ReplyBackend],
        clock: Clock,
        ai_phrases: AiPhrases,
        commitments: CommitmentDetector | None,
        vocabulary: TagVocabulary | None = None,
        bubble_cap: int = 8,
        examples_k: int = 8,
        context_turns: int = 6,
        memory_tokens: int = 800,
        limits: Callable[[], BudgetLimits] | None = None,
        processor: PostProcessor | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if FALLBACK_BACKEND not in backends:
            raise ValueError(f"the pipeline needs the {FALLBACK_BACKEND!r} backend to fall back on")
        self._backends = dict(backends)
        self._clock = clock
        self._ai_phrases = ai_phrases
        self._commitments = commitments
        self._vocabulary = vocabulary
        self._bubble_cap = bubble_cap
        self._examples_k = examples_k
        self._context_turns = context_turns
        self._memory_tokens = memory_tokens
        self._limits = limits
        self._processor = processor or PostProcessor()
        self._rng = rng or random.Random()  # noqa: S311 - sticker draws, not security

    @classmethod
    def from_services(
        cls,
        services: Services,
        runtime: LlmRuntime,
        *,
        extra_backends: Mapping[str, ReplyBackend] | None = None,
        rng: random.Random | None = None,
    ) -> ReplyPipeline:
        """The pipeline of the running application: DeepSeek, and any further backend given."""
        settings = services.settings
        root = services.paths.root
        vocabulary = load_vocabulary(settings, root)
        builder = PromptBuilder.from_services(services, vocabulary.tags)
        deepseek = DeepSeekBackend(runtime.client, builder, auto_rules=settings.thinking.auto_rules)
        return cls(
            backends={deepseek.name: deepseek, **(extra_backends or {})},
            clock=services.clock,
            ai_phrases=load_ai_phrases(locate_list_file(root, settings.engine.ai_phrases_file)),
            commitments=CommitmentDetector.from_services(services),
            vocabulary=vocabulary,
            bubble_cap=settings.engine.max_bubbles,
            examples_k=settings.engine.examples_k,
            context_turns=settings.retrieval.context_turns,
            memory_tokens=settings.memory.block_tokens,
            limits=runtime.budget.limits,
            rng=rng,
        )

    def has_backend(self, name: str) -> bool:
        return name in self._backends

    # ------------------------------------------------------------------------- run

    async def run(self, context: ReplyContext, data: ReplyDataView) -> ReplyDraft:
        """Produce the reply to ``context`` (see the module description)."""
        started = self._clock.monotonic()
        state = _Run()
        material, gathering = await self._gather(context, data)
        state.actions.extend(gathering)
        gathered = self._clock.monotonic()
        rng = random.Random(context.seed) if context.seed is not None else self._rng  # noqa: S311
        profile = await asyncio.to_thread(lambda: data.profile)
        style = StyleLimits.from_profile(profile, bubble_cap=self._bubble_cap)

        backend_name = context.backend if context.backend in self._backends else FALLBACK_BACKEND
        if backend_name != context.backend:
            state.actions.append(PostAction("backend_unavailable", detail=context.backend))
        primary = self._backends[backend_name]
        deepseek = self._backends[FALLBACK_BACKEND]
        # (backend, no thinking): the chosen backend twice, then DeepSeek without thinking
        sequence: list[tuple[ReplyBackend, bool]] = [(primary, False), (primary, False)]
        sequence.append((deepseek, True))
        notes: tuple[str, ...] = ()
        last: tuple[BackendResult, PostResult] | None = None
        reason: FallbackReason = "violations"
        position = 0
        while position < len(sequence):
            backend, non_thinking = sequence[position]
            position += 1
            state.attempts = position
            request = BackendRequest(context, data, material, notes, position, non_thinking)
            try:
                result = await backend.generate(request)
            except Exception as exc:  # a failed call ends the sequence; the engine retries later
                log.warning("backend_failed", backend=backend.name, reason=type(exc).__name__)
                state.actions.append(PostAction("backend_error", detail=type(exc).__name__))
                if backend is not deepseek:
                    sequence = [*sequence[:position], (deepseek, True)]
                    continue
                reason = "backend_error"
                break
            state.add(result)
            if result.refused:
                reason = "refused"
                last = (result, PostResult((), None, False, (), ()))
                break
            if result.skip_reply:
                state.actions.append(PostAction("plan_no_reply", detail=result.skip_reason))
                return self._finish(
                    state, result, PostResult((), None, True, (), ()), started, gathered
                )
            if position == 2 and backend is deepseek and not result.thinking:
                sequence = sequence[:2]  # the third attempt would be this very request again
            post_started = self._clock.monotonic()
            final = position == len(sequence)
            post = await asyncio.to_thread(
                self._process, result.text, context, data, material, style, rng, final
            )
            state.post_ms += round((self._clock.monotonic() - post_started) * 1000)
            last = (result, post)
            if post.ok:
                return self._finish(state, result, post, started, gathered)
            state.note_violations(post.violations)
            notes = self._notes(post.violations)
            log.info("reply_violations", attempt=position, kinds=state.attempt_kinds[-1])
        return self._failed(context, state, last, reason, started, gathered)

    async def preview(self, context: ReplyContext, data: ReplyDataView) -> list[ChatMessage]:
        """The prompt the DeepSeek backend would send for this round (no call is made).

        The evaluation prices a one-time batch from it (R-LLM-014): the very prompt the run uses.
        """
        material, _ = await self._gather(context, data)
        deepseek = self._backends[FALLBACK_BACKEND]
        if not isinstance(deepseek, DeepSeekBackend):
            raise TypeError("the fallback backend of the pipeline is not the DeepSeek backend")
        return deepseek.preview(BackendRequest(context, data, material))

    # --------------------------------------------------------------------- material

    async def _gather(
        self, context: ReplyContext, data: ReplyDataView
    ) -> tuple[ReplyMaterial, list[PostAction]]:
        actions: list[PostAction] = []
        limits = self._limits() if self._limits is not None else None
        k = limits.examples_k if limits is not None else self._examples_k
        factor = limits.memory_budget_factor if limits is not None else 1.0
        history = context.history[-self._context_turns :]
        topic = "\n".join(
            [*(turn.text for turn in context.history[-TOPIC_TURNS:]), context.user_text]
        )
        turns = [QueryTurn(her=turn.role == "bot", text=turn.text) for turn in history]
        turns.append(QueryTurn(her=False, text=context.user_text))

        def recall() -> MemoryBlock:
            return data.memory_block(MemoryQuery(text=topic), int(self._memory_tokens * factor))

        async def examples() -> list[Example]:
            return await data.examples(turns, k=k) if k > 0 else []

        found = await asyncio.gather(asyncio.to_thread(recall), examples(), return_exceptions=True)
        memory = found[0]
        if isinstance(memory, BaseException):
            log.warning("memory_unavailable", reason=type(memory).__name__)
            actions.append(PostAction("memory_unavailable", detail=type(memory).__name__))
            memory = MemoryBlock("")
        retrieved = found[1]
        if isinstance(retrieved, BaseException):
            log.warning("examples_unavailable", reason=type(retrieved).__name__)
            actions.append(PostAction("examples_unavailable", detail=type(retrieved).__name__))
            retrieved = []

        def read() -> _Read:
            """What the data view answers without a search (database reads, kept off the loop)."""
            return _Read(
                data.persona_full(),
                tuple(data.lifeline),
                data.her_state(),
                data.local,
                data.profile,
                data.emoji_codes,  # read now, so the backend finds it already loaded
            )

        plain = await asyncio.to_thread(read)
        persona, lifeline = plain.persona, plain.lifeline
        last = context.last_inbound
        closing: ClosingHint | None = None
        if is_closing_message(last.kind, _own_words(last.text)):
            profile = plain.profile
            rate = profile.metrics.scalar("her", "closing_no_reply_rate") if profile else None
            closing = ClosingHint(rate or 0.0)
        material = ReplyMaterial(
            local=plain.local,
            state=plain.state,
            lifeline=lifeline[0].line() if lifeline else None,
            persona=persona.text if persona is not None else "",
            persona_ref=f"{persona.scope} v{persona.number}" if persona is not None else None,
            memory_text=memory.text,
            memory_item_ids=tuple(item.item_id for item in memory.items),
            examples=tuple(retrieved),
            examples_text=render_examples(retrieved) if retrieved else "",
            closing=closing,
            asks_if_ai=any(
                asks_if_ai(item.text) for item in context.inbound if item.kind == "text"
            ),
        )
        return material, actions

    # ---------------------------------------------------------------- post-processing

    def _process(
        self,
        raw: str,
        context: ReplyContext,
        data: ReplyDataView,
        material: ReplyMaterial,
        style: StyleLimits,
        rng: random.Random,
        final: bool,
    ) -> PostResult:
        """Post-process one output (runs in a worker thread: it reads the sticker library)."""
        selector = data.sticker_selector(rng)
        closing = material.closing
        tail = [turn.text for turn in context.history[-STICKER_CONTEXT_TURNS:]]
        post_context = PostContext(
            style=style,
            emoji=data.emoji_codes,
            ai_phrases=self._ai_phrases,
            commitments=self._commitments,
            rng=rng,
            allow_ai_admission=material.asks_if_ai,
            user_text=context.user_text,
            supports_quote=context.limits.supports_quote,
            quota=context.limits.max_bubbles,
            no_reply_allowed=closing is not None and closing.no_reply_rate > 0,
            known_tags=self._vocabulary.tags if self._vocabulary is not None else (),
            chooser=selector.choose,
            rate=data.sticker_rate(),
            sticker_context="\n".join([*tail, context.user_text]),
            recent_stickers=context.recent_stickers,
            last_attempt=final,
        )
        return self._processor.process(raw, post_context)

    @staticmethod
    def _notes(violations: Sequence[Violation]) -> tuple[str, ...]:
        return tuple(VIOLATION_NOTES[v.kind] for v in violations if v.kind in VIOLATION_NOTES)

    # ----------------------------------------------------------------------- results

    def _timings(self, state: _Run, started: float, gathered: float) -> dict[str, int]:
        total = round((self._clock.monotonic() - started) * 1000)
        return {
            "gather": round((gathered - started) * 1000),
            "generate": state.generate_ms,
            "post": state.post_ms,
            "total": total,
        }

    def _finish(
        self,
        state: _Run,
        result: BackendResult,
        post: PostResult,
        started: float,
        gathered: float,
    ) -> ReplyDraft:
        actions = [*state.actions, *post.actions]
        if state.attempts > 1:
            actions.append(PostAction("regenerated", state.attempts - 1))
        meta: dict[str, Any] = {**result.meta, "attempts": state.attempts}
        if state.attempt_kinds:
            meta["attempt_violations"] = state.attempt_kinds
        return ReplyDraft(
            bubbles=post.bubbles,
            quote=post.quote,
            no_reply=post.no_reply,
            needs_fallback=False,
            fallback_reason=None,
            backend=result.backend,
            thinking=result.thinking,
            reasoning=result.reasoning,
            plan=result.plan,
            cost_usd=state.cost,
            usage=state.usage,
            timings_ms=self._timings(state, started, gathered),
            actions=tuple(actions),
            violations=tuple(state.violations),
            attempts=state.attempts,
            meta=meta,
        )

    def _failed(
        self,
        context: ReplyContext,
        state: _Run,
        last: tuple[BackendResult, PostResult] | None,
        reason: FallbackReason,
        started: float,
        gathered: float,
    ) -> ReplyDraft:
        result = last[0] if last is not None else None
        actions = [*state.actions, *(last[1].actions if last is not None else ())]
        return ReplyDraft(
            bubbles=(),
            quote=None,
            no_reply=False,
            needs_fallback=True,
            fallback_reason=reason,
            backend=result.backend if result is not None else context.backend,
            thinking=result.thinking if result is not None else False,
            reasoning=None,
            plan=result.plan if result is not None else None,
            cost_usd=state.cost,
            usage=state.usage,
            timings_ms=self._timings(state, started, gathered),
            actions=tuple(actions),
            violations=tuple(state.violations),
            attempts=state.attempts,
            meta={"attempt_violations": state.attempt_kinds},
        )
