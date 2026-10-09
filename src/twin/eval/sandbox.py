"""The evaluation sandbox: the running bot's reply, made for a past moment (R-EVAL-009).

::

    kit = build_sandbox(services, mode=SandboxMode.HOLDOUT, batch_id=batch)
    reply = await kit.sandbox.reply(SandboxRequest(inbound, history, at=t, backend="deepseek"))
    reply.candidate          # what reached the in-memory channel, in the shared shape
    await kit.aclose()

**One pipeline.**  :meth:`EvalSandbox.reply` calls ``ReplyPipeline.run(context, data_view)`` - the
function the running engine calls (round 09) - and nothing else produces a reply: the prompts, the
backends, the post-processing and the sticker choice are the bot's own.  What differs is only
what surrounds it.

*The data view.*  ``HOLDOUT`` mode hands the pipeline ``AsOfView(t)`` (round 07): the pre-holdout
persona card, profile and routine, the memory as it was known at ``t``, her real replies from
before ``t`` only, the sticker library as of ``t``, her typical state at ``t``, no life line and no
conversation with the bot (R-TRN-013).  ``LIVE`` mode (the memory test) hands it the live view -
the present and all the data - and is still isolated.  The conversation before the reply is the
context of the sample: **at most eight merged turns for every backend** (the style backend and the
training set use eight), so a comparison between backends is fair.

*The channel.*  The reply is delivered to an :class:`~twin.eval.channel.InMemoryChannel`
(text through ``send_text``, stickers through the engine's own
:class:`~twin.engine.sticker_sender.StickerSender`), immediately: no typing, no pauses, no pacing.
The candidate is what the channel accepted.  A sandbox never builds a WeChat channel.

*Writes.*  The whole run is inside :func:`~twin.eval.isolation.isolated_writes`: only the
evaluation tables, the cost ledger, jobs, alerts and the three settings the cost machinery needs
can be written.  The calls are booked as ``purpose = eval`` on the one-time batch of the run
(:func:`~twin.llm.scope.call_scope`), with the model of a real reply.

*Time.*  The pipeline reads the time only from the data view, so ``t`` is the moment of the reply;
the in-memory channel's clock reads ``t`` too (:class:`SampleClock`).
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from twin.channel.base import MediaNotAllowed, QuoteTarget
from twin.channel.policy import StickerAllowList
from twin.clock import Clock, ensure_aware
from twin.config.runtime import ThinkingMode
from twin.engine.dataview import LiveDataSource, ReplyDataView
from twin.engine.pipeline import ReplyPipeline
from twin.engine.sticker_sender import StickerSender
from twin.engine.style_models import StyleModels
from twin.engine.style_runtime import StyleRuntime
from twin.engine.types import InboundItem, ReplyContext, ReplyDraft, SendLimits
from twin.eval.channel import InMemoryChannel, SentMessage
from twin.eval.isolation import isolated_writes
from twin.eval.liveview import EvalLiveSource
from twin.eval.render import Candidate, candidate_from_bubbles
from twin.llm.runtime import LlmRuntime, build_llm_runtime
from twin.llm.scope import call_scope
from twin.llm.types import ChatMessage, LedgerTag, Purpose
from twin.memory.asof import AsOfSource
from twin.memory.recent import Turn
from twin.ops.logging import get_logger
from twin.stickers.catalog import StickerCatalog
from twin.stickers.library import StickerLibraryAllowList
from twin.training import lf_template

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.eval.sandbox")

BACKENDS = ("deepseek", "style", "hybrid")
STYLE_BACKENDS = ("style", "hybrid")
CONTEXT_TURNS = 8  # merged turns of the conversation every backend is given (R-EVAL-009)
SEED_MODULUS = 2**31


class SandboxMode(StrEnum):
    HOLDOUT = "holdout"  # a past moment, through AsOfView(t)
    LIVE = "live"  # now, the live data; still isolated


class SandboxError(RuntimeError):
    """The sandbox cannot be used the way it was asked."""


class SampleClock:
    """A clock that reads the moment of the sample; durations and sleeping stay real."""

    def __init__(self, real: Clock, moment: datetime | None = None) -> None:
        self._real = real
        self._moment = ensure_aware(moment) if moment is not None else None

    def set(self, moment: datetime) -> None:
        self._moment = ensure_aware(moment)

    def now_utc(self) -> datetime:
        return self._moment if self._moment is not None else self._real.now_utc()

    def monotonic(self) -> float:
        return self._real.monotonic()

    async def sleep(self, seconds: float) -> None:
        await self._real.sleep(seconds)


@dataclass(frozen=True)
class BackendStatus:
    """Whether a backend can be evaluated now, and what to tell the user if not."""

    name: str
    available: bool
    message: str = ""


def backend_status(services: Services, name: str) -> BackendStatus:
    """``style`` and ``hybrid`` need a registered, active model that this code can render."""
    if name == "deepseek":
        return BackendStatus(name, True)
    if name not in STYLE_BACKENDS:
        return BackendStatus(name, False, f"unknown backend {name!r}: use {', '.join(BACKENDS)}")
    models = StyleModels(services.db, mode=services.settings.style_model.mode)
    active = models.active()
    if active is None:
        return BackendStatus(
            name,
            False,
            f"the {name} backend is not deployed: no style model is registered and active "
            "(未部署; it becomes available after the training and deployment of round 14)",
        )
    if active.versions.template_version != lf_template.TEMPLATE_VERSION:
        return BackendStatus(
            name,
            False,
            f"the {name} backend is not deployed: the active model is bound to template "
            f"{active.versions.template_version}, this code renders {lf_template.TEMPLATE_VERSION}",
        )
    return BackendStatus(name, True)


@dataclass(frozen=True)
class SandboxRequest:
    """One round to answer for the evaluation."""

    inbound: tuple[InboundItem, ...]
    history: tuple[Turn, ...]
    at: datetime
    backend: str = "deepseek"
    thinking_mode: ThinkingMode = "off"
    seed: int | None = None


@dataclass(frozen=True)
class SandboxReply:
    """What the sandbox produced for one request."""

    draft: ReplyDraft
    candidate: Candidate
    delivered: tuple[SentMessage, ...]
    requested_backend: str
    skipped_stickers: int = 0

    @property
    def backend(self) -> str:
        return self.draft.backend

    @property
    def fell_back(self) -> bool:
        """The reply was written by another backend than the one that was asked for."""
        return self.draft.backend != self.requested_backend

    @property
    def usable(self) -> bool:
        """A reply that can be shown: the pipeline did not give up and said something."""
        return (
            not self.draft.needs_fallback and not self.draft.no_reply and not self.candidate.empty
        )


def merged_turns(history: Sequence[Turn], limit: int = CONTEXT_TURNS - 1) -> tuple[Turn, ...]:
    """The newest ``limit`` merged turns of a history (the round being answered is the eighth)."""
    return tuple(history[-limit:]) if limit > 0 else ()


class EvalSandbox:
    """Replies for the evaluation (see the module description)."""

    def __init__(
        self,
        services: Services,
        pipeline: ReplyPipeline,
        *,
        mode: SandboxMode,
        channel: InMemoryChannel,
        clock: SampleClock,
        tag: LedgerTag | None = None,
        past: AsOfSource | None = None,
        live: LiveDataSource | None = None,
    ) -> None:
        if mode is SandboxMode.HOLDOUT and past is None:
            raise SandboxError("the hold-out mode needs the as-of source")
        if mode is SandboxMode.LIVE and live is None:
            raise SandboxError("the live mode needs the live data source")
        self._services = services
        self._pipeline = pipeline
        self._mode = mode
        self._channel = channel
        self._clock = clock
        self._tag = tag
        self._past = past
        self._live = live
        self._catalog = StickerCatalog(services)
        self._sender = StickerSender(channel, services.media)

    @property
    def mode(self) -> SandboxMode:
        return self._mode

    @property
    def channel(self) -> InMemoryChannel:
        return self._channel

    @property
    def pipeline(self) -> ReplyPipeline:
        return self._pipeline

    def view(self, at: datetime) -> ReplyDataView:
        """The data view of the sample moment: ``AsOfView(at)`` or the live view."""
        moment = ensure_aware(at)
        if self._mode is SandboxMode.HOLDOUT and self._past is not None:
            return self._past.at(moment)
        if self._mode is SandboxMode.LIVE and self._live is not None:
            return self._live.view(moment)
        raise SandboxError("the sandbox has no data source for its mode")

    def context(self, request: SandboxRequest) -> ReplyContext:
        """The round as the pipeline gets it: at most eight merged turns, the channel's limits."""
        return ReplyContext(
            inbound=request.inbound,
            history=merged_turns(request.history),
            thinking_mode=request.thinking_mode,
            backend=request.backend,
            limits=SendLimits(supports_quote=self._channel.capabilities().supports_quote),
            seed=request.seed,
        )

    async def preview(self, request: SandboxRequest) -> list[ChatMessage]:
        """The prompt of the DeepSeek backend for this request; the estimate of a batch uses it."""
        self._clock.set(request.at)
        with isolated_writes():
            return await self._pipeline.preview(self.context(request), self.view(request.at))

    async def reply(self, request: SandboxRequest) -> SandboxReply:
        """Make the reply the bot would have given, deliver it to the in-memory channel."""
        self._clock.set(request.at)
        context = self.context(request)
        data = self.view(request.at)
        with isolated_writes():
            if self._tag is not None:
                with call_scope(Purpose.EVAL, self._tag):
                    draft = await self._pipeline.run(context, data)
            else:
                draft = await self._pipeline.run(context, data)
            delivered, candidate, skipped = await self._deliver(draft, request.inbound)
        return SandboxReply(draft, candidate, delivered, request.backend, skipped)

    # ----------------------------------------------------------------- delivery

    @staticmethod
    def _quote_target(fragment: str | None, items: Sequence[InboundItem]) -> QuoteTarget | None:
        """The message the quote fragment comes from (as the engine finds it)."""
        if not fragment:
            return None
        for item in reversed(items):
            if fragment in item.text:
                return QuoteTarget(item.id, fragment)
        return QuoteTarget(items[-1].id, fragment)

    async def _deliver(
        self, draft: ReplyDraft, items: Sequence[InboundItem]
    ) -> tuple[tuple[SentMessage, ...], Candidate, int]:
        """Hand the bubbles to the channel at once; the candidate is what it accepted."""
        self._channel.clear()
        quote = (
            self._quote_target(draft.quote, items)
            if self._channel.capabilities().supports_quote
            else None
        )
        shown: list[tuple[str, str | None, str]] = []
        skipped = 0
        quoted: str | None = None
        for bubble in draft.bubbles:
            if bubble.is_sticker:
                record = (
                    await asyncio.to_thread(self._catalog.get, bubble.sticker_md5)
                    if bubble.sticker_md5
                    else None
                )
                try:
                    if record is None:
                        raise MediaNotAllowed("the sticker is not in the library")
                    result = await self._sender.send_sticker(record)
                except (MediaNotAllowed, OSError, LookupError):
                    skipped += 1  # as the bubble sender does: a sticker that cannot go is skipped
                    continue
                if result.ok:
                    shown.append(("sticker", bubble.sticker_md5, ""))
                else:
                    skipped += 1
                continue
            attach = quote if quote is not None and quoted is None else None
            result = await self._channel.send_text(bubble.text, attach)
            if not result.ok:
                continue
            if attach is not None:
                quoted = attach.text
            shown.append(("text", None, bubble.text))
        return tuple(self._channel.sent), candidate_from_bubbles(shown, quoted), skipped


@dataclass
class SandboxKit:
    """A sandbox and what must be closed with it."""

    sandbox: EvalSandbox
    llm: LlmRuntime
    style: StyleRuntime
    closers: tuple[Callable[[], Awaitable[None]], ...] = ()

    async def aclose(self) -> None:
        for close in self.closers:
            await close()
        await self.style.aclose()
        await self.llm.client.aclose()


def build_sandbox(
    services: Services,
    *,
    mode: SandboxMode,
    batch_id: str | None = None,
    llm: LlmRuntime | None = None,
    style: StyleRuntime | None = None,
    allow: StickerAllowList | None = None,
) -> SandboxKit:
    """The sandbox wired the way the application wires its engine (``ReplyPipeline`` and all).

    ``batch_id`` books the calls on that one-time batch (R-LLM-014); without it the sandbox can
    only be asked to ``preview``, since an unbooked evaluation call would count as daily spending.
    """
    runtime = llm or build_llm_runtime(services)
    style_runtime = style or StyleRuntime.from_services(services, runtime)
    pipeline = ReplyPipeline.from_services(services, runtime, extra_backends=style_runtime.backends)
    clock = SampleClock(services.clock)
    channel = InMemoryChannel(
        clock=clock,
        media_policy=allow if allow is not None else StickerLibraryAllowList(services.db),
    )
    tag = LedgerTag("one_time", batch_id) if batch_id else None
    past = AsOfSource(services) if mode is SandboxMode.HOLDOUT else None
    live = EvalLiveSource(services) if mode is SandboxMode.LIVE else None
    sandbox = EvalSandbox(
        services, pipeline, mode=mode, channel=channel, clock=clock, tag=tag, past=past, live=live
    )
    closers: list[Callable[[], Awaitable[None]]] = []
    if past is not None:
        closers.append(past.retriever.aclose)
    if live is not None:
        closers.append(live.aclose)
    return SandboxKit(sandbox, runtime, style_runtime, tuple(closers))


def with_backend(request: SandboxRequest, backend: str) -> SandboxRequest:
    """The same round, asked of another backend."""
    return replace(request, backend=backend)


def stable_seed(*parts: str) -> int:
    """A seed that depends on nothing but the parts (the same sample draws the same stickers)."""
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % SEED_MODULUS
