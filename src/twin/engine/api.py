"""What later rounds import from the reply engine (round 09, steps 1 and 2).

::

    from twin.engine.api import (
        ReplyPipeline, ReplyContext, InboundItem, SendLimits,       # one round in, a draft out
        LiveDataSource, ReplyDataView,                               # what the pipeline may know
        BotTurnStore, ConversationStateStore, FeedbackStore,         # the conversation on disk
        HistoryLoader, InboundRenderer, CrisisHandler,
        StyleRuntime, BackendSelector,                               # the style model's backends
        StylePromptBuilder, StyleTurn, TokenBudget, PlanFields,      # its prompt (the export's too)
        CommandPort, CommandContext, CommandOutcome,                 # the commands (twin.commands)
    )

    llm = build_llm_runtime(services)
    style = StyleRuntime.from_services(services, llm)                # client, backends, selector
    llm.budget.set_style_status(style.selector)                      # R-LLM-008, last level
    pipeline = ReplyPipeline.from_services(services, llm, extra_backends=style.backends)
    data = LiveDataSource(services)                                  # one per process
    choice = await style.selector.choose()                           # deepseek, style or hybrid
    draft = await pipeline.run(replace(context, backend=choice.name), data.view())
    style.selector.record(choice.name, draft)                        # failures and violations
    if draft.needs_fallback: ...                                     # the engine's own business

``ReplyPipeline.run(context, data_view)`` is the single implementation of "write a reply" for the
running bot and for the evaluation sandbox (round 09b, which passes ``AsOfView(t)`` as the data
view).  The backends (``deepseek``, ``style`` and ``hybrid``) implement :class:`ReplyBackend`; the
state machine, the pacing and the sending are the engine's (step 3) and use the stores below.

``StylePromptBuilder`` renders the style model's prompt for the running bot and for the training
set (``twin train export`` passes ``AsOfView(t)`` as the data and, to fit ``cutoff_len``, a
``TokenBudget``).  ``CommandPort`` is what the engine asks to run a command message; the router
that implements it is :class:`twin.commands.router.CommandRouter`.
"""

from __future__ import annotations

from twin.engine.backend import BackendRequest, BackendResult, ReplyBackend
from twin.engine.backend_select import (
    BackendChoice,
    BackendMonitorComponent,
    BackendSelector,
    StyleStatus,
    SwitchCheck,
)
from twin.engine.command_port import CommandContext, CommandOutcome, CommandPort
from twin.engine.dataview import LiveDataSource, LiveDataView, ReplyDataView
from twin.engine.deepseek_backend import DeepSeekBackend
from twin.engine.feedback import FeedbackRecord, FeedbackStore
from twin.engine.history import HistoryLoader
from twin.engine.hybrid_backend import HybridBackend, HybridPlan
from twin.engine.inbound import InboundRenderer
from twin.engine.pipeline import FALLBACK_BACKEND, ReplyPipeline
from twin.engine.postprocess import (
    PostContext,
    PostProcessor,
    PostResult,
    StyleLimits,
    fit_bubbles_to_quota,
)
from twin.engine.prompt import PromptBuilder
from twin.engine.safety.crisis import CrisisHandler, CrisisOutcome
from twin.engine.safety.notifier import EmergencyNotice, EmergencyNotifier
from twin.engine.state_store import ConversationSnapshot, ConversationStateStore
from twin.engine.style_backend import StyleBackend, StyleWriter
from twin.engine.style_models import ActiveStyleModel, StyleModels
from twin.engine.style_prompt import (
    LockedVersionError,
    NormalizedContext,
    PlanFields,
    PromptParts,
    StylePromptBuilder,
    StylePromptError,
    StyleTurn,
    TokenBudget,
    normalize_context,
    style_turns,
)
from twin.engine.style_runtime import ConfiguredStyleClient, StyleRuntime
from twin.engine.turns import (
    BotTurnMessages,
    BotTurnStore,
    OutboundBubble,
    ReplyMeta,
    TurnRecord,
)
from twin.engine.types import (
    Bubble,
    InboundItem,
    PostAction,
    ReplyContext,
    ReplyDraft,
    ReplyMaterial,
    SendLimits,
    UsageSummary,
    Violation,
)

__all__ = [
    "FALLBACK_BACKEND",
    "ActiveStyleModel",
    "BackendChoice",
    "BackendMonitorComponent",
    "BackendRequest",
    "BackendResult",
    "BackendSelector",
    "BotTurnMessages",
    "BotTurnStore",
    "Bubble",
    "CommandContext",
    "CommandOutcome",
    "CommandPort",
    "ConfiguredStyleClient",
    "ConversationSnapshot",
    "ConversationStateStore",
    "CrisisHandler",
    "CrisisOutcome",
    "DeepSeekBackend",
    "EmergencyNotice",
    "EmergencyNotifier",
    "FeedbackRecord",
    "FeedbackStore",
    "HistoryLoader",
    "HybridBackend",
    "HybridPlan",
    "InboundItem",
    "InboundRenderer",
    "LiveDataSource",
    "LiveDataView",
    "LockedVersionError",
    "NormalizedContext",
    "OutboundBubble",
    "PlanFields",
    "PostAction",
    "PostContext",
    "PostProcessor",
    "PostResult",
    "PromptBuilder",
    "PromptParts",
    "ReplyBackend",
    "ReplyContext",
    "ReplyDataView",
    "ReplyDraft",
    "ReplyMaterial",
    "ReplyMeta",
    "ReplyPipeline",
    "SendLimits",
    "StyleBackend",
    "StyleLimits",
    "StyleModels",
    "StylePromptBuilder",
    "StylePromptError",
    "StyleRuntime",
    "StyleStatus",
    "StyleTurn",
    "StyleWriter",
    "SwitchCheck",
    "TokenBudget",
    "TurnRecord",
    "UsageSummary",
    "Violation",
    "fit_bubbles_to_quota",
    "normalize_context",
    "style_turns",
]
