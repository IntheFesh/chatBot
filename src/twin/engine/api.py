"""What later rounds import from the reply engine (round 09, step 1: the stateless part).

::

    from twin.engine.api import (
        ReplyPipeline, ReplyContext, InboundItem, SendLimits,       # one round in, a draft out
        LiveDataSource, ReplyDataView,                               # what the pipeline may know
        BotTurnStore, ConversationStateStore, FeedbackStore,         # the conversation on disk
        HistoryLoader, InboundRenderer, CrisisHandler,
    )

    pipeline = ReplyPipeline.from_services(services, build_llm_runtime(services))
    data = LiveDataSource(services)                                  # one per process
    draft = await pipeline.run(context, data.view())                 # no waiting, no sending
    if draft.needs_fallback: ...                                     # the engine's own business

``ReplyPipeline.run(context, data_view)`` is the single implementation of "write a reply" for the
running bot and for the evaluation sandbox (round 09b, which passes ``AsOfView(t)`` as the data
view).  The backends (``deepseek`` here, ``style`` and ``hybrid`` in step 2) implement
:class:`ReplyBackend`; the state machine, the pacing, the commands and the sending are the
engine's (step 3) and use the stores below.
"""

from __future__ import annotations

from twin.engine.backend import BackendRequest, BackendResult, ReplyBackend
from twin.engine.dataview import LiveDataSource, LiveDataView, ReplyDataView
from twin.engine.deepseek_backend import DeepSeekBackend
from twin.engine.feedback import FeedbackRecord, FeedbackStore
from twin.engine.history import HistoryLoader
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
    "BackendRequest",
    "BackendResult",
    "BotTurnMessages",
    "BotTurnStore",
    "Bubble",
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
    "InboundItem",
    "InboundRenderer",
    "LiveDataSource",
    "LiveDataView",
    "OutboundBubble",
    "PostAction",
    "PostContext",
    "PostProcessor",
    "PostResult",
    "PromptBuilder",
    "ReplyBackend",
    "ReplyContext",
    "ReplyDataView",
    "ReplyDraft",
    "ReplyMaterial",
    "ReplyMeta",
    "ReplyPipeline",
    "SendLimits",
    "StyleLimits",
    "TurnRecord",
    "UsageSummary",
    "Violation",
    "fit_bubbles_to_quota",
]
