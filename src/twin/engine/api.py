"""What later rounds import from the reply engine (round 09: the pipeline and the state machine).

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

    from twin.engine.api import (
        ConversationEngine, build_engine, register_engine,           # the running bot (step 3)
        CommandPort, CommandContext, CommandOutcome,                 # what the command router fits
        StickerSender, Sticker,                                      # the one door for pictures
        PacingModel, Decider, Decision,                              # when and how fast she answers
    )

    engine = build_engine(services, channel)                         # wired as the app does it
    await engine.handle_message(message)                             # stored, queued; returns

``ReplyPipeline.run(context, data_view)`` is the single implementation of "write a reply" for the
running bot and for the evaluation sandbox (round 09b, which passes ``AsOfView(t)`` as the data
view).  The backends (``deepseek`` here, ``style`` and ``hybrid`` in step 2) implement
:class:`ReplyBackend`.  The state machine (:class:`ConversationEngine`), the pacing, the sending and
the failure fallback are step 3; they use the stores below.  The command router of step 2 plugs in
through :class:`CommandPort` at ``command_port_for`` in :mod:`twin.engine.component`.
"""

from __future__ import annotations

from twin.engine.backend import BackendRequest, BackendResult, ReplyBackend
from twin.engine.command_port import CommandContext, CommandOutcome, CommandPort
from twin.engine.component import (
    EngineComponent,
    build_engine,
    command_port_for,
    register_engine,
)
from twin.engine.dataview import LiveDataSource, LiveDataView, ReplyDataView
from twin.engine.decision import Decider, Decision
from twin.engine.deepseek_backend import DeepSeekBackend
from twin.engine.fallback import ShortAnswers
from twin.engine.feedback import FeedbackRecord, FeedbackStore
from twin.engine.history import HistoryLoader
from twin.engine.inbound import InboundRenderer
from twin.engine.machine import ConversationEngine, QuietWindow
from twin.engine.pacing import PacingModel
from twin.engine.pipeline import FALLBACK_BACKEND, ReplyPipeline
from twin.engine.postprocess import (
    PostContext,
    PostProcessor,
    PostResult,
    StyleLimits,
    fit_bubbles_to_quota,
)
from twin.engine.prompt import PromptBuilder
from twin.engine.rounds import RoundStore
from twin.engine.safety.crisis import CrisisHandler, CrisisOutcome
from twin.engine.safety.notifier import EmergencyNotice, EmergencyNotifier
from twin.engine.sender import BubbleSender, OutBubble, StopReason
from twin.engine.state_store import ConversationSnapshot, ConversationStateStore
from twin.engine.sticker_sender import Sticker, StickerSender
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
    "BubbleSender",
    "CommandContext",
    "CommandOutcome",
    "CommandPort",
    "ConversationEngine",
    "ConversationSnapshot",
    "ConversationStateStore",
    "CrisisHandler",
    "CrisisOutcome",
    "Decider",
    "Decision",
    "DeepSeekBackend",
    "EmergencyNotice",
    "EmergencyNotifier",
    "EngineComponent",
    "FeedbackRecord",
    "FeedbackStore",
    "HistoryLoader",
    "InboundItem",
    "InboundRenderer",
    "LiveDataSource",
    "LiveDataView",
    "OutBubble",
    "OutboundBubble",
    "PacingModel",
    "PostAction",
    "PostContext",
    "PostProcessor",
    "PostResult",
    "PromptBuilder",
    "QuietWindow",
    "ReplyBackend",
    "ReplyContext",
    "ReplyDataView",
    "ReplyDraft",
    "ReplyMaterial",
    "ReplyMeta",
    "ReplyPipeline",
    "RoundStore",
    "SendLimits",
    "ShortAnswers",
    "Sticker",
    "StickerSender",
    "StopReason",
    "StyleLimits",
    "TurnRecord",
    "UsageSummary",
    "Violation",
    "build_engine",
    "command_port_for",
    "fit_bubbles_to_quota",
    "register_engine",
]
