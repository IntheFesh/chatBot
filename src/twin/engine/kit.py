"""The parts a running engine is made of, for the code that works beside it (round 10).

The proactive scheduler writes to the user through the same pipeline, data source, sender and
stores as the engine, and with the same style model and DeepSeek client, so it asks the engine
for them instead of building a second set.  :func:`~twin.engine.component.build_engine` fills
``engine.kit``; an engine built by hand (a test) has none, and nothing that needs it is started.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from twin.channel.base import Channel
    from twin.engine.dataview import LiveDataSource
    from twin.engine.machine import DraftWriter
    from twin.engine.sender import StickerLookup
    from twin.engine.sticker_sender import StickerSender
    from twin.engine.style_runtime import StyleRuntime
    from twin.engine.turns import BotTurnStore
    from twin.llm.runtime import LlmRuntime


@dataclass(frozen=True)
class EngineKit:
    """What the engine was built from (see the module description)."""

    llm: LlmRuntime
    style: StyleRuntime
    data: LiveDataSource
    pipeline: DraftWriter
    channel: Channel
    stickers: StickerSender
    lookup: StickerLookup
    store: BotTurnStore
    rng: random.Random
