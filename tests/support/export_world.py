"""A synthetic conversation with everything the training export reads (round 13b tests).

``build_world`` writes a conversation (``tests.support.synth_chat``), computes the profile and the
routine in both scopes, adds a pre-holdout and a live persona card with *different* markers, tags
some stickers, gives some of her quote replies a quoted text, and puts two facts into the memory:
one that was known long before the conversation ends and one that is only learned after it.
Nothing here is a real conversation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from tests.support.embedding import HashingBackend
from tests.support.memory import add_fact, make_memory
from tests.support.synth_chat import ChatSpec, build_chat
from twin.memory.memory import Memory
from twin.profile.builder import rebuild
from twin.profile.holdout import holdout_cutoff
from twin.profile.persona import compose
from twin.profile.persona.store import PersonaStore
from twin.services import Services
from twin.stickers.catalog import StickerCatalog
from twin.storage.chat_models import Message

EARLY_FACT = "早就知道的事：对方住在二楼"
LATE_FACT = "之后才知道的事：对方换了工作"
PAST_MARKER = "过去的口头禅嘿嘿"
LIVE_STYLE = "现在的口头禅哈哈"
FACT_AUTO = "养了一只叫豆包的猫"
FACT_MANUAL = "她的生日在三月三日"
CORRECTION = "不要说得太书面"
STICKER_TAGS = {0: "开心", 1: "大笑", 2: "撒娇"}
QUOTED = "昨天说好的那家火锅店，下次一起去吧，我请客，不见不散哦，记得提前订位置"


def past_card() -> str:
    description = f"### 风格\n- 口头禅是{PAST_MARKER}\n\n### 基本情况\n- 过去的事实{FACT_AUTO}\n\n"
    return (
        compose.stats_block(["几乎不用逗号"])
        + compose.auto_block(description)
        + compose.manual_block(["喜欢拖长音"], None)
    )


def live_card() -> str:
    description = f"### 风格\n- {LIVE_STYLE}\n\n### 基本情况\n- {FACT_AUTO}\n\n"
    return (
        compose.stats_block(["几乎不用逗号"])
        + compose.auto_block(description)
        + compose.manual_block(["喜欢拖长音"], [FACT_MANUAL])
        + compose.dont_block([CORRECTION])
    )


@dataclass
class World:
    services: Services
    memory: Memory
    store: PersonaStore
    cutoff: datetime


def quote_some(services: Services, every: int = 3) -> int:
    """Give every ``every``-th quote reply of hers a quoted text; returns how many."""
    changed = 0
    with services.db.transaction(bump_state=False) as session:
        rows = session.scalars(
            select(Message).where(Message.kind == "quote").order_by(Message.create_time_utc)
        ).all()
        for index, row in enumerate(rows):
            if index % every == 0:
                row.quote = {"quoteContent": QUOTED, "quoteTitle": "她"}
                changed += 1
    return changed


def build_world(
    services: Services,
    embedder: HashingBackend,
    *,
    days: int = 40,
    spec: ChatSpec | None = None,
    tag: bool = True,
) -> World:
    services.settings.retrieval.model = embedder.info.model
    build_chat(services, spec or ChatSpec(days=days))
    rebuild(services, "all")
    store = PersonaStore(services.db, services.clock)
    store.add_version("pre_holdout", past_card(), reason="generate")
    store.add_version("live", live_card(), reason="generate")
    memory = make_memory(services)
    add_fact(memory, EARLY_FACT, datetime(2026, 8, 10, tzinfo=UTC))
    add_fact(memory, LATE_FACT, datetime(2026, 9, 20, tzinfo=UTC))
    if tag:
        catalog = StickerCatalog(services)
        for number, label in STICKER_TAGS.items():
            if catalog.get(f"{number:032x}") is not None:
                catalog.set_manual(f"{number:032x}", [label])
    quote_some(services)
    return World(services, memory, store, holdout_cutoff(services))


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
