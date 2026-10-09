"""The pipeline on real data: the live view and ``AsOfView`` as one protocol (R-TRN-013).

One conversation, one profile, one persona card, one memory and one example library are built from
synthetic data; then the same ``ReplyPipeline`` answers once through ``LiveDataView`` and once
through ``AsOfView(t)`` of a past moment - which is how the evaluation sandbox (round 09b) will use
it.
"""

from __future__ import annotations

import json
import random
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta

import pytest
import respx
from sqlalchemy import select

from tests.support.deepseek import API, TEST_KEY, ok
from tests.support.embedding import HashingBackend
from tests.support.memory import add_fact, make_memory
from tests.support.synth_chat import ChatSpec, build_chat
from twin.engine.dataview import LiveDataSource, LiveDataView, ReplyDataView
from twin.engine.pipeline import ReplyPipeline
from twin.engine.turns import BotTurnStore, OutboundBubble, ReplyMeta
from twin.engine.types import InboundItem, ReplyContext
from twin.llm.runtime import DEEPSEEK_SECRET, LlmRuntime, build_llm_runtime
from twin.memory.api import AsOfSource, AsOfView, Memory
from twin.memory.asof import PUBLIC_API
from twin.memory.blocks import MemoryQuery
from twin.profile.builder import rebuild
from twin.profile.persona import compose
from twin.profile.persona.store import PersonaStore
from twin.retrieval.embedder import EmbeddingService
from twin.retrieval.indexer import run_index
from twin.retrieval.query import ExampleRetriever
from twin.schedule.time_service import BotTimeService
from twin.services import Services
from twin.stickers.catalog import StickerCatalog
from twin.stickers.selector import LIVE_VIEW
from twin.storage.chat_models import StickerUse

PAST = datetime(2026, 8, 25, 18, 0, tzinfo=UTC)
FACT = "养了一只叫豆包的猫"


def card(marker: str) -> str:
    description = f"### 风格\n- 口头禅：{marker}\n\n### 基本情况\n- 事实：{FACT}\n\n"
    return compose.stats_block(["几乎不用逗号"]) + compose.auto_block(description)


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def world(services: Services, embedder: HashingBackend) -> Services:
    services.settings.retrieval.model = embedder.info.model
    build_chat(services, ChatSpec(days=40))
    rebuild(services, "all")
    run_index(services)
    store = PersonaStore(services.db, services.clock)
    store.add_version("live", card("现在的口头禅是嘿嘿"), reason="generate")
    store.add_version("pre_holdout", card("过去的口头禅是哈哈"), reason="generate")
    add_fact(make_memory(services), FACT, datetime(2026, 8, 10, tzinfo=UTC))
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    return services


@pytest.fixture
async def runtime(world: Services) -> AsyncIterator[LlmRuntime]:
    built = build_llm_runtime(world)
    yield built
    await built.client.aclose()


def source(world: Services, embedder: HashingBackend) -> LiveDataSource:
    retriever = ExampleRetriever(world, EmbeddingService(embedder))
    return LiveDataSource(world, memory=Memory(world), retriever=retriever, rng=random.Random(3))


def context(text: str, at: datetime) -> ReplyContext:
    return ReplyContext(inbound=(InboundItem("in-1", at, "text", text),), seed=1)


def request_text(route: respx.Route) -> tuple[str, str]:
    body = json.loads(route.calls[0].request.content)
    return body["messages"][0]["content"], body["messages"][-1]["content"]


def test_the_live_view_answers_every_member_of_the_protocol(
    world: Services, embedder: HashingBackend
) -> None:
    view = source(world, embedder).view()
    assert isinstance(view, ReplyDataView)
    assert view.at == world.clock.now_utc() and view.local.zone == "America/Chicago"
    assert view.local.local.utcoffset() == timedelta(hours=-5)  # daylight time in October
    full, compact = view.persona_full(), view.persona_compact()
    assert full is not None and "现在的口头禅是嘿嘿" in full.text and full.scope == "live"
    assert compact is not None and FACT not in compact.text  # the compact card has no facts
    assert view.profile is not None and view.activity is not None
    assert view.profile is view.profile and view.emoji_codes is view.emoji_codes  # read once
    assert view.her_state() in {"deep_sleep", "sleep_edge", "busy", "free"}
    assert FACT in view.memory_block(MemoryQuery(FACT), 800).text
    assert view.sticker_view is LIVE_VIEW and view.sticker_selector() is view.sticker_selector()
    assert view.sticker_selector(random.Random(1)) is not view.sticker_selector()
    assert view.lifeline == () and view.bot_turns == ()
    # the members of AsOfView are all there (the sandbox adapts it without translation)
    for name in PUBLIC_API - {"memory", "at"}:
        assert hasattr(view, name), name


def test_bubbles_sent_so_far_are_the_bot_turns_the_view_shows(
    world: Services, embedder: HashingBackend
) -> None:
    store = BotTurnStore(world.db, world.clock)
    store.add_inbound(at=world.clock.now_utc(), kind="text", text="在吗")
    store.add_reply(
        [
            OutboundBubble("在呢", world.clock.now_utc()),
            OutboundBubble("[表情包:开心]", world.clock.now_utc(), "sticker", "a" * 32),
        ],
        ReplyMeta("deepseek"),
    )
    view = source(world, embedder).view()
    assert [(m.role, m.text) for m in view.bot_turns] == [
        ("user", "在吗"),
        ("bot", "在呢"),
        ("bot", "[表情包:开心]"),
    ]
    rate = view.sticker_rate()
    assert rate.status().bubbles == 2 and rate.share() > 0  # read from bot_turns, not remembered


async def test_a_live_reply_uses_the_card_the_memory_and_her_real_replies(
    world: Services, embedder: HashingBackend, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(return_value=ok(content="好呀，一起去"))
    pipeline = ReplyPipeline.from_services(world, runtime, rng=random.Random(2))
    data = source(world, embedder)
    draft = await pipeline.run(context(f"你还记得{FACT}吗", world.clock.now_utc()), data.view())
    assert draft.usable and [b.text for b in draft.bubbles] == ["好呀", "一起去"]
    system, last = request_text(route)
    assert "现在的口头禅是嘿嘿" in system and "几乎不用逗号" in system
    assert FACT in last  # the memory block is in the last message
    assert "仅供模仿语气，不要照抄内容" in last and "对方：" in last  # her real replies
    assert "examples_unavailable" not in {a.step for a in draft.actions}
    assert "memory_unavailable" not in {a.step for a in draft.actions}
    assert draft.meta["persona"] == "live v1"
    await data.aclose()


async def test_a_sticker_line_becomes_one_of_her_real_stickers(
    world: Services, embedder: HashingBackend, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    with world.db.session() as session:
        used = sorted(session.scalars(select(StickerUse.sticker_md5).where(StickerUse.by_her)))
    md5s = sorted(set(used))
    assert md5s
    catalog = StickerCatalog(world)
    for md5 in md5s:
        catalog.save_vision(md5, ["开心"], "笑得很开心的猫", "好消息", at=world.clock.now_utc())
    api.post(API).mock(return_value=ok(content="好呀\n[表情包:开心]\n[表情包:不存在的标签]"))
    pipeline = ReplyPipeline.from_services(world, runtime, rng=random.Random(2))
    data = source(world, embedder)
    draft = await pipeline.run(context("给你看个好消息", world.clock.now_utc()), data.view())
    assert [b.kind for b in draft.bubbles] == ["text", "sticker"]
    assert draft.bubbles[1].sticker_md5 in md5s and draft.bubbles[1].sticker_tag == "开心"
    assert draft.bubbles[1].text == "[表情包:开心]"
    assert "sticker_unknown_tag" in {a.step for a in draft.actions}
    await data.aclose()


async def test_the_same_pipeline_answers_for_a_past_moment_through_as_of_view(
    world: Services, embedder: HashingBackend, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(return_value=ok(content="好呀"))
    pipeline = ReplyPipeline.from_services(world, runtime, rng=random.Random(2))
    retriever = ExampleRetriever(world, EmbeddingService(embedder))
    view = AsOfView(AsOfSource(world, memory=Memory(world), retriever=retriever), PAST)
    assert isinstance(view, ReplyDataView)  # the sandbox needs no adapter
    draft = await pipeline.run(context("今天怎么样", PAST), view)
    assert draft.usable and [b.text for b in draft.bubbles] == ["好呀"]
    system, last = request_text(route)
    assert "过去的口头禅是哈哈" in system and "现在的口头禅是嘿嘿" not in system
    assert "你刚醒" not in last and "她这会儿大概在" not in last  # no life line in the past
    assert "2026年8月25日" in last  # the clock of the moment, not today's
    assert draft.meta["persona"] == "pre_holdout v1"
    await retriever.aclose()


def test_a_view_is_for_one_moment(world: Services, embedder: HashingBackend) -> None:
    data = source(world, embedder)
    moment = datetime(2026, 10, 9, 3, 0, tzinfo=UTC)
    view = data.view(moment)
    assert isinstance(view, LiveDataView) and view.at == moment
    assert view.local.local.hour == 22 and view.local.weekday == 3  # Thursday evening in Chicago
    assert view.local.slot == 22 * 4


def test_a_fresh_installation_has_a_view_too(services: Services) -> None:
    """Nothing imported yet: no profile, no card, no memory - and still a working view."""
    view = LiveDataSource(services).view()
    assert view.profile is None and view.activity is None and view.emoji_codes is None
    assert view.persona_full() is None and view.persona_compact() is None
    assert view.lifeline == () and view.bot_turns == ()
    assert view.her_state() == "free"  # the day plan has a default routine to start from
    unplanned = BotTimeService(services.clock, lambda: "America/Chicago")  # no day plans attached
    assert LiveDataSource(services, time_service=unplanned).view().her_state() is None
