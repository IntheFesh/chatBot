"""``AsOfView(t)``: nothing of the future reaches a past moment (R-TRN-013, R-MEM-010).

The injection test is the one the design rests on: the data holds a fact that appears for the
first time in the very reply block the sample asks about, summaries of the day itself and of later
days, a follow-up that was closed afterwards, and ``memory_view(t)`` and ``AsOfView(t)`` must not
show any of it.
"""

from __future__ import annotations

import random
from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import select

from tests.support.embedding import HashingBackend
from tests.support.memory import (
    add_event,
    add_fact,
    add_followup,
    add_summary,
    make_memory,
    utc,
)
from tests.support.synth_chat import ChatSpec, build_chat
from twin.config.settings import TzRange
from twin.memory.api import AsOfSource, AsOfView, Memory, MemoryQuery, memory_view
from twin.memory.asof import PUBLIC_API
from twin.profile.api import load_activity_model, load_profile
from twin.profile.builder import rebuild
from twin.profile.holdout import holdout_cutoff
from twin.profile.persona import compose
from twin.profile.persona.store import PersonaStore
from twin.retrieval.indexer import run_index
from twin.retrieval.query import QueryTurn
from twin.services import Services
from twin.stickers.selector import StickerSelector
from twin.storage.retrieval_models import ExampleWindow

T = utc(2026, 3, 10, 18)  # 13:00 on 10 March in Chicago: the sample's reply block starts here
SECRET = "偷偷报了潜水课"
TOPIC = "潜水课 火锅 面试 法语 牙医"
FACT = "养了一只叫豆包的猫"
CORRECTION = "不要把话说得太正式"


@pytest.fixture
def memory(services: Services, embedder: HashingBackend) -> Memory:
    services.settings.memory.recall_min_similarity = 0.3
    return make_memory(services)


@pytest.fixture
def world(memory: Memory) -> Memory:
    """What the bot knows, before, at and after the sample time ``T``."""
    add_fact(memory, "她喜欢吃火锅", utc(2026, 3, 9))
    add_fact(memory, f"她{SECRET}", T)  # proved by the target reply block itself
    add_fact(memory, "她开始学法语", T + timedelta(hours=1))
    add_summary(memory, "real", date(2026, 3, 9), "周一他们聊了火锅和面试")
    add_summary(memory, "real", date(2026, 3, 10), f"周二她说自己{SECRET}")  # the day of the sample
    add_summary(memory, "real", date(2026, 3, 11), f"周三他们继续聊{SECRET}的细节")
    add_followup(
        memory,
        "她周二晚上有面试",
        utc(2026, 3, 11, 1),
        utc(2026, 3, 8),
        close_at=utc(2026, 3, 12),  # closed two days after the sample
        status="done",
    )
    add_followup(memory, "她周五去看牙医", utc(2026, 3, 14, 20), T + timedelta(hours=2))
    add_followup(
        memory, "她周一交论文", utc(2026, 3, 9, 20), utc(2026, 3, 2), close_at=utc(2026, 3, 9, 22)
    )
    return memory


def rendered(block_text: str) -> str:
    return block_text.replace("\n", " ")


# ---------------------------------------------------------------- the injection test


def test_neither_the_memory_view_nor_the_as_of_view_shows_the_future(world: Memory) -> None:
    services = world.services
    for topic in ("想吃火锅", "潜水课", "法语 牙医 论文"):
        query = MemoryQuery(topic)
        via_view = memory_view(services, T, memory=world).render(query, 2000)
        via_asof = AsOfView(services, T).memory_block(query, 2000)
        for block in (via_view, via_asof):
            text = rendered(block.text)
            assert "周一他们聊了火锅和面试" in text  # the summary of an earlier day
            assert "她周二晚上有面试" in text  # a follow-up that was still open at T
            for hidden in (
                SECRET,  # first said in the target block
                "偷偷",
                "法语",  # known an hour later
                "周二她说自己",  # the summary of the day itself
                "周三他们继续聊",  # a later summary
                "看牙医",  # created after T
                "交论文",  # closed before T
            ):
                assert hidden not in text, (topic, hidden)
        assert via_view.text == via_asof.text  # one code path for the live view and the export
    assert "她喜欢吃火锅" in AsOfView(services, T).memory_block(MemoryQuery("想吃火锅"), 2000).text
    # a moment later the data is all there: it was the time that hid it, not the data
    later = memory_view(services, T + timedelta(seconds=1), memory=world).render(
        MemoryQuery("潜水课"), 2000
    )
    assert SECRET in later.text
    next_morning = memory_view(services, utc(2026, 3, 12, 14), memory=world).render(
        MemoryQuery("潜水课"), 2000
    )
    assert "周二她说自己" in next_morning.text
    soon = memory_view(services, utc(2026, 3, 12, 14), memory=world).render(
        MemoryQuery("学法语"), 2000
    )
    assert "法语" in soon.text


def test_the_structured_views_agree_with_the_rendered_text(world: Memory) -> None:
    services = world.services
    view = AsOfView(services, T)
    assert [f.text for f in view.memory.facts()] == ["她喜欢吃火锅"]
    assert [s.text for s in view.memory.summaries()] == ["周一他们聊了火锅和面试"]
    (follow,) = view.memory.followups()
    assert (follow.text, follow.status, follow.closed_at) == ("她周二晚上有面试", "open", None)
    assert view.lifeline == () and view.bot_turns == ()


def test_the_bot_era_is_not_part_of_a_past_moment(world: Memory) -> None:
    """Even with a life line and bot facts in the data, the view of a past sample has none."""
    world.store.mark_bot_online(utc(2026, 3, 1))
    add_fact(world, "她昨天去爬山了", utc(2026, 3, 9), source="bot_invented")
    add_event(world, date(2026, 3, 9), "去爬山", created_at=utc(2026, 3, 9))
    view = AsOfView(world.services, T)
    assert view.lifeline == ()
    assert "生活线" not in view.memory_block(MemoryQuery("爬山"), 2000).text
    plain = memory_view(world.services, T, memory=world)
    assert [e.activity for e in plain.lifeline()] == ["去爬山"]  # the live view has it


# ---------------------------------------------------------- no way to read everything


MEMORY_VIEW_API = {
    "as_of",
    "bot_era",
    "scope",
    "today",
    "zone",
    "vector_bound",
    "fact_visible",
    "fact",
    "facts",
    "dated_facts",
    "core_facts",
    "summary_visible",
    "summary_for",
    "summary",
    "summaries",
    "followups",
    "lifeline",
    "search_facts",
    "search_summaries",
    "render",
}


def test_the_as_of_view_offers_nothing_that_reads_all_the_data(world: Memory) -> None:
    view = AsOfView(world.services, T)
    public = {name for name in dir(view) if not name.startswith("_")}
    assert public == PUBLIC_API
    for forbidden in (
        "services",
        "source",
        "store",
        "corpus",
        "vectors",
        "all_facts",
        "everything",
        "load",
        "query",
        "session",
    ):
        assert not hasattr(view, forbidden), forbidden
    inside = {name for name in dir(view.memory) if not name.startswith("_")}
    assert inside == MEMORY_VIEW_API
    for forbidden in ("memory", "store", "corpus", "services", "services_", "all_facts"):
        assert not hasattr(view.memory, forbidden), forbidden


def test_no_method_of_the_views_takes_a_moment_or_a_scope_to_look_elsewhere(world: Memory) -> None:
    import inspect

    for owner in (AsOfView, type(AsOfView(world.services, T).memory)):
        for name in dir(owner):
            if name.startswith("_") or not callable(getattr(owner, name)):
                continue
            parameters = set(inspect.signature(getattr(owner, name)).parameters)
            assert not parameters & {"as_of", "at", "moment", "before", "services"}, name


def test_a_view_is_for_one_moment_and_needs_a_time_zone(world: Memory) -> None:
    with pytest.raises(ValueError, match="naive"):
        AsOfView(world.services, datetime(2026, 3, 10, 18, 0))  # noqa: DTZ001 - the point of the test
    source = AsOfSource(world.services)
    first, second = source.at(T), source.at(T + timedelta(days=3))
    assert first.at == T and second.at == T + timedelta(days=3)
    assert len(first.memory.summaries()) < len(second.memory.summaries())


# --------------------------------------------------- pre-holdout data, routine, examples


def live_card() -> str:
    description = f"### 风格\n- 口头禅：哈哈\n\n### 基本情况\n- 事实：{FACT}\n\n"
    return (
        compose.stats_block(["几乎不用逗号"])
        + compose.auto_block(description)
        + compose.manual_block(["- 称呼：宝宝"], ["- 生日在秋天"])
        + compose.dont_block([CORRECTION])
    )


def past_card() -> str:
    description = f"### 风格\n- 口头禅：哈哈\n\n### 基本情况\n- 事实：{FACT}\n\n"
    return (
        compose.stats_block(["几乎不用逗号"])
        + compose.auto_block(description)
        + compose.manual_block(["- 称呼：宝宝"], None)
    )


@pytest.fixture
def chat(services: Services, embedder: HashingBackend) -> Services:
    """Six weeks of conversation with a known routine, profiles of both scopes, the index."""
    services.settings.retrieval.model = embedder.info.model
    services.settings.memory.recall_min_similarity = 0.3
    build_chat(services, ChatSpec())
    rebuild(services, "all", reason="test")
    run_index(services)
    store = PersonaStore(services.db, services.clock)
    store.add_version("live", live_card(), reason="generate")
    store.add_version("pre_holdout", past_card(), reason="generate")
    return services


def test_the_view_carries_the_pre_holdout_data_and_the_routine_of_that_moment(
    chat: Services,
) -> None:
    source = AsOfSource(chat)
    cutoff = holdout_cutoff(chat)
    tuesday_night = datetime(2026, 8, 18, 8, 0, tzinfo=UTC)  # 03:00 in Chicago
    tuesday_noon = datetime(2026, 8, 18, 17, 0, tzinfo=UTC)  # 12:00 in Chicago
    assert tuesday_night < cutoff
    night, noon = source.at(tuesday_night), source.at(tuesday_noon)
    assert night.her_state() == "deep_sleep" and noon.her_state() == "free"
    assert (night.local.day_type, night.local.zone, night.local.weekday_name) == (
        "workday",
        "America/Chicago",
        "周二",
    )
    assert (noon.local.local.hour, noon.local.slot) == (12, 48)
    # profile and routine are those of the pre_holdout scope, not the live ones
    assert noon.profile is not None and noon.profile.version.scope == "pre_holdout"
    live = load_profile(chat, "live")
    assert live is not None and live.version.id != noon.profile.version.id
    pre = load_activity_model(chat, "pre_holdout")
    assert pre is not None and noon.activity is not None
    assert noon.activity.sleep.window_for("workday") == pre.sleep.window_for("workday")
    # persona: the compact card has no facts; the full card has them and the live corrections
    compact, full = noon.persona_compact(), noon.persona_full()
    assert compact is not None and full is not None
    assert compact.scope == full.scope == "pre_holdout" and compact.kind == "compact"
    assert FACT not in compact.text and CORRECTION not in compact.text
    assert FACT in full.text and CORRECTION in full.text
    assert "称呼：宝宝" in compact.text


def test_her_state_is_read_in_the_zone_she_was_in(chat: Services) -> None:
    chat.settings.time.source_timezone_ranges = [
        TzRange.model_validate({"from": "2026-08-15", "to": "2026-08-25", "tz": "Asia/Shanghai"})
    ]
    instant = datetime(
        2026, 8, 18, 3, 0, tzinfo=UTC
    )  # 11:00 in Shanghai, 22:00 the day before in Chicago
    view = AsOfView(chat, instant)
    assert view.local.zone == "Asia/Shanghai" and view.local.local.hour == 11
    assert view.local.day == date(2026, 8, 18) and view.local.minute == 11 * 60
    elsewhere = AsOfView(chat, datetime(2026, 8, 28, 3, 0, tzinfo=UTC))
    assert elsewhere.local.zone == "America/Chicago" and elsewhere.local.local.hour == 22


async def test_examples_come_from_before_the_moment_only(chat: Services) -> None:
    cutoff = holdout_cutoff(chat)
    early = datetime(2026, 8, 12, 16, 0, tzinfo=UTC)
    source = AsOfSource(chat)
    view = source.at(early)
    found = await view.examples([QueryTurn(False, "哈哈")], k=6)
    assert found and all(example.reply_at < early for example in found)
    with chat.db.session() as session:
        replied = dict(session.execute(select(ExampleWindow.id, ExampleWindow.reply_at_utc)).all())
    assert all(replied[e.window_id] < early for e in found)
    after_cut = source.at(cutoff + timedelta(days=1))
    later = await after_cut.examples([QueryTurn(False, "哈哈")], k=40)
    assert later and all(example.reply_at < cutoff for example in later)  # the hold-out stays out


def test_stickers_and_emoji_codes_are_those_of_the_moment(chat: Services) -> None:
    view = AsOfView(chat, datetime(2026, 8, 12, 16, 0, tzinfo=UTC))
    assert (view.sticker_view.scope, view.sticker_view.as_of) == ("pre_holdout", view.at)
    selector = view.sticker_selector()
    assert isinstance(selector, StickerSelector) and selector.view == view.sticker_view
    assert view.sticker_selector(random.Random(1)).view == view.sticker_view
    assert view.emoji_codes is not None and view.emoji_codes.codes
    profile = view.profile
    assert profile is not None
    controller = view.sticker_rate()
    assert controller.her_share == profile.metrics.scalar("her", "sticker_share")
    assert controller.share() == pytest.approx(
        controller.her_share or 0.0
    )  # no history: at her share


def test_a_view_works_before_any_profile_or_card_exists(
    services: Services, embedder: HashingBackend
) -> None:
    view = AsOfView(services, T)
    assert view.profile is None and view.activity is None and view.her_state() is None
    assert view.persona_full() is None and view.persona_compact() is None
    assert view.emoji_codes is None
    assert view.memory_block(MemoryQuery("你好")).empty
