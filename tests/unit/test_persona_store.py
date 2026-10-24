"""Versions of the persona card, refreshing the statistics, the byte-exact manual section
(R-PERS-002, R-PERS-003, R-PERS-005, R-IMP-011)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from tests.support.persona import sticker_scenario
from tests.support.synth_chat import ChatSpec, append_texts, build_chat
from twin.profile.builder import rebuild
from twin.profile.holdout import holdout_cutoff
from twin.profile.persona import compose
from twin.profile.persona.api import load_persona, read_corrections, write_corrections
from twin.profile.persona.refresh import (
    MIN_DESCRIBED_MESSAGES,
    count_her_messages,
    description_due,
    refresh_all,
    refresh_stats,
    sync_manual_style,
    write_description,
)
from twin.profile.persona.sections import AUTO, DONT, MANUAL, STATS, block_text, split_card
from twin.profile.persona.store import PersonaStore, PersonaVersionError
from twin.services import Services
from twin.storage.state import read_state_version

ODD_MANUAL = (
    "## [手动]  \r\n### 风格\r\n- 称呼：宝宝  \r\n\r\n\t\r\n### 事实\r\n- 生日在秋天\r\n\r\n"
)
ODD_DONT = "## [不要这样]\r\n- 不要用句号  \r\n\r\n\r\n"


@pytest.fixture
def chat(services: Services) -> Services:
    build_chat(services, ChatSpec(days=30))
    rebuild(services, "all")
    return services


def store_of(services: Services) -> PersonaStore:
    return PersonaStore(services.db, services.clock)


def minimal(extra: str = "") -> str:
    return (
        compose.stats_block(["r"])
        + compose.empty_auto_block()
        + compose.manual_block(["- a"], ["- b"])
        + compose.dont_block()
        + extra
    )


# ---------------------------------------------------------------- versions


def test_versions_are_numbered_per_scope_and_chained(services: Services) -> None:
    store = store_of(services)
    assert store.active("live") is None and store.latest("live") is None
    first = store.add_version("live", minimal(), reason="generate")
    second = store.add_version("live", minimal(), reason="edit")
    other = store.add_version("pre_holdout", minimal(), reason="stats")
    assert (first.number, second.number, other.number) == (1, 2, 1)
    assert first.parent_id is None and second.parent_id == first.id and other.parent_id is None
    assert store.active("live") and store.active("live").id == second.id  # type: ignore[union-attr]
    assert store.active_id("pre_holdout") == other.id
    assert [v.number for v in store.history("live")] == [2, 1]
    assert {v.scope for v in store.history()} == {"live", "pre_holdout"}
    assert second.active and not store.get(first.id).active  # type: ignore[union-attr]
    assert second.label == "v2"


def test_a_text_that_is_not_a_card_or_a_wrong_scope_is_never_stored(services: Services) -> None:
    store = store_of(services)
    with pytest.raises(ValueError, match="section"):
        store.add_version("live", "just text", reason="edit")
    with pytest.raises(ValueError, match="scope"):
        store.add_version("elsewhere", minimal(), reason="edit")
    assert store.history() == []


def test_a_new_version_signals_the_running_application(services: Services) -> None:
    with services.db.session() as session:
        before = read_state_version(session)
    store_of(services).add_version("live", minimal(), reason="generate")
    with services.db.session() as session:
        assert read_state_version(session) > before


def test_versions_are_found_by_number_by_age_and_by_id_prefix(services: Services) -> None:
    store = store_of(services)
    versions = [store.add_version("live", minimal(), reason="edit") for _ in range(3)]
    assert store.resolve("v2", "live").id == versions[1].id
    assert store.resolve("V3", "live").id == versions[2].id
    assert store.resolve("~0", "live").id == versions[2].id
    assert store.resolve("~2", "live").id == versions[0].id
    assert store.resolve(versions[1].id.lower(), "live").id == versions[1].id
    for bad in ("v9", "~9", "xyz", "ZZZZZZZZ", "01"):
        with pytest.raises(PersonaVersionError):
            store.resolve(bad, "live")
    with pytest.raises(PersonaVersionError):
        store.resolve("v1", "pre_holdout")


def test_an_id_prefix_that_fits_two_versions_is_refused(services: Services) -> None:
    store = store_of(services)
    for _ in range(2):
        store.add_version("live", minimal(), reason="edit")
    with pytest.raises(PersonaVersionError, match="matches 2"):
        store.resolve(store.history()[0].id[:5], "live")


def test_rollback_points_at_the_old_version_and_the_next_one_continues_from_it(
    services: Services,
) -> None:
    store = store_of(services)
    first = store.add_version("live", minimal(), reason="generate")
    second = store.add_version("live", minimal(" "), reason="edit")
    back = store.rollback(first.id)
    assert back.active and store.active("live").id == first.id  # type: ignore[union-attr]
    third = store.add_version("live", minimal("  "), reason="edit")
    assert third.parent_id == first.id and third.number == 3
    assert second.id != third.id
    with pytest.raises(PersonaVersionError):
        store.rollback("NOSUCHVERSION")


def test_the_evidence_record_is_stored_with_a_version(services: Services) -> None:
    store = store_of(services)
    plain = store.add_version("live", minimal(), reason="stats")
    backed = store.add_version(
        "live", minimal(), reason="generate", provenance={"segments": {"S01": {"messages": ["m1"]}}}
    )
    assert store.provenance(plain.id) is None
    assert store.provenance(backed.id) == {"segments": {"S01": {"messages": ["m1"]}}}
    with pytest.raises(PersonaVersionError):
        store.provenance("missing")


def test_the_card_is_sealed_in_the_database(services: Services) -> None:
    secret = "养了一只叫豆包的猫"
    store_of(services).add_version(
        "live", minimal().replace("- a", f"- {secret}"), reason="edit", provenance={"x": secret}
    )
    raw = services.paths.db_path.read_bytes()
    wal = services.paths.db_path.with_name(services.paths.db_path.name + "-wal")
    if wal.exists():
        raw += wal.read_bytes()
    assert secret.encode() not in raw


# ------------------------------------------------------ the statistics section


def test_the_first_refresh_creates_a_card_with_the_statistics_only(chat: Services) -> None:
    results = refresh_all(chat)
    assert [(r.scope, r.status) for r in results[:2]] == [
        ("live", "created"),
        ("pre_holdout", "created"),
    ]
    live = load_persona(chat, "live")
    past = load_persona(chat, "pre_holdout")
    assert live is not None and past is not None
    assert live.sections.names() == (STATS, AUTO, MANUAL, DONT)
    assert past.sections.names() == (STATS, AUTO, MANUAL)
    assert "几乎不用句号" in live.text or "句号" in live.text
    assert live.described_her_messages is None and live.profile_version_id
    assert compose.correction_lines(live.sections) == []


def test_a_refresh_with_nothing_new_writes_no_version(chat: Services) -> None:
    refresh_stats(chat, "live")
    again = refresh_stats(chat, "live")
    assert again.status == "unchanged"
    assert len(store_of(chat).history("live")) == 1
    forced = refresh_stats(chat, "live", force=True)
    assert forced.status == "created" and len(store_of(chat).history("live")) == 2


def test_new_rules_replace_only_the_statistics_section(chat: Services) -> None:
    refresh_stats(chat, "live")
    store = store_of(chat)
    original = store.active("live")
    assert original is not None
    store.add_version(
        "live",
        original.sections.with_block(MANUAL, ODD_MANUAL).with_block(DONT, ODD_DONT).assemble(),
        reason="edit",
    )
    last = chat_end(chat)
    append_texts(
        chat,
        [(last + timedelta(minutes=2 * i + 1), True, "好，嗯，行，是的，对啊") for i in range(400)],
    )
    rebuild(chat, "live")
    result = refresh_stats(chat, "live")
    assert result.status == "created"
    before, after = store.history("live")[1], store.history("live")[0]
    assert before.sections.block(STATS) != after.sections.block(STATS)
    for name in (MANUAL, DONT):  # byte for byte
        assert after.sections.block(name) == before.sections.block(name)
    assert after.sections.block(MANUAL) == ODD_MANUAL and after.sections.block(DONT) == ODD_DONT
    assert after.text.encode("utf-8").count(ODD_MANUAL.encode("utf-8")) == 1


def chat_end(services: Services):  # type: ignore[no-untyped-def]
    from sqlalchemy import func, select

    from twin.storage.chat_models import Message

    with services.db.session() as session:
        return session.scalar(select(func.max(Message.create_time_utc)))


def test_no_profile_means_nothing_to_refresh(services: Services) -> None:
    result = refresh_stats(services, "live")
    assert result.status == "skipped" and "profile" in result.note
    assert store_of(services).active("live") is None


def test_the_description_is_replaced_and_the_hand_written_parts_keep_their_bytes(
    chat: Services,
) -> None:
    refresh_stats(chat, "live")
    store = store_of(chat)
    base = store.active("live")
    assert base is not None
    store.add_version(
        "live",
        base.sections.with_block(MANUAL, ODD_MANUAL).with_block(DONT, ODD_DONT).assemble(),
        reason="edit",
    )
    body = "### 风格\n- 口头禅：哈哈\n\n### 基本情况\n- 事实：养猫\n\n"
    card = write_description(
        chat,
        "live",
        body,
        provenance={"statements": []},
        template_version="persona_map@1,persona_reduce@1",
        her_messages=123,
        at=chat.clock.now_utc(),
    )
    assert card.reason == "generate" and card.described_her_messages == 123
    assert card.template_version == "persona_map@1,persona_reduce@1"
    assert "口头禅：哈哈" in card.sections.body(AUTO)
    assert card.sections.block(MANUAL) == ODD_MANUAL and card.sections.block(DONT) == ODD_DONT
    # the statistics were refreshed in the same version
    assert card.sections.body(STATS).strip() and card.profile_version_id


def test_a_first_description_creates_the_card_of_either_scope(chat: Services) -> None:
    for scope in ("live", "pre_holdout"):
        card = write_description(
            chat, scope, "### 风格\n- 语气：轻快\n\n", provenance={}, template_version="t@1",
            her_messages=50, at=chat.clock.now_utc(),
        )  # fmt: skip
        assert card.number == 1 and "语气：轻快" in card.text
    past = load_persona(chat, "pre_holdout")
    assert past is not None and past.sections.names() == (STATS, AUTO, MANUAL)


def test_a_statistics_refresh_keeps_the_description_bookkeeping(chat: Services) -> None:
    write_description(
        chat, "live", "### 风格\n- 语气：轻快\n\n", provenance={"k": 1}, template_version="t@1",
        her_messages=77, at=chat.clock.now_utc(),
    )  # fmt: skip
    refreshed = refresh_stats(chat, "live", force=True).card
    assert refreshed is not None and refreshed.reason == "stats"
    assert refreshed.described_her_messages == 77 and refreshed.template_version == "t@1"
    assert store_of(chat).provenance(refreshed.id) == {"k": 1}


# ------------------------------------------------ the pre-holdout card follows


def test_the_past_card_carries_the_style_lines_of_the_live_manual_section_only(
    chat: Services,
) -> None:
    refresh_all(chat)
    store = store_of(chat)
    live = store.active("live")
    assert live is not None
    store.add_version(
        "live",
        live.sections.with_block(
            MANUAL, compose.manual_block(["- 称呼：宝宝"], ["- 生日在秋天"])
        ).assemble(),
        reason="edit",
    )
    result = sync_manual_style(chat)
    assert result.status == "created"
    past = load_persona(chat, "pre_holdout")
    assert past is not None
    assert "称呼：宝宝" in past.sections.body(MANUAL) and "生日" not in past.text
    assert sync_manual_style(chat).status == "unchanged"
    # the same lines reach a pre-holdout card that is created later
    fresh = PersonaStore(chat.db, chat.clock)
    assert fresh.active("pre_holdout") is not None


def test_a_missing_past_card_is_not_invented_by_the_style_sync(services: Services) -> None:
    assert sync_manual_style(services).status == "skipped"


def test_a_new_past_card_starts_with_the_live_style_lines(chat: Services) -> None:
    refresh_stats(chat, "live")
    store = store_of(chat)
    live = store.active("live")
    assert live is not None
    store.add_version(
        "live",
        live.sections.with_block(MANUAL, compose.manual_block(["- 称呼：宝宝"], [])).assemble(),
        reason="edit",
    )
    refresh_stats(chat, "pre_holdout")
    past = load_persona(chat, "pre_holdout")
    assert past is not None and "称呼：宝宝" in past.text


# ------------------------------------------------------------- corrections


def test_corrections_are_written_and_read_without_touching_the_other_sections(
    chat: Services,
) -> None:
    refresh_stats(chat, "live")
    store = store_of(chat)
    base = store.active("live")
    assert base is not None
    store.add_version(
        "live", base.sections.with_block(MANUAL, ODD_MANUAL).assemble(), reason="edit"
    )
    assert read_corrections(chat) == []
    written = write_corrections(chat, ["不要用句号", "  不要   说教 ", "", "不要用句号"])
    assert read_corrections(chat) == ["不要用句号", "不要 说教"]
    assert written.reason == "corrections" and written.sections.block(MANUAL) == ODD_MANUAL
    again = write_corrections(chat, ["不要用句号", "不要 说教"])
    assert again.id == written.id  # nothing changed, no new version
    # a statistics refresh and a new description keep the corrections byte for byte
    block = written.sections.block(DONT)
    refreshed = refresh_stats(chat, "live", force=True).card
    assert refreshed is not None and refreshed.sections.block(DONT) == block
    generated = write_description(
        chat, "live", "### 风格\n- 语气：x\n\n", provenance={}, template_version="t@1",
        her_messages=40, at=chat.clock.now_utc(),
    )  # fmt: skip
    assert generated.sections.block(DONT) == block


def test_corrections_can_start_the_card(services: Services) -> None:
    card = write_corrections(services, ["不要用句号"])
    assert card.number == 1 and read_corrections(services) == ["不要用句号"]
    assert card.sections.names() == (STATS, AUTO, MANUAL, DONT)
    split_card(block_text(DONT, "x"))  # a block text on its own is a card too


# --------------------------------------------------------- when to regenerate


def test_the_description_is_due_when_never_written_and_when_her_messages_grew(
    chat: Services,
) -> None:
    holdout_cutoff(chat)
    live = description_due(chat, "live")
    assert live.due and "never" in live.reason and live.her_messages >= MIN_DESCRIBED_MESSAGES
    write_description(
        chat, "live", "### 风格\n- 语气：x\n\n", provenance={}, template_version="t@1",
        her_messages=live.her_messages, at=chat.clock.now_utc(),
    )  # fmt: skip
    settled = description_due(chat, "live")
    assert not settled.due and settled.described_messages == live.her_messages
    # 9 % more messages: below the 10 % limit
    last = chat_end(chat)
    nine = int(live.her_messages * 0.09)
    append_texts(chat, [(last + timedelta(minutes=i + 1), True, "新") for i in range(nine)])
    assert not description_due(chat, "live").due
    # and a few more cross it
    last = chat_end(chat)
    extra = int(live.her_messages * 0.03)
    append_texts(chat, [(last + timedelta(minutes=i + 1), True, "更新") for i in range(extra)])
    grown = description_due(chat, "live")
    assert grown.due and "grew" in grown.reason


def test_the_past_description_counts_only_her_messages_before_the_cutoff(chat: Services) -> None:
    cutoff = holdout_cutoff(chat)
    total = count_her_messages(chat, "live")
    before = count_her_messages(chat, "pre_holdout")
    assert 0 < before < total
    write_description(
        chat, "pre_holdout", "### 风格\n- 语气：x\n\n", provenance={}, template_version="t@1",
        her_messages=before, at=chat.clock.now_utc(),
    )  # fmt: skip
    # messages after the cutoff do not make the past description stale
    append_texts(chat, [(cutoff + timedelta(days=30, minutes=i), True, "后来") for i in range(200)])
    stale = description_due(chat, "pre_holdout")
    assert not stale.due and stale.her_messages == before


def test_too_few_messages_are_never_enough(services: Services) -> None:
    append_texts(
        services,
        [(services.clock.now_utc() - timedelta(days=1, minutes=i), True, "嗯") for i in range(5)],
    )
    due = description_due(services, "live")
    assert not due.due and "only 5" in due.reason
    assert not description_due(services, "pre_holdout").due  # no hold-out can be split yet


def test_the_scenario_helper_makes_a_splittable_conversation(services: Services) -> None:
    scenario = sticker_scenario(services)
    assert holdout_cutoff(services) > scenario.episode_time(30)


def test_a_read_only_check_never_splits_the_holdout(services: Services) -> None:
    from twin.profile.holdout import HoldoutError, get_holdout

    sticker_scenario(services)
    assert get_holdout(services) is None
    unsplit = description_due(services, "pre_holdout", split=False)
    assert not unsplit.due and "not been split" in unsplit.reason and get_holdout(services) is None
    with pytest.raises(HoldoutError):
        count_her_messages(services, "pre_holdout", split=False)
    assert count_her_messages(services, "live", split=False) == 91  # live needs no split
    holdout_cutoff(services)
    assert description_due(services, "pre_holdout", split=False).her_messages > 0
