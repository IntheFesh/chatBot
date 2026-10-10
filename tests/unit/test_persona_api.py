"""What later rounds import: loading and rendering the card (R-PERS-004, R-PERS-005, R-TRN-013)."""

from __future__ import annotations

from twin.profile.persona import compose
from twin.profile.persona.api import load_persona, render_compact, render_full
from twin.profile.persona.sections import DONT, MANUAL
from twin.profile.persona.store import PersonaStore
from twin.services import Services

FACT = "养了一只叫豆包的猫"
CORRECTION = "不要把话说得太正式"


def live_card(style: str = "哈哈") -> str:
    description = f"### 风格\n- 口头禅：{style}\n- 语气：轻快\n\n### 基本情况\n- 事实：{FACT}\n\n"
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


def test_no_card_means_nothing_to_load_or_render(services: Services) -> None:
    assert load_persona(services) is None and load_persona(services, "pre_holdout") is None
    assert render_full(services) is None and render_compact(services, "pre_holdout") is None


def test_the_version_in_force_is_loaded_unless_one_is_named(services: Services) -> None:
    store = PersonaStore(services.db, services.clock)
    first = store.add_version("live", live_card("哈哈"), reason="generate")
    store.add_version("live", live_card("嘿嘿"), reason="edit")
    assert (load_persona(services) or first).number == 2
    named = load_persona(services, "live", "v1")
    assert named is not None and named.id == first.id
    older = render_compact(services, version="v1")
    newer = render_compact(services)
    assert older is not None and newer is not None
    assert "哈哈" in older.text and "嘿嘿" in newer.text
    assert (older.number, newer.number) == (1, 2) and older.scope == newer.scope == "live"
    assert older.kind == newer.kind == "compact" and older.version_id == first.id


def test_the_budgets_come_from_the_settings(services: Services) -> None:
    PersonaStore(services.db, services.clock).add_version("live", live_card(), reason="generate")
    full, compact = render_full(services), render_compact(services)
    assert full is not None and compact is not None
    assert (full.budget, compact.budget) == (1500, 400)
    services.settings.persona.compact_max_tokens = 12
    tight = render_compact(services)
    assert tight is not None and tight.budget == 12 and tight.tokens <= 12 and tight.dropped > 0
    assert "几乎不用逗号" in tight.text  # the statistics rules are the last to go


def test_the_compact_card_has_no_fact_and_no_correction_but_the_full_card_has_both(
    services: Services,
) -> None:
    PersonaStore(services.db, services.clock).add_version("live", live_card(), reason="generate")
    full, compact = render_full(services), render_compact(services)
    assert full is not None and compact is not None
    for text in (FACT, "生日在秋天", CORRECTION):
        assert text in full.text and text not in compact.text
    assert "称呼：宝宝" in compact.text and "口头禅：哈哈" in compact.text


def test_the_evaluation_card_adds_the_live_corrections_to_the_past_full_card_only(
    services: Services,
) -> None:
    store = PersonaStore(services.db, services.clock)
    store.add_version("live", live_card(), reason="generate")
    store.add_version("pre_holdout", past_card(), reason="generate")
    plain = render_full(services, "pre_holdout")
    sandbox = render_full(services, "pre_holdout", with_live_corrections=True)
    compact = render_compact(services, "pre_holdout")
    assert plain is not None and sandbox is not None and compact is not None
    assert CORRECTION not in plain.text and CORRECTION in sandbox.text
    assert CORRECTION not in compact.text  # the compact card never has them
    assert (
        "生日在秋天" not in sandbox.text and FACT in sandbox.text
    )  # past facts, no live hand facts
    assert sandbox.scope == "pre_holdout" and sandbox.number == 1
    live = render_full(services, with_live_corrections=True)
    assert live is not None and live.text.count(CORRECTION) == 1  # not added twice to the live card


def test_without_a_live_card_the_sandbox_gets_no_corrections(services: Services) -> None:
    PersonaStore(services.db, services.clock).add_version(
        "pre_holdout", past_card(), reason="stats"
    )
    sandbox = render_full(services, "pre_holdout", with_live_corrections=True)
    assert sandbox is not None and "不要这样" not in sandbox.text


def test_the_corrections_section_is_the_one_the_next_round_writes(services: Services) -> None:
    from twin.profile.persona.api import read_corrections, write_corrections

    PersonaStore(services.db, services.clock).add_version("live", live_card(), reason="generate")
    assert read_corrections(services) == [CORRECTION]
    write_corrections(services, ["尽量别用句号"])
    card = load_persona(services)
    assert card is not None and card.sections.block(MANUAL) == compose.manual_block(
        ["- 称呼：宝宝"], ["- 生日在秋天"]
    )
    assert "尽量别用句号" in (card.sections.block(DONT) or "")
