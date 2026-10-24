"""The Markdown layout of the persona card and its sections (R-PERS-002)."""

from __future__ import annotations

import pytest

from twin.profile.persona import compose
from twin.profile.persona.sections import (
    AUTO,
    DONT,
    MANUAL,
    STATS,
    CardFormatError,
    Tier,
    block_text,
    card_lines,
    classify_line,
    clean_line,
    split_card,
    split_label,
    split_subsections,
    subsection,
)

CARD = (
    "## [自动-统计规则]\n- 几乎不用逗号\n- 单条通常 3–8 字\n\n"
    "## [自动-描述]\n### 风格\n- 语气：轻快\n- 口头禅：哈哈哈\n- 开心时：哇塞\n"
    "- 说话禁忌：不说教\n\n### 基本情况\n- 事实：养了一只猫\n- 话题：猫和美食\n"
    "- 对用户的态度：亲近\n\n"
    "## [手动]\n### 风格\n- 称呼：宝宝\n\n### 事实\n- 生日在秋天\n\n"
    "## [不要这样]\n- 不要用句号\n"
)


def test_a_card_is_cut_into_its_sections_without_changing_a_byte() -> None:
    card = split_card(CARD)
    assert card.names() == (STATS, AUTO, MANUAL, DONT)
    assert card.assemble() == CARD
    assert card.block(STATS) == "## [自动-统计规则]\n- 几乎不用逗号\n- 单条通常 3–8 字\n\n"
    assert card.body(DONT) == "- 不要用句号\n"
    assert card.block("[unknown]") is None and card.body("[unknown]") == ""


def test_odd_whitespace_and_crlf_survive_a_replacement_of_the_other_blocks() -> None:
    manual = (
        "## [手动]  \r\n### 风格\r\n- 称呼：宝宝  \r\n\r\n\t\r\n### 事实\r\n- 生日在秋天\r\n\r\n"
    )
    text = CARD.replace(CARD[CARD.index("## [手动]") : CARD.index("## [不要这样]")], manual)
    card = split_card(text)
    assert card.block(MANUAL) == manual
    changed = card.with_block(STATS, block_text(STATS, "- 新的规则")).with_block(
        AUTO, block_text(AUTO, "### 风格\n- 语气：新的")
    )
    assert changed.block(MANUAL) == manual.encode().decode()
    assert changed.assemble().count(manual) == 1
    assert changed.block(DONT) == card.block(DONT)


@pytest.mark.parametrize(
    "text",
    ["", "no heading at all\n", "## [别的分区]\nx\n", "## [手动]\na\n## [手动]\nb\n"],
)
def test_text_that_is_not_a_card_is_refused(text: str) -> None:
    with pytest.raises(CardFormatError):
        split_card(text)


def test_a_missing_block_is_put_in_its_place_by_the_section_order() -> None:
    card = split_card("## [自动-描述]\n\n## [不要这样]\n- x\n")
    with_stats = card.with_block(STATS, block_text(STATS, "- r"))
    assert with_stats.names() == (STATS, AUTO, DONT)
    with_manual = with_stats.with_block(MANUAL, block_text(MANUAL, "### 风格"))
    assert with_manual.names() == (STATS, AUTO, MANUAL, DONT)
    with pytest.raises(CardFormatError):
        card.with_block("[x]", "## [x]\n")


def test_preamble_text_before_the_first_heading_is_kept() -> None:
    card = split_card("note\n## [手动]\n### 风格\n")
    assert card.preamble == "note\n" and card.assemble() == "note\n## [手动]\n### 风格\n"


def test_subsections_lines_labels_and_comments() -> None:
    parts = split_subsections("before\n### 风格\n- a\n\n### 事实\n* b\n")
    assert parts == {"": ["before"], "风格": ["- a", ""], "事实": ["* b"]}
    assert clean_line("  - 语气：好  ") == "语气：好"
    assert clean_line("<!-- note -->") == ""
    assert clean_line("* 星号") == "星号"
    assert split_label("口头禅：好的：呢") == ("口头禅", "好的：呢")
    assert split_label("口头禅:好") == ("口头禅", "好")
    assert split_label("没有冒号") == (None, "没有冒号")
    assert split_label("：开头") == (None, "：开头")


@pytest.mark.parametrize(
    ("section", "part", "text", "tier"),
    [
        (STATS, "", "几乎不用逗号", Tier.STATS),
        (DONT, "", "不要用句号", Tier.DONT),
        (AUTO, "风格", "口头禅：哈哈", Tier.ADDRESS),
        (AUTO, "风格", "称呼：宝宝", Tier.ADDRESS),
        (MANUAL, "风格", "称呼：宝宝", Tier.ADDRESS),
        (AUTO, "风格", "开心时：哇塞", Tier.EMOTIONS),
        (AUTO, "风格", "撒娇：嘛", Tier.EMOTIONS),
        (AUTO, "风格", "语气：轻快", Tier.OTHER),
        (AUTO, "风格", "说话禁忌：不说教", Tier.OTHER),
        (MANUAL, "风格", "没有标签的一句", Tier.OTHER),
        (MANUAL, "事实", "生日在秋天", Tier.MANUAL_FACTS),
        (AUTO, "基本情况", "事实：养猫", Tier.BASICS),
        (AUTO, "基本情况", "对用户的态度：亲近", Tier.BASICS),
        (AUTO, "基本情况", "话题：猫", Tier.TOPICS),
    ],
)
def test_lines_are_classified_into_the_priority_tiers_of_the_spec(
    section: str, part: str, text: str, tier: Tier
) -> None:
    assert classify_line(section, part, text) is tier


def test_the_tiers_run_in_the_order_of_r_pers_004() -> None:
    order = [
        Tier.STATS,
        Tier.DONT,
        Tier.ADDRESS,
        Tier.EMOTIONS,
        Tier.MANUAL_FACTS,
        Tier.BASICS,
        Tier.TOPICS,
        Tier.OTHER,
    ]
    assert sorted(order) == order and len({int(t) for t in order}) == len(order)


def test_card_lines_list_every_content_line_with_its_place() -> None:
    lines = card_lines(split_card(CARD))
    assert [line.text for line in lines][:2] == ["几乎不用逗号", "单条通常 3–8 字"]
    assert {line.section for line in lines} == {STATS, AUTO, MANUAL, DONT}
    assert [line.order for line in lines] == list(range(len(lines)))
    assert not any(line.text.startswith("###") or not line.text for line in lines)


def test_new_cards_have_the_sections_of_their_scope() -> None:
    live = split_card(
        compose.new_card("live", compose.stats_block(["r1"]), compose.empty_auto_block())
    )
    assert live.names() == (STATS, AUTO, MANUAL, DONT)
    assert "### 事实" in live.body(MANUAL) and "### 风格" in live.body(MANUAL)
    past = split_card(
        compose.new_card(
            "pre_holdout", compose.stats_block(["r1"]), compose.empty_auto_block(), ["- 称呼：宝宝"]
        )
    )
    assert past.names() == (STATS, AUTO, MANUAL)  # no corrections, no hand-written facts
    assert "### 事实" not in past.body(MANUAL) and "称呼：宝宝" in past.body(MANUAL)


def test_the_style_lines_of_the_manual_section_are_what_the_past_card_gets() -> None:
    card = split_card(CARD)
    assert compose.manual_style_lines(card) == ["- 称呼：宝宝"]
    assert compose.correction_lines(card) == ["不要用句号"]
    block = compose.pre_holdout_manual(CARD)
    assert "称呼：宝宝" in block and "生日" not in block
    assert compose.pre_holdout_manual(None) == compose.manual_block((), None)
    assert subsection("风格", []) == "### 风格\n\n"
