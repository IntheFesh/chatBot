"""The full and the compact rendering of the persona card (R-PERS-004, R-TRN-013)."""

from __future__ import annotations

import pytest

from twin.profile.persona.compose import (
    auto_block,
    dont_block,
    manual_block,
    new_card,
    stats_block,
)
from twin.profile.persona.render import RenderedPersona, count_tokens, render_card
from twin.profile.persona.sections import split_card

FACT = "养了一只叫豆包的猫"
TOPIC = "常聊猫和美食"
MANUAL_FACT = "生日在十月十五日"
CORRECTION = "不要把话说得太正式"


def card(*, rules: int = 3, facts: bool = True) -> str:
    style = [
        "- 语气：轻快",
        "- 口头禅：哈哈哈",
        "- 称呼：宝宝",
        "- 开心时：哇塞",
        "- 撒娇时：嘛嘛",
        "- 说话禁忌：不说教",
    ]
    basics = [f"- 事实：{FACT}", f"- 话题：{TOPIC}", "- 对用户的态度：亲近"] if facts else []
    body = "### 风格\n" + "\n".join(style) + "\n\n### 基本情况\n" + "\n".join(basics) + "\n\n"
    return (
        stats_block([f"规则{i}：" + "说话简短而随意" * 4 for i in range(rules)])
        + auto_block(body)
        + manual_block(["- 手写风格：爱用叠词"], [f"- {MANUAL_FACT}"])
        + dont_block([CORRECTION])
    )


def render(text: str, kind: str, budget: int) -> RenderedPersona:
    return render_card(
        text,
        kind=kind,
        scope="live",
        version_id="V1",
        number=3,
        budget=budget,  # type: ignore[arg-type]
    )


def test_the_compact_rendering_has_style_only_and_never_a_fact_or_a_correction() -> None:
    rendered = render(card(), "compact", 400)
    for forbidden in (FACT, TOPIC, MANUAL_FACT, CORRECTION, "基本情况", "不要这样", "补充（事实）"):
        assert forbidden not in rendered.text
    for style in ("规则0", "哈哈哈", "宝宝", "哇塞", "爱用叠词", "轻快"):
        assert style in rendered.text
    assert rendered.kind == "compact" and rendered.dropped == 0


def test_the_full_rendering_has_every_section() -> None:
    rendered = render(card(), "full", 1500)
    for expected in (FACT, TOPIC, MANUAL_FACT, CORRECTION, "规则0", "哈哈哈", "爱用叠词"):
        assert expected in rendered.text
    assert [h for h in rendered.text.splitlines() if h.startswith("## ")] == [
        "## 数字规则",
        "## 风格",
        "## 基本情况",
        "## 补充（风格）",
        "## 补充（事实）",
        "## 不要这样",
    ]


def test_the_rendering_carries_the_version_and_the_scope_of_its_card() -> None:
    rendered = render(card(), "compact", 400)
    assert (rendered.version_id, rendered.number, rendered.scope) == ("V1", 3, "live")
    assert str(rendered) == rendered.text and rendered.fits
    assert rendered.tokens == count_tokens(rendered.text) and rendered.budget == 400


def test_rendering_the_same_card_twice_gives_the_same_text() -> None:
    assert render(card(), "full", 900).text == render(card(), "full", 900).text


def kept(text: str) -> dict[str, bool]:
    return {
        "stats": "规则0" in text,
        "dont": CORRECTION in text,
        "address": "哈哈哈" in text and "宝宝" in text,
        "emotions": "哇塞" in text,
        "manual_facts": MANUAL_FACT in text,
        "basics": FACT in text,
        "topics": TOPIC in text,
        "other": "轻快" in text or "不说教" in text or "爱用叠词" in text,
    }


def test_the_full_rendering_is_cut_in_the_order_of_the_spec() -> None:
    text = card(rules=6)
    full = render(text, "full", 1500)
    assert all(kept(full.text).values())
    order = ["stats", "dont", "address", "emotions", "manual_facts", "basics", "topics", "other"]
    seen_losses: list[str] = []
    for budget in range(full.tokens, 0, -3):
        lost = [name for name, ok in kept(render(text, "full", budget).text).items() if not ok]
        # whatever is lost at a smaller budget is lost together with everything of lower priority
        for name in lost:
            for lower in order[order.index(name) + 1 :]:
                assert lower in lost, (budget, name, lower)
        for name in lost:
            if name not in seen_losses:
                seen_losses.append(name)
    assert seen_losses == order[::-1]  # the last tier goes first, the statistics last


def test_the_compact_rendering_is_cut_statistics_first_kept_then_address_moods_the_rest() -> None:
    text = card(rules=6)
    compact = render(text, "compact", 400)
    assert kept(compact.text)["stats"] and kept(compact.text)["address"]
    order = ["stats", "address", "emotions", "other"]
    seen_losses: list[str] = []
    for budget in range(compact.tokens, 0, -2):
        lost = [
            n
            for n, ok in kept(render(text, "compact", budget).text).items()
            if not ok and n in order
        ]
        for name in lost:
            for lower in order[order.index(name) + 1 :]:
                assert lower in lost, (budget, name, lower)
            if name not in seen_losses:
                seen_losses.append(name)
    assert seen_losses == order[::-1]


@pytest.mark.parametrize("kind", ["full", "compact"])
def test_a_long_card_never_exceeds_its_budget(kind: str) -> None:
    long_card = card(rules=90)
    budget = 1500 if kind == "full" else 400
    rendered = render(long_card, kind, budget)
    assert rendered.tokens <= budget and rendered.dropped > 0
    unlimited = render(long_card, kind, 10**6)
    assert unlimited.tokens > budget and unlimited.dropped == 0


def test_extra_corrections_are_added_to_the_full_rendering_only() -> None:
    past = new_card("pre_holdout", stats_block(["规则0"]), auto_block("### 风格\n- 口头禅：哈\n\n"))
    extra = ["不要用句号", CORRECTION, CORRECTION]
    full = render_card(
        past,
        kind="full",
        scope="pre_holdout",
        version_id="P",
        number=1,
        budget=1500,
        extra_dont=extra,
    )
    assert "## 不要这样" in full.text and full.text.count(CORRECTION) == 1
    compact = render_card(
        past,
        kind="compact",
        scope="pre_holdout",
        version_id="P",
        number=1,
        budget=400,
        extra_dont=extra,
    )
    assert "不要这样" not in compact.text and CORRECTION not in compact.text


def test_an_empty_or_unlabelled_card_renders_to_nothing_without_failing() -> None:
    blank = new_card("live", stats_block([]), auto_block(""))
    assert render(blank, "full", 1500).text == ""
    assert render(blank, "compact", 400).tokens == 0
    odd = split_card("## [自动-描述]\n无小节的一行\n").assemble()
    assert render(odd, "full", 1500).text == ""
