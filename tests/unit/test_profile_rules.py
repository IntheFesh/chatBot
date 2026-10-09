"""The numeric style rules and their template table (R-PROF-005)."""

from __future__ import annotations

from pathlib import Path

import pytest

from twin.config.lists import locate_list_file
from twin.config.settings import Settings
from twin.profile.distribution import EmpiricalDistribution
from twin.profile.rules import (
    FACT_DOC,
    FACT_NAMES,
    Rule,
    RuleError,
    evaluate,
    generate_rules,
    load_rules,
    parse_condition,
    parse_rules,
    rule_facts,
)
from twin.profile.snapshot import ProfileMetrics, assemble_metrics
from twin.profile.values import Dist, Leaf, Rates, Scalar

ROOT = Path(__file__).resolve().parents[2]


def dist(counts: dict[float, int]) -> Dist:
    return Dist(EmpiricalDistribution.from_counter(counts, discrete=True))


def metrics_with(**leaves: Leaf) -> ProfileMetrics:
    """A profile of her with neutral values, some of them replaced."""
    her: dict[str, Leaf] = {
        "punct_rate": Rates({"comma": 0.4, "exclaim": 0.0, "tilde": 0.0, "ellipsis": 0.0}, 500),
        "end_rate": Rates({"period": 0.2}, 500),
        "question_rate": Scalar(0.0, 500),
        "text_length": dist({10: 50, 20: 50}),
        "burst_size": dist({1: 100}),
        "burst_gap_s": dist({30: 100}),
        "emoji_code_rate": Scalar(0.01, 500),
        "emoji_code_freq": Rates({}, 0),
        "emoji_code_run_length": Dist(EmpiricalDistribution.empty(discrete=True)),
        "unicode_emoji_rate": Scalar(0.0, 500),
        "unicode_emoji_freq": Rates({}, 0),
        "sticker_share": Scalar(0.01, 600),
        "quote_rate": Scalar(0.0, 500),
        "laugh_rate": Scalar(0.0, 500),
        "laugh_length": Dist(EmpiricalDistribution.empty(discrete=True)),
        "final_particle_rate": Scalar(0.0, 500),
        "final_particle_freq": Rates({}, 0),
        "reply_latency_s": Dist(EmpiricalDistribution.empty(discrete=True)),
    }
    her.update(leaves)
    document = assemble_metrics(
        scope="live",
        config={},
        weight=0.6,
        full={"her": her, "user": {}},
        recent=None,
        full_info={},
        recent_info={},
    )
    return ProfileMetrics(document)


def repo_rules() -> list[Rule]:
    return load_rules(locate_list_file(ROOT, Settings().profile.rules_file))


def lines_for(**leaves: Leaf) -> list[str]:
    return generate_rules(metrics_with(**leaves), repo_rules())


def has(lines: list[str], *fragments: str) -> bool:
    return any(all(fragment in line for fragment in fragments) for line in lines)


# ---------------------------------------------------------------- every branch


def test_comma_rules() -> None:
    low = lines_for(punct_rate=Rates({"comma": 0.03}, 500))
    assert has(low, "几乎不用逗号，用分条代替")
    middle = lines_for(punct_rate=Rates({"comma": 0.10}, 500))
    assert has(middle, "偶尔用逗号（约 10% 的消息）") and not has(middle, "几乎不用逗号")
    high = lines_for(punct_rate=Rates({"comma": 0.45}, 500))
    assert has(high, "经常用逗号（约 45% 的消息）")


def test_full_stop_rules() -> None:
    assert has(lines_for(end_rate=Rates({"period": 0.002}, 500)), "几乎不用句号结尾")
    assert has(lines_for(end_rate=Rates({"period": 0.10}, 500)), "偶尔用句号结尾（约 10%）")
    assert has(lines_for(end_rate=Rates({"period": 0.40}, 500)), "常用句号结尾（约 40%）")


def test_length_rule_states_the_typical_range_the_median_and_the_tail() -> None:
    lines = lines_for(text_length=dist({3: 25, 5: 25, 6: 25, 8: 20, 12: 5}))
    assert has(lines, "单条通常 4–6 字", "中位数 6 字", "很少超过 8 字")
    assert has(lines_for(text_length=dist({5: 100})), "单条通常 5 字，中位数 5 字")


def test_burst_rules() -> None:
    many = lines_for(burst_size=dist({2: 50, 3: 30, 6: 20}))
    assert has(many, "常连发 2–3 条")
    single = lines_for(burst_size=dist({1: 85, 2: 10, 4: 5}))
    assert has(single, "多数时候一次只发 1 条") and not has(single, "常连发")
    assert has(lines_for(burst_gap_s=dist({6: 100})), "连发的条与条之间间隔很短（中位约 6 秒）")
    assert not has(lines_for(burst_gap_s=dist({40: 100})), "间隔很短")


def test_emoji_code_rules() -> None:
    common = lines_for(
        emoji_code_rate=Scalar(0.035, 500),
        emoji_code_freq=Rates({"[拥抱]": 0.5, "[亲亲]": 0.3, "[捂脸]": 0.15, "[抱抱]": 0.05}, 80),
        emoji_code_run_length=dist({1: 60, 3: 40}),
    )
    assert has(common, "常用微信表情代码 [拥抱][亲亲][捂脸]", "3.5%")
    assert has(common, "表情代码常连用 3 个")
    assert has(lines_for(emoji_code_rate=Scalar(0.004, 500)), "很少用微信表情代码")


def test_emoji_character_rule() -> None:
    lines = lines_for(
        unicode_emoji_rate=Scalar(0.06, 500), unicode_emoji_freq=Rates({"😂": 0.6, "👍": 0.4}, 30)
    )
    assert has(lines, "会用 emoji 字符（约 6.0% 的消息），常见 😂👍")
    assert not has(lines_for(), "emoji 字符")


def test_sticker_and_quote_rules() -> None:
    assert has(lines_for(sticker_share=Scalar(0.105, 600)), "经常发表情包（约占全部消息的 10%）")
    assert has(lines_for(sticker_share=Scalar(0.01, 600)), "很少发表情包")
    assert has(lines_for(quote_rate=Scalar(0.064, 500)), "有时用引用回复（约占文字消息的 6%）")
    assert not has(lines_for(quote_rate=Scalar(0.01, 500)), "引用回复")


def test_laughter_particle_and_question_rules() -> None:
    laugh = lines_for(laugh_rate=Scalar(0.05, 500), laugh_length=dist({3: 70, 5: 30}))
    assert has(laugh, "笑的时候常写成 3 个“哈”")
    particles = lines_for(
        final_particle_rate=Scalar(0.3, 500),
        final_particle_freq=Rates({"啊": 0.5, "呀": 0.3, "呢": 0.2}, 150),
    )
    assert has(particles, "句末常带语气词 啊、呀、呢（约 30% 的消息）")
    assert has(lines_for(question_rate=Scalar(0.2, 500)), "爱提问，约 20% 的消息是问句")


def test_punctuation_style_rules() -> None:
    marks = Rates({"comma": 0.03, "exclaim": 0.2, "tilde": 0.1, "ellipsis": 0.05}, 500)
    lines = lines_for(punct_rate=marks)
    assert has(lines, "常用感叹号（约 20%")
    assert has(lines, "常用波浪号～（约 10%")
    assert has(lines, "有时用省略号（约 5%")


def test_latency_rule() -> None:
    lines = lines_for(reply_latency_s=dist({15: 50, 20: 40, 300: 10}))
    assert has(lines, "回复通常很快（中位约 18 秒）", "可以到 48 秒以上")


def test_facts_without_data_make_their_rules_silent() -> None:
    quiet = ProfileMetrics(
        assemble_metrics(
            scope="live",
            config={},
            weight=0.6,
            full={"her": {}, "user": {}},
            recent=None,
            full_info={},
            recent_info={},
        )
    )
    facts = rule_facts(quiet)
    assert set(facts) == FACT_NAMES and all(value is None for value in facts.values())
    assert generate_rules(quiet, repo_rules()) == []


# ------------------------------------------------------- the table and its language


def test_the_shipped_table_is_valid_and_every_fact_is_documented() -> None:
    rules = repo_rules()
    assert len(rules) >= 20 and len({r.id for r in rules}) == len(rules)
    used = set()
    for rule in rules:
        parse_condition(rule.when)
        used.update(name for name in FACT_NAMES if name in rule.when or "{" + name in rule.text)
    assert used <= set(FACT_DOC)
    assert set(FACT_DOC) == FACT_NAMES


@pytest.mark.parametrize(
    "text",
    [
        "rules: []",
        "- not a mapping",
        "rules:\n  - {id: a, when: 'comma_rate < 1'}",
        "rules:\n  - {id: a, when: 'comma_rate <', text: x}",
        "rules:\n  - {id: a, when: 'no_such_fact < 1', text: x}",
        "rules:\n  - {id: a, when: 'abs(comma_rate) < 1', text: x}",
        "rules:\n  - {id: a, when: 'comma_rate in [1]', text: x}",
        "rules:\n  - {id: a, when: 'comma_rate < 1', text: '{unknown}'}",
        (
            "rules:\n  - {id: a, when: 'comma_rate < 1', text: x}"
            "\n  - {id: a, when: 'comma_rate < 2', text: y}"
        ),
        "rules: [unclosed",
    ],
)
def test_a_malformed_table_is_refused(text: str) -> None:
    with pytest.raises(RuleError):
        parse_rules(text)


def test_conditions_are_evaluated_without_executing_anything() -> None:
    facts: dict[str, float | str | None] = {"comma_rate": 0.1, "length_median": None}
    assert evaluate(parse_condition("comma_rate < 0.2 and comma_rate >= 0.05"), facts) is True
    assert evaluate(parse_condition("comma_rate > 0.2 or comma_rate == 0.1"), facts) is True
    assert evaluate(parse_condition("not comma_rate > 0.2"), facts) is True
    assert evaluate(parse_condition("0 < comma_rate < 0.2"), facts) is True
    assert evaluate(parse_condition("0.2 < comma_rate < 0.3"), facts) is False
    assert evaluate(parse_condition("length_median > 3"), facts) is False  # missing fact
    assert evaluate(parse_condition("-comma_rate < 0"), facts) is True
    assert evaluate(parse_condition("final_particle_top == 'x'"), facts) is False


def test_a_missing_or_unreadable_table_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(RuleError, match="cannot read"):
        load_rules(tmp_path / "absent.yaml")
    path = tmp_path / "rules.yaml"
    path.write_text(
        "rules:\n  - id: only\n    when: 'comma_rate < 1'\n    text: '逗号 {comma_pct:.0f}'\n",
        encoding="utf-8",
    )
    (rule,) = load_rules(path)
    assert rule.id == "only"
    metrics = metrics_with(punct_rate=Rates({"comma": 0.5}, 10))
    assert generate_rules(metrics, [rule]) == ["逗号 50"]
