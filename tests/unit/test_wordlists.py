"""Shipped word lists and their loader (R-CFG-005; content for R-ENG-008, R-SAFE-001/002)."""

from __future__ import annotations

from pathlib import Path

import pytest

from twin.config.lists import WordListError, load_regex_list, load_word_list

LISTS = Path(__file__).resolve().parents[2] / "config" / "lists"


def test_loader_skips_comments_blanks_and_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "list.txt"
    path.write_text("# comment\n\n  alpha  \nbeta\nalpha\n#hash\nwith # inside\n", encoding="utf-8")
    assert load_word_list(path) == ["alpha", "beta", "with # inside"]


def test_loader_reports_unreadable_files_and_bad_regexes(tmp_path: Path) -> None:
    with pytest.raises(WordListError, match="cannot read"):
        load_word_list(tmp_path / "missing.txt")
    bad = tmp_path / "bad.txt"
    bad.write_text("fine\n(unclosed\n", encoding="utf-8")
    with pytest.raises(WordListError, match="invalid regular expression"):
        load_regex_list(bad)


def test_ai_phrases_cover_the_spec_examples() -> None:
    phrases = load_word_list(LISTS / "ai_phrases.txt")
    for required in ("作为AI", "人工智能", "语言模型", "希望对你有帮助", "有什么可以帮你", "请注意", "总之"):
        assert required in phrases
    assert len(phrases) >= 120
    assert all(phrase == phrase.strip() for phrase in phrases)


def test_crisis_keywords_cover_self_harm_despair_and_harm_to_others() -> None:
    words = load_word_list(LISTS / "crisis_keywords.txt")
    for required in ("自杀", "想死", "不想活了", "割腕", "跳楼", "自残", "绝望", "杀人", "suicide"):
        assert required in words
    assert len(words) >= 150


@pytest.mark.parametrize(
    "sentence",
    [
        "我明天给你打电话",
        "等下给你发个语音",
        "我马上转你100",
        "我给你寄点吃的",
        "周末见面吧",
        "我现在开视频",
        "我给你发张照片",
        "我去找你",
        "红包发你了",
        "我给你买单",
        "我等下去你家做饭",
    ],
)
def test_commitment_patterns_catch_promises_of_real_world_actions(sentence: str) -> None:
    patterns = load_regex_list(LISTS / "commitment_patterns.txt")
    assert any(pattern.search(sentence) for pattern in patterns), sentence


@pytest.mark.parametrize(
    "sentence",
    ["今天吃什么", "好烦啊", "我在上班", "哈哈哈哈", "你吃饭了吗", "晚安", "我刚到家", "好困"],
)
def test_commitment_patterns_leave_ordinary_chat_alone(sentence: str) -> None:
    patterns = load_regex_list(LISTS / "commitment_patterns.txt")
    assert not any(pattern.search(sentence) for pattern in patterns), sentence
