"""What one text contains: punctuation, codes, emoji, particles, laughter (R-PROF-002)."""

from __future__ import annotations

import pytest

from twin.profile.textstats import (
    analyse_text,
    emoji_code_runs,
    ending_class,
    final_particle,
    find_emoji_codes,
    find_unicode_emoji,
    has_latin_laugh,
    is_question,
    laugh_runs,
    punctuation_groups,
    strip_trailing,
)


def test_punctuation_groups_cover_the_listed_marks() -> None:
    assert punctuation_groups("你好，吃饭了吗？") == {"comma", "question"}
    assert punctuation_groups("好的。") == {"period"}
    assert punctuation_groups("太好了！！") == {"exclaim"}
    assert punctuation_groups("嗯嗯～ 好") == {"tilde", "space"}
    assert punctuation_groups("等一下…") == {"ellipsis"}
    assert punctuation_groups("苹果、香蕉") == {"pause"}
    assert punctuation_groups("a, b. c? d! e~") == {
        "comma",
        "period",
        "question",
        "exclaim",
        "tilde",
        "space",
    }
    assert punctuation_groups("3.14 和 v2.0") == {"space"}  # decimal points are no full stops
    assert punctuation_groups("等等...") == {"ellipsis"}
    assert punctuation_groups("没有标点") == frozenset()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("好的。", "period"),
        ("真的吗？", "question"),
        ("哈哈！", "exclaim"),
        ("嗯，", "comma"),
        ("好呀～", "tilde"),
        ("嗯…", "ellipsis"),
        ("等一下...", "ellipsis"),
        ("在吗[拥抱]", "emoji_code"),
        ("晚安", "none"),
        ("   ", "none"),
        ("好的  ", "none"),
    ],
)
def test_how_a_message_ends(text: str, expected: str) -> None:
    assert ending_class(text) == expected


def test_emoji_codes_are_found_and_runs_counted() -> None:
    text = "来了[拥抱][拥抱][亲亲]然后[Emm]"
    assert find_emoji_codes(text) == ["[拥抱]", "[拥抱]", "[亲亲]", "[Emm]"]
    assert emoji_code_runs(text) == [3, 1]
    assert emoji_code_runs("没有") == []
    assert find_emoji_codes("[图片:x] [12]") == []  # digits and long text are not codes


@pytest.mark.parametrize(
    ("text", "count"),
    [
        ("开心😂", 1),
        ("❤️和👍🏽", 2),
        ("全家👨‍👩‍👧福", 1),
        ("国旗🇨🇳", 1),
        ("1️⃣第一", 1),
        ("©版权™", 0),
        ("纯文字 abc", 0),
        ("☀️晴天🌧️雨", 2),
    ],
)
def test_unicode_emoji_are_recognised_as_whole_sequences(text: str, count: int) -> None:
    assert len(find_unicode_emoji(text)) == count


def test_final_particles_ignore_trailing_marks_and_codes() -> None:
    assert final_particle("好啊") == "啊"
    assert final_particle("好啊！！") == "啊"
    assert final_particle("嗯呢[亲亲]") == "呢"
    assert final_particle("好的") is None
    assert final_particle("") is None
    assert final_particle("。。") is None
    assert strip_trailing("嗯呢～[拥抱]。 ") == "嗯呢"


def test_laughter_and_questions() -> None:
    assert laugh_runs("哈哈哈哈好笑哈哈") == [4, 2]
    assert laugh_runs("哈？") == []  # a single 哈 is an interjection, not a laugh
    assert has_latin_laugh("hhhh") and not has_latin_laugh("hh")
    assert is_question("你吃了吗") and is_question("what?") and is_question("什么？")
    assert not is_question("吃了")


def test_analyse_text_collects_everything_once() -> None:
    facts = analyse_text("  哈哈哈，你在干嘛呀？[捂脸][捂脸] ")
    assert facts.length == len("哈哈哈，你在干嘛呀？[捂脸][捂脸]")
    assert facts.groups == {"comma", "question"}
    assert facts.ending == "emoji_code" and facts.code_runs == [2]
    assert facts.laughs == [3] and facts.question and facts.particle == "呀"
    assert facts.emoji == [] and not facts.latin_laugh


def test_the_shipped_code_table_only_lists_codes_the_pattern_can_find() -> None:
    from pathlib import Path

    from twin.config.lists import load_word_list, locate_list_file
    from twin.config.settings import Settings

    root = Path(__file__).resolve().parents[2]
    entries = load_word_list(locate_list_file(root, Settings().profile.emoji_codes_file))
    assert len(entries) >= 100 and len(set(entries)) == len(entries)
    for name in entries:
        assert find_emoji_codes(f"[{name}]") == [f"[{name}]"], name
    for common in ("微笑", "拥抱", "亲亲", "捂脸", "Emm", "OK"):
        assert common in entries
