"""Sticker tag vocabulary and how tags from different sources combine (R-STK-003/004)."""

from __future__ import annotations

from pathlib import Path

import pytest

from twin.config.lists import WordListError
from twin.config.loader import load_settings
from twin.stickers.tags import (
    MAX_TAGS,
    SOURCE_CONTEXT,
    SOURCE_MANUAL,
    SOURCE_VISION,
    TagError,
    TagVocabulary,
    effective_tags,
    load_vocabulary,
    load_vocabulary_files,
    merge_tags,
    parse_neighbors,
)

ROOT = Path(__file__).resolve().parents[2]
SPEC_TAGS = [
    "开心",
    "大笑",
    "撒娇",
    "委屈",
    "难过",
    "生气",
    "无语",
    "震惊",
    "困",
    "晚安",
    "早安",
    "亲亲",
    "抱抱",
    "加油",
    "好的",
    "拒绝",
    "调皮",
    "害羞",
    "疑问",
    "饿",
    "其他",
]


@pytest.fixture
def vocabulary() -> TagVocabulary:
    return load_vocabulary(load_settings(), ROOT)


def test_the_shipped_vocabulary_is_the_one_in_the_spec(vocabulary: TagVocabulary) -> None:
    assert list(vocabulary.tags) == SPEC_TAGS
    assert "开心" in vocabulary and "不存在" not in vocabulary


def test_every_neighbour_is_a_tag_and_every_tag_has_an_entry(vocabulary: TagVocabulary) -> None:
    assert set(vocabulary.neighbors) == set(vocabulary.tags)
    for tag, nearby in vocabulary.neighbors.items():
        assert tag not in nearby
        assert set(nearby) <= set(vocabulary.tags)
        assert len(set(nearby)) == len(nearby)
    assert vocabulary.near("撒娇")[0] == "委屈"
    assert vocabulary.near("其他") == ()
    assert vocabulary.near("没有这个标签") == ()


def test_related_tags_are_equal_or_neighbours_in_either_direction(
    vocabulary: TagVocabulary,
) -> None:
    assert vocabulary.related("开心", "开心")
    assert vocabulary.related("开心", "大笑") and vocabulary.related("大笑", "开心")
    assert vocabulary.related("早安", "加油") and vocabulary.related("加油", "早安")
    assert not vocabulary.related("生气", "晚安")


def test_check_accepts_only_known_tags_and_removes_duplicates(vocabulary: TagVocabulary) -> None:
    assert vocabulary.check(["开心", "开心", "调皮"]) == ["开心", "调皮"]
    with pytest.raises(TagError, match="不存在"):
        vocabulary.check(["开心", "不存在"])
    with pytest.raises(TagError, match="allowed"):
        vocabulary.check(["x"])


def test_the_files_can_be_replaced_by_the_settings(tmp_path: Path) -> None:
    (tmp_path / "tags.txt").write_text("# tags\n甲\n乙\n甲\n", encoding="utf-8")
    (tmp_path / "near.yaml").write_text("甲: [乙]\n乙: []\n", encoding="utf-8")
    custom = load_vocabulary_files(tmp_path / "tags.txt", tmp_path / "near.yaml")
    assert custom.tags == ("甲", "乙") and custom.near("甲") == ("乙",)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("甲: [丙]\n", "丙"),
        ("丙: [甲]\n", "丙"),
        ("甲: 乙\n", "must be a list"),
        ("- 甲\n- 乙\n", "mapping"),
    ],
)
def test_a_bad_neighbour_table_is_refused(text: str, message: str) -> None:
    with pytest.raises(WordListError, match=message):
        parse_neighbors(text, ["甲", "乙"])


def test_empty_and_missing_files_are_refused(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text("# nothing\n", encoding="utf-8")
    (tmp_path / "near.yaml").write_text("{}\n", encoding="utf-8")
    with pytest.raises(WordListError, match="lists no tags"):
        load_vocabulary_files(tmp_path / "empty.txt", tmp_path / "near.yaml")
    (tmp_path / "tags.txt").write_text("甲\n", encoding="utf-8")
    with pytest.raises(WordListError, match="cannot read"):
        load_vocabulary_files(tmp_path / "tags.txt", tmp_path / "missing.yaml")


# ------------------------------------------------------------------ combining


def test_the_picture_alone_decides_when_there_is_no_other_evidence(
    vocabulary: TagVocabulary,
) -> None:
    assert effective_tags(["开心", "大笑"], None, None, vocabulary) == (
        ["开心", "大笑"],
        SOURCE_VISION,
    )
    assert effective_tags([], [], None, vocabulary) == ([], None)
    assert effective_tags(None, None, None, vocabulary) == ([], None)
    assert effective_tags(["a", "b", "c", "d"], None, None, vocabulary)[0] == ["a", "b", "c"]


def test_her_use_outweighs_the_picture(vocabulary: TagVocabulary) -> None:
    # the picture says happy, her use says wronged: the use comes first and the contradicting
    # visual tag is dropped; a visual tag next to a used one is kept
    tags, source = effective_tags(["开心", "调皮"], ["委屈"], None, vocabulary)
    assert source == SOURCE_CONTEXT and tags == ["委屈"]
    tags, _ = effective_tags(["撒娇", "晚安"], ["委屈"], None, vocabulary)
    assert tags == ["委屈", "撒娇"]  # 撒娇 is near 委屈; 晚安 is not
    assert merge_tags(["开心"], [], vocabulary) == ["开心"]
    assert merge_tags(["委屈"], ["委屈", "难过", "撒娇", "生气"], vocabulary) == [
        "委屈",
        "难过",
        "撒娇",
    ]


def test_hand_set_tags_replace_everything(vocabulary: TagVocabulary) -> None:
    assert effective_tags(["开心"], ["委屈"], ["晚安", "困"], vocabulary) == (
        ["晚安", "困"],
        SOURCE_MANUAL,
    )
    assert effective_tags(["开心"], ["委屈"], [], vocabulary)[1] == SOURCE_CONTEXT
    assert (
        len(effective_tags(None, None, ["开心"] * 2 + ["晚安", "困", "饿"], vocabulary)[0])
        <= MAX_TAGS
    )
