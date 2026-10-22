"""Shipped word lists and their loader (R-CFG-005; content for R-ENG-008, R-SAFE-001/002)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from twin.config.lists import WordListError, load_regex_list, load_word_list
from twin.engine.postprocess.phrases import AiPhrases, load_ai_phrases

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
    for required in (
        "作为AI",
        "人工智能",
        "语言模型",
        "希望对你有帮助",
        "有什么可以帮你",
        "请注意",
        "总之",
    ):
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
        "我拍给你看",  # a photo without the word for it
        "晚点拍给你看哈",
        "拍个照给你看",
        "我录个视频给你看",
        "我给你拍一张",
        "我拍一下发你",
        "我明天去找你",  # a time between "I" and "go"
        "我后天就过去看你",
        "我周末来接你",
    ],
)
def test_commitment_patterns_catch_promises_of_real_world_actions(sentence: str) -> None:
    patterns = load_regex_list(LISTS / "commitment_patterns.txt")
    assert any(pattern.search(sentence) for pattern in patterns), sentence


@pytest.mark.parametrize(
    "sentence",
    [
        "今天吃什么",
        "好烦啊",
        "我在上班",
        "哈哈哈哈",
        "你吃饭了吗",
        "晚安",
        "我刚到家",
        "好困",
        "你拍的照片发给我看看",  # asking for one is not promising one
        "照片拍得好好看",
        "我在拍作业呢",
        "他们拍给你们看了",
        "录取结果出来了没",
        "你明天来找我吗",
        "我明天去上课",
    ],
)
def test_commitment_patterns_leave_ordinary_chat_alone(sentence: str) -> None:
    patterns = load_regex_list(LISTS / "commitment_patterns.txt")
    assert not any(pattern.search(sentence) for pattern in patterns), sentence


# ----------------------------------------- R17-37: the lists hit promises, not everyday speech
# (D-671).  Every group of commitment_patterns.txt and of ai_phrases.txt has examples it must
# keep catching (recall is not loosened) and everyday sentences it must leave alone; before round
# 17 the negatives below were deleted or rewritten by the bot's post-processor.

GROUP_HEADER = re.compile(r"^#\s*---\s*(?P<title>.+?)\s*$")


def commitment_groups() -> dict[str, list[re.Pattern[str]]]:
    """``commitment_patterns.txt`` split by its ``# --- title`` comments."""
    groups: dict[str, list[re.Pattern[str]]] = {}
    current = ""
    for raw in (LISTS / "commitment_patterns.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        header = GROUP_HEADER.match(line)
        if header:
            current = header["title"]
            groups[current] = []
        elif line and not line.startswith("#"):
            groups[current].append(re.compile(line))
    return groups


def all_commitment_patterns() -> list[re.Pattern[str]]:
    return [pattern for patterns in commitment_groups().values() for pattern in patterns]


COMMITMENT_TABLE: dict[str, tuple[list[str], list[str]]] = {
    "phone calls": (
        ["我明天给你打电话", "等下给你打个电话", "你接一下电话", "我现在打过去", "我们打个电话吧"],
        ["他在打电话呢", "我手机没电了", "电话费好贵啊", "你昨天的电话我没接到"],
    ),
    "video calls and voice": (
        ["我现在开视频", "等下给你发语音", "我们视频吧", "听听我的声音"],
        ["这个视频好好笑", "你发的语音我听不清", "视频卡住了", "语音条好长啊"],
    ),
    "photos and videos": (
        ["我给你发张照片", "我拍给你看", "我给你拍一张", "你想看我的自拍吗"],
        ["照片拍得好好看", "你拍的照片发给我看看", "他们拍给你们看了", "我在拍作业呢"],
    ),
    "meeting in person": (
        [
            "周末见面吧",
            "我们明天见面",
            "周末不见不散",
            "我明天去找你",
            "我周末来接你",
            "我在你楼下等你",
            "我在门口等你",
            "我在你楼下了",
            "我带你去吃好吃的",
            "带你出去玩",
            "明天带你吃火锅",
            "我陪你一起去",
            "我去你家",
        ],
        [
            "明天见",
            "晚安，明天见",
            "下次见啦",
            "周末见",
            "我明天见老师",
            "我在路上了",
            "我在车站了",
            "我在门口了",
            "好呀，我陪你吃",
            "我陪你聊天",
            "你在学校吗",
        ],
    ),
    "money, red packets and gifts": (
        [
            "我马上转你100",
            "我转账给你",
            "我给你转点钱",
            "红包发你了",
            "我给你买单",
            "我请你吃饭",
            "我来请你喝奶茶",
            "周末请你吃大餐",
            "给你买个礼物",
        ],
        [
            "请你别生气啦",
            "请你理解一下",
            "他请你吃饭了吗",
            "我转给你看",
            "我转给你看看这个",
            "抢红包好好玩",
        ],
    ),
    "mail, parcels and deliveries": (
        ["我给你寄点吃的", "快递已经寄给你了", "我给你带点东西", "给你的礼物已经寄出"],
        ["我的快递在路上", "快递到了吗", "我在等快递", "我妈给我寄了好多东西", "礼物发了"],
    ),
    "other real-world actions": (
        [
            "我等下去你家做饭",
            "我给你煮面",
            "我开车过去",
            "我买了机票",
            "我们订了高铁票",
            "我帮你预约",
            "我帮你挂号",
            "我帮你取快递",
            "我给你收拾房间",
            "定个闹钟叫你起床",
        ],
        [
            "抱抱你",
            "亲亲你",
            "摸摸你的头",
            "抱抱你哦",
            "我抱你一下",
            "我会一直陪在你身边",
            "帮你整理一下思路",
            "帮你拿主意",
            "我买了电影票",
            "我们订了演唱会的票",
        ],
    ),
}


def test_every_group_of_the_commitment_list_has_a_table_entry() -> None:
    assert set(commitment_groups()) == set(COMMITMENT_TABLE)
    assert all(patterns for patterns in commitment_groups().values())


@pytest.mark.parametrize("group", sorted(COMMITMENT_TABLE))
def test_each_commitment_group_catches_its_promises_and_leaves_its_everyday_phrases(
    group: str,
) -> None:
    """R17-37: a positive and a negative for every group, each negative from everyday talk."""
    positives, negatives = COMMITMENT_TABLE[group]
    everything = all_commitment_patterns()
    assert positives and negatives
    for sentence in positives:
        assert any(p.search(sentence) for p in everything), f"promise not caught: {sentence}"
    for sentence in negatives:
        hits = [p.pattern for p in everything if p.search(sentence)]
        assert not hits, f"everyday sentence caught: {sentence} by {hits}"


PROMISE_TIMES = ["", "明天", "后天", "周末", "今晚", "等下", "晚点", "过几天"]
PROMISE_STEMS = [
    "我{t}给你打电话",
    "我{t}去找你",
    "{t}我给你转100",
    "我{t}给你发张照片",
    "{t}我去你家做饭",
    "{t}我给你寄点吃的",
    "{t}我请你吃饭",
]
FILLER = st.text(
    alphabet=list("哈啊呀吧呢嘛嗯哦好乖不哭睡早安累饿困烦~，。！ "), min_size=0, max_size=8
)


@given(st.sampled_from(PROMISE_STEMS), st.sampled_from(PROMISE_TIMES), FILLER, FILLER)
def test_a_promise_is_still_caught_whatever_chat_surrounds_it(
    stem: str, time: str, before: str, after: str
) -> None:
    """R17-37 hypothesis: narrowing the list did not narrow what it catches."""
    sentence = before + stem.format(t=time) + after
    assert any(p.search(sentence) for p in all_commitment_patterns()), sentence


@given(
    st.sampled_from(["抱抱", "亲亲", "摸摸", "贴贴", "揉揉", "戳戳"]),
    st.sampled_from(["你", "你的头", "你的脸", "你哦", "你呀"]),
    FILLER,
    FILLER,
    st.sampled_from(["", "我", "好想", "乖，", "来，"]),
)
def test_a_cuddle_in_chat_is_never_a_promise(
    verb: str, tail: str, before: str, after: str, lead: str
) -> None:
    """R17-37 hypothesis: 抱抱你 / 亲亲你 / 摸摸你的头 are said every day (L77/L78 are gone)."""
    sentence = before + lead + verb + tail + after
    assert not any(p.search(sentence) for p in all_commitment_patterns()), sentence


@given(
    st.sampled_from(["明天见", "周末见", "下次见", "晚安，明天见", "改天见啦", "下周见哦"]),
    FILLER,
    FILLER,
)
def test_saying_goodbye_until_tomorrow_is_not_a_promise_to_meet(
    farewell: str, before: str, after: str
) -> None:
    sentence = before + farewell + after
    assert not any(p.search(sentence) for p in all_commitment_patterns()), sentence


def phrases() -> AiPhrases:
    return load_ai_phrases(LISTS / "ai_phrases.txt")


def test_the_ai_phrase_list_keeps_its_three_groups_and_its_size() -> None:
    found = phrases()
    assert found.self_reference and found.register and found.markup
    assert len(found.self_reference) + len(found.register) + len(found.markup) >= 120


AI_TABLE: dict[str, tuple[list[str], list[str]]] = {
    "self": (
        [
            "作为AI，我无法回答",
            "我是一个语言模型",
            "我没有真实的情感",
            "作为一个没有感情的机器人",
            "我的训练数据截止到",
            "人工智能",
        ],
        ["我没有感情了啦", "你再这样我就没有感情了", "我没有感情的呀", "我没有那么多情绪"],
    ),
    "register": (
        [
            "希望对你有帮助",
            "有什么可以帮你的吗",
            "综上所述，你应该",
            "首先，你要冷静",
            "总之，别想太多",
            "请注意休息",
            "我理解你的感受",
            "照顾好自己很重要",
            "需要我为你",
            "我可以帮助你",
            "以上就是我的建议",
        ],
        [
            "当然可以呀",
            "你说得很对",
            "你说的很有道理",
            "你的想法很棒",
            "记得照顾好自己哦",
            "一定要照顾好自己",
            "好的，我明白了",
            "好的，让我想想",
            "如果你需要我就在",
            "如果您需要的话",
            "需要我帮你吗",
            "我可以帮你看看",
        ],
    ),
    "markup": (["**重点**", "```", "- **要点**"], ["我吃了1个苹果", "好的~", "哈哈哈#"]),
}


@pytest.mark.parametrize("group", sorted(AI_TABLE))
def test_each_ai_phrase_group_catches_its_phrases_and_leaves_everyday_speech(group: str) -> None:
    """R17-37: the register group no longer deletes a bubble for 当然可以 / 你说得很对 ..."""
    found = phrases()
    detect = {
        "self": found.find_self_reference,
        "register": found.find_register,
        "markup": found.has_markup,
    }[group]
    positives, negatives = AI_TABLE[group]
    for sentence in positives:
        assert detect(sentence), f"{group} phrase not caught: {sentence}"
    for sentence in negatives:
        assert not (
            found.find_self_reference(sentence)
            or found.find_register(sentence)
            or found.has_markup(sentence)
        ), f"everyday sentence caught: {sentence}"


@given(st.sampled_from(AI_TABLE["register"][1]), FILLER, FILLER)
def test_everyday_agreement_and_care_survive_in_any_chat(
    core: str, before: str, after: str
) -> None:
    """R17-37 hypothesis: the narrowed register group leaves these in any chat around them."""
    sentence = before + core + after
    found = phrases()
    assert not found.find_register(sentence) and not found.find_self_reference(sentence), sentence


@given(st.sampled_from(sorted(set(phrases().register) | set(phrases().self_reference))), FILLER)
def test_every_phrase_left_in_the_list_is_still_found_inside_a_longer_line(
    phrase: str, filler: str
) -> None:
    found = phrases()
    line = filler + phrase.upper() + filler
    assert found.find_register(line) or found.find_self_reference(line), line
