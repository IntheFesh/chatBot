"""The post-processing steps, one rule at a time (R-ENG-007/008/012, R-STK-005, R-SAFE)."""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import pytest

from tests.support.reply_view import StaticSelector, sticker_record
from twin.config.lists import WordListError
from twin.engine.parsing import parse_reply
from twin.engine.postprocess import (
    STEPS,
    PostContext,
    PostProcessor,
    StyleLimits,
    Working,
    fit_bubbles_to_quota,
    load_ai_phrases,
)
from twin.engine.postprocess import steps as step_module
from twin.engine.postprocess.model import Line
from twin.engine.postprocess.phrases import parse_ai_phrases
from twin.engine.postprocess.stickers import canonical_tag
from twin.engine.postprocess.text import (
    clamp_marks,
    split_after,
    split_after_codes,
    split_semantic,
    strip_list_marker,
)
from twin.engine.safety.commitments import CommitmentDetector
from twin.engine.types import Bubble
from twin.stickers.emoji_codes import EmojiCodePolicy
from twin.stickers.rate import MemoryBubbleHistory, StickerRateController

LISTS = Path(__file__).resolve().parents[2] / "config" / "lists"
PHRASES = load_ai_phrases(LISTS / "ai_phrases.txt")
COMMITMENTS = CommitmentDetector.from_file(LISTS / "commitment_patterns.txt")
HER_STYLE = StyleLimits(
    max_chars=15,
    max_bubbles=5,
    comma_rate=0.03,
    period_end_rate=0.002,
    median_chars=5,
    max_code_run=2,
)
EMOJI = EmojiCodePolicy({"拥抱": 0.6, "亲亲": 0.4}, rate=0.5)


def context(**changes: object) -> PostContext:
    base = PostContext(
        style=HER_STYLE,
        emoji=EMOJI,
        ai_phrases=PHRASES,
        commitments=COMMITMENTS,
        rng=random.Random(1),
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def work(raw: str) -> Working:
    parsed = parse_reply(raw)
    return Working(parsed.quote, [Line(line.kind, line.text) for line in parsed.lines])


def texts(w: Working) -> list[str]:
    return [line.text for line in w.lines]


def actions(w: Working) -> dict[str, int]:
    return {a.step: a.count for a in w.actions}


def kinds(w: Working) -> list[str]:
    return [v.kind for v in w.violations]


def run(raw: str, **changes: object):  # type: ignore[no-untyped-def]
    return PostProcessor().process(raw, context(**changes))


# ----------------------------------------------------------------- thinking blocks


@pytest.mark.parametrize(
    ("raw", "kept", "count"),
    [
        ("<think>让我想想</think>好呀", "好呀", 1),
        ("<think>跨\n多行\n</think>\n好呀", "好呀", 1),
        ("好呀<think>没有结尾", "好呀", 1),
        ("想了很久</think>好呀", "好呀", 1),
        ("<THINK>x</THINK>好呀<think>y</think>", "好呀", 2),
        ("好呀", "好呀", 0),
    ],
)
def test_thinking_is_cut_out_and_counted(raw: str, kept: str, count: int) -> None:
    text, found = step_module.remove_think(raw)
    assert text == kept and found == count


def test_a_thinking_block_in_the_output_is_a_recorded_violation() -> None:
    result = run("<think>先想想</think>好呀")
    assert [b.text for b in result.bubbles] == ["好呀"]
    assert [v.kind for v in result.violations] == ["think_tag"]
    assert {a.step: a.count for a in result.actions}["think_removed"] == 1


# ------------------------------------------------------------------ AI tone, tokens


def test_markdown_and_list_marks_are_taken_out_of_a_line() -> None:
    w = work("**好啊**\n- 我也是\n1. 哈哈\n3.5万块")
    step_module.ai_tone(w, context())
    assert texts(w) == ["好啊", "我也是", "哈哈", "3.5万块"]
    assert actions(w) == {"markup_removed": 1, "list_marker_removed": 2}


def test_a_line_that_was_only_a_mark_disappears() -> None:
    w = work("**\n****\n好")
    step_module.ai_tone(w, context())
    assert texts(w) == ["好"]


def test_a_missing_phrase_list_is_reported() -> None:
    with pytest.raises(WordListError, match="cannot read"):
        load_ai_phrases(LISTS / "no-such-list.txt")


def test_customer_service_wording_costs_the_bubble() -> None:
    w = work("好啊\n希望对你有帮助\n总之你早点睡")
    step_module.ai_tone(w, context())
    assert texts(w) == ["好啊"] and actions(w) == {"ai_phrase_removed": 2}
    assert not w.violations


def test_calling_herself_an_ai_is_a_violation() -> None:
    w = work("作为AI我没有感情")
    step_module.ai_tone(w, context())
    assert kinds(w) == ["ai_self_reference"]
    other = work("我是一个语言模型")
    step_module.ai_tone(other, context())
    assert kinds(other) == ["ai_self_reference"]


def test_an_honest_answer_to_a_sincere_question_is_kept() -> None:
    """R-SAFE-003: the admission is not deleted and not held against the reply."""
    admission = "嗯…我是一个AI，照着她聊天的样子做出来的"
    w = work(f"{admission}\n你别难过呀")
    step_module.ai_tone(w, context(allow_ai_admission=True))
    assert texts(w) == [admission, "你别难过呀"]
    assert not w.violations and actions(w) == {"ai_admission_kept": 1}
    # the same words are a violation when nobody asked
    unasked = work(admission)
    step_module.ai_tone(unasked, context())
    assert kinds(unasked) == ["ai_self_reference"]
    # the customer-service register stays forbidden even then
    again = work(f"{admission}\n有什么可以帮你的吗")
    step_module.ai_tone(again, context(allow_ai_admission=True))
    assert texts(again) == [admission]


def run_tone(raw: str, ctx: PostContext) -> Working:
    w = work(raw)
    step_module.ai_tone(w, ctx)
    return w


def test_a_word_the_user_brought_up_is_a_topic_not_a_slip() -> None:
    """Chatting about 人工智能 with someone who raised it is not calling herself one."""
    said = "最近好多人在聊人工智能，你怎么看"
    topic = run_tone("我觉得人工智能挺吓人的", context(user_text=said))
    assert texts(topic) == ["我觉得人工智能挺吓人的"]
    assert not topic.violations and actions(topic) == {"ai_topic_kept": 1}
    # the same line out of nowhere is still a slip
    assert kinds(run_tone("我觉得人工智能挺吓人的", context(user_text="今天吃什么"))) == [
        "ai_self_reference"
    ]
    assert kinds(run_tone("我觉得人工智能挺吓人的", context())) == ["ai_self_reference"]
    # a longer phrase the user did not say is the model speaking about itself
    assert kinds(run_tone("我是人工智能啦", context(user_text=said))) == ["ai_self_reference"]
    assert kinds(run_tone("作为AI我没有感情", context(user_text="你觉得AI会取代人吗"))) == [
        "ai_self_reference"
    ]
    # the match ignores letter case, like the list does
    assert not run_tone("作为ai我也不懂", context(user_text="作为AI你怎么看")).violations


def test_the_phrase_list_is_read_in_groups() -> None:
    phrases = parse_ai_phrases(
        "# --- self-identification\n作为AI\n# --- register\n总之\n# --- markdown\n**\n# note\n"
    )
    assert phrases.find_self_reference("作为AI,我") and phrases.find_register("总之好吧")
    assert phrases.strip_markup("**好**") == "好" and phrases.has_markup("**重点")
    plain = parse_ai_phrases("总之\n首先，")
    assert plain.self_reference == () and plain.register == ("总之", "首先，")


def test_the_shipped_list_has_all_three_groups() -> None:
    assert "人工智能" in PHRASES.self_reference and "作为ai" in PHRASES.self_reference
    assert "希望对你有帮助" in PHRASES.register and "**" in PHRASES.markup


def test_a_redaction_token_in_the_output_is_a_violation() -> None:
    w = work("你的电话是[手机号]吧")
    step_module.token_leak(w, context())
    assert kinds(w) == ["token_leak"] and w.violations[0].detail == "[手机号]"
    quoted = work("[引用:发到[邮箱#2]]\n好")
    step_module.token_leak(quoted, context())
    assert kinds(quoted) == ["token_leak"]
    clean = work("[拥抱]好的")
    step_module.token_leak(clean, context())
    assert not clean.violations


# --------------------------------------------------------- event text, promises, language


def test_event_text_lines_are_deleted() -> None:
    w = work("好呀\n[图片]\n[语音 5 秒：你好]\n[通话 37 分钟]\n[转账]\n[位置：公司]\n哈哈")
    step_module.event_text(w, context())
    assert texts(w) == ["好呀", "哈哈"] and actions(w) == {"event_text_removed": 5}
    assert not w.violations


def test_nothing_left_after_event_text_is_a_violation() -> None:
    w = work("[图片：一只猫]\n[红包]")
    step_module.event_text(w, context())
    assert w.lines == [] and kinds(w) == ["event_text_only"]


def test_a_sticker_line_is_not_event_text() -> None:
    w = work("[表情包:开心]\n[拥抱]")
    step_module.event_text(w, context())
    assert len(w.lines) == 2 and not w.violations


@pytest.mark.parametrize(
    "line",
    ["我明天给你打电话", "等下给你发个语音", "我马上转你100", "周末见面吧", "我给你发张照片"],
)
def test_a_promise_of_a_real_world_action_is_a_violation(line: str) -> None:
    w = work(f"好呀\n{line}")
    step_module.commitments(w, context())
    assert kinds(w) == ["commitment"] and w.lines[1].promises and not w.lines[0].promises
    assert (w.violations[0].detail or "").startswith("pattern#")


def test_on_the_last_attempt_the_promise_is_cut_out_instead() -> None:
    w = work("好呀\n我明天给你打电话\n等我哦")
    step_module.commitments(w, context(last_attempt=True))
    assert texts(w) == ["好呀", "等我哦"] and actions(w) == {"commitment_removed": 1}
    assert not w.violations


def test_ordinary_chat_makes_no_promise() -> None:
    w = work("今天吃什么\n哈哈哈哈\n我刚到家")
    step_module.commitments(w, context())
    assert not w.violations
    step_module.commitments(w, context(commitments=None))
    assert not w.violations


def test_a_reply_that_is_not_chinese_chat_is_a_violation() -> None:
    english = work("I am doing fine, thank you for asking")
    step_module.language(english, context())
    assert kinds(english) == ["not_chinese"]
    for fine in ("ok 好的", "哈哈 lol", "[拥抱]", "好的 ok 的"):
        w = work(fine)
        step_module.language(w, context())
        assert not w.violations, fine


# -------------------------------------------------------------------- punctuation


def normalise(text: str, **changes: object) -> list[str]:
    return step_module.normalise_line(text, context(**changes))


def test_the_final_stop_goes_and_sentences_become_bubbles() -> None:
    assert normalise("我在吃饭。") == ["我在吃饭"]
    assert normalise("我在吃饭。你呢？") == ["我在吃饭", "你呢？"]
    assert normalise("没事……") == ["没事……"]  # an ellipsis is not a stop
    assert normalise("版本3.5很好") == ["版本3.5很好"]


def test_comma_sentences_are_split_where_she_writes_without_commas() -> None:
    assert normalise("好呀，我也想去，等下说") == ["好呀", "我也想去", "等下说"]
    assert normalise("好，吧") == ["好吧"]  # a piece of one character stays with its neighbour
    assert normalise("好呀，") == ["好呀"]
    assert normalise("你吃了吗？我还没") == ["你吃了吗？", "我还没"]


def test_a_writer_with_commas_keeps_hers() -> None:
    style = replace(HER_STYLE, comma_rate=0.5, period_end_rate=0.5)
    assert normalise("好呀，我也想去。等下说", style=style) == ["好呀，我也想去。等下说"]
    assert normalise("好呀，", style=style) == ["好呀"]  # a trailing comma is never right


def test_a_sometimes_writer_splits_only_long_comma_sentences() -> None:
    style = replace(HER_STYLE, comma_rate=0.15, median_chars=5)
    assert normalise("好呀，我来", style=style) == ["好呀，我来"]
    long_line = "今天真的好累啊，我想早点回家休息一下"
    assert normalise(long_line, style=style) == ["今天真的好累啊", "我想早点回家休息一下"]


def test_without_a_profile_the_punctuation_stays() -> None:
    style = StyleLimits()
    assert normalise("好呀，我也想去。", style=style) == ["好呀，我也想去。"]


def test_repeated_marks_are_clamped_and_spaces_squeezed() -> None:
    assert clamp_marks("真的吗？？？？？？") == "真的吗？？？"
    assert normalise("真的!!!!!!   好") == ["真的!!! 好"]


def test_the_punctuation_step_counts_the_lines_it_changed() -> None:
    w = work("好呀，我来。\n哈哈")
    step_module.punctuation(w, context())
    assert texts(w) == ["好呀", "我来", "哈哈"] and actions(w) == {"punctuation_normalised": 1}


# ------------------------------------------------------------- length and bubble cap


def test_a_long_line_is_split_at_sentence_ends_and_never_cut() -> None:
    line = "今天真的很开心。因为我们去了很多好玩的地方！还吃了好吃的东西，回来的路上还看到了彩虹"
    pieces = split_semantic(line, 15)
    assert "".join(pieces) == line and len(pieces) >= 4
    assert all(len(piece) <= 15 for piece in pieces)
    assert pieces[0].endswith("。")


def test_pieces_are_packed_so_that_small_ones_share_a_bubble() -> None:
    assert split_semantic("好。嗯。行。我去。" + "字" * 10, 12) == ["好。嗯。行。我去。", "字" * 10]


def test_a_piece_without_a_place_to_cut_stays_whole() -> None:
    word = "字" * 40
    assert split_semantic(word, 15) == [word]
    assert split_semantic("ab", 15) == ["ab"]
    with pytest.raises(ValueError, match="at least one"):
        split_semantic("ab", 0)


def test_emoji_codes_are_a_place_to_cut() -> None:
    assert split_after_codes("哈哈[捂脸]真的假的[拥抱]") == ["哈哈[捂脸]", "真的假的[拥抱]"]
    assert split_after("a。b!c", "。!") == ["a。", "b!", "c"]
    assert strip_list_marker("2、你好") == "你好" and strip_list_marker("2号楼") == "2号楼"


def test_the_length_step_splits_only_what_is_too_long() -> None:
    w = work("短短的\n" + "今天真的很开心。因为我们去了很多好玩的地方。还有好吃的")
    step_module.length(w, context())
    assert all(len(line.text) <= 15 for line in w.lines) and len(w.lines) > 2
    assert w.lines[0].text == "短短的" and actions(w) == {"long_bubble_split": 1}


def test_too_many_bubbles_cost_the_stickers_first_then_the_tail() -> None:
    w = work("一\n二\n[表情包:开心]\n三\n四\n五\n六")
    step_module.bubble_cap(w, context(style=replace(HER_STYLE, max_bubbles=5)))
    assert [(line.kind, line.text) for line in w.lines] == [
        ("text", "一"),
        ("text", "二"),
        ("text", "三"),
        ("text", "四"),
        ("text", "五"),
    ]
    assert actions(w) == {"bubble_cap": 2}
    four = work("一\n二\n三\n四")
    step_module.bubble_cap(four, context(style=replace(HER_STYLE, max_bubbles=2)))
    assert texts(four) == ["一", "二"]


def test_the_same_bubble_twice_in_a_row_is_sent_once() -> None:
    w = work("好的\n好的\n哈哈\n好的\n[表情包:开心]\n[表情包:开心]\n[表情包:大笑]")
    step_module.dedupe(w, context())
    assert [(line.kind, line.text) for line in w.lines] == [
        ("text", "好的"),
        ("text", "哈哈"),
        ("text", "好的"),
        ("sticker", "开心"),
        ("sticker", "大笑"),
    ]
    assert actions(w) == {"duplicate_removed": 2}


# ---------------------------------------------------------------------- emoji codes


def test_only_her_codes_survive() -> None:
    w = work("好呀[拥抱]\n[旺柴]\n哈哈[旺柴]")
    step_module.emoji_codes(w, context())
    assert texts(w) == ["好呀[拥抱]", "哈哈"] and actions(w)["emoji_code_removed"] == 2


def test_without_a_profile_no_code_is_trusted() -> None:
    w = work("好呀[拥抱]")
    step_module.emoji_codes(w, context(emoji=None))
    assert texts(w) == ["好呀"]


def test_a_run_of_codes_is_cut_to_her_longest_run() -> None:
    w = work("好呀[拥抱][亲亲][拥抱][亲亲]")
    step_module.emoji_codes(w, context())
    assert texts(w) == ["好呀[拥抱][亲亲]"] and actions(w)["emoji_run_trimmed"] == 1
    kept = work("好[拥抱]呀[亲亲]")  # separate codes are not a run
    step_module.emoji_codes(kept, context())
    assert texts(kept) == ["好[拥抱]呀[亲亲]"]


def test_only_her_share_of_bubbles_carry_a_code() -> None:
    # her rate is 0.5: of four bubbles at most two may have a code
    w = work("一[拥抱]\n二[亲亲]\n三[拥抱]\n四[亲亲]")
    step_module.emoji_codes(w, context())
    assert texts(w) == ["一[拥抱]", "二[亲亲]", "三", "四"]
    assert actions(w)["emoji_rate_trimmed"] == 2
    bare_code = work("好\n[拥抱]\n[亲亲]\n[拥抱]")
    step_module.emoji_codes(bare_code, context(emoji=EmojiCodePolicy({"拥抱": 1}, rate=0.25)))
    assert texts(bare_code) == ["好", "[拥抱]"]  # a bubble that is only a code goes with it


def test_a_woman_who_never_uses_codes_gets_none() -> None:
    w = work("好呀[拥抱]")
    step_module.emoji_codes(w, context(emoji=EmojiCodePolicy({}, rate=0.0)))
    assert texts(w) == ["好呀"]


# ----------------------------------------------------------------------- stickers


def sticker_context(table: dict[str, str], *, flags: list[bool] | None = None) -> dict[str, object]:
    selector = StaticSelector({tag: sticker_record(md5, tag) for tag, md5 in table.items()})
    rate = StickerRateController(
        0.105, MemoryBubbleHistory(200, flags or []), window=200, tolerance=0.2
    )
    return {
        "chooser": selector.choose,
        "rate": rate,
        "known_tags": ("开心", "大笑", "晚安"),
        "sticker_context": "对方说他晚安",
        "recent_stickers": ("c" * 32,),
        "_selector": selector,
    }


def sticker_work(
    raw: str, table: dict[str, str], **extra: object
) -> tuple[Working, StaticSelector]:
    changes = sticker_context(table)
    selector = changes.pop("_selector")
    assert isinstance(selector, StaticSelector)
    w = work(raw)
    step_module.stickers(w, context(**{**changes, **extra}))
    return w, selector


def test_a_tag_becomes_one_of_her_stickers() -> None:
    w, selector = sticker_work("晚安啦\n[表情包:晚安]", {"晚安": "a" * 32})
    assert [(line.kind, line.md5) for line in w.lines] == [("text", None), ("sticker", "a" * 32)]
    tag, wanted_context, recent = selector.asked[0]
    assert (tag, wanted_context) == ("晚安", "对方说他晚安")
    assert recent == ("c" * 32, None)  # the last bubbles, then the text bubble before it


def test_an_unknown_tag_a_missing_sticker_and_a_high_share_delete_the_line() -> None:
    w, _ = sticker_work("[表情包:旺柴]\n[表情包:大笑]\n好", {"开心": "a" * 32})
    assert texts(w) == ["好"]
    assert actions(w) == {"sticker_unknown_tag": 1, "sticker_no_match": 1}
    crowded = sticker_context({"开心": "a" * 32}, flags=[False] * 175 + [True] * 25)
    selector = crowded.pop("_selector")
    assert isinstance(selector, StaticSelector)
    limited = work("[表情包:开心]\n好")
    step_module.stickers(limited, context(**crowded))
    assert texts(limited) == ["好"] and actions(limited) == {"sticker_rate_drop": 1}


def test_two_stickers_in_one_reply_are_judged_together() -> None:
    # 24 of the last 200 bubbles are stickers: one more is 12.5 %, under her limit (10.5 % + 20 %
    # = 12.6 %), but once that one is in this reply a second would make 13 %
    changes = sticker_context(
        {"开心": "a" * 32, "大笑": "b" * 32}, flags=[False] * 176 + [True] * 24
    )
    changes.pop("_selector")
    w = work("[表情包:开心]\n[表情包:大笑]")
    step_module.stickers(w, context(**changes))
    assert [(line.kind, line.md5) for line in w.lines] == [("sticker", "a" * 32)]
    assert actions(w) == {"sticker_rate_drop": 1}


def test_a_tag_is_matched_to_the_vocabulary() -> None:
    known = ("开心", "大笑", "晚安")
    assert canonical_tag("开心", known) == "开心"
    assert canonical_tag(" “晚安” ", known) == "晚安"
    assert canonical_tag("开心的", known) == "开心"
    assert canonical_tag("旺柴", known) is None and canonical_tag("  ", known) is None
    assert canonical_tag("任何", ()) == "任何"


def test_without_a_library_no_sticker_is_sent() -> None:
    w = work("[表情包:开心]\n好")
    step_module.stickers(w, context(known_tags=("开心",)))
    assert texts(w) == ["好"] and actions(w) == {"sticker_no_match": 1}


# -------------------------------------------------------------- quote, silence, empty


def test_a_quote_the_channel_cannot_show_is_dropped() -> None:
    w = work("[引用:今天吃什么]\n吃面吧")
    step_module.quote(w, context(supports_quote=False))
    assert w.quote is None and actions(w) == {"quote_removed": 1}
    kept = work("[引用:今天吃什么]\n吃面吧")
    step_module.quote(kept, context())
    assert kept.quote == "今天吃什么" and not kept.actions
    only_sticker = work("[引用:今天吃什么]\n[表情包:开心]")
    step_module.quote(only_sticker, context())
    assert only_sticker.quote is None


def test_the_silence_marker_counts_only_when_allowed_and_alone() -> None:
    allowed = work("[不回]")
    step_module.no_reply(allowed, context(no_reply_allowed=True))
    assert allowed.no_reply and allowed.lines == [] and actions(allowed) == {"no_reply": 1}
    mixed = work("好的\n[不回]")
    step_module.no_reply(mixed, context(no_reply_allowed=True))
    assert not mixed.no_reply and texts(mixed) == ["好的"] and not mixed.violations
    refused = work("[不回]")
    step_module.no_reply(refused, context(no_reply_allowed=False))
    assert not refused.no_reply and kinds(refused) == ["no_reply_not_allowed"]
    none = work("好")
    step_module.no_reply(none, context())
    assert not none.actions


def test_a_reply_with_nothing_in_it_is_a_violation() -> None:
    empty = work("")
    step_module.emptiness(empty, context())
    assert kinds(empty) == ["empty"]
    silent = work("")
    silent.no_reply = True
    step_module.emptiness(silent, context())
    assert not silent.violations
    explained = work("")
    explained.violate("event_text_only")
    step_module.emptiness(explained, context())
    assert kinds(explained) == ["event_text_only"]


# ------------------------------------------------------------------------- quota


def b(text: str) -> Bubble:
    return Bubble("text", text)


def s(tag: str) -> Bubble:
    return Bubble("sticker", f"[表情包:{tag}]", sticker_md5="a" * 32, sticker_tag=tag)


def test_enough_quota_changes_nothing() -> None:
    bubbles = [b("一"), s("开心"), b("二")]
    fitted, done = fit_bubbles_to_quota(bubbles, 3)
    assert fitted == bubbles and done == []


def test_neighbouring_text_is_merged_with_a_space_shortest_pair_first() -> None:
    fitted, done = fit_bubbles_to_quota([b("长长长长长"), b("短"), b("也短"), b("最后")], 3)
    assert [x.text for x in fitted] == ["长长长长长", "短 也短", "最后"]
    assert [(a.step, a.count) for a in done] == [("quota_merge", 1)]


def test_stickers_go_only_when_no_two_texts_touch() -> None:
    bubbles = [b("一"), s("开心"), b("二"), s("大笑"), b("三")]
    # nothing to merge, so the last sticker goes first; then two texts touch and are merged
    fitted, done = fit_bubbles_to_quota(bubbles, 3)
    assert [x.text for x in fitted] == ["一", "[表情包:开心]", "二 三"]
    assert [(a.step, a.count) for a in done] == [("quota_merge", 1), ("quota_sticker_drop", 1)]
    fitted, done = fit_bubbles_to_quota(bubbles, 4)
    assert [x.text for x in fitted] == ["一", "[表情包:开心]", "二", "三"]
    assert [(a.step, a.count) for a in done] == [("quota_sticker_drop", 1)]


def test_at_least_one_bubble_is_always_left() -> None:
    fitted, done = fit_bubbles_to_quota([b("一"), b("二"), b("三")], 0)
    assert [x.text for x in fitted] == ["一 二 三"] and done[0].count == 2
    only_stickers, actions_done = fit_bubbles_to_quota([s("开心"), s("大笑")], 1)
    assert [x.sticker_tag for x in only_stickers] == ["开心"] and actions_done[0].count == 1


def test_the_quote_stays_with_the_first_bubble_when_merging() -> None:
    first = Bubble("text", "好", quote="吃什么")
    fitted, _ = fit_bubbles_to_quota([first, b("嗯")], 1)
    assert fitted[0].text == "好 嗯" and fitted[0].quote == "吃什么"


# ----------------------------------------------------------------------- the whole


def test_the_steps_run_in_the_order_of_the_requirement() -> None:
    assert PostProcessor().step_names == tuple(name for name, _ in STEPS)
    assert PostProcessor().step_names == (
        "ai_tone",
        "token_leak",
        "event_text",
        "commitments",
        "language",
        "punctuation",
        "length",
        "bubble_cap",
        "dedupe",
        "emoji_codes",
        "stickers",
        "quote",
        "no_reply",
        "emptiness",
    )


def test_a_realistic_reply_comes_out_as_her_bubbles() -> None:
    raw = (
        "[引用:今天吃什么]\n**吃面吧**，你呢[拥抱][亲亲][拥抱]。\n[图片]\n哈哈哈哈\n哈哈哈哈\n"
        "[表情包:开心]"
    )
    selector = StaticSelector({"开心": sticker_record("a" * 32, "开心")})
    result = run(
        raw,
        known_tags=("开心",),
        chooser=selector.choose,
        rate=StickerRateController(0.105, MemoryBubbleHistory(200), window=200, tolerance=0.2),
    )
    assert result.ok and not result.no_reply
    assert [b.text for b in result.bubbles] == [
        "吃面吧",
        "你呢[拥抱][亲亲]",
        "哈哈哈哈",
        "[表情包:开心]",
    ]
    assert result.bubbles[0].quote == "今天吃什么" and result.quote == "今天吃什么"
    assert result.bubbles[3].sticker_md5 == "a" * 32
    steps_done = {a.step for a in result.actions}
    assert {"markup_removed", "event_text_removed", "duplicate_removed", "emoji_run_trimmed"} <= (
        steps_done
    )
    for action in result.actions:  # the trail never carries the text of the reply
        assert "吃面" not in repr(action) and "哈哈" not in repr(action)


def test_the_quote_goes_with_the_first_text_bubble_even_after_a_sticker() -> None:
    selector = StaticSelector({"开心": sticker_record("a" * 32, "开心")})
    result = run(
        "[引用:周末去看电影]\n[表情包:开心]\n好呀",
        known_tags=("开心",),
        chooser=selector.choose,
        rate=StickerRateController(0.105, MemoryBubbleHistory(200), window=200, tolerance=0.2),
    )
    assert [b.kind for b in result.bubbles] == ["sticker", "text"]
    assert result.bubbles[0].quote is None and result.bubbles[1].quote == "周末去看电影"
    assert result.quote == "周末去看电影"


def test_silence_through_the_whole_pipeline() -> None:
    result = run("[不回]", no_reply_allowed=True)
    assert result.ok and result.no_reply and result.bubbles == ()
    forbidden = run("[不回]")
    assert [v.kind for v in forbidden.violations] == ["no_reply_not_allowed"]


def test_the_platform_quota_is_applied_at_the_end() -> None:
    result = run("一\n二\n三\n四", quota=2, style=replace(HER_STYLE, max_bubbles=8))
    assert len(result.bubbles) == 2
    assert [a.step for a in result.actions if a.step.startswith("quota")] == ["quota_merge"]
    again = PostProcessor().with_quota(
        run("一\n二\n三\n四", style=replace(HER_STYLE, max_bubbles=8)), 1
    )
    assert [b.text for b in again.bubbles] == ["一 二 三 四"]
