"""Event text, its detector and reproducibility (R-IMP-007, R-IMP-009, R-SAFE-006)."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from twin.ingest.events import (
    EVENT_KEY_ALIASES,
    EVENT_TEMPLATES,
    KINDS,
    CallStatus,
    EventTextDetector,
    Kind,
    default_detector,
    fold_event_text,
    format_call_duration,
    is_reproducible,
    key_for_template,
    pattern_for_template,
    render_event_text,
)

SAMPLES = {
    "seconds": "5",
    "duration": "37 分钟",
    "caption": "一只猫趴在窗台上",
    "transcript": "好的我马上到",
    "title": "周末去哪里玩",
    "place": "中央公园",
}
FIELD = re.compile(r"\{([a-z_]+)\}")
# characters of ordinary chat: Chinese, ASCII letters and digits, punctuation (no brackets)
CHAT_ALPHABET = st.sampled_from(
    list(
        "的一是不了人我在有他这中大来上好吃饭睡觉今晚想你早安嗯哈"
        "啊呀吧呢嘛 ，。！？~…abcXYZ0123456789"
    )
)
EMOJI_CODES = ["[拥抱]", "[亲亲]", "[流泪]", "[捂脸]", "[旺柴]", "[爱心]", "[笑哭]"]


@dataclass
class Msg:
    kind: str
    is_sent: bool = False
    text: str | None = None
    call_status: str | None = None
    call_duration_s: int | None = None
    voice_seconds: int | None = None
    has_transcript: bool = False


def fill(template: str) -> str:
    return FIELD.sub(lambda m: SAMPLES[m.group(1)], template)


@pytest.mark.parametrize("name", sorted(EVENT_TEMPLATES))
def test_every_template_rendering_is_detected_as_event_text(name: str) -> None:
    rendered = fill(EVENT_TEMPLATES[name])
    assert default_detector.is_event_text(rendered), rendered
    assert default_detector.match(rendered) is not None


def test_every_rendering_of_render_event_text_is_detected() -> None:
    messages = [
        Msg("image"),
        Msg("video"),
        Msg("voice", text="你好", voice_seconds=5, has_transcript=True),
        Msg("voice", voice_seconds=5),
        Msg("voice", text="你好", has_transcript=True),
        Msg("voice"),
        Msg("call", call_status="connected", call_duration_s=2232),
        Msg("call", call_status="cancelled"),
        Msg("call", call_status="other_device"),
        Msg("call"),
        Msg("transfer"),
        Msg("redpacket"),
        Msg("link", text="标题"),
        Msg("link"),
        Msg("file", text="文件.pdf"),
        Msg("file"),
        Msg("location", text="中央公园"),
        Msg("location"),
        Msg("chathistory"),
        Msg("system", text="你撤回了一条消息"),
        Msg("system", text="对方撤回了一条消息"),
        Msg("system", text="“她”拍了拍你"),
        Msg("system", text="其他系统提示"),
        Msg("unknown"),
    ]
    for message in messages:
        text = render_event_text(message)
        assert text is not None and default_detector.is_event_text(text), (message, text)
    assert default_detector.is_event_text(render_event_text(Msg("image"), caption="海边日落") or "")


@given(
    st.text(CHAT_ALPHABET, min_size=1, max_size=30),
    st.sampled_from(["", *EMOJI_CODES]),
)
def test_ordinary_chat_lines_are_never_taken_for_event_text(sentence: str, code: str) -> None:
    assert not default_detector.is_event_text(sentence + code)
    assert not default_detector.is_event_text(code + sentence)


def test_the_detector_is_derived_from_the_template_table() -> None:
    table = {**EVENT_TEMPLATES, "sticker_pack": "[表情包盒子 {name}]"}
    custom = EventTextDetector(table)
    line = "[表情包盒子 春天]"
    assert custom.is_event_text(line) and not default_detector.is_event_text(line)
    assert custom.match(line) == "sticker_pack"
    assert set(custom.names) == set(table)


def test_the_bot_may_still_send_a_sticker_line_and_emoji_codes() -> None:
    assert not default_detector.is_event_text("[表情包:开心]")
    assert not default_detector.is_event_text("[拥抱]")


def test_full_width_punctuation_and_spacing_variants_are_detected() -> None:
    assert default_detector.is_event_text("[图片：一只猫]")
    assert default_detector.is_event_text("［图片：一只猫］")
    assert default_detector.is_event_text("  [语音 5 秒，未转写]  ")
    assert default_detector.is_event_text("[未接通话]")


def test_strip_removes_only_event_lines() -> None:
    text = "好呀\n[图片]\n[语音 3 秒：晚点说]\n我到了"
    kept, removed = default_detector.strip(text)
    assert kept == "好呀\n我到了" and removed == 2
    assert default_detector.lines_without_events(["[红包]", "收到"]) == ["收到"]


def test_pattern_for_template_finds_its_literal_prefix_anywhere_in_a_line() -> None:
    """R17-3 (D-670): was "whole lines only"; the pattern now finds the fragment inside a line.

    The old test pinned ``^[链接：{title}]$`` as the rule, which is why a line such as
    ``看这个[链接:标题]`` or ``[链接:标题] 谢谢`` slipped through.  R-SAFE-006 says a *line* that
    holds event text is deleted, so the fragment is what the template stands for.  It is searched
    in folded text (NFKC, no whitespace, no format characters); a line that merely mentions the
    word is no hit.
    """
    pattern = pattern_for_template("[链接：{title}]")
    assert pattern.search(fold_event_text("[链接:标题]"))
    assert pattern.search(fold_event_text("看这个[链接:标题]"))
    assert pattern.search(fold_event_text("[链接:标题] 谢谢"))
    assert pattern.search(fold_event_text("【 链 接 】"))
    assert not pattern.search(fold_event_text("发你一个链接"))
    assert not pattern.search(fold_event_text("(链接)"))
    assert not pattern.search(fold_event_text("[图片]"))


def test_a_template_without_a_bracketed_literal_has_no_key() -> None:
    assert key_for_template("[语音 {seconds} 秒：{transcript}]") == "语音"
    assert key_for_template("[通话已在其它设备接听]") == "通话已在其它设备接听"
    assert key_for_template("【表情包盒子 {name}】") == "表情包盒子"
    assert key_for_template("{caption}") is None
    assert key_for_template("图片") is None
    assert key_for_template("[{caption}]") is None
    with pytest.raises(ValueError, match="bracket"):
        pattern_for_template("{caption}")
    # such a template never detects anything (it did not before either)
    assert not EventTextDetector({"odd": "{caption}"}).is_event_text("{caption}")


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (0, "0 秒"),
        (45, "45 秒"),
        (60, "1 分钟"),
        (37 * 60 + 12, "37 分钟"),
        (3600, "1 小时"),
        (3600 + 5 * 60, "1 小时 5 分钟"),
    ],
)
def test_call_durations_are_rounded_down_to_whole_units(seconds: int, text: str) -> None:
    assert format_call_duration(seconds) == text


def test_call_event_texts() -> None:
    connected = Msg("call", call_status=CallStatus.CONNECTED.value, call_duration_s=37 * 60 + 12)
    assert render_event_text(connected) == "[通话 37 分钟]"
    for status in ("cancelled", "rejected", "missed"):
        assert render_event_text(Msg("call", call_status=status)) == "[未接通话]"
    assert render_event_text(Msg("call", call_status="other_device")) == "[通话已在其它设备接听]"
    assert render_event_text(Msg("call", call_status="unknown")) == "[通话]"
    assert render_event_text(Msg("call", call_status="connected")) == "[通话]"


def test_voice_event_texts_with_and_without_transcript() -> None:
    spoken = Msg("voice", text="我到楼下了", voice_seconds=5, has_transcript=True)
    assert render_event_text(spoken) == "[语音 5 秒：我到楼下了]"
    assert render_event_text(Msg("voice", voice_seconds=5)) == "[语音 5 秒，未转写]"
    assert render_event_text(Msg("voice")) == "[语音，未转写]"
    assert render_event_text(Msg("voice", text="嗯", has_transcript=True)) == "[语音：嗯]"
    multi = Msg("voice", text="第一句\n第二句", voice_seconds=9, has_transcript=True)
    assert render_event_text(multi) == "[语音 9 秒：第一句 第二句]"


def test_system_event_texts() -> None:
    assert render_event_text(Msg("system", text="你撤回了一条消息")) == "[你撤回了一条消息]"
    assert render_event_text(Msg("system", text="对方撤回了一条消息")) == "[她撤回了一条消息]"
    assert render_event_text(Msg("system", text="x撤回", is_sent=True)) == "[你撤回了一条消息]"
    assert render_event_text(Msg("system", text="“她”拍了拍“我”")) == "[拍一拍]"
    assert render_event_text(Msg("system", text="群公告")) == "[系统消息]"


def test_link_file_and_location_texts_are_one_short_line() -> None:
    long_title = "很长的标题" * 30
    text = render_event_text(Msg("link", text=long_title))
    assert text is not None and "\n" not in text and len(text) < 80 and text.endswith("…]")
    assert render_event_text(Msg("file", text="报告.pdf")) == "[文件：报告.pdf]"
    assert render_event_text(Msg("file")) == "[文件]"
    assert render_event_text(Msg("location", text="东京塔")) == "[位置：东京塔]"
    assert render_event_text(Msg("location")) == "[位置]"
    assert render_event_text(Msg("chathistory")) == "[聊天记录]"
    assert render_event_text(Msg("transfer")) == "[转账]"
    assert render_event_text(Msg("redpacket")) == "[红包]"


def test_image_and_video_event_texts_use_the_caption_when_there_is_one() -> None:
    assert render_event_text(Msg("image")) == "[图片]"
    assert render_event_text(Msg("image"), caption="  海边\n日落 ") == "[图片：海边 日落]"
    assert render_event_text(Msg("video"), caption="") == "[视频]"
    assert render_event_text(Msg("video"), caption="操场") == "[视频：操场]"


@pytest.mark.parametrize("kind", ["text", "sticker", "quote"])
def test_ordinary_chat_kinds_have_no_event_text(kind: str) -> None:
    assert render_event_text(Msg(kind, text="你好")) is None


def test_is_reproducible_follows_what_the_bot_can_send() -> None:
    reproducible = {k for k in KINDS if is_reproducible(k)}
    assert reproducible == {"text", "sticker", "quote"}
    assert is_reproducible(Msg("text")) and is_reproducible(Msg("quote"))
    for kind in (Kind.IMAGE, Kind.VOICE, Kind.VIDEO, Kind.CALL, Kind.TRANSFER, Kind.REDPACKET):
        assert not is_reproducible(Msg(kind.value))
    for kind in (Kind.LOCATION, Kind.FILE, Kind.LINK, Kind.CHATHISTORY, Kind.SYSTEM):
        assert not is_reproducible(kind.value)
    assert not is_reproducible(Msg("unknown"))


def test_alternative_templates_change_the_rendering() -> None:
    table = {**EVENT_TEMPLATES, "image": "[相片]"}
    assert render_event_text(Msg("image"), templates=table) == "[相片]"


# ------------------------------------------------- R17-3: the fragment is the rule (D-670)
# A generated line that carries event text *anywhere* is deleted; before round 17 only a line
# that was exactly a template rendering was (`^…$`, and it had to start with "[").

REVIEWER_LINES = [
    "给你听[语音 8 秒]",
    "[图片：自拍] 好看吗",
    "看这个[链接：周末去哪里玩]",
    "我把[位置]发你",
    "好的 [转账] 收一下",
    "[通话 37 分钟] 刚挂",
    "【图片】",
    "［视频：操场］",
    "[ 图片 ]",
    "[语音 8 秒",  # the output was cut off before the closing bracket
    "【语音 8 秒：好的我马上到",
    "[图\u200b片]",  # a zero-width space inside the key
    "[撤回了一条消息]",  # neutral wording of the two revoke templates
    "[对方撤回了一条消息]",
    "哈哈哈[拍一拍]哈哈",
    "[红包]恭喜发财",
    "[未接通话",
    "[聊天记录] 看一下",
    "\u3000\u3000[其他消息]",
    "好[系统消息]吧",
]


@pytest.mark.parametrize("line", REVIEWER_LINES)
def test_event_text_anywhere_in_a_line_is_detected(line: str) -> None:
    """R17-3: fails before the fix (every one of these slipped through the full-line anchor)."""
    assert default_detector.is_event_text(line), line
    assert default_detector.match(line) in EVENT_TEMPLATES


WORDS = ["图片", "语音", "视频", "通话", "转账", "红包", "链接", "文件", "位置", "拍一拍"]
OPENINGS = ["[", "［", "【"]
CLOSINGS = ["]", "］", "】", ""]  # "" = the output was cut off
PADDING = ["", "", " ", "\u3000", "\u00a0", "\u200b", "\ufeff", "\u2060"]


def _scatter(text: str, rng: random.Random) -> str:
    """``text`` with whitespace / format characters dropped between some of its characters."""
    return "".join(rng.choice(PADDING) + char for char in text)


@given(
    st.sampled_from(sorted(EVENT_TEMPLATES)),
    st.sampled_from(OPENINGS),
    st.sampled_from(CLOSINGS),
    st.text(CHAT_ALPHABET, max_size=12),
    st.text(CHAT_ALPHABET, max_size=12),
    st.randoms(use_true_random=False),
)
def test_every_template_in_every_bracket_and_spacing_variant_is_detected(
    name: str, opening: str, closing: str, before: str, after: str, rng: random.Random
) -> None:
    """R17-3 hypothesis: any template x any bracket pair x any spacing x any chat around it."""
    rendered = fill(EVENT_TEMPLATES[name])
    inner = rendered[1:-1]
    line = before + opening + _scatter(inner, rng) + rng.choice(PADDING) + closing + after
    assert default_detector.is_event_text(line), repr(line)
    assert default_detector.is_event_text(line.strip()), repr(line)


@given(
    st.text(CHAT_ALPHABET, min_size=1, max_size=20),
    st.sampled_from(WORDS),
    st.sampled_from(["", "(", "（", "<", "「", "{", "《"]),
    st.sampled_from(["", ")", "）", ">", "」", "}", "》"]),
)
def test_the_same_words_without_a_square_bracket_are_chat(
    sentence: str, word: str, opening: str, closing: str
) -> None:
    """R17-3: ``我发了图片`` / ``(图片)`` stay ordinary chat; only [ ] 【 】 start event text."""
    assert not default_detector.is_event_text(f"{sentence}{opening}{word}{closing}{sentence}")
    assert not default_detector.is_event_text(f"{opening}{word}{closing}")


@pytest.mark.parametrize(
    "line",
    [
        "[表情包:开心]",
        "[表情包：开心]",
        "[引用:你说的那个]",
        "(图片)",
        "（图片）",
        "我发了图片",
        "我发了个语音给你",
        "图片",
        "「红包」",
        "<链接>",
        "[1]",
        "[ ]",
        "[]",
        "[图]",
        "[开心]",
        "[拥抱]哈哈",
        "好的[亲亲]",
        "[旺柴][捂脸]",
        "你好[",
        "[",
        "",
    ],
)
def test_chat_the_bot_may_send_is_never_event_text(line: str) -> None:
    assert not default_detector.is_event_text(line), line


def test_no_wechat_emoji_code_is_event_text_but_the_one_d_138_already_names() -> None:
    """The prefix keys must not swallow her emoji vocabulary (D-138/D-205: only [红包] collides)."""
    table = Path(__file__).parents[2] / "config" / "lists" / "wechat_emoji_codes.txt"
    codes = [
        line.strip()
        for line in table.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(codes) > 100
    clashes = [code for code in codes if default_detector.is_event_text(f"[{code}]")]
    assert clashes == ["红包"]


def test_a_neutral_revoke_notice_has_an_alias_and_belongs_to_the_revoke_templates() -> None:
    assert set(EVENT_KEY_ALIASES) <= set(EVENT_TEMPLATES)
    assert default_detector.match("[撤回了一条消息]") == "revoke_her"
    assert default_detector.match("[你撤回了一条消息]") == "revoke_user"
    assert default_detector.match("[她撤回了一条消息]") == "revoke_her"
    assert default_detector.match("[我撤回了一条消息]") == "revoke_user"
    # a table that has no revoke template has no alias for it
    assert not EventTextDetector({"image": "[图片]"}).is_event_text("[撤回了一条消息]")


def test_match_names_the_family_of_the_longest_key_in_the_line() -> None:
    assert default_detector.match("看[图片：自拍]吧") == "image"
    assert default_detector.match("[语音 5 秒，未转写]") == "voice"
    assert default_detector.match("[通话已在其它设备接听]") == "call_other_device"
    assert default_detector.match("[未接通话]") == "call_missed"
    assert default_detector.match("[通话 3 分钟]") == "call"
    assert default_detector.match("今天吃什么") is None
    custom = EventTextDetector({**EVENT_TEMPLATES, "sticker_pack": "[表情包盒子 {name}]"})
    assert custom.match("看这个[表情包盒子 春天]") == "sticker_pack"
    assert custom.match("[表情包:开心]") is None


def test_the_fold_removes_whitespace_format_characters_and_full_width_forms() -> None:
    assert fold_event_text("［ 语\u200b音\u3000８ 秒］") == "[语音8秒]"
    assert fold_event_text("【图\u00a0片】") == "【图片】"
    assert fold_event_text("") == ""


def test_an_empty_template_table_detects_nothing() -> None:
    nothing = EventTextDetector({})
    assert nothing.names == () and not nothing.is_event_text("[图片]")


def test_strip_deletes_a_line_that_holds_event_text_anywhere() -> None:
    text = "好呀\n给你听[语音 8 秒]\n【图片】\n我到了\n[图片：自拍] 好看吗"
    kept, removed = default_detector.strip(text)
    assert kept == "好呀\n我到了" and removed == 3
    assert default_detector.lines_without_events(["我把[位置]发你", "收到", "[拥抱]"]) == [
        "收到",
        "[拥抱]",
    ]
