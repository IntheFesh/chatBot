"""The WeChat emoji codes she uses and how often (R-STK-001, R-TRN-013, D-138)."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from tests.support.embedding import H, U, day, write_dialogue
from tests.support.synth_chat import ChatSpec, build_chat
from twin.profile.builder import rebuild
from twin.profile.distribution import EmpiricalDistribution
from twin.services import Services
from twin.stickers.emoji_codes import EmojiCodePolicy, bare, bracketed

FREQUENCIES = {"[拥抱]": 0.6, "[亲亲]": 0.3, "[捂脸]": 0.1}


def policy(**kwargs: object) -> EmojiCodePolicy:
    options: dict[str, object] = {"rate": 0.035, "known": {"拥抱", "亲亲", "捂脸", "红包", "旺柴"}}
    options.update(kwargs)
    return EmojiCodePolicy(FREQUENCIES, **options)  # type: ignore[arg-type]


def test_only_codes_she_used_are_allowed_with_or_without_brackets() -> None:
    p = policy()
    assert p.is_allowed("[拥抱]") and p.is_allowed("拥抱") and p.is_allowed(" [亲亲] ")
    assert not p.is_allowed("[旺柴]")  # a real WeChat code, but she never wrote it
    assert not p.is_allowed("[微笑]") and not p.is_allowed("")
    assert p.codes == ("[拥抱]", "[亲亲]", "[捂脸]")  # most frequent first
    assert (bare("[拥抱]"), bare("拥抱"), bracketed("拥抱"), bracketed("[拥抱]")) == (
        "拥抱",
    ) * 2 + ("[拥抱]",) * 2


def test_the_red_packet_is_never_an_emoji_code() -> None:
    # "[红包]" is in the WeChat table and she may have typed it, but it is also the event text
    # of a red packet message and is deleted as such (D-138): it must not come back as an emoji
    p = EmojiCodePolicy({"[红包]": 0.5, "[拥抱]": 0.5}, rate=0.1, known={"红包", "拥抱"})
    assert not p.is_allowed("[红包]") and p.codes == ("[拥抱]",)
    assert p.frequency("[拥抱]") == 1.0


def test_event_texts_in_general_are_excluded() -> None:
    p = EmojiCodePolicy({"[转账]": 1, "[图片]": 1, "[通话]": 1, "[拥抱]": 1}, rate=0.1)
    assert p.codes == ("[拥抱]",)


def test_a_bracket_text_outside_the_wechat_table_is_not_an_emoji() -> None:
    p = EmojiCodePolicy({"[拥抱]": 1, "[随便写的]": 3}, rate=0.1, known={"拥抱"})
    assert p.codes == ("[拥抱]",)
    open_ended = EmojiCodePolicy({"[拥抱]": 1, "[随便写的]": 3}, rate=0.1)
    assert open_ended.codes == ("[随便写的]", "[拥抱]")  # no table given: nothing is checked


def test_frequencies_are_shares_of_the_allowed_codes() -> None:
    p = EmojiCodePolicy({"[拥抱]": 0.3, "[亲亲]": 0.1, "[红包]": 0.6}, rate=0.1)
    assert p.frequency("拥抱") == pytest.approx(0.75) and p.frequency("[亲亲]") == pytest.approx(
        0.25
    )
    assert p.frequency("[旺柴]") == 0.0
    assert EmojiCodePolicy({}, rate=0.5).frequency("拥抱") == 0.0


def test_the_expected_rate_is_hers_and_zero_without_any_code() -> None:
    assert policy().expected_rate() == pytest.approx(0.035)
    assert EmojiCodePolicy({}, rate=0.5).expected_rate() == 0.0
    assert EmojiCodePolicy({"[拥抱]": 1}, rate=3.0).expected_rate() == 1.0
    assert EmojiCodePolicy({"[拥抱]": 0}, rate=0.5).codes == ()


def test_codes_are_drawn_by_frequency_and_runs_by_her_distribution() -> None:
    rng = random.Random(4)
    drawn = Counter(policy().sample_code(rng) for _ in range(4000))
    assert drawn["[拥抱]"] / 4000 == pytest.approx(0.6, abs=0.04)
    assert drawn["[捂脸]"] / 4000 == pytest.approx(0.1, abs=0.03)
    assert EmojiCodePolicy({}, rate=0.0).sample_code(rng) is None
    runs = EmpiricalDistribution.from_samples([1.0] * 70 + [2.0] * 20 + [3.0] * 10, discrete=True)
    p = policy(run_lengths=runs)
    lengths = Counter(p.sample_repeat(random.Random(i)) for i in range(2000))
    assert set(lengths) <= {1, 2, 3} and lengths[1] > lengths[2] > lengths[3]
    assert policy().sample_repeat(rng) == 1  # no distribution: one code at a time


def test_codes_outside_her_vocabulary_are_deleted_from_a_reply() -> None:
    p = policy()
    assert p.strip_disallowed("好呀[拥抱][旺柴]") == "好呀[拥抱]"
    assert p.strip_disallowed("[旺柴] 好呀") == "好呀"
    assert p.strip_disallowed("好 [旺柴] 呀") == "好 呀"
    assert p.strip_disallowed("[红包]") == ""
    assert p.strip_disallowed("[表情包:开心]") == "[表情包:开心]"  # not an emoji code (colon)
    assert p.strip_disallowed("没有表情") == "没有表情"
    assert p.strip_disallowed("[拥抱] [亲亲]") == "[拥抱] [亲亲]"


# --------------------------------------------------------------- from a profile


@pytest.fixture
def chat(services: Services) -> Services:
    build_chat(services, ChatSpec(days=45))
    rebuild(services, "all")
    return services


def test_the_policy_is_read_from_her_profile(chat: Services) -> None:
    live = EmojiCodePolicy.from_profile(chat, "live")
    assert live is not None
    assert set(live.codes) == {"[拥抱]", "[亲亲]", "[捂脸]", "[抱抱]"}
    assert 0.0 < live.expected_rate() < 0.1  # one text in 25 carries a code
    assert all(live.is_allowed(code) for code in live.codes) and not live.is_allowed("[旺柴]")
    assert live.sample_repeat(random.Random(1)) >= 1
    assert sum(live.frequency(code) for code in live.codes) == pytest.approx(1.0)


def test_the_past_scope_reads_the_pre_holdout_profile(chat: Services) -> None:
    past = EmojiCodePolicy.from_profile(chat, "pre_holdout")
    assert past is not None and past.codes


def test_a_code_she_wrote_only_in_the_held_out_period_is_not_in_the_past_vocabulary(
    services: Services,
) -> None:
    messages = [U("你好"), H("好呀[拥抱]"), H("嗯")]
    episodes = [(day(n, 12), [*messages, H("再见")]) for n in range(30)]
    late = (day(31, 12), [U("来了"), H("来啦[旺柴]"), H("好")])
    write_dialogue(services, [*episodes, late])
    rebuild(services, "all")
    live = EmojiCodePolicy.from_profile(services, "live")
    past = EmojiCodePolicy.from_profile(services, "pre_holdout")
    assert live is not None and past is not None
    assert live.is_allowed("[旺柴]") and not past.is_allowed("[旺柴]")
    assert past.is_allowed("[拥抱]")


def test_without_a_profile_there_is_no_policy(services: Services) -> None:
    assert EmojiCodePolicy.from_profile(services, "live") is None
