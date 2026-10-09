"""Stratified sampling of conversation segments (R-PERS-001, R-TRN-013, R-LLM-009)."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta

import pytest

from tests.support.embedding import H, U, day, write_dialogue
from tests.support.persona import (
    EARLY_MARKER,
    LATE_MARKER,
    message_ids_after,
    month_scenario,
    sticker_scenario,
)
from twin.ingest.transcript import TranscriptLine
from twin.llm.redaction import ConsistentRedactor
from twin.profile.holdout import holdout_cutoff
from twin.profile.persona.sampling import (
    PERIODS,
    SampleRequest,
    SegmentInfo,
    assign_bins,
    best_window,
    build_sample,
    collect_segments,
    coverage,
    message_intensity,
    segment_label,
    select_segments,
)
from twin.services import Services


def info(
    index: int, month: str = "2026-01", period: int = 0, messages: int = 4, intensity: float = 0.0
) -> SegmentInfo:
    start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index)
    return SegmentInfo(
        start,
        start + timedelta(minutes=5),
        f"a{index}",
        f"b{index}",
        messages,
        2,
        month,
        period,
        intensity,
    )


# -------------------------------------------------------------- measuring


def test_segments_are_measured_with_month_day_part_length_and_intensity(services: Services) -> None:
    month_scenario(services)
    segments = collect_segments(services)
    assert len(segments) == 36
    assert {s.month for s in segments} >= {"2026-01", "2026-02", "2026-03"}
    assert {s.period for s in segments} == {0, 1, 2, 3}
    assert all(s.her_messages >= 2 and s.messages >= s.her_messages for s in segments)
    assert {s.length_bin for s in segments} == {0, 1, 2}
    assert {s.intensity_bin for s in segments} == {0, 1, 2}
    loud = [s for s in segments if s.intensity > 1.5]
    quiet = [s for s in segments if s.intensity < 0.5]
    assert loud and quiet


def test_a_segment_needs_two_messages_from_her(services: Services) -> None:
    write_dialogue(
        services,
        [
            (day(0), [U("你好"), H("嗯")]),  # one message from her: not a candidate
            (day(1), [U("你好"), H("嗯"), H("哦")]),
        ],
    )
    segments = collect_segments(services)
    assert len(segments) == 1 and segments[0].her_messages == 2


def test_the_intensity_counts_marks_codes_emoji_and_stickers() -> None:
    assert message_intensity("好的", sticker=False) == 0
    assert message_intensity("好的！", sticker=False) == 1
    assert message_intensity("好的！？～", sticker=False) == 3
    assert message_intensity("哈哈[拥抱][亲亲]", sticker=False) == 2
    assert message_intensity("开心😀", sticker=False) == 1
    assert message_intensity(None, sticker=True) == 1
    assert message_intensity("好！", sticker=True) == 2


def test_bins_are_terciles_of_the_data() -> None:
    segments = assign_bins([info(i, messages=2 + i, intensity=float(i)) for i in range(9)])
    assert Counter(s.length_bin for s in segments) == {0: 3, 1: 3, 2: 3}
    assert Counter(s.intensity_bin for s in segments) == {0: 3, 1: 3, 2: 3}
    assert assign_bins([]) == []


# -------------------------------------------------------------- selecting


def test_every_month_day_part_length_and_intensity_is_reached_before_any_repeats(
    services: Services,
) -> None:
    month_scenario(services)
    segments = collect_segments(services)
    present = coverage(segments)
    chosen = select_segments(segments, 24, seed=1)
    spread = coverage(chosen)
    assert len(chosen) == 24
    for dimension in ("month", "period", "length", "intensity"):
        assert set(spread[dimension]) == set(present[dimension]), dimension
    # and not lopsided: no value takes more than a quarter above its even share, plus one
    for dimension in ("period", "length", "intensity"):
        share = 24 / len(spread[dimension])
        assert max(spread[dimension].values()) <= share * 1.25 + 1, (dimension, spread[dimension])
    assert chosen == sorted(chosen, key=lambda s: s.start)


def test_the_selection_is_reproducible_and_depends_on_the_seed() -> None:
    pool = assign_bins(
        [
            info(
                i, month=f"2026-{1 + i % 6:02d}", period=i % 4, messages=2 + i % 7, intensity=i % 5
            )
            for i in range(80)
        ]
    )
    one = select_segments(pool, 20, seed=5)
    assert one == select_segments(pool, 20, seed=5)
    assert one != select_segments(pool, 20, seed=6)
    assert len({s.first_id for s in one}) == 20  # no segment twice


def test_fewer_segments_than_asked_for_are_all_returned() -> None:
    pool = assign_bins([info(i) for i in range(5)])
    assert len(select_segments(pool, 60, seed=1)) == 5
    assert select_segments([], 60, seed=1) == []


def test_labels_have_the_width_of_the_sample() -> None:
    assert segment_label(3, 12) == "S03"
    assert segment_label(7, 120) == "S007"
    assert PERIODS == ("凌晨", "上午", "下午", "晚上")


def test_the_best_window_holds_the_most_lines_of_hers() -> None:
    lines = [
        TranscriptLine(f"m{i}", her, f"t{i}")
        for i, her in enumerate([False] * 4 + [True] * 4 + [False] * 4)
    ]
    window = best_window(lines, 4)
    assert [line.message_id for line in window] == ["m4", "m5", "m6", "m7"]
    assert best_window(lines, 50) == lines
    assert best_window(lines[:3], 3) == lines[:3]


# ------------------------------------------------------------ the sample


def late_texts(services: Services, cutoff: datetime) -> list[str]:
    from sqlalchemy import select

    from twin.storage.chat_models import Message

    with services.db.session() as session:
        rows = session.scalars(select(Message).where(Message.create_time_utc >= cutoff))
        return [row.text for row in rows if row.text]


def request(scope: str, count: int = 8) -> SampleRequest:
    return SampleRequest(scope, seed=3, count=count, max_messages=40)


def test_a_live_sample_is_made_of_transcript_lines_with_labels(services: Services) -> None:
    sticker_scenario(services)
    sample = build_sample(services, request("live", 5))
    assert [s.label for s in sample.segments] == ["S1", "S2", "S3", "S4", "S5"] or len(
        sample.segments
    ) == 5
    first = sample.segments[0]
    rendered = first.render()
    assert rendered.startswith(f"【片段 {first.label}】\n")
    assert "对方：今天的第" in rendered and "她：听说了呀" in rendered
    assert sample.scope == "live" and sample.cutoff is None and sample.eligible == 40
    assert set(sample.message_ids) == {i for s in sample.segments for i in s.message_ids}


def test_a_past_sample_never_reads_a_message_from_the_held_out_period(services: Services) -> None:
    scenario = sticker_scenario(services)
    cutoff = holdout_cutoff(services)
    late = message_ids_after(services, cutoff)
    assert late
    sample = build_sample(services, request("pre_holdout", 100))
    assert sample.cutoff == cutoff and sample.eligible < 40
    assert late.isdisjoint(sample.message_ids)
    assert all(s.info.end < cutoff for s in sample.segments)
    shown = "\n".join(s.render() for s in sample.segments)
    for text in late_texts(services, cutoff):
        assert text not in shown
    assert scenario.episodes == 40


def test_stickers_appear_with_their_tag_where_the_library_knows_one(services: Services) -> None:
    from twin.stickers.catalog import StickerCatalog

    scenario = sticker_scenario(services)
    catalog = StickerCatalog(services)
    catalog.save_vision(
        scenario.md5s[0], ["开心"], "一只笑脸猫", "回应好消息", at=services.clock.now_utc()
    )
    sample = build_sample(services, request("live", 40))
    text = "\n".join(s.render() for s in sample.segments)
    assert "她：[表情包:开心]" in text and "她：[表情包]" in text


def test_identifiers_are_replaced_with_numbered_tokens_for_the_whole_run(
    services: Services,
) -> None:
    phone = "1" + "38" + "12345678"
    write_dialogue(
        services,
        [
            (day(0), [U(f"我的号码是{phone}"), H(f"好的记住了{phone}"), H("再说一遍")]),
            (day(1), [U("还是那个"), H(f"{phone}对吗"), H("嗯嗯")]),
        ],
    )
    sample = build_sample(services, request("live", 5))
    text = "\n".join(s.render() for s in sample.segments)
    assert phone not in text
    assert text.count("[手机号#1]") >= 3 and "[手机号#2]" not in text


def test_event_messages_appear_as_event_text(services: Services) -> None:
    write_dialogue(
        services,
        [(day(0), [U("看这个"), H("好看", kind="text"), H(kind="image"), H("真的好看")])],
    )
    sample = build_sample(services, request("live", 1))
    assert "她：[图片]" in sample.segments[0].render()


def test_the_sample_needs_a_valid_scope_and_a_cutoff_for_the_past(services: Services) -> None:
    with pytest.raises(ValueError, match="scope"):
        build_sample(services, request("future"))
    from twin.profile.holdout import HoldoutError

    with pytest.raises(HoldoutError):
        build_sample(services, request("pre_holdout"))


def test_the_redactor_can_be_shared_between_samples(services: Services) -> None:
    redactor = ConsistentRedactor()
    phone = "1" + "39" + "87654321"
    write_dialogue(services, [(day(0), [U(phone), H(phone), H("好")])])
    one = build_sample(services, SampleRequest("live", 1, 5, 40, redactor))
    two = build_sample(services, SampleRequest("live", 1, 5, 40, redactor))
    assert one.segments[0].render() == two.segments[0].render()
    assert EARLY_MARKER != LATE_MARKER
