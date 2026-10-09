"""Bursts, segments, reply latency and initiations (R-PROF-001, task section A)."""

from __future__ import annotations

from datetime import date

import pytest

from twin.profile.localtime import LocalStamp
from twin.profile.units import BurstSegmenter, Rec, Step, UnitTracker

STAMP = LocalStamp(date(2026, 1, 5), 600.0, 40, "UTC")


def rec(ts: float, her: bool, kind: str = "text") -> Rec:
    return Rec(f"m{ts}", ts, her, kind, "x", None, STAMP, "workday")


def run(
    messages: list[tuple[float, bool]], burst: float = 120, segment: float = 3600
) -> tuple[list[Step], UnitTracker]:
    tracker = UnitTracker(burst, segment)
    return [tracker.feed(rec(ts, her)) for ts, her in messages], tracker


def test_neighbours_within_the_burst_gap_form_one_burst() -> None:
    steps, tracker = run([(0, True), (5, True), (125, True), (130, False)])
    assert [s.new_block for s in steps] == [True, False, False, True]
    assert [s.intra_gap_s for s in steps] == [None, 5, 120, None]  # 120 s is still inside
    closed = steps[3].closed
    assert closed is not None and (closed.her, closed.size) == (True, 3)
    last = tracker.finish()
    assert last is not None and (last.her, last.size) == (False, 1)
    assert tracker.finish() is None


def test_a_pause_longer_than_the_burst_gap_starts_a_new_burst_without_a_reply() -> None:
    steps, _ = run([(0, True), (121, True)])
    assert steps[1].new_block and steps[1].closed is not None and steps[1].closed.size == 1
    assert steps[1].latency_s is None and not steps[1].delayed_reply


def test_the_reply_latency_runs_from_the_last_message_of_the_other_burst() -> None:
    steps, _ = run([(0, False), (10, False), (45, True), (50, True)])
    reply = steps[2]
    assert reply.latency_s == 35  # from the user's last message at 10 to her first at 45
    assert reply.answered is not None and reply.answered.ts == 10
    assert steps[3].latency_s is None and steps[3].intra_gap_s == 5


def test_a_reply_after_more_than_the_segment_gap_is_counted_as_delayed() -> None:
    steps, tracker = run([(0, False), (3601, True), (3602, True)])
    late = steps[1]
    assert late.new_segment and late.delayed_reply and late.latency_s is None
    assert late.initiation  # the first message after a long silence opens the conversation
    assert tracker.segments == 2
    assert not steps[2].new_segment and not steps[2].initiation


def test_the_segment_gap_boundary_is_strict_for_segments_and_inclusive_for_initiations() -> None:
    steps, tracker = run([(0, False), (3600, True)])
    boundary = steps[1]
    assert not boundary.new_segment  # "more than" 60 minutes starts a segment
    assert boundary.initiation  # "at least" 60 minutes of silence is an initiation
    assert boundary.latency_s == 3600 and not boundary.delayed_reply
    assert tracker.segments == 1


def test_the_first_message_of_the_data_is_not_an_initiation() -> None:
    steps, tracker = run([(0, True), (10, False)])
    assert not steps[0].initiation and steps[0].new_segment and tracker.segments == 1


def test_a_long_pause_by_the_same_sender_is_an_initiation_but_no_delayed_reply() -> None:
    steps, _ = run([(0, True), (7200, True)])
    assert steps[1].initiation and steps[1].new_segment
    assert not steps[1].delayed_reply and steps[1].latency_s is None


def test_simultaneous_messages_stay_in_the_burst() -> None:
    steps, _ = run([(10, True), (10, True), (10, False)])
    assert steps[1].intra_gap_s == 0 and steps[2].latency_s == 0


def test_the_segmenter_uses_bare_times_and_senders() -> None:
    segmenter = BurstSegmenter(60, 1800)
    first = segmenter.feed(0.0, True)
    assert first.new_block and first.initiation is False
    joined = segmenter.feed(30.0, True)
    assert not joined.new_block and joined.intra_gap_s == 30
    answer = segmenter.feed(100.0, False)
    assert answer.reply_to_ts == 30.0 and answer.closed is not None and answer.closed.size == 2


@pytest.mark.parametrize(("gap", "expected"), [(119.9, False), (120.0, False), (120.1, True)])
def test_burst_gap_is_configurable_and_inclusive(gap: float, expected: bool) -> None:
    steps, _ = run([(0, True), (gap, True)])
    assert steps[1].new_block is expected
