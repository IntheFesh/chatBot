"""The basic units of the profile and the routine model (R-PROF-001, R-ACT-002).

Definitions (all gaps are measured between the *times of two consecutive messages* of the
conversation, whoever sent them; ``system`` notices are not messages of either side and are
left out before any of this):

**Burst** (合并连发块)
    consecutive messages of the *same sender* whose neighbours are at most
    ``profile.burst_gap_s`` (120 s) apart.  A message from the other side, or a pause longer
    than that, ends the burst.  Messages of every kind except ``system`` count as burst
    members (a sticker between two texts is part of the burst); text metrics read only the
    ``text`` kind and the reply body of ``quote`` (R-PROF-002).

**Segment** (会话段)
    a conversation segment ends when two consecutive messages are more than
    ``profile.segment_gap_min`` (60 minutes) apart; the next message starts a new segment.
    The same setting is used by the example windows of the retrieval library (R-RET-001).

**Reply latency** (回复延迟)
    when a burst of one side directly follows a burst of the other side, the time from the
    last message of the earlier burst to the first message of the later one.  Only counted
    if both lie in the same segment; a reply that comes after a gap of more than the
    segment gap is counted separately as a *delayed reply* (未即时回复) and contributes no
    latency sample.

**Initiation** (先开口)
    the first message after a silence of at least ``profile.segment_gap_min`` minutes; the
    sender of that message initiates.  (A silence of exactly 60:00 therefore starts an
    initiation while it still belongs to the same segment: the two rules are stated with
    "at least" and "more than" respectively.)  The very first message of the data is not
    counted: the silence before it is unknown.

:class:`UnitTracker` walks the messages in time order, one at a time, and reports for each
message what it started or ended, so that the profile and the routine model can be computed
in one pass over millions of messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from twin.profile.localtime import LocalStamp


class Rec(NamedTuple):
    """One message as the statistics see it."""

    id: str  # the message id (only used to prove what a scope has read)
    ts: float  # epoch seconds
    her: bool  # True: she sent it
    kind: str
    text: str | None  # only for kinds text and quote
    sticker_md5: str | None
    stamp: LocalStamp
    day_type: str


@dataclass(slots=True)
class Block:
    """A finished burst."""

    her: bool
    size: int
    start_ts: float
    end_ts: float


@dataclass(slots=True)
class Step:
    """What one message did to the units (see :class:`UnitTracker`)."""

    rec: Rec
    new_block: bool = False
    closed: Block | None = None  # the burst this message ended, if any
    intra_gap_s: float | None = None  # gap to the previous message inside a burst
    initiation: bool = False
    new_segment: bool = False
    latency_s: float | None = None  # reply latency of the burst this message starts
    answered: Rec | None = None  # the message that latency is measured from
    delayed_reply: bool = False  # the burst answers across a segment boundary


@dataclass(slots=True)
class Boundary:
    """What one message did to the units, without reference to who it was (see below)."""

    new_block: bool = False
    closed: Block | None = None  # the burst this message ended, if any
    intra_gap_s: float | None = None  # the same gap, when the message joins the open burst
    new_segment: bool = False
    initiation: bool = False
    reply_to_ts: float | None = None  # end of the burst this one answers, same segment
    delayed_reply: bool = False  # the burst answers across a segment boundary


class BurstSegmenter:
    """The definitions above on bare ``(time, sender)`` pairs, fed in time order.

    It is the single implementation of bursts and segments: the profile statistics
    (:class:`UnitTracker`) and the hold-out split (:mod:`twin.profile.holdout`) both use it.
    """

    def __init__(self, burst_gap_s: float, segment_gap_s: float) -> None:
        self._burst_gap = float(burst_gap_s)
        self._segment_gap = float(segment_gap_s)
        self._previous_ts: float | None = None
        self._previous_her = False
        self._open: Block | None = None
        self.segments = 0

    def feed(self, ts: float, her: bool) -> Boundary:
        out = Boundary()
        previous_ts = self._previous_ts
        previous_her = self._previous_her
        self._previous_ts = ts
        self._previous_her = her
        if previous_ts is None:
            self._open = Block(her, 1, ts, ts)
            out.new_block = True
            out.new_segment = True
            self.segments += 1
            return out
        gap = max(0.0, ts - previous_ts)
        same_sender = her == previous_her
        out.new_segment = gap > self._segment_gap
        if out.new_segment:
            self.segments += 1
        out.initiation = gap >= self._segment_gap
        if same_sender and gap <= self._burst_gap and self._open is not None:
            self._open.size += 1
            self._open.end_ts = ts
            out.intra_gap_s = gap
            return out
        out.new_block = True
        out.closed = self._open
        self._open = Block(her, 1, ts, ts)
        if not same_sender:
            if out.new_segment:
                out.delayed_reply = True
            else:
                out.reply_to_ts = previous_ts
        return out

    def finish(self) -> Block | None:
        """The burst still open at the end of the data."""
        block, self._open = self._open, None
        return block


class UnitTracker:
    """Streams messages in time order and reports bursts, segments, latencies, initiations."""

    def __init__(self, burst_gap_s: float, segment_gap_s: float) -> None:
        self._segmenter = BurstSegmenter(burst_gap_s, segment_gap_s)
        self._previous: Rec | None = None

    @property
    def segments(self) -> int:
        return self._segmenter.segments

    def feed(self, rec: Rec) -> Step:
        boundary = self._segmenter.feed(rec.ts, rec.her)
        previous = self._previous
        self._previous = rec
        step = Step(
            rec,
            new_block=boundary.new_block,
            closed=boundary.closed,
            intra_gap_s=boundary.intra_gap_s,
            initiation=boundary.initiation,
            new_segment=boundary.new_segment,
            delayed_reply=boundary.delayed_reply,
        )
        if boundary.reply_to_ts is not None and previous is not None:
            step.latency_s = rec.ts - boundary.reply_to_ts
            step.answered = previous
        return step

    def finish(self) -> Block | None:
        """The burst still open at the end of the data."""
        return self._segmenter.finish()
