"""Stratified sampling of conversation segments for the persona card (R-PERS-001).

A *segment* is a stretch of the conversation that ends when two consecutive messages are more
than ``profile.segment_gap_min`` apart (the one definition of
:class:`~twin.profile.units.BurstSegmenter`).  One pass over the messages measures every
segment that she took part in; :func:`select_segments` then picks ``persona.sample_segments`` of
them so that each of four properties is spread as evenly as the data allows:

* the month (in the time zone she was in),
* the time of day (night / morning / afternoon / evening, four six-hour bins of the same local
  clock),
* the length of the segment (three bins by message count),
* the emotional intensity - a simple, explainable number: question marks, exclamation marks,
  tildes, ellipses, emoji codes, emoji characters and stickers per message of hers - in three
  bins.

The selection is greedy: each step takes a segment of the stratum whose values have been chosen
the fewest times so far, so every month and every bin is reached before any is repeated.  Ties
are broken by a seeded shuffle, so the same data and seed give the same sample.

The scope decides which messages exist (R-TRN-013): ``live`` reads everything, ``pre_holdout``
stops at the hold-out cutoff and never reads a message at or after it.  A chosen segment is
loaded as a transcript (event texts for non-text messages, sticker tags where known) and
redacted with one :class:`~twin.llm.redaction.ConsistentRedactor` for the whole run before it is
shown to a model.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from twin.ingest.corpus import conversation_messages, messages_between
from twin.ingest.times import SourceTime
from twin.ingest.transcript import (
    HER_LABEL,
    USER_LABEL,
    StickerTagOf,
    TranscriptLine,
    transcript_lines,
)
from twin.llm.redaction import ConsistentRedactor
from twin.profile.holdout import holdout_cutoff
from twin.profile.localtime import LocalClock
from twin.profile.textstats import find_emoji_codes, find_unicode_emoji, punctuation_groups
from twin.profile.units import BurstSegmenter
from twin.services import Services
from twin.stickers.catalog import StickerCatalog

MIN_HER_MESSAGES = 2
PERIODS = ("凌晨", "上午", "下午", "晚上")
INTENSE_GROUPS = frozenset({"question", "exclaim", "tilde", "ellipsis"})
DIMENSIONS = ("month", "period", "length", "intensity")


@dataclass(frozen=True)
class SegmentInfo:
    """What the sampler knows about one segment (no text)."""

    start: datetime
    end: datetime
    first_id: str
    last_id: str
    messages: int
    her_messages: int
    month: str
    period: int
    intensity: float
    length_bin: int = 0
    intensity_bin: int = 0

    @property
    def stratum(self) -> tuple[str, int, int, int]:
        return (self.month, self.period, self.length_bin, self.intensity_bin)


@dataclass(frozen=True)
class SampledSegment:
    """A chosen segment, loaded: the lines a model will see (already redacted)."""

    label: str
    info: SegmentInfo
    lines: tuple[TranscriptLine, ...]

    @property
    def message_ids(self) -> tuple[str, ...]:
        return tuple(line.message_id for line in self.lines)

    def render(self) -> str:
        body = "\n".join(
            line.render(her_label=HER_LABEL, user_label=USER_LABEL) for line in self.lines
        )
        return f"【片段 {self.label}】\n{body}"


@dataclass(frozen=True)
class Sample:
    """The result of one sampling run."""

    scope: str
    cutoff: datetime | None
    segments: tuple[SampledSegment, ...]
    eligible: int  # segments that were candidates
    seed: int

    @property
    def labels(self) -> frozenset[str]:
        return frozenset(segment.label for segment in self.segments)

    @property
    def message_ids(self) -> tuple[str, ...]:
        return tuple(i for segment in self.segments for i in segment.message_ids)


# ----------------------------------------------------------------------- measuring


@dataclass
class _Open:
    start: datetime
    first_id: str
    month: str
    period: int
    last: datetime
    last_id: str
    messages: int = 0
    her_messages: int = 0
    intensity_sum: float = 0.0

    def close(self) -> SegmentInfo:
        average = self.intensity_sum / self.her_messages if self.her_messages else 0.0
        return SegmentInfo(
            self.start,
            self.last,
            self.first_id,
            self.last_id,
            self.messages,
            self.her_messages,
            self.month,
            self.period,
            average,
        )


def message_intensity(text: str | None, *, sticker: bool) -> float:
    """The intensity of one message of hers: marks, emoji codes, emoji characters, stickers."""
    score = 1.0 if sticker else 0.0
    if text:
        score += len(INTENSE_GROUPS & punctuation_groups(text))
        score += len(find_emoji_codes(text)) + len(find_unicode_emoji(text))
    return score


def collect_segments(services: Services, *, before: datetime | None = None) -> list[SegmentInfo]:
    """Every segment she took part in (messages at or after ``before`` are never read)."""
    settings = services.settings.profile
    segmenter = BurstSegmenter(settings.burst_gap_s, settings.segment_gap_min * 60.0)
    clock = LocalClock(SourceTime.from_config(services.settings.time))
    found: list[SegmentInfo] = []
    current: _Open | None = None

    def close() -> None:
        if current is not None and current.her_messages >= MIN_HER_MESSAGES:
            found.append(current.close())

    stmt = conversation_messages().execution_options(yield_per=5000)
    with services.db.session() as session:
        for row in session.scalars(stmt):
            if row.kind == "system":
                continue
            moment = row.create_time_utc
            if before is not None and moment >= before:
                break  # rows are in time order: nothing later may be read
            boundary = segmenter.feed(moment.timestamp(), not row.is_sent)
            if boundary.new_segment or current is None:
                close()
                stamp = clock.stamp(moment)
                current = _Open(
                    moment,
                    row.id,
                    f"{stamp.day:%Y-%m}",
                    min(3, int(stamp.minute // 360)),
                    moment,
                    row.id,
                )
            current.messages += 1
            current.last = moment
            current.last_id = row.id
            if not row.is_sent:
                current.her_messages += 1
                current.intensity_sum += message_intensity(
                    row.text if row.kind in ("text", "quote") else None,
                    sticker=row.kind == "sticker",
                )
    close()
    return assign_bins(found)


def _thirds(values: Sequence[float]) -> tuple[float, float]:
    """The values that end the lower and the middle third (a third may hold many ties)."""
    ordered = sorted(values)
    count = len(ordered)
    return ordered[max(0, count // 3 - 1)], ordered[max(0, (2 * count) // 3 - 1)]


def _bin(value: float, cuts: tuple[float, float]) -> int:
    return 0 if value <= cuts[0] else 1 if value <= cuts[1] else 2


def assign_bins(segments: Sequence[SegmentInfo]) -> list[SegmentInfo]:
    """The same segments with their length and intensity bins (terciles of this data)."""
    if not segments:
        return []
    length_cuts = _thirds([float(s.messages) for s in segments])
    intensity_cuts = _thirds([s.intensity for s in segments])
    return [
        SegmentInfo(
            s.start,
            s.end,
            s.first_id,
            s.last_id,
            s.messages,
            s.her_messages,
            s.month,
            s.period,
            s.intensity,
            _bin(float(s.messages), length_cuts),
            _bin(s.intensity, intensity_cuts),
        )
        for s in segments
    ]


# ----------------------------------------------------------------------- selecting


def select_segments(segments: Sequence[SegmentInfo], count: int, seed: int) -> list[SegmentInfo]:
    """``count`` segments spread over the months, day parts, lengths and intensities.

    Returned in time order.  If there are fewer segments than ``count`` all are returned.
    """
    rng = random.Random(seed)  # noqa: S311 - a reproducible sample, not security
    pool = list(segments)
    rng.shuffle(pool)
    strata: dict[tuple[str, int, int, int], list[SegmentInfo]] = {}
    for segment in pool:
        strata.setdefault(segment.stratum, []).append(segment)
    tallies: tuple[Counter[object], ...] = tuple(Counter() for _ in DIMENSIONS)
    chosen: list[SegmentInfo] = []
    while len(chosen) < count and strata:
        key = min(
            strata, key=lambda k: sum(tallies[i][value] for i, value in enumerate(k))
        )  # ties: the first stratum in the (shuffled) insertion order
        segment = strata[key].pop()
        if not strata[key]:
            del strata[key]
        for index, value in enumerate(segment.stratum):
            tallies[index][value] += 1
        chosen.append(segment)
    chosen.sort(key=lambda s: (s.start, s.first_id))
    return chosen


def coverage(segments: Sequence[SegmentInfo]) -> dict[str, Counter[object]]:
    """How many chosen segments fall in each value of each dimension."""
    result: dict[str, Counter[object]] = {name: Counter() for name in DIMENSIONS}
    for segment in segments:
        for name, value in zip(DIMENSIONS, segment.stratum, strict=True):
            result[name][value] += 1
    return result


# ----------------------------------------------------------------------- loading


def best_window(lines: Sequence[TranscriptLine], size: int) -> list[TranscriptLine]:
    """The stretch of ``size`` consecutive lines that holds the most lines of hers."""
    if len(lines) <= size:
        return list(lines)
    her = [1 if line.her else 0 for line in lines]
    running = sum(her[:size])
    best, best_start = running, 0
    for start in range(1, len(lines) - size + 1):
        running += her[start + size - 1] - her[start - 1]
        if running > best:
            best, best_start = running, start
    return list(lines[best_start : best_start + size])


def load_segment(
    services: Services,
    info: SegmentInfo,
    label: str,
    *,
    before: datetime | None,
    max_messages: int,
    sticker_tag_of: StickerTagOf | None,
    redactor: ConsistentRedactor,
) -> SampledSegment:
    """Read the messages of a segment and turn them into redacted transcript lines."""
    with services.db.session() as session:
        rows = list(session.scalars(messages_between(info.start, info.end, before=before)))
        lines = transcript_lines(rows, sticker_tag_of=sticker_tag_of)
    shown = best_window(lines, max_messages)
    clean = tuple(
        TranscriptLine(line.message_id, line.her, redactor.redact_text(line.text)) for line in shown
    )
    return SampledSegment(label, info, clean)


def segment_label(index: int, total: int) -> str:
    return f"S{index:0{max(2, len(str(total)))}d}"


@dataclass
class SampleRequest:
    """What :func:`build_sample` is asked for."""

    scope: str
    seed: int
    count: int
    max_messages: int
    redactor: ConsistentRedactor = field(default_factory=ConsistentRedactor)


def build_sample(services: Services, request: SampleRequest) -> Sample:
    """Measure, choose and load the segments of one scope."""
    if request.scope not in ("live", "pre_holdout"):
        raise ValueError("scope must be live or pre_holdout")
    cutoff = holdout_cutoff(services) if request.scope == "pre_holdout" else None
    infos = collect_segments(services, before=cutoff)
    chosen = select_segments(infos, request.count, request.seed)
    tag_of = StickerCatalog(services).tag_lookup()
    segments = tuple(
        load_segment(
            services,
            info,
            segment_label(index, len(chosen)),
            before=cutoff,
            max_messages=request.max_messages,
            sticker_tag_of=tag_of,
            redactor=request.redactor,
        )
        for index, info in enumerate(chosen, start=1)
    )
    return Sample(request.scope, cutoff, segments, len(infos), request.seed)
