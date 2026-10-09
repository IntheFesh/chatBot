"""The style metrics of one time window, computed in a single pass (R-PROF-002).

A :class:`WindowCollector` is fed the messages of one window in time order.  It runs the unit
tracker (:mod:`twin.profile.units`) and keeps, for her and for the user separately, a
:class:`PartyAccumulator`.  Accumulators only keep counters (histograms), so the memory use
does not depend on the number of messages.  :meth:`WindowCollector.leaves` turns the counters
into the metric values of :mod:`twin.profile.values`: rates, vocabularies of closed sets and
sampleable distributions.  No leaf contains message text: the vocabularies are bracket emoji
codes, emoji characters, sentence-final particles and punctuation group names.

Metric names (the same for both sides)::

    kind_mix              fraction of each message kind
    text_length           characters per text message (distribution)
    punct_rate            fraction of text messages with each punctuation group
    end_rate              how text messages end (punctuation group, emoji code, none)
    emoji_code_rate       fraction of text messages with a bracket emoji code
    emoji_code_freq       share of each emoji code among all codes
    emoji_code_known_share  share of codes that are in the WeChat code table
    emoji_code_run_length length of runs of adjacent codes (distribution)
    unicode_emoji_rate / unicode_emoji_freq   the same for emoji characters
    sticker_share         stickers among all messages
    sticker_distinct / sticker_top3_share / sticker_usage   sticker variety, use per md5
    quote_rate            quotes among text and quote messages
    final_particle_rate / final_particle_freq   sentence-final particles
    laugh_rate / laugh_length   runs of "哈" (distribution of their length)
    question_rate         questions among text messages
    burst_size / burst_gap_s    messages per burst, seconds between them (distributions)
    reply_latency_s / reply_latency_by_hour   seconds to answer (all day, per local hour)
    delayed_reply_rate    answers after more than the segment gap, among all answers
    closing_no_reply_rate (her only) how often a closing message of the user - a short reply that
                          asks nothing, :mod:`twin.profile.closing` - was not answered inside its
                          segment
    initiations_per_day / initiation_hour   conversations opened, and at which local hours
    messages_per_day / message_hour   volume and its local time of day
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from twin.profile.closing import is_closing_message
from twin.profile.distribution import BucketedDistribution, EmpiricalDistribution
from twin.profile.textstats import END_CLASSES, PUNCT_GROUPS, analyse_text
from twin.profile.units import Block, Rec, Step, UnitTracker
from twin.profile.values import Dist, Hourly, Leaf, Rates, Scalar, Table

TEXT_KINDS = frozenset({"text", "quote"})
MAX_STICKER_ROWS = 3000
LATENCY_BUCKET_MIN_SAMPLES = 20
MIN_DAYS = 7


def _ratio(part: float, whole: float) -> float:
    return part / whole if whole else 0.0


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class PartyAccumulator:
    """Counters of one side of the conversation in one window."""

    codes_known: Collection[str]
    messages: int = 0
    kinds: Counter[str] = field(default_factory=Counter)
    text_n: int = 0
    length: Counter[float] = field(default_factory=Counter)
    groups: Counter[str] = field(default_factory=Counter)
    endings: Counter[str] = field(default_factory=Counter)
    code_messages: int = 0
    codes: Counter[str] = field(default_factory=Counter)
    code_runs: Counter[float] = field(default_factory=Counter)
    emoji_messages: int = 0
    emoji: Counter[str] = field(default_factory=Counter)
    particle_messages: int = 0
    particles: Counter[str] = field(default_factory=Counter)
    laugh_messages: int = 0
    laugh_length: Counter[float] = field(default_factory=Counter)
    question_messages: int = 0
    stickers: Counter[str] = field(default_factory=Counter)
    sticker_last: dict[str, float] = field(default_factory=dict)
    burst_sizes: Counter[float] = field(default_factory=Counter)
    burst_gaps: Counter[float] = field(default_factory=Counter)
    latency: Counter[float] = field(default_factory=Counter)
    latency_by_hour: dict[int, Counter[float]] = field(default_factory=dict)
    delayed_replies: int = 0
    initiations: int = 0
    initiation_hours: Counter[int] = field(default_factory=Counter)
    hours: Counter[int] = field(default_factory=Counter)

    def feed(self, rec: Rec, step: Step) -> None:
        self.messages += 1
        self.kinds[rec.kind] += 1
        self.hours[int(rec.stamp.minute // 60)] += 1
        if rec.kind == "sticker" and rec.sticker_md5:
            self.stickers[rec.sticker_md5] += 1
            self.sticker_last[rec.sticker_md5] = max(
                self.sticker_last.get(rec.sticker_md5, 0.0), rec.ts
            )
        if rec.kind in TEXT_KINDS and rec.text and rec.text.strip():
            self._feed_text(rec.text)
        if step.intra_gap_s is not None:
            self.burst_gaps[float(round(step.intra_gap_s))] += 1
        if step.initiation:
            self.initiations += 1
            self.initiation_hours[int(rec.stamp.minute // 60)] += 1
        if step.latency_s is not None and step.answered is not None:
            seconds = float(round(step.latency_s))
            self.latency[seconds] += 1
            hour = int(step.answered.stamp.minute // 60)
            self.latency_by_hour.setdefault(hour, Counter())[seconds] += 1
        if step.delayed_reply:
            self.delayed_replies += 1

    def _feed_text(self, text: str) -> None:
        facts = analyse_text(text)
        self.text_n += 1
        self.length[float(facts.length)] += 1
        for group in facts.groups:
            self.groups[group] += 1
        self.endings[facts.ending] += 1
        if facts.codes:
            self.code_messages += 1
            self.codes.update(facts.codes)
            for run in facts.code_runs:
                self.code_runs[float(run)] += 1
        if facts.emoji:
            self.emoji_messages += 1
            self.emoji.update(facts.emoji)
        if facts.particle is not None:
            self.particle_messages += 1
            self.particles[facts.particle] += 1
        if facts.laughs:
            self.laugh_messages += 1
            for run in facts.laughs:
                self.laugh_length[float(run)] += 1
        if facts.question:
            self.question_messages += 1

    def close_block(self, block: Block) -> None:
        self.burst_sizes[float(block.size)] += 1

    # ------------------------------------------------------------------ leaves

    def leaves(self, days: int) -> dict[str, Leaf]:
        total = self.messages
        text_n = self.text_n
        quote_base = self.kinds["text"] + self.kinds["quote"]
        code_total = sum(self.codes.values())
        known_total = sum(n for code, n in self.codes.items() if code[1:-1] in self.codes_known)
        emoji_total = sum(self.emoji.values())
        sticker_total = sum(self.stickers.values())
        top3 = sum(n for _, n in self.stickers.most_common(3))
        latency_n = sum(self.latency.values())
        answers = latency_n + self.delayed_replies
        usage = {
            md5: {"n": n, "last": _iso(self.sticker_last[md5])}
            for md5, n in self.stickers.most_common(MAX_STICKER_ROWS)
        }
        hourly = BucketedDistribution.from_counters(
            self.latency_by_hour,
            sizes=(1,),
            min_samples=LATENCY_BUCKET_MIN_SAMPLES,
            discrete=True,
        )
        out: dict[str, Leaf] = {
            "kind_mix": Rates({k: _ratio(n, total) for k, n in self.kinds.items()}, total),
            "text_length": Dist(EmpiricalDistribution.from_counter(self.length, discrete=True)),
            "punct_rate": Rates({g: _ratio(self.groups[g], text_n) for g in PUNCT_GROUPS}, text_n),
            "end_rate": Rates({c: _ratio(self.endings[c], text_n) for c in END_CLASSES}, text_n),
            "emoji_code_rate": Scalar(_ratio(self.code_messages, text_n), text_n),
            "emoji_code_freq": Rates(
                {code: _ratio(n, code_total) for code, n in self.codes.items()}, code_total
            ),
            "emoji_code_known_share": Scalar(_ratio(known_total, code_total), code_total),
            "emoji_code_run_length": Dist(
                EmpiricalDistribution.from_counter(self.code_runs, discrete=True)
            ),
            "unicode_emoji_rate": Scalar(_ratio(self.emoji_messages, text_n), text_n),
            "unicode_emoji_freq": Rates(
                {e: _ratio(n, emoji_total) for e, n in self.emoji.items()}, emoji_total
            ),
            "sticker_share": Scalar(_ratio(sticker_total, total), total),
            "sticker_distinct": Scalar(float(len(self.stickers)), sticker_total),
            "sticker_top3_share": Scalar(_ratio(top3, sticker_total), sticker_total),
            "sticker_usage": Table(usage, sticker_total),
            "quote_rate": Scalar(_ratio(self.kinds["quote"], quote_base), quote_base),
            "final_particle_rate": Scalar(_ratio(self.particle_messages, text_n), text_n),
            "final_particle_freq": Rates(
                {p: _ratio(n, self.particle_messages) for p, n in self.particles.items()},
                self.particle_messages,
            ),
            "laugh_rate": Scalar(_ratio(self.laugh_messages, text_n), text_n),
            "laugh_length": Dist(
                EmpiricalDistribution.from_counter(self.laugh_length, discrete=True)
            ),
            "question_rate": Scalar(_ratio(self.question_messages, text_n), text_n),
            "burst_size": Dist(EmpiricalDistribution.from_counter(self.burst_sizes, discrete=True)),
            "burst_gap_s": Dist(EmpiricalDistribution.from_counter(self.burst_gaps, discrete=True)),
            "reply_latency_s": Dist(
                EmpiricalDistribution.from_counter(self.latency, discrete=True)
            ),
            "reply_latency_by_hour": Hourly(hourly),
            "delayed_reply_rate": Scalar(_ratio(self.delayed_replies, answers), answers),
            "initiations_per_day": Scalar(_ratio(self.initiations, days), days, MIN_DAYS),
            "initiation_hour": Rates(
                {str(h): _ratio(n, self.initiations) for h, n in self.initiation_hours.items()},
                self.initiations,
            ),
            "messages_per_day": Scalar(_ratio(total, days), days, MIN_DAYS),
            "message_hour": Rates({str(h): _ratio(n, total) for h, n in self.hours.items()}, total),
        }
        return out


class WindowCollector:
    """Metrics of both sides for one window of messages, fed in time order."""

    def __init__(
        self, burst_gap_s: float, segment_gap_s: float, known_codes: Collection[str]
    ) -> None:
        self.tracker = UnitTracker(burst_gap_s, segment_gap_s)
        self.sides = {
            True: PartyAccumulator(known_codes),
            False: PartyAccumulator(known_codes),
        }
        self._previous: Rec | None = None
        self._closings = 0  # closing messages of the user that a later message has followed
        self._closings_answered = 0  # ... and that she answered inside the same segment
        self._days: set[date] = set()
        self.first_ts: float | None = None
        self.last_ts: float | None = None
        self._closed = False

    def feed(self, rec: Rec) -> Step:
        step = self.tracker.feed(rec)
        if self.first_ts is None:
            self.first_ts = rec.ts
        self.last_ts = rec.ts
        self._days.add(rec.stamp.day)
        if step.closed is not None:
            self.sides[step.closed.her].close_block(step.closed)
            previous = self._previous
            if (
                not step.closed.her
                and previous is not None
                and is_closing_message(previous.kind, previous.text)
            ):
                self._closings += 1
                self._closings_answered += int(rec.her and step.latency_s is not None)
        self._previous = rec
        self.sides[rec.her].feed(rec, step)
        return step

    def finish(self) -> None:
        if not self._closed:
            self._closed = True
            block = self.tracker.finish()
            if block is not None:
                self.sides[block.her].close_block(block)

    @property
    def days(self) -> int:
        return len(self._days)

    def info(self) -> dict[str, object]:
        return {
            "start": _iso(self.first_ts) if self.first_ts is not None else None,
            "end": _iso(self.last_ts) if self.last_ts is not None else None,
            "days": self.days,
            "her_messages": self.sides[True].messages,
            "user_messages": self.sides[False].messages,
            "segments": self.tracker.segments,
        }

    def leaves(self) -> dict[str, dict[str, Leaf]]:
        self.finish()
        days = max(1, self.days)
        her = self.sides[True].leaves(days)
        her["closing_no_reply_rate"] = Scalar(
            _ratio(self._closings - self._closings_answered, self._closings), self._closings
        )
        return {"her": her, "user": self.sides[False].leaves(days)}
