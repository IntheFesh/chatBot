"""Style metrics of what the bot wrote, against her profile (R-EVAL-002, R-PROF-002).

The six core metrics of the style profile - the median length of a text, the share of texts with
a comma, the median size of a burst, the share of messages that are stickers, the share of texts
with an emoji code and the share of text replies that quote - are measured on the bot's output and
compared with the same metric of her profile.  **The measuring is the profile's own**: the bot's
messages are fed to :class:`~twin.profile.metrics.WindowCollector` (the code that builds the
profile from her messages), its leaves become a :class:`~twin.profile.snapshot.ProfileMetrics`
document by the same :func:`~twin.profile.snapshot.assemble_metrics`, and both sides are read by the
one function :func:`read_metrics`.  Nothing here counts a comma.

Two sources (R-EVAL-002):

* ``live`` - the bubbles the bot really sent in the last ``days`` days (read through
  :mod:`twin.engine.conversation_log`, commands left out), against the **live** profile;
* ``eval_items`` - the replies the sandbox generated for the contexts of a blind run
  (hold-out, ``pre_holdout`` profile), for one backend (the M5 gate).  The same metrics of her
  real replies in those very contexts are shown next to them, as a feeling for the noise of a
  small sample.

**A metric the channel cannot produce is not measurable** (round 18, user decision, D-851): a
metric names the ability of the channel it depends on (the quote rate: ``supports_quote``), and in
both sources, for every backend alike, it is shown as "not measurable" and left out of the verdict
when the channel the replies went through lacks that ability - the running channel for ``live``
(:func:`twin.channel.configured.configured_capabilities`), the sandbox's channel for
``eval_items`` (each generated pair records the abilities of the channel it went through; a pair
that does not say, generated before round 18, leaves the metric without data).  For the replies of
the sandbox the report also gives, **outside the verdict**, the share of replies in which the model
wrote a ``[引用:…]`` line before the post-processing removed it (a count each pair keeps).

A metric passes when its relative deviation ``(bot - her) / her`` is within +-30 %.  When her value
is 0 the bot has to be 0 as well.  A metric with no data on either side does not pass.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from twin.channel.configured import abilities_of, configured_capabilities
from twin.engine.conversation_log import conversation_since
from twin.eval.render import Candidate
from twin.eval.stats import wilson_interval
from twin.eval.store import EvalStore, ItemView, RunView
from twin.profile.api import load_profile
from twin.profile.localtime import LocalStamp, slot_of_minute
from twin.profile.metrics import WindowCollector
from twin.profile.snapshot import ProfileMetrics, assemble_metrics
from twin.profile.units import Rec

if TYPE_CHECKING:
    from twin.services import Services

TOLERANCE = 0.30  # R-EVAL-002: each metric within +-30 % of hers
EPSILON = 1e-9  # floating-point slack only: 1.3 against 1.0 is 30 %, not 30.000000000000004 %
ITEM_SPACING_S = 7200.0  # between two replies: far more than a burst gap
BUBBLE_SPACING_S = 2.0  # between the bubbles of one reply: well inside a burst
EPOCH = datetime(2026, 1, 1, tzinfo=UTC)

Status = Literal["pass", "fail", "n/a", "no_data"]
Abilities = Mapping[str, bool | None]  # an ability of a channel -> it has it (None: not known)


@dataclass(frozen=True)
class MetricReading:
    """A metric of one side: its value and the number of observations it rests on."""

    value: float | None
    n: int


@dataclass(frozen=True)
class MetricSpec:
    """One of the six core metrics: where it is read, and what it needs of the channel."""

    key: str
    label: str
    read: Callable[[ProfileMetrics], MetricReading]
    proportion: bool = True
    needs: str | None = None  # the ability of the channel (``supports_*``) it depends on (D-851)

    def measurable(self, channel: Abilities | None) -> bool | None:
        """Whether replies through ``channel`` can show it (``None``: the record does not say).

        ``channel`` ``None`` is replies that went through no channel at all (the raw output of a
        model, as the comparison of two adapter stacks reads it): every metric can be measured.
        """
        if self.needs is None or channel is None:
            return True
        return channel.get(self.needs)


def _median_of(name: str) -> Callable[[ProfileMetrics], MetricReading]:
    def read(metrics: ProfileMetrics) -> MetricReading:
        dist = metrics.distribution("her", name)
        return MetricReading(dist.median() if dist is not None else None, dist.n if dist else 0)

    return read


def _scalar_of(name: str) -> Callable[[ProfileMetrics], MetricReading]:
    def read(metrics: ProfileMetrics) -> MetricReading:
        leaf = metrics.leaf("her", name)
        value = metrics.scalar("her", name)
        return MetricReading(value, getattr(leaf, "n", 0) if value is not None else 0)

    return read


def _comma(metrics: ProfileMetrics) -> MetricReading:
    leaf = metrics.leaf("her", "punct_rate")
    rates = metrics.rates("her", "punct_rate")
    if leaf is None or "comma" not in rates:
        return MetricReading(None, 0)
    n = getattr(leaf, "n", 0)
    return MetricReading(rates["comma"] if n > 0 else None, n)


STYLE_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("text_length", "文字长度中位数", _median_of("text_length"), proportion=False),
    MetricSpec("comma_rate", "逗号率", _comma),
    MetricSpec("burst_size", "连发中位数", _median_of("burst_size"), proportion=False),
    MetricSpec("sticker_share", "表情包占比", _scalar_of("sticker_share")),
    MetricSpec("emoji_code_rate", "表情代码率", _scalar_of("emoji_code_rate")),
    MetricSpec("quote_rate", "引用率", _scalar_of("quote_rate"), needs="supports_quote"),
)


def read_metrics(metrics: ProfileMetrics) -> dict[str, MetricReading]:
    """The six core metrics of her side of a profile document (profile or measured alike)."""
    return {spec.key: spec.read(metrics) for spec in STYLE_METRICS}


# ------------------------------------------------------------------ measuring


def _stamp(moment: datetime) -> LocalStamp:
    minute = moment.hour * 60 + moment.minute + moment.second / 60.0
    return LocalStamp(moment.date(), minute, slot_of_minute(minute), "UTC")


def _rec(
    message_id: str, moment: datetime, *, her: bool, kind: str, text: str | None, md5: str | None
) -> Rec:
    return Rec(message_id, moment.timestamp(), her, kind, text, md5, _stamp(moment), "workday")


def candidate_recs(candidates: Iterable[Candidate]) -> Iterator[Rec]:
    """Her side of a conversation made of these replies, one burst per reply."""
    for index, candidate in enumerate(candidates):
        base = EPOCH + timedelta(seconds=index * ITEM_SPACING_S)
        quote_left = bool(candidate.quote)
        for position, line in enumerate(candidate.lines):
            moment = base + timedelta(seconds=position * BUBBLE_SPACING_S)
            ident = f"c{index}-{position}"
            if line.kind == "sticker":
                yield _rec(ident, moment, her=True, kind="sticker", text=None, md5=line.sticker_md5)
                continue
            kind = "quote" if quote_left else "text"
            quote_left = False  # the quote goes with the first text line only
            yield _rec(ident, moment, her=True, kind=kind, text=line.text, md5=None)


def live_recs(services: Services, since: datetime) -> Iterator[Rec]:
    """The conversation with the bot since ``since``: his messages and what the bot sent."""
    for turn in conversation_since(services, since):
        yield _rec(
            turn.id,
            turn.at,
            her=turn.outbound,
            kind=turn.kind,
            text=turn.text if turn.kind == "text" else None,
            md5=turn.sticker_md5 if turn.kind == "sticker" else None,
        )


def measure(recs: Iterable[Rec], burst_gap_s: float, segment_gap_min: float) -> ProfileMetrics:
    """The profile metrics of a stream of messages in time order, computed as the profile is."""
    collector = WindowCollector(burst_gap_s, segment_gap_min * 60.0, frozenset())
    for rec in recs:
        collector.feed(rec)
    return ProfileMetrics(
        assemble_metrics(
            scope="measured",
            config={},
            weight=0.0,
            full=collector.leaves(),
            recent=None,
            full_info=collector.info(),
            recent_info={},
        )
    )


# ------------------------------------------------------------------ comparing


@dataclass(frozen=True)
class MetricResult:
    """One metric: hers, the bot's, the relative deviation and the verdict."""

    key: str
    label: str
    reference: MetricReading
    measured: MetricReading
    deviation: float | None
    status: Status
    interval: tuple[float, float] | None = None
    real: MetricReading | None = None  # her real replies in the same contexts, when known

    @property
    def counts(self) -> bool:
        return self.status != "n/a"

    def to_json(self) -> dict[str, Any]:
        """The numbers of the metric (no text): hers, the bot's, the deviation and the verdict."""

        def reading(value: MetricReading | None) -> dict[str, Any] | None:
            return None if value is None else {"value": value.value, "n": value.n}

        return {
            "key": self.key,
            "label": self.label,
            "reference": reading(self.reference),
            "measured": reading(self.measured),
            "deviation": self.deviation,
            "status": self.status,
            "interval": list(self.interval) if self.interval else None,
            "real": reading(self.real),
        }


def deviation_of(reference: float | None, measured: float | None) -> float | None:
    """``(measured - reference) / reference``; 0 when both are 0; ``None`` when undefined."""
    if reference is None or measured is None:
        return None
    if reference == 0:
        return 0.0 if measured == 0 else None
    return (measured - reference) / reference


def judge_metric(
    spec: MetricSpec,
    reference: MetricReading,
    measured: MetricReading,
    *,
    measurable: bool | None = True,
    real: MetricReading | None = None,
) -> MetricResult:
    """The verdict on one metric (see the module description).

    ``measurable`` ``False``: the channel cannot produce it - "not measurable", left out of the
    verdict; ``None``: nobody knows whether it could - no data, which is not a pass.
    """
    if measurable is False:
        return MetricResult(spec.key, spec.label, reference, measured, None, "n/a", None, real)
    if measurable is None:
        return MetricResult(spec.key, spec.label, reference, measured, None, "no_data", None, real)
    interval = None
    if spec.proportion and measured.value is not None and measured.n > 0:
        hits = min(measured.n, max(0, round(measured.value * measured.n)))
        interval = wilson_interval(hits, measured.n)
    deviation = deviation_of(reference.value, measured.value)
    if reference.value is None or measured.value is None:
        status: Status = "no_data"
    elif deviation is not None and abs(deviation) <= TOLERANCE + EPSILON:
        status = "pass"
    else:
        status = "fail"
    return MetricResult(
        spec.key, spec.label, reference, measured, deviation, status, interval, real
    )


@dataclass(frozen=True)
class StyleReport:
    """The six metrics of a source against the profile of the matching scope."""

    source: str  # live | eval_items
    scope: str  # live | pre_holdout
    results: list[MetricResult]
    backend: str | None = None
    run_id: str | None = None
    days: int | None = None
    messages: int = 0  # the bot's messages (bubbles) measured
    notes: list[str] = field(default_factory=list)
    quoted: tuple[int, int] | None = None  # (replies the model began with a quote, replies)

    @property
    def quoted_share(self) -> float | None:
        """Outside the verdict (D-851): the share of replies the model wrote ``[引用:…]`` in."""
        if self.quoted is None or self.quoted[1] == 0:
            return None
        return self.quoted[0] / self.quoted[1]

    @property
    def passed(self) -> bool:
        """Every metric that can be measured is within the tolerance (and at least one was)."""
        counted = [r for r in self.results if r.counts]
        return bool(counted) and all(r.status == "pass" for r in counted)

    def to_json(self) -> dict[str, Any]:
        """The report as numbers and closed names, to be kept with a ``style`` evaluation run."""
        return {
            "source": self.source,
            "scope": self.scope,
            "backend": self.backend,
            "blind_run": self.run_id,
            "days": self.days,
            "messages": self.messages,
            "passed": self.passed,
            "tolerance": TOLERANCE,
            "metrics": [result.to_json() for result in self.results],
            "notes": list(self.notes),
            "quoted": None
            if self.quoted is None
            else {"replies": self.quoted[1], "with_quote": self.quoted[0], "counted": False},
        }

    @property
    def worst(self) -> list[MetricResult]:
        """The measurable metrics, the largest deviation first (for what to improve)."""
        measurable = [r for r in self.results if r.counts]
        return sorted(
            measurable,
            key=lambda r: abs(r.deviation) if r.deviation is not None else float("inf"),
            reverse=True,
        )


def compare(
    reference: ProfileMetrics,
    measured: ProfileMetrics,
    *,
    channel: Abilities | None = None,
    real: ProfileMetrics | None = None,
) -> list[MetricResult]:
    """Judge all six metrics of ``measured`` against ``reference``.

    ``channel`` is what the channel the replies went through could do (``None``: no channel);
    a metric it cannot produce is not measurable (D-851), whatever the source or the backend.
    """
    ours, hers = read_metrics(measured), read_metrics(reference)
    reals = read_metrics(real) if real is not None else {}
    return [
        judge_metric(
            spec,
            hers[spec.key],
            ours[spec.key],
            measurable=spec.measurable(channel),
            real=reals.get(spec.key),
        )
        for spec in STYLE_METRICS
    ]


def channel_notes(channel: Abilities | None, where: str) -> list[str]:
    """What the report says about the metrics the channel leaves out or does not tell about."""
    notes: list[str] = []
    for spec in STYLE_METRICS:
        able = spec.measurable(channel)
        if able is False:
            notes.append(
                f"{where} cannot do {spec.needs}: the {spec.key} is not measurable here and "
                "is left out of the verdict"
            )
        elif able is None:
            notes.append(
                f"the records do not say whether {where} could do {spec.needs} (generated before "
                f"round 18): the {spec.key} has no data; generate the replies again"
            )
    return notes


# -------------------------------------------------------------------- sources


class StyleError(RuntimeError):
    """The report cannot be made; the message says what is missing."""


def style_of_live(services: Services, days: int) -> StyleReport:
    """The last ``days`` days of the bot against the live profile (``--source live``)."""
    if days < 1:
        raise StyleError("--days must be at least 1")
    profile = load_profile(services, "live")
    if profile is None:
        raise StyleError("there is no live profile; run `twin profile rebuild` first")
    since = services.clock.now_utc() - timedelta(days=days)
    settings = services.settings.profile
    recs = list(live_recs(services, since))
    measured = measure(recs, settings.burst_gap_s, settings.segment_gap_min)
    bubbles = sum(1 for rec in recs if rec.her)
    channel = abilities_of(configured_capabilities(services))
    notes = channel_notes(channel, f"the {services.settings.channel.kind} channel")
    if bubbles == 0:
        notes.append(f"the bot sent nothing in the last {days} day(s)")
    return StyleReport(
        "live",
        "live",
        compare(profile.metrics, measured, channel=channel),
        days=days,
        messages=bubbles,
        notes=notes,
    )


def candidates_of(items: Sequence[ItemView], side: Literal["bot", "real"]) -> list[Candidate]:
    """The candidates of the items that have a reply to measure (one real reply per context)."""
    seen: set[str] = set()
    found: list[Candidate] = []
    for item in items:
        if side == "real":
            if item.sample_key in seen or "real" not in item.payload:
                continue
            seen.add(item.sample_key)
            found.append(Candidate.from_json(item.payload["real"]))
        elif "bot" in item.payload:
            found.append(Candidate.from_json(item.payload["bot"]))
    return found


def recorded_abilities(items: Sequence[ItemView]) -> dict[str, bool | None]:
    """What the channel of the generation could do, from the pairs that have a reply (D-851).

    An ability is there only when every pair says so; one pair without it is enough to leave the
    metric out; a pair that does not say (generated before round 18) makes it unknown.
    """
    generated = [item for item in items if "bot" in item.payload]
    names = {spec.needs for spec in STYLE_METRICS if spec.needs is not None}
    names |= {str(name) for item in generated for name in item.payload.get("channel") or {}}
    found: dict[str, bool | None] = {}
    for name in sorted(names):
        said = [(item.payload.get("channel") or {}).get(name) for item in generated]
        if any(value is False for value in said):
            found[name] = False
        elif said and all(value is True for value in said):
            found[name] = True
        else:
            found[name] = None
    return found


def quoted_replies(items: Sequence[ItemView]) -> tuple[int, int] | None:
    """``(replies with a [引用:…] line before the post-processing, replies)`` - outside the verdict.

    Only the pairs that keep the count (generated from round 18 on) are counted; ``None`` when
    none does.
    """
    counts = [
        int(item.payload["draft"]["quote_lines"])
        for item in items
        if "bot" in item.payload and "quote_lines" in (item.payload.get("draft") or {})
    ]
    if not counts:
        return None
    return sum(1 for count in counts if count > 0), len(counts)


def style_of_items(services: Services, store: EvalStore, run: RunView, backend: str) -> StyleReport:
    """The replies generated for a blind run's contexts against the pre-holdout profile."""
    return style_of_runs(services, store, [run], backend)


def style_of_runs(
    services: Services, store: EvalStore, runs: Sequence[RunView], backend: str
) -> StyleReport:
    """The same for the contexts of several blind runs together (the evaluation of one model
    may be done in more than one run, with contexts no run shares: ``twin model evaluate``)."""
    for run in runs:
        if run.kind != "blind":
            raise StyleError(f"run {run.id} is a {run.kind} run; style metrics need a blind run")
    profile = load_profile(services, "pre_holdout")
    if profile is None:
        raise StyleError("there is no pre-holdout profile; run `twin profile rebuild` first")
    items = [
        item
        for run in runs
        for item in store.items(run.id, backend=backend, status=["generated", "judged", "skipped"])
    ]
    if not items:
        names = ", ".join(run.id for run in runs)
        raise StyleError(f"run {names} has no generated replies of the {backend} backend")
    bots = candidates_of(items, "bot")
    channel = recorded_abilities(items)
    notes = [f"{len(bots)} replies of the {backend} backend; her real replies to the same contexts"]
    if len(runs) > 1:
        notes.append(f"the contexts of {len(runs)} evaluation runs together")
    notes += channel_notes(channel, "the channel of the evaluation sandbox")
    return style_of_candidates(
        services,
        profile.metrics,
        bots,
        real=candidates_of(items, "real"),
        backend=backend,
        run_id=runs[0].id,
        notes=notes,
        channel=channel,
        quoted=quoted_replies(items),
    )


def style_of_candidates(
    services: Services,
    reference: ProfileMetrics,
    bots: Sequence[Candidate],
    *,
    real: Sequence[Candidate] | None = None,
    backend: str | None = None,
    run_id: str | None = None,
    notes: Sequence[str] = (),
    channel: Abilities | None = None,
    quoted: tuple[int, int] | None = None,
) -> StyleReport:
    """The six metrics of ``bots`` (replies to hold-out contexts) against the pre-holdout profile.

    The one place that turns replies into a ``pre_holdout`` report: the blind runs
    (:func:`style_of_runs`) and the comparison of two adapter stacks on the same test contexts
    (``twin.training.stack_choice``, D-655) both come through here, so a stack is judged by the
    very measuring and the very +-30 % of the M5 gate.  ``channel`` is what the channel the
    replies went through could do (``None``: they went through none, as the stacks' raw output).
    """
    settings = services.settings.profile
    measured = measure(candidate_recs(bots), settings.burst_gap_s, settings.segment_gap_min)
    measured_real = (
        measure(candidate_recs(real), settings.burst_gap_s, settings.segment_gap_min)
        if real is not None
        else None
    )
    return StyleReport(
        "eval_items",
        "pre_holdout",
        compare(reference, measured, channel=channel, real=measured_real),
        backend=backend,
        run_id=run_id,
        messages=sum(len(c.lines) for c in bots),
        notes=list(notes),
        quoted=quoted,
    )
