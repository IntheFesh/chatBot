"""Which adapter stack is kept after a preference run: SFT alone, or SFT with DPO (D-655, D-862).

``eval.sh`` evaluates both stacks on the **same** hold-out material - the same test contexts (one
generated reply each, the same fixed sampling and seed), the same validation set, and the same
**held-out preference pairs** (the fifth of the pairs ``twin train export-dpo`` kept out of the
DPO training, ``dpo_holdout.jsonl``, D-861), whose chosen and rejected replies
``tools/pair_logprobs.py`` scores under each stack.  ``twin train remote eval`` decides here, by
the rule the user set in round 18 ("SFT/DPO stack choice"; it replaces the order of D-655):

1. **Main criterion - does DPO prefer her wording?**  On every held-out pair the implicit reward
   of DPO relative to SFT, ``r(y) = beta * (log pi_dpo(y|x) - log pi_sft(y|x))``, is computed for
   the chosen and the rejected reply; the pair is *right* when ``r(chosen) > r(rejected)`` (beta
   > 0 drops out of the comparison).  DPO is better when the bootstrap 90 % interval of the share
   of right pairs has its lower bound **above 50 %**.  When the profile's DPO run fell back to the
   base as its reference (the 4-bit profile, D-651), the two models' **length-normalised** log
   probabilities are compared instead: a pair is right when DPO's
   ``log p(chosen) / n(chosen) - log p(rejected) / n(rejected)`` (``n`` the reply's tokens) is
   larger than SFT's; the same interval and the same 50 % (D-863).
2. **Guard - is DPO's style not significantly worse?**  A paired bootstrap over the hold-out test
   contexts: the contexts are resampled with replacement, the same draw for both stacks; each
   stack's replies of the draw are measured by the M5 gate's own measuring
   (:func:`twin.eval.style_metrics.style_of_candidates`, against her ``pre_holdout`` profile);
   the statistic is "mean absolute style deviation of DPO minus that of SFT" (:func:`style_gap`).
   The guard fails when the 90 % interval of that difference lies **wholly above 0** (D-864).
3. DPO is kept only when **both** hold; otherwise SFT.  The validation loss is shown (it is in
   the report) and decides nothing.

The bootstrap is the percentile bootstrap with :data:`RESAMPLES` draws and the fixed seed
:data:`SEED`, so the same files give the same verdict; the 90 % interval is the 5th to the 95th
percentile of the resampled statistic (:func:`bootstrap_interval`).  The verdict holds numbers and
closed words only - never a reply - and is written next to the evaluation
(``artifacts/eval/stack_choice.json``); ``state/stack_choice`` on the instance holds the winner,
which ``export.sh`` reads.  Passing the M5 gate (``twin eval gate M5``, the human blind test)
stays a separate, later question about the model that was kept.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from twin.engine.parsing import parse_reply
from twin.eval.render import Candidate, CandidateLine, text_lines
from twin.eval.style_metrics import TOLERANCE, MetricResult, StyleReport, style_of_candidates
from twin.ingest.events import quote_fragment
from twin.profile.api import load_profile

if TYPE_CHECKING:
    from twin.services import Services

STACKS: Final = ("sft", "dpo")
SCHEMA: Final = 2
LEVEL: Final = 0.90  # the user's 90 % interval (both criteria)
CHANCE: Final = 0.5  # the main criterion: the interval's lower bound must be above 50 %
RESAMPLES: Final = 1000  # draws of the percentile bootstrap
SEED: Final = 20261023  # fixed: the same files give the same verdict
STAT_IMPLICIT: Final = "implicit_reward_accuracy"
STAT_NORMALISED: Final = "length_normalised_preference"


class StackChoiceError(ValueError):
    """The two stacks cannot be compared (the message never contains a reply)."""


# --------------------------------------------------------------------------- the style


@dataclass(frozen=True)
class MetricLine:
    """One style metric of a stack: the numbers and the verdict, no text."""

    key: str
    status: str
    deviation: float | None
    value: float | None
    reference: float | None

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "status": self.status,
            "deviation": self.deviation,
            "value": self.value,
            "reference": self.reference,
        }


@dataclass(frozen=True)
class StackScore:
    """The six style metrics of one stack's replies and its validation loss (shown only)."""

    stack: str
    metrics: tuple[MetricLine, ...]
    val_loss: float | None
    messages: int

    @property
    def counted(self) -> int:
        return sum(1 for line in self.metrics if line.status != "n/a")

    @property
    def passed(self) -> int:
        return sum(1 for line in self.metrics if line.status == "pass")

    @property
    def undefined(self) -> int:
        return sum(1 for line in self.metrics if line.status != "n/a" and line.deviation is None)

    @property
    def mean_abs_deviation(self) -> float | None:
        known = [
            abs(line.deviation)
            for line in self.metrics
            if line.status != "n/a" and line.deviation is not None
        ]
        return sum(known) / len(known) if known else None

    def to_json(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "counted": self.counted,
            "undefined": self.undefined,
            "mean_abs_deviation": self.mean_abs_deviation,
            "val_loss": self.val_loss,
            "messages": self.messages,
            "metrics": [line.to_json() for line in self.metrics],
        }


def score_of(stack: str, report: StyleReport, val_loss: float | None) -> StackScore:
    """The comparison's view of a style report: its six metrics and the validation loss."""
    return StackScore(
        stack,
        tuple(_line(result) for result in report.results),
        val_loss,
        report.messages,
    )


def _line(result: MetricResult) -> MetricLine:
    return MetricLine(
        result.key,
        result.status,
        result.deviation,
        result.measured.value,
        result.reference.value,
    )


def style_gap(sft: StackScore, dpo: StackScore) -> float:
    """Mean absolute style deviation of DPO minus that of SFT (positive: DPO is worse).

    Over the metrics both stacks are measured on (status not ``n/a``).  A deviation that does not
    exist - her value is 0 and the stack's is not, or the stack has no data - is worse than any
    that does: the stack with **more** such metrics is infinitely worse (``+inf`` when it is DPO,
    ``-inf`` when it is SFT); with as many on both sides, the means are taken over the metrics
    whose deviation both stacks have.  Nothing to compare is no difference (``0``).
    """
    lines = {line.key: line for line in sft.metrics if line.status != "n/a"}
    pairs = [
        (lines[line.key], line)
        for line in dpo.metrics
        if line.status != "n/a" and line.key in lines
    ]
    undefined_sft = sum(1 for ours, _ in pairs if ours.deviation is None)
    undefined_dpo = sum(1 for _, theirs in pairs if theirs.deviation is None)
    if undefined_dpo != undefined_sft:
        return math.inf if undefined_dpo > undefined_sft else -math.inf
    both = [
        (abs(ours.deviation), abs(theirs.deviation))
        for ours, theirs in pairs
        if ours.deviation is not None and theirs.deviation is not None
    ]
    if not both:
        return 0.0
    return sum(theirs for _, theirs in both) / len(both) - sum(ours for ours, _ in both) / len(both)


# ------------------------------------------------------------------------ the bootstrap


@dataclass(frozen=True)
class Interval:
    """The central 90 % of the resampled statistic (the ends may be infinite, see style_gap)."""

    low: float
    high: float

    def to_json(self) -> dict[str, Any]:
        return {"low": _number(self.low), "high": _number(self.high), "level": LEVEL}


def _number(value: float) -> float | str:
    """A float for JSON: the infinities of an undefined style deviation as words."""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def bootstrap_interval(
    count: int,
    statistic: Callable[[Sequence[int]], float],
    *,
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> Interval:
    """The percentile bootstrap: ``statistic`` of ``resamples`` draws of ``count`` indices.

    Each draw takes ``count`` indices out of ``range(count)`` with replacement (one
    ``random.Random(seed)`` for all draws); the interval is the ``round(resamples * 5 %)``-th
    smallest and largest value - with 1000 draws, 50 values lie below it and 50 above.
    """
    if count < 1:
        raise StackChoiceError("there is nothing to resample")
    rng = random.Random(seed)  # noqa: S311 - a reproducible resampling, not security
    values = sorted(
        statistic([rng.randrange(count) for _ in range(count)]) for _ in range(resamples)
    )
    tail = round(resamples * (1 - LEVEL) / 2)
    return Interval(values[tail], values[resamples - 1 - tail])


# ------------------------------------------------------------------- the main criterion


@dataclass(frozen=True)
class PairScore:
    """The log probabilities of one held-out pair's replies under one stack (numbers only)."""

    id: str
    chosen: float  # the sum of the log probabilities of the chosen reply's tokens
    rejected: float
    chosen_tokens: int
    rejected_tokens: int


def read_pair_scores(text: str) -> list[PairScore]:
    """The lines ``tools/pair_logprobs.py`` wrote for one stack, in file order."""
    found: list[PairScore] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            item = PairScore(
                str(row["id"]),
                float(row["chosen_logp"]),
                float(row["rejected_logp"]),
                int(row["chosen_tokens"]),
                int(row["rejected_tokens"]),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            raise StackChoiceError(
                f"line {number} of the pair scores is not an id with four numbers"
            ) from None
        if not (math.isfinite(item.chosen) and math.isfinite(item.rejected)):
            raise StackChoiceError(f"line {number} of the pair scores holds a non-finite number")
        if item.chosen_tokens < 1 or item.rejected_tokens < 1:
            raise StackChoiceError(f"line {number} of the pair scores counts no tokens")
        found.append(item)
    return found


def pair_correct(sft: PairScore, dpo: PairScore, *, normalised: bool) -> bool:
    """Does DPO prefer the chosen reply more than SFT does (see the module description)?

    Not normalised: ``r(chosen) > r(rejected)`` with ``r(y) / beta = log pi_dpo(y) - log
    pi_sft(y)``.  Normalised (the base was DPO's reference): DPO's per-token margin
    ``log p(chosen) / n(chosen) - log p(rejected) / n(rejected)`` is larger than SFT's.
    """
    if normalised:
        dpo_margin = dpo.chosen / dpo.chosen_tokens - dpo.rejected / dpo.rejected_tokens
        sft_margin = sft.chosen / sft.chosen_tokens - sft.rejected / sft.rejected_tokens
        return dpo_margin > sft_margin
    return (dpo.chosen - sft.chosen) > (dpo.rejected - sft.rejected)


@dataclass(frozen=True)
class PreferenceCheck:
    """The main criterion: the share of held-out pairs DPO gets right, and its interval."""

    statistic: str
    pairs: int
    accuracy: float | None
    interval: Interval | None
    reference: str  # sft | base: DPO's reference model, which picks the statistic

    @property
    def passed(self) -> bool:
        return self.interval is not None and self.interval.low > CHANCE

    def to_json(self) -> dict[str, Any]:
        return {
            "statistic": self.statistic,
            "reference": self.reference,
            "pairs": self.pairs,
            "accuracy": self.accuracy,
            "interval": self.interval.to_json() if self.interval is not None else None,
            "needs_lower_bound_above": CHANCE,
            "passed": self.passed,
        }

    def line(self) -> str:
        if self.interval is None or self.accuracy is None:
            return "  preference: no held-out preference pairs were scored"
        what = (
            "share of pairs where dpo's length-normalised margin beats sft's"
            if self.statistic == STAT_NORMALISED
            else "implicit-reward accuracy of dpo over sft"
        )
        return (
            f"  preference ({self.pairs} held-out pairs, reference {self.reference}): {what} "
            f"{self.accuracy:.1%}, {LEVEL:.0%} interval {self.interval.low:.1%} to "
            f"{self.interval.high:.1%} (the lower bound must be above {CHANCE:.0%}): "
            f"{'passed' if self.passed else 'not passed'}"
        )


def preference_check(
    sft: Sequence[PairScore], dpo: Sequence[PairScore], *, reference_is_base: bool
) -> PreferenceCheck:
    """The main criterion on the held-out pairs scored under both stacks."""
    statistic = STAT_NORMALISED if reference_is_base else STAT_IMPLICIT
    reference = "base" if reference_is_base else "sft"
    if [item.id for item in sft] != [item.id for item in dpo]:
        raise StackChoiceError(
            "the two stacks were not scored on the same held-out pairs (their ids differ); "
            "run the evaluation again"
        )
    for ours, theirs in zip(sft, dpo, strict=True):
        if (ours.chosen_tokens, ours.rejected_tokens) != (
            theirs.chosen_tokens,
            theirs.rejected_tokens,
        ):
            raise StackChoiceError(
                "a held-out pair has different numbers of tokens under the two stacks; "
                "run the evaluation again"
            )
    if not sft:
        return PreferenceCheck(statistic, 0, None, None, reference)
    right = [
        1 if pair_correct(ours, theirs, normalised=reference_is_base) else 0
        for ours, theirs in zip(sft, dpo, strict=True)
    ]

    def share(indices: Sequence[int]) -> float:
        return sum(right[index] for index in indices) / len(indices)

    accuracy = sum(right) / len(right)
    return PreferenceCheck(
        statistic, len(right), accuracy, bootstrap_interval(len(right), share), reference
    )


# --------------------------------------------------------------------------- the guard


ScoreResample = Callable[[Sequence[int]], tuple[StackScore, StackScore]]
"""The two stacks' scores on a draw of the test contexts (indices into both lists)."""


@dataclass(frozen=True)
class StyleGuard:
    """The guard: the paired bootstrap of :func:`style_gap` over the test contexts."""

    contexts: int
    gap: float  # on all the contexts
    interval: Interval
    sft_mean: float | None
    dpo_mean: float | None
    sft_undefined: int
    dpo_undefined: int

    @property
    def passed(self) -> bool:
        """DPO's style is not significantly worse: the interval does not lie wholly above 0."""
        return not self.interval.low > 0

    def to_json(self) -> dict[str, Any]:
        return {
            "contexts": self.contexts,
            "gap": _number(self.gap),
            "interval": self.interval.to_json(),
            "mean_abs_deviation": {"sft": self.sft_mean, "dpo": self.dpo_mean},
            "undefined": {"sft": self.sft_undefined, "dpo": self.dpo_undefined},
            "fails_when_wholly_above": 0,
            "passed": self.passed,
        }

    def line(self) -> str:
        return (
            f"  style (paired bootstrap over {self.contexts} test contexts): mean deviation of "
            f"dpo minus sft {_points(self.gap)} points, {LEVEL:.0%} interval "
            f"{_points(self.interval.low)} to {_points(self.interval.high)} (must not lie wholly "
            f"above 0): {'passed' if self.passed else 'not passed'}"
        )


def _points(value: float) -> str:
    """A difference of deviations in percentage points (an infinity as a word)."""
    return str(_number(value)) if math.isinf(value) else f"{value * 100:+.1f}"


def style_guard(
    contexts: int, measure: ScoreResample, *, full: tuple[StackScore, StackScore] | None = None
) -> StyleGuard:
    """The guard over ``contexts`` test contexts; ``full`` are the scores on all of them."""
    sft, dpo = full if full is not None else measure(range(contexts))

    def gap(indices: Sequence[int]) -> float:
        return style_gap(*measure(indices))

    return StyleGuard(
        contexts,
        style_gap(sft, dpo),
        bootstrap_interval(contexts, gap),
        sft.mean_abs_deviation,
        dpo.mean_abs_deviation,
        sft.undefined,
        dpo.undefined,
    )


# ------------------------------------------------------------------------ the decision


@dataclass(frozen=True)
class StackVerdict:
    """Which stack is kept and why."""

    keep: str
    reason: str
    sft: StackScore
    dpo: StackScore
    samples: int
    preference: PreferenceCheck
    style: StyleGuard

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "keep": self.keep,
            "reason": self.reason,
            "tolerance": TOLERANCE,
            "samples": self.samples,
            "preference": self.preference.to_json(),
            "style": self.style.to_json(),
            "validation_loss": {
                "sft": self.sft.val_loss,
                "dpo": self.dpo.val_loss,
                "decides": False,
            },
            "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "level": LEVEL},
            "stacks": {"sft": self.sft.to_json(), "dpo": self.dpo.to_json()},
        }

    def lines(self) -> list[str]:
        """The verdict for the terminal and the run record (numbers and words only)."""
        shown = [f"stack kept: {self.keep} ({self.reason})", self.preference.line()]
        shown.append(self.style.line())
        for score in (self.sft, self.dpo):
            deviation = score.mean_abs_deviation
            loss = "-" if score.val_loss is None else f"{score.val_loss:.4f}"
            shown.append(
                f"  {score.stack}: {score.passed} of {score.counted} style metrics within "
                f"+-{TOLERANCE:.0%}, mean deviation "
                f"{'-' if deviation is None else f'{deviation:.1%}'}, validation loss {loss} "
                "(shown only)"
            )
        return shown


def choose_stack(
    sft: StackScore,
    dpo: StackScore,
    preference: PreferenceCheck,
    style: StyleGuard,
) -> StackVerdict:
    """The rule of this module's description: DPO only when both criteria hold."""
    samples = max(sft.messages, dpo.messages)
    keep, reason = _decide(preference, style)
    return StackVerdict(keep, reason, sft, dpo, samples, preference, style)


def _decide(preference: PreferenceCheck, style: StyleGuard) -> tuple[str, str]:
    if preference.interval is None or preference.accuracy is None:
        return "sft", (
            "no held-out preference pairs were scored, so dpo cannot be shown better and sft stays"
        )
    numbers = (
        f"{preference.accuracy:.1%} of {preference.pairs} pairs, {LEVEL:.0%} interval "
        f"{preference.interval.low:.1%} to {preference.interval.high:.1%}"
    )
    if not preference.passed:
        return "sft", (
            f"the held-out preference pairs do not show dpo better ({numbers}; the lower bound is "
            f"not above {CHANCE:.0%})"
        )
    if not style.passed:
        return "sft", (
            f"dpo prefers her wording on the held-out pairs ({numbers}) but its style is "
            "significantly worse (the interval of its mean deviation minus sft's lies wholly "
            "above 0)"
        )
    return "dpo", (
        f"dpo is better on the held-out preference pairs ({numbers}) and its style is not "
        "significantly worse"
    )


# ------------------------------------------------------------------------- the files


@dataclass(frozen=True)
class StackFiles:
    """What ``eval.sh`` and ``dpo.sh`` left for the comparison.

    ``generations``: ``eval_generations.jsonl`` of each stack; ``metrics``: its
    ``eval_metrics.json``; ``pair_scores``: ``pair_logps.jsonl`` of each stack (the held-out pairs,
    absent when the package had none); ``reference``: ``dpo/dpo_reference.json``.
    """

    generations: Mapping[str, str]
    metrics: Mapping[str, Mapping[str, Any]]
    pair_scores: Mapping[str, str] = field(default_factory=dict)
    reference: Mapping[str, Any] | None = None


StackJudge = Callable[[StackFiles], StackVerdict]


def read_generations(text: str) -> list[tuple[str, str]]:
    """``(sample id, generated reply)`` of an ``eval_generations.jsonl``, in file order."""
    rows: list[tuple[str, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            rows.append((str(row["id"]), str(row["text"])))
        except (json.JSONDecodeError, KeyError, TypeError):
            raise StackChoiceError(
                f"line {number} of the generations is not an id and a text"
            ) from None
    return rows


def candidate_of_generation(raw: str) -> Candidate:
    """A generated reply as the candidate the style metrics measure (the output convention).

    The reply is read by the one parser of the engine (:func:`twin.engine.parsing.parse_reply`): a
    line is a text bubble, a ``[表情包:<标签>]`` line a sticker (identified by its tag), a first
    ``[引用:...]`` line the quote; ``[不回]`` is nothing.
    """
    parsed = parse_reply(raw)
    lines: list[CandidateLine] = []
    for line in parsed.lines:
        if line.kind == "sticker":
            tag = hashlib.sha256(line.text.encode("utf-8")).hexdigest()[:32]
            lines.append(CandidateLine("sticker", sticker_md5=tag))
        elif line.kind == "text":
            lines.extend(CandidateLine("text", text=part) for part in text_lines(line.text))
    return Candidate(tuple(lines), quote_fragment(parsed.quote))


def _val_loss(metrics: Mapping[str, Any]) -> float | None:
    value = metrics.get("val_loss")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _reference_is_base(reference: Mapping[str, Any] | None) -> bool:
    if reference is None:
        raise StackChoiceError(
            "artifacts/dpo/dpo_reference.json is missing: which reference model the DPO run had "
            "decides how the held-out pairs are compared; run `twin train remote dpo` again"
        )
    return reference.get("fallback") is True or reference.get("reference") == "base"


def _preference_of(files: StackFiles) -> PreferenceCheck:
    present = [stack for stack in STACKS if stack in files.pair_scores]
    if not present:
        return preference_check([], [], reference_is_base=False)
    for stack in STACKS:
        if stack not in files.pair_scores:
            raise StackChoiceError(
                f"the held-out pairs of the {stack} stack were not scored; run the evaluation again"
            )
    return preference_check(
        read_pair_scores(files.pair_scores["sft"]),
        read_pair_scores(files.pair_scores["dpo"]),
        reference_is_base=_reference_is_base(files.reference),
    )


def judge_stacks(services: Services, files: StackFiles) -> StackVerdict:
    """Compare the two stacks on the material they were evaluated on (see the module)."""
    profile = load_profile(services, "pre_holdout")
    if profile is None:
        raise StackChoiceError(
            "there is no pre-holdout profile to judge the stacks against; "
            "run `twin profile rebuild` first"
        )
    read = {}
    for stack in STACKS:
        if stack not in files.generations:
            raise StackChoiceError(f"the generations of the {stack} stack are missing")
        read[stack] = read_generations(files.generations[stack])
    if [sample for sample, _ in read["sft"]] != [sample for sample, _ in read["dpo"]]:
        raise StackChoiceError(
            "the two stacks were not evaluated on the same test contexts "
            "(their sample ids differ); run the evaluation again"
        )
    if not read["sft"]:
        raise StackChoiceError("the evaluation generated no replies")
    preference = _preference_of(files)
    candidates = {
        stack: [candidate_of_generation(text) for _, text in read[stack]] for stack in STACKS
    }
    losses = {stack: _val_loss(files.metrics.get(stack, {})) for stack in STACKS}

    def scored(stack: str, indices: Sequence[int]) -> StackScore:
        replies = [candidates[stack][index] for index in indices]
        report = style_of_candidates(services, profile.metrics, replies, backend=stack)
        return score_of(stack, report, losses[stack])

    def both(indices: Sequence[int]) -> tuple[StackScore, StackScore]:
        return scored("sft", indices), scored("dpo", indices)

    full = both(range(len(read["sft"])))
    style = style_guard(len(read["sft"]), both, full=full)
    return choose_stack(full[0], full[1], preference, style)
