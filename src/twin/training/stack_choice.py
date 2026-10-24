"""Which adapter stack is kept after a preference run: SFT alone, or SFT with DPO (D-655).

``eval.sh`` evaluates both stacks on the **same** hold-out samples - the same test contexts (one
generated reply each, the same fixed sampling and seed) and the same validation set - and
``twin train remote eval`` compares them here, with the project's own measures and nothing new:

* the six core **style metrics** of the generated replies against her ``pre_holdout`` profile,
  measured by the profile's own collector and judged within the +-30 % of the M5 gate
  (:func:`twin.eval.style_metrics.style_of_candidates`, the function the blind runs use);
* the **validation loss** ``eval.sh`` measured.

The rule (:func:`choose_stack`; no threshold of the project is touched and no new one is added -
only an order):

1. the stack with **more metrics within the tolerance** is better (a metric without data does not
   count as within it);
2. at equal numbers, the one with **fewer metrics whose deviation is undefined** (her value is 0
   and the stack's is not, or a side has no data);
3. then the one with the **smaller mean absolute deviation** of the defined ones;
4. then the one with the **lower validation loss**;
5. a tie keeps ``sft``: the DPO adapter has to be shown to be better, not merely not worse.

Rule 1 is the gate's own shape (all six within the tolerance), so a stack that is closer to passing
it is the better one; the later rules only break ties.  The verdict holds numbers and closed words
only - never a reply - and is written next to the evaluation (``artifacts/eval/stack_choice.json``);
``state/stack_choice`` on the instance holds the winner, which ``export.sh`` reads.  Passing the
M5 gate (``twin eval gate M5``, the human blind test) stays a separate, later question about the
model that was kept.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from twin.engine.parsing import parse_reply
from twin.eval.render import Candidate, CandidateLine, text_lines
from twin.eval.style_metrics import TOLERANCE, MetricResult, StyleReport, style_of_candidates
from twin.ingest.events import quote_fragment
from twin.profile.api import load_profile

if TYPE_CHECKING:
    from twin.services import Services

STACKS: Final = ("sft", "dpo")
SCHEMA: Final = 1
DIGITS: Final = 6  # deviations and losses are compared to this many decimals


class StackChoiceError(ValueError):
    """The two stacks cannot be compared (the message never contains a reply)."""


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
    """What the comparison knows about one stack."""

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


@dataclass(frozen=True)
class StackVerdict:
    """Which stack is kept and why."""

    keep: str
    reason: str
    sft: StackScore
    dpo: StackScore
    samples: int

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "keep": self.keep,
            "reason": self.reason,
            "tolerance": TOLERANCE,
            "samples": self.samples,
            "stacks": {"sft": self.sft.to_json(), "dpo": self.dpo.to_json()},
        }

    def lines(self) -> list[str]:
        """The verdict for the terminal (numbers only)."""
        shown = []
        for score in (self.sft, self.dpo):
            deviation = score.mean_abs_deviation
            loss = "-" if score.val_loss is None else f"{score.val_loss:.4f}"
            shown.append(
                f"  {score.stack}: {score.passed} of {score.counted} style metrics within "
                f"+-{TOLERANCE:.0%}, mean deviation "
                f"{'-' if deviation is None else f'{deviation:.1%}'}, validation loss {loss}"
            )
        return [f"stack kept: {self.keep} ({self.reason})", *shown]


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


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, DIGITS)


def choose_stack(sft: StackScore, dpo: StackScore) -> StackVerdict:
    """The rule of this module's description."""
    samples = max(sft.messages, dpo.messages)
    keep, reason = _decide(sft, dpo)
    return StackVerdict(keep, reason, sft, dpo, samples)


def _decide(sft: StackScore, dpo: StackScore) -> tuple[str, str]:
    if dpo.passed != sft.passed:
        better = "dpo" if dpo.passed > sft.passed else "sft"
        return better, (
            f"{better} has more style metrics within +-{TOLERANCE:.0%} of hers: "
            f"{dpo.passed} for dpo against {sft.passed} for sft"
        )
    if dpo.undefined != sft.undefined:
        better = "dpo" if dpo.undefined < sft.undefined else "sft"
        return better, (
            f"as many metrics within the tolerance, {better} has fewer without a deviation: "
            f"{dpo.undefined} for dpo against {sft.undefined} for sft"
        )
    mean_sft, mean_dpo = _rounded(sft.mean_abs_deviation), _rounded(dpo.mean_abs_deviation)
    if mean_sft is not None and mean_dpo is not None and mean_sft != mean_dpo:
        better = "dpo" if mean_dpo < mean_sft else "sft"
        return better, (
            f"as many metrics within the tolerance, {better} has the smaller mean deviation: "
            f"{mean_dpo:.1%} for dpo against {mean_sft:.1%} for sft"
        )
    loss_sft, loss_dpo = _rounded(sft.val_loss), _rounded(dpo.val_loss)
    if loss_sft is not None and loss_dpo is not None and loss_sft != loss_dpo:
        better = "dpo" if loss_dpo < loss_sft else "sft"
        return better, (
            f"the same style numbers, {better} has the lower validation loss: "
            f"{loss_dpo:.4f} for dpo against {loss_sft:.4f} for sft"
        )
    return "sft", "a tie on every measure: dpo has to be better to be kept, so sft stays"


# ------------------------------------------------------------------------- the generations


@dataclass(frozen=True)
class StackFiles:
    """What ``eval.sh`` left for the two stacks: ``eval_generations.jsonl`` and the metrics."""

    generations: Mapping[str, str]
    metrics: Mapping[str, Mapping[str, Any]]


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


def judge_stacks(services: Services, files: StackFiles) -> StackVerdict:
    """Compare the two stacks on the hold-out contexts they were evaluated on (see the module)."""
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
    scores = []
    for stack in STACKS:
        report = style_of_candidates(
            services,
            profile.metrics,
            [candidate_of_generation(text) for _, text in read[stack]],
            backend=stack,
            notes=[f"{len(read[stack])} generated replies of the {stack} stack"],
        )
        scores.append(score_of(stack, report, _val_loss(files.metrics.get(stack, {}))))
    return choose_stack(scores[0], scores[1])
