"""R-TRN-009, R18-E6 (D-862..D-864): which adapter stack is kept - SFT, or SFT with DPO.

The user's rule (round 18, "SFT/DPO stack choice"), replacing the order of round 17 (D-655):

1. **main criterion** - on the preference pairs held out of the DPO training, the implicit-reward
   accuracy of DPO over SFT (``r(y) = beta * (log pi_dpo(y|x) - log pi_sft(y|x))``, correct when
   ``r(chosen) > r(rejected)``) has a bootstrap 90 % interval whose lower bound is above 50 %; when
   the profile fell back to the base as DPO's reference, the same comparison of the two models'
   length-normalised log probabilities;
2. **guard** - a paired bootstrap over the hold-out test contexts: the 90 % interval of
   "mean style deviation of DPO minus that of SFT" must not lie wholly above 0;
3. DPO is kept only when both hold; the validation loss is shown, not used.

The tests pin the statistics, the decision, a simulation of two stacks from one distribution
(DPO kept rarely, not half of the time as with the old rule), and the whole judge on generated
replies with a profile whose values are known.
"""

from __future__ import annotations

import ast
import json
import math
import random
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.unit.test_eval_style import BURST_GAP, SEGMENT_GAP, known_replies, profile_of
from twin.eval.render import Candidate
from twin.eval.style_metrics import TOLERANCE, candidate_recs, measure, read_metrics
from twin.services import Services
from twin.training import stack_choice
from twin.training.stack_choice import (
    CHANCE,
    LEVEL,
    RESAMPLES,
    SEED,
    STAT_IMPLICIT,
    STAT_NORMALISED,
    Interval,
    MetricLine,
    PairScore,
    PreferenceCheck,
    StackChoiceError,
    StackFiles,
    StackScore,
    StyleGuard,
    bootstrap_interval,
    candidate_of_generation,
    choose_stack,
    judge_stacks,
    pair_correct,
    preference_check,
    read_generations,
    read_pair_scores,
    style_gap,
    style_guard,
)

KEYS = ("text_length", "comma_rate", "burst_size", "sticker_share", "emoji_code_rate", "quote_rate")


def score(
    stack: str,
    deviation: float = 0.1,
    *,
    undefined: int = 0,
    val_loss: float | None = 1.2,
) -> StackScore:
    """Six metrics: the first ``undefined`` without a deviation, the others at ``deviation``."""
    lines = []
    for index, key in enumerate(KEYS):
        if index < undefined:
            lines.append(MetricLine(key, "fail", None, 1.0, 0.0))
        else:
            status = "pass" if abs(deviation) <= TOLERANCE else "fail"
            lines.append(MetricLine(key, status, deviation, 1.0 + deviation, 1.0))
    return StackScore(stack, tuple(lines), val_loss, 100)


def scores(prefix: str, values: Sequence[tuple[float, float]], tokens: int = 10) -> list[PairScore]:
    return [
        PairScore(f"{prefix}{index}", chosen, rejected, tokens, tokens)
        for index, (chosen, rejected) in enumerate(values)
    ]


def preference_of(correct: int, total: int) -> tuple[list[PairScore], list[PairScore]]:
    """SFT and DPO scores of ``total`` pairs on which DPO is right exactly ``correct`` times."""
    sft = scores("p", [(-20.0, -20.0)] * total)
    dpo = scores(
        "p", [(-19.0, -20.0) if index < correct else (-21.0, -20.0) for index in range(total)]
    )
    return sft, dpo


def guard(low: float, high: float, gap: float = 0.0) -> StyleGuard:
    return StyleGuard(400, gap, Interval(low, high), 0.1, 0.1 + gap, 0, 0)


# ----------------------------------------------------------------- the main criterion


def test_a_pair_is_right_when_the_implicit_reward_of_the_chosen_reply_is_higher() -> None:
    """r(y) = beta * (log pi_dpo(y|x) - log pi_sft(y|x)); beta > 0 does not change the sign."""
    sft = PairScore("p", -30.0, -25.0, 10, 8)
    assert pair_correct(sft, PairScore("p", -28.0, -25.0, 10, 8), normalised=False)  # +2 vs 0
    assert not pair_correct(sft, PairScore("p", -30.0, -24.0, 10, 8), normalised=False)
    assert not pair_correct(sft, sft, normalised=False)  # a tie is not right


def test_with_the_base_as_reference_the_comparison_is_length_normalised() -> None:
    """D-863: per pair, (log p(chosen) / its tokens - log p(rejected) / its tokens) of DPO
    against the same quantity of SFT; the raw sums can say the opposite."""
    sft = PairScore("p", -40.0, -10.0, 40, 5)  # per token: -1.0 and -2.0 -> margin +1.0
    dpo = PairScore("p", -36.0, -9.0, 40, 5)  # raw margin -27 > -30, per token -0.9 + 1.8 = +0.9
    assert pair_correct(sft, dpo, normalised=False)
    assert not pair_correct(sft, dpo, normalised=True)


@given(
    values=st.lists(
        st.tuples(*(st.floats(-200.0, -0.01) for _ in range(4)), st.integers(1, 60)),
        min_size=1,
        max_size=5,
    ),
)
def test_a_pair_is_right_exactly_when_dpo_rewards_the_chosen_reply_more(
    values: list[tuple[float, float, float, float, int]],
) -> None:
    """``r(y) / beta = log pi_dpo(y|x) - log pi_sft(y|x)`` (beta > 0 keeps the order)."""
    for sft_c, sft_r, dpo_c, dpo_r, tokens in values:
        sft = PairScore("p", sft_c, sft_r, tokens, tokens + 1)
        dpo = PairScore("p", dpo_c, dpo_r, tokens, tokens + 1)
        assert pair_correct(sft, dpo, normalised=False) is ((dpo_c - sft_c) > (dpo_r - sft_r))
        dpo_margin = dpo_c / tokens - dpo_r / (tokens + 1)
        sft_margin = sft_c / tokens - sft_r / (tokens + 1)
        assert pair_correct(sft, dpo, normalised=True) is (dpo_margin > sft_margin)


def test_dpo_right_on_70_percent_of_40_pairs_passes_the_main_criterion() -> None:
    check = preference_check(*preference_of(28, 40), reference_is_base=False)
    assert check.statistic == STAT_IMPLICIT and check.pairs == 40
    assert check.accuracy == pytest.approx(0.70)
    assert check.interval is not None and check.interval.low > CHANCE and check.passed
    assert check.interval.low <= 0.70 <= check.interval.high


@pytest.mark.parametrize(("correct", "total"), [(20, 40), (23, 40), (5, 10), (0, 40), (25, 40)])
def test_an_accuracy_whose_interval_reaches_down_to_half_does_not_pass(
    correct: int, total: int
) -> None:
    check = preference_check(*preference_of(correct, total), reference_is_base=False)
    assert check.interval is not None and check.interval.low <= CHANCE and not check.passed


def test_the_fallback_reference_uses_the_length_normalised_statistic() -> None:
    sft = [PairScore(f"p{index}", -40.0, -10.0, 40, 5) for index in range(20)]
    dpo = [PairScore(item.id, -36.0, -9.0, 40, 5) for item in sft]  # raw better, per token worse
    raw = preference_check(sft, dpo, reference_is_base=False)
    normalised = preference_check(sft, dpo, reference_is_base=True)
    assert raw.statistic == STAT_IMPLICIT and raw.accuracy == 1.0 and raw.passed
    assert normalised.statistic == STAT_NORMALISED and normalised.accuracy == 0.0
    assert not normalised.passed and normalised.reference == "base"


def test_no_held_out_pairs_cannot_show_dpo_better() -> None:
    check = preference_check([], [], reference_is_base=False)
    assert check.pairs == 0 and check.accuracy is None and check.interval is None
    assert not check.passed


def test_the_pairs_of_the_two_stacks_must_be_the_same_ones() -> None:
    sft, dpo = preference_of(3, 5)
    with pytest.raises(StackChoiceError, match="same held-out pairs"):
        preference_check(sft, dpo[:4], reference_is_base=False)
    with pytest.raises(StackChoiceError, match="same held-out pairs"):
        preference_check(sft, list(reversed(dpo)), reference_is_base=False)
    other = [PairScore(item.id, item.chosen, item.rejected, 11, 10) for item in dpo]
    with pytest.raises(StackChoiceError, match="tokens"):
        preference_check(sft, other, reference_is_base=False)


# ------------------------------------------------------------------- the bootstrap


def test_the_bootstrap_is_reproducible_and_takes_the_central_90_percent() -> None:
    values = [float(index % 7) for index in range(50)]

    def mean_of(indices: Sequence[int]) -> float:
        return sum(values[i] for i in indices) / len(indices)

    first = bootstrap_interval(len(values), mean_of)
    assert first == bootstrap_interval(len(values), mean_of)  # a fixed seed: the same verdict
    assert first.low < sum(values) / len(values) < first.high
    seen: list[float] = []

    def recorded(indices: Sequence[int]) -> float:
        seen.append(mean_of(indices))
        return seen[-1]

    again = bootstrap_interval(len(values), recorded)
    assert len(seen) == RESAMPLES
    ordered = sorted(seen)
    tail = round(RESAMPLES * (1 - LEVEL) / 2)  # 50 of 1000 below the interval, 50 above
    assert again == Interval(ordered[tail], ordered[RESAMPLES - 1 - tail])
    assert (LEVEL, CHANCE, RESAMPLES, SEED) == (0.90, 0.5, 1000, 20261023)


def test_a_constant_statistic_has_a_point_interval() -> None:
    assert bootstrap_interval(3, lambda indices: 1.0) == Interval(1.0, 1.0)
    with pytest.raises(StackChoiceError, match="nothing to resample"):
        bootstrap_interval(0, lambda indices: 1.0)


# --------------------------------------------------------------------- the style guard


def test_the_style_gap_is_the_mean_deviation_of_dpo_minus_that_of_sft() -> None:
    assert style_gap(score("sft", 0.10), score("dpo", 0.25)) == pytest.approx(0.15)
    assert style_gap(score("sft", -0.20), score("dpo", 0.05)) == pytest.approx(-0.15)
    # a deviation that does not exist (her 0, the stack's not) is worse than any that does
    assert style_gap(score("sft", 0.9), score("dpo", 0.0, undefined=1)) == math.inf
    assert style_gap(score("sft", 0.0, undefined=2), score("dpo", 0.9)) == -math.inf
    # as many undefined on both sides: the metrics both can measure are compared
    assert style_gap(score("sft", 0.1, undefined=1), score("dpo", 0.2, undefined=1)) == (
        pytest.approx(0.1)
    )
    na = StackScore("sft", tuple(MetricLine(k, "n/a", None, None, None) for k in KEYS), None, 1)
    assert style_gap(na, na) == 0.0  # nothing to compare: neither is worse


def test_the_guard_fails_only_when_the_whole_interval_is_above_zero() -> None:
    assert guard(-0.01, 0.20).passed and guard(-0.3, -0.1).passed and guard(0.0, 0.0).passed
    assert not guard(0.001, 0.3).passed
    assert not guard(math.inf, math.inf).passed


def synthetic_measure(sft: Sequence[float], dpo: Sequence[float]) -> stack_choice.ScoreResample:
    """Per-context contributions to a stack's mean deviation (cheap stand-in for the profile)."""

    def measure_both(indices: Sequence[int]) -> tuple[StackScore, StackScore]:
        mean_sft = sum(sft[i] for i in indices) / len(indices)
        mean_dpo = sum(dpo[i] for i in indices) / len(indices)
        return score("sft", mean_sft), score("dpo", mean_dpo)

    return measure_both


def test_the_guard_is_a_paired_bootstrap_over_the_contexts() -> None:
    rng = random.Random(3)
    sft = [abs(rng.gauss(0.10, 0.08)) for _ in range(120)]
    worse = [value + 0.05 for value in sft]  # DPO worse on every context: clearly significant
    clearly = style_guard(len(sft), synthetic_measure(sft, worse))
    assert clearly.contexts == 120 and not clearly.passed
    assert clearly.gap == pytest.approx(0.05) and clearly.interval.low > 0
    same = style_guard(len(sft), synthetic_measure(sft, list(sft)))
    assert same.passed and same.interval == Interval(0.0, 0.0)


# ------------------------------------------------------------------------ the decision


def test_dpo_is_kept_only_when_it_is_better_and_not_worse_in_style() -> None:
    better = preference_check(*preference_of(28, 40), reference_is_base=False)
    unsure = preference_check(*preference_of(22, 40), reference_is_base=False)
    fine, worse = guard(-0.02, 0.01), guard(0.01, 0.05, gap=0.03)
    kept = choose_stack(score("sft"), score("dpo"), better, fine)
    assert kept.keep == "dpo" and "better on the held-out preference pairs" in kept.reason
    style = choose_stack(score("sft"), score("dpo"), better, worse)
    assert style.keep == "sft" and "significantly worse" in style.reason
    not_shown = choose_stack(score("sft"), score("dpo"), unsure, fine)
    assert not_shown.keep == "sft" and "do not show dpo better" in not_shown.reason
    nothing = choose_stack(
        score("sft"), score("dpo"), preference_check([], [], reference_is_base=False), fine
    )
    assert nothing.keep == "sft" and "no held-out preference pairs" in nothing.reason


def test_the_validation_loss_is_shown_and_never_decides() -> None:
    better = preference_check(*preference_of(28, 40), reference_is_base=False)
    unsure = preference_check(*preference_of(20, 40), reference_is_base=False)
    fine = guard(-0.02, 0.01)
    low, high = score("sft", val_loss=2.0), score("dpo", val_loss=0.5)
    assert choose_stack(low, high, unsure, fine).keep == "sft"  # a far lower loss does not help
    assert choose_stack(score("sft", val_loss=0.5), score("dpo", val_loss=2.0), better, fine).keep
    verdict = choose_stack(low, high, unsure, fine)
    assert verdict.to_json()["validation_loss"] == {"sft": 2.0, "dpo": 0.5, "decides": False}
    assert "validation loss 2.0000 (shown only)" in "\n".join(verdict.lines())


def test_two_stacks_from_one_distribution_keep_dpo_rarely() -> None:
    """R18-E6: the null case of the audit (sim_stack_choice.py kept DPO 101 times in 200).

    DPO changed nothing: its log probabilities are SFT's plus noise of the same size on the chosen
    and the rejected reply, and its style deviations come from the same distribution.  The old
    rule kept DPO about half of the time; this one keeps it in at most a tenth of the runs (the
    one-sided error of a 90 % interval is about 5 %)."""
    rng = random.Random(11)
    trials, kept = 200, 0
    for _ in range(trials):
        sft = [(rng.gauss(-30.0, 6.0), rng.gauss(-30.0, 6.0)) for _ in range(40)]
        dpo = [(c + rng.gauss(0.0, 1.0), r + rng.gauss(0.0, 1.0)) for c, r in sft]
        preference = preference_check(scores("p", sft), scores("p", dpo), reference_is_base=False)
        sft_style = [abs(rng.gauss(0.12, 0.08)) for _ in range(40)]
        dpo_style = [abs(rng.gauss(0.12, 0.08)) for _ in range(40)]
        style = style_guard(40, synthetic_measure(sft_style, dpo_style))
        kept += choose_stack(score("sft"), score("dpo"), preference, style).keep == "dpo"
    assert kept <= trials // 10


def test_the_verdict_holds_numbers_and_words_only() -> None:
    better = preference_check(*preference_of(28, 40), reference_is_base=False)
    verdict = choose_stack(score("sft"), score("dpo", 0.12), better, guard(-0.02, 0.01))
    data = verdict.to_json()
    assert data["schema"] == 2 and data["keep"] == "dpo"
    assert data["tolerance"] == TOLERANCE == 0.30  # the gate's own number, reused
    assert data["preference"]["statistic"] == STAT_IMPLICIT
    assert data["preference"]["interval"] == {
        "low": better.interval.low if better.interval else None,
        "high": better.interval.high if better.interval else None,
        "level": 0.9,
    }
    assert data["preference"]["passed"] is True and data["style"]["passed"] is True
    assert [m["key"] for m in data["stacks"]["sft"]["metrics"]] == list(KEYS)
    assert json.loads(json.dumps(data)) == data
    unbounded = choose_stack(score("sft"), score("dpo"), better, guard(math.inf, math.inf))
    assert unbounded.to_json()["style"]["interval"]["low"] == "inf"  # still plain JSON
    json.dumps(unbounded.to_json(), allow_nan=False)
    shown = "\n".join(verdict.lines())
    assert shown.startswith("stack kept: dpo (")
    assert "implicit-reward accuracy of dpo over sft 70.0%" in shown
    assert "90% interval" in shown and "style (paired bootstrap over 400 test contexts)" in shown


def test_the_module_adds_no_threshold_beyond_the_users() -> None:
    """R18-E6: the only numbers of the rule are the user's - a 90 % interval and 50 % - and the
    style tolerance is the gate's own, imported, never redefined."""
    source = Path(stack_choice.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assigned = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    } | {
        node.target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert "TOLERANCE" not in assigned
    floats = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, float)
    }
    assert floats - {0.0} == {0.5, 0.9}  # 0.0 is "no gap", the guard's own zero


def test_a_metric_the_profile_cannot_judge_does_not_count_as_inside_the_tolerance() -> None:
    lines = (
        MetricLine("text_length", "pass", 0.0, 8.0, 8.0),
        MetricLine("quote_rate", "n/a", None, None, None),
        MetricLine("comma_rate", "no_data", None, None, 0.1),
    )
    item = StackScore("sft", lines, None, 3)
    assert (item.passed, item.counted, item.undefined) == (1, 2, 1)
    assert item.mean_abs_deviation == 0.0


# ------------------------------------------------------------------ reading the files


def raw_of(candidate: Candidate) -> str:
    """The generated text the model would write for a candidate (the output convention)."""
    lines = [f"[引用:{candidate.quote}]"] if candidate.quote else []
    lines.extend(
        "[表情包:开心]" if line.kind == "sticker" else line.text for line in candidate.lines
    )
    return "\n".join(lines)


def generations_of(replies: list[Candidate], *, prefix: str = "te") -> str:
    return "".join(
        json.dumps({"id": f"{prefix}{index}", "text": raw_of(reply)}, ensure_ascii=False) + "\n"
        for index, reply in enumerate(replies)
    )


def test_a_generated_reply_becomes_the_candidate_the_style_metrics_measure() -> None:
    replies = known_replies()
    rebuilt = [candidate_of_generation(raw_of(reply)) for reply in replies]
    for original, found in zip(replies, rebuilt, strict=True):
        assert [line.text for line in found.lines if line.kind == "text"] == [
            line.text for line in original.lines if line.kind == "text"
        ]
        assert len(found.stickers) == len(original.stickers) and found.quote == original.quote
    measured = read_metrics(measure(candidate_recs(rebuilt), BURST_GAP, SEGMENT_GAP))
    expected = read_metrics(measure(candidate_recs(replies), BURST_GAP, SEGMENT_GAP))
    assert measured == expected


def test_the_output_conventions_are_read_by_the_engines_parser() -> None:
    reply = candidate_of_generation("[引用：你好]\n她：哈哈 [表情包：大笑] 好的\n[不回]\n\n")
    assert reply.quote == "你好"
    assert [(line.kind, line.text) for line in reply.lines] == [
        ("text", "哈哈"),
        ("sticker", ""),
        ("text", "好的"),
    ]
    assert candidate_of_generation("").empty and candidate_of_generation("[不回]").empty
    same_tag = candidate_of_generation("[表情包:开心]").stickers
    assert same_tag == candidate_of_generation("[表情包：开心]").stickers  # one tag, one sticker
    assert same_tag != candidate_of_generation("[表情包:大笑]").stickers


def test_the_generations_file_is_read_in_order_and_a_broken_line_names_only_its_number() -> None:
    text = '{"id": "a", "text": "x"}\n\n{"id": "b", "text": "y"}\n'
    assert read_generations(text) == [("a", "x"), ("b", "y")]
    secret = "a sentence of hers"
    for broken in (secret, '{"id": "a"}', "[1, 2]"):
        with pytest.raises(StackChoiceError, match="line 1") as caught:
            read_generations(broken)
        assert secret not in str(caught.value)


def pair_file(items: Sequence[PairScore]) -> str:
    return "".join(
        json.dumps(
            {
                "id": item.id,
                "chosen_logp": item.chosen,
                "rejected_logp": item.rejected,
                "chosen_tokens": item.chosen_tokens,
                "rejected_tokens": item.rejected_tokens,
            }
        )
        + "\n"
        for item in items
    )


def test_the_pair_scores_are_ids_and_numbers_and_a_broken_line_names_only_its_number() -> None:
    sft, _ = preference_of(1, 3)
    assert read_pair_scores(pair_file(sft) + "\n") == sft
    secret = "她说的话"
    for broken in (
        secret,
        '{"id": "p0", "chosen_logp": -1.0}',
        '{"id": "p0", "chosen_logp": "x", "rejected_logp": -1, "chosen_tokens": 1, '
        '"rejected_tokens": 1}',
        '{"id": "p0", "chosen_logp": NaN, "rejected_logp": -1, "chosen_tokens": 1, '
        '"rejected_tokens": 1}',
        '{"id": "p0", "chosen_logp": -1, "rejected_logp": -1, "chosen_tokens": 0, '
        '"rejected_tokens": 1}',
    ):
        with pytest.raises(StackChoiceError, match="line 1") as caught:
            read_pair_scores(broken)
        assert secret not in str(caught.value)


# ------------------------------------------------------------------------ the whole judge


@pytest.fixture
def profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pre-holdout profile whose values are exactly those of ``known_replies()``."""
    monkeypatch.setattr(
        stack_choice, "load_profile", lambda services, scope: SimpleNamespace(metrics=profile_of())
    )


def without_stickers(replies: list[Candidate]) -> list[Candidate]:
    return [
        Candidate(tuple(line for line in reply.lines if line.kind != "sticker"), reply.quote)
        for reply in replies
    ]


SFT_REFERENCE = {"mode": "layered", "reference": "sft", "fallback": False}
BASE_REFERENCE = {"mode": "continued", "reference": "base", "fallback": True}


def files(
    sft: str,
    dpo: str,
    *,
    correct: int | None = 28,
    total: int = 40,
    reference: dict[str, object] | None = None,
    sft_loss: float = 1.2,
    dpo_loss: float = 1.3,
) -> StackFiles:
    pairs: dict[str, str] = {}
    if correct is not None:
        sft_pairs, dpo_pairs = preference_of(correct, total)
        pairs = {"sft": pair_file(sft_pairs), "dpo": pair_file(dpo_pairs)}
    return StackFiles(
        {"sft": sft, "dpo": dpo},
        {"sft": {"val_loss": sft_loss, "generated": 20}, "dpo": {"val_loss": dpo_loss}},
        pair_scores=pairs,
        reference=reference if reference is not None else SFT_REFERENCE,
    )


@pytest.mark.usefixtures("profile")
def test_the_judge_keeps_dpo_when_it_prefers_her_wording_and_writes_like_sft(
    services: Services,
) -> None:
    good = generations_of(known_replies())
    verdict = judge_stacks(services, files(good, good))
    assert verdict.keep == "dpo"
    assert verdict.preference.passed and verdict.preference.pairs == 40
    assert verdict.style.passed and verdict.style.contexts == 20
    assert verdict.style.interval == Interval(0.0, 0.0)  # the same replies: no gap at all
    assert (verdict.sft.passed, verdict.dpo.passed) == (6, 6)
    assert verdict.sft.val_loss == 1.2 and verdict.dpo.val_loss == 1.3  # a worse loss, shown
    assert "你好" not in json.dumps(verdict.to_json(), ensure_ascii=False)


@pytest.mark.usefixtures("profile")
def test_the_judge_keeps_sft_when_dpo_writes_significantly_worse(services: Services) -> None:
    """No stickers at all: her sticker share is 1/6, DPO's 0 in every resample of the contexts."""
    good = generations_of(known_replies())
    drifted = generations_of(without_stickers(known_replies()))
    verdict = judge_stacks(services, files(good, drifted))
    assert verdict.preference.passed and not verdict.style.passed
    assert verdict.keep == "sft" and "significantly worse" in verdict.reason
    assert verdict.style.interval.low > 0
    sticker = next(m for m in verdict.dpo.metrics if m.key == "sticker_share")
    assert sticker.status == "fail" and sticker.deviation == pytest.approx(-1.0)


@pytest.mark.usefixtures("profile")
def test_the_judge_keeps_sft_when_the_pairs_do_not_show_dpo_better(services: Services) -> None:
    """R18-E6: the closer style that decided before (rule 3 of D-655) decides nothing now."""
    good = generations_of(known_replies())
    drifted = generations_of(without_stickers(known_replies()))
    unsure = judge_stacks(services, files(drifted, good, correct=21))
    assert unsure.style.passed and unsure.style.gap < 0  # DPO's style is the better one
    assert unsure.keep == "sft" and "do not show dpo better" in unsure.reason


@pytest.mark.usefixtures("profile")
def test_without_scored_pairs_a_closer_style_no_longer_keeps_dpo(services: Services) -> None:
    """R18-E6, the regression of the audit: the old rule kept DPO for a smaller mean deviation
    alone (rule 3 of D-655, compared to six decimals); nothing shows it is better, so SFT stays."""
    good = generations_of(known_replies())
    drifted = generations_of(without_stickers(known_replies()))
    verdict = judge_stacks(services, StackFiles({"sft": drifted, "dpo": good}, {}))
    assert verdict.keep == "sft" and "no held-out preference pairs" in verdict.reason
    assert verdict.preference.pairs == 0 and verdict.style.gap < 0


@pytest.mark.usefixtures("profile")
def test_the_profile_that_fell_back_to_the_base_is_judged_by_normalised_log_probabilities(
    services: Services,
) -> None:
    good = generations_of(known_replies())
    sft = [PairScore(f"p{index}", -40.0, -10.0, 40, 5) for index in range(40)]
    dpo = [PairScore(item.id, -36.0, -9.0, 40, 5) for item in sft]
    pairs = {"sft": pair_file(sft), "dpo": pair_file(dpo)}  # the raw sums prefer DPO every time
    as_base = StackFiles(
        {"sft": good, "dpo": good}, {}, pair_scores=pairs, reference=BASE_REFERENCE
    )
    verdict = judge_stacks(services, as_base)
    assert verdict.preference.statistic == STAT_NORMALISED and verdict.keep == "sft"
    assert "length-normalised" in "\n".join(verdict.lines())


@pytest.mark.usefixtures("profile")
def test_the_judge_needs_the_same_samples_in_the_same_order_and_something_to_judge(
    services: Services,
) -> None:
    good = generations_of(known_replies())
    other_ids = generations_of(known_replies(), prefix="x")
    with pytest.raises(StackChoiceError, match="not evaluated on the same test contexts"):
        judge_stacks(services, files(good, other_ids))
    reordered = "".join(reversed(good.splitlines(keepends=True)))
    with pytest.raises(StackChoiceError, match="same test contexts"):
        judge_stacks(services, files(good, reordered))
    with pytest.raises(StackChoiceError, match="generated no replies"):
        judge_stacks(services, files("", ""))
    with pytest.raises(StackChoiceError, match="dpo stack are missing"):
        judge_stacks(services, StackFiles({"sft": good}, {}))
    secret = "a sentence of hers"
    with pytest.raises(StackChoiceError, match="line 1") as caught:
        judge_stacks(services, files(good, secret))
    assert secret not in str(caught.value)
    one_sided = StackFiles(
        {"sft": good, "dpo": good}, {}, pair_scores={"sft": pair_file(preference_of(1, 2)[0])}
    )
    with pytest.raises(StackChoiceError, match="held-out pairs of the dpo stack"):
        judge_stacks(services, one_sided)
    unknown_reference = files(good, good)
    unknown_reference = StackFiles(
        unknown_reference.generations, {}, pair_scores=unknown_reference.pair_scores
    )
    with pytest.raises(StackChoiceError, match=r"dpo_reference\.json"):
        judge_stacks(services, unknown_reference)


def test_the_judge_needs_the_pre_holdout_profile_to_measure_against(
    services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stack_choice, "load_profile", lambda services, scope: None)
    good = generations_of(known_replies())
    with pytest.raises(StackChoiceError, match="no pre-holdout profile"):
        judge_stacks(services, files(good, good))


@pytest.mark.usefixtures("profile")
def test_a_missing_or_odd_validation_loss_is_not_a_number_to_show(services: Services) -> None:
    good = generations_of(known_replies())
    odd = files(good, good)
    odd = StackFiles(
        odd.generations,
        {"sft": {"val_loss": "high"}, "dpo": {"val_loss": float("nan")}},
        pair_scores=odd.pair_scores,
        reference=odd.reference,
    )
    verdict = judge_stacks(services, odd)
    assert verdict.sft.val_loss is None and verdict.dpo.val_loss is None
    assert verdict.keep == "dpo"  # the loss never decides
    assert verdict.to_json()["validation_loss"] == {"sft": None, "dpo": None, "decides": False}


def test_preference_checks_say_what_they_are() -> None:
    check = PreferenceCheck(STAT_IMPLICIT, 0, None, None, "sft")
    assert not check.passed and check.to_json()["interval"] is None
