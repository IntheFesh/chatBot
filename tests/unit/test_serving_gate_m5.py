"""The judge of milestone M5 and the release gate of a model (R-SRV-005, R-EVAL-010).

The rules are pinned here: at least 50 valid judgements per backend, a guess rate lower than the
DeepSeek backend's with a one-sided p below 0.1, and every style metric within +-30 %.  A change of
a threshold or of the way something is counted breaks one of these tests.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from tests.support.model_eval import evaluation_run, style_report
from twin.eval.blind import MIN_VALID
from twin.eval.gates import (
    EXIT_NOT_PASSED,
    EXIT_PASSED,
    GateContext,
    GateVerdict,
    check_gate,
    run_gate,
)
from twin.eval.stats import two_proportion_test
from twin.eval.store import EvalStore, RunView
from twin.eval.style_metrics import TOLERANCE, StyleError, StyleReport
from twin.services import Services
from twin.serving.gate_m5 import (
    P_THRESHOLD,
    WAYS,
    judge_m5,
    judge_model,
    latest_evaluated_model,
)

MODEL = "run-1-Q5_K_M"


def judge(
    services: Services,
    *,
    off: dict[str, Sequence[str]] | None = None,
    model: str = MODEL,
) -> GateVerdict:
    """M5 for the model, with the style metrics of each way as the test says."""
    deviating = off or {}

    def style_of(
        services_: Services, store: EvalStore, runs: Sequence[RunView], backend: str
    ) -> StyleReport:
        return style_report(backend, off=deviating.get(backend, ()))

    ctx = GateContext(services, EvalStore(services.db, services.clock))
    return judge_model(ctx, model, style_of=style_of)


def way_checks(verdict: GateVerdict, way: str) -> dict[str, bool]:
    return {c.name.split("：", 1)[1]: c.passed for c in verdict.checks if c.name.startswith(way)}


def test_the_thresholds_are_the_ones_of_the_spec() -> None:
    assert MIN_VALID == 50 and P_THRESHOLD == 0.1 and TOLERANCE == 0.30
    assert WAYS == ("style", "hybrid")


def test_a_model_that_was_never_evaluated_asks_for_the_evaluation(services: Services) -> None:
    verdict = judge(services)
    assert verdict.verdict == "insufficient" and not verdict.passed
    assert f"twin model evaluate {MODEL}" in verdict.summary and verdict.runs == ()


def test_too_few_judgements_say_how_many_pairs_are_missing(services: Services) -> None:
    evaluation_run(services, MODEL, {"deepseek": (50, 33), "style": (45, 20), "hybrid": (46, 20)})
    verdict = judge(services)
    assert verdict.verdict == "insufficient"
    assert (
        "还需至少 5 对" in verdict.summary
        and f"twin model evaluate {MODEL} --n 5" in verdict.summary
    )
    style = [c for c in verdict.checks if c.name.startswith("style：") and "各有效判断" in c.name]
    assert style[0].detail == "style 45 对，deepseek 50 对（跳过不计）；还需 5 对"
    assert verdict.values["ways"]["style"]["missing"] == 5


def test_the_deepseek_side_needs_fifty_as_well(services: Services) -> None:
    evaluation_run(services, MODEL, {"deepseek": (49, 33), "style": (60, 20), "hybrid": (60, 20)})
    verdict = judge(services)
    assert verdict.verdict == "insufficient" and verdict.values["ways"]["style"]["missing"] == 1


def test_skipped_pairs_do_not_count(services: Services) -> None:
    evaluation_run(
        services, MODEL, {"deepseek": (45, 30), "style": (45, 20), "hybrid": (45, 20)}, skipped=20
    )
    verdict = judge(services)
    assert verdict.verdict == "insufficient" and verdict.values["baseline"]["judged"] == 45


def test_a_model_that_is_not_better_than_deepseek_fails(services: Services) -> None:
    evaluation_run(services, MODEL, {"deepseek": (50, 30), "style": (50, 30), "hybrid": (50, 31)})
    verdict = judge(services)
    assert verdict.verdict == "failed"
    assert way_checks(verdict, "style")["猜对率低于 deepseek（单侧检验 p < 0.1）"] is False
    assert "不在范围内" not in verdict.summary and "猜对率低于 deepseek" in verdict.summary


def test_a_difference_that_could_be_chance_is_not_enough(services: Services) -> None:
    """27/50 against 33/50 is 12 points lower but only p = 0.11 one-sided: not below 0.1."""
    test = two_proportion_test(27, 50, 33, 50)
    assert test is not None and 0.1 < test.p_less < 0.12
    evaluation_run(services, MODEL, {"deepseek": (50, 33), "style": (50, 27), "hybrid": (50, 27)})
    assert judge(services).verdict == "failed"


def test_just_below_the_p_threshold_passes(services: Services) -> None:
    test = two_proportion_test(26, 50, 33, 50)
    assert test is not None and test.p_less < 0.1
    evaluation_run(services, MODEL, {"deepseek": (50, 33), "style": (50, 26), "hybrid": (50, 40)})
    verdict = judge(services)
    assert verdict.passed and verdict.values["winner"] == "style"


def test_winning_the_blind_test_is_not_enough_when_a_style_metric_is_off(
    services: Services,
) -> None:
    evaluation_run(services, MODEL, {"deepseek": (50, 40), "style": (50, 20), "hybrid": (50, 20)})
    verdict = judge(
        services, off={"style": ("comma_rate",), "hybrid": ("sticker_share", "quote_rate")}
    )
    assert verdict.verdict == "failed"
    style = way_checks(verdict, "style")
    assert style["猜对率低于 deepseek（单侧检验 p < 0.1）"] is True
    assert style["留出集风格指标每项偏差在 ±30% 内"] is False
    detail = next(c.detail for c in verdict.checks if c.name.startswith("hybrid：留出集风格"))
    assert "表情包占比" in detail and "引用率" in detail
    assert verdict.values["winner"] is None


def test_a_model_that_passes_everything_wins_with_the_better_way(services: Services) -> None:
    evaluation_run(services, MODEL, {"deepseek": (50, 40), "style": (50, 22), "hybrid": (50, 18)})
    verdict = judge(services)
    assert verdict.verdict == "passed" and verdict.passed
    assert verdict.values["winner"] == "hybrid"  # the lower guess rate
    assert all(c.passed for c in verdict.checks) and len(verdict.checks) == 6
    assert "hybrid" in verdict.summary and "36.0%" in verdict.summary and "80.0%" in verdict.summary


def test_a_tie_between_the_ways_goes_to_the_cheaper_style_backend(services: Services) -> None:
    evaluation_run(services, MODEL, {"deepseek": (50, 40), "style": (50, 20), "hybrid": (50, 20)})
    assert judge(services).values["winner"] == "style"


def test_one_way_that_passes_is_enough_when_the_other_does_not(services: Services) -> None:
    evaluation_run(services, MODEL, {"deepseek": (50, 40), "style": (50, 35), "hybrid": (50, 20)})
    verdict = judge(services)
    assert verdict.passed and verdict.values["winner"] == "hybrid"
    assert verdict.values["ways"]["style"]["verdict"] == "failed"


def test_one_way_without_enough_samples_does_not_stop_the_other(services: Services) -> None:
    evaluation_run(services, MODEL, {"deepseek": (50, 40), "style": (30, 5), "hybrid": (50, 20)})
    verdict = judge(services)
    assert verdict.passed and verdict.values["winner"] == "hybrid"
    assert verdict.values["ways"]["style"]["verdict"] == "insufficient"


def test_exactly_fifty_valid_judgements_are_enough_and_forty_nine_are_not(
    services: Services,
) -> None:
    evaluation_run(services, MODEL, {"deepseek": (50, 40), "style": (50, 20), "hybrid": (49, 20)})
    verdict = judge(services)
    assert verdict.passed and verdict.values["ways"]["hybrid"]["verdict"] == "insufficient"


def test_several_evaluations_of_the_model_are_pooled(services: Services) -> None:
    """A second ``twin model evaluate`` adds contexts: 30 + 20 pairs are 50 pairs."""
    first = evaluation_run(
        services, MODEL, {"deepseek": (30, 24), "style": (30, 14), "hybrid": (30, 20)}, tag="a"
    )
    assert judge(services).verdict == "insufficient"
    second = evaluation_run(
        services, MODEL, {"deepseek": (20, 16), "style": (20, 8), "hybrid": (20, 14)}, tag="b"
    )
    verdict = judge(services)
    assert verdict.passed and verdict.values["ways"]["style"]["judged"] == 50
    assert verdict.values["ways"]["style"]["correct"] == 22
    assert set(verdict.runs) == {first.id, second.id}


def test_the_evaluation_of_another_model_or_a_cancelled_run_counts_for_nothing(
    services: Services,
) -> None:
    evaluation_run(
        services, "other-model", {"deepseek": (50, 40), "style": (50, 10), "hybrid": (50, 10)}
    )
    evaluation_run(
        services,
        MODEL,
        {"deepseek": (50, 40), "style": (50, 10), "hybrid": (50, 10)},
        status="cancelled",
    )
    assert judge(services).verdict == "insufficient"


def test_a_style_report_that_cannot_be_made_is_missing_evidence_not_a_failure(
    services: Services,
) -> None:
    evaluation_run(services, MODEL, {"deepseek": (50, 40), "style": (50, 20), "hybrid": (50, 20)})

    def broken(*args: object) -> StyleReport:
        raise StyleError("there is no pre-holdout profile; run `twin profile rebuild` first")

    ctx = GateContext(services, EvalStore(services.db, services.clock))
    verdict = judge_model(ctx, MODEL, style_of=broken)
    assert verdict.verdict == "insufficient"
    assert any("pre-holdout profile" in c.detail for c in verdict.checks)


def test_the_gate_command_judges_the_model_evaluated_last(services: Services) -> None:
    store = EvalStore(services.db, services.clock)
    assert latest_evaluated_model(store) is None
    assert run_gate(services, "M5").exit_code == EXIT_NOT_PASSED  # nothing evaluated
    evaluation_run(
        services, "old-model", {"deepseek": (50, 40), "style": (50, 20), "hybrid": (50, 20)}
    )
    evaluation_run(services, MODEL, {"deepseek": (50, 30), "style": (50, 30), "hybrid": (50, 30)})
    assert latest_evaluated_model(store) == MODEL
    verdict = judge_m5(GateContext(services, store))
    assert verdict.values["model_id"] == MODEL


def test_the_verdict_is_stored_and_read_back_by_check(services: Services) -> None:
    outcome = run_gate(services, "M5")
    assert outcome.exit_code == EXIT_NOT_PASSED and outcome.status == "insufficient"
    stored = check_gate(services, "M5")
    assert stored.exit_code == EXIT_NOT_PASSED and stored.run is not None
    assert stored.run.milestone == "M5" and stored.run.kind == "gate"
    assert EXIT_PASSED == 0


def test_a_passing_gate_through_the_framework_exits_with_zero(
    services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation_run(services, MODEL, {"deepseek": (50, 40), "style": (50, 20), "hybrid": (50, 22)})
    monkeypatch.setattr(
        "twin.serving.gate_m5.style_of_runs",
        lambda services_, store, runs, backend: style_report(backend),
    )
    outcome = run_gate(services, "M5")
    assert outcome.exit_code == EXIT_PASSED and outcome.verdict is not None
    assert outcome.verdict.values["winner"] == "style"
    assert check_gate(services, "M5").exit_code == EXIT_PASSED
