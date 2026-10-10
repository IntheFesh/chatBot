"""Milestone gates M0, M1, M2 and the framework behind ``twin eval gate`` (R-EVAL-010).

The judges are tested on evaluation data built by hand: a pass, a failure, too few samples, the
read-only ``--check`` and a milestone nobody has wired in yet.  The rules are pinned here: a
threshold or a way of counting that changes breaks one of these tests.
"""

from __future__ import annotations

from datetime import UTC, datetime
from fractions import Fraction

import pytest

from twin.channel.probe.summary import ChannelProbeSummary, save_summary
from twin.config.runtime import BACKEND_ACTIVE
from twin.eval.blind import MIN_VALID
from twin.eval.gates import (
    EXIT_NOT_PASSED,
    EXIT_NOT_REACHED,
    EXIT_PASSED,
    GATE_ROUNDS,
    LLM_GATE_CHECKS,
    M1_MAX_GUESS_RATE,
    M2_MIN_ACCURACY,
    M4_MAX_GUESS_RATE,
    MILESTONES,
    Check,
    GateContext,
    GateError,
    GateRegistry,
    GateVerdict,
    check_gate,
    default_registry,
    load_judges,
    not_reached_message,
    run_gate,
)
from twin.eval.store import EvalStore, NewItem
from twin.llm.capabilities import DOCUMENTED
from twin.llm.probe import CheckResult, ProbeReport, save_probe
from twin.services import Services

WHEN = datetime(2026, 9, 1, 12, tzinfo=UTC)


# ------------------------------------------------------------------- test data


def blind_run(
    services: Services,
    *,
    backend: str = "deepseek",
    valid: int,
    correct: int,
    skipped: int = 0,
    status: str = "running",
) -> str:
    """A blind run whose judged pairs are exactly these numbers."""
    store = EvalStore(services.db, services.clock)
    run = store.create_run("blind", mode="holdout", backends=[backend], status=status)
    total = valid + skipped
    store.add_items(
        run.id,
        [
            NewItem(f"k{n}", backend, WHEN, {}, None, "night", "short", n % 2 == 0)
            for n in range(total)
        ],
    )
    for found in store.items(run.id):
        store.save_generated(found.id, {"bot": {"lines": [], "quote": None}}, cost_usd=0.0)
        if found.seq < correct:
            store.judge(found.id, "correct", score=1.0)
        elif found.seq < valid:
            store.judge(found.id, "wrong", score=0.0)
        else:
            store.judge(found.id, None)
    return run.id


def memory_run(
    services: Services,
    *,
    correct: int,
    partial: int = 0,
    wrong: int = 0,
    real: int = 10,
    reviewed: bool = True,
) -> str:
    store = EvalStore(services.db, services.clock)
    run = store.create_run("memory", mode="live", backends=["deepseek"], status="running")
    outcomes = ["correct"] * correct + ["partial"] * partial + ["wrong"] * wrong
    store.add_items(
        run.id,
        [
            NewItem(
                f"f{n}",
                "deepseek",
                WHEN,
                {"fact": f"事实{n}"},
                "real_record" if n < real else "bot_invented",
            )
            for n in range(len(outcomes))
        ],
    )
    for found, outcome in zip(store.items(run.id), outcomes, strict=True):
        store.save_auto(found.id, {}, auto_outcome=outcome, cost_usd=0.0)
        if reviewed:
            store.judge(found.id, outcome, score={"correct": 1.0, "partial": 0.5}.get(outcome, 0.0))
    return run.id


def llm_probe(
    services: Services, *, passed: tuple[str, ...] = LLM_GATE_CHECKS, fatal: str | None = None
) -> None:
    checks = [
        CheckResult(number, name, name, True, name in passed)
        for number, name in enumerate(LLM_GATE_CHECKS, start=1)
    ]
    checks.append(CheckResult(5, "detail_param", "detail", False, False))  # a measurement
    report = ProbeReport(
        "run-1", WHEN.isoformat(), WHEN.isoformat(), "deepseek-flash", "deepseek-flash",
        checks, DOCUMENTED, fatal,
    )  # fmt: skip
    with services.db.transaction(bump_state=False) as session:
        save_probe(session, report, services.clock)


def channel_probe(services: Services, *, verdict: str = "met", complete: bool = True) -> None:
    summary = ChannelProbeSummary(
        run_id="chan-1", status="completed" if complete else "stopped",
        started_at=WHEN.isoformat(), finished_at=WHEN.isoformat(), complete=complete,
        verdict=verdict, reasons=[] if verdict == "met" else ["the window is shorter"],
        n_messages=8, n_capped=False, window_lower_bound_h=23.0, window_upper_bound_h=25.0,
        gif_animated=True, typing_visible=True, quote_supported=False, quota_shared=True,
        quota_basis="x", suggestions={}, steps={}, failures=[],
    )  # fmt: skip
    with services.db.transaction(bump_state=False) as session:
        save_summary(session, summary, services.clock)


# ------------------------------------------------------------------------- M1


def test_m1_passes_with_fifty_valid_judgements_and_a_guess_rate_of_at_most_seventy_percent(
    services: Services,
) -> None:
    run_id = blind_run(services, valid=50, correct=35)  # 70.0 % exactly
    outcome = run_gate(services, "M1")
    assert outcome.exit_code == EXIT_PASSED and outcome.status == "passed"
    assert outcome.verdict is not None and outcome.verdict.runs == (run_id,)
    values = outcome.verdict.values
    assert (values["valid"], values["correct"], values["rate"]) == (50, 35, 0.7)
    assert values["backend"] == "deepseek" and values["ceiling"] == 0.7
    low, high = values["interval"]
    assert low < 0.7 < high
    assert [c.passed for c in outcome.verdict.checks] == [True, True]


def test_m1_fails_above_seventy_percent_and_one_judgement_more_than_the_line_is_enough(
    services: Services,
) -> None:
    blind_run(services, valid=50, correct=36)  # 72 %
    outcome = run_gate(services, "M1")
    assert outcome.exit_code == EXIT_NOT_PASSED and outcome.status == "failed"
    assert outcome.verdict is not None
    assert [c.passed for c in outcome.verdict.checks] == [True, False]
    assert "36/50" in outcome.verdict.checks[1].detail


def test_m1_has_a_sample_size_of_fifty_and_a_skip_does_not_count(services: Services) -> None:
    blind_run(services, valid=49, correct=10, skipped=20)  # a good rate on too few judgements
    outcome = run_gate(services, "M1")
    assert outcome.exit_code == EXIT_NOT_PASSED and outcome.status == "insufficient"
    assert outcome.verdict is not None and outcome.verdict.values["valid"] == 49
    assert [c.passed for c in outcome.verdict.checks] == [False, False]  # the rate counts too
    assert MIN_VALID == 50


def test_m1_judges_the_latest_blind_test_that_has_the_default_backend(services: Services) -> None:
    older = blind_run(services, valid=50, correct=20)
    services.clock.tick(60)  # type: ignore[attr-defined]
    blind_run(services, backend="style", valid=60, correct=60)  # another backend: not the default
    first = run_gate(services, "M1")
    assert first.verdict is not None and first.verdict.runs == (older,)
    assert first.status == "passed"
    services.clock.tick(60)  # type: ignore[attr-defined]
    newer = blind_run(services, valid=50, correct=45)
    second = run_gate(services, "M1")
    assert second.verdict is not None and second.verdict.runs == (newer,)
    assert second.status == "failed"  # the older, better test no longer counts


def test_m1_follows_the_backend_that_is_the_default_now(services: Services) -> None:
    services.runtime.set(BACKEND_ACTIVE, "style", by="test")
    blind_run(services, backend="deepseek", valid=60, correct=10)
    assert run_gate(services, "M1").status == "insufficient"  # no test of the style backend yet
    blind_run(services, backend="style", valid=55, correct=30)
    outcome = run_gate(services, "M1")
    assert outcome.status == "passed" and outcome.verdict is not None
    assert outcome.verdict.values["backend"] == "style"


def test_m1_without_any_blind_test_asks_for_one(services: Services) -> None:
    outcome = run_gate(services, "M1")
    assert outcome.status == "insufficient" and outcome.exit_code == EXIT_NOT_PASSED
    assert outcome.verdict is not None and "twin eval blind" in outcome.verdict.checks[0].detail


def test_the_two_blind_thresholds_are_the_specs_and_cannot_drift() -> None:
    assert Fraction(7, 10) == M1_MAX_GUESS_RATE and Fraction(6, 10) == M4_MAX_GUESS_RATE
    assert Fraction(4, 5) == M2_MIN_ACCURACY


# ------------------------------------------------------------------------- M2


def test_m2_passes_with_twenty_reviewed_questions_of_both_sources_and_eighty_percent(
    services: Services,
) -> None:
    run_id = memory_run(services, correct=14, partial=4, wrong=2)  # 16 of 20
    outcome = run_gate(services, "M2")
    assert outcome.exit_code == EXIT_PASSED and outcome.verdict is not None
    assert outcome.verdict.runs == (run_id,) and outcome.verdict.values["accuracy"] == 0.8
    assert [c.passed for c in outcome.verdict.checks] == [True, True, True]


def test_m2_fails_below_eighty_percent(services: Services) -> None:
    memory_run(services, correct=15, partial=1, wrong=4)  # 15.5 of 20
    outcome = run_gate(services, "M2")
    assert outcome.status == "failed" and outcome.exit_code == EXIT_NOT_PASSED
    assert outcome.verdict is not None and outcome.verdict.values["points"] == 15.5
    assert [c.passed for c in outcome.verdict.checks] == [True, True, False]


def test_m2_with_a_composition_that_is_not_ten_and_ten_is_not_passed_whatever_the_score(
    services: Services,
) -> None:
    memory_run(services, correct=20, real=12)  # 100 %, but twelve real and eight from the bot
    outcome = run_gate(services, "M2")
    assert outcome.status == "insufficient" and outcome.exit_code == EXIT_NOT_PASSED
    assert outcome.verdict is not None and not outcome.verdict.checks[0].passed
    assert "样本不足" in outcome.verdict.checks[0].detail


def test_m2_counts_only_a_reviewed_run(services: Services) -> None:
    memory_run(services, correct=20, reviewed=False)
    outcome = run_gate(services, "M2")
    assert outcome.status == "insufficient"
    assert outcome.verdict is not None and outcome.verdict.checks[1].detail == "已复核 0/20"


def test_m2_after_a_test_that_had_too_few_facts_from_the_bot_is_not_passed(
    services: Services,
) -> None:
    """The plan found fewer than ten facts from the conversation: no samples, no real fill-in."""
    store = EvalStore(services.db, services.clock)
    store.create_run(
        "memory", mode="live", status="done", verdict="insufficient",
        summary={"total": 0, "passed": False},
    )  # fmt: skip
    outcome = run_gate(services, "M2")
    assert outcome.status == "insufficient" and outcome.exit_code == EXIT_NOT_PASSED
    assert outcome.verdict is not None and outcome.verdict.values["total"] == 0


def test_m2_without_a_memory_test_asks_for_one(services: Services) -> None:
    outcome = run_gate(services, "M2")
    assert outcome.status == "insufficient"
    assert outcome.verdict is not None and "twin eval memory" in outcome.verdict.checks[0].detail


# ------------------------------------------------------------------------- M0


def test_m0_passes_when_both_probes_pass_by_their_rules(services: Services) -> None:
    llm_probe(services)
    channel_probe(services)
    outcome = run_gate(services, "M0")
    assert outcome.exit_code == EXIT_PASSED and outcome.status == "passed"
    assert outcome.verdict is not None and all(c.passed for c in outcome.verdict.checks)
    assert len(outcome.verdict.checks) == len(LLM_GATE_CHECKS) + 2


@pytest.mark.parametrize("failing", LLM_GATE_CHECKS)
def test_m0_fails_when_one_of_the_four_deepseek_checks_fails(
    services: Services, failing: str
) -> None:
    llm_probe(services, passed=tuple(c for c in LLM_GATE_CHECKS if c != failing))
    channel_probe(services)
    outcome = run_gate(services, "M0")
    assert outcome.status == "failed" and outcome.exit_code == EXIT_NOT_PASSED
    assert outcome.verdict is not None
    assert next(c.name for c in outcome.verdict.checks if not c.passed) == f"探针检查 {failing}"


def test_m0_fails_after_a_fatal_probe_even_if_the_checks_look_fine(services: Services) -> None:
    llm_probe(services, fatal="the key was refused")
    channel_probe(services)
    outcome = run_gate(services, "M0")
    assert outcome.status == "failed"
    assert outcome.verdict is not None and "the key was refused" in outcome.verdict.summary


@pytest.mark.parametrize(
    ("verdict", "complete"), [("not_met", True), ("undetermined", True), ("met", False)]
)
def test_m0_needs_a_complete_channel_probe_with_the_verdict_met(
    services: Services, verdict: str, complete: bool
) -> None:
    llm_probe(services)
    channel_probe(services, verdict=verdict, complete=complete)
    outcome = run_gate(services, "M0")
    assert outcome.status == "failed" and outcome.exit_code == EXIT_NOT_PASSED


def test_m0_without_probe_results_is_not_passed_and_says_which_probe_to_run(
    services: Services,
) -> None:
    outcome = run_gate(services, "M0")
    assert outcome.status == "insufficient" and outcome.verdict is not None
    details = " ".join(c.detail for c in outcome.verdict.checks)
    assert "twin llm probe" in details and "twin channel probe" in details
    llm_probe(services)  # only half of M0
    half = run_gate(services, "M0")
    assert half.status == "insufficient" and half.exit_code == EXIT_NOT_PASSED


# ------------------------------------------------------- the framework around them


def test_a_milestone_without_a_judge_is_not_reached_and_names_the_round(
    services: Services,
) -> None:
    for milestone, number in GATE_ROUNDS.items():
        outcome = run_gate(services, milestone, registry=GateRegistry())
        assert outcome.exit_code == EXIT_NOT_REACHED and outcome.status == "not_reached"
        assert f"该门槛在第 {number:02d} 轮接入" in outcome.message
        assert outcome.verdict is None and outcome.run is None
        assert (
            check_gate(services, milestone, registry=GateRegistry()).exit_code == EXIT_NOT_REACHED
        )
    assert GATE_ROUNDS == {"M3": 10, "M4": 12, "M5": 14}
    # until the rounds register theirs, the real registry has the same state (M4: round 12,
    # M5: round 14)
    for milestone in ("M3",):
        assert default_registry.get(milestone) is None
        assert run_gate(services, milestone).exit_code == EXIT_NOT_REACHED
    assert default_registry.get("M4") is not None and default_registry.get("M5") is not None
    assert "第 10 轮" in not_reached_message("M3") and "后续轮次" in not_reached_message("M1")
    assert EvalStore(services.db, services.clock).list_runs("gate") == []  # nothing was stored


def test_the_verdict_of_a_gate_is_stored_with_its_evidence(services: Services) -> None:
    run_id = blind_run(services, valid=50, correct=30)
    outcome = run_gate(services, "m1")  # the name is not case sensitive
    store = EvalStore(services.db, services.clock)
    stored = store.get_run(outcome.run.id if outcome.run else "")
    assert stored.kind == "gate" and stored.milestone == "M1" and stored.verdict == "passed"
    assert stored.status == "done" and stored.summary["runs"] == [run_id]
    assert stored.summary["values"]["valid"] == 50 and stored.summary["checks"]
    assert stored.summary["summary"]


def test_check_only_reads_the_latest_stored_verdict(services: Services) -> None:
    before = check_gate(services, "M1")
    assert before.exit_code == EXIT_NOT_PASSED and before.status == "not_run" and before.checked
    assert "twin eval gate M1" in before.message
    blind_run(services, valid=50, correct=45)
    assert run_gate(services, "M1").status == "failed"
    services.clock.tick(60)  # type: ignore[attr-defined]
    blind_run(services, valid=50, correct=20)  # the data now passes, but nobody judged it yet
    read = check_gate(services, "M1")
    assert read.status == "failed" and read.exit_code == EXIT_NOT_PASSED and read.checked
    assert read.verdict is not None and read.verdict.values["correct"] == 45  # the stored one
    assert (
        len(EvalStore(services.db, services.clock).list_runs("gate")) == 1
    )  # --check wrote nothing
    services.clock.tick(60)  # type: ignore[attr-defined]
    assert run_gate(services, "M1").status == "passed"
    assert check_gate(services, "M1").exit_code == EXIT_PASSED  # the newest verdict


def test_an_unknown_milestone_is_refused(services: Services) -> None:
    with pytest.raises(GateError, match="unknown milestone"):
        run_gate(services, "M9")
    with pytest.raises(GateError, match="unknown milestone"):
        check_gate(services, "x")
    with pytest.raises(GateError, match="unknown milestone"):
        GateRegistry().register("M7", lambda ctx: GateVerdict("M7", "passed", (), ""))
    assert MILESTONES == ("M0", "M1", "M2", "M3", "M4", "M5")


def test_a_later_round_registers_its_judge_and_the_gate_then_runs(services: Services) -> None:
    """How round 10, 12 and 14 plug in: a judge per milestone, a module in JUDGE_MODULES."""
    registry = GateRegistry()

    def judge_m3(ctx: GateContext) -> GateVerdict:
        assert ctx.services is services
        return GateVerdict("M3", "passed", (Check("七天", True, "7/7"),), "ok", values={"days": 7})

    registry.register("M3", judge_m3)
    with pytest.raises(GateError, match="already has a judge"):
        registry.register("M3", judge_m3)
    registry.register("M3", judge_m3, replace=True)
    assert registry.milestones() == {"M3"}
    outcome = run_gate(services, "M3", registry=registry)
    assert outcome.exit_code == EXIT_PASSED and outcome.verdict is not None
    stored = check_gate(services, "M3", registry=registry)
    assert stored.exit_code == EXIT_PASSED and stored.verdict is not None
    assert stored.verdict.values == {"days": 7} and stored.verdict.checks[0].detail == "7/7"
    failing = GateRegistry()
    failing.register("M4", lambda ctx: GateVerdict("M4", "failed", (), "no"))
    assert run_gate(services, "M4", registry=failing).exit_code == EXIT_NOT_PASSED


def test_the_default_registry_holds_exactly_the_judges_of_this_round() -> None:
    registry = load_judges()
    assert registry is default_registry
    assert registry.milestones() == {"M0", "M1", "M2", "M4", "M5"}
