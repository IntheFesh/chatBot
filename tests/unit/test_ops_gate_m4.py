"""The judge of milestone M4: seven unattended days, a drill, a new blind test (R-EVAL-010).

Each test builds an observation week in the tables (health snapshots, alerts, evaluation runs,
jobs, imports) with one thing wrong - or nothing wrong - and runs ``twin eval gate M4`` on it.  The
rules are pinned here: a threshold or a way of counting that changes breaks one of these tests.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

from tests.support.clock import ManualClock
from tests.support.stability_world import (
    NOW,
    WEEK_START,
    blind_run,
    finished_import,
    finished_job,
    observed_week,
    steady,
    without,
)
from twin.eval.gates import EXIT_NOT_PASSED, EXIT_PASSED, GateOutcome, check_gate, run_gate
from twin.eval.store import EvalStore
from twin.ops.stability import run_stability
from twin.services import Services


def judge(services: Services, *, days: float = 7) -> GateOutcome:
    """Make the stability report (as ``twin eval stability`` does) and judge M4."""
    run_stability(services, days)
    return run_gate(services, "M4")


def failed_names(outcome: GateOutcome) -> list[str]:
    assert outcome.verdict is not None
    return [check.name for check in outcome.verdict.checks if not check.passed]


def verdict_of(outcome: GateOutcome) -> str:
    assert outcome.verdict is not None
    return outcome.verdict.verdict


# ------------------------------------------------------------------------------- passing


def test_a_week_that_meets_every_criterion_passes(services: Services, clock: ManualClock) -> None:
    observed_week(services, clock)
    outcome = judge(services)
    assert outcome.exit_code == EXIT_PASSED and verdict_of(outcome) == "passed", failed_names(
        outcome
    )
    assert outcome.verdict is not None and outcome.verdict.summary == "M4 通过"
    assert len(outcome.verdict.checks) == 12 and all(c.passed for c in outcome.verdict.checks)
    values = outcome.verdict.values
    assert values["days"] == 7 and values["unavailable_s"] == 0
    assert 300 < values["max_alert_latency_s"] < 600 and values["rate"] == 0.5
    assert outcome.run is not None and outcome.run.milestone == "M4"
    assert check_gate(services, "M4").exit_code == EXIT_PASSED  # the stored verdict
    store = EvalStore(services.db, services.clock)
    assert store.latest_run("stability", status="done") is not None
    assert [r.milestone for r in store.list_runs("gate")] == ["M4"]


# ------------------------------------------------------------------------ missing evidence


def test_without_a_stability_report_there_is_nothing_to_judge(services: Services) -> None:
    outcome = run_gate(services, "M4")
    assert outcome.exit_code == EXIT_NOT_PASSED and verdict_of(outcome) == "insufficient"
    assert "twin eval stability --days 7" in outcome.message


def test_a_window_shorter_than_seven_days_is_not_enough(
    services: Services, clock: ManualClock
) -> None:
    observed_week(services, clock)
    outcome = judge(services, days=3)
    assert verdict_of(outcome) == "insufficient"  # nothing measured is wrong, evidence is short
    assert failed_names(outcome) == ["稳定性报告覆盖连续 7 天，且不超过两天前"]


def test_a_week_that_has_not_been_observed_for_seven_days_is_not_enough(
    services: Services, clock: ManualClock
) -> None:
    started_late = steady(NOW - timedelta(days=4), NOW)  # four days of snapshots only
    observed_week(services, clock, snaps=started_late)
    outcome = judge(services)
    assert verdict_of(outcome) == "insufficient"
    assert failed_names(outcome) == ["稳定性报告覆盖连续 7 天，且不超过两天前"]


def test_an_old_report_is_not_enough(services: Services, clock: ManualClock) -> None:
    observed_week(services, clock)
    run_stability(services, 7)
    clock.set_time(NOW + timedelta(days=3))
    outcome = run_gate(services, "M4")
    assert verdict_of(outcome) == "insufficient"
    assert failed_names(outcome) == ["稳定性报告覆盖连续 7 天，且不超过两天前"]


def test_a_week_without_a_drill_is_not_enough(services: Services, clock: ManualClock) -> None:
    observed_week(services, clock, drill=False)
    outcome = judge(services)
    assert verdict_of(outcome) == "insufficient" and outcome.exit_code == EXIT_NOT_PASSED
    names = failed_names(outcome)
    assert any("断网演练" in name for name in names) and len(names) == 2  # and nothing to check


def test_a_drill_that_is_too_short_is_not_a_drill(services: Services, clock: ManualClock) -> None:
    observed_week(services, clock, drill_minutes=6, alert_after_min=3)
    outcome = judge(services)
    assert verdict_of(outcome) == "insufficient"
    assert any("断网演练" in name for name in failed_names(outcome))


def test_without_a_blind_test_there_is_nothing_to_judge(
    services: Services, clock: ManualClock
) -> None:
    observed_week(services, clock, blind=False)
    outcome = judge(services)
    assert verdict_of(outcome) == "insufficient"
    assert failed_names(outcome) == ["盲测"]


def test_too_few_valid_judgements_are_not_enough(services: Services, clock: ManualClock) -> None:
    observed_week(services, clock, blind_valid=49, blind_correct=20)
    outcome = judge(services)
    assert verdict_of(outcome) == "insufficient"
    assert any("有效判断" in name for name in failed_names(outcome))


def test_no_learning_in_the_week_is_not_enough(services: Services, clock: ManualClock) -> None:
    observed_week(services, clock, learning=False)
    outcome = judge(services)
    assert verdict_of(outcome) == "insufficient"
    assert failed_names(outcome) == ["观察期内学习（每周整理纠正规则）至少成功一次"]


def test_no_import_in_the_week_is_not_enough(services: Services, clock: ManualClock) -> None:
    observed_week(services, clock, imported=False)
    outcome = judge(services)
    assert verdict_of(outcome) == "insufficient"
    assert failed_names(outcome) == ["观察期内增量导入至少成功一次"]


def test_activity_before_the_week_does_not_count(services: Services, clock: ManualClock) -> None:
    observed_week(services, clock, learning=False, imported=False)
    finished_job(services, "learning_rules", WEEK_START - timedelta(hours=2))
    finished_import(services, WEEK_START - timedelta(hours=2))
    assert len(failed_names(judge(services))) == 2
    finished_job(services, "learning_rules", WEEK_START + timedelta(hours=2))
    assert len(failed_names(run_gate(services, "M4"))) == 1


# ------------------------------------------------------------------------ measured misses


def test_an_application_that_was_down_too_long_fails(
    services: Services, clock: ManualClock
) -> None:
    cut = NOW - timedelta(days=5)
    gap = without(steady(WEEK_START, NOW), cut, cut + timedelta(minutes=12))  # 12 min: 11 counted
    observed_week(services, clock, snaps=gap)
    outcome = judge(services)
    assert verdict_of(outcome) == "failed" and outcome.exit_code == EXIT_NOT_PASSED
    assert failed_names(outcome) == ["应用累计不可用时间 ≤ 10.0 分钟"]


def test_a_short_gap_is_within_the_allowance(services: Services, clock: ManualClock) -> None:
    cut = NOW - timedelta(days=5)
    gap = without(steady(WEEK_START, NOW), cut, cut + timedelta(minutes=10))  # 9 minutes counted
    observed_week(services, clock, snaps=gap)
    assert verdict_of(judge(services)) == "passed"


def test_two_gaps_add_up(services: Services, clock: ManualClock) -> None:
    first, second = NOW - timedelta(days=5), NOW - timedelta(days=2)
    snaps = without(steady(WEEK_START, NOW), first, first + timedelta(minutes=7))
    snaps = without(snaps, second, second + timedelta(minutes=7))  # 6 + 6 = 12 minutes
    observed_week(services, clock, snaps=snaps)
    assert failed_names(judge(services)) == ["应用累计不可用时间 ≤ 10.0 分钟"]


def test_a_process_started_by_hand_breaks_the_unattended_week(
    services: Services, clock: ManualClock
) -> None:
    snaps = steady(WEEK_START, NOW)
    snaps = [
        replace(s, launch="manual") if s.at < WEEK_START + timedelta(hours=2) else s for s in snaps
    ]
    observed_week(services, clock, snaps=snaps)
    outcome = judge(services)
    assert verdict_of(outcome) == "failed"
    assert failed_names(outcome) == ["全部由计划任务启动（无人值守）"]


def test_an_application_that_is_not_running_at_the_end_fails(
    services: Services, clock: ManualClock
) -> None:
    stopped = steady(WEEK_START, NOW - timedelta(minutes=30))
    observed_week(services, clock, snaps=stopped)
    outcome = judge(services)
    assert verdict_of(outcome) == "failed"
    assert "报告时应用仍在运行（重启后已自动恢复）" in failed_names(outcome)


def test_an_alert_later_than_ten_minutes_fails(services: Services, clock: ManualClock) -> None:
    observed_week(services, clock, alert_after_min=11, drill_minutes=20)
    outcome = judge(services)
    assert verdict_of(outcome) == "failed"
    assert failed_names(outcome) == ["每次通道异常都在 10 分钟内发出告警，没有漏报"]


def test_an_alert_just_inside_ten_minutes_passes(services: Services, clock: ManualClock) -> None:
    observed_week(services, clock, alert_after_min=8, alert_toast_s=60, drill_minutes=20)
    assert verdict_of(judge(services)) == "passed"  # 8 min + 1 min + the 30 s before the first miss


def test_an_outage_nobody_was_told_about_fails(services: Services, clock: ManualClock) -> None:
    observed_week(services, clock, alert_after_min=-1)  # the drill happens, no alert is raised
    outcome = judge(services)
    assert verdict_of(outcome) == "failed"
    assert failed_names(outcome) == ["每次通道异常都在 10 分钟内发出告警，没有漏报"]


def test_a_guess_rate_above_sixty_percent_fails(services: Services, clock: ManualClock) -> None:
    observed_week(services, clock, blind_valid=60, blind_correct=37)  # 61.7 %
    outcome = judge(services)
    assert verdict_of(outcome) == "failed"
    assert failed_names(outcome) == ["猜对率点估计 ≤ 60%"]


def test_exactly_sixty_percent_passes(services: Services, clock: ManualClock) -> None:
    observed_week(services, clock, blind_valid=60, blind_correct=36)
    assert verdict_of(judge(services)) == "passed"


def test_a_blind_test_from_before_the_week_does_not_count(
    services: Services, clock: ManualClock
) -> None:
    observed_week(services, clock, blind_at=WEEK_START - timedelta(days=2))
    outcome = judge(services)
    assert verdict_of(outcome) == "failed"
    assert failed_names(outcome) == ["这次盲测是在观察期内做的"]


def test_a_blind_test_that_reuses_earlier_contexts_fails(
    services: Services, clock: ManualClock
) -> None:
    blind_run(services, clock, WEEK_START - timedelta(days=30), valid=60, correct=30)
    observed_week(services, clock)  # the new run uses the same sample keys
    outcome = judge(services)
    assert verdict_of(outcome) == "failed"
    assert failed_names(outcome) == ["盲测用的是新的上下文（没有用过早先盲测的样本）"]


def test_a_blind_test_on_new_contexts_passes_beside_an_old_one(
    services: Services, clock: ManualClock
) -> None:
    blind_run(services, clock, WEEK_START - timedelta(days=30), valid=60, correct=50, keys="old")
    observed_week(services, clock)
    assert verdict_of(judge(services)) == "passed"


def test_a_failure_wins_over_missing_evidence(services: Services, clock: ManualClock) -> None:
    observed_week(services, clock, drill=False, blind_valid=60, blind_correct=45)
    assert verdict_of(judge(services)) == "failed"  # measured misses make it failed, not unknown


def test_the_blind_test_judged_is_the_one_of_the_default_backend(
    services: Services, clock: ManualClock
) -> None:
    observed_week(services, clock, blind=False)
    blind_run(services, clock, NOW - timedelta(days=1), backend="style", valid=60, correct=59)
    clock.set_time(NOW)
    outcome = judge(services)  # only a "style" run exists; the active backend is deepseek
    assert verdict_of(outcome) == "insufficient" and failed_names(outcome) == ["盲测"]


def test_the_module_is_registered_with_the_gate_command() -> None:
    from twin.eval.gates import default_registry, load_judges

    load_judges()
    assert default_registry.get("M4") is not None
