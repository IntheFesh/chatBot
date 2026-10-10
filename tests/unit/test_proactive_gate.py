"""The judge of the M3 gate: seven watched days that pass the audit, and a rating of 4 (SPEC 26)."""

from __future__ import annotations

import pytest

from tests.support.clock import ManualClock
from tests.support.proactive_books import Books, Sent
from twin.eval.gates import (
    EXIT_NOT_PASSED,
    EXIT_PASSED,
    check_gate,
    load_judges,
    run_gate,
)
from twin.eval.proactive_audit import audit_recent
from twin.eval.proactive_gate import M3_DAYS, M3_MIN_RATING, judge_audit
from twin.eval.store import EvalStore
from twin.services import Services


@pytest.fixture
def books(services: Services, clock: ManualClock) -> Books:
    return Books(services, clock)


def verdict_of(books: Books) -> tuple[str, str]:
    audit = audit_recent(
        books.log,
        books.ratings,
        books.services.settings.proactive,
        books.time,
        books.clock,
        days=M3_DAYS,
    )
    verdict, _, summary = judge_audit(audit)
    return verdict, summary


def test_the_rule_is_seven_days_and_an_average_of_four() -> None:
    assert M3_DAYS == 7 and M3_MIN_RATING == 4.0


def test_a_good_week_and_a_good_rating_pass(books: Books) -> None:
    books.good_week()
    books.rate(5, days_ago=1)
    books.rate(4, days_ago=0.5)
    verdict, summary = verdict_of(books)
    assert verdict == "passed" and "M3 通过" in summary and "4.50" in summary


def test_the_average_of_exactly_four_is_enough(books: Books) -> None:
    books.good_week()
    books.rate(5)
    books.rate(3)
    assert verdict_of(books)[0] == "passed"


def test_an_average_below_four_fails(books: Books) -> None:
    books.good_week()
    books.rate(4)
    books.rate(3)
    verdict, summary = verdict_of(books)
    assert verdict == "failed" and "低于 4" in summary


def test_a_week_without_a_rating_is_not_enough_yet(books: Books) -> None:
    books.good_week()
    verdict, summary = verdict_of(books)
    assert verdict == "insufficient" and "没有 /评分" in summary


def test_a_rating_from_before_the_week_does_not_count(books: Books) -> None:
    books.good_week()
    books.rate(5, days_ago=12)
    assert verdict_of(books)[0] == "insufficient"


def test_a_violation_fails_even_with_a_perfect_rating(books: Books) -> None:
    books.good_week()
    books.row(books.ago(2), 3, 0, kind="share", outcome="sent", state="deep_sleep")
    books.rate(5)
    verdict, summary = verdict_of(books)
    assert verdict == "failed" and "审计不合规" in summary and "深睡时段发了 1 条" in summary


def test_a_day_out_of_range_fails(books: Books) -> None:
    books.good_week(days=6, last=2)
    books.day(books.ago(1), (), refused=("spacing",))
    books.rate(5)
    verdict, summary = verdict_of(books)
    assert verdict == "failed" and f"{books.ago(1):%m-%d}" in summary


def test_the_edge_allowance_broken_in_a_week_fails(books: Books) -> None:
    books.good_week()
    for back in (6, 4, 2):
        books.row(books.ago(back), 23, 50, kind="edge", outcome="sent", state="sleep_edge")
    books.rate(5)
    verdict, summary = verdict_of(books)
    assert verdict == "failed" and summary == "审计不合规：边缘消息超过每周上限"


def test_fewer_than_seven_watched_days_is_an_observation_period_not_a_failure(
    books: Books,
) -> None:
    books.good_week(days=5, last=1)  # the two older days were not watched
    books.rate(5)
    verdict, summary = verdict_of(books)
    assert verdict == "insufficient"
    assert summary == "观察期未满：已连续观察 5 天，还差 2 天"


def test_an_unwatched_day_in_the_middle_starts_the_count_again(books: Books) -> None:
    books.good_week(days=3, last=1)
    books.day(books.ago(4), (Sent(10),), opened=False)
    books.good_week(days=3, last=5)
    books.rate(5)
    verdict, summary = verdict_of(books)
    assert verdict == "insufficient" and "已连续观察 3 天，还差 4 天" in summary


def test_an_empty_installation_has_no_observed_days(books: Books) -> None:
    verdict, summary = verdict_of(books)
    assert verdict == "insufficient" and "已连续观察 0 天，还差 7 天" in summary


def test_the_checks_say_what_is_missing(books: Books) -> None:
    books.good_week(days=4, last=1)
    audit = audit_recent(
        books.log,
        books.ratings,
        books.services.settings.proactive,
        books.time,
        books.clock,
        days=7,
    )
    _, checks, _ = judge_audit(audit)
    by_name = {check.name: check for check in checks}
    assert not by_name["连续观察 7 个当地日"].passed
    assert "还差 3 天" in by_name["连续观察 7 个当地日"].detail
    assert not by_name["这段时间有 /评分"].passed
    assert "给她打个分" in by_name["这段时间有 /评分"].detail
    assert by_name["深睡核心时段主动消息 0 次"].passed


# ---------------------------------------------------------------- through the gate command


def test_the_judge_is_registered_for_m3() -> None:
    assert load_judges().get("M3") is not None


def test_running_the_gate_stores_the_verdict_and_the_audit_it_was_made_from(
    books: Books,
) -> None:
    books.good_week()
    books.rate(5)
    outcome = run_gate(books.services, "M3")
    assert outcome.exit_code == EXIT_PASSED and outcome.status == "passed"
    store = EvalStore(books.services.db, books.clock)
    audits = store.list_runs("proactive_audit")
    gates = store.list_runs("gate")
    assert len(audits) == 1 and len(gates) == 1
    assert audits[0].verdict == "passed" and audits[0].summary["streak"] == 7
    assert audits[0].params["days"] == 7
    assert gates[0].milestone == "M3" and gates[0].verdict == "passed"
    assert audits[0].id in str(gates[0].summary)
    assert check_gate(books.services, "M3").exit_code == EXIT_PASSED  # only reads


def test_a_gate_that_is_not_passed_exits_with_one(books: Books) -> None:
    books.good_week(days=3, last=1)
    outcome = run_gate(books.services, "M3")
    assert outcome.exit_code == EXIT_NOT_PASSED and outcome.status == "insufficient"
    assert check_gate(books.services, "M3").exit_code == EXIT_NOT_PASSED
