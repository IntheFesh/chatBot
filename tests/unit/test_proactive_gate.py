"""The judge of the M3 gate: seven days of observation in the last ten, and a rating of 4 (SPEC 26).

Round 18 (user decision, R18-F6, D-850): the window is the last ten completed local days, seven of
them must be days of observation (each strictly inside its range), an excused day counts neither
way and more than three of them are "not enough".  The tests written for the old rule (seven
watched days in a row) were rewritten to the new one; what they protect is unchanged: a day that
is not watched, switched off or planned for ``0-0`` never counts as observed (D-663).
"""

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
from twin.eval.proactive_audit import Audit, audit_recent
from twin.eval.proactive_gate import (
    M3_DAYS,
    M3_EXEMPT_MAX,
    M3_MIN_RATING,
    M3_OBSERVATION_DAYS,
    judge_audit,
)
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


def test_the_rule_is_seven_observation_days_in_ten_and_an_average_of_four() -> None:
    """R18-F6 (D-850): ten days, seven of them observed, at most three excused; a rating of 4."""
    assert M3_DAYS == 10 and M3_OBSERVATION_DAYS == 7 and M3_EXEMPT_MAX == 3
    assert M3_MIN_RATING == 4.0


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


def test_fewer_than_seven_observation_days_is_an_observation_period_not_a_failure(
    books: Books,
) -> None:
    books.good_week(days=5, last=1)  # the five older days of the ten were not watched
    books.rate(5)
    verdict, summary = verdict_of(books)
    assert verdict == "insufficient"
    assert summary == (
        "观察期未满：最近 10 天里有 5 个观察日，还差 2 个（程序没有运行的 5 天不计入观察）"
    )


def test_an_unwatched_day_in_the_middle_counts_for_nothing(books: Books) -> None:
    """Six watched days around an unwatched one are six days of observation, not seven."""
    books.good_week(days=3, last=1)
    books.day(books.ago(4), (Sent(10),), opened=False)
    books.good_week(days=3, last=5)
    books.rate(5)
    verdict, summary = verdict_of(books)
    assert verdict == "insufficient" and "最近 10 天里有 6 个观察日，还差 1 个" in summary


def test_a_week_with_proactive_switched_off_is_insufficient_not_passed(books: Books) -> None:
    """R17-14 (D-663): seven quiet days with the switch off are no evidence that she writes.

    Before the fix such a week passed: every day was "observed" and "inside its range 0-0".
    """
    for back in range(1, 8):
        books.day(books.ago(back), (), enabled=False)
    books.rate(5)
    verdict, summary = verdict_of(books)
    assert verdict == "insufficient"
    assert summary == (
        "观察期未满：最近 10 天里有 0 个观察日，还差 7 个"
        "（主动消息关闭、暂停或范围 0-0 的 7 天、程序没有运行的 3 天不计入观察）"
    )


def test_a_range_of_zero_to_zero_is_insufficient_too(books: Books) -> None:
    for back in range(1, 8):
        books.day(books.ago(back), (), low=0, high=0, enabled=True)
    books.rate(5)
    assert verdict_of(books)[0] == "insufficient"


def test_days_switched_off_inside_the_window_do_not_count(books: Books) -> None:
    books.good_week(days=3, last=1)
    books.day(books.ago(4), (), enabled=False)
    books.good_week(days=3, last=5)
    books.rate(5)
    verdict, summary = verdict_of(books)
    assert verdict == "insufficient" and "最近 10 天里有 6 个观察日，还差 1 个" in summary
    assert "主动消息关闭、暂停或范围 0-0 的 1 天" in summary


def test_a_message_sent_with_the_switch_off_is_still_a_violation_in_the_checks(
    books: Books,
) -> None:
    """The streak is not the only thing the audit reads: sending while off breaks the range 0-0."""
    books.good_week(days=6, last=2)
    books.day(books.ago(1), (Sent(9),), enabled=False)
    books.rate(5)
    audit = audit_recent(
        books.log,
        books.ratings,
        books.services.settings.proactive,
        books.time,
        books.clock,
        days=M3_DAYS,
    )
    assert not audit.compliant and "不在 0-0 范围内" in audit.days[-1].problems[0]
    verdict, checks, summary = judge_audit(audit)
    assert verdict == "insufficient"  # six days of observation: that is said first
    count = next(check for check in checks if check.name == "每天次数在设定范围内")
    assert not count.passed and "发 1 条（0-0）" in count.detail
    assert "主动消息关闭、暂停或范围 0-0 的 1 天" in summary


def test_the_checks_name_the_days_that_do_not_count(books: Books) -> None:
    books.good_week(days=6, last=2)
    books.day(books.ago(1), (), enabled=False)
    audit = audit_recent(
        books.log,
        books.ratings,
        books.services.settings.proactive,
        books.time,
        books.clock,
        days=M3_DAYS,
    )
    _, checks, _ = judge_audit(audit)
    observed = next(check for check in checks if check.name == "最近 10 个当地日里有 7 个观察日")
    assert not observed.passed and "主动消息关闭、暂停或范围 0-0 的 1 天" in observed.detail
    assert observed.value == 6.0 and observed.threshold == 7.0


def test_an_empty_installation_has_no_observed_days(books: Books) -> None:
    verdict, summary = verdict_of(books)
    assert verdict == "insufficient" and "最近 10 天里有 0 个观察日，还差 7 个" in summary


def test_the_checks_say_what_is_missing(books: Books) -> None:
    books.good_week(days=4, last=1)
    audit = audit_recent(
        books.log,
        books.ratings,
        books.services.settings.proactive,
        books.time,
        books.clock,
        days=M3_DAYS,
    )
    _, checks, _ = judge_audit(audit)
    by_name = {check.name: check for check in checks}
    assert not by_name["最近 10 个当地日里有 7 个观察日"].passed
    assert "还差 3 个" in by_name["最近 10 个当地日里有 7 个观察日"].detail
    assert by_name["免责日 ≤ 3 个"].passed
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
    assert audits[0].verdict == "passed" and audits[0].summary["observation_days"] == 7
    assert audits[0].params["days"] == 10 and audits[0].params["observation_days"] == 7
    assert gates[0].milestone == "M3" and gates[0].verdict == "passed"
    assert audits[0].id in str(gates[0].summary)
    assert check_gate(books.services, "M3").exit_code == EXIT_PASSED  # only reads


def test_a_gate_that_is_not_passed_exits_with_one(books: Books) -> None:
    books.good_week(days=3, last=1)
    outcome = run_gate(books.services, "M3")
    assert outcome.exit_code == EXIT_NOT_PASSED and outcome.status == "insufficient"
    assert check_gate(books.services, "M3").exit_code == EXIT_NOT_PASSED


# ------------------------------------------------- round 18: excused days (R18-F6, D-850)


def m3_audit(books: Books) -> Audit:
    return audit_recent(
        books.log,
        books.ratings,
        books.services.settings.proactive,
        books.time,
        books.clock,
        days=M3_DAYS,
    )


def test_r18_f6_a_week_in_which_the_window_stopped_every_message_is_not_a_pass(
    books: Books,
) -> None:
    """R18-F6 (D-850): seven days on which nothing went out because the window was closed.

    The audit's repro: every day was below its minimum and "excused" by a refusal of the window,
    so the old rule found seven compliant days and passed M3 without one proactive message.  An
    excused day is no day of observation: four or more of them are "not enough", and the reasons
    of the refusals are listed.
    """
    for back in range(1, 8):
        books.day(books.ago(back), (), refused=("window_closed",))
    books.rate(5)
    verdict, summary = verdict_of(books)
    assert verdict == "insufficient"
    assert "免责日 7 个" in summary and "被窗口抑制 7" in summary


def test_r18_f6_seven_observation_days_in_the_last_ten_pass_with_three_excused_days(
    books: Books,
) -> None:
    """R18-F6 (D-850): an excused day is neither a violation nor a day of observation."""
    books.good_week(days=7, last=4)
    for back in (1, 2, 3):
        books.day(books.ago(back), (), refused=("quota_exhausted",))
    books.rate(5)
    verdict, summary = verdict_of(books)
    assert verdict == "passed", summary
    audit = m3_audit(books)
    assert audit.observation_days == 7 and audit.exempt_days == 3
    assert (audit.first_day, audit.last_day) == (books.ago(10), books.ago(1))


def test_r18_f6_four_excused_days_are_not_enough_and_the_reasons_are_listed(
    books: Books,
) -> None:
    """R18-F6 (D-850): more than three excused days in the ten: "not enough", with the reasons."""
    books.good_week(days=6, last=1)
    reasons = ("window_closed", "quota_exhausted", "budget", "window_closed")
    for back, reason in zip((7, 8, 9, 10), reasons, strict=True):
        books.day(books.ago(back), (), refused=(reason,))
    books.rate(5)
    audit = m3_audit(books)
    verdict, checks, summary = judge_audit(audit)
    assert verdict == "insufficient"
    assert audit.exempt_reasons == {"window_closed": 2, "quota_exhausted": 1, "budget": 1}
    assert "免责日 4 个" in summary
    assert "被窗口抑制 2" in summary and "平台条数用完 1" in summary and "预算降级" in summary
    excused = next(check for check in checks if check.name.startswith("免责日"))
    assert not excused.passed and "被窗口抑制 2" in excused.detail


def test_r18_f6_an_unwatched_day_does_not_restart_the_count(books: Books) -> None:
    """R18-F6 (D-850): seven observation days anywhere in the ten, not seven in a row."""
    books.good_week(days=3, last=1)
    books.day(books.ago(4), (Sent(10),), opened=False)
    books.good_week(days=4, last=5)
    books.rate(5)
    assert verdict_of(books)[0] == "passed"


def test_r18_f6_a_day_the_user_paused_is_not_an_observation_day(books: Books) -> None:
    """R18-F6 (D-850): ``/暂停`` (and ``/主动 关``) days do not count, they do not excuse."""
    books.good_week(days=6, last=1)
    books.day(books.ago(7), (), refused=("paused",))
    books.rate(5)
    verdict, summary = verdict_of(books)
    assert verdict == "insufficient" and "还差 1 个" in summary
    audit = m3_audit(books)
    paused = next(day for day in audit.days if day.day == books.ago(7))
    assert not paused.active and not paused.exempt and not paused.observation
    assert paused.compliant and audit.exempt_days == 0 and audit.inactive_days == 1


def test_r18_f6_an_observation_day_above_its_maximum_fails(books: Books) -> None:
    """R18-F6 (D-850): every observation day must be strictly inside its range."""
    books.good_week(days=7, last=2)
    books.day(books.ago(1), tuple(Sent(7 + 2 * n) for n in range(7)), high=6)
    books.rate(5)
    verdict, summary = verdict_of(books)
    assert verdict == "failed" and f"{books.ago(1):%m-%d}" in summary


def test_r18_f6_a_day_below_its_minimum_with_nothing_external_is_a_violation(
    books: Books,
) -> None:
    """R18-F6 (D-850): only a refusal of the window, count, login or budget makes a day excused."""
    books.good_week(days=7, last=2)
    books.day(books.ago(1), (), refused=("planner_declined",))
    books.rate(5)
    assert verdict_of(books)[0] == "failed"


def test_r18_f6_a_day_in_range_with_a_refusal_of_the_window_still_counts(books: Books) -> None:
    """R18-F6 (D-850): a day inside its range is an observation day whatever was refused."""
    books.good_week(days=6, last=2)
    books.day(books.ago(1), (Sent(9), Sent(14)), refused=("window_closed",))
    books.rate(5)
    assert verdict_of(books)[0] == "passed"
