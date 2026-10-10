"""scripts/soak.py: the long-run tool (R-NFR-002, R-ARCH-004): days, verdicts, report."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, date, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.support.scripts import SCRIPTS, load_script

soak = load_script("soak")
FIRST = date(2026, 10, 26)


# ---------------------------------------------------------------------------------- arguments


def test_the_arguments_have_the_defaults_of_the_task() -> None:
    args = soak.parse_args(["--accelerated"])
    assert (args.days, args.seed, args.platform) == (14, 1, "console")
    assert args.first_day == "2026-10-26" and args.output is None and args.json is None
    assert args.rss_limit_mb == 1536.0  # R-NFR-002: 1.5 GB


@pytest.mark.parametrize(
    "argv",
    [
        [],  # the clock is not optional
        ["--days", "14"],
        ["--accelerated", "--days", "0"],
        ["--accelerated", "--first-day", "tomorrow"],
        ["--accelerated", "--platform", "telegram"],
    ],
)
def test_a_wrong_argument_is_refused(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        soak.parse_args(argv)
    assert stopped.value.code == 2
    assert capsys.readouterr().err


# ------------------------------------------------------------------------------- the user's days


def test_the_same_seed_gives_the_same_days_and_another_seed_other_days() -> None:
    assert soak.plan_days(3, FIRST, 14) == soak.plan_days(3, FIRST, 14)
    assert soak.plan_days(3, FIRST, 14) != soak.plan_days(4, FIRST, 14)


def test_the_days_are_the_days_of_a_person() -> None:
    plans = soak.plan_days(1, FIRST, 14)
    assert [plan.day for plan in plans] == [FIRST + timedelta(days=n) for n in range(14)]
    kinds = {event.kind for plan in plans for event in plan.events}
    assert kinds == {"text", "picture", "sticker", "command"}  # pictures, stickers, commands too
    assert {plan.mood for plan in plans} >= {"normal", "silent"}
    for plan in plans:
        times = [event.at_s for event in plan.events]
        assert times == sorted(times)
        assert all(0 <= at < soak.SECONDS_PER_DAY for at in times), plan.day
        if plan.mood == "silent":
            assert plan.events == ()
        else:
            assert any(event.kind == "text" for event in plan.events)
    night = [e for p in plans for e in p.events if e.at_s < 5 * 3600]
    evening = [e for p in plans for e in p.events if 19 * 3600 <= e.at_s < 23 * 3600 + 3600]
    assert night and len(evening) > 3 * len(night)  # now and then at night, mostly in the evening


def test_a_run_of_eight_days_or_more_has_three_days_without_a_word() -> None:
    for seed in range(1, 8):
        silent = [plan.mood == "silent" for plan in soak.plan_days(seed, FIRST, 14)]
        runs = "".join("s" if flag else "." for flag in silent)
        assert "sss" in runs, (seed, runs)
        assert silent[0] is False and silent[1] is False  # the first two days fill the caches


def test_asking_for_more_days_keeps_the_first_ones() -> None:
    short = soak.plan_days(2, FIRST, 6)
    longer = soak.plan_days(2, FIRST, 7)
    assert longer[:6] == short  # (below eight days there is no long silence to move)


def test_a_soak_needs_at_least_one_day() -> None:
    with pytest.raises(ValueError, match="at least one day"):
        soak.plan_days(1, FIRST, 0)


# ------------------------------------------------------------------------------------ verdicts


def sample(index: int, **changes: object) -> Any:
    base = soak.Sample(
        day_index=index,
        day=(FIRST + timedelta(days=index)).isoformat(),
        local_end="2026-10-27 00:00:00",
        rss_mb=400.0,
        rss_anon_mb=270.0,
        rss_file_mb=130.0,
        tasks=27,
        threads=10,
        jobs_pending=0,
        jobs_awaiting_approval=0,
        jobs_running=0,
        jobs_failed=0,
        oldest_pending_s=0.0,
        db_mb=10.0,
        messages_in=30 * (index + 1),
        bubbles_out=30 * (index + 1),
        proactive_sent=3 * (index + 1),
        errors=0,
        warnings=0,
        clock_steps=300 * (index + 1),
    )
    return replace(base, **changes)


def healthy(days: int = 14) -> Any:
    report = soak.SoakReport(days, 1, "console", FIRST.isoformat())
    report.samples = [sample(n, rss_mb=400.0 + (n % 2), db_mb=10.0 + 0.1 * n) for n in range(days)]
    return report


def verdict(report: Any, name: str) -> Any:
    found = [v for v in soak.evaluate(report) if v.name.startswith(name)]
    assert len(found) == 1, name
    return found[0]


def test_a_healthy_run_passes_every_verdict() -> None:
    report = healthy()
    report.verdicts = soak.evaluate(report)
    assert report.passed, soak.render(report)
    assert {v.name.split(" (")[0] for v in report.verdicts} >= {
        "memory limit",
        "memory growth",
        "job queue",
        "database size",
        "tasks and threads",
        "exceptions",
        "promises of the scenarios",
    }


def test_a_run_without_a_day_does_not_pass() -> None:
    report = soak.SoakReport(14, 1, "console", FIRST.isoformat())
    verdicts = soak.evaluate(report)
    assert [v.ok for v in verdicts] == [False]
    report.verdicts = verdicts
    assert not report.passed
    assert not soak.SoakReport(1, 1, "console", FIRST.isoformat()).passed  # no verdicts, no pass


def test_the_memory_limit_is_the_one_of_the_specification() -> None:
    report = healthy()
    report.samples[5] = sample(5, rss_mb=1600.0)
    assert not verdict(report, "memory limit").ok
    just_below = healthy()
    just_below.samples[5] = sample(5, rss_mb=1535.0)
    assert verdict(just_below, "memory limit").ok
    just_below.peak_rss_mb = 1600.0
    assert not verdict(just_below, "memory limit").ok


def test_a_leak_shows_as_a_slope_and_noise_does_not() -> None:
    leaking = healthy(30)
    leaking.samples = [sample(n, rss_mb=400.0 + 8.0 * n) for n in range(30)]
    found = verdict(leaking, "memory growth")
    assert not found.ok and "+8.00 MB/day over days 22-30" in found.detail
    noisy = healthy(30)
    noisy.samples = [sample(n, rss_mb=400.0 + (5.0 if n % 2 else -5.0)) for n in range(30)]
    assert verdict(noisy, "memory growth").ok


def test_the_first_weeks_fill_the_caches_and_are_not_a_slope() -> None:
    """A process that grows 43 MB in its first week and levels off is not leaking."""
    levelling = healthy(28)
    curve = [382, 390, 399, 407, 414, 424, 425, 425, 426, 427, 430, 439, 441, 442, 444, 451]
    curve += [452, 457, 457, 457, 461, 462, 462, 462, 462, 462, 462, 463]  # a 28-day run
    levelling.samples = [sample(n, rss_mb=float(mb)) for n, mb in enumerate(curve)]
    found = verdict(levelling, "memory growth")
    assert found.ok and "over days 22-28" in found.detail
    still_growing = healthy(28)
    still_growing.samples = [sample(n, rss_mb=float(mb)) for n, mb in enumerate(curve[:21])]
    still_growing.samples += [sample(n, rss_mb=461.0 + 6.0 * (n - 20)) for n in range(21, 28)]
    assert not verdict(still_growing, "memory growth").ok  # it did not stop at the end


def test_a_short_run_says_that_the_trend_was_not_judged_and_still_judges_the_limit() -> None:
    short = healthy(14)
    short.samples = [sample(n, rss_mb=380.0 + 4.0 * n) for n in range(14)]  # +4 MB/day, 14 days
    found = verdict(short, "memory growth")
    assert found.ok and "not judged" in found.detail and "needs 26 days" in found.detail
    assert "--days 28" in found.detail
    short.samples[5] = sample(5, rss_mb=2000.0)
    assert not verdict(short, "memory limit").ok  # the limit of R-NFR-002 is judged on any run


def test_the_slope_is_the_least_squares_one() -> None:
    assert soak.slope([(0, 1), (1, 3), (2, 5)]) == pytest.approx(2.0)
    assert soak.slope([(0, 5), (1, 5), (2, 5)]) == 0.0
    assert soak.slope([(1, 2)]) == 0.0 and soak.slope([]) == 0.0
    assert soak.slope([(2, 1), (2, 9)]) == 0.0  # no spread in x


def test_a_backlog_or_a_failed_job_fails_the_queue_verdict() -> None:
    backlog = healthy()
    backlog.samples[7] = sample(7, jobs_pending=41)
    assert not verdict(backlog, "job queue").ok
    stale = healthy()
    stale.samples[7] = sample(7, jobs_pending=2, oldest_pending_s=7 * 3600.0)
    assert not verdict(stale, "job queue").ok
    failed = healthy()
    failed.samples[-1] = sample(13, jobs_failed=1)
    assert not verdict(failed, "job queue").ok
    waiting = healthy()
    waiting.samples[-1] = sample(13, jobs_awaiting_approval=9)
    found = verdict(waiting, "job queue")
    assert found.ok and "9 wait for approval" in found.detail


def test_the_database_may_grow_by_what_is_written_and_no_more() -> None:
    fast = healthy()
    fast.samples = [sample(n, db_mb=10.0 + 15.0 * n) for n in range(14)]
    assert not verdict(fast, "database size").ok
    assert verdict(healthy(), "database size").ok


def test_a_loop_that_gathers_tasks_or_threads_fails() -> None:
    tasks = healthy()
    tasks.samples[-1] = sample(13, tasks=27 + 4)
    assert not verdict(tasks, "tasks and threads").ok
    threads = healthy()
    threads.samples[-1] = sample(13, threads=14)
    assert not verdict(threads, "tasks and threads").ok
    same = healthy()
    same.samples[-1] = sample(13, tasks=30, threads=13)
    assert verdict(same, "tasks and threads").ok


def test_any_exception_fails_and_the_first_ones_are_named() -> None:
    logged = healthy()
    logged.samples[-1] = sample(13, errors=2)
    logged.error_events = ["twin.jobs:job_failed"]
    found = verdict(logged, "exceptions")
    assert not found.ok and "twin.jobs:job_failed" in found.detail
    lost = healthy()
    lost.task_errors = 1
    assert not verdict(lost, "exceptions").ok
    engine = healthy()
    engine.engine_errors = 1
    assert not verdict(engine, "exceptions").ok


def test_a_broken_promise_of_the_scenarios_fails() -> None:
    report = healthy()
    report.invariants = ["the platform's count: 11 messages after ..."]
    found = verdict(report, "promises")
    assert not found.ok and "the platform's count" in found.detail


def test_the_limits_can_be_changed() -> None:
    report = healthy()
    strict = soak.Thresholds(rss_limit_mb=300.0)
    names = {v.name: v.ok for v in soak.evaluate(report, strict)}
    assert names["memory limit (R-NFR-002)"] is False


# ------------------------------------------------------------------------------------- report


def test_the_report_has_a_line_for_every_day_and_every_verdict() -> None:
    report = healthy(4)
    report.summary = {"messages in": 120}
    report.calls = {"reply": 90}
    report.verdicts = soak.evaluate(report)
    text = soak.render(report)
    assert text.startswith("soak: 4 days from 2026-10-26, seed 1, console channel")
    day_lines = [line for line in text.splitlines() if line.startswith("  ") and "2026-10-" in line]
    assert len(day_lines) == 4
    assert text.count("[ok]") == len(report.verdicts)
    assert "result: PASSED" in text and "reply 90" in text and "messages in 120" in text
    report.invariants = ["x"]
    report.verdicts = soak.evaluate(report)
    assert "[FAIL] promises of the scenarios" in soak.render(report)
    assert "result: FAILED" in soak.render(report)


def test_the_json_holds_the_numbers() -> None:
    report = healthy(3)
    report.verdicts = soak.evaluate(report)
    data = json.loads(soak.report_json(report))
    assert data["days"] == 3 and len(data["samples"]) == 3
    assert data["samples"][0]["rss_mb"] == 400.0
    assert [v["ok"] for v in data["verdicts"]] == [True] * len(data["verdicts"])


# ----------------------------------------------------------------------------------- machinery


def test_the_log_tally_counts_levels_and_keeps_names_not_content() -> None:
    tally = soak.LogTally()
    logger = logging.getLogger("soak.test")
    logger.addHandler(tally)
    logger.propagate = False
    try:
        logger.warning("slow_thing")
        logger.error("job_failed")
        logger.info("fine")
    finally:
        logger.removeHandler(tally)
    assert (tally.errors, tally.warnings) == (1, 1)
    assert tally.events == ["soak.test:job_failed"]
    for number in range(40):
        tally.emit(logging.LogRecord("n", logging.ERROR, "f", 1, f"e{number}", None, None))
    assert len(tally.events) == 20  # the report names the first few, not all


def test_the_world_starts_at_six_in_the_morning_of_the_bots_zone() -> None:
    summer = soak.world_start(date(2026, 10, 26))
    winter = soak.world_start(date(2026, 11, 2))
    assert (summer.hour, summer.tzinfo) == (11, UTC)  # 06:00 CDT = 11:00 UTC
    assert (winter.hour, winter.tzinfo) == (
        12,
        UTC,
    )  # 06:00 CST: the clocks went back on 1 November


def test_a_two_day_run_writes_the_report_and_the_numbers_and_touches_nothing_else(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    text, numbers = tmp_path / "soak.txt", tmp_path / "soak.json"
    environment = {k: v for k, v in os.environ.items() if not k.startswith("TWIN_")}
    environment["PYTHONPATH"] = str(SCRIPTS.parent)
    done = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "soak.py"),
            "--days",
            "2",
            "--accelerated",
            "--output",
            str(text),
            "--json",
            str(numbers),
        ],
        capture_output=True,
        text=True,
        timeout=900,
        env=environment,
        cwd=work,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout == ""  # with --output the report is in the file
    assert "day 1/2" in done.stderr and "day 2/2" in done.stderr  # and progress on stderr
    report = text.read_text(encoding="utf-8")
    assert report.startswith("soak: 2 days from 2026-10-26")
    assert "result: PASSED" in report and "[FAIL]" not in report
    data = json.loads(numbers.read_text(encoding="utf-8"))
    assert [s["day"] for s in data["samples"]] == ["2026-10-26", "2026-10-27"]
    assert data["samples"][-1]["messages_in"] > 0 and data["samples"][-1]["bubbles_out"] > 0
    assert data["samples"][-1]["rss_mb"] < 1536 and data["invariants"] == []
    assert data["calls"]["reply"] > 0 and data["error_events"] == []
    assert not list(work.iterdir()), "the run wrote into the working directory"
