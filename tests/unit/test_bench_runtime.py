"""scripts/bench_runtime.py: the arithmetic and the verdicts of the runtime numbers (R-NFR)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.process_metrics import (
    child_cpu_seconds,
    child_rss_bytes,
    cpu_seconds,
    current_rss_bytes,
    megabytes,
    peak_rss_bytes,
)
from tests.support.scripts import load_script

bench = load_script("bench_runtime")


def test_the_percentile_is_the_nearest_rank() -> None:
    values = [float(n) for n in range(1, 101)]  # 1 .. 100
    assert bench.percentile(values, 95) == 95.0 and bench.percentile(values, 50) == 50.0
    assert bench.percentile(values, 100) == 100.0 and bench.percentile(values, 1) == 1.0
    assert bench.percentile([3.0, 1.0, 2.0], 95) == 3.0  # unsorted in, the largest of three
    assert bench.percentile([7.0], 95) == 7.0 and bench.percentile([], 95) == 0.0


def test_the_statistics_of_a_sample() -> None:
    found = bench.stats_of([0.1, 0.2, 0.3, 0.4, 2.0])
    assert found.count == 5 and found.maximum == 2.0 and found.p50 == 0.3 and found.p95 == 2.0
    assert found.mean == pytest.approx(0.6)
    assert bench.stats_of([]) == bench.Stats(0, 0.0, 0.0, 0.0, 0.0)


def engine(p95: float = 0.09) -> dict[str, float]:
    return {
        "replies": 100,
        "mean_s": 0.08,
        "p50_s": 0.078,
        "p95_s": p95,
        "max_s": 0.2,
        "budget_s": bench.GENERATION_LIMIT_S * bench.ENGINE_SHARE,
        "limit_s": bench.GENERATION_LIMIT_S,
        "limit_thinking_s": bench.GENERATION_LIMIT_THINKING_S,
    }


def process(**changes: object) -> dict[str, object]:
    base: dict[str, object] = {
        "startup_s": 3.0,
        "startup_limit_s": bench.STARTUP_LIMIT_S,
        "idle_seconds": 30.0,
        "idle_cpu_percent": 0.4,
        "idle_cpu_limit_percent": bench.IDLE_CPU_LIMIT_PERCENT,
        "rss_mb": 203.0,
        "rss_limit_mb": bench.RSS_LIMIT_MB,
    }
    return base | changes


def test_numbers_inside_the_limits_are_ok() -> None:
    result = bench.Result(engine=engine(), idle=process(), startup=process())
    bench.judge(result)
    assert result.problems == []
    text = bench.render(result)
    assert "result: OK" in text and "p95 90 ms" in text and "3.0 s (limit 30 s)" in text
    assert "processor 0.40 % of one core (limit 2 %)" in text and "resident memory 203 MB" in text


@pytest.mark.parametrize(
    ("changes", "words"),
    [
        ({"startup_s": 30.0}, "startup 30.0 s is not below 30 s"),  # R-NFR-003: below, not equal
        ({"idle_cpu_percent": 2.0}, "idle processor use 2.00 % >= 2 %"),  # R-NFR-002
        ({"rss_mb": 1536.0}, "resident memory 1536 MB >= 1.5 GB"),
    ],
)
def test_a_number_at_its_limit_is_a_problem(changes: dict[str, object], words: str) -> None:
    result = bench.Result(idle=process(**changes))
    bench.judge(result)
    assert any(words in problem for problem in result.problems), result.problems
    assert "result: ABOVE A LIMIT" in bench.render(result)


def test_the_engine_may_use_a_tenth_of_the_generation_time() -> None:
    inside = bench.Result(engine=engine(p95=1.99))
    bench.judge(inside)
    assert inside.problems == []
    above = bench.Result(engine=engine(p95=2.01))
    bench.judge(above)
    assert above.problems and "of the 20 s of R-NFR-001" in above.problems[0]


def test_a_platform_that_cannot_read_the_numbers_does_not_fail_the_limits() -> None:
    result = bench.Result(idle=process(idle_cpu_percent=None, rss_mb=None))
    bench.judge(result)
    assert result.problems == []
    assert "processor n/a" in bench.render(result) and "resident memory n/a" in bench.render(result)


def test_what_was_not_measured_is_not_in_the_report() -> None:
    text = bench.render(bench.Result())
    assert "R-NFR-001" not in text and "R-NFR-002" not in text and "R-NFR-003" not in text
    assert text.splitlines()[0].count(",") >= 3  # the machine, always


def test_the_arguments() -> None:
    args = bench.parse_args([])
    assert (args.engine_messages, args.idle_seconds, args.settle_seconds) == (100, 30.0, 10.0)
    assert args.only == ["engine", "idle", "startup"] and args.json is None
    only = bench.parse_args(["--only", "startup", "--json", "out.json"])
    assert only.only == ["startup"] and only.json == Path("out.json")
    with pytest.raises(SystemExit):
        bench.parse_args(["--idle-seconds", "0"])
    with pytest.raises(SystemExit):
        bench.parse_args(["--only", "nothing"])


def test_the_result_can_be_written_as_json() -> None:
    result = bench.Result(engine=engine(), idle=process())
    from dataclasses import asdict

    data = json.loads(json.dumps(asdict(result)))
    assert data["engine"]["p95_s"] == 0.09 and data["machine"]["cores"] >= 1
    assert data["problems"] == [] and data["startup"] is None


# ------------------------------------------------------------ what the system says of a process


def test_this_process_has_a_resident_set_and_uses_processor_time() -> None:
    assert 20 < megabytes(current_rss_bytes()) < 4096
    assert peak_rss_bytes() >= current_rss_bytes() * 0.5
    before = cpu_seconds()
    sum(n * n for n in range(200_000))
    assert cpu_seconds() > before


def test_another_process_is_read_the_same_way() -> None:
    import os

    pid = os.getpid()
    assert child_rss_bytes(pid) is not None and child_rss_bytes(pid) > 10_000_000
    first = child_cpu_seconds(pid)
    sum(n * n for n in range(300_000))
    second = child_cpu_seconds(pid)
    assert first is not None and second is not None and second >= first
    assert child_rss_bytes(2**22 + 12345) is None  # no such process
    assert child_cpu_seconds(2**22 + 12345) is None
