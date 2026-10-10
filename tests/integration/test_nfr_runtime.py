"""The runtime numbers of R-NFR-001 to R-NFR-003, measured on the machine the suite runs on.

The tool is ``scripts/bench_runtime.py``; the numbers of the sandbox and the ones still to be taken
on the user's Windows computer are in ``docs/PERFORMANCE.md``.  The limits here are the ones of the
specification, so a machine that cannot meet them fails; what a run prints (``-s``) is its numbers.

* R-NFR-001: the engine's own share of the generation time (the model is the ``live`` test of
  ``test_nfr_live.py``): a tenth of the 20 seconds;
* R-NFR-002: resident memory below 1.5 GB and the processor at rest below 2 %;
* R-NFR-003: from the start of the process to ``application_running`` in under 30 seconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.scripts import load_script

pytestmark = pytest.mark.integration

bench = load_script("bench_runtime")


async def test_the_engine_needs_a_small_part_of_the_time_the_specification_allows(
    tmp_path: Path,
) -> None:
    """R-NFR-001: p95 of a reply, without the model and without the deliberate delay."""
    home = tmp_path / "engine"
    home.mkdir()
    found = await bench.measure_engine(30, home)
    print(
        f"engine: {found['replies']} replies, p50 {found['p50_s'] * 1000:.0f} ms, "
        f"p95 {found['p95_s'] * 1000:.0f} ms, max {found['max_s'] * 1000:.0f} ms"
    )
    assert found["replies"] >= 30  # one run of the pipeline for every message that was answered
    assert found["p95_s"] < found["budget_s"] == bench.GENERATION_LIMIT_S * bench.ENGINE_SHARE
    assert found["max_s"] < bench.GENERATION_LIMIT_S  # not one reply on its own near the limit


def test_a_real_application_starts_within_thirty_seconds_and_rests_within_its_limits(
    tmp_path: Path,
) -> None:
    """R-NFR-003 and R-NFR-002 on a real ``twin run`` (a child process, the system clock)."""
    home = tmp_path / "process"
    home.mkdir()
    found = bench.measure_process(idle_seconds=10.0, settle_seconds=3.0, home=home)
    print(
        f"startup {found['startup_s']:.1f} s, at rest: processor "
        f"{found['idle_cpu_percent']} %, resident {found['rss_mb']} MB"
    )
    assert found["startup_s"] < bench.STARTUP_LIMIT_S  # R-NFR-003
    if found["rss_mb"] is not None:  # (a platform that cannot say does not fail the limit)
        assert found["rss_mb"] < bench.RSS_LIMIT_MB  # R-NFR-002
    if found["idle_cpu_percent"] is not None:
        assert found["idle_cpu_percent"] < bench.IDLE_CPU_LIMIT_PERCENT  # R-NFR-002
