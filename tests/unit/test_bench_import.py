"""The import benchmark script (R-IMP-013): it must really generate, import and measure."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_bench() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "bench_import", ROOT / "scripts" / "bench_import.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_benchmark_generates_imports_and_measures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bench = load_bench()
    result_path = tmp_path / "result.json"
    code = bench.main(
        [
            "--messages", "700",
            "--batch-size", "100",
            "--workdir", str(tmp_path / "work"),
            "--json", str(result_path),
        ]
    )  # fmt: skip
    output = capsys.readouterr().out
    assert code == 0, output
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "done" and result["processed"] == 700 and result["invalid"] == 0
    assert result["inserted"] == 700 and result["batch_size"] == 100
    assert result["messages_per_s"] > 0 and result["seconds"] > 0
    assert result["peak_rss_bytes"] > 10 * 1024 * 1024
    assert result["db_bytes"] > 0 and result["tracemalloc_peak_bytes"] is None
    assert "messages imported      700" in output and "messages/s" in output
    assert "R-IMP-013 time" in output and "not judged (fewer than 100,000" in output
    assert "R-IMP-013 memory" in output and "PASS" in output
    assert (tmp_path / "work" / "export" / "report.json").is_file()  # --workdir keeps the files


def test_the_benchmark_can_trace_python_allocations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bench = load_bench()
    result_path = tmp_path / "traced.json"
    code = bench.main(
        [
            "--messages",
            "300",
            "--workdir",
            str(tmp_path / "w"),
            "--tracemalloc",
            "--json",
            str(result_path),
        ]
    )
    assert code == 0, capsys.readouterr().out
    traced = json.loads(result_path.read_text(encoding="utf-8"))["tracemalloc_peak_bytes"]
    assert isinstance(traced, int) and traced > 0


def test_peak_memory_and_formatting_helpers() -> None:
    bench = load_bench()
    assert bench.peak_rss_bytes() > 0
    assert bench.mb(5 * 1024 * 1024) == "5.0 MB"


def test_the_verdicts_follow_the_specification_targets() -> None:
    bench = load_bench()
    fast = {
        "processed": 1_000_000,
        "peak_rss_bytes": 200 * 1024 * 1024,
        "tracemalloc_peak_bytes": None,
    }
    assert bench.verdicts(fast, 20 * 60) == ("PASS", "PASS")
    assert bench.verdicts(fast, 31 * 60)[0] == "FAIL"
    heavy = {**fast, "peak_rss_bytes": 600 * 1024 * 1024}
    assert bench.verdicts(heavy, 60) == ("PASS", "FAIL")
    small = {**fast, "processed": 500}
    assert bench.verdicts(small, 99_999)[0].startswith("not judged")
    traced = {**fast, "tracemalloc_peak_bytes": 1}
    assert bench.verdicts(traced, 99_999)[0] == "not judged (traced run)"
