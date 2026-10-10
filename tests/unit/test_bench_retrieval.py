"""The retrieval benchmark script (R-RET-006): it must really build a library and measure it."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from tests.support.embedding import HashingBackend
from twin.retrieval.embedder import EmbeddingService

ROOT = Path(__file__).resolve().parents[2]


def load_bench() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "bench_retrieval", ROOT / "scripts" / "bench_retrieval.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_benchmark_builds_a_library_and_measures_encoding_and_queries(tmp_path: Path) -> None:
    bench = load_bench()
    result = bench.run_benchmark(
        tmp_path, days=14, embedder=EmbeddingService(HashingBackend()), queries=4
    )
    assert result["windows"] > 100 and result["encoded"] > 50 and result["held_out"] > 0
    assert result["encoded"] + result["held_out"] <= result["windows"]
    assert result["dimension"] == HashingBackend().info.dimension
    assert result["encode_seconds"] >= 0 and result["query_ms_mean"] > 0
    assert result["peak_rss_mb"] > 0 and result["messages"] > 500


def test_the_vector_benchmark_writes_searches_and_reports(tmp_path: Path) -> None:
    bench = load_bench()
    result = bench.bench_vector_table(tmp_path, 700, dimension=16, chunk=256, searches=3)
    assert result["vectors"] == 700 and result["hits"] == 50 and len(result["search_ms"]) == 3
    assert result["write_seconds"] > 0 and result["optimize_seconds"] >= 0
