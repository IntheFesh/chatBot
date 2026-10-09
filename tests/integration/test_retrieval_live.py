"""The real embedding model: download, encode, rank.  Skipped unless ``TWIN_LIVE=1``.

Downloads ``BAAI/bge-small-zh-v1.5`` (about 100 MB) from Hugging Face once into
``data/models/embeddings`` (override with ``TWIN_LIVE_MODEL_DIR``) and runs the same code path
as ``twin retrieval rebuild`` on a synthetic conversation.  The timing is recorded in
``docs/PERFORMANCE.md``; this test only checks that the result is sane.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from twin.retrieval.embedder import (
    EmbeddingService,
    EncodeKind,
    SentenceTransformerBackend,
    read_manifest,
)

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parents[2]
MODEL = "BAAI/bge-small-zh-v1.5"


def cache_dir() -> Path:
    return Path(os.environ.get("TWIN_LIVE_MODEL_DIR", ROOT / "data" / "models" / "embeddings"))


@pytest.fixture(scope="module")
def service() -> EmbeddingService:
    return EmbeddingService(SentenceTransformerBackend(MODEL, "cpu", cache_dir()))


def load_bench() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "bench_retrieval", ROOT / "scripts" / "bench_retrieval.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_real_model_downloads_loads_and_encodes(service: EmbeddingService) -> None:
    info = service.info
    assert (info.model, info.dimension) == (MODEL, 512)
    assert len(info.revision) == 40 and len(info.weights_sha256) == 64
    record = read_manifest(cache_dir(), MODEL)
    assert record is not None
    assert record["revision"] == info.revision and record["weights_sha256"] == info.weights_sha256

    vectors = service.encode(["今天中午想吃火锅", "晚上我们去吃火锅吧", "明天要交论文了"])
    assert vectors.shape == (3, 512) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    same_topic = float(vectors[0] @ vectors[1])
    other_topic = float(vectors[0] @ vectors[2])
    assert same_topic > other_topic + 0.05


def test_a_query_is_closer_to_its_answer_than_to_unrelated_text(service: EmbeddingService) -> None:
    query = service.encode_one("附近有什么好吃的饭店", EncodeKind.QUERY)
    related, unrelated = service.encode(
        ["推荐你去吃楼下新开的川菜馆，味道很不错", "明天上午九点开会，记得带电脑"],
        EncodeKind.PASSAGE,
    )
    assert float(query @ related) > float(query @ unrelated)


def test_the_library_is_built_and_queried_with_the_real_model(
    service: EmbeddingService, tmp_path: Path
) -> None:
    result = load_bench().run_benchmark(tmp_path, days=14, embedder=service, queries=3)
    assert result["encoded"] > 100 and result["held_out"] > 0
    assert result["dimension"] == 512 and result["device"] == "cpu"
    assert result["seconds_per_1000"] > 0 and result["query_ms_mean"] > 0
    print(
        f"\nretrieval benchmark: {result['encoded']} windows, "
        f"{result['seconds_per_1000']:.1f} s per 1,000 windows, "
        f"query {result['query_ms_mean']:.0f} ms, load {result['model_load_seconds']:.1f} s, "
        f"peak {result['peak_rss_mb']:.0f} MB"
    )
