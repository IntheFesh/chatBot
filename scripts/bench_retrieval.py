"""Retrieval benchmark (R-RET-006): encoding time per 1,000 windows and query latency.

Writes a synthetic conversation (``tests/support/synth_chat.py``: random characters with the
timing of a real chat), builds the example library with the **real** embedding model through the
same code path as ``twin retrieval rebuild`` and reports

* the time the model needs to load (the first start also downloads it),
* the seconds per 1,000 encoded windows, and windows per second,
* the milliseconds per query (embedding of the question, nearest neighbours, MMR),
* the peak memory of the process.

The numbers depend on the CPU or GPU; they go into ``docs/PERFORMANCE.md`` together with the
machine they were measured on.

``--vectors N`` skips the model and measures only the LanceDB side with N random vectors:
the seconds to write them in chunks of 256 (as the indexer does), to compact the table, and the
milliseconds of one 50-nearest-neighbour search with the time pre-filter.

Usage::

    uv run python scripts/bench_retrieval.py --days 60
    uv run python scripts/bench_retrieval.py --days 30 --model BAAI/bge-m3 --device cpu
    uv run python scripts/bench_retrieval.py --vectors 100000
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

QUESTIONS = ("你今天吃饭了吗", "在干嘛呢", "晚上想去哪里玩", "今天好累啊", "周末有空吗")


def peak_rss_mb() -> float:
    """Peak resident memory in MB (0.0 if the platform cannot say)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return counters.PeakWorkingSetSize / (1024 * 1024)
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def run_benchmark(
    workdir: Path, days: int, embedder: Any, queries: int = 10, seed: int = 11
) -> dict[str, Any]:
    """Build a synthetic library of ``days`` days with ``embedder`` and time it."""
    os.environ["TWIN_HOME"] = str(workdir / "home")
    os.environ["TWIN_SECRETS_DIR"] = str(workdir / "secrets")
    os.environ["TWIN_KEYRING_BACKEND"] = "file"
    (workdir / "home").mkdir(parents=True, exist_ok=True)

    from tests.support.synth_chat import ChatSpec, build_chat

    from twin.config.loader import load_settings
    from twin.retrieval.indexer import collect_stats, run_index
    from twin.retrieval.query import ExampleQuery, ExampleRetriever, QueryTurn
    from twin.services import build_services
    from twin.storage import migrate

    settings = load_settings(None, {"paths": {"data_dir": str(workdir / "data")}})
    migrate.upgrade(Path(settings.paths.data_dir) / "twin.db")
    services = build_services(settings, root=workdir / "home")
    try:
        truth = build_chat(services, ChatSpec(days=days, seed=seed))
        started = time.perf_counter()
        info = embedder.info  # loads the model (and downloads it the first time)
        loaded = time.perf_counter()
        result = run_index(services, mode="rebuild", embedder=embedder)
        built = time.perf_counter()
        stats = collect_stats(services)
        retriever = ExampleRetriever(services, embedder)
        rank_times = []
        for index in range(queries):
            question = QUESTIONS[index % len(QUESTIONS)]
            query = ExampleQuery([QueryTurn(False, question)], 12 * 60.0, "workday", None, 8)
            t0 = time.perf_counter()
            retriever.rank(query)
            rank_times.append(time.perf_counter() - t0)
        encode_seconds = result.seconds
        return {
            "model": info.model,
            "device": info.device,
            "dimension": info.dimension,
            "messages": truth.messages,
            "windows": stats.windows,
            "encoded": result.encoded,
            "held_out": stats.held_out,
            "model_load_seconds": loaded - started,
            "index_seconds": built - loaded,
            "encode_seconds": encode_seconds,
            "seconds_per_1000": result.per_thousand_s,
            "windows_per_second": result.encoded / encode_seconds if encode_seconds else None,
            "query_ms_mean": 1000 * sum(rank_times) / len(rank_times),
            "query_ms_first": 1000 * rank_times[0],
            "peak_rss_mb": peak_rss_mb(),
            "python": platform.python_version(),
            "machine": f"{platform.system()} {platform.machine()}, {os.cpu_count()} logical cores",
        }
    finally:
        services.close()


def bench_vector_table(
    workdir: Path, count: int, dimension: int = 512, chunk: int = 256, searches: int = 5
) -> dict[str, Any]:
    """Write ``count`` random unit vectors to a window table and time writing and searching."""
    import numpy as np

    from twin.retrieval.vector_store import VectorStore
    from twin.storage.vector_schema import WINDOW_SCHEMA

    rng = np.random.default_rng(1)
    table = VectorStore(workdir / "vectors").table(WINDOW_SCHEMA)
    started = time.perf_counter()
    for start in range(0, count, chunk):
        vectors = rng.normal(size=(min(chunk, count - start), dimension)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        rows = [
            {
                "window_id": f"w{start + i:022d}",
                "vector": vectors[i].tolist(),
                "reply_at_utc": 1_700_000_000 + (start + i) * 600,
                "local_slot": (start + i) % 96,
                "day_type": "workday",
            }
            for i in range(len(vectors))
        ]
        table.upsert(rows, dimension=dimension)
    written = time.perf_counter()
    table.optimize()
    optimised = time.perf_counter()
    query = rng.normal(size=dimension).astype(np.float32)
    query /= np.linalg.norm(query)
    bound = 1_700_000_000 + count * 300  # keeps about the older half
    times = []
    for _ in range(searches):
        t0 = time.perf_counter()
        hits = table.search(query, 50, at_or_before=bound)
        times.append(time.perf_counter() - t0)
    return {
        "vectors": table.count(),
        "dimension": dimension,
        "write_seconds": written - started,
        "optimize_seconds": optimised - written,
        "search_ms": [1000 * t for t in times],
        "hits": len(hits),
        "peak_rss_mb": peak_rss_mb(),
        "machine": f"{platform.system()} {platform.machine()}, {os.cpu_count()} logical cores",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--days", type=int, default=40, help="days of synthetic conversation")
    parser.add_argument(
        "--vectors", type=int, help="only time LanceDB with this many random vectors (no model)"
    )
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data" / "models" / "embeddings",
        help="where the model is (downloaded to, if missing)",
    )
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--workdir", type=Path, help="work directory (default: a temporary one)")
    parser.add_argument("--json", type=Path, help="write the result as JSON to this file")
    args = parser.parse_args(argv)

    if args.vectors:
        workdir = args.workdir or Path(tempfile.mkdtemp(prefix="twin-bench-vectors-"))
        workdir.mkdir(parents=True, exist_ok=True)
        measured = bench_vector_table(workdir, args.vectors)
        sys.stdout.write(json.dumps(measured, indent=2) + "\n")
        return 0

    from twin.retrieval.embedder import EmbeddingService, SentenceTransformerBackend

    backend = SentenceTransformerBackend(args.model, args.device, args.cache_dir)
    embedder = EmbeddingService(backend)
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="twin-bench-retrieval-"))
    workdir.mkdir(parents=True, exist_ok=True)
    result = run_benchmark(workdir, args.days, embedder, args.queries)
    sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
