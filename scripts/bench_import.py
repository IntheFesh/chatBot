"""Import benchmark (R-IMP-013): messages per second and peak memory of the structured import.

Generates a synthetic export of ``--messages`` messages with ``tests/fixtures/synth_export.py``
(streamed to disk, so a million messages need no memory), then imports it in a *fresh* child
process so that the child's peak memory is the import's own:

* wall-clock time and messages per second of the import phases (messages, media, finalize,
  hooks); no sticker download and no model call takes part;
* peak resident memory of the child process (``ru_maxrss`` / ``PeakWorkingSetSize``), and with
  ``--tracemalloc`` also the peak of Python's own allocations (this slows the import down, so
  do not read the speed of such a run);
* the database size.

Target of the specification: 1,000,000 messages in less than 30 minutes with a peak below
500 MB.  The script prints the verdict for both.

Usage::

    uv run python scripts/bench_import.py --messages 200000
    uv run python scripts/bench_import.py --messages 1000000 --workdir D:/bench
    uv run python scripts/bench_import.py --messages 100000 --tracemalloc
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

TARGET_SECONDS = 30 * 60
TARGET_PEAK_MB = 500
DEFAULT_BATCH = 2000
REPORT_EVERY = 100_000
MIN_JUDGED_MESSAGES = 100_000


def peak_rss_bytes() -> int:
    """Peak resident set size of this process in bytes (0 if the platform cannot say)."""
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
        kernel = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
        psapi = ctypes.WinDLL("psapi")  # type: ignore[attr-defined]
        # Without these declarations ctypes passes the pseudo handle (-1) as a 32-bit int, which
        # is not the all-ones 64-bit handle on a 64-bit Windows, and the call fails.
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.GetCurrentProcess.argtypes = []
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        handle = kernel.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
        return 0
    if sys.platform.startswith("linux"):
        # ``ru_maxrss`` survives fork + exec on Linux: a child started by a big parent (a pytest
        # process that has already run thousands of tests) would report the parent's high-water
        # mark.  ``VmHWM`` belongs to the current address space, which exec replaces, so it is
        # the peak of this process alone (D-493).
        try:
            for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass  # no procfs: fall back to getrusage below
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def mb(value: float) -> str:
    return f"{value / (1024 * 1024):,.1f} MB"


# ----------------------------------------------------------------------- child


def run_import_child(
    workdir: Path, export: Path, username: str, batch_size: int, use_tracemalloc: bool
) -> dict[str, Any]:
    """Import ``export`` into a fresh database below ``workdir`` and describe the run."""
    os.environ["TWIN_HOME"] = str(workdir / "home")
    os.environ["TWIN_SECRETS_DIR"] = str(workdir / "secrets")
    os.environ["TWIN_KEYRING_BACKEND"] = "file"
    (workdir / "home").mkdir(parents=True, exist_ok=True)

    from twin.config.loader import load_settings
    from twin.ingest.importer import BatchEvent, ImportRunner, prepare_import
    from twin.services import build_services
    from twin.storage import migrate

    settings = load_settings(None, {"paths": {"data_dir": str(workdir / "data")}})
    db_path = Path(settings.paths.data_dir) / "twin.db"
    migrate.upgrade(db_path)
    services = build_services(settings, root=workdir / "home")
    if use_tracemalloc:
        tracemalloc.start()
    prepared = prepare_import(services, export, target_username=username)
    started = time.perf_counter()
    last_report = [started, 0]

    def progress(event: BatchEvent) -> None:
        if event.processed - last_report[1] >= REPORT_EVERY:
            now = time.perf_counter()
            rate = (event.processed - last_report[1]) / (now - last_report[0])
            sys.stderr.write(
                f"  {event.processed:>9,} messages  {rate:>8,.0f}/s  "
                f"peak so far {mb(peak_rss_bytes())}\n"
            )
            sys.stderr.flush()
            last_report[0], last_report[1] = now, event.processed

    outcome = ImportRunner(services, batch_size=batch_size, on_batch=progress).run(prepared.run_id)
    elapsed = time.perf_counter() - started
    traced_peak = tracemalloc.get_traced_memory()[1] if use_tracemalloc else None
    services.close()
    wal = db_path.with_name(db_path.name + "-wal")
    return {
        "status": outcome.status,
        "processed": outcome.run.processed,
        "inserted": outcome.run.inserted,
        "invalid": outcome.run.invalid,
        "seconds": elapsed,
        "messages_per_s": outcome.run.processed / elapsed if elapsed else 0.0,
        "peak_rss_bytes": peak_rss_bytes(),
        "tracemalloc_peak_bytes": traced_peak,
        "db_bytes": db_path.stat().st_size + (wal.stat().st_size if wal.exists() else 0),
        "batch_size": batch_size,
    }


# ---------------------------------------------------------------------- parent


def generate(workdir: Path, messages: int, seed: int) -> tuple[Path, str, float]:
    from tests.fixtures.synth_export import SynthOptions, build_export

    options = SynthOptions(
        target_messages=messages,
        other_conversations=0,
        include_group=False,
        media=False,
        keep_texts=False,
        keep_ids=False,
        seed=seed,
        average_gap_s=30.0,
    )
    started = time.perf_counter()
    export = build_export(workdir / "export", options)
    return export.root, export.target_username, time.perf_counter() - started


def verdicts(result: dict[str, Any], projected_seconds: float) -> tuple[str, str]:
    """``(time verdict, memory verdict)`` against the R-IMP-013 targets.

    The time is judged only for an untraced run of at least ``MIN_JUDGED_MESSAGES`` messages:
    a small run is dominated by start-up costs and a traced run is several times slower, so
    projecting either to a million messages would say nothing.
    """
    if result["tracemalloc_peak_bytes"] is not None:
        time_verdict = "not judged (traced run)"
    elif result["processed"] < MIN_JUDGED_MESSAGES:
        time_verdict = f"not judged (fewer than {MIN_JUDGED_MESSAGES:,} messages)"
    else:
        time_verdict = "PASS" if projected_seconds < TARGET_SECONDS else "FAIL"
    memory_ok = result["peak_rss_bytes"] < TARGET_PEAK_MB * 1024 * 1024
    return time_verdict, "PASS" if memory_ok else "FAIL"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--messages", type=int, default=200_000, help="messages in the export")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--workdir", type=Path, help="work directory (default: a temporary one)")
    parser.add_argument(
        "--keep", action="store_true", help="keep the generated export and database"
    )
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--tracemalloc", action="store_true", help="also trace Python allocations")
    parser.add_argument("--json", type=Path, help="write the result as JSON to this file")
    parser.add_argument(
        "--child", nargs=3, metavar=("EXPORT", "USERNAME", "RESULT"), help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)

    if args.child:  # the fresh process that does the measured import
        export, username, result_path = args.child
        result = run_import_child(
            args.workdir, Path(export), username, args.batch_size, args.tracemalloc
        )
        Path(result_path).write_text(json.dumps(result), encoding="utf-8")
        return 0

    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="twin-bench-"))
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        sys.stdout.write(f"generating {args.messages:,} synthetic messages in {workdir} ...\n")
        sys.stdout.flush()
        export, username, generated_in = generate(workdir, args.messages, args.seed)
        size = (next((export / "conversations").iterdir()) / "messages.json").stat().st_size
        sys.stdout.write(f"  generated in {generated_in:,.1f} s, messages.json {mb(size)}\n")
        sys.stdout.write("importing in a fresh process ...\n")
        sys.stdout.flush()
        result_path = workdir / "result.json"
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--workdir", str(workdir), "--batch-size", str(args.batch_size),
            "--child", str(export), username, str(result_path),
        ]  # fmt: skip
        if args.tracemalloc:
            command.append("--tracemalloc")
        done = subprocess.run(command, check=False)
        if done.returncode != 0 or not result_path.exists():
            sys.stderr.write("the import process failed\n")
            return 1
        result = json.loads(result_path.read_text(encoding="utf-8"))
    finally:
        if not args.keep and args.workdir is None:
            shutil.rmtree(workdir, ignore_errors=True)

    seconds, peak = result["seconds"], result["peak_rss_bytes"]
    projected = seconds * (1_000_000 / max(1, result["processed"]))  # seconds
    lines = [
        "",
        f"messages imported      {result['processed']:,} "
        f"({result['status']}, {result['invalid']} invalid)",
        f"import time            {seconds:,.1f} s",
        f"speed                  {result['messages_per_s']:,.0f} messages/s",
        f"peak memory (RSS)      {mb(peak)}",
        f"database size          {mb(result['db_bytes'])}",
        f"batch size             {result['batch_size']}",
    ]
    if result["tracemalloc_peak_bytes"] is not None:
        lines.append(
            f"python allocations     {mb(result['tracemalloc_peak_bytes'])} peak (traced run)"
        )
    lines.append(f"projected for 1,000,000 messages: {projected / 60:,.1f} min")
    verdict_time, verdict_memory = verdicts(result, projected)
    lines.append(f"R-IMP-013 time   (< 30 min for 1M)   {verdict_time}")
    lines.append(f"R-IMP-013 memory (< {TARGET_PEAK_MB} MB peak)      {verdict_memory}")
    sys.stdout.write("\n".join(lines) + "\n")
    if args.json:
        result["generated_seconds"] = generated_in
        result["projected_minutes_for_1m"] = projected / 60
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if "FAIL" not in (verdict_time, verdict_memory) else 2


if __name__ == "__main__":
    raise SystemExit(main())
