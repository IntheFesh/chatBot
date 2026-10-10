"""The runtime numbers of the non-functional requirements (R-NFR-001, R-NFR-002, R-NFR-003).

A development tool, not a ``twin`` command.  It measures on the machine it runs on - and says which
machine that is - three things the specification gives a limit for:

``engine``  R-NFR-001: the time the application itself needs to produce a reply, without the
            deliberate delays (the quiet window, her typing) and without the model.  The
            **production** reply pipeline runs on a console world with a fast-forward clock and a
            made-up DeepSeek that answers at once; the wall-clock time of every
            ``ReplyPipeline.run`` is taken with ``perf_counter``.  What is measured is the share of
            the engine in the 20 seconds (60 with thinking) that the specification allows for the
            whole generation; the model's own latency is measured by the ``live`` test
            ``tests/integration/test_nfr_live.py``, with the real API.
``idle``    R-NFR-002: the resident memory and the processor use of ``twin run`` doing nothing.
            A real ``twin run`` (a child process, the console channel, an empty installation, the
            system clock) is left alone for ``--idle-seconds`` and its processor time is read twice.
``startup`` R-NFR-003: the time from starting the process to ``application_running`` in its log
            (the interpreter, the imports, the migration check, the assembly, every component's
            start), the vector model being loaded lazily.

Usage::

    uv run python scripts/bench_runtime.py                   # all three, with the defaults
    uv run python scripts/bench_runtime.py --only startup idle --idle-seconds 60
    uv run python scripts/bench_runtime.py --engine-messages 200 --json bench.json

Numbers from this tool are for **this** machine; the Windows computer of the user has its own
(``docs/PENDING_USER_ACTIONS.md``).  The exit code is 1 when a number is above its limit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # `tests.support` lives next to `src`

# R-NFR-001: generation time without the deliberate delay: p95 below 20 s (60 s with thinking)
GENERATION_LIMIT_S = 20.0
GENERATION_LIMIT_THINKING_S = 60.0
# The engine may use a tenth of that; the rest is the model and the network.
ENGINE_SHARE = 0.10
# R-NFR-002
RSS_LIMIT_MB = 1536.0
IDLE_CPU_LIMIT_PERCENT = 2.0
# R-NFR-003
STARTUP_LIMIT_S = 30.0

ONLY = ("engine", "idle", "startup")


# ----------------------------------------------------------------------------- the numbers


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile (0-100) by the nearest-rank rule (0 for no values)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


@dataclass(frozen=True)
class Stats:
    count: int
    mean: float
    p50: float
    p95: float
    maximum: float


def stats_of(values: Sequence[float]) -> Stats:
    if not values:
        return Stats(0, 0.0, 0.0, 0.0, 0.0)
    return Stats(
        len(values),
        sum(values) / len(values),
        percentile(values, 50),
        percentile(values, 95),
        max(values),
    )


@dataclass
class Machine:
    """Where the numbers come from."""

    system: str = field(default_factory=lambda: f"{platform.system()} {platform.release()}")
    cpu: str = field(default_factory=lambda: _cpu_name())
    cores: int = field(default_factory=lambda: os.cpu_count() or 0)
    python: str = field(default_factory=lambda: platform.python_version())
    measured_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


@dataclass
class Result:
    """Everything the tool measured (``None`` for what it was not asked to measure)."""

    machine: Machine = field(default_factory=Machine)
    engine: dict[str, Any] | None = None
    idle: dict[str, Any] | None = None
    startup: dict[str, Any] | None = None
    problems: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- the engine


async def measure_engine(messages: int, home: Path) -> dict[str, Any]:
    """Wall-clock time of ``ReplyPipeline.run`` for ``messages`` replies (see the description)."""
    from datetime import timedelta

    from tests.support.life_env import life_environment
    from tests.support.proactive_world import opening_curve, proactive_model

    from twin.engine.pipeline import ReplyPipeline

    durations: list[float] = []
    original = ReplyPipeline.run

    async def timed(self: ReplyPipeline, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return await original(self, *args, **kwargs)
        finally:
            durations.append(time.perf_counter() - started)

    ReplyPipeline.run = timed  # type: ignore[method-assign]
    try:
        start = datetime(2026, 10, 9, 16, 0, tzinfo=UTC)  # 11:00 in Chicago, she is up
        async with life_environment(
            home, start=start, model=proactive_model(opening_curve(base=0.02))
        ) as env:
            world = env.world
            world.deepseek.book.default = ("嗯嗯",)
            for number in range(messages):
                world.deepseek.book.queue.append(f"好呀{number % 7}")
                await world.say(f"这是第{number}条消息，今天天气怎么样")
                await world.run_until_idle()
                if world.local_text(world.now) > "2026-10-09 22:00:00":  # keep her awake
                    world.clock.set_time(world.now + timedelta(hours=14))
    finally:
        ReplyPipeline.run = original  # type: ignore[method-assign]
    result = stats_of(durations)
    return {
        "replies": result.count,
        "mean_s": result.mean,
        "p50_s": result.p50,
        "p95_s": result.p95,
        "max_s": result.maximum,
        "budget_s": GENERATION_LIMIT_S * ENGINE_SHARE,
        "limit_s": GENERATION_LIMIT_S,
        "limit_thinking_s": GENERATION_LIMIT_THINKING_S,
    }


# --------------------------------------------------------------- the process: idle, startup


def _environment(home: Path) -> dict[str, str]:
    environment = {k: v for k, v in os.environ.items() if not k.startswith("TWIN_")}
    environment.update(
        {
            "TWIN_HOME": str(home),
            "TWIN_SECRETS_DIR": str(home / "secrets"),
            "TWIN_KEYRING_BACKEND": "file",
            "PYTHONUTF8": "1",
            "NO_PROXY": "*",
        }
    )
    return environment


def _twin(home: Path, *args: str, stdin: str | None = None) -> None:
    done = subprocess.run(
        [sys.executable, "-m", "twin", "--set", "channel.kind=console", *args],
        input=stdin,
        capture_output=True,
        text=True,
        env=_environment(home),
        cwd=home,
        timeout=300,
        check=False,
    )
    if done.returncode != 0:
        raise RuntimeError(f"twin {' '.join(args)} failed: {done.stderr.strip()[-300:]}")


def _running(log_path: Path) -> bool:
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    return any('"application_running"' in line for line in lines)


def measure_process(idle_seconds: float, settle_seconds: float, home: Path) -> dict[str, Any]:
    """Start ``twin run`` and read the time to ready, then its memory and processor use at rest."""
    from tests.support.process_metrics import child_cpu_seconds, child_rss_bytes, megabytes

    _twin(home, "db", "upgrade")
    _twin(home, "secrets", "set", "deepseek_api_key", "--stdin", stdin="sk-bench-not-a-key\n")
    log_path = home / "data" / "logs" / "twin.log"
    started = time.perf_counter()
    child = subprocess.Popen(
        [sys.executable, "-m", "twin", "--set", "channel.kind=console", "run"],
        stdin=subprocess.PIPE,  # (an open pipe: the console channel stops when its input ends)
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_environment(home),
        cwd=home,
    )
    try:
        deadline = started + STARTUP_LIMIT_S * 4
        while not _running(log_path):
            if child.poll() is not None:
                raise RuntimeError(f"twin run ended with code {child.returncode} while starting")
            if time.perf_counter() > deadline:
                raise RuntimeError("twin run was not ready after two minutes")
            time.sleep(0.05)
        ready = time.perf_counter() - started
        time.sleep(settle_seconds)  # whatever the start left to do
        first = child_cpu_seconds(child.pid)
        began = time.perf_counter()
        time.sleep(idle_seconds)
        second = child_cpu_seconds(child.pid)
        window = time.perf_counter() - began
        rss = child_rss_bytes(child.pid)
    finally:
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    cpu_percent = (
        100.0 * (second - first) / window if first is not None and second is not None else None
    )
    return {
        "startup_s": ready,
        "startup_limit_s": STARTUP_LIMIT_S,
        "idle_seconds": window,
        "idle_cpu_percent": cpu_percent,
        "idle_cpu_limit_percent": IDLE_CPU_LIMIT_PERCENT,
        "rss_mb": megabytes(rss) if rss is not None else None,
        "rss_limit_mb": RSS_LIMIT_MB,
    }


# -------------------------------------------------------------------------------- report


def judge(result: Result) -> None:
    """Put what is above a limit into ``result.problems``."""
    engine = result.engine
    if engine is not None and engine["p95_s"] > engine["budget_s"]:
        result.problems.append(
            f"engine p95 {engine['p95_s']:.2f} s is above its share ({engine['budget_s']:.1f} s) "
            f"of the {GENERATION_LIMIT_S:.0f} s of R-NFR-001"
        )
    for part in (result.idle, result.startup):
        if part is None:
            continue
        if part["startup_s"] >= part["startup_limit_s"]:
            result.problems.append(
                f"startup {part['startup_s']:.1f} s is not below {part['startup_limit_s']:.0f} s"
            )
        if part["idle_cpu_percent"] is not None and (
            part["idle_cpu_percent"] >= part["idle_cpu_limit_percent"]
        ):
            result.problems.append(f"idle processor use {part['idle_cpu_percent']:.2f} % >= 2 %")
        if part["rss_mb"] is not None and part["rss_mb"] >= part["rss_limit_mb"]:
            result.problems.append(f"resident memory {part['rss_mb']:.0f} MB >= 1.5 GB")


def render(result: Result) -> str:
    machine = result.machine
    lines = [
        f"{machine.system}, {machine.cpu}, {machine.cores} logical cores, Python {machine.python}, "
        f"measured {machine.measured_at}",
        "",
    ]
    if result.engine is not None:
        e = result.engine
        lines += [
            "R-NFR-001  engine time per reply (no model, no deliberate delay):",
            f"  {e['replies']} replies: mean {e['mean_s'] * 1000:.0f} ms, "
            f"p50 {e['p50_s'] * 1000:.0f} ms, p95 {e['p95_s'] * 1000:.0f} ms, "
            f"max {e['max_s'] * 1000:.0f} ms "
            f"(its share of the limit: {e['budget_s']:.1f} s of {e['limit_s']:.0f} s / "
            f"{e['limit_thinking_s']:.0f} s with thinking)",
        ]
    if result.startup is not None:
        p = result.startup
        lines += [
            "R-NFR-003  start to ready (twin run, console channel, empty installation):",
            f"  {p['startup_s']:.1f} s (limit {p['startup_limit_s']:.0f} s)",
        ]
    if result.idle is not None:
        p = result.idle
        cpu = "n/a" if p["idle_cpu_percent"] is None else f"{p['idle_cpu_percent']:.2f} %"
        rss = "n/a" if p["rss_mb"] is None else f"{p['rss_mb']:.0f} MB"
        lines += [
            f"R-NFR-002  at rest for {p['idle_seconds']:.0f} s:",
            f"  processor {cpu} of one core (limit {p['idle_cpu_limit_percent']:.0f} %), "
            f"resident memory {rss} (limit {p['rss_limit_mb']:,.0f} MB)",
        ]
    lines.append("")
    lines += [f"PROBLEM: {problem}" for problem in result.problems]
    lines.append("result: " + ("OK" if not result.problems else "ABOVE A LIMIT"))
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--only", nargs="+", choices=ONLY, default=list(ONLY))
    parser.add_argument("--engine-messages", type=int, default=100, help="replies to time (100)")
    parser.add_argument("--idle-seconds", type=float, default=30.0, help="the rest window (30)")
    parser.add_argument("--settle-seconds", type=float, default=10.0, help="before the window (10)")
    parser.add_argument("--json", type=Path, default=None, help="write the numbers here")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.engine_messages < 1 or args.idle_seconds <= 0 or args.settle_seconds < 0:
        parser.error("the sizes must be positive")
    return args


def run(args: argparse.Namespace) -> Result:
    result = Result()
    with tempfile.TemporaryDirectory(prefix="twin-bench-") as folder:
        root = Path(folder)
        if "engine" in args.only:
            from tests.support.life_env import isolate

            home = root / "engine"
            home.mkdir()
            isolate(home)
            result.engine = asyncio.run(measure_engine(args.engine_messages, home))
        if "startup" in args.only or "idle" in args.only:
            home = root / "process"
            home.mkdir()
            measured = measure_process(args.idle_seconds, args.settle_seconds, home)
            if "startup" in args.only:
                result.startup = {k: v for k, v in measured.items() if k.startswith("startup")}
                result.startup |= {"idle_cpu_percent": None, "idle_cpu_limit_percent": 2.0}
                result.startup |= {"rss_mb": None, "rss_limit_mb": RSS_LIMIT_MB}
            if "idle" in args.only:
                result.idle = measured
    judge(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(render(result))
    if args.json is not None:
        args.json.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n", "utf-8"
        )
    return 0 if not result.problems else 1


if __name__ == "__main__":
    sys.exit(main())
