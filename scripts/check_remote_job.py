#!/usr/bin/env python3
"""Does the job tool of the instance work under THIS interpreter?  (R-TRN-010, D-630)

Standard library only.

``training/autodl/tools/remote_job.py`` runs on the AutoDL instance with whatever ``python3`` that
image has.  LLaMA-Factory 0.9.5 needs Python 3.11 or newer (``TWIN_PYTHON_MIN`` in
``training/autodl/versions.env``), so the instance may well have 3.11 or 3.12.0-3.12.6, older than
the 3.12 the tests of this repository run under.  ``argparse`` before 3.12.7 treats the ``--`` in
``start ... -- COMMAND`` differently, which once made every remote step fail with "unrecognized
arguments".  Nothing but running the tool under such an interpreter shows that, so:

* every Python file the instance runs (the tools, ``bundle_crypto`` and the two ``twin`` modules in
  the package's ``pylib``) must parse with the syntax of the minimum version, and
* ``start``, ``status``, ``tail`` and ``stop`` are run for real with ``sys.executable``: a command
  whose arguments look like options, a command that exits with a code, a missing command, and (on
  POSIX) a job that is stopped.

    python scripts/check_remote_job.py [--root DIR]

Exit code 0 when everything works, 1 otherwise (every problem is printed to standard error).  CI
runs it under Python 3.11 and 3.12.3 (job ``remote-tools``); ``tests/unit/test_autodl_tools.py``
runs it under every interpreter of the machine.  Nothing here imports the ``twin`` package, so it
needs no virtual environment.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent.parent
PYLIB_MODULES = ("lf_template", "parity_check")
"""The ``twin`` modules the instance imports (``twin.training.bundle.PYLIB_MODULES``)."""
TOOL_MODULES = ("bundle_crypto",)
"""``twin`` modules uploaded as tools of the instance (``autodl/tools/bundle_crypto.py``)."""
WAIT_SECONDS = 30.0
POSIX = os.name == "posix"


class CheckFailedError(Exception):
    """One scenario did not behave as the instance needs it to."""


def minimum_python(root: Path) -> tuple[int, int]:
    """``TWIN_PYTHON_MIN`` of ``versions.env`` as ``(major, minor)``."""
    text = (root / "training" / "autodl" / "versions.env").read_text(encoding="utf-8")
    found = re.search(r"^TWIN_PYTHON_MIN=(\d+)\.(\d+)\s*$", text, re.MULTILINE)
    if found is None:
        raise CheckFailedError("versions.env has no TWIN_PYTHON_MIN=<major>.<minor> line")
    return int(found.group(1)), int(found.group(2))


def instance_files(root: Path) -> list[Path]:
    tools = sorted((root / "training" / "autodl" / "tools").glob("*.py"))
    package = root / "src" / "twin" / "training"
    return [*tools, *(package / f"{name}.py" for name in (*TOOL_MODULES, *PYLIB_MODULES))]


def check_syntax(root: Path, floor: tuple[int, int]) -> list[str]:
    """Every file the instance runs parses with the grammar of the minimum Python."""
    grammar = min(floor, sys.version_info[:2])  # a newer minimum than this interpreter: its own
    problems = []
    for path in instance_files(root):
        label = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=label, feature_version=grammar)
        except OSError as exc:
            problems.append(f"{label}: cannot be read ({exc})")
        except SyntaxError as exc:
            problems.append(
                f"{label}:{exc.lineno}: {exc.msg} (the instance may run Python "
                f"{floor[0]}.{floor[1]})"
            )
    return problems


def last_line(text: str) -> str:
    """The last line of an error output (the ``error: ...`` line of argparse, a traceback)."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1].strip()[:300] if lines else "(no output)"


class Tool:
    """``remote_job.py`` run as the orchestrator runs it: a separate process, JSON on stdout."""

    def __init__(self, script: Path, jobs: Path) -> None:
        self.script = script
        self.jobs = jobs

    def raw(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(self.script), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=60,
        )

    def call(self, action: str, name: str, *args: str) -> dict[str, object]:
        done = self.raw(action, "--dir", str(self.jobs), "--name", name, *args)
        if done.returncode != 0:
            raise CheckFailedError(
                f"remote_job.py {action} exited with {done.returncode}: {last_line(done.stderr)}"
            )
        try:
            answer = json.loads(done.stdout)
        except json.JSONDecodeError:
            raise CheckFailedError(
                f"remote_job.py {action} printed no JSON: {done.stdout!r}"
            ) from None
        if not isinstance(answer, dict):
            raise CheckFailedError(f"remote_job.py {action} printed {answer!r}, not an object")
        return answer

    def state(self, name: str) -> str:
        return str(self.call("status", name)["state"])

    def log(self, name: str) -> str:
        return str(self.call("tail", name, "--offset", "0")["data"])


def wait_for(condition: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + WAIT_SECONDS
    while not condition():
        if time.monotonic() > deadline:
            raise CheckFailedError(f"{what}: not reached within {WAIT_SECONDS:.0f} seconds")
        time.sleep(0.05)


def run_to_the_end(tool: Tool, name: str, *args: str) -> None:
    started = tool.call("start", name, *args)
    if started["state"] not in ("started", "running", "exited"):
        raise CheckFailedError(f"start reported {started!r}")
    wait_for(lambda: tool.state(name) in ("exited", "lost"), f"job {name} to end")


def scenario_exit_code_log_environment(tool: Tool, work: Path) -> None:
    """The exit code, both output streams, the environment and the directory reach the job."""
    code = (
        "import os, sys; print('hello', os.environ['TWIN_CHECK'], os.path.realpath(os.getcwd())); "
        "print('oops', file=sys.stderr); sys.exit(3)"
    )
    run_to_the_end(
        tool,
        "exit-code",
        "--cwd",
        str(work),
        "--env",
        "TWIN_CHECK=a=b",
        "--",
        sys.executable,
        "-c",
        code,
    )
    done = tool.call("status", "exit-code")
    if done["exit_code"] != 3:
        raise CheckFailedError(f"the exit code of the job was {done['exit_code']!r}, not 3")
    log = tool.log("exit-code")
    if f"hello a=b {os.path.realpath(work)}" not in log or "oops" not in log:
        raise CheckFailedError(f"the log of the job is {log!r}")
    if tool.call("tail", "exit-code", "--offset", "0")["eof"] is not True:
        raise CheckFailedError("tail did not report the end of the finished log")


def scenario_the_command_may_look_like_options(tool: Tool, work: Path) -> None:
    """Everything after the first ``--`` is the command, whatever it looks like."""
    words = ["--name", "x", "--dir", "y", "--", "--env", "K=V", "start"]
    code = "import sys; print(repr(sys.argv[1:]))"
    run_to_the_end(tool, "options", "--cwd", str(work), "--", sys.executable, "-c", code, *words)
    expected = repr(words)
    if tool.log("options").strip() != expected:
        raise CheckFailedError(f"the command got {tool.log('options').strip()!r}, not {expected}")
    if tool.call("status", "options")["exit_code"] != 0:
        raise CheckFailedError("the job with option-like arguments did not end with code 0")


def scenario_start_needs_a_command(tool: Tool, work: Path) -> None:
    for tail in ([], ["--"]):
        done = tool.raw("start", "--dir", str(tool.jobs), "--name", "no-command", *tail)
        if done.returncode != 2 or "needs a command" not in done.stderr:
            raise CheckFailedError(
                f"start without a command ended with {done.returncode}: {last_line(done.stderr)}"
            )


def scenario_stop(tool: Tool, work: Path) -> None:
    """A running job is not started twice, and ``stop`` ends it."""
    run_args = ("--cwd", str(work), "--", sys.executable, "-c", "import time; time.sleep(120)")
    try:
        tool.call("start", "stop", *run_args)
        wait_for(lambda: (tool.jobs / "stop" / "child_pid").exists(), "the child process to start")
        if tool.state("stop") != "running":
            raise CheckFailedError("a job that has just started is not reported as running")
        again = tool.call("start", "stop", *run_args)
        if again["state"] != "running":
            raise CheckFailedError(f"starting a running job again reported {again!r}")
    finally:
        tool.raw("stop", "--dir", str(tool.jobs), "--name", "stop")  # never leave it running
    wait_for(lambda: tool.state("stop") in ("exited", "lost"), "the stopped job to end")
    if tool.call("status", "stop")["exit_code"] == 0:
        raise CheckFailedError("a stopped job ended with code 0")


def check_tool(root: Path) -> list[str]:
    script = root / "training" / "autodl" / "tools" / "remote_job.py"
    scenarios = [
        scenario_exit_code_log_environment,
        scenario_the_command_may_look_like_options,
        scenario_start_needs_a_command,
    ]
    if POSIX:  # SIGTERM semantics: the job runs on the Linux instance
        scenarios.append(scenario_stop)
    problems = []
    with tempfile.TemporaryDirectory(
        prefix="remote-job-check-", ignore_cleanup_errors=True
    ) as scratch:
        work = Path(scratch) / "work"
        work.mkdir()
        tool = Tool(script, Path(scratch) / "jobs")
        for scenario in scenarios:
            try:
                scenario(tool, work)
            except CheckFailedError as exc:
                problems.append(f"{scenario.__name__}: {exc}")
            except (OSError, subprocess.SubprocessError) as exc:
                problems.append(f"{scenario.__name__}: {type(exc).__name__}: {exc}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="the repository to check")
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    version = ".".join(str(part) for part in sys.version_info[:3])
    try:
        floor = minimum_python(root)
    except (OSError, CheckFailedError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    problems = []
    if sys.version_info[:2] < floor:
        problems.append(f"Python {version} is older than the minimum {floor[0]}.{floor[1]}")
    problems += check_syntax(root, floor)
    problems += check_tool(root)
    if problems:
        sys.stderr.write(f"remote_job.py does NOT work under Python {version}:\n")
        for problem in problems:
            sys.stderr.write(f"  - {problem}\n")
        return 1
    sys.stdout.write(f"remote_job.py works under Python {version}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
