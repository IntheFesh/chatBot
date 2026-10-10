#!/usr/bin/env python3
"""Background jobs on the instance that survive a dropped SSH connection (standard library only).

``twin train remote`` starts every long step through this tool, so the step keeps running when
the connection breaks, and the next connection finds it again by name and continues to read its
log from where it stopped.

    remote_job.py start  --dir JOBS --name NAME [--cwd DIR] [--env K=V ...] -- COMMAND ...
    remote_job.py status --dir JOBS --name NAME
    remote_job.py tail   --dir JOBS --name NAME --offset N [--limit BYTES]
    remote_job.py stop   --dir JOBS --name NAME

Every command prints one JSON object.  A job lives in ``JOBS/NAME/``: ``log`` (output of the
command), ``exit_code`` (written when it ends), ``heartbeat`` (touched while the supervisor
runs), ``child_pid``.  ``start`` on a name whose job still runs does nothing and reports
``running``; on a finished job it starts a new run and truncates the log.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HEARTBEAT_SECONDS = 2.0
HEARTBEAT_STALE_SECONDS = 20.0
DEFAULT_TAIL_LIMIT = 1 << 18


def job_dir(root: Path, name: str) -> Path:
    if not name or any(part in name for part in ("/", "\\", "..")):
        raise ValueError(f"invalid job name {name!r}")
    return root / name


def read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def status(directory: Path) -> dict[str, object]:
    if not directory.is_dir():
        return {"state": "none", "exit_code": None, "log_size": 0}
    log = directory / "log"
    size = log.stat().st_size if log.exists() else 0
    exit_code = read_int(directory / "exit_code")
    if exit_code is not None:
        return {"state": "exited", "exit_code": exit_code, "log_size": size}
    heartbeat = directory / "heartbeat"
    try:
        age = time.time() - heartbeat.stat().st_mtime
    except OSError:
        age = HEARTBEAT_STALE_SECONDS + 1
    if age > HEARTBEAT_STALE_SECONDS:
        return {"state": "lost", "exit_code": None, "log_size": size}
    return {"state": "running", "exit_code": None, "log_size": size}


def start(
    root: Path, name: str, command: list[str], cwd: str | None, env: list[str]
) -> dict[str, object]:
    directory = job_dir(root, name)
    current = status(directory)
    if current["state"] == "running":
        return current
    directory.mkdir(parents=True, exist_ok=True)
    for stale in ("exit_code", "child_pid", "heartbeat"):
        (directory / stale).unlink(missing_ok=True)
    (directory / "log").write_bytes(b"")
    (directory / "job.json").write_text(
        json.dumps({"command": command, "cwd": cwd, "env": env}), encoding="utf-8"
    )
    (directory / "heartbeat").write_text("starting", encoding="utf-8")
    options: dict[str, object] = {"start_new_session": True}
    if sys.platform == "win32":
        options = {"creationflags": 0x00000008 | 0x00000200}  # DETACHED_PROCESS | NEW_PROCESS_GROUP
    subprocess.Popen(  # noqa: S603 - the interpreter is this one, the arguments are ours
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "_supervise",
            "--dir",
            str(root),
            "--name",
            name,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        **options,  # type: ignore[arg-type]
    )
    return {"state": "started", "exit_code": None, "log_size": 0}


def supervise(root: Path, name: str) -> int:
    directory = job_dir(root, name)
    spec = json.loads((directory / "job.json").read_text(encoding="utf-8"))
    environment = os.environ.copy()
    for item in spec["env"]:
        key, _, value = item.partition("=")
        environment[key] = value
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(HEARTBEAT_SECONDS):
            (directory / "heartbeat").write_text(str(time.time()), encoding="utf-8")

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    code = 127
    try:
        with (directory / "log").open("ab") as log:
            try:
                child = subprocess.Popen(  # noqa: S603
                    spec["command"],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=spec["cwd"],
                    env=environment,
                )
            except OSError as exc:
                log.write(f"could not start the command: {exc}\n".encode())
            else:
                (directory / "child_pid").write_text(str(child.pid), encoding="utf-8")
                code = child.wait()
    finally:
        stop.set()
        thread.join()
        partial = directory / "exit_code.part"
        partial.write_text(str(code), encoding="utf-8")
        os.replace(partial, directory / "exit_code")
    return 0


def tail(directory: Path, offset: int, limit: int) -> dict[str, object]:
    log = directory / "log"
    if not log.exists():
        return {"offset": offset, "data": "", "eof": True}
    with log.open("rb") as stream:
        stream.seek(offset)
        chunk = stream.read(limit)
        end_of_file = stream.read(1) == b""
    for cut in range(4):
        try:
            text = chunk[: len(chunk) - cut].decode("utf-8")
            consumed = len(chunk) - cut
            break
        except UnicodeDecodeError:
            continue
    else:
        text, consumed = chunk.decode("utf-8", errors="replace"), len(chunk)
    return {
        "offset": offset + consumed,
        "data": text,
        "eof": end_of_file and consumed == len(chunk),
    }


def stop(directory: Path) -> dict[str, object]:
    pid = read_int(directory / "child_pid")
    if pid is not None and status(directory)["state"] == "running":
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)
    return status(directory)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "status", "tail", "stop", "_supervise"])
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--cwd")
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=DEFAULT_TAIL_LIMIT)
    parser.add_argument("command", nargs="*")
    args = parser.parse_args(argv)

    if args.action == "_supervise":
        return supervise(args.dir, args.name)
    if args.action == "start":
        if not args.command:
            sys.stderr.write("error: start needs a command after --\n")
            return 2
        result = start(args.dir, args.name, args.command, args.cwd, args.env)
    elif args.action == "status":
        result = status(job_dir(args.dir, args.name))
    elif args.action == "tail":
        result = tail(job_dir(args.dir, args.name), args.offset, args.limit)
    else:
        result = stop(job_dir(args.dir, args.name))
    sys.stdout.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
