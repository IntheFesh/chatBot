"""The managed llama-server: start, load, crash, hang, restart, stop (R-SRV-002).

The program is ``tests/support/llama_server_sim.py``, started as a real child process with the
very command line the program builds for ``llama-server``; it answers the same HTTP.  Waiting is
for conditions (``wait_until``), never for a length of time.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from tests.support.llama_sim import SIM, free_port, make_model_file, sim_prefix, spec_for
from tests.support.waiting import wait_until
from tests.support.win32 import FakeWin32
from twin.clock import SystemClock
from twin.llm import style_client
from twin.llm.style_client import LlamaCppCompletionClient
from twin.ops.jobobject import ProcessJob
from twin.serving.llamacpp import ServeError, ServerSpec
from twin.serving.server import (
    LlamaServerManager,
    ServerBlocked,
    ServerSnapshot,
    ServerState,
    ServerTimings,
    _creation_flags,
    log_tail,
    rotate_log,
    same_model_file,
    served_model_path,
)

FAST = ServerTimings(
    start_timeout_s=20.0,
    backoff_start_s=0.05,
    backoff_max_s=0.2,
    stable_after_s=60.0,
    health_interval_s=0.1,
    unhealthy_limit=3,
    stop_grace_s=5.0,
)


@pytest.fixture(autouse=True)
def quick_health_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stuck server must be noticed in a fraction of a second, not after five."""
    monkeypatch.setattr(style_client, "HEALTH_TIMEOUT_S", 0.4)


def make_manager(
    tmp_path: Path,
    *sim_options: str,
    timings: ServerTimings = FAST,
    port: int | None = None,
    model: Path | None = None,
    job: ProcessJob | None = None,
    on_ready: object = None,
    changes: list[ServerSnapshot] | None = None,
) -> tuple[LlamaServerManager, LlamaCppCompletionClient, Path]:
    chosen = port or free_port()
    model_file = model or make_model_file(tmp_path / "models")
    spec = spec_for(model_file, chosen, *sim_options)
    client = LlamaCppCompletionClient(f"http://127.0.0.1:{chosen}", clock=SystemClock())
    log_path = tmp_path / "logs" / "llama-server.log"
    manager = LlamaServerManager(
        spec,
        clock=SystemClock(),
        client=client,
        log_path=log_path,
        timings=timings,
        job=job,
        on_ready=on_ready,  # type: ignore[arg-type]
        on_change=(lambda snap: changes.append(snap)) if changes is not None else None,
    )
    return manager, client, log_path


@contextlib.asynccontextmanager
async def running(manager: LlamaServerManager) -> AsyncIterator[asyncio.Task[None]]:
    task = asyncio.create_task(manager.run())
    try:
        yield task
    finally:
        await manager.stop()
        with contextlib.suppress(asyncio.CancelledError, ServeError):
            await asyncio.wait_for(task, 20)


async def until_state(manager: LlamaServerManager, *states: ServerState, limit: float = 20) -> None:
    await wait_until(lambda: manager.state in states, limit_s=limit)


async def test_the_server_is_started_with_the_documented_command_line_and_becomes_ready(
    tmp_path: Path,
) -> None:
    argv_file = tmp_path / "argv.json"
    manager, client, log_path = make_manager(tmp_path, "--sim-argv", str(argv_file))
    async with running(manager):
        await until_state(manager, ServerState.READY)
        assert (await client.health()).ok
        snapshot = manager.snapshot()
        assert snapshot.pid is not None and snapshot.restarts == 0
        assert snapshot.started_at is not None and snapshot.ready_at is not None
    recorded = json.loads(argv_file.read_text(encoding="utf-8"))
    spec = manager.spec
    assert recorded == spec.argv()[3:]  # (python, -I, script) are not the server's own arguments
    assert recorded[recorded.index("-m") + 1] == str(spec.model)
    assert recorded[recorded.index("--host") + 1] == "127.0.0.1"  # this computer only
    assert recorded[recorded.index("--port") + 1] == str(spec.port)
    assert (
        recorded[recorded.index("-c") + 1] == "4096"
        and recorded[recorded.index("-ngl") + 1] == "999"
    )
    assert recorded[recorded.index("--parallel") + 1] == "1"
    assert recorded[recorded.index("--chat-template") + 1] == "chatml" and "--no-webui" in recorded
    # the output of the program is in a log file of its own, not in ours
    await wait_until(lambda: "listening on 127.0.0.1" in log_path.read_text(encoding="utf-8"))
    assert manager.state is ServerState.STOPPED
    assert not (await client.health()).ok  # the process is gone


async def test_a_model_that_is_still_loading_is_not_ready_and_is_told_so(tmp_path: Path) -> None:
    manager, client, _ = make_manager(tmp_path, "--sim-load-s", "1.5")
    async with running(manager):
        await wait_until(lambda: manager.state is ServerState.STARTING)
        loading = await wait_for_health(client, loading=True)
        assert not loading.ok and loading.loading and "loading" in loading.detail
        await until_state(manager, ServerState.READY)
        assert (await client.health()).ok


async def wait_for_health_gone(client: LlamaCppCompletionClient) -> None:
    async def poll() -> None:
        while (await client.health()).ok:  # noqa: ASYNC110 - polls a child process over HTTP
            await asyncio.sleep(0.02)

    await asyncio.wait_for(poll(), 15)


async def wait_for_health(
    client: LlamaCppCompletionClient, *, loading: bool
) -> style_client.StyleHealth:
    found: list[style_client.StyleHealth] = []

    async def poll() -> None:
        while True:
            health = await client.health()
            if health.loading == loading and not health.ok:
                found.append(health)
                return
            await asyncio.sleep(0.02)

    await asyncio.wait_for(poll(), 15)
    return found[0]


async def test_a_crash_is_followed_by_a_restart(tmp_path: Path) -> None:
    control = tmp_path / "control"
    changes: list[ServerSnapshot] = []
    manager, client, _ = make_manager(tmp_path, "--sim-control", str(control), changes=changes)
    async with running(manager):
        await until_state(manager, ServerState.READY)
        first_pid = manager.snapshot().pid
        control.write_text("exit", encoding="utf-8")
        await wait_until(lambda: manager.snapshot().restarts == 1)
        await until_state(manager, ServerState.READY)
        snapshot = manager.snapshot()
        assert snapshot.restarts == 1 and snapshot.last_exit == 7
        assert snapshot.pid != first_pid and (await client.health()).ok
    states = [change.state for change in changes]
    assert ServerState.BACKOFF in states
    assert states.index(ServerState.BACKOFF) > states.index(ServerState.READY)


async def test_a_server_that_stops_answering_is_killed_and_started_again(tmp_path: Path) -> None:
    control = tmp_path / "control"
    manager, client, _ = make_manager(tmp_path, "--sim-control", str(control))
    async with running(manager):
        await until_state(manager, ServerState.READY)
        control.write_text("hang", encoding="utf-8")
        await wait_until(lambda: manager.snapshot().restarts >= 1, limit_s=30)
        control.write_text("ok", encoding="utf-8")  # the new process must not hang again
        await until_state(manager, ServerState.READY)
        assert (await client.health()).ok
        assert "/health failed" in manager.snapshot().detail or manager.snapshot().restarts >= 1


async def test_the_wait_before_a_restart_doubles_up_to_the_limit(tmp_path: Path) -> None:
    """A program that dies at once is started again after 0.05 s, 0.1 s, 0.2 s, 0.2 s, ..."""
    sleeps: list[float] = []
    model = make_model_file(tmp_path / "models")
    spec = ServerSpec((sys.executable, "-I", "-c", "import sys; sys.exit(3)"), model, free_port())

    class Recording(SystemClock):
        async def sleep(self, seconds: float) -> None:
            sleeps.append(seconds)
            await super().sleep(0)

    client = LlamaCppCompletionClient(f"http://127.0.0.1:{spec.port}", clock=SystemClock())
    manager = LlamaServerManager(
        spec, clock=Recording(), client=client, log_path=tmp_path / "log.txt", timings=FAST
    )
    task = asyncio.create_task(manager.run())
    try:
        await wait_until(lambda: manager.snapshot().restarts >= 5, limit_s=30)
        last_exit = manager.snapshot().last_exit
    finally:
        await manager.stop()
        await asyncio.wait_for(task, 20)
    backoffs = [s for s in sleeps if s in (0.05, 0.1, 0.2)]
    assert backoffs[:4] == [0.05, 0.1, 0.2, 0.2]
    assert last_exit == 3  # the exit code of the program that kept dying


async def test_a_run_that_was_stable_starts_the_waiting_over(tmp_path: Path) -> None:
    timings = ServerTimings(
        start_timeout_s=20.0,
        backoff_start_s=0.07,
        backoff_max_s=5.0,
        stable_after_s=0.0,  # every run counts as stable
        health_interval_s=0.1,
    )
    sleeps: list[float] = []

    class Recording(SystemClock):
        async def sleep(self, seconds: float) -> None:
            sleeps.append(seconds)
            await super().sleep(0)

    model = make_model_file(tmp_path / "models")
    spec = ServerSpec((sys.executable, "-I", "-c", "import sys; sys.exit(3)"), model, free_port())
    client = LlamaCppCompletionClient(f"http://127.0.0.1:{spec.port}", clock=SystemClock())
    manager = LlamaServerManager(
        spec, clock=Recording(), client=client, log_path=tmp_path / "log.txt", timings=timings
    )
    task = asyncio.create_task(manager.run())
    try:
        await wait_until(lambda: manager.snapshot().restarts >= 4, limit_s=30)
    finally:
        await manager.stop()
        await asyncio.wait_for(task, 20)
    assert {s for s in sleeps if 0.05 <= s < 0.5} == {0.07}  # never doubled


async def test_a_model_that_never_finishes_loading_is_given_up_and_tried_again(
    tmp_path: Path,
) -> None:
    timings = ServerTimings(start_timeout_s=0.6, backoff_start_s=0.05, health_interval_s=0.1)
    manager, _, _ = make_manager(tmp_path, "--sim-load-s", "600", timings=timings)
    async with running(manager):
        await wait_until(lambda: manager.snapshot().restarts >= 1, limit_s=30)
        assert "not loaded after" in manager.snapshot().detail or manager.state in (
            ServerState.BACKOFF,
            ServerState.STARTING,
        )


async def test_a_refusal_after_loading_stops_the_server_and_keeps_it_stopped(
    tmp_path: Path,
) -> None:
    refusals = {"count": 0}

    async def refuse() -> None:
        refusals["count"] += 1
        raise ServerBlocked("tokenizer", "extra BOS")

    manager, client, _ = make_manager(tmp_path, on_ready=refuse)
    async with running(manager):
        await until_state(manager, ServerState.BLOCKED)
        assert manager.blocked is not None and manager.blocked.reason == "tokenizer"
        await wait_for_health_gone(client)  # the process is gone ...
        await manager.wait_state({ServerState.READY}, 1.0)  # ... and nothing starts it again
        assert refusals["count"] == 1 and manager.snapshot().restarts == 0
        assert manager.state is ServerState.BLOCKED and not (await client.health()).ok


async def test_a_blocked_server_is_tried_again_after_unblock(tmp_path: Path) -> None:
    verdicts = [ServerBlocked("tokenizer", "extra BOS"), None]

    async def hook() -> None:
        verdict = verdicts.pop(0)
        if verdict is not None:
            raise verdict

    manager, client, _ = make_manager(tmp_path, on_ready=hook)
    async with running(manager):
        await until_state(manager, ServerState.BLOCKED)
        manager.unblock()
        await until_state(manager, ServerState.READY)
        assert (await client.health()).ok and manager.blocked is None


async def test_a_missing_model_file_blocks_instead_of_restarting_for_ever(tmp_path: Path) -> None:
    manager, _, _ = make_manager(tmp_path, model=tmp_path / "not-there.gguf")
    async with running(manager):
        await until_state(manager, ServerState.BLOCKED)
        assert "does not exist" in manager.snapshot().detail
        assert manager.blocked is not None and manager.blocked.reason == "start_failed"


async def test_a_program_that_cannot_be_started_blocks_with_the_reason(tmp_path: Path) -> None:
    model = make_model_file(tmp_path / "models")
    spec = ServerSpec((str(tmp_path / "no-such-llama-server.exe"),), model, free_port())
    client = LlamaCppCompletionClient(f"http://127.0.0.1:{spec.port}", clock=SystemClock())
    manager = LlamaServerManager(
        spec, clock=SystemClock(), client=client, log_path=tmp_path / "log.txt", timings=FAST
    )
    async with running(manager):
        await until_state(manager, ServerState.BLOCKED)
        assert "cannot start" in manager.snapshot().detail


async def test_a_port_used_by_another_program_is_refused(tmp_path: Path) -> None:
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        port = int(occupied.getsockname()[1])
        manager, _, _ = make_manager(tmp_path, port=port)
        with pytest.raises(ServeError, match="not a llama-server"):
            await manager.run()


async def start_outside(tmp_path: Path, model: Path, port: int) -> asyncio.subprocess.Process:
    process = await asyncio.create_subprocess_exec(
        *sim_prefix(),
        "-m",
        str(model),
        "--port",
        str(port),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    client = LlamaCppCompletionClient(f"http://127.0.0.1:{port}", clock=SystemClock())

    async def ready() -> None:
        while not (await client.health()).ok:  # noqa: ASYNC110 - polls a child process over HTTP
            await asyncio.sleep(0.05)

    await asyncio.wait_for(ready(), 20)
    return process


async def test_a_server_of_another_process_that_serves_this_model_is_adopted_and_left_alone(
    tmp_path: Path,
) -> None:
    model = make_model_file(tmp_path / "models")
    port = free_port()
    outside = await start_outside(tmp_path, model, port)
    try:
        manager, client, _ = make_manager(tmp_path, port=port, model=model)
        async with running(manager):
            await until_state(manager, ServerState.EXTERNAL)
            assert manager.snapshot().pid is None  # not ours
        assert outside.returncode is None  # stop() did not touch it
        assert (await client.health()).ok
    finally:
        outside.kill()
        await outside.wait()


async def test_a_server_that_serves_another_model_is_refused(tmp_path: Path) -> None:
    served = make_model_file(tmp_path / "a", "other-model.gguf")
    port = free_port()
    outside = await start_outside(tmp_path, served, port)
    try:
        wanted = make_model_file(tmp_path / "b", "wanted-model.gguf")
        manager, _, _ = make_manager(tmp_path, port=port, model=wanted)
        with pytest.raises(ServeError, match="serves another model"):
            await manager.run()
    finally:
        outside.kill()
        await outside.wait()


async def test_the_child_is_put_in_the_job_object_when_there_is_one(tmp_path: Path) -> None:
    api = FakeWin32()
    job = ProcessJob(api, platform="win32")
    assert job.open()
    manager, _, _ = make_manager(tmp_path, job=job)
    async with running(manager):
        await until_state(manager, ServerState.READY)
        pid = manager.snapshot().pid
        assert pid is not None and api.open_processes == [pid]
        assert api.jobs[next(iter(api.jobs))] == [9000 + pid]


async def test_stop_before_the_first_start_ends_the_loop(tmp_path: Path) -> None:
    manager, _, _ = make_manager(tmp_path)
    await manager.stop()
    await asyncio.wait_for(manager.run(), 10)
    assert manager.state is ServerState.STOPPED and manager.snapshot().pid is None


async def test_wait_state_reports_whether_the_state_was_reached(tmp_path: Path) -> None:
    manager, _, _ = make_manager(tmp_path)
    assert not await manager.wait_state({ServerState.READY}, 0.1)
    async with running(manager):
        assert await manager.wait_state({ServerState.READY}, 20)


# --------------------------------------------------------------------- helpers


def test_a_log_that_grew_too_big_is_moved_aside_and_the_old_ones_age(tmp_path: Path) -> None:
    folder = tmp_path / "logs"
    folder.mkdir()
    log = folder / "llama-server.log"
    log.write_text("a" * 100, encoding="utf-8")
    rotate_log(log, keep=3, max_bytes=1000)
    assert log.exists()  # small: stays
    for number in range(1, 5):
        log.write_text(f"run{number}" + "x" * 2000, encoding="utf-8")
        rotate_log(log, keep=3, max_bytes=1000)
        assert not log.exists()
    names = sorted(p.name for p in folder.iterdir())
    assert names == ["llama-server.log.1", "llama-server.log.2", "llama-server.log.3"]
    assert (folder / "llama-server.log.1").read_text(encoding="utf-8").startswith("run4")
    assert (folder / "llama-server.log.3").read_text(encoding="utf-8").startswith("run2")


def test_the_tail_of_the_log_has_the_last_non_empty_lines_cut_short(tmp_path: Path) -> None:
    log = tmp_path / "log.txt"
    log.write_text("first\n\nsecond\nthird\n" + "z" * 500 + "\n", encoding="utf-8")
    tail = log_tail(log, lines=3)
    parts = tail.split(" | ")
    assert parts[0] == "second" and parts[1] == "third" and len(parts[2]) == 200
    assert log_tail(tmp_path / "missing.txt") == ""


def test_a_model_file_is_recognised_by_its_name_whatever_the_slashes() -> None:
    expected = Path("D:/data/models/run-1/Q5_K_M.gguf")
    assert same_model_file("D:\\data\\models\\run-1\\Q5_K_M.gguf", expected)
    assert same_model_file("/other/place/Q5_K_M.gguf", expected)
    assert not same_model_file("/other/place/Q4_K_M.gguf", expected)
    assert not same_model_file(None, expected) and not same_model_file("", expected)


async def test_the_model_path_of_a_server_is_read_from_props(tmp_path: Path) -> None:
    model = make_model_file(tmp_path / "models")
    port = free_port()
    outside = await start_outside(tmp_path, model, port)
    try:
        reported = await served_model_path(f"http://127.0.0.1:{port}")
        assert reported is not None and same_model_file(reported, model)
    finally:
        outside.kill()
        await outside.wait()
    assert await served_model_path(f"http://127.0.0.1:{port}") is None


def test_no_console_window_is_asked_for_off_windows_and_the_flag_on_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    assert _creation_flags("win32") == 0x08000000
    assert _creation_flags("linux") == 0
    assert SIM.is_file()
