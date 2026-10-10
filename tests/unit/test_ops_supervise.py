"""``twin supervise``: growing waits, normal exit, stop, kill, job object (R-OPS-001)."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from tests.support.clock import ManualClock
from tests.support.ops import FakeChild, FakeLauncher
from tests.support.waiting import wait_until
from tests.support.win32 import FakeWin32
from twin.clock import SystemClock
from twin.config.settings import SuperviseConfig
from twin.ops.jobobject import ProcessJob
from twin.ops.process_model import ExitCode
from twin.ops.supervise import (
    HISTORY_KEY,
    LAUNCH_ENV,
    MAX_HISTORY,
    RestartLog,
    SubprocessLauncher,
    Supervisor,
    child_arguments,
    describe_exit,
)
from twin.services import Services
from twin.storage.settings_store import get_setting

CONFIG = SuperviseConfig(backoff_start_s=5, backoff_max_s=300, stable_after_min=30, stop_grace_s=40)


class Harness:
    """A supervisor on a manual clock with scripted children."""

    def __init__(
        self,
        services: Services,
        clock: ManualClock,
        *,
        config: SuperviseConfig = CONFIG,
        launcher: FakeLauncher | None = None,
        stop_file: Path | None = None,
        job: ProcessJob | None = None,
        launch: str = "task",
        env: dict[str, str] | None = None,
    ) -> None:
        self.services = services
        self.clock = clock
        self.launcher = launcher or FakeLauncher()
        self.history = RestartLog(services.db, services.clock)
        self.stop = asyncio.Event()
        self.supervisor = Supervisor(
            self.launcher,
            clock,
            config,
            env=env if env is not None else {"PATH": "/usr/bin"},
            alerts=services.alerts,
            history=self.history,
            job=job,
            stop_file=stop_file,
            launch=launch,
        )
        self.task: asyncio.Task[int] | None = None

    def start(self) -> None:
        self.task = asyncio.ensure_future(self.supervisor.run(self.stop))

    @property
    def children(self) -> list[FakeChild]:
        return self.launcher.children

    async def child(self, number: int) -> FakeChild:
        await wait_until(lambda: len(self.children) >= number)
        return self.children[number - 1]

    async def crash(self, number: int, code: int = 1) -> float:
        """End child ``number`` with ``code``; returns the wait the supervisor starts."""
        child = await self.child(number)
        mark = len(self.clock.sleeps)
        known = len(self.history.entries())
        child.finish(code)
        await wait_until(lambda: len(self.history.entries()) == known + 1)
        entry = self.history.entries()[-1]
        delay = float(str(entry["delay_s"]))
        await wait_until(lambda: delay in self.clock.sleeps[mark:])  # the wait has begun
        return delay

    async def finish(self) -> int:
        assert self.task is not None
        return await asyncio.wait_for(self.task, 5)


@pytest.fixture
def harness(services: Services, clock: ManualClock) -> Harness:
    return Harness(services, clock)


async def test_a_crashing_child_is_restarted_after_5_10_20_up_to_5_minutes(
    harness: Harness,
) -> None:
    harness.start()
    delays = []
    for number in range(1, 9):
        delay = await harness.crash(number)
        delays.append(delay)
        await harness.clock.advance(delay)
    assert delays == [5, 10, 20, 40, 80, 160, 300, 300]
    await harness.child(9)
    assert harness.supervisor.restarts == 8 and len(harness.children) == 9
    harness.stop.set()
    assert await harness.finish() == 0


async def test_every_restart_is_logged_recorded_and_alerted_with_a_reason(
    harness: Harness, services: Services
) -> None:
    harness.start()
    for number, code in enumerate((1, int(ExitCode.SCHEMA), 1), start=1):
        await harness.clock.advance(await harness.crash(number, code))
    harness.stop.set()
    await harness.finish()
    entries = harness.history.entries()
    assert [e["exit_code"] for e in entries] == [1, int(ExitCode.SCHEMA), 1]
    assert entries[0]["reason"] == "exit code 1"
    assert entries[1]["reason"] == "the database needs `twin db upgrade`"
    assert all(e["at"] for e in entries) and entries[0]["delay_s"] == 5
    alerts = [a for a in services.alerts.recent() if a.kind == "alert"]
    assert alerts and {a.category for a in alerts} == {"process_restarted"}
    assert len([a for a in alerts if not a.suppressed]) == 1  # the rate limit: one notice
    assert any("restarting in" in a.title for a in alerts)


async def test_the_fifth_quick_failure_in_a_row_is_critical_and_passes_the_rate_limit(
    harness: Harness, services: Services
) -> None:
    harness.start()
    for number in range(1, 6):
        await harness.clock.advance(await harness.crash(number))
    harness.stop.set()
    await harness.finish()
    announced = [a for a in services.alerts.recent() if a.kind == "alert" and not a.suppressed]
    assert sorted(a.severity for a in announced) == ["critical", "warning"]


async def test_a_child_that_ran_for_half_an_hour_starts_the_wait_over(
    harness: Harness,
) -> None:
    harness.start()
    for number in (1, 2, 3):
        await harness.clock.advance(await harness.crash(number))
    child = await harness.child(4)
    await harness.clock.advance(31 * 60)  # it lived long enough to count as stable
    assert not child._exit.done()
    delay = await harness.crash(4)
    assert delay == 5  # not 40
    await harness.clock.advance(delay)
    await harness.child(5)
    harness.stop.set()
    await harness.finish()
    assert [e["delay_s"] for e in harness.history.entries()] == [5, 10, 20, 5]
    assert harness.history.entries()[-1]["ran_s"] >= 31 * 60


async def test_a_child_that_ended_normally_is_not_started_again(
    harness: Harness, services: Services
) -> None:
    harness.start()
    child = await harness.child(1)
    child.finish(0)
    assert await harness.finish() == 0
    assert len(harness.children) == 1 and harness.supervisor.restarts == 0
    assert harness.history.entries() == [] and harness.supervisor.last_exit == 0
    assert services.alerts.recent() == []


async def test_stopping_asks_the_child_to_stop_and_waits_for_it(harness: Harness) -> None:
    harness.start()
    child = await harness.child(1)
    harness.stop.set()
    assert await harness.finish() == 0
    assert child.stop_requests == 1 and not child.killed
    assert harness.history.entries() == []  # a requested stop is not a crash


async def test_a_stop_during_the_wait_for_a_restart_ends_at_once(harness: Harness) -> None:
    harness.start()
    await harness.crash(1)  # the supervisor is now waiting 5 s
    harness.stop.set()
    assert await harness.finish() == 0
    assert len(harness.children) == 1  # nothing was started again


async def test_a_child_that_does_not_stop_is_killed_after_the_grace_time(
    services: Services, clock: ManualClock
) -> None:
    config = SuperviseConfig(stop_grace_s=0.05)
    harness = Harness(services, clock, config=config, launcher=FakeLauncher(stops_gracefully=False))
    harness.start()
    child = await harness.child(1)
    harness.stop.set()
    assert await harness.finish() == 0
    assert child.stop_requests == 1 and child.killed
    assert harness.supervisor.last_exit == -9


async def test_the_stop_file_of_twin_service_stop_stops_the_supervisor(
    services: Services, clock: ManualClock, tmp_path: Path
) -> None:
    flag = tmp_path / "supervisor.stop"
    harness = Harness(services, clock, stop_file=flag)
    harness.start()
    child = await harness.child(1)
    flag.write_text("stop\n", encoding="utf-8")
    for _ in range(50):  # the watcher looks once per second of the clock
        await clock.advance(1)
        if harness.task is not None and harness.task.done():
            break
    assert await harness.finish() == 0
    assert child.stop_requests == 1 and not flag.exists()  # the request is cleaned up


async def test_a_stop_request_from_before_the_start_is_stale(
    services: Services, clock: ManualClock, tmp_path: Path
) -> None:
    flag = tmp_path / "supervisor.stop"
    flag.write_text("stop\n", encoding="utf-8")
    harness = Harness(services, clock, stop_file=flag)
    harness.start()
    child = await harness.child(1)
    await clock.advance(5)
    assert not harness.task.done() and child.stop_requests == 0  # type: ignore[union-attr]
    harness.stop.set()
    await harness.finish()


async def test_the_launch_kind_and_utf8_reach_the_child(
    services: Services, clock: ManualClock
) -> None:
    harness = Harness(services, clock, launch="task", env={"PATH": "/bin"})
    harness.start()
    await harness.child(1)
    assert harness.launcher.environments[0] == {
        "PATH": "/bin",
        LAUNCH_ENV: "task",
        "PYTHONUTF8": "1",
    }
    harness.stop.set()
    await harness.finish()
    other = Harness(services, clock, launch="manual", env={"PYTHONUTF8": "0"})
    other.start()
    await other.child(1)
    assert other.launcher.environments[0][LAUNCH_ENV] == "manual"
    assert other.launcher.environments[0]["PYTHONUTF8"] == "0"  # an explicit choice stays
    other.stop.set()
    await other.finish()


async def test_a_history_that_cannot_be_written_does_not_stop_the_restarts(
    services: Services, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(services, clock)

    def broken(restart: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(harness.history, "add", broken)
    harness.start()
    child = await harness.child(1)
    child.finish(1)
    await wait_until(lambda: harness.supervisor.restarts == 1)
    await wait_until(lambda: 5 in clock.sleeps)
    await clock.advance(5)
    await harness.child(2)
    harness.stop.set()
    await harness.finish()


async def test_a_supervisor_without_database_still_restarts(clock: ManualClock) -> None:
    launcher = FakeLauncher()
    stop = asyncio.Event()
    supervisor = Supervisor(launcher, clock, CONFIG, env={})
    task = asyncio.ensure_future(supervisor.run(stop))
    await wait_until(lambda: len(launcher.children) == 1)
    launcher.children[0].finish(2)
    await wait_until(lambda: 5 in clock.sleeps)
    await clock.advance(5)
    await wait_until(lambda: len(launcher.children) == 2)
    stop.set()
    assert await asyncio.wait_for(task, 5) == 0


# ---------------------------------------------------------------------------- job object


async def test_the_child_is_put_in_the_job_and_the_job_is_closed_at_the_end(
    services: Services, clock: ManualClock
) -> None:
    api = FakeWin32()
    job = ProcessJob(api, platform="win32")
    assert job.open()
    harness = Harness(services, clock, job=job)
    harness.start()
    await harness.child(1)
    assert api.open_processes == [1000]
    (handle,) = api.jobs
    assert api.jobs[handle] == [9000 + 1000]
    harness.stop.set()
    await harness.finish()
    assert handle in api.closed and not job.active


def test_a_job_object_exists_on_windows_only() -> None:
    api = FakeWin32()
    elsewhere = ProcessJob(api, platform="linux")
    assert not elsewhere.open() and not elsewhere.adopt_current_process()
    assert not elsewhere.assign(1) and not elsewhere.active
    elsewhere.close()
    assert api.jobs == {}


def test_a_process_can_adopt_itself_and_then_its_children_are_in_the_job() -> None:
    api = FakeWin32()
    job = ProcessJob(api, platform="win32")
    assert job.adopt_current_process() and job.active
    (handle,) = api.jobs
    assert api.jobs[handle] == [-1]  # the pseudo handle of the current process
    job.close()  # adopted: closing would end this very process, the system does it at exit
    assert handle not in api.closed and not job.active


def test_job_object_failures_are_reported_not_raised() -> None:
    api = FakeWin32()
    api.job_fails = True
    job = ProcessJob(api, platform="win32")
    assert not job.open() and not job.adopt_current_process() and not job.assign(5)
    api.job_fails = False
    api.assign_fails = True
    assert not job.adopt_current_process()
    assert not job.assign(7)
    assert 9007 in api.closed  # the process handle is closed even when the assignment failed

    class NoProcess(FakeWin32):
        def open_process(self, pid: int) -> int | None:
            return None

    assert not ProcessJob(NoProcess(), platform="win32").assign(1)


# ------------------------------------------------------------------------ small pieces


def test_the_exit_codes_have_words() -> None:
    assert describe_exit(None) == "the process ended without an exit code"
    assert describe_exit(1) == "exit code 1"
    assert "db upgrade" in describe_exit(int(ExitCode.SCHEMA))
    assert "another instance" in describe_exit(int(ExitCode.BUSY))
    assert "secret" in describe_exit(int(ExitCode.SECRETS))
    assert "configuration" in describe_exit(int(ExitCode.CONFIG))
    assert "consent" in describe_exit(int(ExitCode.CONSENT))


def test_the_child_runs_twin_in_this_environment_with_the_same_options() -> None:
    args = child_arguments(["--set", 'a.b="1"', "--log-level", "DEBUG"])
    assert args[:4] == [sys.executable, "-X", "utf8", "-m"] and args[4] == "twin"
    assert args[5:] == ["--set", 'a.b="1"', "--log-level", "DEBUG", "run"]


def test_the_restart_history_keeps_the_last_200_and_the_session(
    services: Services, clock: ManualClock
) -> None:
    from twin.ops.supervise import Restart

    log = RestartLog(services.db, services.clock)
    assert log.entries() == [] and log.session() is None
    for number in range(MAX_HISTORY + 5):
        log.add(Restart(clock.now_utc(), number, 1.0, 5.0, f"r{number}"))
    entries = log.entries()
    assert len(entries) == MAX_HISTORY and entries[0]["exit_code"] == 5
    log.start_session("task")
    session = log.session()
    assert session == {"started_at": clock.now_utc().isoformat(), "launch": "task"}
    with services.db.session() as db_session:
        assert len(get_setting(db_session, HISTORY_KEY)) == MAX_HISTORY


# ------------------------------------------------------------------- real child processes


FLAKY = """
import pathlib, sys
counter = pathlib.Path(sys.argv[1])
runs = int(counter.read_text()) if counter.exists() else 0
counter.write_text(str(runs + 1))
sys.exit(3 if runs == 0 else 0)
"""

# What the supervisor sends to ask a child to stop: CTRL_BREAK_EVENT on Windows (the child is
# alone in its process group; Python reports it as SIGBREAK), SIGTERM elsewhere.  A Windows
# process cannot be sent SIGTERM, and its SIGBREAK handler only runs while the main thread is
# executing Python code (a long time.sleep() does not return for it), so these children wait
# in short sleeps.
TERM = """
import pathlib, signal, sys, time
marker = pathlib.Path(sys.argv[1])
def stop(*_):
    marker.write_text("stopped")
    sys.exit(0)
signal.signal(getattr(signal, "SIGBREAK", signal.SIGTERM), stop)
pathlib.Path(sys.argv[1] + ".ready").write_text("1")
while True:
    time.sleep(0.05)
"""

STUBBORN = """
import pathlib, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
if hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, signal.SIG_IGN)
pathlib.Path(sys.argv[1]).write_text("1")
while True:
    time.sleep(0.05)
"""

# the application's own shutdown handling (what `twin run` installs), in a real process
APPLICATION = """
import asyncio, pathlib, sys
from twin.app import ShutdownSignals

async def main():
    stop = asyncio.Event()
    signals = ShutdownSignals(asyncio.get_running_loop(), stop)
    signals.install()
    pathlib.Path(sys.argv[1] + ".ready").write_text("1")
    await stop.wait()
    pathlib.Path(sys.argv[1]).write_text(signals.reason or "?")
    signals.uninstall()
    signals.notify_done()

asyncio.run(main())
"""

ENVIRONMENT = """
import os, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(
    os.environ.get("TWIN_LAUNCH", "?") + "," + os.environ.get("PYTHONUTF8", "?")
)
"""


def real_supervisor(script: str, argument: Path, **options: object) -> Supervisor:
    config = SuperviseConfig(backoff_start_s=0.05, backoff_max_s=0.2, stop_grace_s=0.5)
    launcher = SubprocessLauncher([sys.executable, "-c", script, str(argument)])
    return Supervisor(launcher, SystemClock(), config, env={}, **options)  # type: ignore[arg-type]


async def test_a_real_child_that_crashes_once_is_started_again_and_then_ends_normally(
    tmp_path: Path,
) -> None:
    counter = tmp_path / "runs"
    supervisor = real_supervisor(FLAKY, counter)
    assert await asyncio.wait_for(supervisor.run(asyncio.Event()), 20) == 0
    assert counter.read_text() == "2" and supervisor.restarts == 1 and supervisor.last_exit == 0


WINDOWS = sys.platform == "win32"


@pytest.mark.parametrize(
    "platform",
    [
        pytest.param(
            "linux",
            marks=pytest.mark.skipif(
                WINDOWS,
                reason="the POSIX request is SIGTERM; a Windows child cannot be sent one "
                "(terminate() is a hard kill there): the 'win32' case is its counterpart",
            ),
        ),
        "win32",
    ],
)
async def test_a_real_child_is_asked_to_stop_and_saves_its_state(
    tmp_path: Path, platform: str
) -> None:
    marker = tmp_path / "marker"
    config = SuperviseConfig(stop_grace_s=10)
    launcher = SubprocessLauncher([sys.executable, "-c", TERM, str(marker)], platform=platform)
    supervisor = Supervisor(launcher, SystemClock(), config, env={})
    stop = asyncio.Event()
    task = asyncio.ensure_future(supervisor.run(stop))
    await wait_until(lambda: Path(f"{marker}.ready").exists(), limit_s=20)
    stop.set()
    assert await asyncio.wait_for(task, 20) == 0
    assert marker.read_text() == "stopped" and supervisor.last_exit == 0


async def test_a_real_child_that_ignores_the_request_is_killed(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    supervisor = real_supervisor(STUBBORN, ready)
    stop = asyncio.Event()
    task = asyncio.ensure_future(supervisor.run(stop))
    await wait_until(ready.exists, limit_s=20)
    stop.set()
    assert await asyncio.wait_for(task, 20) == 0
    # Killed, not stopped: a SIGKILL shows as -9.  Windows has no signals; TerminateProcess ends
    # the process with the exit code Python passes to it, 1.  Neither is the 0 of a normal exit,
    # and the child never got to write anything but its "ready" file.
    assert supervisor.last_exit == (1 if WINDOWS else -9)


async def test_a_real_application_process_stops_when_the_supervisor_asks(tmp_path: Path) -> None:
    """The request the supervisor sends is the one the application's shutdown handler hears.

    This is the production pairing - ``ShutdownSignals`` in the child, the platform's stop
    request from ``SubprocessChild`` - with nothing replaced.
    """
    reason = tmp_path / "reason"
    config = SuperviseConfig(stop_grace_s=30)
    launcher = SubprocessLauncher([sys.executable, "-X", "utf8", "-c", APPLICATION, str(reason)])
    supervisor = Supervisor(launcher, SystemClock(), config, env=dict(os.environ))
    stop = asyncio.Event()
    task = asyncio.ensure_future(supervisor.run(stop))
    await wait_until(lambda: Path(f"{reason}.ready").exists(), limit_s=60)
    stop.set()
    assert await asyncio.wait_for(task, 60) == 0
    assert supervisor.last_exit == 0  # it finished by itself, it was not killed
    heard = {"SIGBREAK", "console_ctrl_1"} if WINDOWS else {"SIGTERM"}
    assert reason.read_text() in heard


async def test_a_real_child_sees_how_it_was_launched(tmp_path: Path) -> None:
    seen = tmp_path / "seen"
    supervisor = real_supervisor(ENVIRONMENT, seen, launch="task")
    assert await asyncio.wait_for(supervisor.run(asyncio.Event()), 20) == 0
    assert seen.read_text() == "task,1"
