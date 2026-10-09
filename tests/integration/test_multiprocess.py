"""Several OS processes sharing one SQLite database and one running application.

R-ARCH-006.4: WAL + busy_timeout + short IMMEDIATE transactions make concurrent writers safe;
R-ARCH-006.2: a LIGHT command in another process is noticed by the running application.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from tests.support.waiting import wait_until_sync
from twin.storage import migrate
from twin.storage.crypto import KeyRing, generate_key, use_keyring
from twin.storage.db import Database
from twin.storage.models import Job
from twin.storage.settings_store import get_setting
from twin.storage.state import read_state_version

pytestmark = pytest.mark.integration

WRITER = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    from twin.clock import SystemClock
    from twin.ops.jobs import JobQueue
    from twin.storage.crypto import KeyRing, use_keyring
    from twin.storage.db import Database, WritePolicy, use_write_policy
    from twin.storage.settings_store import get_setting, put_setting

    db_path, key_hex, tag, count = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
    clock = SystemClock()
    db = Database(db_path, clock=clock)
    with use_keyring(KeyRing({1: bytes.fromhex(key_hex)}, 1)):
        queue = JobQueue(db, clock)
        for index in range(count):
            queue.enqueue("load_test", {"writer": tag, "index": index})
            with use_write_policy(WritePolicy(bump_state=True)), db.transaction() as session:
                put_setting(session, f"counter.{tag}", index, clock=clock, by=tag)
    print("finished", tag, flush=True)
    """
)


def test_two_processes_writing_jobs_and_settings_do_not_lose_updates(tmp_path: Path) -> None:
    db_path = tmp_path / "shared.db"
    migrate.upgrade(db_path)
    key = generate_key()
    count = 60
    env = {**os.environ, "PYTHONUTF8": "1"}
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", WRITER, str(db_path), key.hex(), tag, str(count)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for tag in ("alpha", "beta")
    ]
    outputs = [proc.communicate(timeout=120) for proc in procs]
    for proc, (out, err) in zip(procs, outputs, strict=True):
        assert proc.returncode == 0, err
        assert "finished" in out
        assert "locked" not in err.lower()

    db = Database(db_path)
    try:
        with use_keyring(KeyRing({1: key}, 1)), db.session() as session:
            jobs = session.query(Job).all()
            assert len(jobs) == 2 * count
            seen = {(j.payload["writer"], j.payload["index"]) for j in jobs}
            assert len(seen) == 2 * count  # nothing lost, nothing duplicated
            assert get_setting(session, "counter.alpha") == count - 1
            assert get_setting(session, "counter.beta") == count - 1
            # every transaction bumped the shared counter atomically: no lost increments
            assert read_state_version(session) == 2 * count
    finally:
        db.dispose()


def run_twin(
    args: list[str],
    env: dict[str, str],
    *,
    timeout: float = 60,
    stdin_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "twin", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        input=stdin_text,
    )


def read_log(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.mark.skipif(
    sys.platform == "win32", reason="signals are POSIX here; Windows has its own tests"
)
def test_running_application_notices_a_cli_change_within_two_seconds_and_stops_gracefully(
    tmp_path: Path,
) -> None:
    home = tmp_path / "proj"
    home.mkdir()
    env = {
        **os.environ,
        "TWIN_HOME": str(home),
        "TWIN_SECRETS_DIR": str(tmp_path / "secrets"),
        "TWIN_KEYRING_BACKEND": "file",
        "PYTHONUTF8": "1",
    }
    assert run_twin(["db", "upgrade"], env).returncode == 0
    # the conversation engine needs the DeepSeek key to start (09-3); this one is only a test value
    stored = run_twin(
        ["secrets", "set", "deepseek_api_key", "--stdin"],
        env,
        stdin_text="sk-test-for-the-multiprocess-test\n",
    )
    assert stored.returncode == 0, stored.stderr
    log_path = home / "data" / "logs" / "twin.log"

    app = subprocess.Popen(
        [sys.executable, "-m", "twin", "run"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_until_sync(
            lambda: any(e["event"] == "application_running" for e in read_log(log_path)),
            timeout=40,
        )
        # a second instance is refused while the first runs
        second = run_twin(["run"], env)
        assert second.returncode == 4 and "already running" in second.stderr
        # exclusive commands are refused as well
        refused = run_twin(["db", "upgrade"], env)
        assert refused.returncode == 4 and "exclusive" in refused.stderr
        # the application keeps running
        assert app.poll() is None

        started = time.monotonic()
        changed = run_twin(["settings", "set", "time.bot_timezone", "Asia/Shanghai"], env)
        assert changed.returncode == 0, changed.stderr
        wait_until_sync(
            lambda: any(e["event"] == "state_changed" for e in read_log(log_path)), timeout=10
        )
        assert time.monotonic() - started < 2.0 + 3.0  # poll interval plus process start-up slack
    finally:
        app.send_signal(signal.SIGINT)
        try:
            app.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            app.kill()
            app.communicate()
            raise
    assert app.returncode == 0
    events = [e["event"] for e in read_log(log_path)]
    assert events.index("application_stopping") < events.index("application_stopped")
    assert "component_stopped" in events
    heartbeat_written = run_twin(["settings", "list"], env)
    assert "Asia/Shanghai" in heartbeat_written.stdout


def test_cli_entry_points_work_as_a_module(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "TWIN_HOME": str(tmp_path),
        "TWIN_SECRETS_DIR": str(tmp_path / "secrets"),
        "TWIN_KEYRING_BACKEND": "file",
    }
    helped = run_twin(["--help"], env)
    assert helped.returncode == 0 and "doctor" in helped.stdout
    doctor = run_twin(["doctor"], env)
    assert doctor.returncode == 0, doctor.stdout + doctor.stderr
    assert "keyring" in doctor.stdout
