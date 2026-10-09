"""R-TRN-010: login, host keys, uploads that resume, downloads that verify, background jobs.

The tests talk to a real SSH server (``asyncssh``'s server side) on localhost, so they run on
Windows too: the tools on the "instance" are Python programs.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
from pathlib import Path

import asyncssh
import pytest

from tests.support.clock import ManualClock
from tests.support.ssh_server import PASSWORD, USER, LocalSSHServer
from tests.support.waiting import wait_until
from twin.clock import SystemClock
from twin.config.secrets import SecretStore
from twin.config.settings import AutoDlConfig
from twin.training.layout import RemoteLayout
from twin.training.remote.connection import (
    HashMismatchError,
    HostKeyError,
    LoginRefusedError,
    RemoteError,
    RemoteTarget,
    has_known_host,
    open_connection,
    target_from_settings,
)
from twin.training.remote.jobs import RemoteJobs
from twin.training.remote.session import RemoteSession
from twin.training.remote.transfer import run_command

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "training" / "autodl" / "tools"


def target_for(server: LocalSSHServer, tmp_path: Path, **overrides: object) -> RemoteTarget:
    values: dict[str, object] = {
        "host": "127.0.0.1",
        "port": server.port,
        "user": USER,
        "auth": "password",
        "known_hosts": tmp_path / "known_hosts",
        "password": PASSWORD,
    }
    values.update(overrides)
    return RemoteTarget(**values)  # type: ignore[arg-type]


def trusting(fingerprints: list[str] | None = None):  # type: ignore[no-untyped-def]
    def trust(label: str, fingerprint: str) -> bool:
        if fingerprints is not None:
            fingerprints.append(fingerprint)
        return True

    return trust


def session_for(server: LocalSSHServer, tmp_path: Path, **overrides: object) -> RemoteSession:
    return RemoteSession(
        target_for(server, tmp_path, **overrides),
        SystemClock(),
        trust=trusting(),
        backoff=(0.0, 0.0, 0.0, 0.0),
    )


def random_file(path: Path, size: int) -> str:
    data = os.urandom(size)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------------------ settings


def test_the_target_comes_from_the_settings_and_the_credential_store(
    secret_store: SecretStore, tmp_path: Path
) -> None:
    config = AutoDlConfig(host="connect.example.test", port=10309)
    with pytest.raises(RemoteError, match="twin secrets set autodl_password"):
        target_from_settings(config, secret_store, tmp_path / "kh")
    secret_store.set("autodl_password", "s3cret-value")
    target = target_from_settings(config, secret_store, tmp_path / "kh")
    assert (target.host, target.port, target.user, target.auth) == (
        "connect.example.test",
        10309,
        "root",
        "password",
    )
    assert "s3cret-value" not in repr(target) and "s3cret-value" not in target.label
    with pytest.raises(RemoteError, match=r"autodl\.host and autodl\.port"):
        target_from_settings(AutoDlConfig(), secret_store, tmp_path / "kh")


def test_key_login_needs_an_existing_key_file(secret_store: SecretStore, tmp_path: Path) -> None:
    config = AutoDlConfig(host="h", port=22, auth="key")
    with pytest.raises(RemoteError, match="key_path"):
        target_from_settings(config, secret_store, tmp_path / "kh")
    missing = AutoDlConfig(host="h", port=22, auth="key", key_path=str(tmp_path / "nope"))
    with pytest.raises(RemoteError, match="does not exist"):
        target_from_settings(missing, secret_store, tmp_path / "kh")
    key = tmp_path / "id"
    key.write_text("x", encoding="utf-8")
    ok = target_from_settings(
        AutoDlConfig(host="h", port=22, auth="key", key_path=str(key)),
        secret_store,
        tmp_path / "kh",
    )
    assert ok.auth == "key" and ok.key_path == key and ok.password is None


# ----------------------------------------------------------------------------- host keys


async def test_the_first_connection_asks_to_trust_the_host_key_and_remembers_it(
    tmp_path: Path,
) -> None:
    async with LocalSSHServer(tmp_path / "srv") as server:
        target = target_for(server, tmp_path)
        asked: list[str] = []
        connection = await open_connection(target, trust=trusting(asked))
        connection.close()
        assert asked == [server.fingerprint()]
        assert has_known_host(target.known_hosts, "127.0.0.1", server.port)
        line = target.known_hosts.read_text("utf-8").strip()
        assert line.startswith(f"[127.0.0.1]:{server.port} ssh-ed25519 ")
        again = await open_connection(target, trust=None)  # known now: no question
        again.close()
        assert asked == [server.fingerprint()]


async def test_an_untrusted_host_key_stops_the_connection_and_saves_nothing(
    tmp_path: Path,
) -> None:
    async with LocalSSHServer(tmp_path / "srv") as server:
        target = target_for(server, tmp_path)
        with pytest.raises(HostKeyError, match="not trusted"):
            await open_connection(target, trust=lambda label, fingerprint: False)
        with pytest.raises(HostKeyError, match="twin train remote connect"):
            await open_connection(target, trust=None)
        assert not target.known_hosts.exists()


async def test_a_changed_host_key_is_refused(tmp_path: Path) -> None:
    async with LocalSSHServer(tmp_path / "srv") as server:
        target = target_for(server, tmp_path)
        other = asyncssh.generate_private_key("ssh-ed25519").export_public_key("openssh").decode()
        target.known_hosts.write_text(
            f"[127.0.0.1]:{server.port} {' '.join(other.split()[:2])}\n", encoding="utf-8"
        )
        with pytest.raises(HostKeyError, match="differs"):
            await open_connection(target, trust=trusting())


async def test_a_wrong_password_is_a_login_error_that_is_not_retried(tmp_path: Path) -> None:
    async with LocalSSHServer(tmp_path / "srv") as server:
        session = session_for(server, tmp_path, password="wrong")
        with pytest.raises(LoginRefusedError, match="refused the login"):
            await session.retrying(session.connection)
        assert server.state.logins == 1  # no second attempt
        assert session.reconnects == 0


async def test_an_unreachable_host_is_a_remote_error(tmp_path: Path) -> None:
    target = RemoteTarget("127.0.0.1", 1, USER, "password", tmp_path / "kh", password="x")
    with pytest.raises(RemoteError, match="cannot connect"):
        await open_connection(target, trust=trusting(), limit_s=5.0)


async def test_key_login_works(tmp_path: Path) -> None:
    key = asyncssh.generate_private_key("ssh-ed25519")
    key_file = tmp_path / "id_ed25519"
    key_file.write_bytes(key.export_private_key())
    async with LocalSSHServer(tmp_path / "srv") as server:
        server.state.authorized_key = key
        target = target_for(server, tmp_path, auth="key", key_path=key_file, password=None)
        connection = await open_connection(target, trust=trusting())
        done = await run_command(connection, "python3 -c \"print('hi')\"")
        connection.close()
        assert done.ok and done.stdout.strip() == "hi"


# ------------------------------------------------------------------------------ commands


async def test_commands_run_on_the_instance_with_input_and_report_their_status(
    tmp_path: Path,
) -> None:
    async with LocalSSHServer(tmp_path / "srv") as server, session_for(server, tmp_path) as session:
        echo = await session.run(
            'python3 -c "import sys; print(sys.stdin.readline().strip().upper())"',
            stdin="secret words\n",
        )
        assert echo.ok and echo.stdout.strip() == "SECRET WORDS"
        failing = await session.run('python3 -c "import sys; sys.exit(7)"')
        assert failing.status == 7 and not failing.ok
        missing = await session.run("no-such-program-xyz")
        assert missing.status == 127
        assert all("secret words" not in command for command in server.state.commands)


# ------------------------------------------------------------------------------- upload


async def test_an_upload_is_verified_and_a_second_upload_sends_nothing(tmp_path: Path) -> None:
    local = tmp_path / "bundle.enc"
    digest = random_file(local, 3 * (1 << 20) + 123)
    async with LocalSSHServer(tmp_path / "srv") as server, session_for(server, tmp_path) as session:
        files = await session.files()
        progress: list[tuple[int, int]] = []
        first = await files.upload(
            local, "work/bundle.enc", progress=lambda a, b: progress.append((a, b))
        )
        assert (first.size, first.sha256, first.bytes_sent, first.skipped) == (
            local.stat().st_size,
            digest,
            local.stat().st_size,
            False,
        )
        assert progress[-1] == (local.stat().st_size, local.stat().st_size)
        stored = server.root / "work" / "bundle.enc"
        assert hashlib.sha256(stored.read_bytes()).hexdigest() == digest
        assert not (server.root / "work" / "bundle.enc.part").exists()
        second = await files.upload(local, "work/bundle.enc")
        assert second.skipped and second.bytes_sent == 0


async def test_an_interrupted_upload_resumes_from_the_verified_part_file(tmp_path: Path) -> None:
    local = tmp_path / "bundle.enc"
    size = 4 * (1 << 20) + 777
    digest = random_file(local, size)
    async with LocalSSHServer(tmp_path / "srv") as server:
        server.state.drop_after_bytes["bundle.enc.part"] = 2 * (1 << 20)
        async with session_for(server, tmp_path) as session:
            result = await session.retrying(
                lambda: _upload(session, local), what="uploading the package"
            )
            assert session.reconnects == 1 and server.state.drops == 1
        assert result.resumed_from >= 2 * (1 << 20)
        assert result.bytes_sent == size - result.resumed_from
        written = sum(v for k, v in server.state.bytes_written.items() if k.endswith(".part"))
        assert written == size  # nothing was sent twice
        stored = server.root / "work" / "bundle.enc"
        assert hashlib.sha256(stored.read_bytes()).hexdigest() == digest


async def _upload(session: RemoteSession, local: Path):  # type: ignore[no-untyped-def]
    files = await session.files()
    return await files.upload(local, "work/bundle.enc")


async def test_a_part_file_that_is_not_the_start_of_the_local_file_is_replaced(
    tmp_path: Path,
) -> None:
    local = tmp_path / "bundle.enc"
    size = 2 * (1 << 20) + 5
    digest = random_file(local, size)
    async with LocalSSHServer(tmp_path / "srv") as server, session_for(server, tmp_path) as session:
        (server.root / "work").mkdir()
        (server.root / "work" / "bundle.enc.part").write_bytes(b"garbage" * 1000)
        files = await session.files()
        result = await files.upload(local, "work/bundle.enc")
        assert result.resumed_from == 0 and result.bytes_sent == size
        assert (
            hashlib.sha256((server.root / "work" / "bundle.enc").read_bytes()).hexdigest() == digest
        )


async def test_a_changed_remote_copy_is_replaced_by_the_new_file(tmp_path: Path) -> None:
    local = tmp_path / "bundle.enc"
    digest = random_file(local, 1000)
    async with LocalSSHServer(tmp_path / "srv") as server, session_for(server, tmp_path) as session:
        (server.root / "work").mkdir()
        (server.root / "work" / "bundle.enc").write_bytes(b"an older package")
        files = await session.files()
        result = await files.upload(local, "work/bundle.enc")
        assert not result.skipped
        assert (
            hashlib.sha256((server.root / "work" / "bundle.enc").read_bytes()).hexdigest() == digest
        )


@pytest.mark.parametrize("with_tool", [True, False])
async def test_the_remote_hash_comes_from_the_tool_or_from_the_bytes(
    tmp_path: Path, with_tool: bool
) -> None:
    local = tmp_path / "bundle.enc"
    digest = random_file(local, 100_000)
    async with LocalSSHServer(tmp_path / "srv") as server, session_for(server, tmp_path) as session:
        files = await session.files()
        if with_tool:
            await files.write_bytes(
                "work/tools/file_hash.py", (TOOLS / "file_hash.py").read_bytes()
            )
            files.use_hash_tool("work/tools/file_hash.py")
        else:
            files.use_hash_tool("work/tools/missing.py")  # the command fails: SFTP fallback
        await files.upload(local, "work/bundle.enc")
        assert await files.remote_hash("work/bundle.enc") == (100_000, digest)
        prefix = hashlib.sha256(local.read_bytes()[:1234]).hexdigest()
        assert await files.remote_hash("work/bundle.enc", 1234) == (100_000, prefix)
        assert await files.remote_hash("work/absent") == (None, None)
        used_tool = any("file_hash.py" in c and "missing" not in c for c in server.state.commands)
        assert used_tool is with_tool


async def test_small_files_are_written_and_read_back(tmp_path: Path) -> None:
    async with LocalSSHServer(tmp_path / "srv") as server, session_for(server, tmp_path) as session:
        files = await session.files()
        await files.write_bytes("work/autodl/profile.env", b"A=1\n")
        assert await files.read_text("work/autodl/profile.env") == "A=1\n"
        assert await files.read_text("work/none") is None
        assert await files.size("work/autodl/profile.env") == 4
        await files.makedirs("work/a/b")
        assert (server.root / "work" / "a" / "b").is_dir()


# ----------------------------------------------------------------------------- download


async def test_a_download_is_verified_and_a_second_one_is_skipped(tmp_path: Path) -> None:
    async with LocalSSHServer(tmp_path / "srv") as server, session_for(server, tmp_path) as session:
        remote = server.root / "work" / "model.gguf"
        remote.parent.mkdir()
        digest = random_file(remote, 2 * (1 << 20) + 9)
        files = await session.files()
        target = tmp_path / "out" / "gguf" / "model.gguf"
        first = await files.download(
            "work/model.gguf", target, size=remote.stat().st_size, sha256=digest
        )
        assert not first.skipped and hashlib.sha256(target.read_bytes()).hexdigest() == digest
        assert not target.with_name("model.gguf.part").exists()
        second = await files.download(
            "work/model.gguf", target, size=remote.stat().st_size, sha256=digest
        )
        assert second.skipped and second.bytes_sent == 0


async def test_a_download_continues_a_verified_partial_file(tmp_path: Path) -> None:
    async with LocalSSHServer(tmp_path / "srv") as server, session_for(server, tmp_path) as session:
        remote = server.root / "work" / "model.gguf"
        remote.parent.mkdir()
        digest = random_file(remote, 3 * (1 << 20))
        target = tmp_path / "out" / "model.gguf"
        target.parent.mkdir()
        target.with_name("model.gguf.part").write_bytes(remote.read_bytes()[: 1 << 20])
        files = await session.files()
        files.use_hash_tool(None)
        result = await files.download("work/model.gguf", target, size=3 * (1 << 20), sha256=digest)
        assert result.resumed_from == 1 << 20 and result.bytes_sent == 2 * (1 << 20)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        # a partial file with other bytes is not trusted
        target.unlink()
        target.with_name("model.gguf.part").write_bytes(b"x" * 5000)
        again = await files.download("work/model.gguf", target, size=3 * (1 << 20), sha256=digest)
        assert again.resumed_from == 0 and hashlib.sha256(target.read_bytes()).hexdigest() == digest


async def test_a_file_that_does_not_match_its_hash_is_refused_and_discarded(tmp_path: Path) -> None:
    async with LocalSSHServer(tmp_path / "srv") as server, session_for(server, tmp_path) as session:
        remote = server.root / "work" / "model.gguf"
        remote.parent.mkdir()
        random_file(remote, 4096)
        files = await session.files()
        target = tmp_path / "out" / "model.gguf"
        with pytest.raises(RemoteError, match="does not match its sha256"):
            await files.download("work/model.gguf", target, size=4096, sha256="0" * 64)
        assert not target.exists() and not target.with_name("model.gguf.part").exists()


# ------------------------------------------------------------------------------- jobs


async def install_job_tool(session: RemoteSession, layout: RemoteLayout) -> None:
    files = await session.files()
    await files.write_bytes(f"{layout.tools}/remote_job.py", (TOOLS / "remote_job.py").read_bytes())


async def test_a_job_is_followed_to_its_end_and_reports_the_exit_code(tmp_path: Path) -> None:
    layout = RemoteLayout("work")
    async with LocalSSHServer(tmp_path / "srv") as server, session_for(server, tmp_path) as session:
        await install_job_tool(session, layout)
        jobs = RemoteJobs(session, layout, SystemClock(), poll_seconds=0.05)
        seen: list[str] = []
        code = "import sys; print('step one'); print('step two', flush=True); sys.exit(4)"
        outcome = await jobs.run("j1", [sys.executable, "-c", code], emit=seen.append)
        assert (outcome.state, outcome.exit_code, outcome.ok) == ("exited", 4, False)
        assert "".join(seen).split() == ["step", "one", "step", "two"]
        ok = await jobs.run("j2", [sys.executable, "-c", "print('fine')"], emit=seen.append)
        assert ok.ok and "fine" in "".join(seen)
        assert (await jobs.status("j1")).exit_code == 4
        assert (await jobs.status("never")).state == "none"


async def test_a_dropped_connection_does_not_stop_the_job_and_the_log_goes_on_from_the_offset(
    tmp_path: Path,
) -> None:
    layout = RemoteLayout("work")
    marker = tmp_path / "srv" / "release"
    async with LocalSSHServer(tmp_path / "srv") as server, session_for(server, tmp_path) as session:
        await install_job_tool(session, layout)
        jobs = RemoteJobs(session, layout, SystemClock(), poll_seconds=0.05)
        code = (
            "import pathlib, time\n"
            "print('before the break', flush=True)\n"
            f"while not pathlib.Path({str(marker)!r}).exists():\n    time.sleep(0.05)\n"
            "print('after the break', flush=True)\n"
        )
        seen: list[str] = []
        offsets: list[int] = []
        task = asyncio.create_task(
            jobs.run("j3", [sys.executable, "-c", code], emit=seen.append, on_offset=offsets.append)
        )
        await wait_until(lambda: "before the break" in "".join(seen), limit_s=20)
        await session.drop()  # the connection breaks while the job runs
        marker.write_text("go", encoding="utf-8")
        outcome = await asyncio.wait_for(task, 30)
        assert outcome.ok
        assert "".join(seen).count("before the break") == 1  # not printed again after reconnecting
        assert "after the break" in "".join(seen)
        assert offsets and offsets == sorted(offsets)
        # asking for a job that is finished does not start it again
        status = await jobs.start("j3", [sys.executable, "-c", "print('second run')"])
        assert status.state in ("started", "running", "exited")


async def test_starting_a_job_that_runs_does_not_start_a_second_one(tmp_path: Path) -> None:
    layout = RemoteLayout("work")
    marker = tmp_path / "srv" / "release2"
    async with LocalSSHServer(tmp_path / "srv") as server, session_for(server, tmp_path) as session:
        await install_job_tool(session, layout)
        jobs = RemoteJobs(session, layout, SystemClock(), poll_seconds=0.05)
        code = (
            "import pathlib, time\n"
            f"while not pathlib.Path({str(marker)!r}).exists():\n    time.sleep(0.05)\n"
        )
        first = await jobs.start("j4", [sys.executable, "-c", code])
        second = await jobs.start("j4", [sys.executable, "-c", "print('never runs')"])
        assert first.state in ("started", "running") and second.state == "running"
        marker.write_text("go", encoding="utf-8")
        outcome = await jobs.follow("j4", emit=lambda text: None)
        assert outcome.ok
        log = (server.root / "work" / "jobs" / "j4" / "log").read_text("utf-8")
        assert "never runs" not in log


async def test_retrying_gives_up_after_the_backoff_list_and_waits_with_the_clock(
    tmp_path: Path,
) -> None:
    clock = ManualClock()
    target = RemoteTarget("127.0.0.1", 1, USER, "password", tmp_path / "kh", password="x")
    session = RemoteSession(target, clock, trust=trusting(), backoff=(1.0, 2.0))

    async def attempt() -> None:
        await session.run("true")

    task = asyncio.create_task(session.retrying(attempt, what="the test"))
    await wait_until(lambda: clock.pending_sleepers >= 1)
    await clock.advance(1.0)
    await wait_until(lambda: clock.pending_sleepers >= 1)
    await clock.advance(2.0)
    with pytest.raises(RemoteError, match="after 2 reconnects"):
        await asyncio.wait_for(task, 10)
    assert session.reconnects == 2
    assert clock.sleeps == [1.0, 2.0]


async def test_a_hash_mismatch_is_tried_once_more_and_a_second_one_is_final(tmp_path: Path) -> None:
    target = RemoteTarget("127.0.0.1", 1, USER, "password", tmp_path / "kh", password="x")
    session = RemoteSession(target, SystemClock(), trust=trusting(), backoff=(0.0, 0.0, 0.0))
    calls = 0

    async def damaged() -> None:
        nonlocal calls
        calls += 1
        raise HashMismatchError("the file does not match")

    with pytest.raises(HashMismatchError):
        await session.retrying(damaged, what="the test")
    assert calls == 2 and session.reconnects == 0  # no reconnect, no long wait

    outcomes = iter([HashMismatchError("first"), "fine"])

    async def heals() -> str:
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert await session.retrying(heals, what="the test") == "fine"
