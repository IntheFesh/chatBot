"""The SSH tunnel to the instance: forward, reconnect, refuse, hand over (R-SRV-003).

The SSH server is ``asyncssh``'s own server side (``tests/support/ssh_server.py``) with port
forwarding allowed; the "vLLM" behind it is the scripted HTTP server of the style client tests.
Nothing is slept: every wait is for a state the manager reports.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import sys
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path

import asyncssh
import pytest

from tests.support.llama_sim import free_port
from tests.support.ssh_server import PASSWORD, USER, LocalSSHServer
from tests.support.style_server import StyleServer, running_server, vllm_defaults
from tests.support.waiting import wait_until
from twin.clock import SystemClock
from twin.llm.style_client import StyleHealth, VllmCompletionClient
from twin.serving.tunnel import (
    TunnelManager,
    TunnelState,
    TunnelTimings,
    parse_uptime,
)
from twin.training.remote.connection import RemoteError, RemoteTarget, remember_host

FAST = TunnelTimings(backoff_start_s=0.05, backoff_max_s=0.2, stable_after_s=60.0, watch_s=0.05)


@dataclass
class World:
    ssh: LocalSSHServer
    remote: StyleServer
    target: RemoteTarget
    known_hosts: Path


@pytest.fixture
def remote_server() -> Iterator[StyleServer]:
    with running_server() as server:
        vllm_defaults(server, model="lora-a")
        yield server


@pytest.fixture
async def world(tmp_path: Path, remote_server: StyleServer) -> AsyncIterator[World]:
    async with LocalSSHServer(tmp_path / "ssh") as ssh:
        ssh.state.allow_forwarding = True
        known = tmp_path / "known_hosts"
        remember_host(known, "127.0.0.1", ssh.port, ssh.host_key)
        target = RemoteTarget("127.0.0.1", ssh.port, USER, "password", known, password=PASSWORD)
        yield World(ssh, remote_server, target, known)


def make_tunnel(world: World, **options: object) -> tuple[TunnelManager, int]:
    port = free_port()
    manager = TunnelManager(
        world.target,
        local_port=port,
        remote_port=world.remote.port,
        clock=SystemClock(),
        timings=FAST,
        **options,  # type: ignore[arg-type]
    )
    return manager, port


@contextlib.asynccontextmanager
async def running(manager: TunnelManager) -> AsyncIterator[asyncio.Task[None]]:
    task = asyncio.create_task(manager.run())
    try:
        yield task
    finally:
        await manager.stop()
        await asyncio.wait_for(task, 20)


def vllm_client(port: int) -> VllmCompletionClient:
    return VllmCompletionClient(f"http://127.0.0.1:{port}", model="lora-a", clock=SystemClock())


async def until(manager: TunnelManager, state: TunnelState, limit: float = 20) -> None:
    await wait_until(lambda: manager.state is state, limit_s=limit)


async def test_the_local_port_leads_to_the_port_on_the_instance(world: World) -> None:
    manager, port = make_tunnel(world)
    async with running(manager):
        await until(manager, TunnelState.UP)
        client = vllm_client(port)
        assert (await client.health()).ok  # /health and /v1/models went through the tunnel
        assert await client.tokenize("ab") == [97, 98]
        await client.aclose()
        snapshot = manager.snapshot()
        assert snapshot.state is TunnelState.UP and snapshot.local_port == port
        assert snapshot.up_since is not None and snapshot.reconnects == 0
    assert ("127.0.0.1", world.remote.port) in world.ssh.state.forwards  # the instance's own port


async def test_the_local_end_is_not_reachable_from_other_addresses(world: World) -> None:
    other = socket.gethostbyname(socket.gethostname())
    if other.startswith("127."):
        pytest.skip("this computer has no address other than loopback to try")
    manager, port = make_tunnel(world)
    async with running(manager):
        await until(manager, TunnelState.UP)
        with pytest.raises(OSError):
            await asyncio.wait_for(asyncio.open_connection(other, port), 5)
        _, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.close()


async def test_the_forward_is_asked_for_on_loopback_to_the_loopback_of_the_instance(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[tuple[object, ...]] = []
    original = asyncssh.SSHClientConnection.forward_local_port

    async def spy(self: asyncssh.SSHClientConnection, *args: object, **kwargs: object) -> object:
        asked.append(args)
        return await original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(asyncssh.SSHClientConnection, "forward_local_port", spy)
    manager, port = make_tunnel(world)
    async with running(manager):
        await until(manager, TunnelState.UP)
    assert asked == [("127.0.0.1", port, "127.0.0.1", world.remote.port)]


async def test_a_dropped_connection_is_logged_in_again(world: World) -> None:
    manager, port = make_tunnel(world)
    async with running(manager):
        await until(manager, TunnelState.UP)
        for connection in list(world.ssh.state.connections):
            connection.abort()
        await wait_until(lambda: manager.snapshot().reconnects >= 1, limit_s=30)
        await until(manager, TunnelState.UP, limit=30)
        client = vllm_client(port)
        assert (await client.health()).ok  # the new connection forwards again
        await client.aclose()
        assert world.ssh.state.logins == 2 and manager.snapshot().detail == ""


async def test_the_pause_before_a_login_doubles_up_to_the_limit(world: World) -> None:
    sleeps: list[float] = []

    class Recording(SystemClock):
        async def sleep(self, seconds: float) -> None:
            sleeps.append(seconds)
            await super().sleep(0)

    attempts = {"n": 0}

    async def unreachable(target: RemoteTarget) -> asyncssh.SSHClientConnection:
        attempts["n"] += 1
        raise RemoteError("cannot connect to the instance: timed out")

    manager = TunnelManager(
        world.target,
        local_port=free_port(),
        remote_port=1,
        clock=Recording(),
        timings=FAST,
        connect=unreachable,
    )
    task = asyncio.create_task(manager.run())
    try:
        await wait_until(lambda: attempts["n"] >= 6, limit_s=20)
        assert manager.state in (TunnelState.BACKOFF, TunnelState.CONNECTING)
        assert "timed out" in manager.snapshot().detail or manager.state is TunnelState.CONNECTING
    finally:
        await manager.stop()
        await asyncio.wait_for(task, 20)
    assert sleeps[:5] == [0.05, 0.1, 0.2, 0.2, 0.2]


async def test_a_refused_password_is_not_retried_until_asked(world: World) -> None:
    password = {"value": "wrong-password"}

    async def login(target: RemoteTarget) -> asyncssh.SSHClientConnection:
        from twin.training.remote.connection import open_connection

        return await open_connection(
            RemoteTarget(
                target.host,
                target.port,
                target.user,
                "password",
                target.known_hosts,
                password=password["value"],
            ),
            trust=None,
        )

    manager, port = make_tunnel(world, connect=login)
    async with running(manager):
        await until(manager, TunnelState.BLOCKED)
        assert "refused the login" in manager.snapshot().detail
        logins = world.ssh.state.logins
        await asyncio.sleep(0.3)  # it must not knock again by itself
        assert world.ssh.state.logins == logins and manager.state is TunnelState.BLOCKED
        password["value"] = PASSWORD
        manager.retry()
        await until(manager, TunnelState.UP)
        client = vllm_client(port)
        assert (await client.health()).ok
        await client.aclose()


async def test_an_unknown_host_key_is_not_trusted_by_a_background_tunnel(
    world: World, tmp_path: Path
) -> None:
    unknown = RemoteTarget(
        "127.0.0.1", world.ssh.port, USER, "password", tmp_path / "nothing-known", password=PASSWORD
    )
    manager = TunnelManager(
        unknown,
        local_port=free_port(),
        remote_port=world.remote.port,
        clock=SystemClock(),
        timings=FAST,
    )
    async with running(manager):
        await until(manager, TunnelState.BLOCKED)
        assert "not trusted yet" in manager.snapshot().detail
        assert "twin train remote connect" in manager.snapshot().detail


async def test_a_local_port_taken_by_another_program_is_an_error_and_not_a_tunnel(
    world: World,
) -> None:
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        port = int(occupied.getsockname()[1])

        async def not_a_tunnel() -> StyleHealth:
            return StyleHealth(False, "unreachable")

        manager = TunnelManager(
            world.target,
            local_port=port,
            remote_port=world.remote.port,
            clock=SystemClock(),
            timings=FAST,
            local_health=not_a_tunnel,
        )
        async with running(manager):
            await wait_until(lambda: manager.snapshot().reconnects >= 1)
            assert "not a tunnel" in manager.snapshot().detail or manager.state in (
                TunnelState.CONNECTING,
                TunnelState.BACKOFF,
            )


async def test_a_listener_is_noticed_before_the_bind_which_windows_would_let_through(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """asyncssh binds with SO_REUSEADDR; on Windows that lets a second listener share the port, so
    the tunnel asks first whether something listens (the other process' tunnel, or a program)."""
    asked: list[tuple[object, ...]] = []
    original = asyncssh.SSHClientConnection.forward_local_port

    async def spy(self: asyncssh.SSHClientConnection, *args: object, **kwargs: object) -> object:
        asked.append(args)
        return await original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(asyncssh.SSHClientConnection, "forward_local_port", spy)
    with socket.socket() as occupied:
        occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        port = int(occupied.getsockname()[1])

        async def healthy() -> StyleHealth:
            return StyleHealth(True, "ok")

        manager = TunnelManager(
            world.target,
            local_port=port,
            remote_port=world.remote.port,
            clock=SystemClock(),
            timings=FAST,
            local_health=healthy,
        )
        async with running(manager):
            await until(manager, TunnelState.EXTERNAL)
        assert asked == []  # nothing was forwarded: the port was left alone


async def test_a_tunnel_of_another_process_is_used_and_taken_over_when_it_goes(
    world: World,
) -> None:
    first, port = make_tunnel(world)
    client = vllm_client(port)
    second = TunnelManager(
        world.target,
        local_port=port,
        remote_port=world.remote.port,
        clock=SystemClock(),
        timings=FAST,
        local_health=client.health,
    )
    async with running(first):
        await until(first, TunnelState.UP)
        async with running(second):
            await until(second, TunnelState.EXTERNAL)
            await first.stop()
            await until(
                second, TunnelState.UP, limit=30
            )  # the port is free: the second one holds it
            assert (await client.health()).ok
    await client.aclose()


async def test_stopping_closes_the_local_port(world: World) -> None:
    manager, port = make_tunnel(world)
    async with running(manager):
        await until(manager, TunnelState.UP)
    assert manager.state is TunnelState.STOPPED
    with pytest.raises(OSError):
        await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), 5)


@pytest.mark.skipif(
    sys.platform == "win32", reason="/proc/uptime is the Linux instance's: no such file on Windows"
)
async def test_the_uptime_of_the_instance_is_read_over_the_same_login(world: World) -> None:
    manager, _ = make_tunnel(world)
    assert await manager.instance_uptime() is None  # not connected yet
    async with running(manager):
        await until(manager, TunnelState.UP)
        uptime = await manager.instance_uptime()
        assert uptime is not None and uptime > 0
        assert "cat /proc/uptime" in world.ssh.state.commands


async def test_an_uptime_that_cannot_be_read_is_none(world: World) -> None:
    world.ssh.state.fail_exec = True
    manager, _ = make_tunnel(world)
    async with running(manager):
        await until(manager, TunnelState.UP)
        assert await manager.instance_uptime() is None


def test_the_text_of_proc_uptime_is_parsed() -> None:
    assert parse_uptime("12345.67 98765.43\n") == 12345.67
    assert parse_uptime("") is None and parse_uptime("n/a") is None and parse_uptime("-1 2") is None
