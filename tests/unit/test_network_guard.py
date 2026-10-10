"""The suite reaches no computer but this one (R-NFR-004): the guard of ``tests/conftest.py``.

"Unit tests do not touch the network" is only true if something stops the test that forgot its
mock.  The guard refuses every connection to an address that is not this computer, records it, and
fails the test when it ends.  These tests check the guard on both kinds of client - blocking
sockets and the event loop - and that a ``respx`` mock, a server on loopback and a Unix socket still
work.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import respx

from tests.support.offline import (
    EXEMPT_BECAUSE,
    NetworkBlocked,
    guard_network,
    is_local,
    network_allowed,
)

NOWHERE = "192.0.2.1"  # TEST-NET-1 (RFC 5737): never a real host


@pytest.mark.parametrize(
    "address",
    [
        ("127.0.0.1", 80),
        ("127.255.0.3", 9),
        ("::1", 80, 0, 0),
        ("::ffff:127.0.0.1", 80, 0, 0),
        ("::1%lo", 80),
        ("localhost", 8080),
        ("LOCALHOST.", 8080),
        ("api.localhost", 1),
        (b"127.0.0.1", 5),
        "/tmp/a-unix-socket",  # a path: the file system
        b"/tmp/a-unix-socket",
    ],
)
def test_this_computer_is_not_the_network(address: object) -> None:
    assert is_local(address)


@pytest.mark.parametrize(
    "address",
    [
        (NOWHERE, 443),
        ("8.8.8.8", 53),
        ("2001:db8::1", 443, 0, 0),
        ("api.deepseek.com", 443),
        ("example.org", 80),
        ("localhost.example.org", 80),
        ("10.0.0.5", 22),  # the local network is a network
        ("0.0.0.0", 80),  # noqa: S104 - an address the guard refuses, not one it listens on
        (None, 1),
        (123, 1),
        (),
        None,
    ],
)
def test_everything_else_is_the_network(address: object) -> None:
    assert not is_local(address)


def test_a_blocking_connection_to_another_computer_is_refused_and_recorded() -> None:
    with guard_network() as guard:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(NetworkBlocked, match="respx") as stopped:
                client.connect((NOWHERE, 443))
            assert isinstance(stopped.value, OSError)  # a client sees "cannot connect"
            with pytest.raises(NetworkBlocked):  # the other way to connect is refused the same
                client.connect_ex(("api.deepseek.com", 443))
        finally:
            client.close()
        assert guard.attempts == [f"connect {NOWHERE}:443", "connect_ex api.deepseek.com:443"]
        assert "api.deepseek.com:443" in guard.report()


def test_a_library_that_connects_by_name_is_refused_at_the_address_it_found() -> None:
    with guard_network() as guard, pytest.raises(NetworkBlocked):
        socket.create_connection(("api.deepseek.com", 443), timeout=1)
    [attempt] = guard.attempts  # (the name is looked up first: the connection is what is stopped)
    assert attempt.startswith("connect ") and attempt.endswith(":443")


def test_a_server_on_loopback_still_answers() -> None:
    with guard_network() as guard:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
            accepted, _ = server.accept()
            client.sendall(b"ping")
            assert accepted.recv(4) == b"ping"
            accepted.close()
        server.close()
    assert guard.attempts == []


@pytest.mark.skipif(sys.platform == "win32", reason="a Unix socket needs a path the platform has")
def test_a_unix_socket_is_this_computer(tmp_path: Path) -> None:
    path = str(tmp_path / "s.sock")
    with guard_network() as guard:
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(path)
        server.listen(1)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(path)
        client.close()
        server.close()
    assert guard.attempts == []


async def test_the_event_loop_connects_to_loopback_and_not_to_another_computer() -> None:
    """``open_connection`` (httpx, asyncssh) goes through the loop, not ``socket.connect``."""

    async def hello(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(b"hello")
        await writer.drain()
        writer.close()

    with guard_network() as guard:
        server = await asyncio.start_server(hello, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        assert await reader.read(5) == b"hello"
        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()
        assert guard.attempts == []
        with pytest.raises(NetworkBlocked):
            await asyncio.open_connection(NOWHERE, 443)
        loop = asyncio.get_running_loop()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(False)
        try:
            with pytest.raises(NetworkBlocked):
                await loop.sock_connect(sock, (NOWHERE, 80))
        finally:
            sock.close()
    assert guard.attempts == [f"connect {NOWHERE}:443", f"connect {NOWHERE}:80"]


def test_an_http_client_that_nobody_mocked_cannot_reach_out() -> None:
    with guard_network() as guard:
        with pytest.raises(httpx.ConnectError) as failed:
            httpx.get(f"http://{NOWHERE}/", timeout=2)
        assert "the tests do not reach the network" in str(failed.value)
    assert guard.attempts == [f"connect {NOWHERE}:80"]


async def test_the_async_http_client_is_stopped_too() -> None:
    with guard_network() as guard:
        async with httpx.AsyncClient(timeout=2) as client:
            with pytest.raises(httpx.ConnectError):
                await client.get("https://api.deepseek.com/v1/models")
    [attempt] = guard.attempts
    assert attempt.endswith(":443")


def test_what_respx_answers_never_reaches_a_socket() -> None:
    with guard_network() as guard, respx.mock(assert_all_called=True) as router:
        route = router.get("https://api.deepseek.com/v1/models").respond(200, json={"data": []})
        assert httpx.get("https://api.deepseek.com/v1/models").json() == {"data": []}
        assert route.call_count == 1
    assert guard.attempts == []


def test_a_route_that_passes_through_to_another_computer_is_stopped() -> None:
    with guard_network() as guard, respx.mock() as router:
        router.get(f"http://{NOWHERE}/").pass_through()
        with pytest.raises(httpx.ConnectError):
            httpx.get(f"http://{NOWHERE}/", timeout=2)
    assert guard.attempts == [f"connect {NOWHERE}:80"]


def test_a_route_that_passes_through_to_loopback_is_let_through() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    answered = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"

    def serve() -> None:
        connection, _ = server.accept()
        connection.recv(4096)
        connection.sendall(answered)
        connection.close()

    import threading

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    with guard_network() as guard, respx.mock() as router:
        router.route(host="127.0.0.1").pass_through()
        assert httpx.get(f"http://127.0.0.1:{port}/", timeout=5).text == "ok"
    thread.join(5)
    server.close()
    assert guard.attempts == []


def test_a_proxy_in_the_environment_does_not_open_a_way_around_the_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")  # a local proxy, as developers have
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:7890")
    with guard_network() as guard:
        assert "HTTPS_PROXY" not in os.environ and os.environ["NO_PROXY"] == "*"
        with pytest.raises(httpx.ConnectError):
            httpx.get("https://api.deepseek.com/v1/models", timeout=2)  # direct, so refused
    assert guard.attempts and guard.attempts[0].endswith(":443")
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7890"  # and put back afterwards


def test_the_patches_are_gone_when_the_block_ends() -> None:
    before = (socket.socket.connect, socket.socket.connect_ex)
    with guard_network():
        assert socket.socket.connect is not before[0]
    assert (socket.socket.connect, socket.socket.connect_ex) == before


def test_a_failure_inside_the_block_still_undoes_the_patches() -> None:
    before = socket.socket.connect
    with pytest.raises(RuntimeError, match="boom"), guard_network():
        raise RuntimeError("boom")
    assert socket.socket.connect is before


# ---------------------------------------------------------------------------------- exemptions


def item(*markers: str) -> SimpleNamespace:
    return SimpleNamespace(get_closest_marker=lambda name: name if name in markers else None)


def test_only_live_tests_and_integration_tests_with_the_switch_may_use_the_network() -> None:
    assert not network_allowed(item(), live_enabled=False)
    assert not network_allowed(item(), live_enabled=True)  # a unit test never
    assert network_allowed(item("live"), live_enabled=False)  # (skipped unless TWIN_LIVE=1)
    assert network_allowed(item("live"), live_enabled=True)
    assert not network_allowed(item("integration"), live_enabled=False)  # R-NFR-004
    assert network_allowed(item("integration"), live_enabled=True)
    assert not network_allowed(item("windows"), live_enabled=True)
    assert set(EXEMPT_BECAUSE) == {"live", "integration"}  # and each exemption says why
    assert all(len(reason) > 40 for reason in EXEMPT_BECAUSE.values())


# ------------------------------------------------------------------------------ the fixture

SAMPLE = textwrap.dedent(
    """
    import socket

    import pytest

    from tests.conftest import no_network  # noqa: F401 - the fixture under test


    def reach() -> None:
        try:
            socket.create_connection(("192.0.2.1", 443), timeout=1)
        except OSError:
            pass  # the client's own handling: "cannot connect"


    def test_a_unit_test_that_forgot_its_mock():
        reach()


    @pytest.mark.live
    def test_a_live_test_may():
        reach()


    def test_a_unit_test_on_loopback():
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        socket.create_connection(server.getsockname(), timeout=1).close()
        server.close()
    """
)


def test_the_fixture_fails_the_test_that_reached_out_even_when_the_client_swallowed_the_error(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "test_sample.py"
    sample.write_text(SAMPLE, encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    environment = {k: v for k, v in os.environ.items() if not k.startswith("TWIN_")}
    environment["PYTHONPATH"] = str(root)
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(sample),
            "-p",
            "no:cacheprovider",
            "-c",
            str(root / "pyproject.toml"),
            "--rootdir",
            str(tmp_path),
            "--import-mode=importlib",
            "-q",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
        cwd=tmp_path,
        check=False,
    )
    output = done.stdout + done.stderr
    assert done.returncode == 1, output
    assert "network access in a test: connect 192.0.2.1:443" in output
    assert "1 error" in output  # (in the teardown of the test that reached out)
    assert "3 passed" in output  # the three bodies ran: the live and the loopback test are fine


def _own_address_other_than_loopback() -> str:
    names = {socket.gethostname(), socket.getfqdn()}
    for name in sorted(names):
        try:
            found = socket.getaddrinfo(name, None, socket.AF_INET)
        except OSError:
            continue
        for _family, _type, _proto, _canon, sockaddr in found:
            if not str(sockaddr[0]).startswith("127."):
                return str(sockaddr[0])
    pytest.skip("this computer has no address other than loopback")


def test_an_address_of_this_computer_is_refused_until_it_is_allowed() -> None:
    own = _own_address_other_than_loopback()
    with guard_network() as guard:
        with pytest.raises(NetworkBlocked):
            guard.check((own, 9), "connect")
        guard.allow_own_address(own)
        guard.check((own, 9), "connect")  # now it is "this computer"
        with pytest.raises(NetworkBlocked):
            guard.check((NOWHERE, 9), "connect")  # and nothing else is
    assert guard.attempts == [f"connect {own}:9", f"connect {NOWHERE}:9"]


def test_a_real_connection_to_an_address_of_this_computer_goes_through_the_fixture(
    allow_own_address: Callable[[str], None],
) -> None:
    own = _own_address_other_than_loopback()
    allow_own_address(own)
    listener = socket.socket()
    listener.bind((own, 0))
    listener.listen(1)
    try:
        with socket.socket() as client:
            client.connect(
                (own, listener.getsockname()[1])
            )  # the packet never leaves this computer
    finally:
        listener.close()


def test_only_an_address_that_this_computer_owns_can_be_allowed() -> None:
    with guard_network() as guard:
        with pytest.raises(ValueError, match="not an address of this computer"):
            guard.allow_own_address(NOWHERE)
        with pytest.raises(ValueError, match="not an IP address"):
            guard.allow_own_address("example.com")
        with pytest.raises(NetworkBlocked):
            socket.socket().connect((NOWHERE, 9))  # the refusal is still in force
    assert guard.attempts == [f"connect {NOWHERE}:9"]
