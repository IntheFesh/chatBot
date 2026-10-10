"""The suite does not reach the network (R-NFR-004): a guard on every way a connection is made.

"Unit tests do not touch the network (``respx`` intercepts); integration tests may use the real
DeepSeek, switched on by an explicit environment variable."  ``respx`` stops an ``httpx`` request
before it reaches a socket, so what is left to catch is the request that nobody mocked - the new
code path that goes to ``api.deepseek.com`` in a test, the library that looks something up.  This
module is that net:

* ``socket.socket.connect`` and ``connect_ex`` (every blocking client: ``urllib``, ``requests``,
  ``smtplib``, ``imaplib`` ...);
* ``sock_connect`` of the event loops (``asyncio.open_connection``, ``httpx.AsyncClient``,
  ``asyncssh``).  On Windows the default loop connects through ``ConnectEx`` and never calls
  ``socket.connect``, so the loops are patched themselves, on every platform.

An attempt to reach anything but this computer is **recorded** and **refused** with an
:class:`OSError` (a client sees "cannot connect" and behaves as it does without a network; the test
then fails when it ends, because the record is not empty).  This computer is: the loopback
addresses (127.0.0.0/8, ::1), the name ``localhost`` and Unix sockets - the test servers of the
suite (an SSH server, a model server, a WeChat double) all listen there.

Subprocesses are not covered (they are other processes); the few tests that start one use
loopback only.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest

LOCAL_NAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"})

# why a kind of test may leave the guard (the fixture in ``tests/conftest.py`` applies this table)
EXEMPT_BECAUSE = {
    "live": (
        "a live test is the test of the real service (DeepSeek, Hugging Face); it is skipped "
        "unless TWIN_LIVE=1 is set, which is the explicit switch of R-NFR-004"
    ),
    "integration": (
        "R-NFR-004: integration tests may use the real DeepSeek when TWIN_LIVE=1 is set; "
        "without it they are guarded like any other test"
    ),
}


def network_allowed(item: Any, live_enabled: bool) -> bool:
    """May this test reach the network?  See :data:`EXEMPT_BECAUSE` for why some may."""
    if item.get_closest_marker("live") is not None:
        return True
    return live_enabled and item.get_closest_marker("integration") is not None


PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def without_proxies(patches: pytest.MonkeyPatch) -> None:
    """Make an ``httpx`` client in a test go direct (and so through the guard).

    A proxy is a computer on loopback to the guard: a connection to it is allowed, and the
    request it forwards is not seen.  Developers behind a proxy (a local one on 127.0.0.1 is
    common) would otherwise reach the real network from a test that forgot its mock.
    ``NO_PROXY=*`` also stops ``httpx`` from reading the proxy of the system (the Windows
    registry).
    """
    for name in PROXY_VARIABLES:
        patches.delenv(name, raising=False)
    patches.setenv("NO_PROXY", "*")
    patches.setenv("no_proxy", "*")


class NetworkBlocked(OSError):
    """A test tried to reach a computer other than this one."""


def is_local(address: Any) -> bool:
    """True for an address that stays on this computer (see the module description)."""
    if isinstance(address, str | bytes | bytearray):
        return True  # a Unix socket path: the file system, not the network
    if not isinstance(address, tuple) or not address:
        return False
    host = address[0]
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    name = host.strip().lower().rstrip(".")
    if name in LOCAL_NAMES or name.endswith(".localhost"):
        return True
    try:
        parsed = ipaddress.ip_address(name.split("%", 1)[0])  # (an IPv6 zone: fe80::1%lo)
    except ValueError:
        return False  # a name that has to be looked up is a lookup on the network
    mapped = getattr(parsed, "ipv4_mapped", None)
    return (mapped or parsed).is_loopback


class NetworkGuard:
    """What the guard saw: the addresses that were refused."""

    def __init__(self) -> None:
        self.attempts: list[str] = []

    def check(self, address: Any, how: str) -> None:
        if not is_local(address):
            shown = f"{address[0]}:{address[1]}" if isinstance(address, tuple) else str(address)
            self.attempts.append(f"{how} {shown}")
            raise NetworkBlocked(
                f"the tests do not reach the network: {how} to {shown} was refused "
                "(mock it with respx, or mark the test `live`)"
            )

    def report(self) -> str:
        return "network access in a test: " + ", ".join(self.attempts)


def _loop_classes() -> list[type]:
    found: list[type] = []
    for module, name in (
        ("asyncio.selector_events", "BaseSelectorEventLoop"),
        ("asyncio.proactor_events", "BaseProactorEventLoop"),
    ):
        try:
            imported = __import__(module, fromlist=[name])
        except ImportError:  # a platform without that loop
            continue
        found.append(getattr(imported, name))
    return found


@contextmanager
def guard_network() -> Iterator[NetworkGuard]:
    """Install the guard for the duration of the block (patches are undone, whatever happens)."""
    guard = NetworkGuard()
    patches = pytest.MonkeyPatch()
    without_proxies(patches)
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def connect(self: socket.socket, address: Any) -> None:
        guard.check(address, "connect")
        original_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        guard.check(address, "connect_ex")
        return original_connect_ex(self, address)

    patches.setattr(socket.socket, "connect", connect)
    patches.setattr(socket.socket, "connect_ex", connect_ex)
    for loop_class in _loop_classes():
        patches.setattr(loop_class, "sock_connect", _guarded_sock_connect(guard, loop_class))
    try:
        yield guard
    finally:
        patches.undo()


def _guarded_sock_connect(
    guard: NetworkGuard, loop_class: type
) -> Callable[[Any, socket.socket, Any], Any]:
    original = loop_class.sock_connect  # type: ignore[attr-defined]

    async def sock_connect(self: Any, sock: socket.socket, address: Any) -> Any:
        guard.check(address, "connect")
        return await original(self, sock, address)

    return sock_connect
