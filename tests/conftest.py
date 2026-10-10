"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from tests.support.clock import ManualClock
from tests.support.embedding import HashingBackend
from tests.support.network import OfflineTransport
from tests.support.offline import NetworkGuard, guard_network, network_allowed
from twin.clock import SystemClock, set_active_clock
from twin.config.loader import load_settings
from twin.config.secrets import SecretStore, select_backend
from twin.config.settings import Settings
from twin.engine.turns import install_bot_turn_reader
from twin.ops.jobs import get_offpeak_policy, set_offpeak_policy
from twin.ops.logging import shutdown_logging
from twin.retrieval import embedder as embedder_module
from twin.retrieval.embedder import reset_embedding_services
from twin.services import CliContext, Services, build_services, set_cli_context
from twin.storage import migrate
from twin.storage.crypto import KeyRing, generate_key, set_active_keyring, use_keyring
from twin.storage.db import Database

# read when the suite starts: the fixture below deletes every TWIN_ variable from the environment
LIVE_ENABLED = os.environ.get("TWIN_LIVE") == "1"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    live_enabled = os.environ.get("TWIN_LIVE") == "1"
    for item in items:
        if "live" in item.keywords and not live_enabled:
            item.add_marker(pytest.mark.skip(reason="live test: set TWIN_LIVE=1 to run"))
        if "windows" in item.keywords and sys.platform != "win32":
            item.add_marker(pytest.mark.skip(reason="runs only on Windows"))


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest) -> Iterator[NetworkGuard | None]:
    """No test reaches a computer other than this one (R-NFR-004; see ``tests/support/offline.py``).

    ``respx`` stops a request before it reaches a socket; this catches the one nobody mocked.
    ``live`` tests, and ``integration`` tests when ``TWIN_LIVE=1`` is set, are exempt - the
    reasons are in ``tests.support.offline.EXEMPT_BECAUSE``.  The fixture keeps no state between
    tests, so it is the same under any sharding of the suite and on every platform.
    """
    if network_allowed(request.node, LIVE_ENABLED):
        yield None
        return
    with guard_network() as guard:
        yield guard
    if guard.attempts:
        pytest.fail(guard.report(), pytrace=False)


@pytest.fixture
def allow_own_address(no_network: NetworkGuard | None) -> Callable[[str], None]:
    """Let the test connect to one of this computer's own non-loopback addresses.

    For the test that proves a server is not reachable by the computer's other address; see
    :meth:`tests.support.offline.NetworkGuard.allow_own_address` for why it is still offline.
    """

    def allow(host: str) -> None:
        if no_network is not None:  # live tests are not guarded
            no_network.allow_own_address(host)

    return allow


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Keep every test away from the real home, config, credential store and clock."""
    for name in list(os.environ):
        if name.startswith("TWIN_"):
            monkeypatch.delenv(name)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("TWIN_HOME", str(home))
    monkeypatch.setenv("TWIN_SECRETS_DIR", str(tmp_path / "secrets"))
    monkeypatch.setenv("TWIN_KEYRING_BACKEND", "file")  # never the real credential manager
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("COLUMNS", "200")  # keep rich tables on one line in captured output
    previous_policy = get_offpeak_policy()  # building an application registers the real one
    # the application registers the reader of ``bot_turns`` once, when twin.engine.turns is
    # imported, for the whole process: a test that unregisters or replaces it (the memory and
    # schedule tests do) must not take it away from the tests after it, so every test starts with
    # what the application has, whatever ran before and whichever modules this run imported
    install_bot_turn_reader()
    set_cli_context(CliContext(http_transport=OfflineTransport()))  # `twin doctor` stays offline
    yield home
    install_bot_turn_reader()
    set_offpeak_policy(previous_policy)
    shutdown_logging()
    set_cli_context(None)
    set_active_keyring(None)
    set_active_clock(SystemClock())


@pytest.fixture
def clock() -> ManualClock:
    manual = ManualClock()
    set_active_clock(manual)
    return manual


@pytest.fixture
def keyring_ring() -> Iterator[KeyRing]:
    ring = KeyRing({1: generate_key()}, 1)
    with use_keyring(ring):
        yield ring


@pytest.fixture
def secret_store() -> SecretStore:
    backend, info = select_backend()
    return SecretStore(backend, info)


@pytest.fixture
def settings() -> Settings:
    return load_settings()


@pytest.fixture
def db(tmp_path: Path, keyring_ring: KeyRing, clock: ManualClock) -> Iterator[Database]:
    """A migrated database with an active key ring and manual clock."""
    path = tmp_path / "data" / "test.db"
    migrate.upgrade(path)
    database = Database(path, clock=clock)
    yield database
    database.dispose()


@pytest.fixture
def services(tmp_path: Path, clock: ManualClock, secret_store: SecretStore) -> Iterator[Services]:
    """Full services container on a fresh migrated database."""
    home = Path(os.environ["TWIN_HOME"])
    built_settings = load_settings(None, {"paths": {"data_dir": str(tmp_path / "data")}})
    migrate.upgrade(Path(built_settings.paths.data_dir) / "twin.db")
    container = build_services(built_settings, root=home, secrets=secret_store, clock=clock)
    yield container
    container.close()


@pytest.fixture
def embedder(monkeypatch: pytest.MonkeyPatch) -> Iterator[HashingBackend]:
    """The tiny offline embedding model, installed as the model of the retrieval library."""
    backend = HashingBackend()
    monkeypatch.setattr(embedder_module, "backend_factory", lambda config, paths: backend)
    reset_embedding_services()
    yield backend
    reset_embedding_services()
