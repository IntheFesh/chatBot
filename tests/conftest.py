"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.support.clock import ManualClock
from twin.clock import SystemClock, set_active_clock
from twin.config.loader import load_settings
from twin.config.secrets import SecretStore, select_backend
from twin.config.settings import Settings
from twin.ops.logging import shutdown_logging
from twin.services import Services, build_services, set_cli_context
from twin.storage import migrate
from twin.storage.crypto import KeyRing, generate_key, set_active_keyring, use_keyring
from twin.storage.db import Database


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    live_enabled = os.environ.get("TWIN_LIVE") == "1"
    for item in items:
        if "live" in item.keywords and not live_enabled:
            item.add_marker(pytest.mark.skip(reason="live test: set TWIN_LIVE=1 to run"))
        if "windows" in item.keywords and sys.platform != "win32":
            item.add_marker(pytest.mark.skip(reason="runs only on Windows"))


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
    yield home
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
