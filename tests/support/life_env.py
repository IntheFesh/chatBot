"""A life world outside pytest: for the long-run script and the runtime measurements.

The scenario tests get their world from fixtures (``tests/integration/life/conftest.py``).  The
development tools (``scripts/soak.py``, ``scripts/bench_runtime.py``) have no fixtures, so this
module builds the same things once, in the same order: the isolated environment, the migrated data
directory, the services on a :class:`~tests.support.life_clock.LifeClock`, the offline embedding
model, the router that stands for the network, and the :class:`~tests.support.life_world.LifeWorld`
on top.  Nothing here is imported by ``src``.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import respx

from tests.support.embedding import HashingBackend
from tests.support.life_clock import LifeClock
from tests.support.life_world import LifeWorld
from twin.clock import set_active_clock
from twin.config.loader import load_settings
from twin.config.secrets import SecretStore, select_backend
from twin.engine.turns import install_bot_turn_reader
from twin.retrieval import embedder as embedder_module
from twin.retrieval.embedder import reset_embedding_services
from twin.services import Services, build_services
from twin.storage import migrate


def isolate(home: Path) -> None:
    """Keep the process away from the real home, credentials and clock (as the tests do)."""
    for name in list(os.environ):
        if name.startswith("TWIN_"):
            del os.environ[name]
    os.environ["TWIN_HOME"] = str(home)
    os.environ["TWIN_SECRETS_DIR"] = str(home / "secrets")
    os.environ["TWIN_KEYRING_BACKEND"] = "file"
    os.environ["XDG_CONFIG_HOME"] = str(home / "xdg")
    os.environ["COLUMNS"] = "160"


@dataclass
class LifeEnvironment:
    """What :func:`life_environment` hands over."""

    world: LifeWorld
    services: Services
    clock: LifeClock
    embedder: HashingBackend
    api: respx.MockRouter
    home: Path


@asynccontextmanager
async def life_environment(
    home: Path, *, start: datetime, **options: Any
) -> AsyncIterator[LifeEnvironment]:
    """Build a started :class:`LifeWorld` below ``home`` (call :func:`isolate` first).

    ``options`` go to :meth:`LifeWorld.create` (``model``, ``platform``, ``history_days`` ...).
    Everything is closed and every process-wide slot given back when the block ends.
    """
    clock = LifeClock()
    set_active_clock(clock)
    install_bot_turn_reader()
    settings = load_settings(None, {"paths": {"data_dir": str(home / "data")}})
    migrate.upgrade(Path(settings.paths.data_dir) / "twin.db")
    backend, info = select_backend()
    services = build_services(settings, root=home, secrets=SecretStore(backend, info), clock=clock)
    embedder = HashingBackend()
    original_factory = embedder_module.backend_factory
    embedder_module.backend_factory = lambda config, paths: embedder
    reset_embedding_services()
    pictures = home / "pictures"
    pictures.mkdir(exist_ok=True)
    try:
        with respx.mock(assert_all_called=False) as api:
            world = await LifeWorld.create(
                services, clock, embedder, api, workdir=pictures, start=start, **options
            )
            try:
                yield LifeEnvironment(world, services, clock, embedder, api, home)
            finally:
                await world.close()
    finally:
        embedder_module.backend_factory = original_factory
        reset_embedding_services()
        services.close()
