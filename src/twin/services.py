"""The services container and the per-invocation CLI context.

:func:`build_services` wires configuration, credential store, encryption keys,
database and runtime settings.  It is the only place that creates the first
database master key, and only when the database holds no data yet (a missing key
for an existing database is reported, never silently replaced).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select

from twin.clock import Clock, SystemClock, set_active_clock
from twin.config.loader import DataPaths, load_settings, resolve_paths
from twin.config.runtime import RuntimeSettings
from twin.config.secrets import SecretStore
from twin.config.settings import Settings
from twin.ops.alerts import AlertSink, DbAlertSink
from twin.storage.crypto import KeyRing, set_active_keyring
from twin.storage.db import Database
from twin.storage.keystore import KeyStore
from twin.storage.media import MediaStore
from twin.storage.migrate import SchemaState, require_current_schema, schema_status
from twin.storage.rotate import encrypted_tables


@dataclass
class Services:
    """Everything a command or component needs, built once per process."""

    settings: Settings
    paths: DataPaths
    clock: Clock
    secrets: SecretStore
    db: Database
    keystore: KeyStore
    keyring: KeyRing
    runtime: RuntimeSettings
    media: MediaStore
    alerts: AlertSink

    def close(self) -> None:
        self.db.dispose()


def database_has_data(db: Database) -> bool:
    """True if any table with encrypted columns already holds a row."""
    with db.session() as session:
        return any(
            session.execute(select(spec.pk[0]).limit(1)).first() is not None
            for spec in encrypted_tables()
        )


def build_services(
    settings: Settings,
    *,
    root: Path | None = None,
    secrets: SecretStore | None = None,
    clock: Clock | None = None,
    require_schema: bool = True,
    allow_create_key: bool | None = None,
) -> Services:
    """Create the services container.

    ``require_schema`` raises :class:`~twin.storage.migrate.SchemaOutdatedError` (with the
    command to run) unless the database is at the current migration.
    ``allow_create_key`` overrides the default rule "create a master key only for an
    empty database".
    """
    paths = resolve_paths(settings, root)
    paths.ensure()
    the_clock = clock or SystemClock()
    set_active_clock(the_clock)
    store = secrets or SecretStore.default()
    if require_schema:
        require_current_schema(paths.db_path)
    db = Database(paths.db_path, clock=the_clock)
    keystore = KeyStore(store)
    if allow_create_key is None:
        status = schema_status(paths.db_path)
        allow_create_key = status.state is not SchemaState.CURRENT or not database_has_data(db)
    ring = keystore.load_or_create(allow_create=allow_create_key)
    set_active_keyring(ring)
    return Services(
        settings=settings,
        paths=paths,
        clock=the_clock,
        secrets=store,
        db=db,
        keystore=keystore,
        keyring=ring,
        runtime=RuntimeSettings(db, settings, the_clock),
        media=MediaStore(paths.media_dir, paths.tmp_dir),
        alerts=DbAlertSink(db, the_clock),
    )


@dataclass
class CliContext:
    """Options given to the root command, plus lazily built settings and services."""

    config_path: Path | None = None
    overrides: dict[str, Any] = field(default_factory=dict)
    log_level: str = "INFO"
    secrets: SecretStore | None = None  # injected by tests; otherwise the default store
    http_transport: httpx.BaseTransport | None = None  # injected by tests; network checks only
    _settings: Settings | None = field(default=None, repr=False)
    _services: Services | None = field(default=None, repr=False)

    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings(self.config_path, self.overrides)
        return self._settings

    def paths(self) -> DataPaths:
        return resolve_paths(self.settings())

    def secret_store(self) -> SecretStore:
        if self.secrets is None:
            self.secrets = SecretStore.default()
        return self.secrets

    def services(self) -> Services:
        if self._services is None:
            self._services = build_services(
                self.settings(), root=self.paths().root, secrets=self.secret_store()
            )
        return self._services

    def reset(self) -> None:
        if self._services is not None:
            self._services.close()
        self._services = None
        self._settings = None


_cli_context: CliContext | None = None


def get_cli_context() -> CliContext:
    global _cli_context
    if _cli_context is None:
        _cli_context = CliContext()
    return _cli_context


def set_cli_context(context: CliContext | None) -> None:
    global _cli_context
    _cli_context = context
