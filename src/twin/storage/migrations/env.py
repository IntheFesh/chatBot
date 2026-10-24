"""Alembic environment.

The database location comes from (in order): ``config.attributes["db_path"]``
(programmatic use by ``twin db upgrade``), the ``sqlalchemy.url`` option, or the
project settings (``paths.data_dir``) when invoked as ``alembic upgrade head``.
"""

from __future__ import annotations

from logging.config import fileConfig
from pathlib import Path

from alembic import context

from twin.storage.db import create_sqlite_engine
from twin.storage.models import Base

config = context.config
target_metadata = Base.metadata

if config.config_file_name is not None and not config.attributes.get("programmatic"):
    fileConfig(config.config_file_name, disable_existing_loggers=False)


def _db_path() -> Path:
    explicit = config.attributes.get("db_path")
    if explicit is not None:
        return Path(explicit)
    url = config.get_main_option("sqlalchemy.url")
    if url and url.startswith("sqlite:///"):
        return Path(url.removeprefix("sqlite:///"))
    from twin.config.loader import load_settings, resolve_paths

    return resolve_paths(load_settings()).db_path


def run_migrations_offline() -> None:
    context.configure(
        url=f"sqlite:///{_db_path().as_posix()}",
        target_metadata=target_metadata,
        literal_binds=True,
        render_as_batch=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_sqlite_engine(_db_path())
    try:
        with engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                render_as_batch=True,
                compare_type=True,
                transactional_ddl=True,
            )
            with context.begin_transaction():
                context.run_migrations()
            connection.commit()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
