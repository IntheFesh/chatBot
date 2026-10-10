"""Alembic helpers: upgrade, schema status and the start-up version check.

Migrations live in ``twin/storage/migrations``; every round adds its tables as
a new revision.  Start-up never migrates silently: :func:`require_current_schema`
raises an error that tells the user to run ``twin db upgrade`` (or
``alembic upgrade head``), which is an EXCLUSIVE command (R-ARCH-006).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


class SchemaState(StrEnum):
    MISSING = "missing"  # no database file yet
    EMPTY = "empty"  # file exists, no migrations applied
    OUTDATED = "outdated"
    CURRENT = "current"
    AHEAD = "ahead"  # database is newer than this code


@dataclass(frozen=True)
class SchemaStatus:
    state: SchemaState
    current: str | None
    head: str

    @property
    def ok(self) -> bool:
        return self.state is SchemaState.CURRENT

    def hint(self) -> str:
        match self.state:
            case SchemaState.MISSING | SchemaState.EMPTY:
                return (
                    "database is not initialised: run `twin db upgrade` (or `alembic upgrade head`)"
                )
            case SchemaState.OUTDATED:
                return (
                    f"database schema {self.current} is older than {self.head}: "
                    "stop the application (`twin service stop`) and run `twin db upgrade` "
                    "(or `alembic upgrade head`)"
                )
            case SchemaState.AHEAD:
                return (
                    f"database schema {self.current} is newer than this program ({self.head}): "
                    "update the program"
                )
            case SchemaState.CURRENT:
                return f"database schema is up to date ({self.head})"


class SchemaOutdatedError(RuntimeError):
    """The database schema does not match the program version."""

    def __init__(self, status: SchemaStatus) -> None:
        self.status = status
        super().__init__(status.hint())


def alembic_config(db_path: Path) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.attributes["db_path"] = db_path
    config.attributes["programmatic"] = True
    return config


def head_revision() -> str:
    head = ScriptDirectory.from_config(alembic_config(Path("unused.db"))).get_current_head()
    if head is None:
        raise RuntimeError("no migration revisions found")
    return head


def revision_history() -> list[str]:
    """All revision ids, oldest first."""
    scripts = ScriptDirectory.from_config(alembic_config(Path("unused.db")))
    return [rev.revision for rev in reversed(list(scripts.walk_revisions()))]


def current_revision(db_path: Path) -> str | None:
    """The applied revision, or ``None`` if the database is missing or unmigrated."""
    if not db_path.exists():
        return None
    connection = sqlite3.connect(db_path, timeout=20)
    try:
        row = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        connection.close()
    return str(row[0]) if row else None


def schema_status(db_path: Path) -> SchemaStatus:
    head = head_revision()
    if not db_path.exists():
        return SchemaStatus(SchemaState.MISSING, None, head)
    current = current_revision(db_path)
    if current is None:
        return SchemaStatus(SchemaState.EMPTY, None, head)
    if current == head:
        return SchemaStatus(SchemaState.CURRENT, current, head)
    history = revision_history()
    if current in history and history.index(current) < history.index(head):
        return SchemaStatus(SchemaState.OUTDATED, current, head)
    return SchemaStatus(SchemaState.AHEAD, current, head)


def require_current_schema(db_path: Path) -> SchemaStatus:
    status = schema_status(db_path)
    if not status.ok:
        raise SchemaOutdatedError(status)
    return status


def upgrade(db_path: Path, revision: str = "head") -> None:
    command.upgrade(alembic_config(db_path), revision)


def downgrade(db_path: Path, revision: str) -> None:
    command.downgrade(alembic_config(db_path), revision)
