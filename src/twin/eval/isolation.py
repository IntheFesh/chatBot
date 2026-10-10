"""Write isolation of the evaluation sandbox (R-EVAL-009).

The sandbox runs the code of the running bot, and that code writes: the conversation, the state
machine, the memory.  None of that may happen while a past moment is replayed or while the memory
test asks the bot questions.  The rule is enforced, not just promised:

:func:`isolated_writes` opens a scope, tied to the running task (a context variable, so a worker
thread started from the task is inside it and the other tasks of the process are not).  In the
scope

* a **statement** that inserts into, updates or deletes from a table outside the allowed set -
  or changes the schema - raises :class:`IsolationViolation` before it reaches the database;
* an **object** flushed by an ORM session is checked the same way, and a ``settings`` row only
  passes for the keys the sandbox may touch.

What a sandbox may write (SPEC R-EVAL-009): ``eval_runs`` and ``eval_items`` (its results),
``cost_ledger`` (the calls, ``purpose = eval``), ``jobs`` (the batch jobs and the queued picture
descriptions), ``alerts``, and in ``settings`` only the state version, the token-estimate
calibration and the records of the one-time batches (``onetime.batch.*``, the "batch records" of
R-LLM-014).

:func:`snapshot` and :func:`changes` are the proof used by the tests: the row count and a checksum
of every table before and after a run, and which tables (and settings keys) differ.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Engine, event, inspect
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session

from twin.llm.onetime import STATE_PREFIX
from twin.llm.tokens import CALIBRATION_KEY
from twin.storage.db import Database
from twin.storage.models import Base
from twin.storage.state import STATE_VERSION_KEY

ALLOWED_TABLES = frozenset({"eval_runs", "eval_items", "cost_ledger", "jobs", "alerts", "settings"})
ALLOWED_SETTING_KEYS = frozenset({STATE_VERSION_KEY, CALIBRATION_KEY})
ALLOWED_SETTING_PREFIXES = (STATE_PREFIX,)

_WRITE = re.compile(
    r"^\s*(?:INSERT(?:\s+OR\s+\w+)?\s+INTO|REPLACE\s+INTO|UPDATE(?:\s+OR\s+\w+)?|DELETE\s+FROM)"
    r"\s+[\"`\[]?(\w+)",
    re.IGNORECASE,
)
_SCHEMA = re.compile(r"^\s*(?:CREATE|DROP|ALTER)\b", re.IGNORECASE)


class IsolationViolation(RuntimeError):
    """The sandbox tried to write something it must not."""


@dataclass(frozen=True)
class WriteScope:
    """What may be written inside an isolated block."""

    tables: frozenset[str] = ALLOWED_TABLES
    setting_keys: frozenset[str] = ALLOWED_SETTING_KEYS
    setting_prefixes: tuple[str, ...] = ALLOWED_SETTING_PREFIXES

    def allows_setting(self, key: str) -> bool:
        return key in self.setting_keys or key.startswith(self.setting_prefixes)


SANDBOX_WRITES = WriteScope()
_scope: ContextVar[WriteScope | None] = ContextVar("twin_eval_write_scope", default=None)


def active_scope() -> WriteScope | None:
    """The write scope of the running task, or ``None`` when writes are unrestricted."""
    return _scope.get()


@contextmanager
def isolated_writes(scope: WriteScope = SANDBOX_WRITES) -> Iterator[WriteScope]:
    """Restrict the database writes of the running task to ``scope`` for the block."""
    token = _scope.set(scope)
    try:
        yield scope
    finally:
        _scope.reset(token)


def written_table(statement: str) -> str | None:
    """The table an INSERT / UPDATE / DELETE statement writes to, else ``None``."""
    found = _WRITE.match(statement)
    return found.group(1) if found else None


def _refuse(what: str) -> IsolationViolation:
    return IsolationViolation(f"the evaluation sandbox may not write {what}")


def _check_statement(
    _conn: Connection,
    _cursor: Any,
    statement: str,
    _parameters: Any,
    _context: Any,
    _executemany: bool,
) -> None:
    scope = _scope.get()
    if scope is None:
        return
    if _SCHEMA.match(statement):
        raise _refuse("to the schema")
    table = written_table(statement)
    if table is not None and table not in scope.tables:
        raise _refuse(f"to the table {table!r}")


def _check_flush(session: Session, _context: Any, _instances: Any) -> None:
    scope = _scope.get()
    if scope is None:
        return
    for obj in (
        *session.new,
        *session.deleted,
        *(o for o in session.dirty if session.is_modified(o)),
    ):
        table = getattr(obj, "__tablename__", None)
        if table is None:
            continue
        if table not in scope.tables:
            raise _refuse(f"to the table {table!r}")
        if table == "settings" and not scope.allows_setting(str(getattr(obj, "key", ""))):
            raise _refuse(f"the setting {getattr(obj, 'key', '')!r}")


def install() -> None:
    """Register the two checks (once; they do nothing outside an isolated block)."""
    if not event.contains(Engine, "before_cursor_execute", _check_statement):
        event.listen(Engine, "before_cursor_execute", _check_statement)
    if not event.contains(Session, "before_flush", _check_flush):
        event.listen(Session, "before_flush", _check_flush)


install()


# ------------------------------------------------------------------ the proof


@dataclass(frozen=True)
class TableState:
    rows: int
    checksum: str


@dataclass(frozen=True)
class Snapshot:
    """Row count and checksum of every table, and a checksum per ``settings`` key."""

    tables: dict[str, TableState]
    settings: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Changes:
    """What differs between two snapshots."""

    tables: frozenset[str]
    settings: frozenset[str]

    def outside(self, scope: WriteScope = SANDBOX_WRITES) -> list[str]:
        """The changes a sandbox run must not have made (empty when the run was clean)."""
        found = sorted(table for table in self.tables if table not in scope.tables)
        found += sorted(f"settings:{key}" for key in self.settings if not scope.allows_setting(key))
        return found


def _digest(rows: Any) -> tuple[int, str]:
    sha = hashlib.sha256()
    count = 0
    for row in rows:
        sha.update(repr(tuple(row)).encode("utf-8"))
        sha.update(b"\x00")
        count += 1
    return count, sha.hexdigest()


def snapshot(db: Database) -> Snapshot:
    """Count and checksum every table of the database (for the tests; it reads everything)."""
    tables: dict[str, TableState] = {}
    settings: dict[str, str] = {}
    with db.engine.connect() as connection:
        present = set(inspect(connection).get_table_names())
        for table in Base.metadata.sorted_tables:
            if table.name not in present:
                continue
            rows = connection.exec_driver_sql(f'SELECT * FROM "{table.name}" ORDER BY rowid')
            count, checksum = _digest(rows)
            tables[table.name] = TableState(count, checksum)
        if "settings" in present:
            for row in connection.exec_driver_sql('SELECT * FROM "settings" ORDER BY rowid'):
                settings[str(row[0])] = _digest([row])[1]
    return Snapshot(tables, settings)


def changes(before: Snapshot, after: Snapshot) -> Changes:
    """The tables (and settings keys) whose rows differ between two snapshots."""
    names = set(before.tables) | set(after.tables)
    differing = {name for name in names if before.tables.get(name) != after.tables.get(name)}
    keys = set(before.settings) | set(after.settings)
    moved = {key for key in keys if before.settings.get(key) != after.settings.get(key)}
    return Changes(frozenset(differing), frozenset(moved))
