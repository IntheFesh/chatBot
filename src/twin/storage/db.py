"""SQLite engine, sessions and transactions (R-STO-001, R-ARCH-006.4).

* WAL journal, ``foreign_keys=ON`` and a ``busy_timeout`` on every connection;
* explicit transaction control: read sessions use a deferred ``BEGIN`` (many
  readers alongside one writer under WAL), write transactions use
  ``BEGIN IMMEDIATE`` so two processes never deadlock on a lock upgrade and a
  writer simply waits (``busy_timeout``) for the other one to finish;
* write transactions are kept short; callers never hold one across ``await``;
* the process model (R-ARCH-006) plugs in through :data:`write_policy`:
  READ commands may not write, LIGHT commands bump ``settings.state_version``
  in the same transaction as their writes.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session

from twin.clock import Clock, get_clock
from twin.storage.state import bump_state_version

T = TypeVar("T")

DEFAULT_BUSY_TIMEOUT_MS = 20_000
_IMMEDIATE_OPTION = "twin_immediate"


class ReadOnlyViolationError(RuntimeError):
    """A READ-class command tried to write to the database."""


@dataclass(frozen=True)
class WritePolicy:
    """Constraints the current CLI command places on database writes."""

    read_only: bool = False
    bump_state: bool = False


_write_policy: ContextVar[WritePolicy | None] = ContextVar("twin_write_policy", default=None)
DEFAULT_WRITE_POLICY = WritePolicy()


def get_write_policy() -> WritePolicy:
    """The write policy of the current command (unrestricted by default)."""
    return _write_policy.get() or DEFAULT_WRITE_POLICY


@contextmanager
def use_write_policy(policy: WritePolicy) -> Iterator[WritePolicy]:
    token = _write_policy.set(policy)
    try:
        yield policy
    finally:
        _write_policy.reset(token)


def _on_connect(dbapi_connection: sqlite3.Connection, busy_timeout_ms: int) -> None:
    # pysqlite must not emit BEGIN on its own; transaction control is explicit below.
    dbapi_connection.isolation_level = None
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


def _on_begin(conn: Connection) -> None:
    immediate = bool(conn.get_execution_options().get(_IMMEDIATE_OPTION))
    conn.exec_driver_sql("BEGIN IMMEDIATE" if immediate else "BEGIN")


def create_sqlite_engine(path: Path, *, busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS) -> Engine:
    """Create an engine for ``path`` with the project's SQLite settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{path.as_posix()}",
        connect_args={"check_same_thread": False, "timeout": busy_timeout_ms / 1000},
    )

    @event.listens_for(engine, "connect")
    def _connect(dbapi_connection: Any, _record: Any) -> None:
        _on_connect(dbapi_connection, busy_timeout_ms)

    event.listen(engine, "begin", _on_begin)
    return engine


class Database:
    """A SQLite database file with read sessions and write transactions."""

    def __init__(
        self,
        path: Path,
        *,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        clock: Clock | None = None,
    ) -> None:
        self.path = path
        self._clock = clock
        self.engine = create_sqlite_engine(path, busy_timeout_ms=busy_timeout_ms)
        self._write_engine = self.engine.execution_options(**{_IMMEDIATE_OPTION: True})

    @property
    def clock(self) -> Clock:
        return self._clock or get_clock()

    @contextmanager
    def session(self) -> Iterator[Session]:
        """A read session (deferred BEGIN).  Nothing is committed on exit."""
        with Session(self.engine, expire_on_commit=False) as session:
            try:
                yield session
            finally:
                session.rollback()

    @contextmanager
    def transaction(self, *, bump_state: bool | None = None) -> Iterator[Session]:
        """A short write transaction (``BEGIN IMMEDIATE``).

        Commits on success and rolls back on any exception.  When ``bump_state``
        is true (the default inside LIGHT commands) ``settings.state_version`` is
        incremented in the same transaction (R-ARCH-006.2).
        """
        policy = get_write_policy()
        if policy.read_only:
            raise ReadOnlyViolationError(
                "this command is declared read-only (READ) but attempted a database write"
            )
        do_bump = policy.bump_state if bump_state is None else bump_state
        with Session(self._write_engine, expire_on_commit=False) as session:
            try:
                yield session
                if do_bump:
                    bump_state_version(session, self.clock)
                session.commit()
            except BaseException:
                session.rollback()
                raise

    async def arun(self, func: Callable[[Session], T], *, write: bool = False) -> T:
        """Run ``func(session)`` in a worker thread so the event loop never blocks."""

        def _call() -> T:
            if write:
                with self.transaction() as session:
                    return func(session)
            with self.session() as session:
                return func(session)

        return await asyncio.to_thread(_call)

    def dispose(self) -> None:
        self.engine.dispose()

    def exists(self) -> bool:
        return self.path.exists()
