"""A pytest plugin that makes every database access slow, to see on Linux what Windows shows.

On the ``windows-latest`` runners a thread hop plus a SQLite commit takes tens of milliseconds,
on Linux one or two.  A test that observes the engine "one step too early" - the channel has the
bubble but the state does not note it yet, the row is stored but the message is not queued yet -
passes on Linux by luck of timing and fails there every time.  With the plugin the luck is gone::

    SLOWDB_MS=30 uv run pytest -p tests.support.slow_db tests/unit/test_engine_e2e.py

Every read session starts ``SLOWDB_MS`` late and every write commits ``SLOWDB_MS`` late.  Without
the variable (or with ``0``) the plugin does nothing.  A test must pass with both.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Iterator
from typing import Any

from twin.storage import db as database

DELAY_S = float(os.environ.get("SLOWDB_MS", "0")) / 1000

if DELAY_S > 0:
    _transaction = database.Database.transaction
    _session = database.Database.session

    @contextlib.contextmanager
    def slow_transaction(self: Any, *, bump_state: bool | None = None) -> Iterator[Any]:
        with _transaction(self, bump_state=bump_state) as session:
            yield session
            time.sleep(DELAY_S)  # the commit comes late

    @contextlib.contextmanager
    def slow_session(self: Any) -> Iterator[Any]:
        time.sleep(DELAY_S)  # the read starts late
        with _session(self) as session:
            yield session

    database.Database.transaction = slow_transaction  # type: ignore[method-assign,assignment]
    database.Database.session = slow_session  # type: ignore[method-assign,assignment]
