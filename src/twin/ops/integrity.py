"""The weekly integrity check: SQLite and the vector index (R-OPS-003).

Every Sunday at 03:30 local time the monitor runs :func:`run_integrity`:

* ``PRAGMA integrity_check`` on the database (the SQLite page-level check);
* for each vector table that exists, every id in the index must be a row of the database table
  it was made from (``example_windows``, ``facts``, ``daily_summaries``, ``stickers``).  An index
  row whose source is gone only wastes space, but it means the index and the database drifted
  apart - a restored backup, a crash between the two writes - so it is reported with the command
  that repairs it.

The result is one :class:`IntegrityReport`; the monitor keeps it in the ``ops.integrity.last``
setting and raises ``integrity_failed`` when it is not clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import InstrumentedAttribute

from twin.retrieval.vector_store import VectorStore
from twin.storage.chat_models import Sticker
from twin.storage.db import Database
from twin.storage.memory_models import DailySummary, Fact
from twin.storage.retrieval_models import ExampleWindow
from twin.storage.vector_schema import (
    FACT_SCHEMA,
    STICKER_SCHEMA,
    SUMMARY_SCHEMA,
    WINDOW_SCHEMA,
    VectorTableSchema,
)

MAX_MESSAGES = 20
REPAIRS = {
    "example_windows": "twin retrieval rebuild",
    "memory_facts": "twin memory reindex",
    "memory_summaries": "twin memory reindex",
    "sticker_descriptions": "twin stickers tag-all",
}


@dataclass(frozen=True)
class VectorSource:
    """A vector table and the database column its ids must be found in."""

    schema: VectorTableSchema
    column: InstrumentedAttribute[str]


VECTOR_SOURCES = (
    VectorSource(WINDOW_SCHEMA, ExampleWindow.id),
    VectorSource(FACT_SCHEMA, Fact.id),
    VectorSource(SUMMARY_SCHEMA, DailySummary.id),
    VectorSource(STICKER_SCHEMA, Sticker.md5),
)


@dataclass(frozen=True)
class TableCheck:
    """One vector table: how many rows it holds and how many have no source row."""

    table: str
    rows: int
    orphans: int

    @property
    def ok(self) -> bool:
        return self.orphans == 0

    def to_json(self) -> dict[str, Any]:
        return {"table": self.table, "rows": self.rows, "orphans": self.orphans}


@dataclass(frozen=True)
class IntegrityReport:
    """The outcome of one run."""

    at: datetime
    database_ok: bool
    database_messages: tuple[str, ...]
    vectors: tuple[TableCheck, ...]

    @property
    def ok(self) -> bool:
        return self.database_ok and all(check.ok for check in self.vectors)

    def problems(self) -> list[str]:
        """What is wrong, one line each, with the repair command where there is one."""
        lines = [
            f"database: {message}" for message in self.database_messages if not self.database_ok
        ]
        for check in self.vectors:
            if not check.ok:
                fix = REPAIRS.get(check.table, "rebuild the index")
                lines.append(f"{check.table}: {check.orphans} id(s) not in the database; run {fix}")
        return lines

    def to_json(self) -> dict[str, Any]:
        return {
            "at": self.at.isoformat(),
            "ok": self.ok,
            "database_ok": self.database_ok,
            "database_messages": list(self.database_messages),
            "vectors": [check.to_json() for check in self.vectors],
        }


def check_database(db: Database) -> tuple[bool, tuple[str, ...]]:
    """``PRAGMA integrity_check``: ``(clean, messages)``.

    A file so damaged that SQLite cannot even run the check (``database disk image is
    malformed``) is a result too, not an exception.
    """
    try:
        with db.engine.connect() as connection:
            rows = connection.exec_driver_sql("PRAGMA integrity_check").fetchall()
    except DBAPIError as exc:
        return False, (f"SQLite cannot read the database: {exc.orig}",)
    messages = tuple(str(row[0]) for row in rows)[:MAX_MESSAGES]
    return messages == ("ok",), messages


def check_vectors(db: Database, store: VectorStore) -> tuple[TableCheck, ...]:
    """Every vector id of every existing table must be a row of its source table."""
    checks: list[TableCheck] = []
    for source in VECTOR_SOURCES:
        table = store.table(source.schema)
        if not table.exists():
            continue
        indexed = table.ids()
        with db.session() as session:
            known = set(session.scalars(select(source.column)))
        orphans = sum(1 for value in indexed if value not in known)
        checks.append(TableCheck(source.schema.name, len(indexed), orphans))
    return tuple(checks)


def run_integrity(db: Database, store: VectorStore, now: datetime) -> IntegrityReport:
    """Run both checks (blocking: call it from a worker thread)."""
    clean, messages = check_database(db)
    return IntegrityReport(now, clean, messages, check_vectors(db, store))
