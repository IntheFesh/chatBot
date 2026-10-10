"""A small synthetic installation to back up, restore and delete (round 12 tests).

Everything is invented: a few facts with vectors (the memory of the bot), a few jobs and settings
with encrypted payloads, two media files.  The texts carry a marker that must never be found in a
backup file, in a log or in a mail.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tests.support.memory import add_fact, make_memory, utc
from twin.ops.backup.service import BackupService
from twin.retrieval.vector_store import VectorStore
from twin.services import Services
from twin.storage.media import MediaKind
from twin.storage.memory_models import Fact
from twin.storage.models import Job, Setting
from twin.storage.vector_schema import FACT_SCHEMA

MARKER = "绝密测试暗号"
IMAGE_BYTES = (MARKER.encode("utf-8") + bytes(range(256))) * 30
STICKER_BYTES = b"\x89STICKER" + bytes(reversed(range(256))) * 12


@dataclass
class BackupWorld:
    """What :func:`build_backup_world` put into the installation."""

    services: Services
    fact_texts: list[str]
    media: dict[str, bytes] = field(default_factory=dict)  # sha256 -> content
    backup: BackupService = field(init=False)

    def __post_init__(self) -> None:
        self.backup = BackupService.from_services(self.services)


def build_backup_world(services: Services, *, facts: int = 6) -> BackupWorld:
    """Facts (with vectors), jobs, settings and two media files."""
    memory = make_memory(services)
    texts = [f"{MARKER}-事实-{n}-她喜欢第{n}号的小店" for n in range(facts)]
    for number, text in enumerate(texts):
        add_fact(memory, text, utc(2026, 3, 10 + number))
    with services.db.transaction() as session:
        for number in range(8):
            session.add(Job(type="demo", payload={"n": number, "text": f"{MARKER}-任务{number}"}))
        session.add(Setting(key="demo.alpha", value={"note": f"{MARKER}-设置"}))
    stored = {
        services.media.put(IMAGE_BYTES, MediaKind.IMAGE).sha256: IMAGE_BYTES,
        services.media.put(STICKER_BYTES, MediaKind.STICKER).sha256: STICKER_BYTES,
    }
    return BackupWorld(services, texts, stored)


def table_counts(db_path: Path) -> dict[str, int]:
    """Row count of every table of a database file (plain SQL; the values stay sealed)."""
    connection = sqlite3.connect(str(db_path))
    try:
        names = [
            str(name)
            for (name,) in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            if name not in ("alembic_version", "sqlite_sequence")
        ]
        return {
            name: int(connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
            for name in sorted(names)
        }
    finally:
        connection.close()


def vector_ids(services: Services) -> list[str]:
    """The ids in the fact vector table of the installation."""
    table = VectorStore(services.paths.vectors_dir).table(FACT_SCHEMA)
    return sorted(table.ids()) if table.exists() else []


def fact_texts(services: Services) -> list[str]:
    """The decrypted fact texts of the installation, sorted."""
    from sqlalchemy import select

    with services.db.session() as session:
        return sorted(str(fact.text) for fact in session.scalars(select(Fact)))


def on_day(offset: int, base: datetime | None = None) -> datetime:
    """15:00 UTC on a day ``offset`` days after ``base`` (2026-09-01): 10:00 in Chicago."""
    first = base or datetime(2026, 9, 1, 15, tzinfo=UTC)
    return first + timedelta(days=offset)
