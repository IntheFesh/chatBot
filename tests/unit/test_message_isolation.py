"""``messages`` holds real messages only (R-STO-007, CLAUDE.md rule 7)."""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import inspect, select

from tests.support.ingest import make_export, run_import
from twin.ingest.corpus import her_messages, her_reproducible_messages
from twin.ingest.events import REPRODUCIBLE_KINDS
from twin.services import Services
from twin.storage.chat_models import Message
from twin.storage.db import Database
from twin.storage.models import Base

SRC = Path(__file__).resolve().parents[2] / "src" / "twin"
# packages that may write to ``messages``: the import, and the storage layer itself
WRITERS = ("ingest", "storage")


def test_messages_and_bot_turns_are_different_tables() -> None:
    assert "messages" in Base.metadata.tables
    columns = set(Base.metadata.tables["messages"].columns.keys())
    assert "source_export_id" in columns
    assert Base.metadata.tables["messages"].c.source_export_id.nullable is False
    # nothing of the bot's own conversation is modelled in the real-message table
    assert not {"bot", "reply_to", "generated", "backend"} & columns


def test_a_row_without_an_export_cannot_be_stored(services: Services) -> None:
    connection = sqlite3.connect(services.paths.db_path)
    try:
        connection.execute(
            "INSERT INTO conversations (id, username, is_group, message_count, created_at, "
            "updated_at) VALUES ('c', x'00', 0, 0, '2026-01-01', '2026-01-01')"
        )
        with pytest.raises(sqlite3.IntegrityError, match="source_export_id"):
            connection.execute(
                "INSERT INTO messages (id, conversation_id, create_time_utc, is_sent, kind, raw, "
                "has_transcript, created_at, updated_at) VALUES ('m', 'c', '2026-01-01', 0, "
                "'text', x'00', 0, '2026-01-01', '2026-01-01')"
            )
    finally:
        connection.close()


def test_the_corpus_queries_read_her_real_messages_only(services: Services, tmp_path: Path) -> None:
    export = make_export(tmp_path, target_messages=300)
    run_import(services, export)
    with services.db.session() as session:
        hers = session.scalars(her_messages()).all()
        assert hers and all(not row.is_sent for row in hers)
        assert len(hers) == sum(n for (_, sent), n in export.counts.items() if not sent)
        times = [row.create_time_utc for row in hers]
        assert times == sorted(times)
        reproducible = session.scalars(her_reproducible_messages()).all()
        assert reproducible and {row.kind for row in reproducible} <= REPRODUCIBLE_KINDS
        assert all(not row.is_sent for row in reproducible)
        mine = session.scalars(select(Message).where(Message.is_sent.is_(True))).all()
        assert mine and {row.id for row in mine}.isdisjoint({row.id for row in hers})
        conversation_id = hers[0].conversation_id
        assert len(session.scalars(her_messages(conversation_id)).all()) == len(hers)
        assert session.scalars(her_messages("no-such-conversation")).all() == []


def test_the_indexes_serve_the_sender_and_time_queries(db: Database) -> None:
    names = {index["name"] for index in inspect(db.engine).get_indexes("messages")}
    assert {"ix_messages_conversation_time", "ix_messages_sent_kind"} <= names


def _writes_messages(tree: ast.AST) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        first = node.args[0] if node.args else None
        targets_message = isinstance(first, ast.Name) and first.id in {"Message", "MESSAGES"}
        if (name in {"insert", "update", "delete"} and targets_message) or (
            isinstance(func, ast.Name) and func.id == "Message"
        ):
            lines.append(node.lineno)
        if isinstance(func, ast.Attribute) and func.attr in {"insert", "update", "delete"}:
            owner = func.value
            if isinstance(owner, ast.Name) and owner.id == "MESSAGES":
                lines.append(node.lineno)
    return lines


def test_only_the_import_writes_to_the_messages_table() -> None:
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        package = path.relative_to(SRC).parts[0]
        if package in WRITERS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders += [f"{path.relative_to(SRC)}:{line}" for line in _writes_messages(tree)]
    assert offenders == []


def test_the_scan_recognises_a_write() -> None:
    sample = ast.parse("insert(Message)\nupdate(Message)\nMESSAGES.insert()\nMessage(id='x')\n")
    assert len(_writes_messages(sample)) == 4
    assert _writes_messages(ast.parse("select(Message)\nprint(Message)\n")) == []
