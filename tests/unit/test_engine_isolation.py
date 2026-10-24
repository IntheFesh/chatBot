"""What the bot says never becomes what she sounds like (R-STO-007, CLAUDE.md rule 7, R-CMD-001)."""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import inspect, select

from tests.support.embedding import HashingBackend
from tests.support.synth_chat import ChatSpec, build_chat
from twin.engine.turns import BotTurnStore, OutboundBubble, ReplyMeta
from twin.ingest.corpus import (
    conversation_messages,
    conversation_skeleton,
    her_messages,
    her_reproducible_messages,
)
from twin.profile.api import load_profile
from twin.profile.builder import rebuild
from twin.retrieval.embedder import EmbeddingService
from twin.retrieval.indexer import run_index
from twin.retrieval.query import ExampleQuery, ExampleRetriever, QueryTurn
from twin.services import Services
from twin.storage.chat_models import Message
from twin.storage.models import Base

SRC = Path(__file__).resolve().parents[2] / "src" / "twin"
MARKER = "机器人专属暗号火锅"
COMMAND_MARKER = "指令回复专属暗号"
# the only code that may touch the table of the bot's conversation
OWNERS = ("engine/", "storage/", "stickers/rate.py")
FORBIDDEN_PACKAGES = ("ingest", "profile", "retrieval", "training", "eval", "learning")


def test_the_two_tables_share_nothing() -> None:
    bot, real = Base.metadata.tables["bot_turns"], Base.metadata.tables["messages"]
    assert not any(fk.column.table is bot for fk in real.foreign_keys)
    assert not any(fk.column.table is real for fk in bot.foreign_keys)
    assert "source_export_id" in real.columns and "source_export_id" not in bot.columns
    assert "is_sent" in real.columns and "is_sent" not in bot.columns  # who spoke is a direction


def test_only_the_engine_and_the_stores_name_the_table_of_the_bots_conversation() -> None:
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        if relative.startswith(OWNERS) or relative in OWNERS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module, *(alias.name for alias in node.names)]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.Name | ast.Attribute):
                names = [node.id if isinstance(node, ast.Name) else node.attr]
            if any(
                name in {"BotTurn", "engine_models", "twin.storage.engine_models"} for name in names
            ):
                offenders.append(f"{relative}:{node.lineno}")
    assert offenders == []


def names_in(tree: ast.AST) -> set[str]:
    """Identifiers, imported names and string values of the code (docstrings left out)."""
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            found.update(alias.name for alias in node.names)
            found.add(node.module or "")
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            found.add(node.value)
    return found


def test_style_retrieval_and_training_packages_never_name_the_conversation() -> None:
    """R-RET-004, R-TRN-004: no import, name or table string of the bot's conversation there."""
    banned = {"BotTurn", "bot_turns", "engine_models", "twin.storage.engine_models"}
    offenders = []
    for package in FORBIDDEN_PACKAGES:
        for path in sorted((SRC / package).rglob("*.py")):
            used = names_in(ast.parse(path.read_text(encoding="utf-8")))
            if banned & used or any("bot_turns" in name for name in used):
                offenders.append(path.relative_to(SRC).as_posix())
    assert offenders == []


def test_the_scan_sees_a_table_name_hidden_in_code() -> None:
    hidden = names_in(ast.parse("rows = session.execute('select * from bot_turns')"))
    assert any("bot_turns" in name for name in hidden)
    assert names_in(ast.parse('"""the bot_turns table"""\nx = 1')) == {"x"}


def test_the_corpus_queries_do_not_read_the_bots_table() -> None:
    used = names_in(ast.parse((SRC / "ingest" / "corpus.py").read_text(encoding="utf-8")))
    assert "BotTurn" not in used and not any("bot_turns" in name for name in used)
    for statement in (her_messages(), her_reproducible_messages(), conversation_messages()):
        assert "bot_turns" not in str(statement)
    assert "bot_turns" not in str(conversation_skeleton())


async def test_nothing_the_bot_said_reaches_her_style_her_library_or_her_messages(
    services: Services, embedder: HashingBackend
) -> None:
    services.settings.retrieval.model = embedder.info.model
    build_chat(services, ChatSpec(days=40))
    store = BotTurnStore(services.db, services.clock)
    start = datetime(2026, 9, 1, 12, tzinfo=UTC)
    for number in range(30):
        at = start + timedelta(minutes=10 * number)
        store.add_inbound(at=at, kind="text", text=f"{MARKER}用户{number}")
        store.add_reply(
            [OutboundBubble(f"{MARKER}回复{number}", at + timedelta(minutes=1))],
            ReplyMeta("deepseek"),
        )
        store.add_reply(
            [OutboundBubble(COMMAND_MARKER, at + timedelta(minutes=2))],
            ReplyMeta("command"),
            is_command=True,
        )

    with services.db.session() as session:
        real = list(session.scalars(select(Message)))
        assert real and not any(
            MARKER in (m.text or "") or COMMAND_MARKER in (m.text or "") for m in real
        )
        for statement in (her_messages(), her_reproducible_messages(), conversation_messages()):
            rows = list(session.scalars(statement))
            assert rows and all(isinstance(row, Message) for row in rows)
    rebuild(services, "all")  # the style profile of her messages, phrases included
    profile = load_profile(services, "live")
    assert profile is not None
    assert MARKER not in json.dumps(profile.phrases() or {}, ensure_ascii=False)
    run_index(services)
    retriever = ExampleRetriever(services, EmbeddingService(embedder))
    asked = [QueryTurn(False, f"{MARKER}用户3"), QueryTurn(True, f"{MARKER}回复3")]
    examples = await retriever.query(ExampleQuery(asked, 720, "workday", None, 8))
    assert examples
    shown = " ".join(
        line.text
        for example in examples
        for line in (*(item for turn in example.context for item in turn.lines), *example.reply)
    )
    assert MARKER not in shown and COMMAND_MARKER not in shown
    assert {"messages", "bot_turns"} <= set(inspect(services.db.engine).get_table_names())


def test_a_command_and_its_reply_are_in_the_table_but_not_in_the_conversation(
    services: Services,
) -> None:
    """R-CMD-001: commands are kept as an audit trail, flagged, and read by nobody else."""
    store = BotTurnStore(services.db, services.clock)
    now = services.clock.now_utc()
    store.add_inbound(at=now, kind="text", text="/状态", external_id="c1", is_command=True)
    store.add_reply([OutboundBubble("⚙️ 一切正常", now)], ReplyMeta("command"), is_command=True)
    assert store.count() == 2 and store.latest_reply() == []
    assert store.recent_stickers(10) == [] and store.last_message_at() is None
