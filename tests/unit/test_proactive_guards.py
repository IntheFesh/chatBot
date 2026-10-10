"""What she says first never becomes what she sounds like (CLAUDE.md rule 7, R-PRO-006, R-LRN-004).

A proactive message is the bot's own words: it is written to ``bot_turns`` (the conversation) and to
``proactive_log`` (the audit, sealed).  Neither the style samples, nor the retrieval library, nor
the SFT set may ever read them - otherwise the bot would learn to imitate itself.  The same holds
for the ratings of ``/评分``.  These tests guard it both ways: by what the code imports and by what
the files hold after a real export.
"""

from __future__ import annotations

import ast
import random
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from tests.support.embedding import HashingBackend
from tests.support.export_world import World, build_world
from tests.support.tiny_tokenizer import tiny_qwen_tokenizer
from twin.retrieval.openers import OpenerExamples
from twin.schedule.proactive.store import NewLog, ProactiveLogStore, RatingStore
from twin.services import Services
from twin.storage.engine_models import BotTurn
from twin.storage.ids import new_id
from twin.training.export import ExportOptions, TrainingSetExporter

SRC = Path(__file__).resolve().parents[2] / "src" / "twin"
MARKER = "只有她自己先开口时才说的话"
NOTE = "评分里写的备注文字"
PLAN_REASON = "规划理由里的文字"
TOKENIZER = tiny_qwen_tokenizer()

# the only modules that may know the tables of the proactive messages or their stores
READERS = {
    "storage/proactive_models.py",
    "storage/__init__.py",
    "storage/migrations/env.py",
    "schedule/proactive/cli.py",
    "schedule/proactive/component.py",
    "schedule/proactive/decide.py",
    "schedule/proactive/rules.py",
    "schedule/proactive/scheduler.py",
    "schedule/proactive/send.py",
    "schedule/proactive/slots.py",
    "schedule/proactive/status.py",
    "schedule/proactive/store.py",
    "eval/cli.py",
    "eval/proactive_audit.py",
    "eval/proactive_gate.py",
    "eval/proactive_report.py",
    "commands/rating.py",
    "cli.py",
}
PROACTIVE_MODULES = ("twin.schedule.proactive", "twin.storage.proactive_models")
PROACTIVE_NAMES = {"ProactiveLog", "ProactiveCandidate", "Rating"}
LIBRARIES = (
    "retrieval",
    "ingest",
    "profile",
    "stickers",
    "memory",
    "training",
    "learning",
    "llm",
    "channel",
)


def imports_of(path: Path) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.extend((node.module, alias.name) for alias in node.names)
        elif isinstance(node, ast.Import):
            found.extend((alias.name, "") for alias in node.names)
    return found


def modules_naming_the_proactive_books() -> set[str]:
    found: set[str] = set()
    for path in SRC.rglob("*.py"):
        relative = path.relative_to(SRC).as_posix()
        for module, name in imports_of(path):
            if module.startswith(PROACTIVE_MODULES) or name in PROACTIVE_NAMES:
                found.add(relative)
    return found


# ---------------------------------------------------------------------- by the imports


def test_only_the_scheduler_its_audit_and_the_rating_command_know_the_proactive_books() -> None:
    found = modules_naming_the_proactive_books()
    assert {"schedule/proactive/store.py", "eval/proactive_audit.py", "commands/rating.py"} <= found
    assert found <= READERS


@pytest.mark.parametrize("package", LIBRARIES)
def test_no_library_that_builds_samples_context_or_training_data_reads_them(package: str) -> None:
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in (SRC / package).rglob("*.py")
        for module, name in imports_of(path)
        if module.startswith(PROACTIVE_MODULES) or name in PROACTIVE_NAMES
    ]
    assert offenders == []


def test_the_engine_does_not_know_the_proactive_log_either() -> None:
    """The engine only exposes its parts; what is written to the log is the scheduler's own."""
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in (SRC / "engine").rglob("*.py")
        for module, name in imports_of(path)
        if module.startswith(PROACTIVE_MODULES) or name in PROACTIVE_NAMES
    ]
    assert offenders == []


# ---------------------------------------------------------------- by what the files hold


@pytest.fixture
def world(services: Services, embedder: HashingBackend) -> World:
    return build_world(services, embedder, days=12)


def write_her_first_words(services: Services) -> None:
    """A proactive message the way the scheduler leaves it: bot_turns, the log, a rating."""
    now = services.clock.now_utc()
    reply_id = new_id()
    with services.db.transaction(bump_state=False) as session:
        session.add(
            BotTurn(
                id=new_id(),
                at=datetime(2026, 8, 6, 15, 0, tzinfo=UTC),
                direction="out",
                kind="text",
                text=MARKER,
                is_command=False,
                reply_id=reply_id,
                actions=[{"step": "proactive", "count": 1, "detail": "greeting"}],
                created_at=now,
                updated_at=now,
            )
        )
    ProactiveLogStore(services.db, services.clock).add(
        NewLog(
            at=datetime(2026, 8, 6, 15, 0, tzinfo=UTC),
            candidate_at=datetime(2026, 8, 6, 15, 0, tzinfo=UTC),
            local_date=date(2026, 8, 6),
            local_at="2026-08-06 10:00",
            timezone="America/Chicago",
            kind="greeting",
            outcome="sent",
            reply_id=reply_id,
            plan_reason=PLAN_REASON,
            content={"bubbles": [MARKER]},
        )
    )
    RatingStore(services.db, services.clock).add(
        4, NOTE, at=datetime(2026, 8, 7, tzinfo=UTC), local_date=date(2026, 8, 7)
    )


def test_a_proactive_message_its_reason_and_a_rating_are_in_no_training_file(
    world: World, tmp_path: Path
) -> None:
    write_her_first_words(world.services)
    result = TrainingSetExporter(
        world.services, TOKENIZER, ExportOptions(out_dir=tmp_path / "sft", plan_ratio=0.0)
    ).run()
    assert result.dataset is not None
    for name in result.dataset.data_files():
        text = result.dataset.file_path(name).read_text(encoding="utf-8")
        assert MARKER not in text and NOTE not in text and PLAN_REASON not in text, name
    for path in result.dataset.path.iterdir():
        if path.is_file():
            body = path.read_text(encoding="utf-8", errors="ignore")
            assert MARKER not in body, path.name


async def test_her_real_openings_never_include_what_the_bot_said_first(
    world: World,
) -> None:
    write_her_first_words(world.services)
    openers = OpenerExamples(world.services, rng=random.Random(3))
    try:
        found = await openers.query(
            local_minute=600.0,
            day_type="workday",
            now=datetime(2026, 10, 9, 16, 0, tzinfo=UTC),
            k=5,
        )
    finally:
        await openers.aclose()
    said = ["".join(line.text for line in example.reply) for example in found]
    assert all(MARKER not in text for text in said)
