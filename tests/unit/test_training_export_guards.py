"""The guards around the training export: only her words, nothing from the future (R-TRN-004,
R-TRN-013, CLAUDE.md rules 7 and 13) and a registered model rendering with the versions it was
trained with (R-SRV-001).
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.support.embedding import HashingBackend
from tests.support.export_world import World, build_world, live_card
from tests.support.tiny_tokenizer import tiny_qwen_tokenizer
from tests.support.training_artifacts import build_artifact_dir
from twin.engine.style_prompt import StylePromptBuilder, StyleTurn
from twin.memory.asof import AsOfSource
from twin.retrieval.records import NotHerMessageError
from twin.services import Services
from twin.stickers.catalog import StickerCatalog
from twin.storage.chat_models import Message
from twin.storage.crypto import KeyRing
from twin.storage.engine_models import BotTurn
from twin.storage.ids import new_id
from twin.training import lf_template
from twin.training.dataset_dir import DpoSample
from twin.training.export import ExportOptions, TrainingSetExporter
from twin.training.export_blocks import (
    LeakError,
    MessageLoader,
    block_from_rows,
    iter_block_refs,
)
from twin.training.export_text import context_text
from twin.training.registry import get_model, register_artifacts

SRC = Path(__file__).resolve().parents[2] / "src" / "twin"
EXPORT_MODULES = (
    "training/export.py",
    "training/export_blocks.py",
    "training/export_text.py",
    "training/export_stats.py",
    "training/export_job.py",
    "training/parity_cases.py",
    "training/plans.py",
)
# repositories that read all of the data (or the bot's own conversation) without a moment
FORBIDDEN_MODULES = {
    "twin.memory.memory",
    "twin.memory.store",
    "twin.memory.api",
    "twin.memory.assemble",
    "twin.memory.followups",
    "twin.memory.lifeline",
    "twin.memory.records",
    "twin.engine.turns",
    "twin.engine.dataview",
    "twin.profile.api",
    "twin.profile.store",
    "twin.profile.persona.api",
    "twin.profile.persona.store",
    "twin.retrieval.query",
    "twin.retrieval.examples",
    "twin.retrieval.indexer",
    "twin.storage.memory_models",
    "twin.storage.engine_models",
}
FORBIDDEN_NAMES = {
    "Memory",
    "MemoryAssembler",
    "FactStore",
    "ExampleRetriever",
    "LiveDataSource",
    "BotTurnStore",
    "BotTurn",
    "LifelineStore",
    "PersonaStore",
    "VersionStore",
    "memory_view",
    "load_profile",
    "load_activity_model",
    "render_compact",
    "render_full",
}


def imports_of(path: Path) -> list[tuple[str, str]]:
    """``(module, name)`` of every import of a file (``name`` is empty for ``import x``)."""
    found: list[tuple[str, str]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.extend((node.module, alias.name) for alias in node.names)
        elif isinstance(node, ast.Import):
            found.extend((alias.name, "") for alias in node.names)
    return found


# ---------------------------------------------------------------------- the structure


def test_the_export_modules_import_no_repository_that_reads_all_the_data() -> None:
    offenders = []
    for relative in EXPORT_MODULES:
        for module, name in imports_of(SRC / relative):
            if module in FORBIDDEN_MODULES or name in FORBIDDEN_NAMES:
                offenders.append(f"{relative}: from {module} import {name}")
    assert offenders == []
    exporter = imports_of(SRC / "training/export.py")
    assert ("twin.memory.asof", "AsOfSource") in exporter


def test_the_export_modules_never_query_the_messages_table_themselves() -> None:
    """Her words are read through ``twin.ingest.corpus`` only (R-STO-007)."""
    for relative in EXPORT_MODULES:
        for node in ast.walk(ast.parse((SRC / relative).read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                first = node.args[0] if node.args else None
                assert not (
                    node.func.id in {"select", "update", "delete", "insert"}
                    and isinstance(first, ast.Name)
                    and first.id in {"Message", "BotTurn", "MESSAGES"}
                ), f"{relative} queries a message table"
    blocks = imports_of(SRC / "training/export_blocks.py")
    assert ("twin.ingest.corpus", "conversation_skeleton") in blocks
    assert ("twin.ingest.corpus", "messages_by_ids") in blocks


def test_the_export_does_not_import_test_code() -> None:
    for relative in EXPORT_MODULES:
        assert not [m for m, _ in imports_of(SRC / relative) if m.split(".")[0] == "tests"]


# ------------------------------------------------------ the door for her real messages


def message(
    kind: str = "text", text: str = "好的", *, sent: bool = False, at: datetime | None = None
) -> Message:
    moment = at or datetime(2026, 8, 5, 12, 0, tzinfo=UTC)
    return Message(
        id=new_id(),
        conversation_id="c",
        create_time_utc=moment,
        sort_seq=0,
        is_sent=sent,
        kind=kind,
        text=text,
        raw={},
        source_export_id="export-1",
        created_at=moment,
        updated_at=moment,
    )


def test_objects_of_the_bots_conversation_and_of_preference_pairs_cannot_enter(
    services: Services,
) -> None:
    now = datetime(2026, 8, 5, tzinfo=UTC)
    bot_row = BotTurn(
        id=new_id(),
        at=now,
        direction="out",
        kind="text",
        text="机器人说的",
        is_command=False,
        created_at=now,
        updated_at=now,
    )
    pair = DpoSample(
        "p1", "系统", (lf_template.Turn("user", "在吗"),), "好的", "您好，有什么可以帮您"
    )
    with pytest.raises(TypeError, match="messages table"):
        block_from_rows([bot_row], [])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="messages table"):
        block_from_rows([pair], [])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="messages table"):
        block_from_rows([message()], [[bot_row]])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        block_from_rows([object()], [])  # type: ignore[list-item]
    block = block_from_rows([message(), message(text="嗯")], [[message(text="在吗", sent=True)]])
    assert [m.data.text for m in block.reply] == ["好的", "嗯"] and len(block.context) == 1


def test_a_message_of_the_user_cannot_be_part_of_a_target(keyring_ring: KeyRing) -> None:
    with pytest.raises(NotHerMessageError):
        block_from_rows([message(sent=True)], [])
    with pytest.raises(ValueError, match="at least one"):
        block_from_rows([], [])


def test_a_context_message_from_after_the_reply_is_a_leak(keyring_ring: KeyRing) -> None:
    moment = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)
    later = message(text="之后才说的", sent=True, at=moment + timedelta(seconds=1))
    with pytest.raises(LeakError):
        block_from_rows([message(at=moment)], [[later]])
    same_second = message(text="同一秒", sent=True, at=moment)
    assert block_from_rows([message(at=moment)], [[same_second]]).context


def test_the_block_loader_refuses_ids_that_are_not_messages(
    services: Services, embedder: HashingBackend
) -> None:
    from dataclasses import replace

    world = build_world(services, embedder, days=3)
    ref = next(iter(iter_block_refs(world.services)))
    broken = replace(ref, reply_ids=(*ref.reply_ids, "a-bot-turn-id"))
    with pytest.raises(LookupError):
        MessageLoader(services).load([broken])
    assert MessageLoader(services).load([ref])[0].sample_id == ref.sample_id


# ------------------------------------------- everything about a moment goes through AsOfView


@pytest.fixture
def world(services: Services, embedder: HashingBackend) -> World:
    return build_world(services, embedder, days=8)


def test_every_prompt_is_built_from_the_view_of_the_moment_of_its_block(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = world.services
    asked: list[datetime] = []
    real = AsOfSource.at

    def recording(self: AsOfSource, moment: datetime) -> Any:
        asked.append(moment)
        return real(self, moment)

    monkeypatch.setattr(AsOfSource, "at", recording)
    result = TrainingSetExporter(
        services, tiny_qwen_tokenizer(), ExportOptions(out_dir=tmp_path / "ds", plan_ratio=0.0)
    ).run()
    assert result.dataset is not None
    times = {ref.sample_id: ref.reply_at for ref in iter_block_refs(services)}
    exported = {
        json.loads(line)["id"]
        for name in ("sft_train.jsonl", "sft_val.jsonl", "sft_test.jsonl")
        for line in result.dataset.file_path(name).read_text(encoding="utf-8").splitlines()
    }
    assert exported and {times[i] for i in exported} <= set(asked)
    assert len(asked) == len(exported)  # one view per sample, none for anything else
    assert all(moment.tzinfo is not None for moment in asked)


def test_the_view_of_a_moment_is_all_the_prompt_builder_reads(world: World, tmp_path: Path) -> None:
    """The builder gets the view and the turns; a fact of the future stays out whatever it says."""
    services = world.services
    source = AsOfSource(services)
    moment = datetime(2026, 8, 20, tzinfo=UTC)
    builder = StylePromptBuilder.from_services(services)
    prompt = builder.build(source.at(moment), [StyleTurn("user", "对方住在哪里呢")])
    assert "之后才知道的事" not in prompt.text and "2026年8月" in prompt.text
    assert prompt.meta is not None and prompt.meta.persona_scope == "pre_holdout"


# --------------------------------------------- a registered model renders what it learnt


def test_the_registry_locks_the_versions_the_export_used_and_renders_with_them(
    world: World, tmp_path: Path
) -> None:
    services = world.services
    result = TrainingSetExporter(
        services, tiny_qwen_tokenizer(), ExportOptions(out_dir=tmp_path / "ds", plan_ratio=0.0)
    ).run()
    dataset = result.dataset
    assert dataset is not None
    meta = dataset.meta
    artifacts = tmp_path / "artifacts"
    build_artifact_dir(
        artifacts,
        run_id="r-lock",
        dataset_version=meta.dataset_version,
        versions={
            "template_version": meta.template_version,
            "persona_version": meta.persona_version,
            "profile_version": meta.profile_version,
        },
    )
    registered = register_artifacts(services.db, artifacts)
    model = get_model(services.db, registered.models[0].id)
    locked = model.versions
    assert (
        locked.template_version,
        locked.persona_version,
        locked.profile_version,
        locked.dataset_version,
    ) == (
        lf_template.TEMPLATE_VERSION,
        meta.persona_version,
        meta.profile_version,
        meta.dataset_version,
    )

    # the card is edited afterwards: a newer pre-holdout version becomes the active one
    world.store.add_version("pre_holdout", live_card().replace("嘿嘿", "新版口头禅"), reason="edit")
    row = json.loads(dataset.file_path("sft_val.jsonl").read_text(encoding="utf-8").splitlines()[0])
    ref = next(r for r in iter_block_refs(services) if r.sample_id == row["id"])
    (block,) = MessageLoader(services).load([ref])
    label = StickerCatalog(services).tag_lookup()
    turns = [
        StyleTurn(
            "user" if t[0].data.is_sent else "assistant", context_text([m.data for m in t], label)
        )
        for t in block.context
    ]
    view = AsOfSource(services).at(block.at)
    served = StylePromptBuilder.from_services(services, locked=locked).compose(view, turns)
    drifted = StylePromptBuilder.from_services(services).compose(view, turns)
    persona = row["system"].split("【此刻】")[0]
    assert served.system.split("【此刻】")[0] == persona  # exactly the card it was trained with
    assert drifted.system.split("【此刻】")[0] != persona  # the active card is another one now
    assert served.meta.locked and served.meta.persona_version == meta.persona_version
