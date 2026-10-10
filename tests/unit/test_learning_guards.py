"""The bot's words never become training data - except as the negative of a DPO pair (R-LRN-004).

CLAUDE.md rule 7 and R-LRN-004: the bot's own replies (``bot_turns``) never enter the style
samples, the retrieval library or the SFT set; ``preference_pairs.rejected`` is the bot's reply and
appears in the DPO export and nowhere else.  These tests guard both ways: by what the code imports
and by what the files hold after a real export.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.support.embedding import HashingBackend
from tests.support.export_world import World, build_world
from tests.support.tiny_tokenizer import tiny_qwen_tokenizer
from twin.learning.pairs import PairRecord, PreferencePairStore, PromptSample
from twin.services import Services
from twin.storage.engine_models import BotTurn
from twin.storage.ids import new_id
from twin.storage.learning_models import PreferencePair
from twin.training import lf_template
from twin.training.dpo_export import export_dpo
from twin.training.export import ExportOptions, TrainingSetExporter
from twin.training.export_blocks import MessageLoader, block_from_rows, iter_block_refs
from twin.training.layout import FILE_DPO

SRC = Path(__file__).resolve().parents[2] / "src" / "twin"
MARKER = "只在偏好对里被拒绝的那句话"
WORDING = "她真正会说的另一句话"
TOKENIZER = tiny_qwen_tokenizer()

# the only modules that may know the table of the pairs or their records
READERS_OF_PAIRS = {
    "storage/learning_models.py",
    "storage/__init__.py",
    "learning/pairs.py",
    "learning/dislike.py",
    "learning/sample.py",
    "learning/corrections.py",
    "commands/learn_commands.py",
    "commands/status_extras.py",
    "commands/router.py",
    "engine/component.py",
    "training/dpo_export.py",
    "training/cli.py",
    "storage/migrations/env.py",
}
PAIR_NAMES = {"PreferencePair", "PreferencePairStore", "PairRecord"}
PAIR_MODULES = ("twin.storage.learning_models",)  # the helpers of twin.learning.pairs are free


def imports_of(path: Path) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.extend((node.module, alias.name) for alias in node.names)
        elif isinstance(node, ast.Import):
            found.extend((alias.name, "") for alias in node.names)
    return found


def modules_naming_the_pairs() -> set[str]:
    found: set[str] = set()
    for path in SRC.rglob("*.py"):
        relative = path.relative_to(SRC).as_posix()
        for module, name in imports_of(path):
            if module in PAIR_MODULES or name in PAIR_NAMES:
                found.add(relative)
    return found


# ---------------------------------------------------------------------- by the imports


def test_only_the_learning_the_commands_and_the_dpo_export_know_the_pairs() -> None:
    assert modules_naming_the_pairs() <= READERS_OF_PAIRS


@pytest.mark.parametrize(
    "package", ["retrieval", "ingest", "profile", "stickers", "memory", "eval", "schedule", "llm"]
)
def test_no_library_that_builds_samples_or_context_reads_the_pairs(package: str) -> None:
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in (SRC / package).rglob("*.py")
        for module, name in imports_of(path)
        if (module.startswith("twin.learning") or module in PAIR_MODULES or name in PAIR_NAMES)
        and path.relative_to(SRC).as_posix() != "profile/persona/cli.py"  # `twin persona rules`
    ]
    assert offenders == []


def test_the_sft_export_modules_never_import_the_learning() -> None:
    exporting = (
        "training/export.py",
        "training/export_blocks.py",
        "training/export_text.py",
        "training/export_stats.py",
        "training/export_job.py",
        "training/parity_cases.py",
        "training/plans.py",
        "training/bundle.py",
        "training/dataset_dir.py",
    )
    for relative in exporting:
        for module, name in imports_of(SRC / relative):
            assert not module.startswith("twin.learning"), (relative, module)
            assert module not in PAIR_MODULES and name not in PAIR_NAMES, (relative, name)


def test_the_dpo_export_is_the_one_training_module_that_reads_the_pairs() -> None:
    readers = {r for r in modules_naming_the_pairs() if r.startswith("training/")}
    assert readers == {"training/dpo_export.py", "training/cli.py"}
    assert ("twin.learning.pairs", "PreferencePairStore") in imports_of(
        SRC / "training/dpo_export.py"
    )


# ---------------------------------------------------------------- by what the code accepts


def test_the_sft_door_accepts_neither_the_pairs_nor_their_records(services: Services) -> None:
    now = datetime(2026, 8, 5, tzinfo=UTC)
    row = PreferencePair(
        id=new_id(),
        reply_id=new_id(),
        source="user_correction",
        template_version="t",
        persona_version="p",
        created_at=now,
        updated_at=now,
    )
    record = PairRecord(
        "p1",
        None,
        "r1",
        PromptSample("系统", (lf_template.Turn("user", "在吗"),)),
        "好的",
        MARKER,
        "user_correction",
        "t",
        "p",
        now,
    )
    for rejected in (row, record, record.sample):
        with pytest.raises(TypeError, match="messages table"):
            block_from_rows([rejected], [])  # type: ignore[list-item]
        with pytest.raises(TypeError, match="messages table"):
            block_from_rows([], [[rejected]])  # type: ignore[list-item]


def test_the_block_loader_does_not_find_a_pair_or_its_reply_among_her_messages(
    services: Services, embedder: HashingBackend
) -> None:
    world = build_world(services, embedder, days=3)
    store = PreferencePairStore(services.db, services.clock)
    pair, _ = store.add(
        reply_id=new_id(),
        sample=PromptSample("系统", (lf_template.Turn("user", "在吗"),)),
        chosen=WORDING,
        rejected=MARKER,
        template_version="t",
        persona_version="p",
    )
    ref = next(iter(iter_block_refs(world.services)))
    for foreign in (pair.id, pair.reply_id):
        with pytest.raises(LookupError):
            MessageLoader(services).load([replace(ref, reply_ids=(*ref.reply_ids, foreign))])


# ---------------------------------------------------------------- by what the files hold


@pytest.fixture
def world(services: Services, embedder: HashingBackend) -> World:
    return build_world(services, embedder, days=12)


def test_a_rejected_reply_is_in_the_dpo_file_and_never_in_the_sft_set(
    world: World, tmp_path: Path
) -> None:
    services = world.services
    store = PreferencePairStore(services.db, services.clock)
    reply_id = new_id()
    with services.db.transaction(bump_state=False) as session:  # the reply, in the bot's table
        now = services.clock.now_utc()
        for number, direction in enumerate(("in", "out")):
            session.add(
                BotTurn(
                    id=new_id(),
                    at=datetime(2026, 8, 6, 15, number, tzinfo=UTC),
                    direction=direction,
                    kind="text",
                    text=MARKER if direction == "out" else "在吗",
                    is_command=False,
                    reply_id=reply_id if direction == "out" else None,
                    created_at=now,
                    updated_at=now,
                )
            )
    store.add(
        reply_id=reply_id,
        sample=PromptSample("系统", (lf_template.Turn("user", "在吗"),), NOW_FOR_PAIR),
        chosen=WORDING,
        rejected=MARKER,
        template_version=lf_template.TEMPLATE_VERSION,
        persona_version="live:v1",
    )
    sft = TrainingSetExporter(
        services, TOKENIZER, ExportOptions(out_dir=tmp_path / "sft", plan_ratio=0.0)
    ).run()
    assert sft.dataset is not None
    for name in sft.dataset.data_files():  # neither the reply nor the wording is a sample
        text = sft.dataset.file_path(name).read_text(encoding="utf-8")
        assert MARKER not in text and WORDING not in text, name
    dpo = export_dpo(
        services, dataset=sft.dataset.path, system_maker=_keep_system, now=NOW_FOR_PAIR
    )
    file = dpo.dataset.path / FILE_DPO
    assert MARKER in file.read_text(encoding="utf-8")  # the negative of the pair, here only
    for path in dpo.dataset.path.iterdir():
        if path.name != FILE_DPO:
            assert MARKER not in path.read_text(encoding="utf-8"), path.name


NOW_FOR_PAIR = datetime(2026, 10, 9, 18, 0, tzinfo=UTC)


def _keep_system(pair: PairRecord) -> PromptSample:
    return pair.sample
