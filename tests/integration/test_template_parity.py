"""Training = inference, token for token, with the pinned LLaMA-Factory (R-TRN-011).

The training set is exported from a synthetic conversation with the real Qwen3 tokenizer, and
``python -m twin.training.parity_check`` - the module of the training package - runs in an
environment that has LLaMA-Factory 0.9.5 installed, the way ``setup.sh verify`` runs it on the
instance:

1. the prompt that ``StylePromptBuilder`` renders, tokenised as one string, equals the tokens
   LLaMA-Factory's ``encode_multiturn`` makes of the ShareGPT sample, minus its last response;
2. the last response decodes to the reply and ``<|im_end|>``;
3. with ``mask_history`` only the last response has labels (the processor's output is checked).

The environment is not part of the project (LLaMA-Factory pulls in a lot): build one with

    uv venv lfvenv && uv pip install --python lfvenv/bin/python torch==2.9.1 \\
        --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python lfvenv/bin/python llamafactory==0.9.5 transformers==4.57.6 \\
        peft==0.18.1 trl==0.24.0 accelerate==1.11.0 datasets==4.0.0

and download the tokenizer files of ``Qwen/Qwen3-8B`` (``tokenizer.json``,
``tokenizer_config.json``, ``vocab.json``, ``merges.txt``; no weights).  Then

    TWIN_LLAMAFACTORY_PYTHON=lfvenv/bin/python TWIN_QWEN_TOKENIZER=<that folder> \\
        uv run pytest tests/integration/test_template_parity.py

Without both variables the tests are skipped.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.support.embedding import HashingBackend
from tests.support.export_world import World, build_world
from tests.support.synth_chat import MessageWriter
from twin.services import Services
from twin.training import bundle, lf_template
from twin.training.dataset_dir import DatasetDir
from twin.training.export import ExportOptions, ExportResult, TrainingSetExporter
from twin.training.plans import PlanStore, StoredPlan, fact_lines
from twin.training.tokenizer import load_tokenizer

pytestmark = pytest.mark.integration

# read when the module is imported: the tests' own fixture clears every TWIN_* variable
PYTHON = os.environ.get("TWIN_LLAMAFACTORY_PYTHON")
TOKENIZER_DIR = os.environ.get("TWIN_QWEN_TOKENIZER")
needs_environment = pytest.mark.skipif(
    not PYTHON or not TOKENIZER_DIR,
    reason="set TWIN_LLAMAFACTORY_PYTHON and TWIN_QWEN_TOKENIZER (see the module docstring)",
)

# beginnings and endings that decide where tokens could merge across the pieces of a prompt
EDGE_REPLIES = [
    "好的",
    "123 号楼",
    "…哈哈哈",
    "~好~",
    "abc 好",
    "ok",
    "（笑）",
    "😂😂",
    "「好」",
    "[拥抱]",
    "[表情包:开心]",
    "收到！！！",
    "a",
    "好 <|im_end|> 的{{content}}",
    "第一行\n第二行\n第三行",
]
EDGE_USER = ["在吗", "1", "...", "A", "好~", "😂", "  嗯", "（括号）", "ａｂｃ", "​零宽"]


def insert_edge_conversations(world: World) -> None:
    """Short exchanges before the conversation starts: each user line and each reply alone."""
    writer = MessageWriter(world.services)
    moment = datetime(2026, 7, 1, 15, 0, tzinfo=UTC)
    for number, reply in enumerate(EDGE_REPLIES):
        user = EDGE_USER[number % len(EDGE_USER)]
        writer.add(moment, False, "text", f"{user}{number}")
        for part, line in enumerate(reply.split("\n")):
            writer.add(moment + timedelta(seconds=20 + 3 * part), True, "text", line)
        moment += timedelta(hours=3)
    # a deep conversation: fifteen turns of two people in a row, then her answer
    for turn in range(15):
        writer.add(moment, turn % 2 == 1, "text", f"第{turn}轮的一句话，说得不短不长")
        moment += timedelta(minutes=2)
    writer.add(moment, True, "text", "收到深的对话了")
    moment += timedelta(hours=3)
    # a long one that has to lose its oldest turns to fit 2,048 tokens
    for turn in range(9):
        her = turn % 2 == 1
        writer.add(moment, her, "text", "嗯" if her else f"长句{turn}" + "很长的一句话" * 180)
        moment += timedelta(minutes=2)
    writer.add(moment, True, "text", "长对话的回答")
    writer.store(append=True)


def register_plans(world: World, tokenizer_dir: str, directory: Path) -> ExportResult:
    """Give every sample that is to carry a plan one (what the batch jobs would have written)."""
    services = world.services
    exporter = TrainingSetExporter(
        services,
        load_tokenizer(Path(tokenizer_dir)),
        ExportOptions(out_dir=directory, plan_ratio=0.4),
    )
    missing, _ = exporter.scan()
    store = PlanStore(services.db)
    store.register(missing)
    for sample_id, source in missing.items():
        facts = tuple(source.facts[:1])
        plan = StoredPlan("回应对方并说说自己的情况", facts, "随意", "两三条短句")
        store.save_done(sample_id, plan, model="m", template_ref="train_plan@1", cost_usd=0.0)
    return exporter.run()


def cases_of(dataset: DatasetDir, limit: int = 400) -> list[dict[str, Any]]:
    """The cases the export chose, then every sample of every split (up to ``limit``)."""
    chosen = [
        json.loads(line)
        for line in dataset.file_path("parity_cases.jsonl").read_text(encoding="utf-8").split("\n")
        if line
    ]
    everything: list[dict[str, Any]] = []
    for name in ("sft_train.jsonl", "sft_val.jsonl", "sft_test.jsonl"):
        for line in dataset.file_path(name).read_text(encoding="utf-8").split("\n"):
            if not line:
                continue
            row = json.loads(line)
            turns = [
                lf_template.Turn("user" if i % 2 == 0 else "assistant", m["value"])
                for i, m in enumerate(row["conversations"][:-1])
            ]
            row["prompt"] = lf_template.render_prompt(row["system"], turns)
            row["tags"] = [name.removesuffix(".jsonl")]
            everything.append(row)
    step = max(1, len(everything) // limit)
    return chosen + everything[::step]


def write_cases(path: Path, cases: list[dict[str, Any]]) -> None:
    lines = [json.dumps(case, ensure_ascii=False) for case in cases]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def package_modules(directory: Path) -> Path:
    """``pylib`` as the training package carries it: the modules the instance imports."""
    for name, content in bundle._pylib_files().items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content if isinstance(content, bytes) else content.read_bytes())
    return directory / "pylib"


def run_check(pylib: Path, cases: Path) -> subprocess.CompletedProcess[str]:
    assert PYTHON is not None and TOKENIZER_DIR is not None
    env = {
        "PYTHONPATH": str(pylib),
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", str(cases.parent)),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
    }
    return subprocess.run(
        [
            PYTHON,
            "-m",
            "twin.training.parity_check",
            "--model-dir",
            TOKENIZER_DIR,
            "--cases",
            str(cases),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )


@pytest.fixture
def exported(services: Services, embedder: HashingBackend, tmp_path: Path) -> DatasetDir:
    assert TOKENIZER_DIR is not None
    world = build_world(services, embedder, days=14)
    insert_edge_conversations(world)
    result = register_plans(world, TOKENIZER_DIR, tmp_path / "dataset")
    assert result.state == "done" and result.dataset is not None
    return result.dataset


@needs_environment
def test_the_exported_samples_are_token_identical_under_llamafactory(
    exported: DatasetDir, tmp_path: Path
) -> None:
    tags = {
        tag
        for line in exported.file_path("parity_cases.jsonl").read_text("utf-8").split("\n")
        if line
        for tag in json.loads(line)["tags"]
    }
    assert {"prelude", "multiline", "sticker", "plan", "deep_context", "trimmed"} <= tags
    cases = tmp_path / "cases.jsonl"
    write_cases(cases, cases_of(exported))
    completed = run_check(package_modules(tmp_path), cases)
    print(completed.stdout)  # shown with -s: the number of cases and the longest sequence
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
    assert "template check (llamafactory):" in completed.stdout
    assert "FAILED" not in completed.stdout
    total = len(cases.read_text(encoding="utf-8").strip().split("\n"))
    assert f"{total} of {total} case(s) token-identical" in completed.stdout


@needs_environment
def test_the_edge_conversations_are_in_the_checked_set(exported: DatasetDir) -> None:
    replies = []
    for name in ("sft_train.jsonl", "sft_val.jsonl", "sft_test.jsonl"):
        for line in exported.file_path(name).read_text(encoding="utf-8").split("\n"):
            if line:
                replies.append(json.loads(line)["conversations"][-1]["value"])
    for reply in ("…哈哈哈", "😂😂", "收到！！！", "ok", "第一行\n第二行\n第三行"):
        assert reply in replies, "an edge reply did not become a sample"
    assert "好  的" in replies  # the control text of the template was taken out of the reply
    assert not any("<|im_end|>" in r or "{{content}}" in r for r in replies)
    assert any(r == "长对话的回答" for r in replies) and any(r == "收到深的对话了" for r in replies)


@needs_environment
def test_the_check_notices_the_think_block_of_the_official_template(
    exported: DatasetDir, tmp_path: Path
) -> None:
    """Qwen3's own chat template adds an empty think block before the answer; this one must not."""
    cases = cases_of(exported, limit=20)
    official = dict(cases[0])
    official["prompt"] = official["prompt"] + "<think>\n\n</think>\n\n"
    path = tmp_path / "cases.jsonl"
    write_cases(path, [official])
    completed = run_check(package_modules(tmp_path), path)
    assert completed.returncode == 1, completed.stdout
    assert "FAILED" in completed.stdout and "not the template rendering" in completed.stdout
    assert "0 of 1 case(s) token-identical" in completed.stdout


def test_the_memory_fact_lines_of_a_prompt_are_not_special_to_the_template() -> None:
    """A guard for the helpers this module imports (it also runs where the check is skipped)."""
    assert fact_lines("【相关的事】\n- 她：喜欢火锅") == ["她：喜欢火锅"]
    assert lf_template.render_response("好").endswith("<|im_end|>\n")
