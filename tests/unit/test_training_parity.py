"""The template check that runs on the instance (R-TRN-011): ``twin.training.parity_check``.

Here it runs with the ``slots`` engine - the encoding rules of the template written out - and a
small byte-level tokenizer that splits text like Qwen's.  The same check against the pinned
LLaMA-Factory release and the real Qwen3 tokenizer is ``tests/integration/test_template_parity.py``.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.embedding import HashingBackend
from tests.support.export_world import build_world
from tests.support.tiny_tokenizer import (
    build_merging_tokenizer,
    build_tiny_tokenizer,
    tiny_qwen_tokenizer,
)
from twin.services import Services
from twin.training import bundle, lf_template, parity_check
from twin.training.dataset_dir import DatasetError, ParityCase, load_dataset_dir
from twin.training.export import ExportOptions, TrainingSetExporter
from twin.training.parity_check import (
    EXIT_DIFFERENT,
    EXIT_OK,
    EXIT_USAGE,
    IGNORE_INDEX,
    Case,
    SlotEngine,
    check_case,
    load_cases,
    main,
    open_engine,
    run_checks,
)
from twin.training.profiles import CUTOFF_LEN

ROOT = Path(__file__).resolve().parents[2]

LONG = "很长的一句话" * 120

EDGE_REPLIES = [
    "好的",
    "好的\n晚安\n[表情包:晚安]",
    "[表情包:开心]",
    "[拥抱]",
    "[引用:上次说的那家店]\n去呀",
    "123 号",
    "…哈哈哈",
    "~好~",
    "abc 好",
    "ok",
    "a",
    "（笑）",
    "😂😂",
    "「好」",
    "a\nb\nc\nd\ne\nf",
]
EDGE_SYSTEMS = [
    "她写得很短。",
    "## 风格\n- 口头禅是嘿嘿\n\n【此刻】\n当地时间：2026年8月5日 周三 20:00\n她现在的状态：空闲",
    "【规划】\n想表达：答应\n会用到：她：喜欢火锅\n语气：随意\n气泡：两三条短句\n\n【前文】\n她：早",
    "",
]


def case_of(system: str, turns: list[str], reply: str, case_id: str = "c") -> Case:
    """A case with ``turns`` as the conversation (user first, alternating) before ``reply``."""
    roles = [
        lf_template.Turn("user" if i % 2 == 0 else "assistant", text)
        for i, text in enumerate(turns)
    ]
    conversations = [
        {"from": "human" if t.role == "user" else "gpt", "value": t.content} for t in roles
    ]
    conversations.append({"from": "gpt", "value": reply})
    return Case(case_id, system, conversations, lf_template.render_prompt(system, roles))


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    directory = tmp_path / "model"
    directory.mkdir()
    build_tiny_tokenizer().save(str(directory / "tokenizer.json"))
    return directory


def slots(folder: Path, cutoff: int = CUTOFF_LEN) -> tuple[parity_check.TokenizerLike, SlotEngine]:
    tokenizer, engine = open_engine("slots", folder, cutoff)
    assert isinstance(engine, SlotEngine)
    return tokenizer, engine


# --------------------------------------------------------------------- the edge cases


@pytest.mark.parametrize("system", EDGE_SYSTEMS)
@pytest.mark.parametrize("reply", EDGE_REPLIES)
def test_a_sample_is_token_identical_in_training_and_inference(
    folder: Path, system: str, reply: str
) -> None:
    tokenizer, engine = slots(folder)
    for turns in (["在吗"], ["在吗", "在呢", "吃饭了吗"], [f"第{n}句" for n in range(1, 8)]):
        result = check_case(case_of(system, turns, reply), tokenizer, engine, CUTOFF_LEN)
        assert result.problems == [] and result.passed
        assert 0 < result.tokens <= CUTOFF_LEN


def test_a_context_that_starts_with_digits_punctuation_and_spaces_still_matches(
    folder: Path,
) -> None:
    tokenizer, engine = slots(folder)
    for first in ("123", "...", "  留白", "（括号）", "A", "ａｂｃ", "​零宽"):
        case = case_of("系统", [first, "好", "嗯"], "好的")
        assert check_case(case, tokenizer, engine, CUTOFF_LEN).problems == []


def test_a_sample_near_the_length_limit_is_accepted_and_one_above_it_is_reported(
    folder: Path,
) -> None:
    tokenizer, engine = slots(folder)
    case = case_of("系统", [LONG], "好的")
    tokens = check_case(case, tokenizer, engine, CUTOFF_LEN).tokens
    assert check_case(case, tokenizer, engine, tokens).passed
    over = check_case(case, tokenizer, engine, tokens - 1)
    assert not over.passed and any("longer than cutoff_len" in p for p in over.problems)


# ------------------------------------------------------------------ what it must notice


def test_a_prompt_that_is_not_the_template_rendering_is_reported(folder: Path) -> None:
    tokenizer, engine = slots(folder)
    case = case_of("系统", ["在吗"], "在呢")
    case.prompt = case.prompt.replace("<|im_start|>assistant\n", "<|im_start|>assistant\n<think>\n")
    result = check_case(case, tokenizer, engine, CUTOFF_LEN)
    assert not result.passed
    assert any("not the template rendering" in p for p in result.problems)
    assert any("does not tokenise like the training sequence" in p for p in result.problems)


def test_a_tokenizer_that_merges_across_the_pieces_is_caught(tmp_path: Path) -> None:
    """An earlier answer follows the assistant opener in the prompt, but not in the pieces."""
    directory = tmp_path / "merging"
    directory.mkdir()
    build_merging_tokenizer().save(str(directory / "tokenizer.json"))
    tokenizer, engine = slots(directory)
    merged = case_of("系统", ["ab", "abc ok", "ab"], "好的")
    single = case_of("系统", ["ab"], "abc ok")  # no earlier answer: nothing meets inside the prompt
    results = run_checks([merged, single], tokenizer, engine, CUTOFF_LEN)
    assert not results[0].passed
    assert any("does not tokenise like the training sequence" in p for p in results[0].problems)
    assert results[1].passed


class TrainsOnTheHistory(SlotEngine):
    """An encoder that forgets ``mask_history`` and gives every position a label."""

    def training(self, case: Case) -> tuple[list[int], list[int]]:
        input_ids, _ = super().training(case)
        return input_ids, list(input_ids)


def test_labels_on_anything_but_the_last_response_are_reported(folder: Path) -> None:
    tokenizer, _ = slots(folder)
    wrong = TrainsOnTheHistory(tokenizer)
    result = check_case(
        case_of("系统", ["在吗", "在呢", "好"], "嗯嗯"), tokenizer, wrong, CUTOFF_LEN
    )
    assert any("only the last response" in p for p in result.problems)


def test_the_correct_encoder_masks_everything_but_the_last_response(folder: Path) -> None:
    tokenizer, engine = slots(folder)
    case = case_of("系统", ["在吗", "在呢", "好"], "嗯嗯")
    input_ids, labels = engine.training(case)
    response = engine.pairs(case)[-1][1]
    assert labels[-len(response) :] == response and labels[: -len(response)] == [IGNORE_INDEX] * (
        len(input_ids) - len(response)
    )
    assert tokenizer.decode(response, skip_special_tokens=False) == "嗯嗯<|im_end|>\n"


def test_a_reply_the_template_cannot_carry_is_reported_and_not_a_crash(folder: Path) -> None:
    tokenizer, engine = slots(folder)
    case = case_of("系统", ["在吗"], "在呢")
    case.conversations[-1]["value"] = "在呢<|im_end|>"
    result = check_case(case, tokenizer, engine, CUTOFF_LEN)
    assert not result.passed and "cannot be rendered" in result.problems[0]


# --------------------------------------------------------------------- command line


def write_cases(path: Path, cases: list[Case]) -> None:
    lines = [
        json.dumps(
            {
                "id": c.id,
                "system": c.system,
                "conversations": c.conversations,
                "prompt": c.prompt,
                "tags": c.tags,
            },
            ensure_ascii=False,
        )
        for c in cases
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def test_the_command_passes_good_cases_and_names_the_engine(
    folder: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [case_of("系统", ["在吗"], "在呢", f"c{n}") for n in range(3)])
    code = main(["--model-dir", str(folder), "--cases", str(cases), "--engine", "slots"])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "template check (slots): 3 of 3 case(s) token-identical" in out
    assert "does not call LLaMA-Factory" in out


def test_a_failing_case_gives_exit_code_one_and_never_prints_the_text(
    folder: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = case_of("系统", ["在吗"], "独特的回复文字", "bad")
    broken.prompt += " "
    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [case_of("系统", ["在吗"], "在呢", "good"), broken])
    code = main(["--model-dir", str(folder), "--cases", str(cases), "--engine", "slots"])
    out = capsys.readouterr().out
    assert code == EXIT_DIFFERENT
    assert "FAILED bad" in out and "1 of 2 case(s) token-identical" in out
    assert "独特的回复文字" not in out and "在吗" not in out


def test_the_limit_option_checks_only_the_first_cases(
    folder: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [case_of("系统", ["在吗"], "在呢", f"c{n}") for n in range(5)])
    main(["--model-dir", str(folder), "--cases", str(cases), "--engine", "slots", "--limit", "2"])
    assert "2 of 2 case(s)" in capsys.readouterr().out


def test_missing_input_is_a_usage_error_not_a_verdict(
    folder: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "nowhere.jsonl"
    assert main(["--model-dir", str(folder), "--cases", str(missing)]) == EXIT_USAGE
    assert "cannot run" in capsys.readouterr().err
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert main(["--model-dir", str(folder), "--cases", str(empty), "--engine", "slots"]) == (
        EXIT_USAGE
    )
    junk = tmp_path / "junk.jsonl"
    junk.write_text('{"id": "x"}\n', encoding="utf-8")
    assert main(["--model-dir", str(folder), "--cases", str(junk), "--engine", "slots"]) == (
        EXIT_USAGE
    )
    assert main(["--engine", "unknown"]) == EXIT_USAGE
    assert (
        main(["--model-dir", str(tmp_path / "none"), "--cases", str(junk), "--engine", "slots"])
        == EXIT_USAGE
    )


def test_without_llamafactory_the_default_engine_says_so_and_does_not_skip(
    folder: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("transformers")
    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [case_of("系统", ["在吗"], "在呢")])
    try:
        import llamafactory  # noqa: F401
    except ImportError:
        code = main(["--model-dir", str(folder), "--cases", str(cases)])
        assert code == EXIT_USAGE and "LLaMA-Factory is not installed" in capsys.readouterr().err


def test_load_cases_reads_what_the_exporter_writes(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    write_cases(path, [case_of("系统", ["在吗", "在呢", "好"], "嗯", "x")])
    (case,) = load_cases(path)
    assert case.id == "x" and case.reply == "嗯" and len(case.turns()) == 3


# ---------------------------------------------------------------- standing on its own


def test_the_check_needs_nothing_of_the_project_but_the_template_constants() -> None:
    tree = ast.parse(Path(parity_check.__file__).read_text(encoding="utf-8"))
    twin_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("twin"):
            twin_imports.add(node.module + ":" + ",".join(a.name for a in node.names))
        elif isinstance(node, ast.Import):
            twin_imports.update(a.name for a in node.names if a.name.startswith("twin"))
    assert twin_imports == {"twin.training:lf_template"}


def test_the_package_carries_the_template_check_and_it_runs_from_there(tmp_path: Path) -> None:
    assert bundle.PYLIB_MODULES == ("lf_template", "parity_check")
    files = bundle._pylib_files()
    assert {"pylib/twin/training/lf_template.py", "pylib/twin/training/parity_check.py"} <= set(
        files
    )
    root = tmp_path / "pylib"
    for name, content in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content if isinstance(content, bytes) else content.read_bytes())
    # `-S`: no site-packages, so the installed project cannot be what answers
    environment = {
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "PATH": os.environ.get("PATH", ""),
    }
    bare = subprocess.run(
        [sys.executable, "-S", "-c", "import twin"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert bare.returncode != 0  # without the package folder there is no `twin` at all
    completed = subprocess.run(
        [sys.executable, "-S", "-m", "twin.training.parity_check", "--help"],
        env={**environment, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--model-dir" in completed.stdout


def test_the_length_limit_is_the_one_of_the_profiles() -> None:
    assert parity_check.CUTOFF_LEN == CUTOFF_LEN == 2048


# ------------------------------------------------- the cases an export puts in the dataset


def test_the_cases_the_exporter_chooses_are_token_identical(
    services: Services, embedder: HashingBackend, tmp_path: Path, folder: Path
) -> None:
    build_world(services, embedder, days=8)
    tokenizer = tiny_qwen_tokenizer()
    result = TrainingSetExporter(
        services, tokenizer, ExportOptions(out_dir=tmp_path / "ds", plan_ratio=0.0)
    ).run()
    assert result.dataset is not None and result.dataset.has_parity
    cases = load_cases(result.dataset.file_path("parity_cases.jsonl"))
    assert len(cases) >= 5 and len({c.id for c in cases}) == len(cases)
    engine_tokenizer, engine = slots(folder)
    results = run_checks(cases, engine_tokenizer, engine, CUTOFF_LEN)
    assert [r.problems for r in results if not r.passed] == []
    assert any("prelude" in r.tags for r in results)


# --------------------------------------------- the dataset directory and the package


def sample_case(prompt_suffix: str = "") -> ParityCase:
    turns = (lf_template.Turn("user", "在吗"),)
    prompt = lf_template.render_prompt("系统", turns) + prompt_suffix
    return ParityCase("c1", "系统", turns, "在呢", prompt, ("test",))


def test_a_dataset_with_template_check_cases_round_trips_into_the_package(tmp_path: Path) -> None:
    from tests.support.training_data import pair, sample
    from twin.training.dataset_dir import write_dataset_dir
    from twin.training.profiles import get_profile

    def make(directory: Path, case: ParityCase) -> Path:
        write_dataset_dir(
            directory,
            dataset_version="ds-parity-1",
            created_at="2026-10-09T12:00:00+00:00",
            template_version=lf_template.TEMPLATE_VERSION,
            persona_version="v1",
            profile_version="p1",
            holdout_cutoff="2026-09-01T00:00:00+00:00",
            train=[sample(i, prefix="tr") for i in range(4)],
            val=[sample(0, prefix="va")],
            test=[sample(0, prefix="te")],
            dpo=[pair(0)],
            parity=[case],
            redacted=True,
        )
        return directory

    dataset = load_dataset_dir(make(tmp_path / "good", sample_case()))
    assert dataset.has_parity and dataset.meta.counts.parity == 1
    assert (
        "parity_cases.jsonl" in dataset.data_files() and "parity_cases.jsonl" in dataset.meta.files
    )
    from datetime import UTC, datetime

    built = bundle.build_bundle(
        dataset,
        get_profile("5090-8b"),
        passphrase="a passphrase long enough",
        out_dir=tmp_path / "out",
        created_at=datetime(2026, 10, 9, tzinfo=UTC),
        kdf_log_n=10,
    )
    members = dict(bundle.iter_bundle(built.path, "a passphrase long enough"))
    assert "data/parity_cases.jsonl" in members  # where setup.sh verify looks for it
    assert "pylib/twin/training/parity_check.py" in members
    with pytest.raises(DatasetError, match="differs from the sample"):
        make(tmp_path / "bad", sample_case(" "))
