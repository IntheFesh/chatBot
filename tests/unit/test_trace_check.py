"""scripts/trace_check.py on small synthetic SPEC / TRACEABILITY documents."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.scripts import SCRIPTS, load_script

trace = load_script("trace_check")

SPEC = """\
# Spec
- **R-AAA-001** first requirement
- **R-AAA-002** second requirement
  continuation line mentioning R-AAA-001 inline
- **R-BBB-001** third, shared by two rounds
text mentioning `R-ZZZ-999` that is not a definition
"""

HEADER = """\
## 表 1：x
| a | b |
| --- | --- |
| P-01 | R-AAA-001 |

## 表 2：SPEC 需求实现表

| 编号 | 摘要 | 轮次 | 状态 | 实现位置 | 测试 |
| --- | --- | --- | --- | --- | --- |
"""


def table(*rows: str) -> str:
    return HEADER + "\n".join(rows) + "\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "mod.py").write_text(
        "CONSTANT = 1\n\nclass Thing:\n    ATTR: int = 1\n\n    def method(self) -> None: ...\n\n"
        "async def run() -> None: ...\n\ndef helper() -> None: ...\n",
        encoding="utf-8",
    )
    (tmp_path / "config.yaml").write_text("a: 1\n", encoding="utf-8")
    return tmp_path


COLLECTED = {"tests/unit/test_a.py::test_one", "tests/unit/test_a.py::test_param[x-1]"}
ROW_OK = (
    "| R-AAA-001 | s | 00 | 已实现 | src/pkg/mod.py::Thing.method; config.yaml "
    "| tests/unit/test_a.py::test_one |"
)
ROW_PENDING = "| R-AAA-002 | s | 01 | 待实现 |  |  |"
ROW_SHARED = (
    "| R-BBB-001 | s | 00,01 | 待实现 | src/pkg/mod.py::helper（部分：第 00 轮） "
    "| tests/unit/test_a.py::test_one |"
)


def run(repo: Path, rows: list[str], round_id: str | None = None, spec: str = SPEC):
    return trace.check(spec, table(*rows), COLLECTED, repo, round_id)


def test_parse_spec_ids_reads_definitions_only() -> None:
    assert trace.parse_spec_ids(SPEC) == ["R-AAA-001", "R-AAA-002", "R-BBB-001"]


def test_parse_table_extracts_the_six_columns() -> None:
    rows = trace.parse_table(table(ROW_OK, ROW_SHARED))
    assert (
        rows[0].req_id == "R-AAA-001" and rows[0].rounds == ("00",) and rows[0].status == "已实现"
    )
    assert rows[1].rounds == ("00", "01")
    with pytest.raises(ValueError, match="no '## 表 2'"):
        trace.parse_table("nothing")
    with pytest.raises(ValueError, match="6"):
        trace.parse_table(table("| R-AAA-001 | only | three |"))
    with pytest.raises(ValueError, match="invalid round"):
        trace.parse_table(table("| R-AAA-001 | s | zero | 待实现 |  |  |"))


def test_a_consistent_table_passes(repo: Path) -> None:
    problems, stats = run(repo, [ROW_OK, ROW_PENDING, ROW_SHARED], "00")
    assert problems == []
    assert (
        stats["spec"] == 3
        and stats["owned"] == 2
        and stats["implemented"] == 1
        and stats["partial"] == 1
    )


def test_missing_rows_unknown_rows_and_duplicates_are_reported(repo: Path) -> None:
    problems, _ = run(repo, [ROW_OK, "| R-ZZZ-999 | s | 00 | 待实现 |  |  |", ROW_OK])
    text = "\n".join(problems)
    assert "R-AAA-002: defined in SPEC but has no row" in text
    assert "R-BBB-001: defined in SPEC but has no row" in text
    assert "R-ZZZ-999: row exists but the SPEC does not define it" in text
    assert "R-AAA-001: duplicate row" in text


def test_implemented_rows_need_locations_and_tests(repo: Path) -> None:
    problems, _ = run(repo, ["| R-AAA-001 | s | 00 | 已实现 |  |  |", ROW_PENDING, ROW_SHARED])
    assert any("without 实现位置" in p for p in problems)
    assert any("without tests" in p for p in problems)


def test_locations_are_checked_with_ast(repo: Path) -> None:
    bad = [
        "| R-AAA-001 | s | 00 | 已实现 | src/pkg/mod.py::Missing "
        "| tests/unit/test_a.py::test_one |",
        "| R-AAA-002 | s | 00 | 已实现 | src/pkg/nope.py::Thing | tests/unit/test_a.py::test_one |",
    ]
    problems, _ = run(repo, [*bad, ROW_SHARED], "00")
    assert any("symbol not found: src/pkg/mod.py::Missing" in p for p in problems)
    assert any("file not found: src/pkg/nope.py" in p for p in problems)
    for good in ("Thing", "Thing.method", "Thing.ATTR", "run", "helper", "CONSTANT"):
        assert trace.check_location(repo, f"src/pkg/mod.py::{good}") is None
    assert trace.check_location(repo, "config.yaml") is None
    assert trace.check_location(repo, "config.yaml::anything") is None  # non-Python: file only


def test_tests_must_be_collected_by_pytest(repo: Path) -> None:
    row = (
        "| R-AAA-001 | s | 00 | 已实现 | src/pkg/mod.py::helper "
        "| tests/unit/test_a.py::test_ghost |"
    )
    problems, _ = run(repo, [row, ROW_PENDING, ROW_SHARED])
    assert any(
        "test not collected by pytest: tests/unit/test_a.py::test_ghost" in p for p in problems
    )
    parametrized = (
        "| R-AAA-001 | s | 00 | 已实现 | src/pkg/mod.py::helper "
        "| tests/unit/test_a.py::test_param |"
    )
    assert run(repo, [parametrized, ROW_PENDING, ROW_SHARED])[0] == []  # test_param[x-1] counts


def test_last_owner_round_must_mark_the_requirement_implemented(repo: Path) -> None:
    row = "| R-AAA-001 | s | 00 | 待实现 |  |  |"
    problems, _ = run(repo, [row, ROW_PENDING, ROW_SHARED], "00")
    assert any("R-AAA-001: round 00 is its last owner" in p for p in problems)


def test_shared_requirement_needs_a_partial_note_from_non_final_rounds(repo: Path) -> None:
    no_note = "| R-BBB-001 | s | 00,01 | 待实现 | src/pkg/mod.py::helper | |"
    problems, _ = run(repo, [ROW_OK, ROW_PENDING, no_note], "00")
    assert any("R-BBB-001" in p and "（部分：第 00 轮）" in p for p in problems)
    empty = "| R-BBB-001 | s | 00,01 | 待实现 |  |  |"
    assert run(repo, [ROW_OK, ROW_PENDING, empty], "00")[0] != []
    # the last owner (round 01) cannot hide behind a partial note
    problems, _ = run(repo, [ROW_OK, ROW_PENDING, ROW_SHARED], "01")
    assert any("R-BBB-001: round 01 is its last owner" in p for p in problems)


def test_rounds_with_suffixes_sort_correctly() -> None:
    assert trace.round_key("09b") > trace.round_key("09") > trace.round_key("08")
    assert trace.round_key("10") > trace.round_key("09b")
    with pytest.raises(ValueError):
        trace.round_key("9")


def test_invalid_status_is_reported(repo: Path) -> None:
    problems, _ = run(repo, ["| R-AAA-001 | s | 00 | done |  |  |", ROW_PENDING, ROW_SHARED])
    assert any("status must be" in p for p in problems)


def test_partial_notes_and_location_splitting() -> None:
    cell = "a.py::f（部分：第 00 轮）（另见第 01 轮的说明）; b.py::g (部分：第 02 轮); c.yaml"
    assert trace.split_locations(cell) == ["a.py::f", "b.py::g", "c.yaml"]
    assert trace.partial_rounds(cell) == {"00", "02"}


def test_main_runs_end_to_end_on_given_files(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = tmp_path / "SPEC.md"
    spec.write_text(SPEC, encoding="utf-8")
    trace_file = tmp_path / "TRACE.md"
    trace_file.write_text(table(ROW_OK, ROW_PENDING, ROW_SHARED), encoding="utf-8")
    monkeypatch.setattr(trace, "collect_tests", lambda root: COLLECTED)
    argv = ["--spec", str(spec), "--trace", str(trace_file), "--root", str(repo)]
    assert trace.main([*argv, "--round", "00"]) == 0
    assert "OK" in capsys.readouterr().out
    trace_file.write_text(table(ROW_PENDING), encoding="utf-8")
    assert trace.main(argv) == 1
    assert "problem(s)" in capsys.readouterr().out


def test_collect_tests_runs_pytest_collection() -> None:
    collected = trace.collect_tests(SCRIPTS.parent)
    assert "tests/unit/test_trace_check.py::test_collect_tests_runs_pytest_collection" in collected
    assert any(
        item.startswith("tests/unit/test_crypto.py::test_round_trip_property") for item in collected
    )


def test_the_real_documents_are_consistent_with_each_other() -> None:
    """SPEC ids and table 2 rows must line up (global consistency, no round filter)."""
    root = SCRIPTS.parent
    problems, stats = trace.check(
        (root / "docs" / "SPEC.md").read_text(encoding="utf-8"),
        (root / "docs" / "TRACEABILITY.md").read_text(encoding="utf-8"),
        trace.collect_tests(root),
        root,
    )
    assert problems == [], "\n".join(problems)
    assert stats["spec"] == stats["rows"] > 150
