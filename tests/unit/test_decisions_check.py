"""scripts/decisions_check.py: numbering, trade-off table and cited tests (R-SCOPE-009)."""

from __future__ import annotations

import textwrap
from collections.abc import Iterable
from pathlib import Path

import pytest

from tests.support.scripts import SCRIPTS, load_script

check = load_script("decisions_check")
ROOT = SCRIPTS.parent


def d(number: int) -> str:
    """A decision id, assembled here so that this file does not cite one by accident."""
    return f"D-{number}"


def doc(text: str) -> str:
    return textwrap.dedent(text).lstrip("\n")


def table(*rows: str) -> str:
    head = (
        "| 取舍点 | 负责轮次 | 采用的顺序 | 优先级理由 | 固定该行为的测试名 |\n"
        "| --- | --- | --- | --- | --- |\n"
    )
    return head + "\n".join(rows) + "\n"


SPEC = (
    "- **R-SCOPE-009** 体验目标按优先级。可验证的落实方式：`docs/DECISIONS.md` 逐条列出每个取舍点"
    "（至少：预算降级顺序、平台条数不足时的气泡合并、睡眠与回复），写明采用的顺序。"
)


# ------------------------------------------------------------------------------ numbering


def test_decision_rows_are_found_with_their_lines() -> None:
    text = doc(
        f"""
        | 编号 | 主题 |
        | --- | --- |
        | {d(101)} | a |
        | {d(105)} | b |
        text mentioning {d(101)} is not a row
        """
    )
    assert check.decision_ids(text) == [(d(101), 3), (d(105), 4)]


def test_a_number_used_twice_is_reported_but_a_gap_is_not() -> None:
    assert check.duplicate_ids([(d(1), 3), (d(2), 4), (d(4), 5)]) == []
    problems = check.duplicate_ids([(d(1), 3), (d(2), 4), (d(1), 9)])
    assert problems == [f"{d(1)} is defined twice (lines 3 and 9)"]


def test_every_mention_of_a_decision_is_found_but_not_longer_numbers_or_words() -> None:
    text = f"see {d(101)} and {d(202)}.\nrange {d(490)}..{d(497)}\nnot D-12 nor D-1234 nor XD-123"
    assert check.references(text) == [
        (d(101), 1),
        (d(202), 1),
        (d(490), 2),
        (d(497), 2),
    ]


def test_a_mention_of_a_decision_that_does_not_exist_is_reported_with_its_place(
    tmp_path: Path,
) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "A.md").write_text(f"ok {d(101)}\nbroken {d(999)}\n", encoding="utf-8")
    (tmp_path / "x.py").write_text(f"# {d(101)}\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text(f"# {d(998)}\n", encoding="utf-8")
    problems = check.unknown_references(tmp_path, {d(101)})
    assert problems == [
        f"docs/A.md:2: {d(999)} is mentioned but docs/DECISIONS.md has no such decision"
    ]


# ------------------------------------------------------------------- trade-off table


def test_the_points_the_spec_requires_are_read_from_r_scope_009() -> None:
    assert check.spec_points(SPEC) == ["预算降级顺序", "平台条数不足时的气泡合并", "睡眠与回复"]
    assert check.spec_points("nothing here") == []
    assert len(check.spec_points((ROOT / "docs" / "SPEC.md").read_text(encoding="utf-8"))) >= 6


def good_row(point: str, node: str = "`tests/unit/test_a.py::test_x`") -> str:
    return f"| {point}（R-X-1） | 09 | ① 先后 | 像她排第一 | {node} |"


def tradeoff_doc(*rows: str) -> str:
    return (
        "## 1. SPEC 偏差\n\n| x |\n\n## 2. 取舍记录（R-SCOPE-009）\n\n"
        + table(*rows)
        + "\n## 3. 实现\n"
    )


def test_the_table_is_read_row_by_row() -> None:
    rows = check.tradeoff_rows(tradeoff_doc(good_row("预算降级顺序"), good_row("睡眠与回复")))
    assert [row.point for row in rows] == ["预算降级顺序（R-X-1）", "睡眠与回复（R-X-1）"]
    assert rows[0].order == "① 先后" and rows[0].line == 9


def test_a_complete_table_has_no_problems() -> None:
    text = tradeoff_doc(good_row("预算降级顺序"), good_row("平台条数不足时的气泡合并"))
    assert check.check_tradeoffs(text, ["预算降级顺序", "平台条数不足时的气泡合并"]) == []


def test_a_missing_trade_off_point_is_reported() -> None:
    text = tradeoff_doc(good_row("预算降级顺序"))
    assert check.check_tradeoffs(text, ["预算降级顺序", "睡眠与回复"]) == [
        "trade-off point missing from the table: 睡眠与回复"
    ]


@pytest.mark.parametrize(
    "row",
    [
        "| 预算降级顺序 | 01 |  | 理由 | `tests/unit/test_a.py::test_x` |",
        "| 预算降级顺序 | 01 | 顺序 |  | `tests/unit/test_a.py::test_x` |",
        "| 预算降级顺序 | 01 | 顺序 | 理由 |  |",
        "| 预算降级顺序 | 01 | 顺序 | 理由 | 见实现 |",
        "| 预算降级顺序 | 01 | 顺序 | 理由 | `tests/unit/test_a.py` |",
        "| 预算降级顺序 | 顺序 | 理由 | `tests/unit/test_a.py::test_x` |",
    ],
)
def test_a_row_without_order_reason_or_a_test_node_is_reported(row: str) -> None:
    problems = check.check_tradeoffs(tradeoff_doc(row), ["预算降级顺序"])
    assert len(problems) == 1 and "预算降级顺序" in problems[0]


def test_a_document_without_the_table_is_reported() -> None:
    assert "no trade-off table" in check.check_tradeoffs("# nothing\n", ["a"])[0]


# ---------------------------------------------------------------------------- test nodes


def test_cited_tests_are_found_in_every_form_a_decision_uses() -> None:
    text = (
        "| D | `tests/unit/test_a.py::test_one`；`::test_two` 与 test_b.py::TestK::test_three |\n"
        "tests/unit/test_c.py::test_param[a-b] `::test_w*` and tests/unit/test_d.py\n"
        "prose test_e.py without a node\n"
    )
    index = {"test_b.py": "tests/unit/test_b.py"}
    found = check.node_refs(text, index)
    assert [(ref.node, ref.line, ref.is_file) for ref in found] == [
        ("tests/unit/test_a.py::test_one", 1, False),
        ("tests/unit/test_a.py::test_two", 1, False),
        ("tests/unit/test_b.py::TestK::test_three", 1, False),
        ("tests/unit/test_c.py::test_param[a-b]", 2, False),
        ("tests/unit/test_c.py::test_w*", 2, False),
        ("tests/unit/test_d.py", 2, True),
    ]


def test_the_short_form_continues_only_the_file_of_its_own_line() -> None:
    found = check.node_refs("tests/unit/test_a.py::test_x\n`::test_y`\n", None)
    assert [ref.node for ref in found] == ["tests/unit/test_a.py::test_x"]


def test_a_path_in_the_source_tree_is_not_taken_for_a_test() -> None:
    assert check.node_refs("src/twin/test_like.py::thing and twin/test_q.py::x", None) == []


def test_the_file_index_lists_only_names_that_are_unique(tmp_path: Path) -> None:
    for name in ("tests/unit/test_a.py", "tests/unit/test_b.py", "tests/integration/test_b.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    assert check.test_file_index(tmp_path) == {"test_a.py": "tests/unit/test_a.py"}


def test_collected_matches_exact_parametrised_class_and_prefix_forms() -> None:
    collected = {
        "tests/unit/test_a.py::test_x",
        "tests/unit/test_a.py::test_p[1]",
        "tests/unit/test_a.py::TestK::test_m",
        "tests/unit/test_a.py::test_redo_one",
    }
    for node in (
        "tests/unit/test_a.py::test_x",
        "tests/unit/test_a.py::test_p",
        "tests/unit/test_a.py::TestK",
        "tests/unit/test_a.py::TestK::test_m",
        "tests/unit/test_a.py::test_redo_*",
    ):
        assert check.is_collected(node, collected), node
    for node in ("tests/unit/test_a.py::test_y", "tests/unit/test_a.py::test_nope_*"):
        assert not check.is_collected(node, collected), node


def fake_repo(tmp_path: Path) -> Path:
    unit = tmp_path / "tests" / "unit"
    unit.mkdir(parents=True)
    (unit / "test_a.py").write_text(
        "import pytest\n\n@pytest.fixture\ndef rig():\n    return 1\n\n"
        "def test_real():\n    pass\n",
        encoding="utf-8",
    )
    support = tmp_path / "tests" / "support"
    support.mkdir()
    (support / "world.py").write_text(
        "class World:\n    def build(self):\n        return 1\n\n"
        "def build_world():\n    return 2\n",
        encoding="utf-8",
    )
    return tmp_path


def collector(found: Iterable[str]):
    names = set(found)

    def collect(root: Path, files: Iterable[str]) -> set[str]:
        assert all(name.endswith(".py") and name.startswith("tests/") for name in files)
        return names

    return collect


def refs(*nodes: str) -> list[object]:
    return [check.NodeRef(node, 1, is_file="::" not in node) for node in nodes]


def test_a_cited_test_that_pytest_does_not_find_is_reported(tmp_path: Path) -> None:
    root = fake_repo(tmp_path)
    ok = ["tests/unit/test_a.py::test_real"]
    gone = refs("tests/unit/test_a.py::test_real", "tests/unit/test_a.py::test_renamed")
    problems = check.check_nodes(gone, root, collector(ok))  # type: ignore[arg-type]
    assert problems == [
        "line 1: not found by pytest --collect-only: tests/unit/test_a.py::test_renamed"
    ]


def test_a_cited_file_that_does_not_exist_is_reported(tmp_path: Path) -> None:
    root = fake_repo(tmp_path)
    problems = check.check_nodes(refs("tests/unit/test_gone.py::test_x"), root, collector([]))  # type: ignore[arg-type]
    assert problems == ["test file does not exist: tests/unit/test_gone.py"]


def test_fixtures_and_helpers_are_checked_by_their_definition_not_by_collection(
    tmp_path: Path,
) -> None:
    root = fake_repo(tmp_path)
    good = refs(
        "tests/unit/test_a.py::rig",
        "tests/support/world.py::build_world",
        "tests/support/world.py::World::build",
    )
    assert check.check_nodes(good, root, collector([])) == []  # type: ignore[arg-type]
    bad = refs("tests/unit/test_a.py::no_such_fixture", "tests/support/world.py::missing")
    assert check.check_nodes(bad, root, collector([])) == [  # type: ignore[arg-type]
        "line 1: not found by the definitions of the module: tests/unit/test_a.py::no_such_fixture",
        "line 1: not found by the definitions of the module: tests/support/world.py::missing",
    ]


def test_a_bare_file_citation_only_has_to_exist(tmp_path: Path) -> None:
    root = fake_repo(tmp_path)
    assert check.check_nodes(refs("tests/unit/test_a.py"), root, collector([])) == []  # type: ignore[arg-type]


# --------------------------------------------------------------- the whole check on files


def write_repo(tmp_path: Path, decisions: str, spec: str = SPEC) -> Path:
    root = fake_repo(tmp_path)
    (root / "docs").mkdir()
    (root / "docs" / "DECISIONS.md").write_text(decisions, encoding="utf-8")
    (root / "docs" / "SPEC.md").write_text(spec, encoding="utf-8")
    return root


def rows_for(*points: str) -> str:
    return "\n".join(good_row(p, "`tests/unit/test_a.py::test_real`") for p in points)


def test_a_consistent_document_passes(tmp_path: Path) -> None:
    decisions = (
        tradeoff_doc(rows_for("预算降级顺序", "平台条数不足时的气泡合并", "睡眠与回复"))
        + f"| {d(101)} | 00 | a | see {d(101)} and `tests/unit/test_a.py::test_real` |\n"
    )
    root = write_repo(tmp_path, decisions)
    problems, stats = check.check(root, collector(["tests/unit/test_a.py::test_real"]))
    assert problems == []
    assert stats == {"decisions": 1, "tradeoffs": 3, "required_points": 3, "cited_tests": 1}


def test_every_kind_of_problem_is_found_in_one_run(tmp_path: Path) -> None:
    decisions = (
        tradeoff_doc(rows_for("预算降级顺序", "睡眠与回复"))
        + f"| {d(101)} | 00 | a | `tests/unit/test_a.py::test_renamed` |\n"
        + f"| {d(101)} | 00 | b | points at {d(555)} |\n"
    )
    root = write_repo(tmp_path, decisions)
    problems, _ = check.check(root, collector(["tests/unit/test_a.py::test_real"]))
    text = "\n".join(problems)
    assert "defined twice" in text and f"{d(555)} is mentioned" in text
    assert "missing from the table: 平台条数不足时的气泡合并" in text
    assert "tests/unit/test_a.py::test_renamed" in text
    assert len(problems) == 4


def test_the_command_reports_and_exits_with_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(check, "collect_nodes", collector(["tests/unit/test_a.py::test_real"]))
    decisions = tradeoff_doc(rows_for("预算降级顺序", "平台条数不足时的气泡合并", "睡眠与回复"))
    root = write_repo(tmp_path / "good", decisions)
    assert check.main(["--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "decisions_check: 0 decisions, 3 trade-off rows (3 required by the SPEC)" in out
    assert out.rstrip().endswith("OK")
    broken = write_repo(tmp_path / "bad", tradeoff_doc(rows_for("预算降级顺序")))
    assert check.main(["--root", str(broken)]) == 1
    out = capsys.readouterr().out
    assert "2 problem(s):" in out and "missing from the table: 睡眠与回复" in out


# ------------------------------------------------------------------------ this repository


def test_the_decisions_of_this_repository_are_consistent() -> None:
    """The real document: unique numbers, no dangling reference, complete trade-off table.

    The cited tests are looked up in the syntax trees of their modules here (pytest would have to
    import every test module to collect them again); ``scripts/decisions_check.py`` itself, run
    by CI, asks ``pytest --collect-only``.
    """

    def by_syntax(root: Path, files: Iterable[str]) -> set[str]:
        import ast

        found: set[str] = set()
        for name in files:
            tree = ast.parse((root / name).read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    found.add(f"{name}::{node.name}")
                elif isinstance(node, ast.ClassDef):
                    for member in node.body:
                        if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
                            found.add(f"{name}::{node.name}::{member.name}")
        return found

    problems, stats = check.check(ROOT, by_syntax)
    assert problems == [], "\n".join(problems)
    assert stats["decisions"] > 300 and stats["tradeoffs"] >= stats["required_points"] >= 6
    assert stats["cited_tests"] > 80


def test_the_trade_off_table_names_every_point_of_the_spec() -> None:
    text = (ROOT / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
    spec = (ROOT / "docs" / "SPEC.md").read_text(encoding="utf-8")
    names = [row.point for row in check.tradeoff_rows(text)]
    for point in check.spec_points(spec):
        assert any(point in name for name in names), point
    for row in check.tradeoff_rows(text):
        assert row.order and row.reason and row.tests, row.point
