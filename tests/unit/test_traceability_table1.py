"""Table 1 of TRACEABILITY.md: every product requirement says how it is verified (round 16).

Table 1 maps the points of the product document to SPEC numbers.  Its last column, "验证方式",
names for each point the tests that fix the behaviour (pytest node ids) and, where there is one,
the command that shows it.  This test holds the column to what it promises:

* no row is empty, and every row names at least one test or one command;
* every test it cites is found by ``pytest --collect-only`` (run on the cited files, so a renamed or
  deleted test is noticed here and not in a year);
* every ``twin`` command it cites exists, with the options it uses; every script it cites exists;
* the SPEC numbers a row refers to are defined in the SPEC.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.support.cli_tree import PENDING_COMMANDS, global_options, resolve
from tests.support.scripts import load_script

ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT / "docs" / "TRACEABILITY.md"
checks = load_script("decisions_check")

COMMAND = re.compile(r"`(?:uv run )?(twin [^`]+)`")
SCRIPT = re.compile(r"`uv run python (scripts/[\w/.\-]+\.py)[^`]*`")
OPTION = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")
SPEC_ID = re.compile(r"R-[A-Z]+-\d{3}")


@dataclass(frozen=True)
class Row:
    ident: str
    point: str
    spec: str
    rounds: str
    how: str


def table_1() -> list[Row]:
    text = TRACE.read_text(encoding="utf-8")
    section = text[text.index("## 表 1") : text.index("## 表 2")]
    rows = []
    for line in section.splitlines():
        if not line.startswith("| P-"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        assert len(cells) == 6, line[:60]
        rows.append(Row(cells[0], cells[1], cells[3], cells[4], cells[5]))
    return rows


def cited_tests(how: str) -> list[str]:
    return [ref.node for ref in checks.node_refs(how) if not ref.is_file]


def test_the_table_has_all_105_points_in_order() -> None:
    rows = table_1()
    assert [row.ident for row in rows] == [f"P-{n:02d}" for n in range(1, len(rows) + 1)]
    assert len(rows) >= 105
    assert "验证方式（第 16 轮填写）" not in TRACE.read_text(encoding="utf-8")


def test_no_point_is_left_without_a_way_to_verify_it() -> None:
    empty = [row.ident for row in table_1() if not row.how.strip()]
    assert empty == []
    nothing_to_run = [
        row.ident
        for row in table_1()
        if not cited_tests(row.how) and not COMMAND.search(row.how) and not SCRIPT.search(row.how)
    ]
    assert nothing_to_run == []


def test_most_points_name_tests_and_every_test_is_named_in_full() -> None:
    rows = table_1()
    with_tests = [row for row in rows if cited_tests(row.how)]
    assert len(with_tests) >= len(rows) - 1
    for row in rows:
        for node in cited_tests(row.how):
            assert node.startswith("tests/") and "::test_" in node, (row.ident, node)


def test_every_test_cited_is_found_by_pytest_collect_only() -> None:
    cited = {node for row in table_1() for node in cited_tests(row.how)}
    assert len(cited) > 150
    collected = checks.collect_nodes(ROOT, {node.split("::")[0] for node in cited})
    missing = sorted(node for node in cited if not checks.is_collected(node, collected))
    assert missing == []


def test_every_command_cited_exists_with_the_options_it_uses() -> None:
    problems = []
    seen = 0
    for row in table_1():
        for text in COMMAND.findall(row.how):
            seen += 1
            words = text.removeprefix("twin ").split()
            info, _ = resolve(words)
            group = tuple(word for word in words if not word.startswith(("-", "<", "[")))[:2]
            if info is None:
                if group not in PENDING_COMMANDS:
                    problems.append(f"{row.ident}: twin {' '.join(words)} does not exist")
                    continue
                allowed = set(PENDING_COMMANDS[group]) | {"--help"}
            else:
                allowed = set(info.options) | set(global_options()) | {"--help"}
            for option in OPTION.findall(text):
                if option not in allowed:
                    problems.append(f"{row.ident}: twin {' '.join(group)} has no {option}")
    assert seen > 50
    assert problems == []


def test_every_script_cited_exists() -> None:
    scripts = [(row.ident, name) for row in table_1() for name in SCRIPT.findall(row.how)]
    assert scripts
    assert [(ident, name) for ident, name in scripts if not (ROOT / name).is_file()] == []


def test_the_spec_numbers_of_every_row_are_defined_in_the_spec() -> None:
    spec = (ROOT / "docs" / "SPEC.md").read_text(encoding="utf-8")
    defined = set(re.findall(r"^\s*-\s+\*\*(R-[A-Z]+-\d{3})\*\*", spec, re.MULTILINE))
    undefined = [
        f"{row.ident}: {ident}"
        for row in table_1()
        for ident in SPEC_ID.findall(row.spec)
        if ident not in defined
    ]
    assert undefined == []


@pytest.mark.parametrize(
    ("how", "tests", "commands"),
    [
        (
            "tests/unit/test_a.py::test_x; tests/unit/test_b.py::test_y; "
            "演示：`uv run twin health`",
            ["tests/unit/test_a.py::test_x", "tests/unit/test_b.py::test_y"],
            ["twin health"],
        ),
        (
            "演示：`uv run twin eval gate M1` `uv run twin cost report`",
            [],
            ["twin eval gate M1", "twin cost report"],
        ),
    ],
)
def test_the_cells_are_read_as_tests_and_commands(
    how: str, tests: list[str], commands: list[str]
) -> None:
    assert cited_tests(how) == tests
    assert COMMAND.findall(how) == commands


def test_a_wrong_citation_would_be_found() -> None:
    cell = "tests/unit/test_decisions_check.py::test_no_such_test_exists; `uv run twin nonsense`"
    collected = checks.collect_nodes(ROOT, {"tests/unit/test_decisions_check.py"})
    assert not checks.is_collected(cited_tests(cell)[0], collected)
    assert resolve(["nonsense"])[0] is None
