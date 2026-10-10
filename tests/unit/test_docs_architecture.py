"""docs/ARCHITECTURE.md describes the code that exists (R-NFR-006).

The architecture document is only useful while it is true.  So the test reads it the way the code
reads itself:

* every ``twin.xxx`` name in a diagram or in the text (a module, or a class, function or constant
  in a module) exists in ``src/twin``;
* the states of the state diagram are the states of ``conversation_state``;
* the table overview lists exactly the tables of the database models - every table of the SPEC
  (R-STO-006) among them;
* the components it lists are the components ``twin run`` registers;
* every decision it cites (D-nnn) is in DECISIONS.md, and every link in the project's documents
  leads to a file and a heading that exist;
* the Mermaid blocks are well formed enough to render (a known diagram type, balanced subgraphs,
  quotes and brackets).
"""

from __future__ import annotations

import ast
import re
import unicodedata
from pathlib import Path

import pytest

from tests.support.scripts import load_script
from twin.storage import engine_models
from twin.storage.models import Base

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "twin"
ARCHITECTURE = ROOT / "docs" / "ARCHITECTURE.md"
DOCUMENTS = [
    ROOT / "README.md",
    ROOT / "docs" / "RUNBOOK.md",
    ROOT / "docs" / "ARCHITECTURE.md",
    ROOT / "docs" / "RELEASE_CHECKLIST.md",
]
_checks = load_script("decisions_check")

DOTTED = re.compile(r"(?<![\w.])twin(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
MERMAID = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)
DIAGRAM_TYPES = ("flowchart", "graph", "sequenceDiagram", "stateDiagram-v2", "classDiagram")


def text() -> str:
    return ARCHITECTURE.read_text(encoding="utf-8")


def mermaid_blocks() -> list[str]:
    return MERMAID.findall(text())


# ---------------------------------------------------------------------- the names


def module_file(parts: list[str]) -> Path | None:
    base = SRC.joinpath(*parts[1:])
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def resolves(dotted: str) -> bool:
    """A module, or a name defined in a module (``twin.engine.machine.ConversationEngine``).

    Anything after the first name inside the module (``Base.metadata``) is an attribute of that
    name and is not looked at further.
    """
    parts = dotted.split(".")
    for end in range(len(parts), 0, -1):
        path = module_file(parts[:end])
        if path is None:
            continue
        rest = parts[end:]
        if not rest:
            return True
        names = _checks.defined_names(path.read_text(encoding="utf-8"))
        return rest[0] in names or (
            path.name == "__init__.py" and module_file([*parts[:end], rest[0]]) is not None
        )
    return False


def test_every_name_in_the_diagrams_exists() -> None:
    blocks = mermaid_blocks()
    assert len(blocks) >= 8
    missing = sorted(
        {name for block in blocks for name in DOTTED.findall(block) if not resolves(name)}
    )
    assert missing == []


def test_every_name_in_the_text_exists() -> None:
    prose = MERMAID.sub("", text())
    spans = re.findall(r"`([^`\n]+)`", prose)
    named = {m for span in spans for m in DOTTED.findall(span)}
    assert len(named) > 5
    assert sorted(name for name in named if not resolves(name)) == []


def test_the_name_check_itself_catches_a_wrong_name() -> None:
    assert resolves("twin.engine.machine.ConversationEngine")
    assert resolves("twin.engine.machine")
    assert resolves("twin.storage.models.Base.metadata")
    assert resolves("twin.engine")
    assert not resolves("twin.engine.machine.NoSuchEngine")
    assert not resolves("twin.engine.nonsense")
    assert not resolves("twin.nonsense")


# -------------------------------------------------------------------- the diagrams


def test_the_diagrams_are_well_formed_enough_to_render() -> None:
    for block in mermaid_blocks():
        first = block.strip().splitlines()[0]
        assert first.startswith(DIAGRAM_TYPES), first
        lines = [line.strip() for line in block.splitlines()]
        opened = sum(1 for line in lines if line.startswith("subgraph "))
        closed = sum(1 for line in lines if line == "end")
        assert opened == closed, f"unbalanced subgraph/end in the diagram that starts: {first}"
        assert block.count('"') % 2 == 0, f"odd number of quotes in: {first}"
        for opening, closing in (("[", "]"), ("(", ")"), ("{", "}")):
            assert block.count(opening) == block.count(closing), (
                f"unbalanced {opening}{closing} in: {first}"
            )


def test_a_message_of_a_sequence_diagram_has_no_semicolon() -> None:
    """In a sequence diagram ``;`` ends a statement: a message that holds one does not parse
    (the real Mermaid parser said so about this document)."""
    for block in mermaid_blocks():
        if block.strip().startswith("sequenceDiagram"):
            assert [line for line in block.splitlines() if ";" in line] == []


def test_there_is_a_component_diagram_the_flows_and_the_state_machine() -> None:
    kinds = [block.strip().splitlines()[0].split()[0] for block in mermaid_blocks()]
    assert kinds.count("sequenceDiagram") >= 1
    assert kinds.count("stateDiagram-v2") == 1
    assert kinds.count("flowchart") >= 6
    for heading in (
        "## 2. 组件图",
        "### 4.1 消息进入 → 回复 → 发送",
        "### 4.2 主动消息",
        "### 4.3 记忆",
        "### 4.5 训练到部署",
        "## 5. 引擎状态机",
        "## 6. 表结构概览",
        "## 7. 关键设计决策",
    ):
        assert heading in text(), heading


def test_the_state_diagram_has_the_states_of_the_conversation_state_table() -> None:
    (diagram,) = [b for b in mermaid_blocks() if b.strip().startswith("stateDiagram-v2")]
    states = set(re.findall(r"\b[A-Z]{4,}\b", diagram))
    assert states == set(engine_models.STATES)
    sequence = re.findall(
        r"IDLE --> COLLECTING|COLLECTING --> DECIDING|DECIDING --> GENERATING"
        r"|GENERATING --> SENDING|SENDING --> IDLE",
        diagram,
    )
    assert len(sequence) == 5  # the main line of R-ENG-001


def test_the_main_line_is_also_the_one_in_the_engine_module() -> None:
    doc = (SRC / "engine" / "machine.py").read_text(encoding="utf-8")
    assert "IDLE --message--> COLLECTING --quiet--> DECIDING --due--> GENERATING" in doc
    for state in engine_models.STATES:
        assert f"STATE_{state}" in doc


# ------------------------------------------------------------------------- tables


def documented_tables() -> list[str]:
    section = text().split("## 6. 表结构概览")[1].split("## 7. ")[0]
    names: list[str] = []
    for line in section.splitlines():
        if not line.startswith("|") or line.startswith("| ---") or "领域" in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        names.extend(re.findall(r"`([a-z_]+)`", cells[1]))
    return names


def spec_tables() -> list[str]:
    spec = (ROOT / "docs" / "SPEC.md").read_text(encoding="utf-8")
    line = next(row for row in spec.splitlines() if row.startswith("- **R-STO-006**"))
    return re.findall(r"`([a-z_]+)`", line)


def test_the_table_overview_lists_exactly_the_tables_of_the_database_models() -> None:
    import importlib
    import pkgutil

    import twin.storage

    for module in pkgutil.iter_modules(twin.storage.__path__):
        if module.name.endswith("models"):
            importlib.import_module(f"twin.storage.{module.name}")
    real = set(Base.metadata.tables)
    listed = documented_tables()
    assert len(listed) == len(set(listed)), "a table is listed twice"
    assert set(listed) == real, sorted(set(listed) ^ real)


def test_every_table_the_spec_requires_exists_and_is_documented() -> None:
    required = set(spec_tables())
    assert len(required) >= 36
    assert required <= set(Base.metadata.tables) | {"channel_state"}
    assert required <= set(documented_tables())


def test_the_overview_says_how_many_tables_there_are() -> None:
    import twin.storage.models  # noqa: F401

    count = len(Base.metadata.tables)
    assert f"共 {count} 张" in text()
    assert f"SPEC R-STO-006 的 {len(spec_tables())} 张" in text()


# ---------------------------------------------------------------------- components

COMPONENT_CONSTANTS = {"name", "COMPONENT_NAME", "CHANNEL_COMPONENT_NAME"}


def component_names_in_code() -> set[str]:
    found: set[str] = set()
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Assign | ast.AnnAssign):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if (
                isinstance(value, ast.Constant)
                and isinstance(value.value, str)
                and any(isinstance(t, ast.Name) and t.id in COMPONENT_CONSTANTS for t in targets)
            ):
                found.add(value.value)
    return found


def test_the_components_the_text_lists_are_components_of_the_code() -> None:
    line = next(row for row in text().splitlines() if row.startswith("`twin run` 里注册的组件"))
    listed = re.findall(r"`([a-z_]+)`", line.split("：", 1)[1])
    assert len(listed) >= 14 and len(set(listed)) == len(listed)
    assert sorted(set(listed) - component_names_in_code()) == []


# ---------------------------------------------------------- decisions and the links


def decision_ids() -> set[str]:
    decisions = (ROOT / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
    return {ident for ident, _ in _checks.decision_ids(decisions)}


def test_every_decision_the_document_cites_exists() -> None:
    cited = {ident for ident, _ in _checks.references(text())}
    assert len(cited) >= 20
    assert sorted(cited - decision_ids()) == []


def github_slug(heading: str) -> str:
    """The anchor GitHub gives a heading: lower case, letters/digits/hyphens/spaces only."""
    kept = []
    for char in heading.strip().lower():
        category = unicodedata.category(char)
        if category[0] in "LMN" or category == "Pc" or char in "- ":
            kept.append(char)
    return "".join(kept).replace(" ", "-")


def headings_of(path: Path) -> set[str]:
    slugs: set[str] = set()
    in_block = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            in_block = not in_block
        elif not in_block and re.match(r"#{1,6} ", line):
            slugs.add(github_slug(re.sub(r"^#+\s+", "", line)))
    return slugs


def links_of(path: Path) -> list[tuple[str, str]]:
    body = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)
    found = []
    for target in re.findall(r"\]\(([^)\s]+)\)", body):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        file, _, anchor = target.partition("#")
        found.append((file, anchor))
    return found


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda p: p.name)
def test_the_links_of_the_documents_lead_somewhere(document: Path) -> None:
    broken = []
    for file, anchor in links_of(document):
        target = document if not file else (document.parent / file).resolve()
        if not target.exists():
            broken.append(f"{file}: no such file")
        elif anchor and target.suffix == ".md" and anchor not in headings_of(target):
            broken.append(f"{file}#{anchor}: no such heading")
    assert broken == []


def test_the_slugger_matches_what_github_makes_of_the_headings_used_here() -> None:
    assert (
        github_slug("0. 先读：命令的进程类别与“先停应用”规则")
        == "0-先读命令的进程类别与先停应用规则"
    )
    assert github_slug("6. 训练风格模型（AutoDL）") == "6-训练风格模型autodl"
    assert github_slug("13. M0–M5 里程碑清单") == "13-m0m5-里程碑清单"
    assert github_slug("附录 A：全部 CLI 命令与进程类别") == "附录-a全部-cli-命令与进程类别"
