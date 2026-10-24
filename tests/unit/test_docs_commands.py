"""The documents name only commands that exist (R-NFR-006).

``README.md`` and ``docs/RUNBOOK.md`` (and the release checklist) tell the person what to type.
A command that was renamed, or never written, must not stay in them, so this test reads the
documents the way a person would follow them:

* every ``twin ...`` in a code block or an inline code span names a command of the real Typer
  tree, and every ``--option`` after it is an option of that command (or a global one);
* every ``/指令`` in them is in the chat command table of the running application;
* the quick-reference table of the README is exactly the table of the application (syntax and
  example), and lists every chat command;
* the appendix of the RUNBOOK lists every CLI command with its process class (R-ARCH-006), and
  agrees with the declarations in the code.

The commands of round 15 (``twin eval consistency|cost|report``) are described in the documents as
``prompts/15-evaluation.md`` describes them; they are accepted as long as the command does not
exist yet and are checked like the others as soon as it does.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.support.cli_tree import PENDING_COMMANDS, command_tree, global_options, resolve
from tests.support.clock import ManualClock
from tests.support.commands_world import CommandWorld, open_world
from tests.support.embedding import HashingBackend
from twin.cli import app
from twin.commands.parse import parse_command
from twin.commands.rating import rating_command
from twin.ops.process_model import CommandKind, get_spec, iter_commands
from twin.schedule.proactive.store import RatingStore
from twin.services import Services

ROOT = Path(__file__).resolve().parents[2]
DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "RUNBOOK.md",
    ROOT / "docs" / "RELEASE_CHECKLIST.md",
]

# typed in `twin chat --local` and handled by the terminal itself, not by the chat command table
TERMINAL_COMMANDS = {"img", "quit", "exit", "help"}

FENCE = re.compile(r"^(\s*)```")
SPAN = re.compile(r"`([^`\n]+)`")
PROMPT = re.compile(r"^(?:PS [^>]*>|PS>|\$|>)\s+")
WORDS = re.compile(r"^[a-z][a-z0-9-]*(?:\|[a-z][a-z0-9-]*)*$")
OPTION = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")
SLASH_NAME = re.compile(r"^[/／]([^\s/／:：]+)")


@dataclass(frozen=True)
class Found:
    """A command written in a document."""

    file: str
    line: int
    text: str  # what follows ``twin``


def unescape(cell: str) -> str:
    return cell.replace("\\|", "|")


def doc_lines(path: Path) -> Iterator[tuple[int, str, bool]]:
    """``(number, text, in_code_block)`` for every line of a document."""
    in_block = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if FENCE.match(line):
            in_block = not in_block
            continue
        yield number, line, in_block


def strip_comment(line: str) -> str:
    return re.split(r"\s+#\s", line, maxsplit=1)[0].rstrip()


def twin_commands(path: Path, root: Path = ROOT) -> list[Found]:
    """Every ``twin ...`` the document tells the person to run."""
    found: list[Found] = []
    name = path.relative_to(root).as_posix()

    def add(number: int, text: str) -> None:
        text = PROMPT.sub("", text.strip())
        text = re.sub(r"^uv run ", "", text)
        if text.startswith("twin "):
            found.append(Found(name, number, strip_comment(text[len("twin ") :])))

    for number, line, in_block in doc_lines(path):
        if in_block:
            add(number, strip_comment(line))
            continue
        for span in SPAN.findall(line):
            add(number, unescape(span))
    return found


def expand(text: str) -> list[list[str]]:
    """The word lists a command line stands for: ``service start|stop`` is two commands."""
    tokens = text.split()
    head = 0
    while head < len(tokens) and WORDS.match(tokens[head]):
        head += 1
    variants: list[list[str]] = [[]]
    for token in tokens[:head]:
        variants = [[*v, option] for v in variants for option in token.split("|")]
    return [[*v, *tokens[head:]] for v in variants]


def slash_commands(path: Path) -> list[tuple[int, str]]:
    """``(line, command text)`` of every chat command written in a code span."""
    found: list[tuple[int, str]] = []
    for number, line, in_block in doc_lines(path):
        if in_block:
            continue
        for span in SPAN.findall(line):
            span = unescape(span)
            named = SLASH_NAME.match(span)
            if named is None:
                continue
            name = named.group(1)
            chinese = re.fullmatch(r"[一-鿿]+", name) is not None
            english_alias = re.fullmatch(r"[a-z_]{3,}", name) is not None and "/" not in span[1:]
            if chinese or english_alias:
                found.append((number, span))
    return found


# ------------------------------------------------------------------------- the CLI


def test_the_documents_exist_and_have_commands_in_them() -> None:
    for path in DOCS:
        assert path.is_file(), path
    assert len(twin_commands(ROOT / "docs" / "RUNBOOK.md")) > 80
    assert len(twin_commands(ROOT / "README.md")) > 25


def test_the_extraction_reads_blocks_and_spans_and_expands_alternatives(tmp_path: Path) -> None:
    sample = tmp_path / "doc.md"
    sample.write_text(
        "text `twin service start|stop` and `uv run twin import <目录> [--resume]`\n"
        "```powershell\n"
        "PS> uv run twin eval gate M1   # judged\n"
        "twin settings set a.b 1\n"
        "echo twin not-a-command\n"
        "```\n"
        "`twin.exe supervise` and `/状态`\n",
        encoding="utf-8",
    )
    texts = [(f.line, f.text) for f in twin_commands(sample, tmp_path)]
    assert texts == [
        (1, "service start|stop"),
        (1, "import <目录> [--resume]"),
        (3, "eval gate M1"),
        (4, "settings set a.b 1"),
    ]
    assert expand("service start|stop") == [["service", "start"], ["service", "stop"]]
    assert expand("import <目录> [--resume]") == [["import", "<目录>", "[--resume]"]]
    assert expand("eval gate M0|M1") == [["eval", "gate", "M0|M1"]]


def all_commands() -> Iterator[tuple[Found, list[str]]]:
    for path in DOCS:
        for found in twin_commands(path):
            for words in expand(found.text):
                yield found, words


def known_options(words: list[str]) -> tuple[set[str], str] | None:
    info, _ = resolve(words)
    if info is None:
        return None
    return set(info.options) | set(global_options()) | {"--help"}, info.name


def test_every_command_in_the_documents_exists() -> None:
    unknown = []
    for found, words in all_commands():
        info, _ = resolve(words)
        if info is not None:
            continue
        group = tuple(word for word in words if not word.startswith(("-", "<", "[")))[:2]
        if group in PENDING_COMMANDS:
            continue
        unknown.append(f"{found.file}:{found.line}: twin {found.text}")
    assert unknown == [], "\n".join(unknown)


def test_every_option_in_the_documents_belongs_to_the_command_it_follows() -> None:
    wrong = []
    for found, words in all_commands():
        options = known_options(words)
        if options is None:
            group = tuple(word for word in words if not word.startswith(("-", "<", "[")))[:2]
            options = (set(PENDING_COMMANDS.get(group, set())) | {"--help"}, " ".join(group))
        allowed, name = options
        for option in OPTION.findall(" ".join(words)):
            if option not in allowed:
                wrong.append(f"{found.file}:{found.line}: twin {name} has no {option}")
    assert wrong == [], "\n".join(wrong)


def test_the_exemption_for_pending_commands_is_only_for_commands_that_do_not_exist_yet() -> None:
    """Once ``twin eval report`` exists, the entry must go: the test then checks it for real."""
    tree = command_tree()
    for group in PENDING_COMMANDS:
        if group in tree:
            name = " ".join(group)
            pytest.fail(f"twin {name} exists now: remove it from PENDING_COMMANDS (cli_tree.py)")


def test_the_terminal_commands_are_the_ones_the_console_channel_handles() -> None:
    source = (ROOT / "src" / "twin" / "channel" / "local.py").read_text(encoding="utf-8")
    for name in TERMINAL_COMMANDS:
        assert f'"/{name}"' in source, name


def test_the_resolver_knows_the_default_command_of_import_and_global_options() -> None:
    info, rest = resolve(["import", "<目录>", "--foreground"])
    assert info is not None and info.name == "import start" and rest == ["<目录>", "--foreground"]
    info, _ = resolve(["--set", "channel.kind=console", "run"])
    assert info is not None and info.name == "run"
    info, _ = resolve(["import", "--resume"])
    assert info is not None and info.name == "import start"
    info, _ = resolve(["--version"])
    assert info is not None and "--version" in info.options
    assert resolve(["nonsense"])[0] is None and resolve(["train", "remote"])[0] is None


# ------------------------------------------------------------------ the chat commands


@pytest.fixture
async def world(
    services: Services, clock: ManualClock, embedder: HashingBackend
) -> AsyncIterator[CommandWorld]:
    start = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
    async with open_world(services, clock, start=start) as built:
        store = RatingStore(services.db, services.clock)
        built.router.register(rating_command(store, services.clock, built.rig.kit.time))
        yield built


async def test_every_chat_command_in_the_documents_is_in_the_table(world: CommandWorld) -> None:
    missing = []
    for path in DOCS:
        for number, span in slash_commands(path):
            parsed = parse_command(span)
            assert parsed is not None, span
            if parsed.name in TERMINAL_COMMANDS:
                continue
            if world.router.registry.find(parsed.name) is None:
                missing.append(f"{path.name}:{number}: {span}")
    assert missing == [], "\n".join(missing)


def readme_table() -> dict[str, tuple[str, str]]:
    """The quick reference of the README: command name -> (syntax, example)."""
    rows: dict[str, tuple[str, str]] = {}
    heading = "### 微信里的指令"
    lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(heading))
    for line in lines[start:]:
        if line.startswith("### ") and not line.startswith(heading):
            break
        if not line.startswith("| `/"):
            continue
        cells = [unescape(c.strip()) for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
        syntax = SPAN.search(cells[0])
        example = SPAN.search(cells[2])
        assert syntax and example, line
        name = parse_command(syntax.group(1))
        assert name is not None, line
        rows[name.name] = (syntax.group(1), example.group(1))
    return rows


async def test_the_readme_table_is_the_table_of_the_application(world: CommandWorld) -> None:
    rows = readme_table()
    specs = {parse_command(f"/{s.name}").name: s for s in world.router.registry.specs()}  # type: ignore[union-attr]
    assert set(rows) == set(specs), sorted(set(rows) ^ set(specs))
    for name, (syntax, example) in rows.items():
        assert syntax == specs[name].syntax, name
        assert example == specs[name].example, name


# ------------------------------------------------------------------ the appendix


KINDS = {
    "只读": CommandKind.READ,
    "轻量修改": CommandKind.LIGHT,
    "重任务": CommandKind.HEAVY,
    "独占": CommandKind.EXCLUSIVE,
}


def appendix() -> dict[str, CommandKind]:
    text = (ROOT / "docs" / "RUNBOOK.md").read_text(encoding="utf-8")
    marker = text.index("## 附录 A")
    rows: dict[str, CommandKind] = {}
    for line in text[marker:].splitlines():
        if not line.startswith("| `twin "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        command = cells[0].strip("`")
        kind = cells[1].split("（")[0].strip()
        rows[command.removeprefix("twin ")] = KINDS[kind]
    return rows


def test_the_appendix_lists_every_command_with_the_class_the_code_declares() -> None:
    declared = {}
    for name, callback in iter_commands(app):
        spec = get_spec(callback)
        assert spec is not None, name
        declared[name] = spec.kind
    listed = appendix()
    assert set(listed) == set(declared), sorted(set(listed) ^ set(declared))
    wrong = {
        name: (listed[name], declared[name]) for name in declared if listed[name] != declared[name]
    }
    assert wrong == {}


def test_the_runbook_names_every_group_of_commands() -> None:
    text = (ROOT / "docs" / "RUNBOOK.md").read_text(encoding="utf-8")
    tops = {path[0] for path in command_tree()}
    missing = sorted(top for top in tops if f"twin {top}" not in text)
    assert missing == []
