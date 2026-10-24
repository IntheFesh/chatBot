"""The commands the program tells the person to type exist (R-NFR-006; round 18 item F8).

``test_docs_commands`` holds the documents to the real command tree; the messages of the program
are held to it here.  Every string constant of ``src/twin`` (docstrings, messages and f-strings,
whose placeholders count as an argument) is searched for ``twin <words>``:

* in a code span (```twin persona regenerate```) it must name a command of the Typer tree, and
  every ``--option`` after it must be an option of that command (or a global one);
* outside one, it is held to the same rules as soon as its first word is a command group or a
  command (``twin model register ...``); other words after "twin" are prose ("the twin is").
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from tests.support.cli_tree import (
    command_tree,
    global_options,
    group_paths,
    resolve,
)

SOURCE = Path(__file__).resolve().parents[2] / "src" / "twin"
WORD = r"(?:--?[a-z][\w-]*(?:=\S+)?|[a-z][\w-]*)"
SPAN = re.compile(r"`twin((?: +" + WORD + r")+)[^`]*`")
TEXT = re.compile(r"(?<![\w/.`-])twin((?: +" + WORD + r")+)")
PLACEHOLDER = " ARG "
COMMAND_WORD = re.compile(r"[a-z][a-z0-9-]*")


@dataclass(frozen=True)
class Hint:
    where: str
    words: tuple[str, ...]
    quoted: bool  # inside a code span


def strings_of(tree: ast.AST) -> Iterator[tuple[int, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value
        elif isinstance(node, ast.JoinedStr):
            parts = [
                part.value if isinstance(part, ast.Constant) else PLACEHOLDER
                for part in node.values
            ]
            yield node.lineno, "".join(str(part) for part in parts)


def hints_in(text: str, where: str) -> Iterator[Hint]:
    for match in SPAN.finditer(text):
        yield Hint(where, tuple(match.group(1).split()), True)
    for match in TEXT.finditer(SPAN.sub(" ", text)):
        yield Hint(where, tuple(match.group(1).split()), False)


def source_hints() -> Iterator[Hint]:
    for path in sorted(SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line, text in strings_of(tree):
            yield from hints_in(text, f"{path.relative_to(SOURCE.parent)}:{line}")


def problem(hint: Hint) -> str | None:
    """Why ``hint`` names no real command, or ``None`` when it does (or is prose)."""
    words = list(hint.words)
    info, rest = resolve(words)
    if info is None:
        groups = group_paths()
        depth = 0
        while depth < len(words) and tuple(words[: depth + 1]) in groups:
            depth += 1
        if depth == 0:
            commands = {path[0] for path in command_tree()}
            if hint.quoted or words[0] in commands:
                return f"no command `twin {' '.join(words)}`"
            return None  # prose: "the twin is ..."
        if depth < len(words) and COMMAND_WORD.fullmatch(words[depth]):
            return f"`twin {' '.join(words[:depth])}` has no command {words[depth]!r}"
        return None  # `twin persona <version>`: an argument, not a command
    for word in rest:
        if word.startswith("--"):
            option = word.split("=", 1)[0]
            if option not in info.options and option not in global_options():
                return f"`twin {info.name}` has no option {option}"
    return None


def test_every_command_the_program_names_exists() -> None:
    """R18-F8 (D-816): ``twin persona generate`` was named in three messages; the command is
    ``twin persona regenerate``."""
    found = list(source_hints())
    assert len(found) > 50  # the search finds the hints at all
    problems = [f"{hint.where}: {why}" for hint in found if (why := problem(hint))]
    assert problems == []


def test_the_check_finds_a_command_that_does_not_exist() -> None:
    """The rule itself: an unknown sub-command, an unknown option and an unknown quoted command
    are found; prose and arguments are not."""
    bad = [
        Hint("x", ("persona", "generate"), False),
        Hint("x", ("persona", "generate"), True),
        Hint("x", ("train", "export", "--nope"), True),
        Hint("x", ("nosuch",), True),
        Hint("x", ("model", "register", "--nope"), False),
    ]
    assert all(problem(hint) for hint in bad)
    good = [
        Hint("x", ("persona", "regenerate"), True),
        Hint("x", ("from", "the", "repository"), False),
        Hint("x", ("train", "export", "--from"), True),
        Hint("x", ("retrieval", "resplit"), True),
        Hint("x", ("import",), False),
    ]
    assert [problem(hint) for hint in good] == [None] * len(good)
    text = "run `twin persona generate` first, then twin model register ARG"
    assert [(h.words, h.quoted) for h in hints_in(text, "x")] == [
        (("persona", "generate"), True),
        (("model", "register"), False),
    ]
