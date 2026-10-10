"""The ``twin`` command tree as the command line sees it (for the documentation and scope tests).

:func:`command_tree` walks the real Typer application: every command by its full path, with the
long options it takes.  :func:`resolve` finds the command a line of documentation means, the way
the command line would: a first argument that is not a sub-command of a group that has a default
command (``twin import <directory>``) is that default command's argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

import typer.main


@dataclass(frozen=True)
class CommandInfo:
    """One leaf command of the tree."""

    path: tuple[str, ...]
    options: frozenset[str]
    flags_with_values: frozenset[str]
    help: str

    @property
    def name(self) -> str:
        return " ".join(self.path)


@cache
def _root() -> Any:
    from twin.cli import app

    return typer.main.get_command(app)


def _children(command: Any) -> dict[str, Any]:
    found = getattr(command, "commands", None)
    return dict(found) if isinstance(found, dict) else {}


def _leaf(path: tuple[str, ...], command: Any) -> CommandInfo:
    options: set[str] = set()
    with_value: set[str] = set()
    for param in command.params:
        names = [*getattr(param, "opts", ()), *getattr(param, "secondary_opts", ())]
        longs = {name for name in names if name.startswith("--")}
        options |= longs
        if longs and not getattr(param, "is_flag", False):
            with_value |= longs
    return CommandInfo(path, frozenset(options), frozenset(with_value), command.help or "")


@cache
def command_tree() -> dict[tuple[str, ...], CommandInfo]:
    """Every leaf command of ``twin`` by its path, e.g. ``("channel", "probe", "start")``."""
    found: dict[tuple[str, ...], CommandInfo] = {}

    def walk(path: tuple[str, ...], command: Any) -> None:
        children = _children(command)
        if not children:
            found[path] = _leaf(path, command)
            return
        for name, child in sorted(children.items()):
            walk((*path, name), child)

    walk((), _root())
    return found


@cache
def group_paths() -> frozenset[tuple[str, ...]]:
    """The paths of the groups (commands that hold commands), the root excluded."""
    groups: set[tuple[str, ...]] = set()

    def walk(path: tuple[str, ...], command: Any) -> None:
        children = _children(command)
        if children:
            if path:
                groups.add(path)
            for name, child in children.items():
                walk((*path, name), child)

    walk((), _root())
    return frozenset(groups)


@cache
def global_options() -> frozenset[str]:
    """The long options of ``twin`` itself (``--config``, ``--set``, ...)."""
    root = _root()
    return frozenset(
        name
        for param in root.params
        for name in getattr(param, "opts", ())
        if name.startswith("--")
    )


def ROOT_INFO() -> CommandInfo:
    """``twin`` itself: no command, only its global options."""
    return CommandInfo((), global_options(), frozenset(), "")


# Commands that ``prompts/15-evaluation.md`` describes and that the documents already mention while
# round 15 is developed in parallel.  A document may name them as long as they do not exist; the
# moment one does, the entry must go (``test_docs_commands`` fails until it does) and the command is
# checked like every other.  The options are the ones the prompt names.
PENDING_COMMANDS: dict[tuple[str, ...], frozenset[str]] = {}

GLOBAL_WITH_VALUE = frozenset({"--config", "-c", "--set", "--log-level"})


def resolve(tokens: list[str]) -> tuple[CommandInfo | None, list[str]]:
    """The command the words after ``twin`` name, and the words left over (its arguments).

    ``None`` when the first word that should be a command is not one.  A group with a default
    command (``import``) treats an unknown word as that command's first argument.
    """
    tree = command_tree()
    groups = group_paths()
    path: tuple[str, ...] = ()
    index = 0
    seen_options = False
    while index < len(tokens):
        token = tokens[index]
        if token in GLOBAL_WITH_VALUE:
            index += 2  # a global option before the command takes the next word as its value
            seen_options = True
            continue
        if token.startswith("-"):
            if path == ("import",) and token != "--help" and (*path, "start") in tree:
                return tree[(*path, "start")], tokens[index:]  # `twin import --resume`
            index += 1
            seen_options = True
            continue
        candidate = (*path, token)
        if candidate in tree:
            return tree[candidate], tokens[index + 1 :]
        if candidate in groups:
            path = candidate
            index += 1
            continue
        default = (*path, "start")
        if path == ("import",) and default in tree:
            return tree[default], tokens[index:]
        return None, tokens[index:]
    if path == () and seen_options and all(token.startswith("-") for token in tokens[::2]):
        return ROOT_INFO(), []  # `twin --version`, `twin --help`
    return None, []
