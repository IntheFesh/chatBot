"""The command table: what a command is, and how later rounds add theirs (R-CMD-002).

Every command is one :class:`CommandSpec` in a :class:`CommandRegistry`.  The spec holds everything
the router and ``/帮助`` need to know - names, group, one-line description, syntax, example, the
accepted choices - and the coroutine that does the work, so the table is the one place to look at
and to test.  Rounds 10 and 11 add their commands by registering specs::

    registry.register(
        CommandSpec(
            name="暂停",
            group="作息与主动",
            summary="暂停回复与主动",
            syntax="/暂停 <时长>",
            example="/暂停 2小时",
            handler=pause_handler,
        )
    )

A handler is ``async def handler(call: CommandCall) -> str | CommandOutcome``.  It returns the text
of its answer (the router adds the ``⚙️ `` prefix) or a :class:`CommandOutcome`; it raises
:class:`UsageError` when the argument is not one it accepts - the router then answers with the
syntax and the example of the spec.  Anything else it raises is caught by the router, logged by
type, and answered with a short failure message: a command never produces a stack trace in the chat.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from twin.commands.parse import fold
from twin.engine.command_port import CommandContext, CommandOutcome

GROUP_ORDER = ("基本", "对话与生成", "作息与主动", "记忆", "学习与评分", "费用与导入")


class UsageError(ValueError):
    """The argument of a command is not valid; the router answers with the command's usage."""


@dataclass(frozen=True)
class CommandCall:
    """One invocation: the spec that matched, the argument text and the message's context."""

    spec: CommandSpec
    args: str
    context: CommandContext

    @property
    def name(self) -> str:
        return self.spec.name


Handler = Callable[[CommandCall], Awaitable[str | CommandOutcome]]


@dataclass(frozen=True)
class CommandSpec:
    """A command of the table (see the module description).

    ``choices`` maps each accepted value of a command with a fixed set of arguments to the
    spellings that mean it; spellings are compared folded (case and width do not matter), the
    value itself is a spelling too.
    """

    name: str
    group: str
    summary: str
    syntax: str
    example: str
    handler: Handler = field(compare=False)
    aliases: tuple[str, ...] = ()
    choices: Mapping[str, tuple[str, ...]] | None = None

    def choose(self, args: str) -> str:
        """The value ``args`` names; :class:`UsageError` if it is none of the choices."""
        if not self.choices:
            raise UsageError(f"{self.name} takes no fixed choices")
        wanted = fold(args)
        for value, spellings in self.choices.items():
            if wanted == fold(value) or wanted in {fold(spelling) for spelling in spellings}:
                return value
        raise UsageError(f"{args!r} is not one of {'|'.join(self.choices)}")


class CommandRegistry:
    """The commands by name, in the order they were registered."""

    def __init__(self) -> None:
        self._specs: list[CommandSpec] = []
        self._by_name: dict[str, CommandSpec] = {}

    def register(self, spec: CommandSpec) -> CommandSpec:
        """Add a command; a name or alias that is taken is an error."""
        names = [fold(spec.name), *(fold(alias) for alias in spec.aliases)]
        for name in names:
            if name in self._by_name:
                raise ValueError(f"the command name {name!r} is already registered")
        self._specs.append(spec)
        for name in names:
            self._by_name[name] = spec
        return spec

    def find(self, folded_name: str) -> CommandSpec | None:
        return self._by_name.get(folded_name)

    def specs(self) -> tuple[CommandSpec, ...]:
        return tuple(self._specs)

    def groups(self) -> list[tuple[str, list[CommandSpec]]]:
        """The commands by group: the known groups in their order, then any other."""
        grouped: dict[str, list[CommandSpec]] = {}
        for spec in self._specs:
            grouped.setdefault(spec.group, []).append(spec)
        known = [name for name in GROUP_ORDER if name in grouped]
        other = [name for name in grouped if name not in GROUP_ORDER]
        return [(name, grouped[name]) for name in (*known, *other)]
