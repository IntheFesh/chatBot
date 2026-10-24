"""The commands of round 11, registered in the table next to those of round 09 (R-CMD-002).

``/时区 /暂停 /恢复 /主动 /作息`` (:mod:`~twin.commands.routine_commands`), ``/记住 /忘掉 /记忆``
(:mod:`~twin.commands.memory_commands`), ``/不像`` (:mod:`~twin.commands.learn_commands`), ``/费用``
(:mod:`~twin.commands.cost_commands`) and ``/导入`` (:mod:`~twin.commands.import_commands`).
``/评分`` belongs to round 10, which registers it itself.  Everything the commands say is in
:mod:`twin.commands.texts`.
"""

from __future__ import annotations

from dataclasses import dataclass

from twin.commands import texts
from twin.commands.cost_commands import CostCommands
from twin.commands.import_commands import ImportCommands
from twin.commands.learn_commands import LearnCommands
from twin.commands.memory_commands import MemoryCommands
from twin.commands.registry import CommandRegistry, CommandSpec, Handler
from twin.commands.routine_commands import RoutineCommands


@dataclass
class Round11Commands:
    """The handler objects of this round (kept so the engine and tests can reach them)."""

    routine: RoutineCommands
    memory: MemoryCommands
    learn: LearnCommands
    cost: CostCommands
    imports: ImportCommands


def register_round11(registry: CommandRegistry, commands: Round11Commands) -> None:
    """Add the commands of this round to ``registry`` (the order is the order of ``/帮助``)."""
    day = ("今天", "今日", "today", "day")
    month = ("本月", "这个月", "当月", "month")
    table: tuple[
        tuple[str, str, Handler, tuple[str, ...], dict[str, tuple[str, ...]] | None], ...
    ] = (
        ("时区", "作息与主动", commands.routine.timezone, ("timezone", "tz"), None),
        ("暂停", "作息与主动", commands.routine.pause, ("pause",), None),
        ("恢复", "作息与主动", commands.routine.resume, ("resume",), None),
        ("主动", "作息与主动", commands.routine.proactive, ("proactive",), None),
        ("作息", "作息与主动", commands.routine.routine, ("routine",), None),
        ("记住", "记忆", commands.memory.remember, ("remember",), None),
        ("忘掉", "记忆", commands.memory.forget, ("forget",), None),
        ("记忆", "记忆", commands.memory.memory, ("memory",), None),
        ("不像", "学习与评分", commands.learn.not_like, ("notlike",), None),
        ("费用", "费用与导入", commands.cost.cost, ("cost",), {"day": day, "month": month}),
        ("导入", "费用与导入", commands.imports.start, ("import",), None),
    )
    for name, group, handler, aliases, choices in table:
        registry.register(
            CommandSpec(
                name=name,
                group=group,
                summary=texts.SUMMARIES[name],
                syntax=texts.SYNTAXES[name],
                example=texts.EXAMPLES[name],
                handler=handler,
                aliases=aliases,
                choices=choices,
            )
        )
