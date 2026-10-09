"""The commands of round 09: ``/帮助``, ``/状态``, ``/思考``, ``/显示思考``, ``/后端``, ``/重来``.

The handlers live in :class:`BuiltinCommands`; :func:`register_builtin` puts them into a
:class:`~twin.commands.registry.CommandRegistry` with the texts of :mod:`twin.commands.texts`.
Rounds 10 and 11 register their commands in the same registry; ``/帮助`` lists whatever is
registered, so a command that is not implemented is not listed.

``/重来`` throws her last reply away: the rows of the reply are marked ``rejected`` (so they are no
longer part of the conversation the prompt, the memory and the sticker share see), the user's
verdict goes to the ``feedback`` table as ``redo`` (a negative example for round 11's learning),
and what the thrown-away reply made up - facts the bot invented, the follow-ups and life line
entries made from them - is deleted.  Writing the new reply is the engine's job: the outcome
says ``redo=True``.  The command refuses when there is nothing to redo: no reply yet, or the last
reply was already thrown away and the next one has not been written.  The one reply it never
throws away is the answer given out of the role to a crisis (R-SAFE-001): that is not "redone".
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

from twin.commands import texts
from twin.commands.parse import fold
from twin.commands.registry import CommandCall, CommandRegistry, CommandSpec
from twin.commands.status import StatusReport, StatusSources
from twin.config.runtime import (
    BACKEND_ACTIVE,
    SHOW_THINKING,
    THINKING_CHAT,
    BackendName,
    ThinkingMode,
)
from twin.engine.backend_select import STYLE_BACKENDS, SwitchCheck
from twin.engine.command_port import CommandOutcome
from twin.engine.feedback import FeedbackStore
from twin.engine.turns import BotTurnStore
from twin.memory.lifeline import shared_ids_of
from twin.ops.logging import get_logger

log = get_logger("twin.commands.builtin")

SAFETY_BACKEND = "safety"  # the backend label of the replies given out of the role (R-SAFE-001)


@dataclass
class CommandDeps:
    """What the commands of this round work with."""

    sources: StatusSources
    turns: BotTurnStore
    feedback: FeedbackStore
    undo: Callable[[Sequence[str]], int] | None = None
    """Deletes what the bot made up in the given rows of its conversation; returns how many."""
    unshare: Callable[[Sequence[str]], int] | None = None
    """Takes the "already told" mark off the given life line entries; returns how many."""


def render_help(registry: CommandRegistry) -> str:
    """Every registered command, by group, one line each."""
    lines = [texts.HELP_HEADER]
    for group, specs in registry.groups():
        lines.append(texts.HELP_GROUP.format(group=group))
        lines.extend(
            texts.HELP_LINE.format(syntax=spec.syntax, summary=spec.summary) for spec in specs
        )
    return "\n".join(lines)


def render_unknown(name: str, registry: CommandRegistry) -> str:
    """The answer to a slash command that is not in the table: a summary of what is."""
    names = " ".join(f"/{spec.name}" for spec in registry.specs())
    return f"{texts.UNKNOWN.format(name=name)}\n{texts.UNKNOWN_HINT.format(names=names)}"


class BuiltinCommands:
    """The handlers of the commands of this round."""

    def __init__(self, deps: CommandDeps, registry: CommandRegistry) -> None:
        self._deps = deps
        self._registry = registry

    # ---------------------------------------------------------------------- /帮助

    async def help(self, call: CommandCall) -> str:
        if not call.args:
            return render_help(self._registry)
        spec = self._registry.find(fold(call.args.lstrip("/／")))
        if spec is None:
            return render_unknown(call.args.lstrip("/／"), self._registry)
        lines = [
            texts.HELP_ONE.format(syntax=spec.syntax, summary=spec.summary, example=spec.example)
        ]
        if spec.aliases:
            lines.append(
                texts.HELP_ALIASES.format(aliases="、".join(f"/{a}" for a in spec.aliases))
            )
        return "\n".join(lines)

    # ---------------------------------------------------------------------- /状态

    async def status(self, call: CommandCall) -> str:
        return await StatusReport(self._deps.sources).render()

    # ---------------------------------------------------------------------- /思考

    async def think(self, call: CommandCall) -> str:
        mode = call.spec.choose(call.args)
        runtime = self._deps.sources.runtime
        await asyncio.to_thread(runtime.set, THINKING_CHAT, cast(ThinkingMode, mode), by="command")
        lines = [texts.THINK_SET.format(mode=texts.THINK_MODES[mode])]
        if mode == "auto":
            lines.append(texts.THINK_AUTO_HINT)
        budget = self._deps.sources.budget
        if mode != "off" and budget is not None and not budget.limits().chat_thinking_allowed:
            lines.append(texts.THINK_BUDGET)
        return "".join(lines)

    # ------------------------------------------------------------------- /显示思考

    async def show_thinking(self, call: CommandCall) -> str:
        value = call.spec.choose(call.args) == "on"
        await asyncio.to_thread(self._deps.sources.runtime.set, SHOW_THINKING, value, by="command")
        return texts.SHOW_SET_ON if value else texts.SHOW_SET_OFF

    # ---------------------------------------------------------------------- /后端

    async def backend(self, call: CommandCall) -> str:
        name = call.spec.choose(call.args)
        selector = self._deps.sources.selector
        refusal = await selector.verify_switch(name)
        if refusal is not None:
            return texts.BACKEND_REFUSED.format(reason=_refusal_text(refusal))
        await asyncio.to_thread(
            self._deps.sources.runtime.set, BACKEND_ACTIVE, cast(BackendName, name), by="command"
        )
        await asyncio.to_thread(selector.note_user_choice)
        reply = texts.BACKEND_SET.format(name=name)
        if name in STYLE_BACKENDS:
            reply += texts.BACKEND_STYLE_HINT
        return reply

    # ---------------------------------------------------------------------- /重来

    async def redo(self, call: CommandCall) -> CommandOutcome:
        deps = self._deps
        reply = await asyncio.to_thread(deps.turns.latest_reply)
        newest_message = await asyncio.to_thread(deps.turns.last_message_at, "in")
        if not reply or (newest_message is not None and newest_message > reply[-1].at):
            return CommandOutcome(texts.REDO_NOTHING)
        reply_id = reply[0].reply_id
        if reply_id is None:
            return CommandOutcome(texts.REDO_NOTHING)
        if reply[0].backend == SAFETY_BACKEND:  # what she said out of the role stays said
            return CommandOutcome(texts.REDO_SAFETY)
        await asyncio.to_thread(deps.turns.reject_reply, reply_id, call.context.at)
        await asyncio.to_thread(deps.feedback.add, "redo", reply_id, bot_turn_id=reply[0].id)
        undone = 0
        if deps.undo is not None:
            undone = await asyncio.to_thread(deps.undo, [row.id for row in reply])
        shared = shared_ids_of(reply[0].actions)
        unshared = 0
        if shared and deps.unshare is not None:  # a proactive message told her day (round 10)
            unshared = await asyncio.to_thread(deps.unshare, shared)
        text = texts.REDO_DONE
        if undone:
            text += texts.REDO_UNDONE.format(count=undone)
        if unshared:
            text += texts.REDO_UNSHARED.format(count=unshared)
        return CommandOutcome(text, redo=True)


def _refusal_text(refusal: SwitchCheck) -> str:
    template = texts.BACKEND_REASONS.get(refusal.code, texts.BACKEND_REASONS["unhealthy"])
    return template.format(detail=refusal.detail)


def register_builtin(registry: CommandRegistry, deps: CommandDeps) -> BuiltinCommands:
    """Add the commands of this round to ``registry``."""
    commands = BuiltinCommands(deps, registry)
    table = (
        ("帮助", "基本", commands.help, ("help",), None),
        ("状态", "基本", commands.status, ("status",), None),
        (
            "思考",
            "对话与生成",
            commands.think,
            ("think", "thinking"),
            {
                "on": ("开", "打开", "开启", "on", "1"),
                "off": ("关", "关闭", "off", "0"),
                "auto": ("自动", "auto"),
            },
        ),
        (
            "显示思考",
            "对话与生成",
            commands.show_thinking,
            ("showthinking", "show_thinking"),
            {"on": ("开", "打开", "开启", "on", "1"), "off": ("关", "关闭", "off", "0")},
        ),
        (
            "后端",
            "对话与生成",
            commands.backend,
            ("backend",),
            {
                "deepseek": ("deepseek", "ds"),
                "style": ("style", "风格", "风格模型"),
                "hybrid": ("hybrid", "混合"),
            },
        ),
        ("重来", "对话与生成", commands.redo, ("redo",), None),
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
    return commands


__all__ = ["BuiltinCommands", "CommandDeps", "register_builtin", "render_help", "render_unknown"]
