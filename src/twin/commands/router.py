"""The command router: a message in, a system-voice answer out (R-CMD-001, R-CMD-003).

:class:`CommandRouter` implements the engine's :class:`~twin.engine.command_port.CommandPort`:

* a message that is **not** a command - it does not start with a slash and a name, see
  :mod:`twin.commands.parse` - returns ``None`` and goes through the normal chat path;
* a command that is in the table runs its handler; an argument the handler does not accept is
  answered with the command's syntax and example (R-CMD-003);
* a slash command that is **not** in the table is answered with a summary of the commands that
  are: it is never taken for chat (R-CMD-001).  This includes the commands of later rounds until
  they register themselves, so ``/不像`` is answered with the help summary today;
* whatever a handler raises is caught - a command never puts an error into the chat.

Every answer starts with ``⚙️ `` and is the system's voice: the engine sends it as it is and writes
its row in ``bot_turns`` with ``is_command`` (and the inbound row too - :func:`is_command_text`
tells it which messages those are), which keeps commands out of the conversation the prompt, the
memory, the learning, the retrieval and the training set see.

Later rounds add commands with :meth:`CommandRouter.register`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING

from twin.channel.base import SessionState
from twin.commands import texts
from twin.commands.builtin import CommandDeps, register_builtin, render_unknown
from twin.commands.parse import is_command_text, parse_command
from twin.commands.registry import CommandCall, CommandRegistry, CommandSpec, UsageError
from twin.commands.status import ProactiveStatus, StatusSources
from twin.engine.backend_select import BackendSelector
from twin.engine.command_port import CommandContext, CommandOutcome, CommandPort
from twin.engine.feedback import FeedbackStore
from twin.engine.turns import BotTurnStore
from twin.memory.lifeline import LifelineStore
from twin.memory.manage import MemoryManager
from twin.memory.memory import Memory
from twin.ops.logging import get_logger

if TYPE_CHECKING:
    from twin.llm.runtime import LlmRuntime
    from twin.services import Services

log = get_logger("twin.commands.router")


def with_prefix(text: str) -> str:
    """``text`` starting with the system prefix (once)."""
    return text if text.startswith(texts.PREFIX) else texts.PREFIX + text


class CommandRouter:
    """Routes command messages (see the module description)."""

    def __init__(self, registry: CommandRegistry) -> None:
        self._registry = registry

    @property
    def registry(self) -> CommandRegistry:
        return self._registry

    def register(self, spec: CommandSpec) -> CommandSpec:
        """Add a command of a later round to the table."""
        return self._registry.register(spec)

    @staticmethod
    def is_command(text: str) -> bool:
        """Does this message count as a command (the engine marks its rows ``is_command``)?"""
        return is_command_text(text)

    async def handle(self, text: str, context: CommandContext) -> CommandOutcome | None:
        parsed = parse_command(text)
        if parsed is None:
            return None
        spec = self._registry.find(parsed.name)
        if spec is None:
            log.info("command_unknown")
            return CommandOutcome(with_prefix(render_unknown(parsed.name, self._registry)))
        call = CommandCall(spec, parsed.args, context)
        try:
            result = await spec.handler(call)
        except UsageError as exc:
            reason = str(exc)
            body = texts.USAGE_WITH_REASON if reason else texts.USAGE
            reply = body.format(syntax=spec.syntax, example=spec.example, reason=reason)
            return CommandOutcome(with_prefix(reply))
        except Exception as exc:  # a command never puts an error into the chat
            log.warning("command_failed", command=spec.name, error=type(exc).__name__)
            return CommandOutcome(with_prefix(texts.FAILED))
        log.info("command_handled", command=spec.name)
        if isinstance(result, CommandOutcome):
            return CommandOutcome(with_prefix(result.reply), result.redo)
        return CommandOutcome(with_prefix(result))

    @classmethod
    def from_services(
        cls,
        services: Services,
        llm: LlmRuntime,
        *,
        selector: BackendSelector,
        turns: BotTurnStore | None = None,
        feedback: FeedbackStore | None = None,
        memory: Memory | None = None,
        session_state: Callable[[], SessionState | None] | None = None,
        proactive: Callable[[], ProactiveStatus | None] | None = None,
        retrain: Callable[[], str | None] | None = None,
        extra_status: Sequence[Callable[[], str | Awaitable[str | None] | None]] = (),
    ) -> CommandRouter:
        """The router of the running application with the commands of round 09 registered.

        ``memory`` lets ``/重来`` delete what the thrown-away reply made up; ``session_state`` is
        the channel's (``channel.session_state``) for the platform window of ``/状态``;
        ``proactive`` and ``retrain`` are the sources of later rounds (without them ``/状态``
        says so); ``extra_status`` are further lines of ``/状态`` (plain or coroutine functions).
        """
        sources = StatusSources(
            runtime=services.runtime,
            settings=services.settings,
            time=llm.time_service,
            selector=selector,
            db=services.db,
            clock=services.clock,
            ledger=llm.ledger,
            budget=llm.budget,
            session_state=session_state,
            proactive=proactive,
            retrain=retrain,
            extra=tuple(extra_status),
        )
        undo: Callable[[Sequence[str]], int] | None = None
        unshare: Callable[[Sequence[str]], int] | None = None
        if memory is not None:
            manager = MemoryManager(memory)
            lifeline = LifelineStore(memory)

            def forget(ids: Sequence[str]) -> int:
                return len(manager.forget_derived_from(ids).deleted)

            def forget_sharing(ids: Sequence[str]) -> int:
                return lifeline.unshare(ids)

            undo = forget
            unshare = forget_sharing
        deps = CommandDeps(
            sources,
            turns or BotTurnStore(services.db, services.clock),
            feedback or FeedbackStore(services.db, services.clock),
            undo,
            unshare,
        )
        registry = CommandRegistry()
        register_builtin(registry, deps)
        return cls(registry)


ENGINE_PORT: type[CommandPort] = CommandRouter
"""The router is what the engine asks for as a :class:`CommandPort` (mypy checks this line)."""
