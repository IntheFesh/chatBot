"""The command router: a message in, a system-voice answer out (R-CMD-001, R-CMD-003).

:class:`CommandRouter` implements the engine's :class:`~twin.engine.command_port.CommandPort`:

* a message that is **not** a command - it does not start with a slash and a name, see
  :mod:`twin.commands.parse` - returns ``None`` and goes through the normal chat path;
* a command that is in the table runs its handler; an argument the handler does not accept is
  answered with the command's syntax and example (R-CMD-003);
* a slash command that is **not** in the table is answered with a summary of the commands that
  are: it is never taken for chat (R-CMD-001).  This includes a command nobody has registered
  (yet), which is answered with the help summary;
* whatever a handler raises is caught - a command never puts an error into the chat.

Every answer starts with ``⚙️ `` and is the system's voice: the engine sends it as it is and writes
its row in ``bot_turns`` with ``is_command`` (and the inbound row too - :func:`is_command_text`
tells it which messages those are), which keeps commands out of the conversation the prompt, the
memory, the learning, the retrieval and the training set see.

The table has the commands of round 09 (``/帮助 /状态 /思考 /显示思考 /后端 /重来``) and of round 11
(:mod:`twin.commands.round11`); the ``/评分`` of round 10 and any later command are added with
:meth:`CommandRouter.register`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING

from twin.channel.base import SessionState
from twin.commands import texts
from twin.commands.builtin import CommandDeps, register_builtin, render_unknown
from twin.commands.cost_commands import CostCommands
from twin.commands.import_commands import ImportCommands
from twin.commands.import_report import ImportNotifyStore
from twin.commands.learn_commands import LearnCommands
from twin.commands.memory_commands import MemoryCommands
from twin.commands.parse import is_command_text, parse_command
from twin.commands.registry import CommandCall, CommandRegistry, CommandSpec, UsageError
from twin.commands.round11 import Round11Commands, register_round11
from twin.commands.routine_commands import KitSchedule, RoutineCommands, ScheduleControl
from twin.commands.status import ProactiveStatus, StatusSources
from twin.commands.status_extras import dpo_line, import_replay_line
from twin.engine.backend_select import BackendSelector
from twin.engine.command_port import CommandContext, CommandOutcome, CommandPort
from twin.engine.dataview import LiveDataSource
from twin.engine.feedback import FeedbackStore
from twin.engine.turns import BotTurnMessages, BotTurnStore
from twin.learning.dislike import NotLikeRecorder
from twin.learning.pairs import PreferencePairStore
from twin.learning.sample import LazyViewSource, SampleBuilder, ViewSource
from twin.memory.lifeline import LifelineStore
from twin.memory.manage import MemoryManager
from twin.memory.memory import Memory
from twin.ops.logging import get_logger
from twin.profile.overrides import RoutineOverrides
from twin.schedule.service import schedule_kit

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
        schedule: ScheduleControl | None = None,
        data: ViewSource | None = None,
        recorder: NotLikeRecorder | None = None,
        import_notes: ImportNotifyStore | None = None,
    ) -> CommandRouter:
        """The router of the running application with the commands of rounds 09 and 11.

        ``memory`` is the memory the commands work on (``/重来`` deletes what the thrown-away
        reply made up, ``/记住 /忘掉 /记忆`` manage it); ``session_state`` is the channel's
        (``channel.session_state``) for the platform window of ``/状态``; ``proactive`` and
        ``retrain`` are the sources of later rounds (without them ``/状态`` says so);
        ``extra_status`` are further lines of ``/状态`` (plain or coroutine functions).
        ``schedule`` is how ``/时区`` and ``/作息`` reach the schedule (the schedule component
        inside ``twin run``, the schedule kit otherwise), ``data`` makes the data view of a
        moment for the situation of a ``/不像`` (the live data source), ``recorder`` and
        ``import_notes`` are shared with the engine's other parts when it has them.
        """
        known_memory = memory or Memory(services)
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
            extra=(*extra_status, import_replay_line(services), dpo_line(services)),
        )
        manager = MemoryManager(known_memory)
        lifeline = LifelineStore(known_memory)

        def forget(ids: Sequence[str]) -> int:
            return len(manager.forget_derived_from(ids).deleted)

        def forget_sharing(ids: Sequence[str]) -> int:
            return lifeline.unshare(ids)

        turn_store = turns or BotTurnStore(services.db, services.clock)
        feedback_store = feedback or FeedbackStore(services.db, services.clock)
        registry = CommandRegistry()
        register_builtin(
            registry, CommandDeps(sources, turn_store, feedback_store, forget, forget_sharing)
        )
        views = data or LazyViewSource(lambda: LiveDataSource(services, memory=known_memory))
        learner = recorder or NotLikeRecorder(
            turn_store,
            feedback_store,
            PreferencePairStore(services.db, services.clock),
            SampleBuilder(services, BotTurnMessages(services.db), views),
        )
        control = schedule or KitSchedule(schedule_kit(services))
        register_round11(
            registry,
            Round11Commands(
                routine=RoutineCommands(
                    sources, control, RoutineOverrides(services.db, services.clock)
                ),
                memory=MemoryCommands(MemoryManager(known_memory, llm.client)),
                learn=LearnCommands(
                    learner, services.alerts, lambda: services.settings.training.dpo_min_pairs
                ),
                cost=CostCommands(sources),
                imports=ImportCommands(services, import_notes or ImportNotifyStore(services)),
            ),
        )
        return cls(registry)


ENGINE_PORT: type[CommandPort] = CommandRouter
"""The router is what the engine asks for as a :class:`CommandPort` (mypy checks this line)."""
