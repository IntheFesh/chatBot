"""A router with everything the commands of round 11 work with, on the real tables.

:class:`CommandWorld` puts together what ``build_engine`` wires for the running bot, minus the
channel: the migrated database, the manual clock, the schedule kit of ``tests/support/routine``
(a fixed routine, so a plan exists), a memory on the hashing embedder, the scripted style client
and the static data view.  ``say`` sends a message through ``CommandRouter.handle`` the way the
engine does; ``chat`` writes a turn of the conversation into ``bot_turns`` without the engine.
Nothing here is imported by ``src``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

from tests.support.clock import ManualClock
from tests.support.engine_harness import ClockData
from tests.support.memory import make_memory
from tests.support.routine import Rig, fixed_model
from tests.support.style_models import ScriptedStyleClient
from twin.commands.router import CommandRouter
from twin.commands.routine_commands import KitSchedule
from twin.engine.backend_select import BackendSelector
from twin.engine.command_port import CommandContext, CommandOutcome
from twin.engine.feedback import FeedbackStore
from twin.engine.style_models import StyleModels
from twin.engine.turns import BotTurnStore, OutboundBubble, ReplyMeta
from twin.llm.runtime import LlmRuntime, build_llm_runtime
from twin.memory.memory import Memory
from twin.profile.activity_model import ActivityModel
from twin.profile.overrides import RoutineOverrides
from twin.schedule.plan_builder import QuotaRange
from twin.schedule.service import KIT_KEY, build_kit
from twin.services import Services

PREFIX = "⚙️ "


class CommandWorld:
    """The router of the commands and the parts a test looks into."""

    def __init__(self, services: Services, llm: LlmRuntime, clock: ManualClock, rig: Rig) -> None:
        services.runtime.initialize()
        self.services, self.llm, self.clock, self.rig = services, llm, clock, rig
        self.client = ScriptedStyleClient()
        self.selector = BackendSelector(
            runtime=services.runtime,
            models=StyleModels(services.db),
            client=self.client,
            config=services.settings.backend,
            clock=clock,
            alerts=services.alerts,
            limits=llm.budget.limits,
        )
        self.turns = BotTurnStore(services.db, clock)
        self.feedback = FeedbackStore(services.db, clock)
        self.memory: Memory = make_memory(services)
        self.views = ClockData(clock)
        self.router = CommandRouter.from_services(
            services,
            llm,
            selector=self.selector,
            turns=self.turns,
            feedback=self.feedback,
            memory=self.memory,
            schedule=KitSchedule(rig.kit),
            data=self.views,
        )
        self._number = 0

    async def say(self, text: str) -> CommandOutcome:
        outcome = await self.router.handle(text, CommandContext(self.clock.now_utc(), "in-1"))
        assert outcome is not None, f"{text!r} was not taken for a command"
        return outcome

    async def reply(self, text: str) -> str:
        """The answer to ``text`` with the system prefix taken off."""
        answer = (await self.say(text)).reply
        assert answer.startswith(PREFIX), answer
        return answer.removeprefix(PREFIX)

    def chat(
        self,
        user: str,
        bot: Sequence[str],
        *,
        at: datetime | None = None,
        backend: str = "deepseek",
        sticker: str | None = None,
    ) -> str:
        """One turn of the conversation: the user's message, then the bot's bubbles.

        Returns the id of the reply.  The clock does not move; the user's message is one second
        before the first bubble.
        """
        moment = at or self.clock.now_utc()
        self._number += 1
        self.turns.add_inbound(
            at=moment - timedelta(seconds=1),
            kind="text",
            text=user,
            external_id=f"in-{self._number}-{moment.timestamp()}",
        )
        bubbles = [
            OutboundBubble(text, moment + timedelta(seconds=index))
            for index, text in enumerate(bot)
        ]
        if sticker is not None:
            bubbles.append(
                OutboundBubble(sticker, moment + timedelta(seconds=len(bot)), "sticker", "a" * 32)
            )
        records = self.turns.add_reply(bubbles, ReplyMeta(backend))
        assert records[0].reply_id is not None
        return records[0].reply_id


@asynccontextmanager
async def open_world(
    services: Services, clock: ManualClock, *, start: datetime | None = None
) -> AsyncIterator[CommandWorld]:
    """A world on the services container, at ``start`` (default: Friday 07:00 in Chicago).

    The schedule rig is made first and installed as the container's schedule kit, so the runtime,
    the router and the commands all read one plan and one time zone.
    """
    clock.set_time(start or datetime(2026, 10, 9, 12, 0, tzinfo=UTC))
    holder: dict[str, ActivityModel | None] = {"model": fixed_model()}
    corrections = RoutineOverrides(services.db, clock)

    def corrected_model() -> ActivityModel | None:
        """The routine with the manual corrections applied, as ``load_activity_model`` does."""
        model = holder["model"]
        return model.with_overrides(corrections.entries()) if model is not None else None

    # no quota source of its own: the plan reads the proactive range from the runtime settings
    kit = build_kit(services, model_source=corrected_model)
    rig = Rig(services, clock, kit, holder, {"quota": QuotaRange(1, 6)})
    services.extras[KIT_KEY] = kit
    llm = build_llm_runtime(services)
    try:
        yield CommandWorld(services, llm, clock, rig)
    finally:
        await llm.client.aclose()
