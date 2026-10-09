"""What later rounds import from the memory (round 07).

::

    from twin.memory.api import Memory, MemoryAssembler, MemoryQuery, memory_view

    memory = Memory(services)                                   # one per process and caller
    block = MemoryAssembler(memory, limits=budget.limits).build(
        MemoryQuery(text=topic), now, budget_tokens=800
    )                                                           # the live memory block
    view = memory_view(services, as_of=t)                       # the memory as known at t
    view.facts(); view.summaries(); view.followups()

``AsOfView(t)`` - the single door the training export and the evaluation sandbox read a past
moment through - is in :mod:`twin.memory.asof`.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from twin.memory.asof import AsOfSource, AsOfView, LocalMoment
from twin.memory.assemble import MemoryAssembler
from twin.memory.blocks import BlockItem, MemoryBlock, MemoryQuery
from twin.memory.followups import FollowupStore
from twin.memory.lifeline import LifelineStore, PlannedEvent
from twin.memory.memory import Memory
from twin.memory.records import LifelineRecord
from twin.memory.view import MemoryView
from twin.schedule.service import time_service_for

if TYPE_CHECKING:
    from twin.services import Services

__all__ = [
    "AsOfSource",
    "AsOfView",
    "BlockItem",
    "FollowupStore",
    "LifelineStore",
    "LocalMoment",
    "Memory",
    "MemoryAssembler",
    "MemoryBlock",
    "MemoryQuery",
    "MemoryView",
    "PlannedEvent",
    "lifeline_at",
    "memory_view",
]


def memory_view(services: Services, as_of: datetime, *, memory: Memory | None = None) -> MemoryView:
    """The memory as it was known at ``as_of`` (R-MEM-010).

    Only facts with ``known_at < as_of`` that are valid and not yet replaced then, summaries of
    local days earlier than the local day of ``as_of``, and follow-ups created before it and
    still open at it.  Before the bot's conversation began, what comes from it is empty.
    ``view.render(query)`` builds the memory block of that moment.
    """
    return MemoryAssembler(memory or Memory(services)).view(as_of)


def lifeline_at(
    services: Services, moment: datetime | None = None, *, memory: Memory | None = None
) -> LifelineRecord | None:
    """What her life line says she is doing at ``moment`` (now if omitted), or ``None``.

    The entry of the bot's local day whose time span contains the moment (R-MEM-005); replies and
    proactive messages use it so that what she says about her day agrees with the day that was
    drawn when she woke.  Pass one long-lived ``Memory`` when asking often.
    """
    store = LifelineStore(memory or Memory(services), time_service=time_service_for(services))
    return store.at(moment if moment is not None else services.clock.now_utc())
