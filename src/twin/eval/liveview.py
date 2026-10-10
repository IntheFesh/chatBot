"""The live data view for the sandbox: the present, read without writing (R-EVAL-009).

The memory test asks the bot its questions in the sandbox's ``live`` mode: the live persona card,
the whole memory, today.  The running bot's live view has one member that writes: ``her_state``
makes the day plan of today on first use (``daily_plans``, and the install salt in ``settings``),
which the sandbox must not do - the plan of a day belongs to the running bot.  :class:`EvalLiveView`
reads the plan that is already there; with none, her state is unknown and the prompt simply has no
line about it, as it has none for a fresh installation.
"""

from __future__ import annotations

from datetime import datetime

from twin.engine.dataview import LiveDataSource, LiveDataView
from twin.schedule.service import schedule_kit


class EvalLiveView(LiveDataView):
    """The live view with a ``her_state`` that only reads."""

    def her_state(self) -> str | None:
        plan = schedule_kit(self._services).planner.store.covering(self._at)
        segment = plan.segment_at(self._at) if plan is not None else None
        return str(segment.kind) if segment is not None else None


class EvalLiveSource(LiveDataSource):
    """A :class:`~twin.engine.dataview.LiveDataSource` whose views never write."""

    def view(self, at: datetime | None = None) -> LiveDataView:
        return EvalLiveView(self, at if at is not None else self.time.now_utc())
