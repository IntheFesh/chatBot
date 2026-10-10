"""Hand-made books of the proactive scheduler: the log and the ratings of whole days (round 10).

The audit tests do not run the scheduler; they write the rows a scheduler would have left - some
correct, some not - and ask what the audit and the M3 judge make of them.  A :class:`Books` has the
real schedule kit (so the local days are the bot's), the three stores, and ``day()`` / ``rate()``
to write a day or a rating at a local time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from tests.support.clock import ManualClock
from tests.support.proactive_world import proactive_model
from tests.support.routine import Rig
from twin.schedule.proactive.store import (
    CandidateStore,
    LogEntry,
    NewLog,
    ProactiveLogStore,
    RatingStore,
)
from twin.schedule.service import KIT_KEY
from twin.services import Services


@dataclass(frozen=True)
class Sent:
    """One message that went out: the local time of day and what the log says of it."""

    hour: int
    minute: int = 0
    state: str = "free"
    kind: str = "share"
    chase: int = 0


class Books:
    """The log and the ratings of a bot in Chicago, with today at 2026-10-09 (a Friday)."""

    def __init__(self, services: Services, clock: ManualClock) -> None:
        self.services = services
        self.clock = clock
        self.rig = Rig.build(services, clock, proactive_model())
        services.extras[KIT_KEY] = self.rig.kit
        self.time = self.rig.kit.time
        self.log = ProactiveLogStore(services.db, clock)
        self.ratings = RatingStore(services.db, clock)
        self.candidates = CandidateStore(services.db, clock)
        self.today = self.time.local_date()

    def at(self, day: date, hour: int, minute: int = 0) -> datetime:
        return self.time.local_to_utc(day, hour * 60 + minute)

    def ago(self, days: int) -> date:
        return self.today - timedelta(days=days)

    def row(
        self,
        day: date,
        hour: int,
        minute: int,
        *,
        kind: str,
        outcome: str,
        reason: str | None = None,
        state: str | None = "free",
        chase: int = 0,
        low: int | None = None,
        high: int | None = None,
        enabled: bool | None = None,
    ) -> LogEntry:
        moment = self.at(day, hour, minute)
        zone = self.time.bot_timezone()
        return self.log.add(
            NewLog(
                at=moment,
                candidate_at=moment,
                local_date=day,
                local_at=f"{moment.astimezone(zone):%Y-%m-%d %H:%M}",
                timezone=zone.key,
                kind=kind,
                outcome=outcome,
                reason=reason,
                her_state=state,
                chase_seq=chase,
                bubbles_sent=1 if outcome == "sent" else 0,
                range_min=low,
                range_max=high,
                enabled=enabled,
                quota_total=high,
            )
        )

    def day(
        self,
        day: date,
        sent: tuple[Sent, ...] = (),
        *,
        low: int = 1,
        high: int = 6,
        enabled: bool = True,
        opened: bool = True,
        refused: tuple[str, ...] = (),
    ) -> None:
        """Write one day: its opening row (the scheduler ran), its messages, its refusals."""
        if opened:
            self.row(day, 0, 10, kind="day", outcome="opened", low=low, high=high, enabled=enabled)
        for message in sent:
            self.row(
                day,
                message.hour,
                message.minute,
                kind=message.kind,
                outcome="sent",
                state=message.state,
                chase=message.chase,
            )
        for number, reason in enumerate(refused):
            self.row(day, 5, number, kind="share", outcome="rejected", reason=reason)

    def good_week(self, *, last: int = 1, days: int = 7) -> None:
        """``days`` completed days ending ``last`` days ago, each with two well-spaced messages."""
        for back in range(last, last + days):
            self.day(
                self.ago(back),
                (Sent(10, 30), Sent(15, 0, chase=1), Sent(20, 45)),
            )

    def rate(self, score: int, *, days_ago: float = 0.0, note: str | None = None) -> None:
        moment = self.clock.now_utc() - timedelta(days=days_ago)
        self.ratings.add(score, note, at=moment, local_date=self.time.local_date(moment))
