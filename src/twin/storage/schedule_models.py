"""Tables of the schedule (round 08; R-STO-006, R-SCH-002, R-SCH-004).

``daily_plans``
    one row per plan of a local day (R-SCH-004).  The plan itself - sleep, busy periods, meals,
    the proactive quota - is sealed JSON (``plan``); the columns hold what the scheduler
    selects by: the local date and zone it was made for, the stretch of time it decides
    (``effective_from`` to ``ends_at``), the wake-up instant, the seed it was drawn with and a
    fingerprint of its inputs.  A plan that is replaced - the time zone changed, the routine was
    corrected - is kept: ``superseded_by`` names the plan that took over and ``superseded_at`` the
    moment it did, so the state at any past moment can still be read.  ``lifeline_job_id`` /
    ``lifeline_done_at`` / ``summary_queued_at`` remember which daily jobs were already queued.
``timezone_history``
    every switch of the bot's time zone (R-SCH-002): when, from where to where, by which route,
    the plan that was made for the rest of the day and when the last wake-up greeting went out
    (the 18-hour rule needs it).
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import BigInteger, CheckConstraint, Date, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from twin.storage.crypto import SealedBlob
from twin.storage.models import Base, TimestampMixin
from twin.storage.types import UTCDateTime, encrypted_column, sealed_json

DAY_TYPES = ("workday", "weekend", "holiday")
SWITCH_SOURCES = ("cli", "command", "app")


def _in_list(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class DailyPlanRow(TimestampMixin, Base):
    """One plan of one local day (R-SCH-004)."""

    __tablename__ = "daily_plans"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    local_date: Mapped[date] = mapped_column(Date, nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False)
    day_type: Mapped[str] = mapped_column(String(8), nullable=False)
    plan_ct: Mapped[SealedBlob] = encrypted_column("plan", json=True)
    plan = sealed_json()
    seed: Mapped[int] = mapped_column(BigInteger, nullable=False)
    effective_from: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    ends_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    wake_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    reason: Mapped[str] = mapped_column(String(24), nullable=False)
    inputs_hash: Mapped[str] = mapped_column(String(40), nullable=False)
    superseded_by: Mapped[str | None] = mapped_column(
        String(26), ForeignKey("daily_plans.id", ondelete="SET NULL"), nullable=True
    )
    superseded_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    lifeline_job_id: Mapped[str | None] = mapped_column(String(26), nullable=True)
    lifeline_done_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    summary_queued_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    __table_args__ = (
        CheckConstraint(_in_list("day_type", DAY_TYPES), name="day_type"),
        CheckConstraint("ends_at > effective_from", name="interval"),
        Index("ix_daily_plans_date_zone", "local_date", "timezone"),
        Index("ix_daily_plans_interval", "effective_from", "ends_at"),
    )


class TimezoneChange(TimestampMixin, Base):
    """One switch of the bot's time zone (R-SCH-002)."""

    __tablename__ = "timezone_history"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    changed_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    from_timezone: Mapped[str] = mapped_column(String(64), nullable=False)
    to_timezone: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(8), nullable=False)
    plan_id: Mapped[str | None] = mapped_column(
        String(26), ForeignKey("daily_plans.id", ondelete="SET NULL"), nullable=True
    )
    last_greeting_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    __table_args__ = (
        CheckConstraint(_in_list("source", SWITCH_SOURCES), name="source"),
        CheckConstraint("from_timezone <> to_timezone", name="differs"),
        Index("ix_timezone_history_changed_at", "changed_at"),
    )
