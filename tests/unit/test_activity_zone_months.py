"""The routine's plausibility check finds a wrong zone month by month (round 18 item C7, D-902).

User decision (export text zone): a wrongly configured ``time.source_timezone(_ranges)`` is found
by the routine's plausibility check - a sleep core in the daytime - and no longer from the
export's text.  The check over the whole data cannot see a trip that is a small part of it, so
every calendar month (cut again where the configured zone changes) is checked on its own.
"""

from __future__ import annotations

from array import array
from datetime import UTC, date, datetime, time, timedelta

from sqlalchemy import select

from tests.support.synth_chat import ChatSpec, build_chat
from twin.config.settings import ActivityConfig, TzRange
from twin.profile.activity_collect import ActivityRaw
from twin.profile.activity_infer import MonthVerdict, month_verdicts, month_warnings
from twin.profile.activity_model import ActivityModel
from twin.profile.api import load_activity_model
from twin.profile.builder import rebuild
from twin.profile.circular import signed_diff
from twin.profile.jobs import TIMEZONE_ALERT, run_rebuild
from twin.profile.overrides import OverrideView
from twin.services import Services
from twin.storage.models import Alert

START = date(2026, 5, 4)  # a Monday; 120 days run to 2026-08-31
AUGUST = {date(2026, 8, 1) + timedelta(days=n): "Asia/Shanghai" for n in range(31)}


def her_range(first: date, last: date, zone: str = "Asia/Shanghai") -> TzRange:
    return TzRange.model_validate({"from": first, "to": last, "tz": zone})


def learn(services: Services, **spec: object) -> ActivityModel:
    build_chat(services, ChatSpec(**spec))  # type: ignore[arg-type]
    rebuild(services, "live", reason="test", force=True)
    model = load_activity_model(services, "live", apply_overrides=False)
    assert model is not None
    return model


def relearn(services: Services) -> ActivityModel:
    rebuild(services, "live", reason="test", force=True)
    model = load_activity_model(services, "live", apply_overrides=False)
    assert model is not None
    return model


def zone_warnings(model: ActivityModel) -> list[str]:
    return [w for w in model.sleep_profile().warnings if "source_timezone" in w]


def within(minute: float, hour: float, tolerance: float = 45.0) -> bool:
    return abs(signed_diff(minute, hour * 60)) <= tolerance


def test_a_month_away_without_a_range_raises_the_plausibility_warning(services: Services) -> None:
    """R18-C7: she spent August in Shanghai and no range says so.  Over all 120 days she still
    sleeps at night on the Chicago clock, so the overall check is quiet; August's own low
    stretch lies in the daytime and the warning names that month (before: no warning at all)."""
    model = learn(services, start=START, days=120, zone_by_day=AUGUST)
    found = model.sleep_profile().windows["all"]
    assert within(found.onset_min, 1.0) and within(found.wake_min, 8.5)
    warnings = zone_warnings(model)
    assert len(warnings) == 1, warnings
    text = warnings[0]
    assert "2026-08" in text and "time.source_timezone_ranges" in text
    assert "America/Chicago" in text and "10:00–18:00" in text
    for other in ("2026-05", "2026-06", "2026-07"):
        assert other not in text
    # with the range the month is read on the Shanghai clock and nothing is left to say
    services.settings.time.source_timezone_ranges = [her_range(date(2026, 8, 1), date(2026, 8, 31))]
    assert zone_warnings(relearn(services)) == []


def test_a_range_on_the_wrong_month_names_both_months(services: Services) -> None:
    """R18-C7: the range says September, she was away in August: August (read in Chicago) and
    September (read in Shanghai while she was home) both sleep in the daytime."""
    days = 151  # to 2026-10-01
    build_chat(services, ChatSpec(start=START, days=days, zone_by_day=AUGUST))
    services.settings.time.source_timezone_ranges = [her_range(date(2026, 9, 1), date(2026, 9, 30))]
    warnings = zone_warnings(relearn(services))
    joined = "\n".join(warnings)
    assert "2026-08" in joined and "America/Chicago" in joined
    assert "2026-09" in joined and "Asia/Shanghai" in joined
    assert "2026-07" not in joined


def test_the_warning_raises_the_routine_alert(services: Services) -> None:
    """R18-C7: the month's warning goes out through the existing ``routine_timezone`` alert."""
    build_chat(services, ChatSpec(start=START, days=120, zone_by_day=AUGUST))
    run_rebuild(services, "live", "manual", True)
    with services.db.session() as session:
        alerts = session.scalars(select(Alert)).all()
        assert [a.dedup_key for a in alerts] == [f"{TIMEZONE_ALERT}-live"]
        detail = alerts[0].detail
    assert detail is not None and any("2026-08" in w for w in detail["warnings"])


def test_a_sleep_set_by_hand_does_not_hide_a_month_in_the_wrong_zone(services: Services) -> None:
    """R18-C7: a manual sleep replaces the inferred one (and its daytime warning), but the month
    read in the wrong zone is about the configuration, so its warning stays."""
    model = learn(services, start=START, days=120, zone_by_day=AUGUST)
    override = OverrideView(
        id="o1",
        kind="sleep",
        params={"start": "01:00", "end": "08:30", "day_types": None},
        enabled=True,
        note=None,
        created_at=datetime(2026, 10, 9, tzinfo=UTC),
    )
    fixed = model.with_overrides([override])
    assert [w for w in fixed.sleep_profile().warnings if "2026-08" in w]
    assert fixed.typical_state(time(3, 0), "workday") == "deep_sleep"


def test_a_whole_record_in_the_wrong_zone_is_said_once(services: Services) -> None:
    """R18-C7: every month is off - the overall warning says it; the months add nothing."""
    model = learn(services, start=START, days=90, zone="Asia/Shanghai")
    warnings = zone_warnings(model)
    assert warnings and all("可能是 time.source_timezone 设置不对" in w for w in warnings)
    assert not [w for w in warnings if "source_timezone_ranges" in w]


def test_short_or_thin_months_are_not_judged(services: Services) -> None:
    """R18-C7: a month with too few days of her messages is too thin for a curve of its own."""
    trip = {date(2026, 8, 1) + timedelta(days=n): "Asia/Shanghai" for n in range(5)}
    model = learn(services, start=date(2026, 6, 1), days=66, zone_by_day=trip)  # Aug 1-5 only
    assert zone_warnings(model) == []


# ------------------------------------------------------------------- the pieces


def raw_of(walls: list[float], zones: list[int], names: list[str]) -> ActivityRaw:
    """An :class:`ActivityRaw` that holds only the clock positions of her messages."""
    return ActivityRaw(
        zone_names=names,
        day_types={},
        counts={},
        inits={},
        latency={},
        latency_workday={},
        her_wall=array("d", walls),
        her_zone=array("b", zones),
        workday_day_counts={},
        her_messages=len(walls),
        initiations=0,
        first_day=None,
        last_day=None,
    )


def test_a_month_without_a_clear_low_stretch_is_not_judged() -> None:
    """R18-C7: a month whose messages are spread evenly over the clock has no valley to judge."""
    first = date(2026, 8, 1).toordinal()
    walls = [(first + day) * 1440.0 + minute for day in range(20) for minute in range(0, 1440, 60)]
    raw = raw_of(walls, [0] * len(walls), ["America/Chicago"])
    assert month_verdicts(raw, ActivityConfig()) == []


def verdict(month: int, daytime: bool, zone: str = "America/Chicago") -> MonthVerdict:
    return MonthVerdict(2026, month, zone, (12 * 60.0, 19 * 60.0), daytime)


def test_consecutive_months_in_one_zone_make_one_warning() -> None:
    """R18-C7: months in a row are one warning; at most three cores are spelled out."""
    months = [verdict(5, False), *(verdict(m, True) for m in (6, 7, 8, 9)), verdict(11, True)]
    warnings = month_warnings(months, whole_daytime=False)
    assert len(warnings) == 2
    assert warnings[0].startswith("2026-06 至 2026-09 按 America/Chicago 读")
    assert "2026-08 12:00–19:00）" not in warnings[0] and "等）" in warnings[0]
    assert warnings[1].startswith("2026-11 按 America/Chicago 读")
    # the same months in another zone start a warning of their own
    split = month_warnings(
        [verdict(6, True), verdict(7, True, "Asia/Shanghai"), verdict(8, False)], False
    )
    assert len(split) == 2 and "Asia/Shanghai" in split[1]
    # every judged month off and the overall warning said so: nothing more
    assert month_warnings([verdict(6, True), verdict(7, True)], whole_daytime=True) == []
    assert month_warnings([verdict(6, False)], whole_daytime=False) == []
