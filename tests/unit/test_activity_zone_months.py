"""The routine's plausibility check finds a wrong zone month by month (round 18 item C7, D-902).

User decision (export text zone): a wrongly configured ``time.source_timezone(_ranges)`` is found
by the routine's plausibility check - a sleep core in the daytime - and no longer from the
export's text.  The check over the whole data cannot see a trip that is a small part of it, so
every calendar month (cut again where the configured zone changes) is checked on its own.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

from sqlalchemy import select

from tests.support.synth_chat import ChatSpec, build_chat
from twin.config.settings import TzRange
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
