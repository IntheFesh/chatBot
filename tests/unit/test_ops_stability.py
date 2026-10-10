"""The stability report: availability, restarts, launches, outages and their alerts (R-EVAL-006)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from tests.support.clock import ManualClock
from tests.support.stability_world import (
    DRILL,
    INTERVAL,
    NOW,
    PROCESS,
    WEEK_START,
    channel_down,
    steady,
    store_snapshots,
    without,
)
from twin.eval.store import EvalStore
from twin.ops.alerts import AlertView
from twin.ops.stability import (
    ALERT_LATENCY_LIMIT,
    DRILL_MIN,
    GAP_FACTOR,
    UNAVAILABLE_LIMIT,
    Outage,
    Snap,
    StabilityReport,
    build_report,
    load_snapshots,
    render_lines,
    run_stability,
)
from twin.ops.supervise import Restart as SupervisorRestart
from twin.ops.supervise import RestartLog
from twin.services import Services


def alert(
    category: str,
    created: datetime,
    *,
    toast_after_s: float | None = 5,
    mail_after_s: float | None = None,
    suppressed: bool = False,
    kind: str = "alert",
) -> AlertView:
    toast = None if toast_after_s is None else created + timedelta(seconds=toast_after_s)
    mail = None if mail_after_s is None else created + timedelta(seconds=mail_after_s)
    first = min((moment for moment in (toast, mail) if moment is not None), default=None)
    return AlertView(
        id=f"a-{category}-{created:%H%M%S}",
        category=category,
        kind=kind,
        severity="warning",
        title="t",
        detail=None,
        created_at=created,
        toast_state="sent" if toast else "none",
        mail_state="sent" if mail else "none",
        toast_at=toast,
        mail_at=mail,
        mail_attempts=0,
        mail_next_at=None,
        resolved_at=None,
        suppressed=suppressed,
        notified_at=first,
    )


def report(
    snaps: list[Snap],
    alerts: list[AlertView] | None = None,
    history: list[dict[str, object]] | None = None,
    *,
    now: datetime = NOW,
    days: float = 7,
) -> StabilityReport:
    return build_report(snaps, alerts or [], history or [], now=now, days=days, interval_s=INTERVAL)


def week() -> list[Snap]:
    return steady(WEEK_START, NOW)


# ---------------------------------------------------------------------------- availability


def test_a_week_without_a_gap_is_fully_available() -> None:
    result = report(week())
    assert result.snapshots == 7 * 24 * 60 + 1 and result.covers_window
    assert result.unavailable_s == 0 and result.longest_gap_s == INTERVAL
    assert result.processes == 1 and result.restarts == () and result.outages == ()
    assert result.launches == {"task": result.snapshots} and result.drill is None
    assert result.longest_run_s == 7 * 24 * 3600 and result.missed_alerts == 0
    assert result.max_latency_s is None and result.notes == ()


def test_a_gap_is_unavailable_time_minus_one_interval() -> None:
    cut = NOW - timedelta(days=3)
    snaps = without(week(), cut, cut + timedelta(minutes=30))
    result = report(snaps)
    assert result.unavailable_s == 30 * 60 - INTERVAL  # the two snapshots around the hole
    assert result.longest_gap_s == 30 * 60


def test_a_small_gap_is_a_late_snapshot_and_not_downtime() -> None:
    tolerance = GAP_FACTOR * INTERVAL  # 150 s
    start = NOW - timedelta(hours=1)
    first = Snap(start, PROCESS, "task", True, start)
    close = replace(first, at=first.at + timedelta(seconds=tolerance - 10))  # late, not down
    far = replace(first, at=close.at + timedelta(seconds=tolerance + 10))  # a gap
    result = report([first, close, far], now=far.at, days=1)
    assert result.unavailable_s == pytest.approx(tolerance + 10 - INTERVAL)
    assert result.longest_gap_s == pytest.approx(tolerance + 10)


def test_time_since_the_last_snapshot_counts_when_it_is_a_gap() -> None:
    snaps = steady(WEEK_START, NOW - timedelta(hours=1))
    result = report(snaps)
    assert result.unavailable_s == 3600 - INTERVAL and result.longest_gap_s == 3600
    recent = report(steady(WEEK_START, NOW - timedelta(seconds=90)))
    assert recent.unavailable_s == 0  # the next snapshot is simply not due yet


def test_a_window_that_does_not_start_with_snapshots_is_not_covered() -> None:
    result = report(steady(NOW - timedelta(days=2), NOW))
    assert not result.covers_window
    just_in_time = report(steady(WEEK_START + timedelta(seconds=GAP_FACTOR * INTERVAL), NOW))
    assert just_in_time.covers_window
    too_late = report(steady(WEEK_START + timedelta(seconds=GAP_FACTOR * INTERVAL + 1), NOW))
    assert not too_late.covers_window


def test_no_snapshots_at_all_is_said_and_not_invented() -> None:
    result = report([])
    assert result.snapshots == 0 and result.first_snapshot_at is None
    assert not result.covers_window and result.processes == 0 and result.longest_run_s == 0
    assert result.unavailable_s == 0 and "did not run" in result.notes[0]
    assert render_lines(result)[1].startswith("健康快照 0 个")
    old = report(steady(NOW - timedelta(days=30), NOW - timedelta(days=20)))
    assert old.snapshots == 0  # outside the window


# --------------------------------------------------------------------- processes and launches


def test_each_new_process_is_a_restart_and_the_supervisor_says_why() -> None:
    second = PROCESS + timedelta(days=2)
    third = PROCESS + timedelta(days=5, hours=3)
    snaps = [
        replace(s, started_at=third)
        if s.at >= third
        else replace(s, started_at=second)
        if s.at >= second
        else s
        for s in week()
    ]
    history = [
        {"at": (second - timedelta(seconds=40)).isoformat(), "reason": "exit code 1"},
        {"at": (third - timedelta(hours=2)).isoformat(), "reason": "too early to be the cause"},
        {"at": (WEEK_START - timedelta(days=1)).isoformat(), "reason": "long before"},
        {"reason": "no time"},
    ]
    result = report(snaps, history=history)
    assert result.processes == 3 and [r.at for r in result.restarts] == [second, third]
    assert result.restarts[0].reason == "exit code 1"
    assert result.restarts[1].reason.startswith("unknown")
    assert result.longest_run_s < 7 * 24 * 3600
    lines = "\n".join(render_lines(result))
    assert "重启 2 次" in lines and "exit code 1" in lines


def test_launch_kinds_are_counted_per_snapshot() -> None:
    mixed = steady(WEEK_START, WEEK_START + timedelta(hours=1), launch="manual")
    mixed += steady(WEEK_START + timedelta(hours=1, minutes=1), NOW)
    result = report(mixed)
    assert result.launches["manual"] == 61 and result.launches["task"] == result.snapshots - 61
    assert "manual 61" in "\n".join(render_lines(result))


# ----------------------------------------------------------------------------- outages


def with_drill(minutes: int = 15) -> list[Snap]:
    return channel_down(
        week(), DRILL, DRILL + timedelta(minutes=minutes), last_ok=DRILL - timedelta(seconds=30)
    )


def test_an_outage_is_measured_from_the_last_good_poll_to_the_alert() -> None:
    sent = alert("channel_poll_stale", DRILL + timedelta(minutes=6), toast_after_s=20)
    result = report(with_drill(), [sent])
    (outage,) = result.outages
    assert outage.start == DRILL - timedelta(seconds=30)
    assert outage.end == DRILL + timedelta(minutes=15)  # the first snapshot that works again
    assert not outage.ongoing and outage.alert_category == "channel_poll_stale"
    assert outage.latency_s == pytest.approx(6 * 60 + 20 + 30)
    assert outage.duration_s == pytest.approx(15 * 60 + 30)
    assert result.max_latency_s == outage.latency_s and result.missed_alerts == 0
    assert result.drill == outage  # fifteen minutes is a drill


def test_an_outage_without_an_alert_is_a_missed_one() -> None:
    result = report(with_drill())
    (outage,) = result.outages
    assert outage.latency_s is None and outage.alert_at is None and outage.alert_category is None
    assert result.missed_alerts == 1 and result.max_latency_s is None
    assert "没有告警" in "\n".join(render_lines(result))


def test_an_alert_that_comes_too_late_has_a_latency_over_the_limit() -> None:
    late = alert("login_lost", DRILL + timedelta(minutes=13), toast_after_s=30)
    result = report(with_drill(20), [late])
    assert result.max_latency_s is not None
    assert result.max_latency_s > ALERT_LATENCY_LIMIT.total_seconds()
    assert result.outages[0].alert_category == "login_lost"


def test_only_announced_alerts_of_the_channel_count() -> None:
    wrong_kind = alert("channel_poll_stale", DRILL + timedelta(minutes=6), kind="recovery")
    suppressed = alert("channel_poll_stale", DRILL + timedelta(minutes=6), suppressed=True)
    other = alert("disk_low", DRILL + timedelta(minutes=6))
    elsewhere = alert("channel_poll_stale", DRILL - timedelta(hours=5))
    result = report(with_drill(), [wrong_kind, suppressed, other, elsewhere])
    assert result.outages[0].latency_s is None  # none of those announced this outage


def test_without_a_toast_the_first_delivery_counts() -> None:
    mail_only = alert(
        "channel_poll_stale", DRILL + timedelta(minutes=6), toast_after_s=None, mail_after_s=40
    )
    result = report(with_drill(), [mail_only])
    assert result.outages[0].latency_s == pytest.approx(6 * 60 + 40 + 30)
    nothing_delivered = alert(
        "channel_poll_stale", DRILL + timedelta(minutes=6), toast_after_s=None
    )
    assert report(with_drill(), [nothing_delivered]).outages[0].latency_s == pytest.approx(
        6 * 60 + 30
    )  # the moment it was raised


def test_an_outage_that_is_still_going_on_is_reported_as_such() -> None:
    snaps = channel_down(
        week(),
        NOW - timedelta(minutes=20),
        NOW + timedelta(minutes=1),
        last_ok=NOW - timedelta(minutes=21),
    )
    result = report(snaps, [alert("channel_poll_stale", NOW - timedelta(minutes=14))])
    (outage,) = result.outages
    assert outage.ongoing and outage.end == snaps[-1].at
    assert "（还在继续）" in "\n".join(render_lines(result))


def test_a_stop_of_the_application_cuts_an_outage_in_two() -> None:
    snaps = channel_down(
        week(), DRILL, DRILL + timedelta(minutes=30), last_ok=DRILL - timedelta(seconds=30)
    )
    snaps = without(snaps, DRILL + timedelta(minutes=10), DRILL + timedelta(minutes=15))
    result = report(snaps)
    assert len(result.outages) == 2  # before and after the hole
    assert result.unavailable_s == 5 * 60 - INTERVAL


def test_a_short_blip_is_an_outage_but_not_the_drill() -> None:
    blip = channel_down(
        week(), DRILL, DRILL + timedelta(minutes=3), last_ok=DRILL - timedelta(seconds=30)
    )
    result = report(blip, [alert("channel_poll_stale", DRILL + timedelta(minutes=1))])
    assert len(result.outages) == 1 and result.drill is None
    assert result.outages[0].duration_s < DRILL_MIN.total_seconds()


def test_the_longest_outage_is_the_drill() -> None:
    first = channel_down(week(), DRILL, DRILL + timedelta(minutes=12), last_ok=DRILL)
    second_at = DRILL + timedelta(days=1)
    both = channel_down(first, second_at, second_at + timedelta(minutes=25), last_ok=second_at)
    result = report(both)
    assert len(result.outages) == 2 and result.drill is not None
    assert result.drill.start == second_at


def test_system_resumes_are_counted_from_the_alerts() -> None:
    result = report(
        week(),
        [
            alert("system_resumed", NOW - timedelta(hours=5)),
            alert("system_resumed", NOW - timedelta(days=9)),
        ],
    )
    assert result.system_resumes == 1  # the one outside the window is not counted


# --------------------------------------------------------------------------- the output


def test_the_json_and_the_text_agree() -> None:
    result = report(with_drill(), [alert("channel_poll_stale", DRILL + timedelta(minutes=6))])
    data = result.to_json()
    assert data["snapshots"] == result.snapshots and data["covers_window"] is True
    assert data["missed_alerts"] == 0 and data["drill"] is not None
    assert data["max_alert_latency_s"] == pytest.approx(result.max_latency_s, abs=0.1)
    assert data["launches"] == {"task": result.snapshots}
    assert isinstance(data["outages"], list) and len(data["outages"]) == 1
    assert Outage(NOW, NOW, False, None, None, None).to_json()["latency_s"] is None
    lines = render_lines(result)
    assert lines[0].startswith("稳定性报告：") and "（7 天）" in lines[0]
    assert any(line.startswith("断网演练：有（") for line in lines)
    assert any("告警延迟" in line and "channel_poll_stale" in line for line in lines)
    plain = render_lines(report(week()))
    assert "通道中断：没有" in plain and "断网演练：还没有（twin ops drill network）" in plain
    assert timedelta(minutes=10) == UNAVAILABLE_LIMIT


# ------------------------------------------------------------------ from the database


def test_the_report_is_computed_from_the_tables_and_stored_as_a_run(
    services: Services, clock: ManualClock
) -> None:
    services.settings.ops.health.interval_s = INTERVAL
    snaps = channel_down(
        steady(NOW - timedelta(days=3), NOW),
        DRILL,
        DRILL + timedelta(minutes=14),
        last_ok=DRILL - timedelta(seconds=30),
    )
    store_snapshots(services, snaps)
    clock.set_time(DRILL + timedelta(minutes=6))
    services.alerts.raise_alert(
        "channel.poll_failing", "the long poll fails", severity="critical", dedup_key="poll"
    )
    (queued,) = services.alerts.claim_due()
    services.alerts.finish_toast(queued.id, ok=True)
    RestartLog(services.db, services.clock).add(
        SupervisorRestart(DRILL - timedelta(days=1), 1, 5.0, 5.0, "exit code 1")
    )
    clock.set_time(NOW)
    result, run = run_stability(services, 3)
    assert result.snapshots == 3 * 24 * 60 + 1 and result.covers_window
    (outage,) = result.outages
    assert outage.alert_category == "channel_poll_stale"
    assert outage.latency_s == pytest.approx(6 * 60 + 30, abs=1)
    assert run.kind == "stability" and run.status == "done" and run.params["days"] == 3
    assert run.summary == result.to_json()
    stored = EvalStore(services.db, services.clock).latest_run("stability", status="done")
    assert stored is not None and stored.id == run.id
    assert len(load_snapshots(services, NOW - timedelta(hours=1))) == 61
