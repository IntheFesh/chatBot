"""Alerts: categories, the cooldown, recovery, delivery, retries, no chat content (R-OPS-004)."""

from __future__ import annotations

import ast
from datetime import timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from tests.support.clock import ManualClock
from tests.support.ops import RecordingMailer, RecordingNotifier
from twin.ops.alert_delivery import MAX_MAIL_ATTEMPTS, AlertDelivery
from twin.ops.alert_text import render_alert, safe_detail
from twin.ops.alerts import (
    ALIASES,
    RECOVERIES,
    SPECS,
    AlertCategory,
    AlertService,
    AlertView,
    canonical_category,
)
from twin.services import Services

CHICAGO = ZoneInfo("America/Chicago")
CANARY = "今晚想吃火锅还是烧烤"


def views(service: AlertService) -> list[AlertView]:
    return list(reversed(service.recent(limit=50)))


def make_delivery(
    services: Services,
    clock: ManualClock,
    *,
    notifier: RecordingNotifier | None = None,
    mailer: RecordingMailer | None = None,
) -> AlertDelivery:
    return AlertDelivery(
        services.alerts,
        clock,
        notifier=notifier,
        mailer=mailer,
        recipient=lambda: "me@example.org",
        zone=lambda: CHICAGO,
    )


# ------------------------------------------------------------------- the categories


def test_there_are_exactly_the_twenty_categories_of_the_plan() -> None:
    assert {c.value for c in AlertCategory} == {
        "login_lost",
        "deepseek_failure",
        "circuit_open",
        "budget_80",
        "budget_level_n",
        "one_time_overrun",
        "style_model_down",
        "style_tokenize_mismatch",
        "backup_failed",
        "backup_mirror_unavailable",
        "disk_low",
        "queue_backlog",
        "retrain_suggested",
        "crisis_detected",
        "channel_window_unexpected",
        "channel_poll_stale",
        "calendar_out_of_range",
        "sleep_timezone_suspect",
        "process_restarted",
        "system_resumed",
    }
    assert all(c.value in SPECS for c in AlertCategory)


@pytest.mark.parametrize(
    ("raw", "detail", "expected"),
    [
        ("channel.auth_expired", None, "login_lost"),
        ("channel.poll_failing", None, "channel_poll_stale"),
        ("channel_session_expired", None, "channel_window_unexpected"),
        ("llm_auth", None, "deepseek_failure"),
        ("llm_balance", None, "deepseek_failure"),
        ("llm_circuit", None, "circuit_open"),
        ("budget", {"period": "daily", "level": None}, "budget_80"),
        ("budget", {"period": "daily", "level": 2}, "budget_level_n"),
        ("batch_overrun", None, "one_time_overrun"),
        ("style_fallback", None, "style_model_down"),
        ("crisis", None, "crisis_detected"),
        ("routine_timezone", None, "sleep_timezone_suspect"),
        ("retrain_suggested", None, "retrain_suggested"),
        ("calendar_out_of_range", None, "calendar_out_of_range"),
        ("something_new", None, "something_new"),
    ],
)
def test_the_names_of_the_earlier_rounds_are_recorded_under_the_categories(
    raw: str, detail: dict[str, Any] | None, expected: str
) -> None:
    assert canonical_category(raw, detail) == expected


def test_alerts_raised_with_the_old_names_are_stored_under_the_new(services: Services) -> None:
    services.alerts.raise_alert("channel.auth_expired", "login expired", severity="critical")
    services.alerts.raise_alert("llm_circuit", "breaker open", severity="critical")
    assert [v.category for v in views(services.alerts)] == ["login_lost", "circuit_open"]


def test_every_category_the_code_raises_is_known_to_the_alert_service() -> None:
    """A new alert name must be added to the service, not appear by accident."""
    src = Path(__file__).resolve().parents[2] / "src" / "twin"
    constants: dict[str, set[str]] = {}
    trees = {path: ast.parse(path.read_text(encoding="utf-8")) for path in src.rglob("*.py")}
    for tree in trees.values():
        for node in tree.body:
            if (
                isinstance(node, ast.Assign | ast.AnnAssign)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        constants.setdefault(target.id, set()).add(node.value.value)
    known = set(SPECS) | set(ALIASES) | set(RECOVERIES) | {c.value for c in AlertCategory}
    unknown: list[str] = []
    checked = 0
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr not in ("raise_alert", "_alert") or not node.args:
                continue
            index = (
                1 if node.func.attr == "_alert" else 0
            )  # DeepSeekClient._alert(key, category, ..)
            if len(node.args) <= index:
                continue
            first = node.args[index]
            names: set[str] = set()
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                names = {first.value}
            elif isinstance(first, ast.Name) and first.id in constants:
                names = constants[first.id]
            else:
                continue  # a computed name: covered by the callers' own tests
            checked += 1
            unknown += [
                f"{path.name}:{node.lineno} {n}"
                for n in names
                if canonical_category(n) not in known and n not in known
            ]
    assert checked > 15 and not unknown, unknown


def test_only_the_alert_service_writes_the_alerts_table() -> None:
    src = Path(__file__).resolve().parents[2] / "src" / "twin"
    allowed = {src / "ops" / "alerts.py", src / "storage" / "models.py"}
    writers = []
    for path in src.rglob("*.py"):
        if path in allowed or "migrations" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "Alert"
            ):
                writers.append(str(path.relative_to(src)))
    assert writers == []


# ------------------------------------------------------------- the cooldown, recovery


def test_a_category_is_announced_once_an_hour(services: Services, clock: ManualClock) -> None:
    alerts = services.alerts
    alerts.raise_alert("channel.poll_failing", "first", dedup_key="x")
    clock.tick(600)
    alerts.raise_alert("channel.poll_failing", "again")
    clock.tick(600)
    alerts.raise_alert("channel.poll_failing", "and again")
    first, second, third = views(alerts)
    assert (first.suppressed, first.toast_state, first.mail_state) == (False, "pending", "pending")
    assert (second.suppressed, second.toast_state, second.mail_state) == (True, "none", "none")
    assert third.suppressed
    clock.tick(3600)
    alerts.raise_alert("channel.poll_failing", "an hour later")
    assert views(alerts)[-1].suppressed is False and views(alerts)[-1].toast_state == "pending"


def test_other_categories_have_their_own_cooldown(services: Services) -> None:
    services.alerts.raise_alert("disk_low", "disk")
    services.alerts.raise_alert("queue_backlog", "queue")
    assert [v.suppressed for v in views(services.alerts)] == [False, False]


def test_a_worse_alert_is_not_held_back_by_the_cooldown(services: Services) -> None:
    alerts = services.alerts
    alerts.raise_alert("disk_low", "getting low", severity="warning")
    alerts.raise_alert("disk_low", "still low", severity="warning")
    alerts.raise_alert("disk_low", "almost full", severity="critical")
    assert [v.suppressed for v in views(alerts)] == [False, True, False]


def test_recovery_announces_once_and_clears_the_cooldown(
    services: Services, clock: ManualClock
) -> None:
    alerts = services.alerts
    alerts.raise_alert("disk_low", "low", severity="critical")
    assert alerts.recover("disk_low", "disk space is back") is True
    assert alerts.recover("disk_low", "again") is False  # nothing open any more
    rows = views(alerts)
    assert [(v.kind, v.resolved_at is not None) for v in rows] == [
        ("alert", True),
        ("recovery", False),
    ]
    assert rows[1].toast_state == "pending" and rows[1].severity == "info"
    # the next failure is a new episode: announced at once, not held back by the old one
    clock.tick(60)
    alerts.raise_alert("disk_low", "low again", severity="critical")
    assert views(alerts)[-1].suppressed is False


def test_recovery_of_something_never_announced_says_nothing(services: Services) -> None:
    assert services.alerts.recover("disk_low", "fine") is False
    services.alerts.raise_alert("job_failed", "quiet", severity="warning")  # record only
    assert services.alerts.recover("job_failed", "fine") is False
    assert all(v.kind == "alert" for v in views(services.alerts))


def test_the_recovered_notices_of_the_earlier_rounds_close_their_alerts(
    services: Services,
) -> None:
    alerts = services.alerts
    alerts.raise_alert("channel.auth_expired", "login lost", severity="critical")
    alerts.raise_alert("channel.auth_recovered", "login works again", severity="info")
    kinds = [(v.category, v.kind) for v in views(alerts)]
    assert kinds == [("login_lost", "alert"), ("login_lost", "recovery")]
    alerts.raise_alert("style_fallback", "down")
    alerts.raise_alert("style_recovered", "up", severity="info")
    assert views(alerts)[-1].category == "style_model_down"
    assert views(alerts)[-1].kind == "recovery"


def test_only_what_matters_is_announced(services: Services) -> None:
    alerts = services.alerts
    alerts.raise_alert("job_failed", "a job failed")  # recorded only
    alerts.raise_alert("task_crashed", "once", severity="warning")  # critical only
    alerts.raise_alert("task_crashed", "five times", severity="critical")
    alerts.raise_alert("dpo_ready", "200 pairs", severity="info")
    states = [(v.category, v.toast_state) for v in views(alerts)]
    assert states == [
        ("job_failed", "none"),
        ("task_crashed", "none"),
        ("task_crashed", "pending"),
        ("dpo_ready", "none"),
    ]


def test_an_unknown_severity_is_refused(services: Services) -> None:
    with pytest.raises(ValueError, match="severity"):
        services.alerts.raise_alert("disk_low", "x", severity="loud")


# ------------------------------------------------------------------------ delivery


async def test_an_alert_becomes_a_notification_and_a_mail(
    services: Services, clock: ManualClock
) -> None:
    notifier, mailer = RecordingNotifier(), RecordingMailer()
    delivery = make_delivery(services, clock, notifier=notifier, mailer=mailer)
    services.alerts.raise_alert(
        "channel.auth_expired", "WeChat login expired", severity="critical", detail={"code": -14}
    )
    assert await delivery.deliver_once() == 1
    assert len(notifier.shown) == 1 and "微信登录失效" in notifier.shown[0][0]
    (mail,) = mailer.sent
    assert mail.to == "me@example.org" and mail.subject == "[wechat-twin] 微信登录失效"
    assert mail.html is not None and "<p>" in mail.html and "微信登录失效" in mail.text
    assert "code = -14" in mail.text
    (view,) = views(services.alerts)
    assert (view.toast_state, view.mail_state) == ("sent", "sent")
    assert view.toast_at == clock.now_utc() and view.mail_at == clock.now_utc()
    assert view.notified_at == clock.now_utc()
    assert await delivery.deliver_once() == 0  # nothing is sent twice


async def test_the_recovered_notice_says_so(services: Services, clock: ManualClock) -> None:
    notifier, mailer = RecordingNotifier(), RecordingMailer()
    delivery = make_delivery(services, clock, notifier=notifier, mailer=mailer)
    services.alerts.raise_alert("disk_low", "low", severity="critical")
    services.alerts.recover("disk_low", "disk space is back")
    await delivery.deliver_once()
    assert [m.subject for m in mailer.sent] == [
        "[wechat-twin] 磁盘剩余空间不足",
        "[wechat-twin] 已恢复：磁盘剩余空间不足",
    ]
    assert "怎么办" not in mailer.sent[1].text


async def test_a_failing_mail_server_does_not_hold_back_the_notification_and_is_retried(
    services: Services, clock: ManualClock
) -> None:
    notifier, mailer = RecordingNotifier(), RecordingMailer()
    mailer.errors = ["network", "network"]
    delivery = make_delivery(services, clock, notifier=notifier, mailer=mailer)
    services.alerts.raise_alert("disk_low", "low", severity="critical")
    await delivery.deliver_once()
    assert len(notifier.shown) == 1 and mailer.attempts == 1
    (view,) = views(services.alerts)
    assert (view.toast_state, view.mail_state, view.mail_attempts) == ("sent", "pending", 1)
    assert await delivery.deliver_once() == 0  # not due yet: one minute
    clock.tick(61)
    await delivery.deliver_once()  # second failure: five minutes
    assert mailer.attempts == 2
    clock.tick(200)
    assert await delivery.deliver_once() == 0
    clock.tick(150)
    await delivery.deliver_once()
    assert mailer.attempts == 3 and len(mailer.sent) == 1
    assert views(services.alerts)[0].mail_state == "sent"
    assert len(notifier.shown) == 1  # the notification was not repeated


async def test_a_mail_is_given_up_after_eight_attempts(
    services: Services, clock: ManualClock
) -> None:
    mailer = RecordingMailer()
    mailer.errors = ["auth"] * 20
    delivery = make_delivery(services, clock, notifier=RecordingNotifier(), mailer=mailer)
    services.alerts.raise_alert("disk_low", "low", severity="critical")
    for _ in range(MAX_MAIL_ATTEMPTS + 3):
        await delivery.deliver_once()
        clock.tick(3601)
    assert mailer.attempts == MAX_MAIL_ATTEMPTS
    (view,) = views(services.alerts)
    assert view.mail_state == "failed" and view.mail_attempts == MAX_MAIL_ATTEMPTS


async def test_without_a_mail_account_the_notification_still_works(
    services: Services, clock: ManualClock
) -> None:
    notifier = RecordingNotifier()
    for mailer in (None, RecordingMailer(configured=False)):
        delivery = make_delivery(services, clock, notifier=notifier, mailer=mailer)
        services.alerts.raise_alert("queue_backlog", "queue", severity="warning")
        await delivery.deliver_once()
        clock.tick(3601)
    assert len(notifier.shown) == 2
    assert {v.mail_state for v in views(services.alerts)} == {"none"}


async def test_a_notification_that_cannot_be_shown_is_noted(
    services: Services, clock: ManualClock
) -> None:
    notifier, mailer = RecordingNotifier(), RecordingMailer()
    notifier.fail = True
    delivery = make_delivery(services, clock, notifier=notifier, mailer=mailer)
    services.alerts.raise_alert("disk_low", "low", severity="critical")
    await delivery.deliver_once()
    (view,) = views(services.alerts)
    assert (view.toast_state, view.mail_state) == ("failed", "sent")
    assert view.notified_at == view.mail_at  # the mail was the first thing that reached him


async def test_a_notice_nobody_delivered_for_a_day_is_not_sent(
    services: Services, clock: ManualClock
) -> None:
    services.alerts.raise_alert("disk_low", "low", severity="critical")
    clock.tick(25 * 3600)
    notifier = RecordingNotifier()
    delivery = make_delivery(services, clock, notifier=notifier, mailer=RecordingMailer())
    assert await delivery.deliver_once() == 0
    assert notifier.shown == []
    (view,) = views(services.alerts)
    assert (view.toast_state, view.mail_state) == ("failed", "failed")


async def test_a_second_process_does_not_send_what_the_first_holds(
    services: Services, clock: ManualClock
) -> None:
    services.alerts.raise_alert("disk_low", "low", severity="critical")
    claimed = services.alerts.claim_due()
    assert len(claimed) == 1
    assert services.alerts.claim_due() == []  # leased for two minutes
    clock.tick(121)
    assert len(services.alerts.claim_due()) == 1  # the lease ran out: it is tried again


async def test_the_component_delivers_when_woken(services: Services, clock: ManualClock) -> None:
    notifier = RecordingNotifier()
    delivery = make_delivery(services, clock, notifier=notifier, mailer=RecordingMailer())
    await delivery.start()
    try:
        services.alerts.raise_alert("disk_low", "low", severity="critical")
        from tests.support.waiting import wait_until

        await wait_until(lambda: len(notifier.shown) == 1)
    finally:
        await delivery.stop()
    assert delivery.health().status.value == "ok"


# ----------------------------------------------------------------- no chat content


def test_the_notice_is_made_of_wording_and_numbers_only(services: Services) -> None:
    detail = {
        "text": CANARY,
        "summary_text": CANARY,
        "reply": CANARY,
        "note": CANARY,  # a sentence is not a code
        "reason": "http_503",  # a code is
        "count": 3,
        "ratio": 0.5,
        "flag": True,
        "nested": {"x": CANARY},
        "list": [CANARY],
    }
    services.alerts.raise_alert("disk_low", "low", severity="critical", detail=detail)
    rendered = render_alert(views(services.alerts)[0], CHICAGO)
    everything = "\n".join(
        [rendered.subject, rendered.text, rendered.html, rendered.toast_title, rendered.toast_body]
    )
    assert CANARY not in everything and "火锅" not in everything
    assert "reason = http_503" in rendered.text and "count = 3" in rendered.text
    assert "ratio = 0.5" in rendered.text and "flag = 是" in rendered.text
    assert [key for key, _ in safe_detail(detail)] == ["reason", "count", "ratio", "flag"]


def test_the_crisis_notice_is_the_fixed_wording_without_the_title(services: Services) -> None:
    services.alerts.raise_alert(
        "crisis",
        f"he said {CANARY}",  # a title that (wrongly) held content would still not get out
        severity="critical",
        detail={"severity": "high", "keyword_hits": 2},
    )
    rendered = render_alert(views(services.alerts)[0], CHICAGO)
    assert CANARY not in rendered.text and CANARY not in rendered.html
    assert "可能需要关心" in rendered.text and "severity = high" in rendered.text


def test_a_lost_login_notice_never_carries_a_qr_code_or_a_link(services: Services) -> None:
    services.alerts.raise_alert(
        "channel.auth_expired",
        "WeChat login expired: run `twin channel login` and scan the QR code",
        severity="critical",
        detail={"source": "getupdates", "code": -14, "errmsg": "https://login.example/qr?x=1"},
    )
    rendered = render_alert(views(services.alerts)[0], CHICAGO)
    for text in (rendered.text, rendered.html, rendered.toast_body):
        assert "http" not in text.lower() and "qrcode" not in text.lower()
    assert "二维码窗口" in rendered.text and "不会通过邮件发送" in rendered.text


def test_the_time_is_written_on_the_clock_of_the_bot(
    services: Services, clock: ManualClock
) -> None:
    services.alerts.raise_alert("disk_low", "low", severity="critical")
    rendered = render_alert(views(services.alerts)[0], CHICAGO)
    assert "2026-10-09 07:00（America/Chicago）" in rendered.text  # 12:00 UTC


def test_old_alerts_are_pruned(services: Services, clock: ManualClock) -> None:
    services.alerts.raise_alert("job_failed", "old")
    clock.tick(int(timedelta(days=91).total_seconds()))
    services.alerts.raise_alert("job_failed", "new")
    assert services.alerts.prune() == 1
    assert [v.title for v in views(services.alerts)] == ["new"]
