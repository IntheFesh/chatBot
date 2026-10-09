"""``twin proactive log`` and ``twin eval proactive`` on a data directory of their own."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tests.support.embedding import HashingBackend
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.eval.store import EvalStore
from twin.schedule.proactive.store import NewLog, ProactiveLogStore, RatingStore
from twin.schedule.service import schedule_kit
from twin.services import Services, build_services

runner = CliRunner()
SECRET_TEXT = "今天好困想找你聊聊"
SECRET_REASON = "他昨晚说没睡好所以问问"


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedder: HashingBackend) -> Path:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    monkeypatch.setenv("TWIN_RETRIEVAL__MODEL", embedder.info.model)
    result = runner.invoke(app, ["db", "upgrade"])
    assert result.exit_code == 0, result.output
    return path


def cli(*args: str, answers: str | None = None) -> Any:
    return runner.invoke(app, list(args), input=answers)


def with_services[T](work: Callable[[Services], T]) -> T:
    settings = load_settings()
    services = build_services(settings, root=resolve_paths(settings).root)
    try:
        return work(services)
    finally:
        services.close()


def write_day(
    services: Services,
    day: date,
    *,
    sent: tuple[tuple[int, str], ...] = ((10, "free"), (15, "free"), (20, "free")),
    opened: bool = True,
) -> None:
    time = schedule_kit(services).time
    log = ProactiveLogStore(services.db, services.clock)
    zone = time.bot_timezone()

    def add(hour: int, minute: int, *, outcome: str, kind: str, state: str | None = "free") -> None:
        moment = time.local_to_utc(day, hour * 60 + minute)
        sealed = outcome == "sent"
        log.add(
            NewLog(
                at=moment,
                candidate_at=moment,
                local_date=day,
                local_at=f"{moment.astimezone(zone):%Y-%m-%d %H:%M}",
                timezone=zone.key,
                kind=kind,
                outcome=outcome,
                her_state=state,
                bubbles_sent=1 if sealed else 0,
                range_min=1 if kind == "day" else None,
                range_max=6 if kind == "day" else None,
                enabled=True if kind == "day" else None,
                plan_reason=SECRET_REASON if sealed else None,
                content={"bubbles": [SECRET_TEXT]} if sealed else None,
                backend="deepseek" if sealed else None,
            )
        )

    if opened:
        add(0, 10, outcome="opened", kind="day")
    for hour, state in sent:
        add(hour, 30, outcome="sent", kind="share", state=state)


def write_week(services: Services, **kwargs: Any) -> None:
    today = schedule_kit(services).time.local_date()
    for back in range(1, 8):
        write_day(services, today - timedelta(days=back), **kwargs)


def rate(services: Services, score: int) -> None:
    now = services.clock.now_utc()
    RatingStore(services.db, services.clock).add(
        score, None, at=now, local_date=schedule_kit(services).time.local_date(now)
    )


# ---------------------------------------------------------------------- twin proactive log


def test_the_log_shows_the_decisions_and_no_words_by_default(data_dir: Path) -> None:
    with_services(lambda s: write_day(s, schedule_kit(s).time.local_date()))
    result = cli("proactive", "log")
    assert result.exit_code == 0, result.output
    assert "主动消息日志" in result.output and "已发" in result.output and "分享" in result.output
    assert SECRET_TEXT not in result.output and SECRET_REASON not in result.output
    assert "当日开始" not in result.output  # the opening mark is bookkeeping, not a decision


def test_the_words_are_shown_only_after_two_confirmations(data_dir: Path) -> None:
    with_services(lambda s: write_day(s, schedule_kit(s).time.local_date()))
    shown = cli("proactive", "log", "--show-text", answers="y\ny\n")
    assert shown.exit_code == 0, shown.output
    assert SECRET_TEXT in shown.output and "内容" in shown.output
    first_no = cli("proactive", "log", "--show-text", answers="n\n")
    assert first_no.exit_code != 0 and SECRET_TEXT not in first_no.output
    second_no = cli("proactive", "log", "--show-text", answers="y\nn\n")
    assert second_no.exit_code != 0 and SECRET_TEXT not in second_no.output


def test_the_log_is_limited_to_the_days_asked_for(data_dir: Path) -> None:
    def seed(services: Services) -> None:
        today = schedule_kit(services).time.local_date()
        write_day(services, today - timedelta(days=5), sent=((9, "free"),))
        write_day(services, today, sent=((11, "free"), (14, "busy")))

    with_services(seed)
    recent = cli("proactive", "log", "--days", "1")
    assert recent.exit_code == 0
    assert recent.output.count("已发") == 2
    wide = cli("proactive", "log", "--days", "7")
    assert wide.output.count("已发") == 3
    newest = cli("proactive", "log", "--days", "7", "--limit", "1")
    assert newest.output.count("已发") == 1 and "14:30" in newest.output
    assert cli("proactive", "log", "--days", "0").exit_code == 2


def test_an_empty_log_is_an_empty_table(data_dir: Path) -> None:
    result = cli("proactive", "log")
    assert result.exit_code == 0 and "主动消息日志" in result.output


# ---------------------------------------------------------------------- twin eval proactive


def test_a_compliant_week_exits_with_zero_and_is_recorded(data_dir: Path) -> None:
    with_services(write_week)
    result = cli("eval", "proactive", "--days", "7")
    assert result.exit_code == 0, result.output
    assert "审计结论：合规" in result.output and "连续观察 7/7 天" in result.output
    assert "发送时刻分布" in result.output and "10:00" in result.output
    assert "共发出 21 条；深睡时段 0 条" in result.output
    assert "这段时间没有 /评分" in result.output
    runs = with_services(lambda s: EvalStore(s.db, s.clock).list_runs("proactive_audit"))
    assert len(runs) == 1 and runs[0].verdict == "passed" and runs[0].summary["sent"] == 21
    assert runs[0].params["days"] == 7
    listed = cli("eval", "runs", "--kind", "proactive_audit")
    assert listed.exit_code == 0 and "proactive_audit" in listed.output


def test_a_message_in_deep_sleep_exits_with_one(data_dir: Path) -> None:
    def seed(services: Services) -> None:
        write_week(services)
        today = schedule_kit(services).time.local_date()
        write_day(services, today - timedelta(days=2), sent=((3, "deep_sleep"),), opened=False)

    with_services(seed)
    result = cli("eval", "proactive")
    assert result.exit_code == 1
    assert "深睡时段发了 1 条" in result.output and "审计结论：不合规或未满" in result.output
    runs = with_services(lambda s: EvalStore(s.db, s.clock).list_runs("proactive_audit"))
    assert runs[0].verdict == "failed"


def test_days_the_program_did_not_watch_are_an_observation_period(data_dir: Path) -> None:
    def seed(services: Services) -> None:
        today = schedule_kit(services).time.local_date()
        for back in range(1, 4):
            write_day(services, today - timedelta(days=back))

    with_services(seed)
    result = cli("eval", "proactive", "--days", "7")
    assert result.exit_code == 1
    assert "连续观察 3/7 天。还差 4 天。" in result.output
    runs = with_services(lambda s: EvalStore(s.db, s.clock).list_runs("proactive_audit"))
    assert runs[0].verdict == "insufficient"


def test_the_week_so_far_can_include_today(data_dir: Path) -> None:
    def seed(services: Services) -> None:
        write_week(services)
        write_day(services, schedule_kit(services).time.local_date(), sent=((9, "free"),))

    with_services(seed)
    result = cli("eval", "proactive", "--days", "2", "--today")
    assert result.exit_code == 0 and "共发出 4 条" in result.output


def test_the_ratings_of_the_week_are_reported_and_the_gate_reads_them(data_dir: Path) -> None:
    def seed(services: Services) -> None:
        write_week(services)
        rate(services, 5)
        rate(services, 4)

    with_services(seed)
    result = cli("eval", "proactive")
    assert result.exit_code == 0 and "/评分 2 次，平均 4.50 分" in result.output
    gate = cli("eval", "gate", "M3")
    assert gate.exit_code == 0, gate.output
    assert "M3：通过" in gate.output
    assert cli("eval", "gate", "M3", "--check").exit_code == 0
    runs = with_services(lambda s: EvalStore(s.db, s.clock).list_runs("gate"))
    assert runs[0].milestone == "M3" and runs[0].verdict == "passed"


def test_the_gate_asks_for_a_rating_and_for_the_missing_days(data_dir: Path) -> None:
    with_services(write_week)
    unrated = cli("eval", "gate", "M3")
    assert unrated.exit_code == 1 and "未通过（样本不足）" in unrated.output
    assert "还没有 /评分" in unrated.output
    with_services(lambda s: rate(s, 3))
    low = cli("eval", "gate", "M3")
    assert low.exit_code == 1 and "未通过" in low.output and "样本不足" not in low.output
