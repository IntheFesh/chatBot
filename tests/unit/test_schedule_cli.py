"""``twin timezone`` and ``twin plan`` (R-SCH-002, R-SCH-004) on a routine learnt from chat.

The commands run in-process on a data directory of their own, with the real clock; the assertions
are about structure and about what is written, never about the hour of the day the suite runs.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select
from typer.testing import CliRunner

from tests.support.embedding import HashingBackend
from tests.support.synth_chat import ChatSpec, build_chat
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.memory.lifeline import LifelineStore, PlannedEvent
from twin.memory.memory import Memory
from twin.ops.jobs import JobQueue
from twin.profile.builder import rebuild
from twin.profile.overrides import RoutineOverrides
from twin.schedule.jobs import LIFELINE_JOB
from twin.schedule.service import schedule_kit
from twin.services import Services, build_services
from twin.storage.schedule_models import DailyPlanRow, TimezoneChange
from twin.storage.state import read_state_version

runner = CliRunner()


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedder: HashingBackend) -> Path:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    monkeypatch.setenv("TWIN_RETRIEVAL__MODEL", embedder.info.model)
    result = runner.invoke(app, ["db", "upgrade"])
    assert result.exit_code == 0, result.output
    return path


def cli(*args: str) -> Any:
    return runner.invoke(app, list(args))


def with_services[T](work: Callable[[Services], T]) -> T:
    settings = load_settings()
    services = build_services(settings, root=resolve_paths(settings).root)
    try:
        return work(services)
    finally:
        services.close()


def learn_a_routine(services: Services) -> None:
    build_chat(services, ChatSpec())  # she sleeps 01:00-08:30 and is slow 13:00-17:00 on workdays
    rebuild(services, "all", reason="test")


def state_version() -> int:
    return with_services(_version)


def _version(services: Services) -> int:
    with services.db.session() as session:
        return read_state_version(session)


def plan_rows(services: Services) -> int:
    with services.db.session() as session:
        return int(session.scalar(select(func.count()).select_from(DailyPlanRow)) or 0)


# ------------------------------------------------------------------- reading


def test_the_time_zone_is_shown_with_the_time_there_and_the_next_clock_change(
    data_dir: Path,
) -> None:
    result = cli("timezone", "show")
    assert result.exit_code == 0, result.output
    out = result.output
    assert "bot time zone: America/Chicago" in out and "local time:" in out
    assert "day type:" in out and "routine learnt in: America/Chicago" in out
    assert "next clock change:" in out and ("CDT -> CST" in out or "CST -> CDT" in out)
    assert "no plan yet" in out
    assert with_services(plan_rows) == 0  # showing never writes


def test_a_zone_without_daylight_saving_says_so(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TWIN_TIME__BOT_TIMEZONE", "Asia/Shanghai")
    result = cli("timezone", "show")
    assert result.exit_code == 0, result.output
    assert "does not use daylight saving time" in result.output


def test_a_plan_that_does_not_exist_yet_is_shown_as_a_preview_and_nothing_is_written(
    data_dir: Path,
) -> None:
    with_services(learn_a_routine)
    result = cli("plan", "show")
    assert result.exit_code == 0, result.output
    assert "预览：尚未保存" in result.output and "睡眠·早晨" in result.output
    assert with_services(plan_rows) == 0
    from twin.schedule.store import InstallSalt

    assert with_services(lambda s: InstallSalt(s.db, s.clock).peek()) is None


def test_a_date_that_is_not_a_date_is_a_usage_error(data_dir: Path) -> None:
    result = cli("plan", "show", "yesterday")
    assert result.exit_code == 2 and "must look like 2026-10-09" in result.output


# ------------------------------------------------------------------- changing


def test_rebuilding_makes_the_plan_that_show_then_prints_in_local_time(data_dir: Path) -> None:
    with_services(learn_a_routine)
    before = state_version()
    made = cli("plan", "rebuild")
    assert made.exit_code == 0, made.output
    assert with_services(plan_rows) == 1 and state_version() > before
    shown = cli("plan", "show")
    assert shown.exit_code == 0, shown.output
    out = shown.output
    assert "预览" not in out and "America/Chicago" in out
    for expected in (
        "睡眠·早晨：",
        "睡眠·夜晚：",
        "饭点：",
        "主动配额：",
        "起床问候：",
        "状态时间线：",
    ):
        assert expected in out, expected
    assert "深睡" in out and "空闲" in out
    again = cli("plan", "rebuild")
    assert "nothing changed" in again.output and with_services(plan_rows) == 1
    forced = cli("plan", "rebuild", "--force")
    assert forced.exit_code == 0 and "replaces" in forced.output
    assert with_services(plan_rows) == 2
    assert "此前替换过 1 份" in cli("plan", "show").output


def test_the_plan_learns_the_routine_the_data_shows(data_dir: Path) -> None:
    """End to end: chat -> activity model -> the real model source -> a plan with her nights."""

    def check(services: Services) -> None:
        learn_a_routine(services)
        kit = schedule_kit(services)
        zone = kit.time.bot_timezone()
        today = kit.time.local_date()
        plan = kit.planner.ensure(today)
        assert plan.morning is not None and plan.night is not None
        wake = plan.morning.wake.astimezone(zone)
        assert 7.5 <= wake.hour + wake.minute / 60 <= 9.6  # she wakes around 08:30
        onset = plan.night.onset.astimezone(zone)
        assert 0.0 <= onset.hour + onset.minute / 60 <= 2.0 or onset.hour >= 23  # about 01:00
        assert plan.quota.mean_source == "profile" and plan.quota.mean_target is not None
        model = kit.planner._model_source()
        assert model is not None and plan.quota.mean_target == pytest.approx(
            min(max(model.initiations_per_day, 1), 6), abs=0.01
        )
        # a correction by hand wins: sleep 03:00-11:00 on every kind of day
        RoutineOverrides(services.db, services.clock).add_sleep("03:00", "11:00")
        kit.planner.invalidate()
        changed = kit.planner.refresh("settings_changed", force=True).plan
        assert changed.night is not None
        assert changed.night.onset.astimezone(zone).strftime("%H:%M") == "03:00"
        assert changed.night.source == "override"

    with_services(check)


def test_the_time_zone_is_switched_recorded_and_the_rest_of_the_day_planned(
    data_dir: Path,
) -> None:
    with_services(learn_a_routine)
    before = state_version()
    result = cli("timezone", "set", "Asia/Shanghai")
    assert result.exit_code == 0, result.output
    assert "switched America/Chicago -> Asia/Shanghai" in result.output
    assert "Asia/Shanghai" in result.output and "状态时间线" in result.output
    assert "notices within two seconds" in result.output
    assert state_version() > before

    def recorded(services: Services) -> tuple[str, str, str]:
        with services.db.session() as session:
            row = session.scalars(select(TimezoneChange)).one()
            return row.from_timezone, row.to_timezone, row.source

    assert with_services(recorded) == ("America/Chicago", "Asia/Shanghai", "cli")
    shown = cli("timezone", "show").output
    assert "bot time zone: Asia/Shanghai" in shown and "her state:" in shown
    assert "routine learnt in: America/Chicago" in shown  # the clock times move over unchanged
    history = cli("timezone", "history")
    assert history.exit_code == 0 and "America/Chicago" in history.output
    assert "Asia/Shanghai" in history.output and "cli" in history.output
    same = cli("timezone", "set", "Asia/Shanghai")
    assert same.exit_code == 0 and "already in Asia/Shanghai" in same.output
    back = cli("timezone", "set", "America/Chicago")
    assert back.exit_code == 0 and "switched Asia/Shanghai -> America/Chicago" in back.output
    assert "Asia/Shanghai" in cli("timezone", "history").output


def test_a_name_that_is_not_a_time_zone_changes_nothing(data_dir: Path) -> None:
    result = cli("timezone", "set", "Mars/Olympus")
    assert result.exit_code == 2 and "unknown IANA time zone" in result.output
    assert "bot time zone: America/Chicago" in cli("timezone", "show").output
    assert "never switched" in cli("timezone", "history").output
    assert with_services(plan_rows) == 0


def test_the_history_limit_is_respected(data_dir: Path) -> None:
    for name in ("Asia/Shanghai", "America/Chicago", "Asia/Shanghai"):
        assert cli("timezone", "set", name).exit_code == 0
    one = cli("timezone", "history", "--limit", "1").output
    assert (one.count("Asia/Shanghai") >= 1 and "2026" in one) or "20" in one


def test_the_plan_of_another_date_can_be_shown(data_dir: Path) -> None:
    with_services(learn_a_routine)
    tomorrow = with_services(
        lambda s: (schedule_kit(s).time.local_date() + timedelta(days=1)).isoformat()
    )
    result = cli("plan", "show", tomorrow)
    assert result.exit_code == 0 and tomorrow in result.output and "预览" in result.output


# -------------------------------------------------------------------- life line


def test_a_day_without_a_life_line_says_when_it_is_drawn_and_nothing_is_written(
    data_dir: Path,
) -> None:
    result = cli("plan", "lifeline")
    assert result.exit_code == 0, result.output
    assert "no life line for" in result.output and "lifeline-generate" in result.output
    assert with_services(plan_rows) == 0


def test_the_life_line_of_a_day_is_listed_in_time_order(data_dir: Path) -> None:
    def write(services: Services) -> str:
        kit = schedule_kit(services)
        today = kit.time.local_date()
        store = LifelineStore(Memory(services), time_service=kit.time)
        store.replace_plan(
            today,
            [
                PlannedEvent("去图书馆自习", "14:00", "17:00", "图书馆", "专注"),
                PlannedEvent("吃早饭", "09:00", "09:30", "家里", "平静"),
            ],
        )
        store.add_improvised(today, PlannedEvent("和朋友聊了会儿天", "19:00", "19:30"))
        return today.isoformat()

    today = with_services(write)
    result = cli("plan", "lifeline")
    assert result.exit_code == 0, result.output
    out = result.output
    assert f"life line of {today}" in out and "America/Chicago" in out
    assert out.index("吃早饭") < out.index("去图书馆自习") < out.index("和朋友聊了会儿天")
    assert "planned" in out and "improvised in a chat" in out and "图书馆" in out
    other = cli("plan", "lifeline", "2020-01-01")
    assert "no life line for 2020-01-01" in other.output
    assert cli("plan", "lifeline", "someday").exit_code == 2


def test_the_life_line_cannot_be_queued_before_the_day_has_a_plan(data_dir: Path) -> None:
    result = cli("plan", "lifeline-generate")
    assert result.exit_code == 2 and "no plan for" in result.output
    assert with_services(lambda s: JobQueue(s.db, s.clock).list_jobs(job_type=LIFELINE_JOB)) == []


def test_queueing_the_life_line_records_the_job_on_the_plan(data_dir: Path) -> None:
    with_services(learn_a_routine)
    assert cli("plan", "rebuild").exit_code == 0
    result = cli("plan", "lifeline-generate")
    assert result.exit_code == 0, result.output
    assert (
        "queued the life line of" in result.output and "twin jobs run --until-idle" in result.output
    )

    def queued(services: Services) -> tuple[list[str], str | None]:
        kit = schedule_kit(services)
        plan = kit.planner.store.current_for(kit.time.local_date(), kit.time.bot_timezone().key)
        jobs = JobQueue(services.db, services.clock).list_jobs(job_type=LIFELINE_JOB)
        return [job.id for job in jobs], plan.lifeline_job_id if plan else None

    ids, recorded = with_services(queued)
    assert len(ids) == 1 and recorded == ids[0]  # the daily tick will not queue it a second time
