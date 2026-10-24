"""``twin profile`` and ``twin routine`` (R-PROF-004, R-ACT-005, R-ACT-006, R-ARCH-006)."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tests.support.synth_chat import ChatSpec, build_chat
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.ops.instance_lock import LOCK_RUN, InstanceLock
from twin.ops.process_model import ExitCode
from twin.profile.overrides import OverrideError, RoutineOverrides, parse_weekdays
from twin.services import Services, build_services

runner = CliRunner()


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
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


def fill(**spec: object) -> None:
    with_services(lambda services: build_chat(services, ChatSpec(**spec)))  # type: ignore[arg-type]


def added_id(output: str) -> str:
    match = re.search(r"added (\w+):", output)
    assert match, output
    return match.group(1)


# ---------------------------------------------------------------- profile


def test_rebuild_show_history_diff_and_rollback(data_dir: Path) -> None:
    fill(days=30)
    queued = cli("profile", "rebuild")
    assert queued.exit_code == 0 and "queued" in queued.output
    again = cli("profile", "rebuild")
    assert "already waiting" in again.output
    done = cli("profile", "rebuild", "--foreground")
    assert done.exit_code == 0, done.output
    assert "live: version" in done.output and "pre_holdout: version" in done.output
    assert "confirm the sleep time" in done.output

    history = cli("profile", "history")
    assert history.exit_code == 0 and "live" in history.output and "pre_holdout" in history.output
    only = cli("profile", "history", "--scope", "pre_holdout")
    assert "live" not in only.output.replace("pre_holdout", "")

    shown = cli("profile", "show")
    assert shown.exit_code == 0, shown.output
    for fragment in (
        "画像版本",
        "风格指标",
        "SPEC §0 样本",
        "数字风格规则",
        "几乎不用逗号",
        "作息概览（当地时间",
        "睡眠·工作日",
        "忙碌·工作日",
        "请确认：上面推断的睡眠时段与实际相符吗",
        "twin routine add sleep",
        "hold-out 切分点",
    ):
        assert fragment in shown.output, fragment
    assert "警告" not in shown.output
    pre = cli("profile", "show", "--scope", "pre_holdout")
    assert pre.exit_code == 0 and "范围 pre_holdout" in pre.output

    forced = cli("profile", "rebuild", "--foreground", "--force")
    assert forced.exit_code == 0, forced.output
    diff = cli("profile", "diff", "~1", "~0", "--scope", "live")
    assert diff.exit_code == 0 and "0 metric(s) changed over 10%" in diff.output
    named = cli("profile", "show", "~1")
    assert named.exit_code == 0 and "画像版本" in named.output

    older = with_services(lambda s: _version_ids(s, "live"))
    rolled = cli("profile", "rollback", older[1])
    assert rolled.exit_code == 0 and older[1] in rolled.output
    assert "routine model is now version" in rolled.output
    assert older[1] in cli("profile", "show").output


def _version_ids(services: Services, scope: str) -> list[str]:
    from twin.profile.store import VersionStore

    return [v.id for v in VersionStore(services.db, services.clock).history(scope)]


def test_phrases_are_shown_on_screen_only_on_request(data_dir: Path) -> None:
    from datetime import UTC, datetime, timedelta

    from tests.support.synth_chat import append_texts

    fill(days=10)
    start = datetime(2026, 9, 20, tzinfo=UTC)

    def add(services: Services) -> None:
        rows = [(start + timedelta(hours=i), True, "晚安宝宝") for i in range(12)]
        append_texts(services, rows)

    with_services(add)
    cli("profile", "rebuild", "--foreground", "--scope", "live")
    shown = cli("profile", "phrases", "--top", "5")
    assert shown.exit_code == 0, shown.output
    assert "form-of-address candidates" in shown.output and "宝宝" in shown.output
    assert "12\t晚安宝宝" in shown.output and "frequent 2-character groups" in shown.output
    assert cli("profile", "phrases", "--scope", "pre_holdout").exit_code == ExitCode.FAILURE
    assert cli("profile", "phrases", "--scope", "bogus").exit_code == ExitCode.USAGE


def test_a_daytime_sleep_is_announced_loudly_in_show(data_dir: Path) -> None:
    fill(days=30, zone="Asia/Shanghai")  # the default source zone is America/Chicago
    assert cli("profile", "rebuild", "--foreground", "--scope", "live").exit_code == 0
    shown = cli("profile", "show")
    assert shown.exit_code == 0
    assert "!!! 警告：推断的睡眠核心" in shown.output and "source_timezone" in shown.output


def test_errors_are_short_and_have_exit_codes(data_dir: Path) -> None:
    missing = cli("profile", "show")
    assert missing.exit_code == ExitCode.FAILURE and "twin profile rebuild" in missing.output
    assert cli("profile", "history").exit_code == ExitCode.FAILURE
    assert cli("profile", "rebuild", "--scope", "bogus").exit_code == ExitCode.USAGE
    assert cli("profile", "history", "--scope", "bogus").exit_code == ExitCode.USAGE
    assert cli("profile", "show", "--scope", "bogus").exit_code == ExitCode.USAGE
    fill(days=20)
    cli("profile", "rebuild", "--foreground", "--scope", "live")
    assert cli("profile", "show", "ZZZZZZ").exit_code == ExitCode.FAILURE
    assert cli("profile", "diff", "ZZZZZZ", "~0").exit_code == ExitCode.FAILURE
    assert cli("profile", "rollback", "~9").exit_code == ExitCode.FAILURE
    assert cli("profile", "rollback", "~0", "--scope", "bogus").exit_code == ExitCode.USAGE
    assert cli("profile", "diff", "~0", "~0", "--scope", "bogus").exit_code == ExitCode.USAGE


def test_an_empty_database_gets_a_clear_message_from_the_foreground_rebuild(
    data_dir: Path,
) -> None:
    done = cli("profile", "rebuild", "--foreground")
    assert done.exit_code == 0 and "nothing computed" in done.output


def test_a_running_application_executes_the_job_itself(data_dir: Path) -> None:
    fill(days=10)
    lock = InstanceLock(LOCK_RUN, locks_dir=resolve_paths(load_settings()).locks_dir)
    assert lock.acquire()
    try:
        result = cli("profile", "rebuild", "--foreground")
    finally:
        lock.release()
    assert result.exit_code == 0 and "the application is running" in result.output


# ---------------------------------------------------------------- routine


def test_routine_corrections_are_listed_added_switched_and_removed(data_dir: Path) -> None:
    fill(days=30)
    assert cli("profile", "rebuild", "--foreground", "--scope", "live").exit_code == 0
    assert "no manual corrections" in cli("routine", "list").output
    sleep = cli("routine", "add", "sleep", "02:00", "10:30", "--days", "weekend,holiday")
    busy = cli("routine", "add", "busy", "09:00", "11:00", "--weekdays", "tue,thu")
    holiday = cli("routine", "add", "holiday", "2026-10-01", "2026-10-07")
    for result in (sleep, busy, holiday):
        assert result.exit_code == 0, result.output
    assert "run `twin profile rebuild`" in holiday.output
    listing = cli("routine", "list").output
    for fragment in (
        "睡眠 02:00–10:30",
        "weekend",
        "忙碌 周二、周四 09:00–11:00",
        "节假日 2026-10-01",
    ):
        assert fragment in listing, fragment
    shown = cli("profile", "show").output
    assert "手动修正（优先于推断）" in shown and "（手动修正）" in shown
    assert "睡眠·周末：02:00–10:30" in shown

    sleep_id = added_id(sleep.output)
    assert cli("routine", "disable", sleep_id).exit_code == 0
    assert "（已停用）" in cli("profile", "show").output
    assert "off" in cli("routine", "list").output
    assert "睡眠·周末：02:00–10:30" not in cli("profile", "show").output
    assert cli("routine", "enable", sleep_id).exit_code == 0
    assert cli("routine", "remove", sleep_id).exit_code == 0
    assert sleep_id not in cli("routine", "list").output
    for command in ("remove", "enable", "disable"):
        assert cli("routine", command, "no-such-id").exit_code == ExitCode.FAILURE


def test_bad_corrections_are_refused_with_a_usage_error(data_dir: Path) -> None:
    for args in (
        ("add", "sleep", "25:00", "08:00"),
        ("add", "sleep", "01:00", "01:00"),
        ("add", "sleep", "01:00", "08:00", "--days", "tomorrow"),
        ("add", "busy", "17:00", "13:00"),
        ("add", "busy", "09:00", "11:00", "--weekdays", "funday"),
        ("add", "holiday", "2026-10-07", "2026-10-01"),
        ("add", "holiday", "yesterday"),
    ):
        result = cli("routine", *args)
        assert result.exit_code == ExitCode.USAGE, args
    assert "no manual corrections" in cli("routine", "list").output


def test_a_holiday_range_is_a_single_day_when_one_date_is_given(data_dir: Path) -> None:
    result = cli("routine", "add", "holiday", "2026-10-01")
    assert result.exit_code == 0 and "2026-10-01 至 2026-10-01" in result.output


# ---------------------------------------------------------------- the API


def test_weekday_names_in_both_languages(services: Services) -> None:
    assert parse_weekdays("mon-fri") == [0, 1, 2, 3, 4]
    assert parse_weekdays("sat,sun") == [5, 6]
    assert parse_weekdays("周一至周五") == [0, 1, 2, 3, 4]
    assert parse_weekdays("周六、周日") == [5, 6]
    assert parse_weekdays("星期三 Fri") == [2, 4]
    assert parse_weekdays("5-1") == [0, 4, 5, 6]  # a range may wrap past Sunday
    assert parse_weekdays("tuesday") == [1]
    for bad in ("", "  ", "mon-tue-wed", "noday"):
        with pytest.raises(OverrideError):
            parse_weekdays(bad)


def test_the_override_api_validates_and_persists(services: Services) -> None:
    api = RoutineOverrides(services.db, services.clock)
    sleep = api.add_sleep(60, "8:30", day_types=["weekend", "weekend"], note="note")
    assert sleep.params == {"start": "01:00", "end": "08:30", "day_types": ["weekend"]}
    assert sleep.note == "note" and sleep.enabled
    busy = api.add_busy([4, 0, 0], "13:00", 17 * 60)
    assert busy.params["weekdays"] == [0, 4] and busy.params["end"] == "17:00"
    holiday = api.add_holiday(date(2026, 8, 1), date(2026, 8, 3))
    assert [v.kind for v in api.entries()] == ["sleep", "busy", "holiday"]
    assert api.holiday_ranges() == [(date(2026, 8, 1), date(2026, 8, 3))]
    assert api.set_enabled(holiday.id, False) and api.holiday_ranges() == []
    assert [v.id for v in api.entries(include_disabled=False)] == [sleep.id, busy.id]
    assert api.remove(sleep.id) and not api.remove(sleep.id) and not api.set_enabled("x", True)
    for call in (
        lambda: api.add_sleep("01:00", "01:00"),
        lambda: api.add_sleep("01:00", "08:00", day_types=["someday"]),
        lambda: api.add_sleep(1440, "08:00"),
        lambda: api.add_busy([7], "09:00", "10:00"),
        lambda: api.add_busy([], "09:00", "10:00"),
        lambda: api.add_busy([1], "10:00", "09:00"),
        lambda: api.add_holiday(date(2026, 8, 3), date(2026, 8, 1)),
        lambda: api.add_sleep("noon", "08:00"),
    ):
        with pytest.raises(OverrideError):
            call()
    assert [v.kind for v in api.entries()] == ["busy", "holiday"]


def test_corrections_are_sealed_at_rest(services: Services) -> None:
    api = RoutineOverrides(services.db, services.clock)
    item = api.add_sleep("01:15", "08:45", note="secret note text")
    from sqlalchemy import select

    from twin.storage.profile_models import RoutineOverride

    with services.db.session() as session:
        row = session.scalars(select(RoutineOverride)).one()
        assert row.id == item.id
        assert b"01:15" not in bytes(row.params_ct) and b"secret note" not in bytes(
            row.note_ct or b""
        )
