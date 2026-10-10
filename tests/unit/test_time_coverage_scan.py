"""scripts/time_coverage_scan.py: every time-related module has a test that can move the clock.

R-NFR-005: "all time-related logic has tests on an injected clock, DST switch days and time zone
switch days included".  The scan reads the source, so it is tested on small trees written here,
and then on this repository: the second part is the requirement itself - a new module that uses a
clock or a zone and has no test fails the suite until it gets one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.scripts import SCRIPTS, load_script

scan_module = load_script("time_coverage_scan")


def write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def tree(root: Path) -> Path:
    """A small package: clock and zone users, a re-exporting package, a module without time."""
    write(root, "src/twin/__init__.py", "")
    write(root, "src/twin/clock.py", "from zoneinfo import ZoneInfo\n")
    write(root, "src/twin/plain.py", "def add(a, b):\n    return a + b\n")
    write(root, "src/twin/sched/__init__.py", "from twin.sched.plan import Plan\n")
    write(root, "src/twin/sched/plan.py", "from twin.clock import Clock\n\nclass Plan: ...\n")
    write(root, "src/twin/sched/zone.py", "from zoneinfo import ZoneInfo\n")
    write(
        root,
        "src/twin/sched/service.py",
        "from twin.schedule.time_service import TimeService\n",
    )
    write(root, "src/twin/mem/__init__.py", "")
    write(root, "src/twin/mem/days.py", "import zoneinfo\n")
    return root


def covered(result: object, name: str) -> list[str]:
    [found] = [m for m in result.modules if m.name == name]  # type: ignore[attr-defined]
    return found.tests


def test_a_module_is_time_related_when_it_uses_a_zone_a_clock_or_the_time_service(
    tmp_path: Path,
) -> None:
    result = scan_module.scan(tree(tmp_path))
    related = {m.name: m.reasons for m in result.related}
    assert related == {
        "twin.clock": ["zoneinfo"],
        "twin.sched.plan": ["Clock"],
        "twin.sched.zone": ["zoneinfo"],
        "twin.sched.service": ["TimeService"],
        "twin.mem.days": ["zoneinfo"],
    }  # (a plain module, and packages that only re-export, are not)
    assert {m.name for m in result.gaps} == set(related)  # and without a test they are all gaps


def test_a_test_covers_the_modules_it_imports_when_it_can_move_time(tmp_path: Path) -> None:
    root = tree(tmp_path)
    write(
        root,
        "tests/unit/test_plan.py",
        "from tests.support.clock import ManualClock\n"
        "from twin.sched.zone import ZoneInfo\n"
        "import twin.mem.days\n"
        "def test_x(clock): ...\n",
    )
    result = scan_module.scan(root)
    assert covered(result, "twin.sched.zone") == ["tests/unit/test_plan.py"]
    assert covered(result, "twin.mem.days") == ["tests/unit/test_plan.py"]
    assert covered(result, "twin.sched.plan") == []  # not imported
    assert {m.name for m in result.gaps} == {"twin.clock", "twin.sched.plan", "twin.sched.service"}


def test_a_name_a_package_re_exports_counts_for_the_module_it_comes_from(tmp_path: Path) -> None:
    root = tree(tmp_path)
    write(
        root,
        "tests/unit/test_plan.py",
        "from twin.sched import Plan\ndef test_x(clock): ...\n",
    )
    assert covered(scan_module.scan(root), "twin.sched.plan") == ["tests/unit/test_plan.py"]


def test_a_submodule_imported_by_its_package_name_counts(tmp_path: Path) -> None:
    root = tree(tmp_path)
    write(root, "tests/unit/test_zone.py", "from twin.sched import zone\ndef test_x(clock): ...\n")
    assert covered(scan_module.scan(root), "twin.sched.zone") == ["tests/unit/test_zone.py"]


def test_a_test_that_cannot_move_time_does_not_count(tmp_path: Path) -> None:
    root = tree(tmp_path)
    write(root, "tests/unit/test_zone.py", "from twin.sched.zone import ZoneInfo\n")
    result = scan_module.scan(root)
    assert covered(result, "twin.sched.zone") == []
    assert "twin.sched.zone" in {m.name for m in result.gaps}


def test_fixed_aware_instants_are_a_way_of_moving_time(tmp_path: Path) -> None:
    root = tree(tmp_path)
    write(
        root,
        "tests/unit/test_zone.py",
        "from datetime import UTC, datetime\nfrom twin.sched.zone import ZoneInfo\n"
        "NOW = datetime(2026, 11, 1, 7, 30, tzinfo=UTC)\n",
    )
    helper = "from twin.mem.days import zoneinfo\nSTART = utc(2026, 3, 8)\n"
    write(root, "tests/unit/test_days.py", helper)
    result = scan_module.scan(root)
    assert covered(result, "twin.sched.zone") == ["tests/unit/test_zone.py"]
    assert covered(result, "twin.mem.days") == ["tests/unit/test_days.py"]


def test_being_imported_by_another_module_of_the_package_is_not_a_test(tmp_path: Path) -> None:
    root = tree(tmp_path)
    write(
        root, "tests/unit/test_service.py", "from twin.sched.service import x\ndef t(clock): ...\n"
    )
    result = scan_module.scan(root)
    assert covered(result, "twin.sched.service") == ["tests/unit/test_service.py"]
    assert covered(result, "twin.sched.plan") == []  # (the service does not import it here)


def test_the_two_days_the_requirement_names_must_have_a_test_each(tmp_path: Path) -> None:
    root = tree(tmp_path)
    write(root, "tests/unit/test_days.py", "def test_x(clock): ...\n")
    result = scan_module.scan(root)
    assert result.dst_tests == [] and result.zone_tests == [] and not result.ok
    write(
        root,
        "tests/unit/test_calendar.py",
        "def test_the_25_hour_day(clock): ...\n"
        "def test_spring_forward_skips_an_hour(clock): ...\n"
        "def test_a_zone_switch_moves_the_plan(clock): ...\n"
        "async def test_switch_timezone_in_the_chat(clock): ...\n"
        "def test_an_unknown_time_zone_changes_nothing(clock): ...\n"
        "def test_the_pipeline_needs_deepseek_to_fall_back_on(clock): ...\n",
    )
    result = scan_module.scan(root)
    assert [name.split("::")[1] for name in result.dst_tests] == [
        "test_the_25_hour_day",
        "test_spring_forward_skips_an_hour",
    ]
    assert [name.split("::")[1] for name in result.zone_tests] == [
        "test_a_zone_switch_moves_the_plan",
        "test_switch_timezone_in_the_chat",
    ]  # ("an unknown time zone changes nothing" and "fall back on" are about something else)


def test_the_result_is_ok_only_without_gaps_and_with_both_kinds_of_day(tmp_path: Path) -> None:
    root = tree(tmp_path)
    for name in ("clock", "sched.plan", "sched.zone", "sched.service", "mem.days"):
        short = name.replace(".", "_")
        write(root, f"tests/unit/test_{short}.py", f"import twin.{name}\ndef test_x(clock): ...\n")
    write(
        root,
        "tests/unit/test_days.py",
        "def test_a_dst_day(clock): ...\ndef test_the_zone_switch(clock): ...\n",
    )
    result = scan_module.scan(root)
    assert result.gaps == [] and result.ok
    text = scan_module.render(result)
    assert "gaps: none" in text and "result: OK" in text
    assert "daylight saving switch tests: 1" in text and "time zone switch tests: 1" in text


def test_the_report_names_the_gaps_and_the_json_has_them(tmp_path: Path) -> None:
    result = scan_module.scan(tree(tmp_path))
    text = scan_module.render(result, everything=True)
    assert "twin.sched.plan  (src/twin/sched/plan.py)  [Clock]" in text
    assert "-- no test --" in text and "MISSING: no test of a daylight saving switch day" in text
    assert "result: GAPS" in text
    import json

    data = json.loads(scan_module.to_json(result))
    assert "twin.mem.days" in data["gaps"] and data["ok"] is False
    assert {m["name"] for m in data["modules"]} == {m.name for m in result.related}


def test_a_file_that_is_not_python_is_skipped(tmp_path: Path) -> None:
    root = tree(tmp_path)
    write(root, "src/twin/broken.py", "def (:\n")
    write(root, "tests/unit/test_broken.py", "def (:\n")
    result = scan_module.scan(root)
    assert "twin.broken" not in {m.name for m in result.modules}


def test_the_command_line_exits_one_with_a_gap_and_writes_the_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tree(tmp_path / "repo")
    out = tmp_path / "found.json"
    assert scan_module.main(["--root", str(root), "--json", str(out)]) == 1
    printed = capsys.readouterr().out
    assert "gaps (a test must import the module and use an injected clock):" in printed
    assert out.is_file()


# --------------------------------------------------------------------------- this repository


def test_every_time_related_module_of_this_repository_has_a_test_that_moves_the_clock() -> None:
    result = scan_module.scan(SCRIPTS.parent)
    assert len(result.related) > 100  # (the scan did read the package)
    assert result.gaps == [], scan_module.render(result)
    assert result.dst_tests and result.zone_tests
    assert result.ok
