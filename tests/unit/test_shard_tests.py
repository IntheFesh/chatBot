"""scripts/shard_tests.py: the file-level sharding of the CI matrix (D-490..D-499)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.scripts import SCRIPTS, load_script

shard = load_script("shard_tests")
REPO = SCRIPTS.parent


def write(root: Path, name: str, text: str = "def test_it() -> None: ...\n") -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_tree(root: Path, count: int = 23) -> list[str]:
    """A tests/ tree with ``count`` files of different sizes plus files pytest ignores."""
    names = []
    for number in range(count):
        package = "unit" if number % 3 else "integration"
        name = f"tests/{package}/test_file_{number:02d}.py"
        write(root, name, "".join(f"def test_{i}() -> None: ...\n" for i in range(1 + number % 7)))
        names.append(name)
    write(root, "tests/conftest.py", "import pytest\n")
    write(root, "tests/support/helper.py", "def test_not_a_module_of_tests() -> None: ...\n")
    write(root, "tests/unit/__pycache__/test_cached.py", "def test_x() -> None: ...\n")
    return sorted(names)


def shards_of(root: Path, count: int, platform: str = "linux") -> list[list[str]]:
    return [
        shard.select_shard(index, count, platform=platform, root=root)[0]
        for index in range(1, count + 1)
    ]


# ------------------------------------------------------------------ the universe


def test_discovery_finds_every_test_file_and_nothing_else(tmp_path: Path) -> None:
    names = make_tree(tmp_path)
    write(tmp_path, "tests/unit/deep/er/other_test.py")  # pytest's second default pattern
    assert shard.discover_test_files(tmp_path) == sorted(
        [*names, "tests/unit/deep/er/other_test.py"]
    )


def test_the_real_repository_is_discovered_without_a_list() -> None:
    files = shard.discover_test_files(REPO)
    assert (
        "tests/unit/test_shard_tests.py" in files
        and "tests/integration/test_multiprocess.py" in files
    )
    assert len(files) > 200
    assert all(name.startswith("tests/") and "\\" not in name for name in files)


# ------------------------------------------------- partition properties of the shards


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 7])
def test_the_shards_partition_the_universe(tmp_path: Path, count: int) -> None:
    universe = make_tree(tmp_path)
    parts = shards_of(tmp_path, count)
    flat = [name for part in parts for name in part]
    assert sorted(flat) == universe  # union is the whole universe ...
    assert len(flat) == len(set(flat))  # ... and no file is in two shards
    assert all(part == sorted(part) for part in parts)


def test_one_shard_is_the_whole_universe(tmp_path: Path) -> None:
    universe = make_tree(tmp_path)
    assert shards_of(tmp_path, 1) == [universe]


def test_the_same_tree_and_weights_always_give_the_same_shards(tmp_path: Path) -> None:
    make_tree(tmp_path)
    first = shards_of(tmp_path, 4)
    assert shards_of(tmp_path, 4) == first
    # the order in which the weights were written does not matter either
    table = {"linux": {name: float(len(name) % 5) for name in shard.discover_test_files(tmp_path)}}
    shard.save_weights(table, tmp_path / "ci" / "test_weights.json")
    again = shards_of(tmp_path, 4)
    assert shards_of(tmp_path, 4) == again


def test_a_new_unregistered_file_is_picked_up_automatically(tmp_path: Path) -> None:
    universe = make_tree(tmp_path)
    table = {"linux": dict.fromkeys(universe, 5.0), "windows": {}}
    shard.save_weights(table, tmp_path / "ci" / "test_weights.json")
    before = shards_of(tmp_path, 3)
    write(tmp_path, "tests/unit/test_brand_new.py", "def test_a() -> None: ...\n" * 3)
    after = shards_of(tmp_path, 3)
    flat = [name for part in after for name in part]
    assert sorted(flat) == sorted([*universe, "tests/unit/test_brand_new.py"])
    assert flat.count("tests/unit/test_brand_new.py") == 1
    _, weighing = shard.select_shard(1, 3, platform="linux", root=tmp_path)
    assert weighing.estimated == 1 and weighing.recorded == len(universe)
    assert sum(len(part) for part in before) == len(universe)


def test_a_new_file_is_priced_from_its_test_count_in_seconds(tmp_path: Path) -> None:
    write(tmp_path, "tests/unit/test_old.py", "def test_a() -> None: ...\n" * 4)  # 4 tests, 40 s
    write(tmp_path, "tests/unit/test_new.py", "def test_a() -> None: ...\n" * 2)  # 2 tests
    table = {"linux": {"tests/unit/test_old.py": 40.0}, "windows": {}}
    weighing = shard.weigh(
        ["tests/unit/test_new.py", "tests/unit/test_old.py"], table, "linux", tmp_path
    )
    assert weighing.weights == {"tests/unit/test_old.py": 40.0, "tests/unit/test_new.py": 20.0}
    nothing = shard.weigh(
        ["tests/unit/test_new.py"], {"linux": {}, "windows": {}}, "linux", tmp_path
    )
    assert nothing.weights == {"tests/unit/test_new.py": 2.0}  # without any time: the test count


def test_without_times_for_a_platform_the_other_platforms_times_are_used(tmp_path: Path) -> None:
    write(tmp_path, "tests/unit/test_a.py")
    write(tmp_path, "tests/unit/test_b.py")
    table = {"linux": {"tests/unit/test_a.py": 9.0, "tests/unit/test_b.py": 1.0}, "windows": {}}
    weighing = shard.weigh(
        ["tests/unit/test_a.py", "tests/unit/test_b.py"], table, "windows", tmp_path
    )
    assert weighing.borrowed_from == "linux" and weighing.weights["tests/unit/test_a.py"] == 9.0


def test_the_test_count_follows_pytest_collection_rules(tmp_path: Path) -> None:
    write(
        tmp_path,
        "tests/unit/test_counting.py",
        "import pytest\n"
        "def test_plain() -> None: ...\n"
        "async def test_async() -> None: ...\n"
        "def helper() -> None: ...\n"
        "@pytest.mark.parametrize('x', [1, 2, 3])\n"
        "def test_three(x: int) -> None: ...\n"
        "@pytest.mark.parametrize('x', [1, 2])\n"
        "@pytest.mark.parametrize('y', 'abc')\n"
        "def test_six(x: int, y: str) -> None: ...\n"
        "@pytest.mark.parametrize('x', range(50))\n"
        "def test_unknown(x: int) -> None: ...\n"
        "class TestGroup:\n"
        "    def test_method(self) -> None: ...\n"
        "    def not_a_test(self) -> None: ...\n"
        "class Helper:\n"
        "    def test_hidden(self) -> None: ...\n",
    )
    assert shard.count_tests(tmp_path / "tests/unit/test_counting.py") == 1 + 1 + 3 + 6 + 1 + 1
    write(tmp_path, "tests/unit/test_empty.py", "VALUE = 1\n")
    write(tmp_path, "tests/unit/test_broken.py", "def (:\n")
    assert shard.count_tests(tmp_path / "tests/unit/test_empty.py") == 1
    assert shard.count_tests(tmp_path / "tests/unit/test_broken.py") == 1
    assert shard.count_tests(tmp_path / "tests/unit/test_missing.py") == 1


# ------------------------------------------------------------------- the balance


def test_the_greedy_assignment_balances_the_weights() -> None:
    weights = {
        f"tests/unit/test_{i:02d}.py": float(w)
        for i, w in enumerate([50, 40, 30, 30, 20, 20, 10, 10, 5, 5])
    }
    parts = shard.assign(weights, 3)
    loads = sorted(sum(weights[name] for name in part) for part in parts)
    assert loads[-1] - loads[0] <= 5
    assert sorted(name for part in parts for name in part) == sorted(weights)


def test_one_heavy_file_gets_a_shard_of_its_own() -> None:
    weights = {
        "tests/unit/test_big.py": 100.0,
        **{f"tests/unit/test_s{i}.py": 1.0 for i in range(9)},
    }
    parts = shard.assign(weights, 2)
    assert ["tests/unit/test_big.py"] in parts
    assert shard.assign(weights, 2) == parts


def test_ties_break_by_path_and_shard_index() -> None:
    weights = {
        "tests/unit/test_b.py": 1.0,
        "tests/unit/test_a.py": 1.0,
        "tests/unit/test_c.py": 1.0,
    }
    assert shard.assign(weights, 2) == [
        ["tests/unit/test_a.py", "tests/unit/test_c.py"],
        ["tests/unit/test_b.py"],
    ]


def test_more_shards_than_files_leave_some_empty() -> None:
    parts = shard.assign({"tests/unit/test_a.py": 1.0}, 3)
    assert parts == [["tests/unit/test_a.py"], [], []]


def test_the_committed_weights_balance_the_real_suite() -> None:
    files = shard.discover_test_files(REPO)
    table = shard.load_weights(REPO / "ci" / "test_weights.json")
    for platform, count in (("linux", 3), ("windows", 5)):
        weighing = shard.weigh(files, table, platform, REPO)
        parts = shard.assign(weighing.weights, count)
        assert sorted(name for part in parts for name in part) == files
        loads = [sum(weighing.weights[name] for name in part) for part in parts]
        biggest_file = max(weighing.weights.values())
        assert max(loads) - min(loads) <= max(biggest_file, 0.15 * sum(loads) / count)


# -------------------------------------------------------------- argument handling


@pytest.mark.parametrize(("index", "count"), [(0, 3), (4, 3), (-1, 2)])
def test_a_shard_outside_the_range_is_an_error(tmp_path: Path, index: int, count: int) -> None:
    make_tree(tmp_path, 4)
    with pytest.raises(shard.ShardError):
        shard.select_shard(index, count, root=tmp_path)


def test_an_empty_tree_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    with pytest.raises(shard.ShardError, match="no test files"):
        shard.select_shard(1, 2, root=tmp_path)


def test_zero_shards_is_an_error() -> None:
    with pytest.raises(shard.ShardError):
        shard.assign({"tests/unit/test_a.py": 1.0}, 0)


def test_the_command_line_prints_the_files_of_a_shard(
    capfdbinary: pytest.CaptureFixture[bytes],
) -> None:
    assert shard.main(["--shard", "2", "--of", "3", "--platform", "linux"]) == 0
    out = capfdbinary.readouterr().out
    assert b"\r" not in out and out.endswith(b"\n")
    files = out.decode("utf-8").splitlines()
    expected = shard.select_shard(2, 3, platform="linux")[0]
    assert files == expected and files


def test_every_file_of_the_real_suite_is_in_exactly_one_command_line_shard(
    capfdbinary: pytest.CaptureFixture[bytes],
) -> None:
    seen: list[str] = []
    for index in range(1, 6):
        assert shard.main(["--shard", str(index), "--of", "5", "--platform", "windows"]) == 0
        seen += capfdbinary.readouterr().out.decode("utf-8").splitlines()
    assert sorted(seen) == shard.discover_test_files(REPO)


def test_the_command_line_rejects_bad_usage(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        shard.main([])
    with pytest.raises(SystemExit):
        shard.main(["--shard", "1"])
    assert shard.main(["--shard", "5", "--of", "2"]) == 2
    assert "between 1 and --of" in capsys.readouterr().err


def test_the_plan_lists_every_shard(capfdbinary: pytest.CaptureFixture[bytes]) -> None:
    assert shard.main(["--of", "3", "--plan", "--platform", "linux"]) == 0
    lines = capfdbinary.readouterr().out.decode("utf-8").splitlines()
    assert lines[0].startswith(f"{len(shard.discover_test_files(REPO))} test files in 3 shards")
    assert [line.split(":")[0].strip() for line in lines[1:]] == [
        "shard 1/3",
        "shard 2/3",
        "shard 3/3",
    ]


# ------------------------------------------------------------- weights and timings


def junit(root: Path, cases: list[tuple[str, str, float]], name: str = "junit.xml") -> Path:
    body = "".join(
        f'<testcase classname="{classname}" name="{test}" time="{seconds}"/>'
        for classname, test, seconds in cases
    )
    path = root / name
    path.write_text(
        f'<testsuites><testsuite name="pytest">{body}</testsuite></testsuites>', encoding="utf-8"
    )
    return path


def test_timings_are_summed_per_file_from_a_junit_report(tmp_path: Path) -> None:
    make_tree(tmp_path, 3)
    write(tmp_path, "tests/unit/test_cls.py")
    report = junit(
        tmp_path,
        [
            ("tests.unit.test_file_01", "test_a", 1.5),
            ("tests.unit.test_file_01", "test_b[x.y]", 2.0),
            ("tests.unit.test_cls.TestInner", "test_c", 4.0),
            ("tests.unit.vanished", "test_d", 9.0),
        ],
    )
    timings = shard.timings_from_junit(report, tmp_path)
    assert timings == {"tests/unit/test_file_01.py": (3.5, 2), "tests/unit/test_cls.py": (4.0, 1)}


def test_the_printed_timing_table_reads_back_out_of_a_ci_log(tmp_path: Path) -> None:
    timings = {"tests/unit/test_a.py": (12.34, 3), "tests/integration/test_b.py": (700.0, 9)}
    table = shard.format_timings(timings, "windows")
    log = "\n".join(f"2026-10-10T03:13:00.1234567Z {line}" for line in table.splitlines())
    parsed = shard.timings_from_log("noise\n" + log + "\nmore noise 12s\n")
    assert parsed == {"tests/unit/test_a.py": (12.3, 0), "tests/integration/test_b.py": (700.0, 0)}


def test_updating_the_weights_merges_one_platform_and_drops_vanished_files(tmp_path: Path) -> None:
    make_tree(tmp_path, 4)
    first, other, unit1 = (
        "tests/integration/test_file_00.py",
        "tests/integration/test_file_03.py",
        "tests/unit/test_file_01.py",
    )
    existing = {
        "linux": {first: 1.0, "tests/unit/test_gone.py": 5.0},
        "windows": {other: 2.0, first: 4.0},
    }
    shard.save_weights(existing, tmp_path / "ci" / "test_weights.json")
    report = junit(
        tmp_path,
        [("tests.unit.test_file_01", "t", 7.0), ("tests.integration.test_file_03", "t", 3.0)],
    )
    assert (
        shard.main(
            [
                "--update-weights",
                str(report),
                "--platform",
                "windows",
                "--root",
                str(tmp_path),
                "--weights",
                str(tmp_path / "ci" / "test_weights.json"),
            ]
        )
        == 0
    )
    saved = json.loads((tmp_path / "ci" / "test_weights.json").read_text(encoding="utf-8"))
    assert saved["version"] == 1
    # the measured files replace the old numbers of that platform; the other files stay
    assert saved["platforms"]["windows"] == {first: 4.0, other: 3.0, unit1: 7.0}
    # the other platform is untouched except that a file which no longer exists is dropped
    assert saved["platforms"]["linux"] == {first: 1.0}


def test_updating_prunes_vanished_files_of_the_updated_tree(tmp_path: Path) -> None:
    names = make_tree(tmp_path, 3)
    table = {"linux": {names[0]: 1.0, "tests/unit/test_gone.py": 5.0}, "windows": {}}
    merged = shard.updated_weights(table, {names[1]: (2.0, 1)}, "linux", tmp_path)
    assert merged["linux"] == {names[0]: 1.0, names[1]: 2.0}


def test_blending_keeps_a_moving_average_over_the_runs(tmp_path: Path) -> None:
    names = make_tree(tmp_path, 3)
    table = {"linux": {names[0]: 10.0}, "windows": {}}
    merged = shard.updated_weights(
        table, {names[0]: (20.0, 1), names[1]: (8.0, 1)}, "linux", tmp_path, blend=0.25
    )
    assert merged["linux"] == {names[0]: 12.5, names[1]: 8.0}  # a new file takes the measurement
    for bad in (0.0, -1.0, 1.5):
        with pytest.raises(shard.ShardError, match="--blend"):
            shard.updated_weights(table, {names[0]: (1.0, 1)}, "linux", tmp_path, blend=bad)


def test_the_blend_option_reaches_the_weights_file(tmp_path: Path) -> None:
    make_tree(tmp_path, 4)
    path = tmp_path / "ci" / "test_weights.json"
    shard.save_weights({"linux": {"tests/unit/test_file_01.py": 10.0}, "windows": {}}, path)
    report = junit(tmp_path, [("tests.unit.test_file_01", "t", 30.0)])
    arguments = ["--update-weights", str(report), "--platform", "linux", "--root", str(tmp_path)]
    assert shard.main([*arguments, "--weights", str(path), "--blend", "0.5"]) == 0
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["platforms"]["linux"] == {"tests/unit/test_file_01.py": 20.0}


def test_the_weights_file_is_validated(tmp_path: Path) -> None:
    path = tmp_path / "weights.json"
    assert shard.load_weights(path) == {"linux": {}, "windows": {}}  # absent: nothing recorded
    for text in ("not json", "[]", '{"version": 2}', '{"version": 1, "platforms": 3}',
                 '{"version": 1, "platforms": {"mac": {}}}',
                 '{"version": 1, "platforms": {"linux": {"a": -1}}}',
                 '{"version": 1, "platforms": {"linux": {"a": true}}}',
                 '{"version": 1, "platforms": {"linux": []}}'):  # fmt: skip
        path.write_text(text, encoding="utf-8")
        with pytest.raises(shard.ShardError):
            shard.load_weights(path)


def test_the_committed_weights_file_is_valid_and_mentions_real_files() -> None:
    table = shard.load_weights(REPO / "ci" / "test_weights.json")
    files = set(shard.discover_test_files(REPO))
    for platform, recorded in table.items():
        assert recorded, f"{platform}: no weights recorded"
        assert set(recorded) <= files, f"{platform}: weights of files that do not exist"


def test_reading_timings_needs_a_source_with_timings(tmp_path: Path) -> None:
    with pytest.raises(shard.ShardError, match="does not exist"):
        shard.read_timings([tmp_path / "nope.xml"])
    empty = tmp_path / "empty.log"
    empty.write_text("nothing here\n", encoding="utf-8")
    with pytest.raises(shard.ShardError, match="no per-file timings"):
        shard.read_timings([empty])
    broken = tmp_path / "broken.xml"
    broken.write_text("<testsuites>", encoding="utf-8")
    with pytest.raises(shard.ShardError, match="cannot read the junit report"):
        shard.read_timings([broken])


def test_a_directory_of_reports_and_later_sources_override_earlier_ones(tmp_path: Path) -> None:
    make_tree(tmp_path, 4)
    reports = tmp_path / "reports"
    reports.mkdir()
    junit(reports, [("tests.unit.test_file_01", "t", 1.0)], "a.xml")
    junit(reports, [("tests.integration.test_file_03", "t", 2.0)], "b.xml")
    assert shard.read_timings([reports], tmp_path) == {
        "tests/unit/test_file_01.py": (1.0, 1),
        "tests/integration/test_file_03.py": (2.0, 1),
    }
    log = tmp_path / "t.log"
    log.write_text("    9.0s  tests/unit/test_file_01.py  1 tests\n", encoding="utf-8")
    assert shard.read_timings([reports, log], tmp_path)["tests/unit/test_file_01.py"] == (9.0, 0)


def test_the_timings_command_prints_the_slowest_first(
    tmp_path: Path, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    log = tmp_path / "t.log"
    log.write_text(
        "    1.0s  tests/unit/test_a.py  1 tests\n  50.0s  tests/unit/test_b.py  2 tests\n",
        encoding="utf-8",
    )
    assert shard.main(["--timings", str(log), "--platform", "windows"]) == 0
    lines = capfdbinary.readouterr().out.decode("utf-8").splitlines()
    assert lines[0].startswith("seconds per test file (windows, 2 files, 51.0s in total)")
    assert lines[1].endswith("tests/unit/test_b.py  0 tests") and lines[2].endswith(
        "tests/unit/test_a.py  0 tests"
    )


# ---------------------------------------------------------------------- the checks


def test_collected_files_come_from_the_node_ids() -> None:
    output = (
        "tests/unit/test_a.py::test_x\n"
        "tests/unit/test_a.py::TestK::test_y[1-2]\n"
        "tests\\unit\\test_b.py::test_z\n"
        "\n5937/5941 tests collected (4 deselected) in 34.53s\n"
    )
    assert shard.collected_files(output) == {"tests/unit/test_a.py", "tests/unit/test_b.py"}


def test_pytest_collects_no_file_that_sharding_would_miss() -> None:
    # a few real files exercise the subprocess path without collecting the whole suite
    targets = ["tests/unit/test_shard_tests.py", "tests/unit/test_gate_scripts.py"]
    assert shard.check_collection(REPO, targets) == []


def test_a_file_that_pytest_collects_but_sharding_does_not_know_is_reported(tmp_path: Path) -> None:
    write(tmp_path, "tests/unit/test_known.py")
    write(tmp_path, "tests/unit/check_unlisted.py")
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\npython_files = test_*.py check_*.py\n", encoding="utf-8"
    )
    assert shard.check_collection(tmp_path, ["tests"]) == [
        "pytest collects tests/unit/check_unlisted.py, which sharding would never select"
    ]


def test_a_failing_collection_is_reported(tmp_path: Path) -> None:
    write(tmp_path, "tests/unit/test_broken.py", "import module_that_does_not_exist\n")
    problems = shard.check_collection(tmp_path, ["tests"])
    assert len(problems) == 1 and problems[0].startswith("pytest --collect-only failed")


def test_the_complete_shard_set_is_recognised() -> None:
    names = [f".coverage.windows-latest.{i}-of-4" for i in (3, 1, 4, 2)]
    assert shard.check_shard_set(names) == []
    assert shard.check_shard_set(names[:3]) == ["missing shards of 4: [2]"]
    assert "shards present more than once: [1]" in shard.check_shard_set([*names, names[1]])
    assert shard.check_shard_set([names[0], ".coverage.windows-latest.1-of-3"])[0].startswith(
        "shard files disagree"
    )
    assert shard.check_shard_set(["coverage"]) == ["coverage: no '<i>-of-<n>' suffix"]
    assert shard.check_shard_set([]) == ["no shard files given"]
    assert shard.check_shard_set([".coverage.x.5-of-4"])[1:] == ["shard numbers outside 1..4: [5]"]


def test_the_check_commands_report_through_the_exit_code(
    capfdbinary: pytest.CaptureFixture[bytes],
) -> None:
    good = [f".coverage.os.{i}-of-2" for i in (1, 2)]
    assert shard.main(["--check-shard-set", *good]) == 0
    assert capfdbinary.readouterr().out == b"ok\n"
    assert shard.main(["--check-shard-set", good[0]]) == 1
    assert "missing shards of 2: [2]" in capfdbinary.readouterr().err.decode("utf-8")
