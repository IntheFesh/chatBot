"""scripts/coverage_gate.py and scripts/privacy_scan.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.scripts import SCRIPTS, load_script
from tests.support.synthetic import export_fragment

gate = load_script("coverage_gate")
scan = load_script("privacy_scan")


def report(files: dict[str, tuple[int, int]]) -> dict[str, object]:
    return {
        "files": {
            path: {"summary": {"covered_lines": covered, "num_statements": statements}}
            for path, (covered, statements) in files.items()
        }
    }


# ------------------------------------------------------------- coverage gate


def test_package_grouping() -> None:
    assert gate.package_of("src/twin/ops/jobs.py") == "ops"
    assert gate.package_of("src/twin/storage/migrations/env.py") == "storage"
    assert gate.package_of("src/twin/cli.py") == gate.TOP_LEVEL
    assert gate.package_of("src\\twin\\config\\settings.py") == "config"
    assert gate.package_of("elsewhere/file.py") == gate.TOP_LEVEL


def test_gate_passes_when_total_and_every_package_meet_the_thresholds() -> None:
    data = report(
        {
            "src/twin/ops/a.py": (90, 100),
            "src/twin/storage/b.py": (80, 100),
            "src/twin/cli.py": (95, 100),
        }
    )
    lines, failures = gate.evaluate(data, 85.0, 75.0)
    assert failures == []
    assert any("TOTAL" in line and "88.3%" in line for line in lines)


def test_gate_fails_for_a_weak_package_even_if_the_total_is_high() -> None:
    data = report({"src/twin/big/a.py": (990, 1000), "src/twin/weak/b.py": (7, 10)})
    _, failures = gate.evaluate(data, 85.0, 75.0)
    assert failures == ["package weak: 70.0% < 75%"]


def test_gate_fails_when_the_total_is_too_low() -> None:
    data = report({"src/twin/a/x.py": (80, 100), "src/twin/b/y.py": (80, 100)})
    _, failures = gate.evaluate(data, 85.0, 75.0)
    assert failures == ["total: 80.0% < 85%"]


def test_nested_directories_are_found_below_their_package() -> None:
    """R17-43 (D-740): every directory below a package is a group of its own."""
    assert gate.nested_packages_of("src/twin/ops/backup/archive.py") == ["ops/backup"]
    assert gate.nested_packages_of("src/twin/ops/a/b/c.py") == ["ops/a", "ops/a/b"]
    assert gate.nested_packages_of("src\\twin\\channel\\ilink\\wire.py") == ["channel/ilink"]
    assert gate.nested_packages_of("src/twin/ops/jobs.py") == []
    assert gate.nested_packages_of("src/twin/cli.py") == []
    assert gate.nested_packages_of("elsewhere/a/b.py") == []


def test_a_weak_directory_below_a_strong_package_fails_the_gate() -> None:
    """The parent as a whole is at 94 %, which used to hide the child that is at 60 %."""
    data = report(
        {
            "src/twin/ops/jobs.py": (900, 900),
            "src/twin/ops/backup/archive.py": (30, 50),
            "src/twin/ops/health/collect.py": (80, 100),
        }
    )
    lines, failures = gate.evaluate(data, 85.0, 75.0)
    assert failures == ["package ops/backup: 60.0% < 75%"]
    assert any(line.strip().startswith("ops/health") for line in lines)
    assert gate.evaluate(data, 85.0, 75.0, nested=False)[1] == []  # the old gate saw nothing


def test_the_total_counts_each_statement_once_whatever_the_groups_are() -> None:
    data = report({"src/twin/ops/a/x.py": (50, 100), "src/twin/ops/y.py": (100, 100)})
    per_group, total = gate.summarise(data, nested=True)
    assert total == (150, 200)
    assert per_group["ops"] == (150, 200) and per_group["ops/a"] == (50, 100)
    assert gate.summarise(data)[0] == {"ops": (150, 200)}


def test_main_gates_the_directories_unless_told_not_to(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "coverage.json"
    path.write_text(
        json.dumps(report({"src/twin/big/a.py": (990, 1000), "src/twin/big/weak/b.py": (6, 10)})),
        encoding="utf-8",
    )
    assert gate.main(["--json", str(path)]) == 1
    assert "package big/weak: 60.0% < 75%" in capsys.readouterr().out
    assert gate.main(["--json", str(path), "--no-nested"]) == 0


def test_the_windows_settings_count_the_win32_only_code_and_differ_in_nothing_else() -> None:
    """R17-43 (D-741): ``ci/coverage-windows.toml`` is pyproject's coverage table minus one line."""
    import tomllib

    root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    windows = tomllib.loads((root / "ci" / "coverage-windows.toml").read_text(encoding="utf-8"))
    linux_table = pyproject["tool"]["coverage"]
    windows_table = windows["tool"]["coverage"]
    assert "# pragma: win32-only" in linux_table["report"]["exclude_also"]
    assert "# pragma: win32-only" not in windows_table["report"]["exclude_also"]
    trimmed = json.loads(json.dumps(linux_table))
    trimmed["report"]["exclude_also"].remove("# pragma: win32-only")
    assert windows_table == trimmed
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "ci/coverage-windows.toml" in workflow and "windows-latest" in workflow


def test_empty_packages_cannot_fail() -> None:
    data = report({"src/twin/ok/a.py": (100, 100), "src/twin/channel/__init__.py": (0, 0)})
    lines, failures = gate.evaluate(data, 85.0, 75.0)
    assert failures == [] and any("channel" in line and "100.0%" in line for line in lines)


def test_main_reads_a_json_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(report({"src/twin/a/x.py": (100, 100)})), encoding="utf-8")
    assert gate.main(["--json", str(path)]) == 0
    assert "coverage gate passed" in capsys.readouterr().out
    path.write_text(json.dumps(report({"src/twin/a/x.py": (1, 100)})), encoding="utf-8")
    assert gate.main(["--json", str(path)]) == 1
    assert "FAILED" in capsys.readouterr().out
    assert gate.main(["--json", str(tmp_path / "missing.json")]) == 2


# ----------------------------------------------------------------- privacy scan
# Pattern-shaped test data is assembled at run time so this file stays clean itself.


def id_card() -> str:
    prefix = "11010519491231002"
    weights = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    return (
        prefix + "10X98765432"[sum(int(d) * w for d, w in zip(prefix, weights, strict=True)) % 11]
    )


def card_number() -> str:
    base = "411111111111111"
    return next(base + d for d in "0123456789" if scan.luhn(base + d))


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        ("contact: " + "wxid_" + "abcdef12345", "wxid"),
        ("group " + "1234567890" + "@chatroom", "wxid"),
        ("phone " + "138" + "00138000", "mobile number"),
        ("tel +86 " + "139" + "00139000", "mobile number"),
        ("id " + id_card(), "ID card number"),
        ("card " + card_number(), "bank card number (Luhn)"),
        ("key = " + "sk-" + "a1b2c3d4e5f6g7h8i9j0k1l2", "API key / private key"),
        ("-----BEGIN " + "RSA PRIVATE KEY-----", "API key / private key"),
    ],
)
def test_sensitive_patterns_are_found_without_echoing_them(line: str, kind: str) -> None:
    findings = scan.scan_text("notes.txt", "harmless\n" + line + "\n")
    assert [(f.line, f.kind) for f in findings] == [(2, kind)]
    assert "abcdef12345" not in findings[0].render()


@pytest.mark.parametrize(
    "line",
    [
        "timestamp 2026-10-09T12:00:00",
        "number 12345678901234567",  # no Luhn / ID structure
        "version 3.12.13 and build 20261009",
        "hash " + "a" * 64,
        "wxid is mentioned in prose, and so is `messages.json`",
        "phone pattern 1[3-9]\\d{9} in a spec",
        "id 110105194912310021",  # wrong check digit
    ],
)
def test_ordinary_text_is_not_flagged(line: str) -> None:
    assert scan.scan_text("doc.md", line + "\n") == []


def test_export_fragments_are_recognised() -> None:
    fragment = export_fragment()
    findings = scan.scan_text("messages.json", fragment)
    assert [f.kind for f in findings] == ["WeChat export (messages.json) fragment"]
    assert scan.scan_text("one.json", '{"isSent": true, "other": 1}') == []


def test_scan_files_skips_binary_large_and_lock_files(tmp_path: Path) -> None:
    bad = "wxid_" + "abcdef12345"
    (tmp_path / "text.txt").write_text(bad, encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"\x00" + bad.encode())
    (tmp_path / "uv.lock").write_text(bad, encoding="utf-8")
    (tmp_path / "pic.png").write_bytes(bad.encode())
    findings = list(scan.scan_files(sorted(tmp_path.iterdir()), tmp_path))
    assert [f.path for f in findings] == ["text.txt"]


def test_main_scans_named_files_and_reports_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean = tmp_path / "clean.txt"
    clean.write_text("nothing here", encoding="utf-8")
    dirty = tmp_path / "dirty.txt"
    dirty.write_text("x " + "wxid_" + "abcdef12345", encoding="utf-8")
    assert scan.main([str(clean), "--root", str(tmp_path)]) == 0
    assert scan.main([str(clean), str(dirty), "--root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "dirty.txt:1: wxid" in out and "abcdef12345" not in out


def test_the_repository_itself_is_clean() -> None:
    root = SCRIPTS.parent
    findings = list(scan.scan_files(scan.tracked_files(root), root))
    assert findings == [], [f.render() for f in findings]


def test_tracked_files_falls_back_outside_a_git_repository(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    assert [p.name for p in scan.tracked_files(tmp_path)] == ["a.txt"]
