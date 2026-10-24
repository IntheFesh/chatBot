"""The PowerShell scripts of the installation: what they run, how they fail (R-OPS-001).

PowerShell is not available where these tests run, so they check the text: it must be plain ASCII
(Windows PowerShell 5.1 reads a file without a byte order mark in the ANSI code page), strict,
stopping at the first error, and every ``twin`` command it calls must exist in this program with
the arguments the script gives it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from twin.cli import app
from twin.ops.process_model import iter_commands

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "windows"
INSTALL = SCRIPTS / "install.ps1"
UNINSTALL = SCRIPTS / "uninstall.ps1"
ARGUMENTS = re.compile(r"-Arguments\s+@\(([^)]*)\)")


def text_of(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def twin_calls(script: str) -> list[list[str]]:
    """The argument lists the script gives to ``twin.exe`` (``-File $Twin -Arguments @(...)``)."""
    calls = []
    for line in script.splitlines():
        if "$Twin" not in line:
            continue
        found = ARGUMENTS.search(line)
        if found:
            calls.append(re.findall(r"'([^']*)'", found.group(1)))
    return calls


@pytest.mark.parametrize("path", [INSTALL, UNINSTALL], ids=lambda p: p.name)
def test_the_scripts_are_plain_ascii_strict_and_stop_at_the_first_error(path: Path) -> None:
    script = text_of(path)
    script.encode("ascii")  # (no byte order mark needed, no mojibake in Windows PowerShell 5.1)
    assert "Set-StrictMode -Version Latest" in script
    assert "$ErrorActionPreference = 'Stop'" in script
    assert "[CmdletBinding()]" in script and ".SYNOPSIS" in script
    assert script.count("{") == script.count("}") and script.count("(") == script.count(")")
    assert "Invoke-Expression" not in script  # nothing is evaluated in the script's own process
    evaluated = [line for line in script.splitlines() if re.search(r"\biex\b", line)]
    assert all("https://astral.sh/uv/install.ps1" in line for line in evaluated)  # the official one


def test_install_installs_syncs_sets_up_migrates_and_registers_in_that_order() -> None:
    script = text_of(INSTALL)
    steps = [
        "Find-Uv",
        "'sync', '--frozen'",
        "@('setup')",
        "@('db', 'upgrade')",
        "@('service', 'install')",
        "@('service', 'status')",
    ]
    positions = [script.index(step) for step in steps]
    assert positions == sorted(positions)
    assert "https://astral.sh/uv/install.ps1" in script  # the official installer, nothing else
    assert "PYTHONUTF8" in script and "Win32NT" in script and "pyproject.toml" in script
    assert "[switch]$SkipSetup" in script and "[switch]$StartNow" in script


def test_install_never_syncs_without_the_lock_and_never_deletes_anything() -> None:
    script = text_of(INSTALL)
    assert "'sync', '--frozen'" in script and not re.search(r"'sync'(?!, '--frozen')", script)
    for forbidden in ("Remove-Item", "Set-ExecutionPolicy", "-Verb RunAs", "schtasks"):
        assert forbidden not in script, forbidden
    assert not re.search(r"(?<![A-Za-z-])(rm|del|rd|rmdir)\s", script)


def test_uninstall_stops_then_removes_the_task_and_leaves_the_data() -> None:
    script = text_of(UNINSTALL)
    assert script.index("@('service', 'stop')") < script.index("@('service', 'uninstall')")
    assert "Remove-Item" not in script and "purge" in script  # only a hint where data is deleted
    assert "'wechat-twin'" in script  # the fallback without the environment names the same task


@pytest.mark.parametrize("path", [INSTALL, UNINSTALL], ids=lambda p: p.name)
def test_every_command_a_script_runs_exists_in_the_program(path: Path) -> None:
    known = {name for name, _ in iter_commands(app)}
    calls = twin_calls(text_of(path))
    assert calls
    for arguments in calls:
        assert " ".join(arguments) in known, arguments


def test_the_task_name_of_the_fallback_is_the_one_the_program_registers() -> None:
    from twin.ops.taskscheduler import TASK_NAME

    assert f"'{TASK_NAME}'" in text_of(UNINSTALL)
