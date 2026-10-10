"""scripts/simulate_proactive.py: the development tool that runs the scheduler over many days."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.scripts import SCRIPTS, load_script

simulate = load_script("simulate_proactive")


def test_the_arguments_have_the_defaults_of_the_task() -> None:
    args = simulate.parse_args([])
    assert (args.days, args.seed, args.user, args.min, args.max) == (14, 1, "evening", 1, 6)
    assert args.silent == "" and args.zone is None and args.first_day == "2026-10-12"


@pytest.mark.parametrize(
    "argv",
    [["--days", "0"], ["--min", "5", "--max", "2"], ["--user", "nobody"], ["--min", "-1"]],
)
def test_a_wrong_argument_is_refused(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        simulate.parse_args(argv)
    assert stopped.value.code == 2
    assert capsys.readouterr().err


def test_a_run_prints_the_days_the_totals_and_the_audit(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    environment = {k: v for k, v in os.environ.items() if not k.startswith("TWIN_")}
    environment["PYTHONPATH"] = str(SCRIPTS.parent)
    done = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "simulate_proactive.py"),
            "--days",
            "2",
            "--user",
            "all-day",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env=environment,
        cwd=work,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "日期" in done.stdout and "日均" in done.stdout and "发送时刻分布" in done.stdout
    assert "审计结论：合规" in done.stdout
    assert "2026-10-12" in done.stdout and "2026-10-13" in done.stdout
    assert not list(work.iterdir()), "the run wrote into the working directory"
