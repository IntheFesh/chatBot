"""R18-C3/E7 (D-818): an import never moves the hold-out cut-off - the program says where it stands.

``holdout_cutoff()`` is computed once and kept by imports (R-RET-003), so every chat imported later
lands in the hold-out: not in the training set (the example library holds it since round 18,
D-880).  The user decided (D-881) that ``twin train export`` moves the cut-off itself; until then
``twin profile show`` counts the blocks now, and once the share after the cut-off is clearly above
``retrieval.holdout_ratio`` the retraining check and the import report say so and name both the
export and ``twin retrieval resplit``; the export says it moves the cut-off first.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tests.support.ingest import make_export, run_import
from tests.support.synth_chat import ChatSpec, append_texts, build_chat
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.profile.holdout import (
    DRIFT_FACTOR,
    DRIFT_MARGIN,
    HoldoutShare,
    drift_warning,
    get_holdout,
    holdout_cutoff,
    holdout_share,
    resplit_holdout,
)
from twin.services import Services, build_services

runner = CliRunner()
RESPLIT = "`twin retrieval resplit`"


def later_chat(services: Services, *, start: datetime, exchanges: int = 900) -> None:
    """Months of conversation after the first import (an incremental import would bring it)."""
    rows = []
    moment = start
    for _ in range(exchanges):
        rows.append((moment, False, "你在干嘛"))
        rows.append((moment + timedelta(seconds=30), True, "在吃饭呢"))
        moment += timedelta(hours=4)
    append_texts(services, rows)


def drifted(services: Services) -> datetime:
    build_chat(services, ChatSpec(start=date(2026, 1, 5), days=30))
    cutoff = holdout_cutoff(services)
    later_chat(services, start=datetime(2026, 2, 10, 15, 0, tzinfo=UTC))
    return cutoff


# ------------------------------------------------------------------------------ the share


def test_the_share_after_the_cutoff_is_counted_now_and_drift_is_named(services: Services) -> None:
    build_chat(services, ChatSpec(start=date(2026, 1, 5), days=30))
    assert holdout_share(services) is None  # nothing stored yet, and a read stores nothing
    assert get_holdout(services) is None
    cutoff = holdout_cutoff(services)
    fresh = holdout_share(services)
    assert fresh is not None and fresh.cutoff == cutoff
    stored = get_holdout(services)
    assert stored is not None
    assert (fresh.blocks, fresh.held_out_blocks) == (stored.blocks, stored.held_out_blocks)
    assert not fresh.drifted and drift_warning(fresh) is None
    later_chat(services, start=datetime(2026, 2, 10, 15, 0, tzinfo=UTC))
    now = holdout_share(services)
    assert now is not None and now.blocks > fresh.blocks
    assert now.share > 0.5 and now.drifted
    assert get_holdout(services) == stored  # still the stored cut-off; nothing was moved
    warning = drift_warning(now)
    assert warning is not None and RESPLIT in warning and f"{now.share:.0%}" in warning
    resplit_holdout(services, by="test")
    after = holdout_share(services)
    assert after is not None and not after.drifted


@pytest.mark.parametrize(
    ("ratio", "share", "expected"),
    [(0.10, 0.20, False), (0.10, 0.21, True), (0.05, 0.15, False), (0.05, 0.16, True)],
)
def test_drift_is_twice_the_ratio_and_at_least_ten_points_more(
    ratio: float, share: float, expected: bool
) -> None:
    assert (DRIFT_FACTOR, DRIFT_MARGIN) == (2.0, 0.10)
    found = HoldoutShare(datetime(2026, 1, 1, tzinfo=UTC), ratio, 100, round(share * 100))
    assert found.drifted is expected
    assert (drift_warning(found) is not None) is expected


# ----------------------------------------------------------------------- the commands


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


def test_profile_show_counts_the_blocks_now(data_dir: Path) -> None:
    with_services(lambda services: build_chat(services, ChatSpec(start=date(2026, 1, 5), days=30)))
    assert cli("profile", "rebuild", "--foreground").exit_code == 0
    stored = with_services(get_holdout)
    assert stored is not None
    shown = cli("profile", "show")
    assert shown.exit_code == 0, shown.output
    assert f"她的回复块现有 {stored.blocks:,} 个，切分点之后 {stored.held_out_blocks:,} 个" in (
        shown.output
    )
    assert "配置的留出比例 10%" in shown.output and "resplit" not in shown.output
    with_services(lambda s: later_chat(s, start=datetime(2026, 2, 10, 15, 0, tzinfo=UTC)))
    share = with_services(holdout_share)
    assert share is not None and share.drifted
    later = cli("profile", "show")
    assert f"她的回复块现有 {share.blocks:,} 个，切分点之后 {share.held_out_blocks:,} 个" in (
        later.output
    )
    assert f"（{share.share:.0%}）" in later.output and RESPLIT in later.output


def test_retrain_check_and_export_name_the_resplit_only_when_the_holdout_drifted(
    data_dir: Path,
) -> None:
    """Changed in round 18 (user decision, D-881): the export no longer tells the user to re-split
    first - it re-splits itself, and says so; the retraining check names both ways."""
    with_services(lambda services: build_chat(services, ChatSpec(start=date(2026, 1, 5), days=30)))
    with_services(holdout_cutoff)
    quiet = cli("train", "retrain-check")
    assert quiet.exit_code == 0 and "resplit" not in quiet.output
    with_services(lambda s: later_chat(s, start=datetime(2026, 2, 10, 15, 0, tzinfo=UTC)))
    loud = cli("train", "retrain-check")
    assert loud.exit_code == 0 and RESPLIT in loud.output
    assert "`twin train export` 会先自动" in loud.output
    queued = cli("train", "export")
    assert queued.exit_code == 0, queued.output
    assert "queued (job " in queued.output
    assert "the export first moves it to the newest 10%" in queued.output
    assert RESPLIT not in queued.output


def test_the_import_report_names_the_resplit_when_the_holdout_drifted(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=80)
    first = run_import(services, export)
    assert RESPLIT not in Path(first.report_path or "").read_text(encoding="utf-8")
    holdout_cutoff(services)
    later_chat(services, start=datetime(2026, 12, 1, 15, 0, tzinfo=UTC), exchanges=300)
    again = run_import(services, export)
    report = Path(again.report_path or "").read_text(encoding="utf-8")
    share = holdout_share(services)
    assert share is not None and share.drifted
    section = report.split("## 数据质量提示", 1)[1].split("\n## ", 1)[0]
    assert RESPLIT in section and f"{share.share:.0%}" in section
