"""``twin persona`` (R-PERS-001 to R-PERS-005, R-ARCH-006)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tests.support.synth_chat import ChatSpec, build_chat
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.ops.instance_lock import LOCK_RUN, InstanceLock
from twin.profile.holdout import get_holdout
from twin.profile.persona import compose, edit
from twin.profile.persona.sections import MANUAL, split_card
from twin.profile.persona.store import PersonaStore
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


@pytest.fixture
def ready(data_dir: Path) -> Path:
    with_services(lambda services: build_chat(services, ChatSpec(days=30)))
    done = cli("profile", "rebuild", "--foreground")
    assert done.exit_code == 0, done.output
    return data_dir


@pytest.fixture
def cards(ready: Path) -> Path:
    done = cli("persona", "regenerate", "--stats-only", "--foreground")
    assert done.exit_code == 0, done.output
    return ready


def test_without_a_card_the_commands_say_what_to_do(data_dir: Path) -> None:
    for args in (["show"], ["history"]):
        result = cli("persona", *args)
        assert result.exit_code == 1 and "persona regenerate" in result.output
    assert cli("persona", "show", "--scope", "elsewhere").exit_code == 2


def test_the_statistics_refresh_is_queued_once_and_runs_in_the_foreground(ready: Path) -> None:
    queued = cli("persona", "regenerate", "--stats-only")
    assert queued.exit_code == 0 and "queued a statistics refresh" in queued.output
    again = cli("persona", "regenerate", "--stats-only")
    assert queued.output.split("job ")[1] == again.output.split("job ")[1]  # the same job
    with_services(lambda s: None)
    done = cli("persona", "regenerate", "--stats-only", "--foreground")
    assert done.exit_code == 0, done.output
    assert "live: created" in done.output and "pre_holdout: created" in done.output
    unchanged = cli("persona", "regenerate", "--stats-only", "--foreground")
    assert "live: unchanged" in unchanged.output


def test_a_running_application_runs_the_refresh_itself(ready: Path) -> None:
    lock = InstanceLock(LOCK_RUN, locks_dir=resolve_paths(load_settings()).locks_dir)
    assert lock.acquire()
    try:
        result = cli("persona", "regenerate", "--stats-only", "--foreground")
    finally:
        lock.release()
    assert result.exit_code == 0 and "queued a statistics refresh" in result.output


def test_show_history_and_the_renderings(cards: Path) -> None:
    shown = cli("persona", "show")
    assert shown.exit_code == 0, shown.output
    for fragment in (
        "persona card v1 (live)",
        "## [自动-统计规则]",
        "## [手动]",
        "## [不要这样]",
        "full ",
        "compact ",
    ):
        assert fragment in shown.output, fragment
    past = cli("persona", "show", "--scope", "pre_holdout")
    assert "## [不要这样]" not in past.output and "(pre_holdout)" in past.output
    full = cli("persona", "show", "--full")
    assert "full rendering of live v1" in full.output and "## 数字规则" in full.output
    compact = cli("persona", "show", "--compact")
    assert "compact rendering of live v1" in compact.output and "/400 tokens" in compact.output
    history = cli("persona", "history")
    assert history.exit_code == 0 and "v1" in history.output and "pre_holdout" in history.output
    only = cli("persona", "history", "--scope", "pre_holdout")
    assert "live" not in only.output.replace("pre_holdout", "")
    assert cli("persona", "show", "v9").exit_code == 1


def test_evidence_is_listed_for_a_version_that_has_it(cards: Path) -> None:
    none = cli("persona", "show", "--evidence")
    assert "no evidence record" in none.output

    def add(services: Services) -> None:
        store = PersonaStore(services.db, services.clock)
        current = store.active("live")
        assert current is not None
        store.add_version(
            "live",
            current.text,
            reason="generate",
            provenance={
                "statements": [{"label": "口头禅", "text": "哈哈", "evidence": ["S01", "S04"]}]
            },
        )

    with_services(add)
    listed = cli("persona", "show", "--evidence")
    assert listed.exit_code == 0 and "口头禅：哈哈  <- S01, S04" in listed.output


def test_diff_and_rollback(cards: Path) -> None:
    def second(services: Services) -> None:
        store = PersonaStore(services.db, services.clock)
        current = store.active("live")
        assert current is not None
        store.add_version(
            "live",
            current.sections.with_block(
                MANUAL, compose.manual_block(["- 称呼：宝宝"], [])
            ).assemble(),
            reason="edit",
        )

    with_services(second)
    same = cli("persona", "diff", "v1", "v1")
    assert "identical" in same.output
    diff = cli("persona", "diff", "v1", "v2")
    assert diff.exit_code == 0 and "+- 称呼：宝宝" in diff.output
    back = cli("persona", "rollback", "v1")
    assert back.exit_code == 0 and "now v1" in back.output
    shown = cli("persona", "show")
    assert "in force" in shown.output and "称呼：宝宝" not in shown.output
    assert cli("persona", "rollback", "v8").exit_code == 1


def test_regenerate_prices_the_work_and_waits_for_the_approval(ready: Path) -> None:
    result = cli("persona", "regenerate")
    assert result.exit_code == 0, result.output
    assert "estimated" in result.output and "upper bound" in result.output
    assert "approve with: twin jobs approve persona-" in result.output
    listed = cli("jobs", "list", "--type", "persona_generate")
    assert listed.exit_code == 0 and listed.output.count("persona_generate") >= 2
    again = cli("persona", "regenerate")
    assert again.exit_code == 0 and "already waiting" in again.output
    single = cli("persona", "regenerate", "--scope", "pre_holdout")
    assert "already waiting" in single.output and "approve with" not in single.output
    assert cli("persona", "regenerate", "--scope", "future").exit_code == 2


def test_edit_runs_the_editor_and_reports_the_result(
    cards: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def change(path: Path) -> None:
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "### 风格\n\n### 事实", "### 风格\n- 称呼：宝宝\n\n### 事实"
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(edit, "make_editor", lambda: change)
    saved = cli("persona", "edit")
    assert saved.exit_code == 0 and "saved as version v2" in saved.output
    assert "称呼：宝宝" in cli("persona", "show").output

    def spoil(path: Path) -> None:
        path.write_text(
            path.read_text(encoding="utf-8").replace("## [自动-描述]", "## [自动-描述]\n- x"),
            encoding="utf-8",
        )

    monkeypatch.setattr(edit, "make_editor", lambda: spoil)
    refused = cli("persona", "edit")
    assert refused.exit_code == 1 and "only the [手动] section may be edited" in refused.output
    assert "v3" not in cli("persona", "history").output

    monkeypatch.setattr(edit, "make_editor", lambda: lambda path: None)
    assert "nothing changed" in cli("persona", "edit").output


def test_status_and_templates(cards: Path) -> None:
    status = cli("persona", "status")
    assert status.exit_code == 0, status.output
    assert "yes: the description was never generated" in status.output and "v1" in status.output
    templates = cli("persona", "templates")
    assert templates.exit_code == 0
    for name in ("persona_map", "persona_reduce", "sticker_tag", "sticker_context"):
        assert name in templates.output
    assert "not loaded yet" in templates.output or "v1" in templates.output


def test_the_split_of_a_stored_card_is_what_show_prints(cards: Path) -> None:
    def text(services: Services) -> str:
        found = PersonaStore(services.db, services.clock).active("live")
        assert found is not None
        return found.text

    stored = with_services(text)
    printed = cli("persona", "show").output
    assert split_card(stored).block(MANUAL).strip().splitlines()[0] in printed  # type: ignore[union-attr]


def test_status_is_read_only_and_does_not_split_the_holdout(data_dir: Path) -> None:
    with_services(lambda services: build_chat(services, ChatSpec(days=30)))
    status = cli("persona", "status")
    assert status.exit_code == 0, status.output
    assert "no hold-out split yet" in status.output
    assert with_services(lambda services: get_holdout(services)) is None


def test_the_past_card_cannot_be_planned_before_there_is_data_to_split(data_dir: Path) -> None:
    result = cli("persona", "regenerate", "--scope", "pre_holdout")
    assert result.exit_code == 1 and "her reply blocks" in result.output
    assert cli("jobs", "list", "--type", "persona_generate").output.count("persona_generate") == 0
