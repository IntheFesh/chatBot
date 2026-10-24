"""``twin eval consistency`` through the real CLI (R-EVAL-004).

The audit runs against a database in the test's data directory; DeepSeek is a scripted ``respx``
route; the screen reads its keys from a string and writes to a string.  The contract: the user
decides each contradiction, a correction of the memory happens only on his ``y``, and the exit code
says whether the week passed (at most one obvious contradiction).
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest
import respx
from rich.console import Console
from typer.testing import CliRunner

from tests.support.clock import ManualClock
from tests.support.consistency_world import (
    LIBRARY,
    SHANGHAI,
    AuditModel,
    Week,
    build_week,
    the_two_contradictions,
)
from tests.support.deepseek import API
from tests.support.embedding import HashingBackend
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.eval.cli import use_interaction
from twin.eval.consistency_store import ConsistencyStore
from twin.eval.store import EvalStore
from twin.eval.ui import LineKeys
from twin.ops.jobs import JobQueue
from twin.services import Services, build_services

runner = CliRunner()


@pytest.fixture
def data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedder: HashingBackend, clock: ManualClock
) -> Path:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    monkeypatch.setenv("TWIN_RETRIEVAL__MODEL", embedder.info.model)
    assert runner.invoke(app, ["db", "upgrade"]).exit_code == 0
    return path


def open_services() -> Services:
    settings = load_settings()
    return build_services(settings, root=resolve_paths(settings).root)


@pytest.fixture
def week(data_dir: Path) -> Iterator[Week]:
    services = open_services()
    try:
        yield build_week(services)
    finally:
        services.close()


@pytest.fixture
def model() -> AuditModel:
    return AuditModel(rules=the_two_contradictions())


@pytest.fixture
def api(model: AuditModel) -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        router.post(API).mock(side_effect=model)
        yield router


def run(*args: str, keys: str = "") -> tuple[int, str, str]:
    """Invoke ``twin eval ...``; returns ``(exit code, what the command printed, the screen)``."""
    screen = io.StringIO()
    console = Console(file=screen, width=140, color_system=None, highlight=False)
    with use_interaction(console, LineKeys(io.StringIO(keys))):
        result = runner.invoke(app, ["eval", *args])
    return result.exit_code, result.output, screen.getvalue()


def stores(week: Week) -> tuple[EvalStore, ConsistencyStore]:
    services = week.memory.services
    return EvalStore(services.db, services.clock), ConsistencyStore(services.db, services.clock)


def run_id_of(screen: str) -> str:
    return screen.split("一致性审计 ", 1)[1].split("：", 1)[0].strip()


def library_is_active(week: Week) -> bool:
    entry = week.memory.store.event(week.library.id)
    return entry is not None and entry.active


def invented_fact_status(week: Week) -> str:
    fact = week.memory.store.fact(week.invented_fact.id)
    assert fact is not None
    return fact.status if fact.superseded_by is None else "superseded"


# ------------------------------------------------------------------- the whole flow


def test_the_user_decides_each_contradiction_and_each_correction(
    week: Week, api: respx.MockRouter, model: AuditModel
) -> None:
    # 1st: real and obvious, then apply the correction; 2nd: real but minor, then keep the memory
    code, out, screen = run("consistency", "--days", "7", keys="y\ny\nm\nn\n")
    assert code == 0, out + screen
    assert "生活安排 3 条、她说过的话 3 段、相关事实" in screen
    assert "等你决定 2 条" in screen and "第 1/2 条" in screen and "第 2/2 条" in screen
    assert "生活安排：" in screen and LIBRARY in screen and "她说过的话" in screen
    assert "记忆里的事实" in screen and "真实聊天记录" in screen and "机器人自己说的" in screen
    assert "理由：两处说法不可能同时为真" in screen
    assert "修正建议 1/1：把这条生活安排标记失效" in screen
    assert "修正建议 1/1：把这条事实改写" in screen and "新：她在北京工作" in screen
    assert "已应用" in screen
    assert "本次：明显矛盾 1，不明显的矛盾 1，不是矛盾 0" in screen
    assert "应用 1" in screen and "不改 1" in screen
    assert "确认的明显矛盾 1 次" in screen and "一致性：通过" in screen
    estore, cstore = stores(week)
    week.memory.refresh()
    assert not library_is_active(week)  # applied
    assert invented_fact_status(week) == "active"  # declined: the memory is as it was
    run_row = estore.list_runs("consistency")[0]
    assert run_row.status == "done" and run_row.verdict == "passed"
    assert [(f.status, f.severity) for f in cstore.findings(run_row.id)] == [
        ("confirmed", "obvious"),
        ("confirmed", "minor"),
    ]
    assert [x.status for x in cstore.run_fixes(run_row.id)] == ["applied", "declined"]
    assert len(model.requests) == 1


def test_nothing_in_the_memory_changes_without_a_yes(week: Week, api: respx.MockRouter) -> None:
    facts_before = {f.id: f for f in week.memory.store.all_facts()}
    code, _, screen = run("consistency", keys="y\nn\ny\nn\n")
    assert code == 1  # two obvious contradictions in a week: not passed
    assert "确认的明显矛盾 2 次" in screen and "一致性：未通过" in screen
    week.memory.refresh()
    assert library_is_active(week)
    assert {f.id: f for f in week.memory.store.all_facts()} == facts_before
    estore, cstore = stores(week)
    run_row = estore.list_runs("consistency")[0]
    assert {fix.status for fix in cstore.run_fixes(run_row.id)} == {"declined"}
    assert run_row.status == "done" and run_row.verdict == "failed"


def test_contradictions_that_are_not_contradictions_change_nothing(
    week: Week, api: respx.MockRouter
) -> None:
    code, _, screen = run("consistency", keys="n\nn\n")
    assert code == 0
    assert "不是矛盾 2" in screen and "修正建议" not in screen and "一致性：通过" in screen
    estore, cstore = stores(week)
    found = cstore.findings(estore.list_runs("consistency")[0].id)
    assert [f.status for f in found] == ["rejected", "rejected"]
    assert cstore.run_fixes(found[0].run_id) == []
    assert library_is_active(week)


def test_leaving_in_the_middle_keeps_what_was_decided_and_goes_on_later(
    week: Week, api: respx.MockRouter, model: AuditModel
) -> None:
    code, _, screen = run("consistency", keys="y\nq\n")
    assert code == 0 and "没做完：twin eval consistency --resume" in screen
    estore, cstore = stores(week)
    run_row = estore.list_runs("consistency")[0]
    assert run_row.status == "running" and run_row.verdict is None
    assert [f.status for f in cstore.findings(run_row.id)] == ["confirmed", "proposed"]
    assert [x.status for x in cstore.run_fixes(run_row.id)] == ["proposed"]
    # later: the correction that was left comes first, then the contradiction that was not decided
    code, _, screen = run("consistency", "--review", keys="n\nm\nn\n")
    assert code == 0 and len(model.requests) == 1  # DeepSeek is not asked again
    assert screen.index("修正建议 1/1：把这条生活安排标记失效") < screen.index("第 1/1 条")
    assert "一致性：通过" in screen
    closed = estore.get_run(run_row.id)
    assert closed.status == "done" and closed.verdict == "passed"
    assert library_is_active(week)  # he said no to it


def test_resume_names_the_run_and_a_finished_run_is_only_reported(
    week: Week, api: respx.MockRouter
) -> None:
    _, _, screen = run("consistency", keys="n\nn\n")
    run_id = run_id_of(screen)
    code, _, again = run("consistency", "--resume", run_id)
    assert code == 0 and "第 1/" not in again and "一致性：通过" in again
    code, out, _ = run("consistency", "--resume", "nope")
    assert code == 1 and "there is no evaluation run" in out
    other = EvalStore(week.memory.services.db, week.memory.services.clock).create_run("blind")
    code, out, _ = run("consistency", "--resume", other.id)
    assert code == 1 and "not a consistency run" in out


def test_review_without_an_audit_that_waits_says_so(week: Week) -> None:
    code, _, screen = run("consistency", "--review")
    assert code == 0 and "没有等你决定的审计" in screen


def test_a_skipped_contradiction_stays_undecided_and_the_week_is_not_passed(
    week: Week, api: respx.MockRouter
) -> None:
    code, _, screen = run("consistency", keys="s\nn\n")
    assert code == 1 and "还没决定 1 次" in screen and "先跳过 1" in screen
    assert "一致性：待确认" in screen
    assert stores(week)[0].list_runs("consistency")[0].status == "running"


def test_the_end_of_the_input_is_like_leaving(week: Week, api: respx.MockRouter) -> None:
    code, _, screen = run("consistency", keys="")
    assert code == 0 and "没做完" in screen


def test_a_window_shorter_than_a_week_can_never_pass(week: Week, api: respx.MockRouter) -> None:
    code, _, screen = run("consistency", "--days", "3", keys="n\nn\n")
    assert code == 1 and "不足一周" in screen and "未通过（样本不足）" in screen


def test_a_week_with_nothing_in_it_says_so_and_is_not_a_pass(
    data_dir: Path, api: respx.MockRouter, model: AuditModel
) -> None:
    code, _, screen = run("consistency")
    assert code == 1 and "没有可审阅的内容" in screen and model.requests == []


def test_an_answer_that_cannot_be_used_is_reported_with_a_failed_exit(
    week: Week, api: respx.MockRouter, model: AuditModel
) -> None:
    model.replies = ["不是 JSON", "还是不是"]
    code, out, _ = run("consistency")
    assert code == 1 and "the audit could not be made" in out
    run_row = stores(week)[0].list_runs("consistency")[0]
    assert run_row.status == "failed"


def test_without_the_key_the_command_says_which_secret_to_set(
    week: Week, api: respx.MockRouter, model: AuditModel
) -> None:
    services = week.memory.services
    services.secrets.delete("deepseek_api_key")
    code, out, _ = run("consistency")
    assert code != 0 and "deepseek_api_key" in out and model.requests == []


# --------------------------------------------------------------------------- queue


def test_the_audit_can_be_queued_for_the_off_peak_hours(week: Week) -> None:
    code, _, screen = run("consistency", "--queue", "--days", "10")
    assert code == 0 and "已排队" in screen
    services = week.memory.services
    jobs = JobQueue(services.db, services.clock).list_jobs(job_type="eval_consistency")
    assert len(jobs) == 1 and jobs[0].offpeak_only and jobs[0].payload == {"days": 10}
    assert jobs[0].deadline is not None
    waits = jobs[0].deadline - jobs[0].created_at  # forced after a day
    assert timedelta(days=1) - timedelta(seconds=5) < waits <= timedelta(days=1)
    code, _, screen = run("consistency", "--queue")
    assert code == 0 and "已经有一次审计在排队" in screen
    assert len(JobQueue(services.db, services.clock).list_jobs(job_type="eval_consistency")) == 1


def test_runs_lists_the_audit_by_its_kind(week: Week, api: respx.MockRouter) -> None:
    run("consistency", keys="n\nn\n")
    code, _, screen = run("runs", "--kind", "consistency")
    assert code == 0 and "consistency" in screen and "passed" in screen
    assert SHANGHAI not in screen and LIBRARY not in screen  # the list shows no chat text
