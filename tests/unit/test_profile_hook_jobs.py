"""The import hook, the job and the import report of the profile recomputation (R-IMP-011)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from sqlalchemy import select

from tests.support.ingest import make_export, run_import
from tests.support.synth_chat import ChatSpec, append_texts, build_chat
from twin.ingest.hooks import HookContext, load_hooks
from twin.ingest.runs import get_run
from twin.ops.jobs import HandlerRegistry, JobQueue, Worker
from twin.profile.api import load_profile
from twin.profile.builder import rebuild
from twin.profile.hook import queue_profile
from twin.profile.jobs import TIMEZONE_ALERT, handle_profile_rebuild, run_rebuild
from twin.profile.queue import PROFILE_JOB, queue_profile_rebuild
from twin.profile.report_section import (
    HEADING,
    refresh_latest_import_report,
    replace_section,
    style_change_lines,
)
from twin.services import Services
from twin.storage.models import Alert


def worker(services: Services) -> Worker:
    handlers = HandlerRegistry()
    handlers.register(PROFILE_JOB, handle_profile_rebuild)
    return Worker(
        JobQueue(services.db, services.clock),
        handlers,
        services.clock,
        services=services,
        alerts=services.alerts,
    )


def context(services: Services, *, changed: int = 0, first: bool = False) -> HookContext:
    return HookContext(
        services=services,
        run_id="run",
        conversation_id="c",
        export_id="e",
        inserted=10,
        changed=changed,
        first_import=first,
    )


def read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def pending(services: Services) -> list[dict[str, object]]:
    queue = JobQueue(services.db, services.clock)
    return [job.payload for job in queue.list_jobs(status="pending", job_type=PROFILE_JOB)]


def test_the_hook_is_registered_with_its_backfill_command() -> None:
    registry = load_hooks()
    hook = next(h for h in registry.hooks() if h.name == "profile")
    assert hook.backfill_command == "profile rebuild"
    assert "pre_holdout" in hook.description


def test_the_hook_waits_for_messages_from_her_and_for_news(services: Services) -> None:
    empty = queue_profile(context(services, first=True))
    assert empty.status == "skipped" and "no messages from her" in empty.detail
    build_chat(services, ChatSpec(days=14))
    first = queue_profile(context(services, first=True))
    assert first.status == "queued" and first.jobs == 1
    assert pending(services) == [
        {"scope": "all", "reason": "import", "run_id": "run", "force": False}
    ]
    again = queue_profile(context(services))
    assert again.status == "queued" and again.jobs == 0 and "already waiting" in again.detail
    assert len(pending(services)) == 1


def test_an_import_without_news_from_her_does_not_recompute(services: Services) -> None:
    build_chat(services, ChatSpec(days=14))
    rebuild(services, "all", reason="test")
    quiet = queue_profile(context(services))
    assert quiet.status == "skipped" and "no new messages" in quiet.detail
    assert pending(services) == []
    changed = queue_profile(context(services, changed=2))
    assert changed.status == "queued"
    last = load_profile(services, "live")
    assert last is not None
    append_texts(
        services,
        [(last.version.created_at + timedelta(days=30 + i), True, f"新消息{i}") for i in range(5)],
    )
    JobQueue(services.db, services.clock).cancel(_only_job(services))
    news = queue_profile(context(services))
    assert news.status == "queued" and news.jobs == 1


def _only_job(services: Services) -> str:
    queue = JobQueue(services.db, services.clock)
    (job,) = queue.list_jobs(status="pending", job_type=PROFILE_JOB)
    return job.id


async def test_the_job_recomputes_and_the_report_gets_its_section(
    services: Services, tmp_path: Path
) -> None:
    outcome = run_import(
        services, make_export(tmp_path, target_messages=300, other_conversations=0)
    )
    assert outcome.run.hooks["profile"]["status"] == "queued"
    assert outcome.report_path is not None
    report = read(outcome.report_path)
    assert HEADING in report and "画像还没有计算" in report
    assert pending(services)[0]["run_id"] == outcome.run.id
    summary = await worker(services).run_until_idle()
    assert summary.done == 1 and summary.failed == 0
    finished = read(outcome.report_path)
    assert "首次画像" in finished and "画像还没有计算" not in finished
    assert finished.count(HEADING) == 1 and "## 导入后钩子" in finished
    assert load_profile(services, "live") is not None
    run = get_run(services.db, outcome.run.id)
    assert run is not None and run.status == "done"


def test_the_report_lists_the_changes_of_a_new_version(services: Services, tmp_path: Path) -> None:
    outcome = run_import(
        services, make_export(tmp_path, target_messages=300, other_conversations=0)
    )
    assert outcome.report_path is not None and outcome.run.finished_at is not None
    run_rebuild(services, "all", "import", False)
    first = load_profile(services, "live")
    assert first is not None and first.metrics.window_info("full")["her_messages"] > 0
    later = outcome.run.finished_at + timedelta(days=2)
    append_texts(
        services,
        [(later + timedelta(minutes=2 * i), True, "好，嗯，行，对，是") for i in range(250)],
    )
    run_rebuild(services, "live", "import", False)
    path = refresh_latest_import_report(services)
    assert path == Path(outcome.report_path)
    text = path.read_text(encoding="utf-8")
    assert "相对上一版变化超过 10% 的指标" in text and "她 " in text
    assert text.count(HEADING) == 1 and "## 导入后钩子" in text.split(HEADING)[1]


def test_the_section_lines_for_each_state(services: Services) -> None:
    assert "还没有计算" in style_change_lines(services)[0]
    build_chat(services, ChatSpec(days=14))
    rebuild(services, "live", reason="test")
    assert "首次画像" in style_change_lines(services)[0]
    rebuild(services, "live", reason="test", force=True)
    assert "没有变化超过 10%" in style_change_lines(services)[0]


def test_replacing_a_section_keeps_the_rest_of_the_report() -> None:
    text = "# 报告\n\n## 甲\n\n内容\n\n## 风格变化\n\n旧的\n\n## 乙\n\n后面\n"
    assert replace_section(text, ["- 新的"]) == (
        "# 报告\n\n## 甲\n\n内容\n\n## 风格变化\n\n- 新的\n\n## 乙\n\n后面\n"
    )
    last = "# 报告\n\n## 风格变化\n\n旧的\n"
    assert replace_section(last, ["- 新的"]).endswith("## 风格变化\n\n- 新的\n\n")
    assert replace_section("# 报告\n", ["- 新的"]) == "# 报告\n\n## 风格变化\n\n- 新的\n"


def test_no_report_means_nothing_to_refresh(services: Services) -> None:
    assert refresh_latest_import_report(services) is None


async def test_a_queued_job_runs_through_the_worker(services: Services) -> None:
    build_chat(services, ChatSpec(days=14))
    queued = queue_profile_rebuild(services, scope="live", reason="manual")
    summary = await worker(services).run_until_idle()
    assert summary.done == 1
    job = JobQueue(services.db, services.clock).get(queued.job_id)
    assert job is not None and job.status == "done"
    assert load_profile(services, "live") is not None
    assert load_profile(services, "pre_holdout") is None


def alerts(services: Services) -> list[tuple[str, str | None]]:
    with services.db.session() as session:
        return [(a.category, a.dedup_key) for a in session.scalars(select(Alert))]


def test_a_daytime_sleep_raises_an_alert_without_chat_content(services: Services) -> None:
    build_chat(services, ChatSpec(zone="Asia/Shanghai", days=30))
    run_rebuild(services, "live", "manual", False)
    assert alerts(services) == [(TIMEZONE_ALERT, f"{TIMEZONE_ALERT}-live")]
    with services.db.session() as session:
        detail = session.scalars(select(Alert)).one().detail
    assert detail is not None and "source_timezone" in detail["warnings"][0]
    assert set(detail) == {"scope", "warnings"}


def test_a_plausible_sleep_raises_no_alert(services: Services) -> None:
    build_chat(services, ChatSpec(days=30))
    run_rebuild(services, "all", "manual", False)
    assert alerts(services) == []
