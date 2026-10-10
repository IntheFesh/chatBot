"""``twin persona rules``: view the rules of ``[不要这样]``, delete one, consolidate (R-LRN-003)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from tests.support.clock import ManualClock
from tests.support.deepseek import API, TEST_KEY, ok, request_json
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.engine.feedback import FeedbackStore
from twin.engine.turns import BotTurnStore, OutboundBubble, ReplyMeta
from twin.learning.jobs import RULES_JOB
from twin.llm.runtime import DEEPSEEK_SECRET
from twin.ops.jobs import JobQueue
from twin.ops.process_model import ExitCode
from twin.profile.persona import compose
from twin.profile.persona.api import read_corrections
from twin.profile.persona.refresh import persona_store
from twin.profile.persona.sections import AUTO, MANUAL, STATS
from twin.services import Services, build_services

runner = CliRunner()


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: ManualClock) -> Path:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    # The CLI runs on the real clock and the consolidation is an off-peak job: without the
    # discount there is nothing to wait for, so the test does not depend on the time of day.
    monkeypatch.setenv("TWIN_PRICING__OFFPEAK_MULTIPLIER", "1.0")
    result = runner.invoke(app, ["db", "upgrade"])
    assert result.exit_code == 0, result.output
    return path


def with_services[T](work: Callable[[Services], T]) -> T:
    settings = load_settings()
    services = build_services(settings, root=resolve_paths(settings).root)
    try:
        return work(services)
    finally:
        services.close()


def put_card(rules: list[str]) -> None:
    def write(services: Services) -> None:
        text = (
            compose.stats_block(["几乎不用句号"])
            + compose.auto_block("### 风格\n- 爱说“哈哈”")
            + compose.manual_block(["我手写的风格行"], ["我手写的事实行"])
            + compose.dont_block(rules)
        )
        persona_store(services).add_version("live", text, reason="test")

    with_services(write)


def rules_now() -> list[str]:
    return with_services(read_corrections)


def test_the_rules_are_listed_with_numbers_by_the_group_and_by_list(data_dir: Path) -> None:
    empty = runner.invoke(app, ["persona", "rules"])
    assert empty.exit_code == 0 and "no rules yet" in empty.output
    put_card(["不要太客气", "少用句号结尾"])
    for arguments in (["persona", "rules"], ["persona", "rules", "list"]):
        result = runner.invoke(app, arguments)
        assert result.exit_code == 0, result.output
        assert "1. 不要太客气" in result.output and "2. 少用句号结尾" in result.output
        assert "2 of at most 30 rule(s)" in result.output


def test_one_rule_is_deleted_by_its_number_as_a_new_version_that_keeps_the_rest(
    data_dir: Path,
) -> None:
    put_card(["不要太客气", "少用句号结尾", "多用口语"])
    before = with_services(lambda s: persona_store(s).active("live"))
    result = runner.invoke(app, ["persona", "rules", "delete", "2"])
    assert result.exit_code == 0, result.output
    assert "deleted rule 2: 少用句号结尾" in result.output
    assert rules_now() == ["不要太客气", "多用口语"]
    after = with_services(lambda s: persona_store(s).active("live"))
    assert before is not None and after is not None and after.number == before.number + 1
    assert after.reason == "rule_deleted"
    for section in (STATS, AUTO, MANUAL):
        assert after.sections.block(section) == before.sections.block(section)


@pytest.mark.parametrize("number", ["0", "4", "-1", "x"])
def test_a_rule_that_is_not_there_is_refused_and_changes_nothing(
    data_dir: Path, number: str
) -> None:
    put_card(["不要太客气", "少用句号结尾", "多用口语"])
    result = runner.invoke(app, ["persona", "rules", "delete", number])
    assert result.exit_code == ExitCode.USAGE
    assert rules_now() == ["不要太客气", "少用句号结尾", "多用口语"]


def seed_feedback(services: Services) -> None:
    store = BotTurnStore(services.db, services.clock)
    store.add_inbound(at=services.clock.now_utc(), kind="text", text="你好", external_id="m1")
    rows = store.add_reply(
        [OutboundBubble("您好，有什么可以帮您", services.clock.now_utc())], ReplyMeta("deepseek")
    )
    FeedbackStore(services.db, services.clock).add(
        "not_like", rows[0].reply_id or "", bot_turn_id=rows[0].id
    )
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)


def test_consolidate_queues_the_job_once_and_can_run_it_in_the_foreground(
    data_dir: Path,
) -> None:
    with_services(seed_feedback)
    queued = runner.invoke(app, ["persona", "rules", "consolidate"])
    assert queued.exit_code == 0 and "queued the consolidation (job" in queued.output
    again = runner.invoke(app, ["persona", "rules", "consolidate"])
    assert "already waiting" in again.output

    def jobs(services: Services) -> list[str]:
        return [
            j.status for j in JobQueue(services.db, services.clock).list_jobs(job_type=RULES_JOB)
        ]

    assert with_services(jobs) == ["pending"]

    def answer(request: httpx.Request) -> httpx.Response:
        system = request_json(request)["messages"][0]["content"]
        if "整理一个聊天机器人" in system:
            return ok(content=json.dumps({"rules": ["不要用客服腔"]}, ensure_ascii=False))
        return ok(content=json.dumps({"verdicts": [{"index": 1, "ok": True, "kind": "style"}]}))

    with respx.mock(assert_all_called=False) as router:
        router.post(API).mock(side_effect=answer)
        done = runner.invoke(app, ["persona", "rules", "consolidate", "--foreground"])
    assert done.exit_code == 0, done.output
    assert rules_now() == ["不要用客服腔"] and with_services(jobs) == ["done"]


def test_a_consolidation_that_cannot_finish_says_so(data_dir: Path) -> None:
    with_services(seed_feedback)
    from tests.support.deepseek import error

    with respx.mock(assert_all_called=False) as router:
        router.post(API).mock(return_value=error(400, "not today"))
        result = runner.invoke(app, ["persona", "rules", "consolidate", "--foreground"])
    assert result.exit_code != 0 and "did not finish" in result.output
    assert rules_now() == []
