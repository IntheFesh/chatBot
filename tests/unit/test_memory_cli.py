"""``twin memory``: replay estimate, start and status, list, remember, forget, summarize, reindex
(R-MEM-009, R-MEM-010, R-IMP-011, R-LLM-014)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tests.support.embedding import HashingBackend
from tests.support.memory import add_fact, add_followup
from tests.support.synth_chat import MessageWriter
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.memory.memory import Memory
from twin.ops.process_model import CommandKind, ExitCode, get_spec, iter_commands
from twin.services import Services, build_services

runner = CliRunner()
BASE = datetime(2026, 3, 5, 15, 0, tzinfo=UTC)  # 09:00 in Chicago


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedder: HashingBackend) -> Path:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    monkeypatch.setenv("TWIN_RETRIEVAL__MODEL", embedder.info.model)
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


def history(services: Services) -> None:
    writer = MessageWriter(services)
    for day in (5, 6, 9):
        at = BASE + timedelta(days=day - 5)
        writer.add(at, False, "text", f"{day}号我要去考试")
        writer.add(at + timedelta(minutes=1), True, "text", "加油呀")
    writer.store()


def facts_of(services: Services) -> list[Any]:
    return Memory(services).store.all_facts()


# ----------------------------------------------------------------- process model


def test_every_memory_command_declares_what_kind_of_command_it_is() -> None:
    kinds = {
        name: get_spec(command).kind
        for name, command in iter_commands(app)
        if name.startswith("memory")
    }
    assert kinds == {
        "memory replay estimate": CommandKind.READ,
        "memory replay start": CommandKind.HEAVY,
        "memory replay status": CommandKind.READ,
        "memory list": CommandKind.READ,
        "memory block": CommandKind.READ,
        "memory remember": CommandKind.LIGHT,
        "memory forget": CommandKind.LIGHT,
        "memory summarize": CommandKind.HEAVY,
        "memory reindex": CommandKind.LIGHT,
    }


# ------------------------------------------------------------------------ replay


def test_the_estimate_of_an_empty_history_says_there_is_nothing_to_do(data_dir: Path) -> None:
    result = cli("memory", "replay", "estimate")
    assert result.exit_code == 0, result.output
    assert "nothing to replay" in result.output and "days to replay" in result.output


def test_the_estimate_lists_days_and_money_and_changes_nothing(data_dir: Path) -> None:
    with_services(history)
    result = cli("memory", "replay", "estimate")
    assert result.exit_code == 0, result.output
    assert "days to replay" in result.output and "3" in result.output
    assert "estimated cost" in result.output and "2026-03-05 / 2026-03-09" in result.output
    assert "twin memory replay start" in result.output
    ranged = cli("memory", "replay", "estimate", "--from", "2026-03-06", "--to", "2026-03-06")
    assert "2026-03-06 / 2026-03-06" in ranged.output
    assert with_services(lambda s: Memory(s).store.replay_days()) == {}
    assert cli("memory", "replay", "status").exit_code == 0


def test_dates_are_checked(data_dir: Path) -> None:
    bad = cli("memory", "replay", "estimate", "--from", "March 5")
    assert bad.exit_code == int(ExitCode.USAGE) and "YYYY-MM-DD" in bad.output.replace(
        "like 2026-03-05", "YYYY-MM-DD"
    )
    backwards = cli("memory", "replay", "start", "--from", "2026-03-09", "--to", "2026-03-05")
    assert (
        backwards.exit_code == int(ExitCode.USAGE) and "--to is before --from" in backwards.output
    )


def test_start_queues_a_batch_that_waits_for_approval_and_status_reports_it(
    data_dir: Path,
) -> None:
    with_services(history)
    started = cli("memory", "replay", "start")
    assert started.exit_code == 0, started.output
    assert (
        "estimated $" in started.output
        and "approve with: twin jobs approve memory-" in started.output
    )
    assert "twin jobs run --until-idle" in started.output
    batch = next(
        line.split("twin jobs approve ")[1].strip()
        for line in started.output.splitlines()
        if line.startswith("approve with:")
    )
    status = cli("memory", "replay", "status")
    assert status.exit_code == 0, status.output
    assert f"batch {batch}: waiting for approval" in status.output
    assert "days waiting" in status.output and "pending 1" in status.output
    again = cli("memory", "replay", "start")
    assert "3 day(s) are already waiting in the queue" in again.output
    assert "nothing new to queue" in again.output
    approved = cli("jobs", "approve", batch, "--yes")
    assert approved.exit_code == 0, approved.output
    assert f"batch {batch}: approved" in cli("memory", "replay", "status").output


# -------------------------------------------------------------- list / remember / forget


def test_the_block_command_shows_what_a_reply_about_a_topic_would_remember(
    data_dir: Path,
) -> None:
    empty = cli("memory", "block", "火锅")
    assert empty.exit_code == 0, empty.output
    assert "nothing to say" in empty.output and "0 item(s)" in empty.output

    def fill(services: Services) -> None:
        memory = Memory(services)
        add_fact(memory, "她很喜欢吃火锅", BASE)
        add_fact(memory, "她的猫叫豆包", BASE + timedelta(days=30))

    with_services(fill)
    now = cli("memory", "block", "她很喜欢吃火锅")
    assert now.exit_code == 0, now.output
    assert "她很喜欢吃火锅" in now.output and "item(s)" in now.output and "tokens" in now.output
    before = cli("memory", "block", "她很喜欢吃火锅", "--at", "2026-03-05T08:00:00-06:00")
    assert (
        "她很喜欢吃火锅" not in before.output and "nothing to say" in before.output
    )  # not yet known
    small = cli("memory", "block", "她很喜欢吃火锅", "--budget", "0")
    assert small.exit_code == 0 and "/0 tokens" in small.output
    naive = cli("memory", "block", "火锅", "--at", "2026-03-05T08:00:00")
    assert naive.exit_code == int(ExitCode.USAGE) and "UTC offset" in naive.output
    junk = cli("memory", "block", "火锅", "--at", "yesterday")
    assert junk.exit_code == int(ExitCode.USAGE) and "--at must look like" in junk.output


def test_remembering_without_an_api_key_stores_the_words_as_they_are(data_dir: Path) -> None:
    result = cli("memory", "remember", "她不吃香菜")
    assert result.exit_code == 0, result.output
    assert "remembered as number 1" in result.output and "stored as written" in result.output
    (fact,) = with_services(facts_of)
    assert (fact.text, fact.source, fact.importance, fact.confidence) == (
        "她不吃香菜",
        "user_command",
        4,
        1.0,
    )
    empty = cli("memory", "remember", "   ")
    assert empty.exit_code == int(ExitCode.USAGE) and "nothing to remember" in empty.output


def test_the_list_shows_numbered_facts_pages_and_open_follow_ups(data_dir: Path) -> None:
    assert "nothing remembered yet" in cli("memory", "list").output

    def fill(services: Services) -> None:
        memory = Memory(services)
        for n in range(12):
            add_fact(memory, f"第{n}条事实内容{n}", BASE + timedelta(minutes=n))
        add_fact(
            memory, "她的生日", BASE, subject="her", category="anniversary",
            event_date=date(2000, 5, 1), recurrence="yearly",
        )  # fmt: skip
        add_followup(memory, "她周三面试", BASE + timedelta(days=30), BASE)

    with_services(fill)
    first = cli("memory", "list")
    assert first.exit_code == 0, first.output
    assert "page 1/2, 13 fact(s)" in first.output and "待跟进：她周三面试" in first.output
    assert "[2000-05-01]" in first.output and "(real_record)" in first.output
    second = cli("memory", "list", "--page", "2")
    assert "page 2/2" in second.output
    found = cli("memory", "list", "--keyword", "生日")
    assert "她的生日" in found.output and "第3条" not in found.output
    assert "no fact contains that word" in cli("memory", "list", "--keyword", "不存在的词").output


def test_forget_deletes_by_number_asks_when_ambiguous_and_deletes_all_on_request(
    data_dir: Path,
) -> None:
    def fill(services: Services) -> None:
        memory = Memory(services)
        add_fact(memory, "她养了一只猫叫豆包", BASE)
        add_fact(memory, "她养了一只猫叫年糕", BASE + timedelta(minutes=1))
        add_fact(memory, "她喜欢吃火锅", BASE + timedelta(minutes=2))

    with_services(fill)
    ambiguous = cli("memory", "forget", "养了一只猫")
    assert (
        ambiguous.exit_code == 0
        and "matches several facts; nothing was deleted" in ambiguous.output
    )
    assert len(with_services(facts_of)) == 3
    single = cli("memory", "forget", "3")
    assert single.exit_code == 0 and "deleted fact (number 3): 她喜欢吃火锅" in single.output
    both = cli("memory", "forget", "养了一只猫", "--all")
    assert both.output.count("deleted fact") == 2
    assert with_services(facts_of) == []
    nothing = cli("memory", "forget", "不存在的事")
    assert nothing.exit_code == int(ExitCode.FAILURE) and "nothing matches that" in nothing.output
    blank = cli("memory", "forget", "  ")
    assert blank.exit_code == int(ExitCode.USAGE)


def test_forgetting_a_follow_up_by_its_words(data_dir: Path) -> None:
    def fill(services: Services) -> None:
        memory = Memory(services)
        add_followup(memory, "她周三面试", BASE + timedelta(days=30), BASE)
        add_followup(memory, "她周五面试", BASE + timedelta(days=32), BASE)

    with_services(fill)
    several = cli("memory", "forget", "面试")
    assert "matches several follow-ups; use --all" in several.output
    gone = cli("memory", "forget", "面试", "--all")
    assert gone.exit_code == 0 and gone.output.count("deleted") == 2


# ------------------------------------------------------------------ summarize / reindex


def test_summarize_queues_the_jobs_once(data_dir: Path) -> None:
    first = cli("memory", "summarize", "2026-03-05")
    assert first.exit_code == 0 and "queued 2 summary job(s)" in first.output
    assert "already queued" in cli("memory", "summarize", "2026-03-05").output
    one = cli("memory", "summarize", "2026-03-06", "--scope", "real", "--force")
    assert "queued 1 summary job(s)" in one.output
    assert cli("memory", "summarize", "yesterday").exit_code == int(ExitCode.USAGE)
    wrong = cli("memory", "summarize", "2026-03-05", "--scope", "everyone")
    assert wrong.exit_code == int(ExitCode.USAGE) and "--scope must be" in wrong.output


def test_reindex_encodes_what_has_no_vector(data_dir: Path) -> None:
    def fill(services: Services) -> None:
        memory = Memory(services)
        add_fact(memory, "她养了一只猫", BASE, embed=False)
        add_fact(memory, "她喜欢吃火锅", BASE + timedelta(minutes=1), embed=False)

    with_services(fill)
    result = cli("memory", "reindex")
    assert result.exit_code == 0, result.output
    assert "encoded 2 record(s)" in result.output
    assert "encoded" in cli("memory", "reindex").output
