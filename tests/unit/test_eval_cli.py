"""``twin eval blind | style | memory | gate | runs`` through the real CLI (R-EVAL-001..003, 010).

The commands run against a database in the test's data directory; the conversation is synthetic,
DeepSeek is a scripted ``respx`` route, and the screens read their keys from a string and write to
a string.  The exit codes are the contract: gates exit 0 / 1 / 2.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import respx
from rich.console import Console
from typer.testing import CliRunner

from tests.support.embedding import HashingBackend
from tests.support.eval_world import (
    DeepSeekScript,
    add_memory_facts,
    build_eval_world,
    memory_script,
    mock_deepseek,
    serve,
)
from tests.support.export_world import World
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.eval.cli import use_interaction
from twin.eval.store import EvalStore, NewItem
from twin.eval.ui import LineKeys
from twin.services import Services, build_services

runner = CliRunner()
KNOWN = datetime(2026, 9, 3, 12, tzinfo=UTC)


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    monkeypatch.setenv("TWIN_RETRIEVAL__MODEL", "test/hashing-bigram")
    assert runner.invoke(app, ["db", "upgrade"]).exit_code == 0
    return path


def open_services() -> Services:
    settings = load_settings()
    return build_services(settings, root=resolve_paths(settings).root)


@pytest.fixture
def cli_world(data_dir: Path, embedder: HashingBackend) -> Iterator[World]:
    services = open_services()
    try:
        yield build_eval_world(services, embedder, days=30)
    finally:
        services.close()


@pytest.fixture
def script() -> DeepSeekScript:
    return DeepSeekScript(lambda body: "好呀\n哈哈[拥抱]")


@pytest.fixture
def api(script: DeepSeekScript) -> Iterator[respx.MockRouter]:
    yield from mock_deepseek(script)


def run(*args: str, keys: str = "") -> tuple[int, str, str]:
    """Invoke ``twin eval ...``; returns ``(exit code, what the command printed, the screen)``."""
    screen = io.StringIO()
    console = Console(file=screen, width=120, color_system=None, highlight=False)
    with use_interaction(console, LineKeys(io.StringIO(keys))):
        result = runner.invoke(app, ["eval", *args])
    return result.exit_code, result.output, screen.getvalue()


def run_id_of(screen: str) -> str:
    return screen.split("盲测 ", 1)[1].split("：", 1)[0].strip()


def batch_of(screen: str) -> str:
    return screen.split("批次 ", 1)[1].split("：", 1)[0].strip()


def test_the_blind_test_is_planned_approved_generated_judged_and_reported(
    cli_world: World, api: respx.MockRouter, script: DeepSeekScript
) -> None:
    code, _, screen = run("blind", "--n", "8", "--seed", "3")
    assert code == 0 and "盲测" in screen and "8 个上下文" in screen
    assert "twin jobs approve" in screen and "--resume" in screen and "估算" in screen
    assert script.requests == []  # planning costs nothing
    run_id, batch = run_id_of(screen), batch_of(screen)
    # before the approval nothing can be generated, and the command says what to do
    code, _, screen = run("blind", "--resume", run_id, "--foreground")
    assert code == 0 and "还有 8 对没有生成" in screen and f"twin jobs approve {batch}" in screen
    assert script.requests == []
    approved = runner.invoke(app, ["jobs", "approve", batch, "--yes"])
    assert approved.exit_code == 0, approved.output
    # approved: the jobs run here, then the screen asks which one is hers
    code, _, screen = run("blind", "--resume", run_id, "--foreground", keys="1\n2\ns\nq\n")
    assert code == 0 and len(script.requests) == 8
    assert (
        "最近的对话" in screen
        and "哪条是她" in screen
        and "deepseek" not in screen.split("盲测")[0]
    )
    assert "本次判了 2 对，跳过 1 对" in screen
    assert "有效判断" in screen and "95% 区间 (Wilson)" in screen and "没做完" in screen
    store = EvalStore(cli_world.services.db, cli_world.services.clock)
    statuses = sorted(i.status for i in store.items(run_id))
    assert statuses == ["generated"] * 5 + ["judged"] * 2 + ["skipped"]
    assert store.get_run(run_id).status == "running"
    # continuing asks only for the five that are left; the run is then done
    code, _, screen = run("blind", "--resume", run_id, keys="1\n1\n1\n1\n1\n")
    assert code == 0 and "第 1/5 对" in screen and "没做完" not in screen
    assert store.get_run(run_id).status == "done"
    # a finished run only reports
    code, _, screen = run("blind", "--resume", run_id)
    assert code == 0 and "有效判断" in screen and "哪条是她" not in screen
    listed = run("runs", "--kind", "blind")
    assert listed[0] == 0 and run_id in listed[2] and "done" in listed[2]


def test_a_backend_that_is_not_deployed_is_refused_with_a_message(
    cli_world: World, api: respx.MockRouter
) -> None:
    code, out, _ = run("blind", "--backend", "style", "--n", "5")
    assert code == 1 and "未部署" in out
    code, out, _ = run("blind", "--backend", "all", "--n", "5")
    assert code == 1 and "未部署" in out
    code, out, _ = run("blind", "--backend", "gpt")
    assert code == 1 and "unknown backend" in out
    assert EvalStore(cli_world.services.db, cli_world.services.clock).list_runs() == []


def test_resuming_a_run_that_does_not_exist_or_is_not_a_blind_run_is_an_error(
    cli_world: World,
) -> None:
    code, out, _ = run("blind", "--resume", "nothere")
    assert code == 1 and "no evaluation run" in out
    store = EvalStore(cli_world.services.db, cli_world.services.clock)
    gate = store.create_run("gate", status="done", milestone="M1", verdict="failed")
    code, out, _ = run("blind", "--resume", gate.id)
    assert code == 1 and "not a blind run" in out


def test_the_style_report_reads_the_live_conversation_or_the_replies_of_a_run(
    cli_world: World,
) -> None:
    code, out, screen = run("style", "--source", "live", "--days", "7")
    assert code == 1  # the bot has said nothing yet: no metric can pass
    assert "风格指标" in screen and "逗号率" in screen and "引用率" in screen and "不可测" in screen
    assert "最近 7 天" in screen
    store = EvalStore(cli_world.services.db, cli_world.services.clock)
    blind = store.create_run("blind", mode="holdout", backends=["deepseek"], status="running")
    store.add_items(
        blind.id,
        [NewItem("k0", "deepseek", KNOWN, {"real": {"lines": [{"k": "text", "t": "你好"}]}})],
    )
    item = store.items(blind.id)[0]
    store.save_generated(
        item.id, {"bot": {"quote": None, "lines": [{"k": "text", "t": "你好呀"}]}}, cost_usd=0.0
    )
    code, _, screen = run(
        "style", "--source", "eval_items", "--run", blind.id, "--backend", "deepseek"
    )
    assert code == 1 and "pre_holdout" in screen and "她的真实回复" in screen
    code, out, _ = run("style", "--source", "eval_items")
    assert code == 2 and "--run" in out
    code, out, _ = run("style", "--source", "elsewhere")
    assert code == 2 and "live or eval_items" in out
    code, out, _ = run("style", "--source", "eval_items", "--run", blind.id, "--backend", "style")
    assert code == 1 and "no generated replies" in out


def test_a_style_report_is_kept_as_a_run_of_numbers_for_the_summary(cli_world: World) -> None:
    """R-EVAL-008 reads the style results from ``eval_runs``: the command keeps what it shows."""
    code, _, screen = run("style", "--source", "live", "--days", "7")
    assert code == 1 and "评估记录：" in screen
    store = EvalStore(cli_world.services.db, cli_world.services.clock)
    live = store.latest_run("style")
    assert live is not None and live.status == "done" and live.verdict == "failed"
    assert live.mode == "live" and live.backends == ()
    assert live.params == {"source": "live", "days": 7, "blind_run": None, "backend": None}
    assert live.summary["source"] == "live" and live.summary["passed"] is False
    assert [m["key"] for m in live.summary["metrics"]] == [
        "text_length",
        "comma_rate",
        "burst_size",
        "sticker_share",
        "emoji_code_rate",
        "quote_rate",
    ]
    assert all(
        set(m) >= {"reference", "measured", "deviation", "status"} for m in live.summary["metrics"]
    )
    blind = store.create_run("blind", mode="holdout", backends=["deepseek"], status="running")
    store.add_items(
        blind.id,
        [NewItem("k0", "deepseek", KNOWN, {"real": {"lines": [{"k": "text", "t": "你好"}]}})],
    )
    item = store.items(blind.id)[0]
    store.save_generated(
        item.id, {"bot": {"quote": None, "lines": [{"k": "text", "t": "你好呀"}]}}, cost_usd=0.0
    )
    run("style", "--source", "eval_items", "--run", blind.id, "--backend", "deepseek")
    held = store.latest_run("style")
    assert held is not None and held.id != live.id
    assert held.mode == "holdout" and held.backends == ("deepseek",)
    assert held.params["source"] == "eval_items" and held.params["blind_run"] == blind.id
    assert "你好" not in str(held.summary)  # numbers and names only, not what was said


def test_the_memory_test_with_too_few_facts_from_the_bot_is_not_passed(cli_world: World) -> None:
    add_memory_facts(cli_world, real=12, bot=4)
    code, _, screen = run("memory")
    assert code == 1 and "未通过（样本不足）" in screen and "chat a few more days" in screen
    assert run("gate", "M2")[0] == 1


def test_the_memory_test_is_planned_asked_reviewed_and_decides_m2(
    cli_world: World, data_dir: Path
) -> None:
    add_memory_facts(cli_world, real=12, bot=12)
    script = memory_script()
    router = serve(script)
    try:
        code, _, screen = run("memory", "--seed", "2")
        assert code == 0 and "记忆测试" in screen and "20 题" in screen
        assert "twin jobs approve" in screen
        run_id = screen.split("记忆测试 ", 1)[1].split("：", 1)[0].strip()
        batch = batch_of(screen)
        store = EvalStore(cli_world.services.db, cli_world.services.clock)
        assert store.counts(run_id)["pending"] == 20 and script.requests == []
        assert runner.invoke(app, ["jobs", "approve", batch, "--yes"]).exit_code == 0
        waiting = run("memory", "--resume", run_id)  # approved, but nobody has run the jobs
        assert waiting[0] == 0 and "还有 20 题没有出完" in waiting[2] and script.requests == []
        # the jobs run here, then the review screen: Enter keeps each verdict, `w` overturns one
        code, _, screen = run("memory", "--resume", run_id, "--foreground", keys="w\n" + "\n" * 19)
    finally:
        router.stop()
    assert script.errors == [] and len(script.requests) == 60
    assert "第 1/20 题" in screen and "自动判分" in screen and "要点：" in screen
    assert "得分" in screen and "题目 20：真实记录 10 + 机器人对话 10" in screen
    run_state = EvalStore(cli_world.services.db, cli_world.services.clock).get_run(run_id)
    assert run_state.status == "done" and run_state.summary["reviewed"] == 20
    # nearly all answers were right, so the score is above 80 % and M2 is passed
    assert code == 0 and run_state.verdict == "passed"
    gate_code, _, gate_screen = run("gate", "M2")
    assert gate_code == 0 and "M2：通过" in gate_screen
    assert run("gate", "M2", "--check")[0] == 0


def test_the_gate_commands_exit_with_0_1_and_2(cli_world: World) -> None:
    code, out, screen = run("gate", "M3")  # wired by round 10: no observed days yet
    assert code == 1 and "还差 7 天" in screen and "未通过（样本不足）" in screen
    # M4 is judged since round 12, M5 since round 14: nothing was evaluated yet, so both are 1
    assert run("gate", "M4")[0] == 1
    code, _, screen = run("gate", "M5")
    assert code == 1 and "twin model evaluate" in screen
    assert run("gate", "M5", "--check")[0] == 1  # judged just before: the stored verdict says so
    code, _, screen = run("gate", "M1")
    assert (
        code == 1 and "还没有包含 deepseek 后端的盲测" in screen and "未通过（样本不足）" in screen
    )
    code, _, screen = run("gate", "M1", "--check")
    assert code == 1 and "读取已存结果" in screen  # the verdict of the run just before
    assert run("gate", "M0")[0] == 1  # no probe results in this database
    code, out, _ = run("gate", "M9")
    assert code == 2 and "unknown milestone" in out
    listed = run("runs", "--kind", "gate")
    assert listed[0] == 0 and "M1" in listed[2] and "insufficient" in listed[2]
