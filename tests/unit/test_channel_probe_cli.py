"""``twin channel probe start | status | answer | stop | report`` (R-CH-009, R-CH-010)."""

from __future__ import annotations

from collections.abc import Coroutine, Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tests.support.ilink import API, BOT, CTX, TOKEN, USER
from twin.channel.ilink.store import Credentials, IlinkStore
from twin.channel.probe.model import (
    Action,
    ActionKind,
    ActionStatus,
    Attempt,
    AttemptPhase,
    PlanStatus,
    ProbePlan,
    Question,
    QuestionKind,
    StepId,
    StepStatus,
)
from twin.channel.probe.store import ProbeStore
from twin.channel.probe.summary import load_channel_probe_summary
from twin.channel.state import ChannelStateStore
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.ops.instance_lock import LOCK_RUN, InstanceLock
from twin.services import Services, build_services
from twin.storage.state import read_state_version

runner = CliRunner()


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    return path


@pytest.fixture
def svc(data_dir: Path) -> Iterator[Services]:
    result = runner.invoke(app, ["db", "upgrade"])
    assert result.exit_code == 0, result.output
    settings = load_settings()
    services = build_services(settings, root=resolve_paths(settings).root)
    yield services
    services.close()


def logged_in(services: Services, *, bound: bool = True) -> IlinkStore:
    store = IlinkStore(ChannelStateStore(services.db), services.clock)
    store.save_credentials(Credentials(TOKEN, BOT, USER, API, services.clock.now_utc().isoformat()))
    if bound:
        store.bind(USER, context_token=CTX)
    return store


def probe_store(services: Services) -> ProbeStore:
    return ProbeStore(services.db, services.clock)


def state_version(services: Services) -> int:
    with services.db.session() as session:
        return read_state_version(session)


def invoke(*args: str, input: str | None = None) -> Any:
    return runner.invoke(app, ["channel", "probe", *args], input=input)


def put_question(store: ProbeStore, question: Question) -> None:
    def change(plan: ProbePlan) -> None:
        step = plan.step(StepId.COUNT)
        step.status = StepStatus.ACTIVE
        step.attempts.append(
            Attempt(
                n=1,
                phase=AttemptPhase.RUNNING,
                armed_at=plan.created_at,
                baseline_inbound_at=None,
                actions=[
                    Action(
                        id=f"ask:{question.id}",
                        kind=ActionKind.ASK,
                        label="ask",
                        status=ActionStatus.ACTIVE,
                        question=question,
                    )
                ],
            )
        )

    store.update(change)


def measured(
    store: ProbeStore, *, n: int = 8, lower: float | None = 23.0, upper: float | None = 25.0
) -> None:
    """Fill the three steps with a finished measurement (the data the state machine writes)."""

    def change(plan: ProbePlan) -> None:
        count = plan.step(StepId.COUNT)
        count.status = StepStatus.DONE
        count.data = {
            "n": n,
            "capped": False,
            "api_ok": n,
            "phone_received": n,
            "mismatch": False,
            "first_problem_index": n + 1,
            "first_failure": None,
            "interval_s": 120.0,
        }
        media = plan.step(StepId.MEDIA)
        media.status = StepStatus.DONE
        media.data = {"gif_animated": True, "typing": {"visible": "yes"}, "images": {}}
        window = plan.step(StepId.WINDOW)
        window.status = StepStatus.DONE
        window.data = {
            "points": [],
            "dropped_hours": [],
            "lower_bound_h": lower,
            "upper_bound_h": upper,
            "mismatch": False,
        }

    store.update(change)


# ------------------------------------------------------------------------ start


def test_start_needs_a_login_and_a_bound_user(svc: Services) -> None:
    result = invoke("start", "--yes")
    assert result.exit_code == 1 and "not logged in" in result.output
    logged_in(svc, bound=False)
    result = invoke("start", "--yes")
    assert result.exit_code == 1 and "nobody is bound" in result.output
    assert probe_store(svc).load() is None


def test_start_refuses_an_expired_login(svc: Services) -> None:
    store = logged_in(svc)
    store.mark_needs_relogin(-14, None)
    result = invoke("start", "--yes")
    assert result.exit_code == 1 and "login expired" in result.output


def test_start_refuses_the_console_channel(svc: Services) -> None:
    logged_in(svc)
    result = runner.invoke(
        app, ["--set", "channel.kind=console", "channel", "probe", "start", "--yes"]
    )
    assert result.exit_code == 1 and "channel.kind is not 'ilink'" in result.output


def test_start_stores_the_plan_and_says_the_application_carries_it_out(svc: Services) -> None:
    logged_in(svc)
    result = invoke("start", "--yes")
    assert result.exit_code == 0, result.output
    plan = probe_store(svc).load()
    assert plan is not None and plan.status is PlanStatus.RUNNING
    assert [s.id for s in plan.steps] == [StepId.COUNT, StepId.MEDIA, StepId.WINDOW]
    assert plan.run_id in result.output
    assert "Start the application (`twin run`)" in result.output
    assert "twin channel probe answer" in result.output


def test_start_says_the_application_will_begin_when_it_is_running(svc: Services) -> None:
    logged_in(svc)
    lock = InstanceLock(LOCK_RUN, locks_dir=svc.paths.locks_dir)
    assert lock.acquire()
    try:
        result = invoke("start", "--yes")
    finally:
        lock.release()
    assert result.exit_code == 0 and "will begin within a few seconds" in result.output


def test_start_can_add_the_empty_token_experiment(svc: Services) -> None:
    logged_in(svc)
    assert invoke("start", "--yes", "--empty-token-experiment").exit_code == 0
    plan = probe_store(svc).load()
    assert plan is not None and plan.options.empty_token_experiment
    assert [s.id for s in plan.steps][-1] is StepId.EMPTY_TOKEN


def test_start_explains_and_asks_first(svc: Services) -> None:
    logged_in(svc)
    declined = invoke("start", input="n\n")
    assert declined.exit_code == 1 and "[测试]" in declined.output
    assert "25 hours of silence" in declined.output
    assert probe_store(svc).load() is None
    accepted = invoke("start", input="y\n")
    assert accepted.exit_code == 0 and probe_store(svc).load() is not None


def test_start_refuses_a_second_plan_while_one_runs(svc: Services) -> None:
    logged_in(svc)
    assert invoke("start", "--yes").exit_code == 0
    again = invoke("start", "--yes")
    assert again.exit_code == 1 and "still running" in again.output


# ----------------------------------------------------------------------- status


def test_status_without_a_plan_points_to_start(svc: Services) -> None:
    result = invoke("status")
    assert result.exit_code == 0 and "no probe plan yet" in result.output


def test_status_shows_the_plan_the_notice_the_steps_and_what_waits_for_you(svc: Services) -> None:
    logged_in(svc)
    store = probe_store(svc)
    plan = store.create()
    put_question(store, Question("count-1-count", QuestionKind.COUNT, "how many?", maximum=8))
    store.update(lambda p: setattr(p, "notice", "Send the bot ONE message"))
    before = state_version(svc)
    result = invoke("status")
    assert result.exit_code == 0, result.output
    out = result.output
    assert f"probe {plan.run_id}: running" in out
    assert "NOT running" in out and ">>> Send the bot ONE message" in out
    assert "1 question(s) waiting: run `twin channel probe answer`" in out
    assert "1/3 count" in out and "2/3 media" in out and "3/3 window" in out
    assert "recent events:" in out and "probe plan created" in out
    assert state_version(svc) == before  # a READ command writes nothing


def test_status_says_when_the_wechat_reminder_could_not_be_sent(svc: Services) -> None:
    store = probe_store(svc)
    store.create()

    def waiting(plan: ProbePlan) -> None:
        step = plan.step(StepId.COUNT)
        step.status = StepStatus.ACTIVE
        step.attempts.append(
            Attempt(
                n=1,
                phase=AttemptPhase.WAITING,
                armed_at=plan.created_at,
                baseline_inbound_at=None,
                announce_note="not delivered (window_rejected: session_expired)",
            )
        )
        plan.notice = "Send the bot ONE message"

    store.update(waiting)
    out = invoke("status").output
    assert "the WeChat reminder was not delivered (window_rejected: session_expired)" in out


def test_the_channel_status_mentions_the_probe_when_there_is_one(svc: Services) -> None:
    before = runner.invoke(app, ["channel", "status"])
    assert "probe:" not in before.output
    store = probe_store(svc)
    plan = store.create()
    store.update(lambda p: setattr(p, "notice", "Send the bot ONE message"))
    after = runner.invoke(app, ["channel", "status"])
    assert after.exit_code == 0, after.output
    assert f"probe: {plan.run_id} running - Send the bot ONE message" in after.output


def test_status_shows_what_each_finished_step_found(svc: Services) -> None:
    store = probe_store(svc)
    store.create()
    measured(store, n=8, lower=23.0, upper=25.0)
    out = invoke("status").output
    assert "N=8, server accepted 8, phone received 8" in out
    assert "delivered up to 23.0 h, first miss at 25.0 h" in out
    assert "2/3 media" in out and "done" in out


def test_status_says_why_a_step_was_skipped(svc: Services) -> None:
    store = probe_store(svc)
    store.create()
    store.update(lambda p: setattr(p.step(StepId.MEDIA), "status", StepStatus.SKIPPED))
    store.update(lambda p: setattr(p.step(StepId.MEDIA), "skip_reason", "no room"))
    assert "no room" in invoke("status").output


def test_status_of_a_finished_plan_shows_how_it_ended(svc: Services) -> None:
    store = probe_store(svc)
    store.create()
    store.finish(PlanStatus.STOPPED, "stopped by the user")
    out = invoke("status").output
    assert "stopped (stopped by the user)" in out and "ended" in out


def test_status_falls_back_to_the_stored_result_when_the_plan_is_gone(svc: Services) -> None:
    store = probe_store(svc)
    store.create()
    store.finish(PlanStatus.STOPPED, "x")
    store.state.delete("probe.plan")
    out = invoke("status").output
    assert "no probe plan yet" in out and "last stored result" in out


# ----------------------------------------------------------------------- answer


def test_answer_without_a_plan_is_an_error(svc: Services) -> None:
    result = invoke("answer")
    assert result.exit_code == 1 and "there is no probe plan" in result.output


def test_answer_with_nothing_pending_says_so(svc: Services) -> None:
    probe_store(svc).create()
    result = invoke("answer")
    assert result.exit_code == 0 and "no question is waiting" in result.output


def test_answer_asks_again_until_the_answer_is_valid(svc: Services) -> None:
    store = probe_store(svc)
    store.create()
    put_question(
        store,
        Question("count-1-count", QuestionKind.COUNT, "How many arrived?", maximum=8),
    )
    result = invoke("answer", input="lots\n9\n5\n")
    assert result.exit_code == 0, result.output
    assert "How many arrived?" in result.output
    assert "whole number" in result.output and "at most 8" in result.output
    assert "answered 1 question(s)" in result.output
    plan = store.load()
    assert plan is not None
    action = plan.step(StepId.COUNT).attempts[0].actions[0]
    assert action.question is not None and action.question.answer == "5"
    assert action.status is ActionStatus.ACTIVE  # the running application finishes the action


def test_answer_accepts_a_choice_and_a_bare_enter_for_ready(svc: Services) -> None:
    store = probe_store(svc)
    store.create()
    put_question(
        store,
        Question(
            "gif", QuestionKind.CHOICE, "Does it move?", choices=["moving", "still", "missing"]
        ),
    )
    out = invoke("answer", input="what\nMoving\n").output
    assert "moving, still, missing" in out and "answered 1" in out
    ready = Question("typing", QuestionKind.READY, "Press Enter when you look")

    def swap(plan: ProbePlan) -> None:
        plan.step(StepId.COUNT).attempts[0].actions[0].question = ready

    store.update(swap)
    assert "answered 1" in invoke("answer", input="\n").output


def test_answer_can_be_postponed(svc: Services) -> None:
    store = probe_store(svc)
    store.create()
    put_question(store, Question("count-1-count", QuestionKind.COUNT, "How many?", maximum=8))
    result = invoke("answer", input="later\n")
    assert result.exit_code == 0 and "answered 0 question(s)" in result.output
    plan = store.load()
    assert plan is not None and store.pending_questions(plan)  # still waiting


def test_answer_with_wait_returns_when_the_plan_ends(svc: Services) -> None:
    store = probe_store(svc)
    store.create()
    store.finish(PlanStatus.STOPPED, "x")
    result = invoke("answer", "--wait")
    assert result.exit_code == 0 and "no question is waiting" in result.output


def test_answer_with_wait_keeps_looking_until_the_plan_ends(
    svc: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = probe_store(svc)
    store.create()
    pauses: list[float] = []

    def pause(coroutine: Coroutine[Any, Any, None]) -> None:
        coroutine.close()
        pauses.append(1.0)
        store.finish(PlanStatus.COMPLETED, None)  # time passes; the application finished the plan

    monkeypatch.setattr("twin.channel.probe.cli.asyncio.run", pause)
    result = invoke("answer", "--wait")
    assert result.exit_code == 0 and pauses == [1.0]


def test_answer_with_wait_can_be_left_with_ctrl_c(
    svc: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe_store(svc).create()

    def interrupt(coroutine: Coroutine[Any, Any, None]) -> None:
        coroutine.close()
        raise KeyboardInterrupt

    monkeypatch.setattr("twin.channel.probe.cli.asyncio.run", interrupt)
    result = invoke("answer", "--wait")
    assert result.exit_code == 0 and "no question is waiting" in result.output


# ------------------------------------------------------------------------- stop


def test_stop_without_a_running_plan_says_so(svc: Services) -> None:
    assert "no probe is running" in invoke("stop").output
    probe_store(svc).create()
    probe_store(svc).finish(PlanStatus.COMPLETED, None)
    assert "no probe is running" in invoke("stop").output


def test_stop_asks_first_and_then_ends_the_plan_keeping_what_was_measured(svc: Services) -> None:
    store = probe_store(svc)
    store.create()
    kept = invoke("stop", input="n\n")
    assert kept.exit_code == 1
    plan = store.load()
    assert plan is not None and plan.status is PlanStatus.RUNNING
    result = invoke("stop", "--yes")
    assert result.exit_code == 0 and "stopped" in result.output
    plan = store.load()
    assert plan is not None and plan.status is PlanStatus.STOPPED
    assert plan.stop_reason == "stopped by the user"
    with svc.db.session() as session:
        assert load_channel_probe_summary(session) is not None


# ----------------------------------------------------------------------- report


def test_report_without_any_result_is_an_error(svc: Services, tmp_path: Path) -> None:
    result = invoke("report", "--output", str(tmp_path / "r.md"))
    assert result.exit_code == 1 and "no probe result yet" in result.output
    assert not (tmp_path / "r.md").exists()


def test_report_writes_the_file_and_the_verdict_and_asks_before_touching_the_configuration(
    svc: Services, tmp_path: Path
) -> None:
    store = probe_store(svc)
    store.create()
    measured(store)
    store.finish(PlanStatus.COMPLETED, None)
    config = tmp_path / "my-config.yaml"
    config.write_text("channel: { kind: ilink, outbound_quota_safe: 8 }\n", encoding="utf-8")
    report = tmp_path / "out" / "CHANNEL_REPORT.md"
    result = runner.invoke(
        app,
        ["--config", str(config), "channel", "probe", "report", "--output", str(report)],
        input="n\n",
    )
    assert result.exit_code == 0, result.output
    assert report.is_file() and "R-CH-010 判定：**达标**" in report.read_text(encoding="utf-8")
    assert "verdict: met; N=8" in result.output
    assert "channel.outbound_quota_safe: 8 -> 7" in result.output
    assert "channel.proactive_window_safe_h: 22.0 -> 20.7" in result.output
    assert "configuration left unchanged" in result.output
    assert (
        config.read_text(encoding="utf-8") == "channel: { kind: ilink, outbound_quota_safe: 8 }\n"
    )


def test_report_writes_the_suggestions_only_after_a_yes(svc: Services, tmp_path: Path) -> None:
    store = probe_store(svc)
    store.create()
    measured(store)
    store.finish(PlanStatus.COMPLETED, None)
    config = tmp_path / "my-config.yaml"
    config.write_text("# mine\nchannel: { kind: ilink }\nretrieval: { device: cpu }\n", "utf-8")
    result = runner.invoke(
        app,
        ["--config", str(config), "channel", "probe", "report", "--output", str(tmp_path / "r.md")],
        input="y\n",
    )
    assert result.exit_code == 0 and f"updated {config}" in result.output
    settings = load_settings(config)
    assert settings.channel.outbound_quota_safe == 7
    assert settings.channel.proactive_window_safe_h == 20.7
    assert config.read_text(encoding="utf-8").startswith("# mine\n")


def test_report_apply_skips_the_question_and_no_apply_skips_the_offer(
    svc: Services, tmp_path: Path
) -> None:
    store = probe_store(svc)
    store.create()
    measured(store)
    store.finish(PlanStatus.COMPLETED, None)
    config = tmp_path / "c.yaml"
    config.write_text("channel: { kind: ilink }\n", encoding="utf-8")
    base = [
        "--config",
        str(config),
        "channel",
        "probe",
        "report",
        "--output",
        str(tmp_path / "r.md"),
    ]
    no = runner.invoke(app, [*base, "--no-apply"])
    assert no.exit_code == 0 and "suggested configuration" not in no.output
    assert config.read_text(encoding="utf-8") == "channel: { kind: ilink }\n"
    yes = runner.invoke(app, [*base, "--apply"])
    assert yes.exit_code == 0 and "suggested configuration" in yes.output
    assert load_settings(config).channel.outbound_quota_safe == 7


def test_report_of_a_failing_channel_writes_it_and_says_stop(svc: Services, tmp_path: Path) -> None:
    store = probe_store(svc)
    store.create()
    measured(store, n=2, lower=6.0, upper=12.0)
    store.finish(PlanStatus.COMPLETED, None)
    report = tmp_path / "r.md"
    result = invoke("report", "--output", str(report))
    assert result.exit_code == 1
    assert "STOP: the channel does not meet R-CH-010" in result.output
    assert "enterprise WeChat" in result.output
    assert "suggested configuration" not in result.output  # nothing to apply when stopping
    text = report.read_text(encoding="utf-8")
    assert "未达标" in text and "建议停机" in text


def test_report_during_a_run_shows_what_is_measured_so_far(svc: Services, tmp_path: Path) -> None:
    store = probe_store(svc)
    store.create()
    report = tmp_path / "r.md"
    result = invoke("report", "--output", str(report))
    assert result.exit_code == 0, result.output
    assert "进行中（未做完）" in report.read_text(encoding="utf-8")
    assert "verdict: undetermined" in result.output


def test_the_default_report_path_is_docs_channel_report_under_the_project_root(
    svc: Services,
) -> None:
    store = probe_store(svc)
    store.create()
    result = invoke("report")
    assert result.exit_code == 0, result.output
    target = svc.paths.root / "docs" / "CHANNEL_REPORT.md"
    assert target.is_file() and f"report written to {target}" in result.output


def test_an_unusable_configuration_is_reported_not_overwritten(
    svc: Services, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = probe_store(svc)
    store.create()
    measured(store)
    store.finish(PlanStatus.COMPLETED, None)
    config = tmp_path / "list.yaml"
    config.write_text("channel: { kind: ilink }\n", encoding="utf-8")

    def refuse(*_args: object, **_kwargs: object) -> None:
        from twin.config.loader import ConfigError

        raise ConfigError("not valid")

    monkeypatch.setattr("twin.channel.probe.cli.set_section_values", refuse)
    result = runner.invoke(
        app,
        [
            "--config",
            str(config),
            "channel",
            "probe",
            "report",
            "--output",
            str(tmp_path / "r.md"),
            "--apply",
        ],
    )
    assert result.exit_code == 1 and "could not update" in result.output
    assert config.read_text(encoding="utf-8") == "channel: { kind: ilink }\n"


def test_probe_commands_are_listed_in_the_help() -> None:
    result = runner.invoke(app, ["channel", "probe", "--help"])
    assert result.exit_code == 0
    for name in ("start", "status", "answer", "stop", "report"):
        assert name in result.output
