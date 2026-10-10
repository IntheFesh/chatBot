"""The doctor's checks of round 12: DeepSeek, GPU, scheduled task, power plan (R-OPS-009)."""

from __future__ import annotations

import httpx
import pytest

from tests.support.ops import POWERCFG, TASK_SPEC, ScriptedRunner, failed, ok
from twin.config.secrets import SecretStore
from twin.config.settings import Settings
from twin.ops import doctor
from twin.ops.doctor import (
    CheckResult,
    CheckStatus,
    DoctorContext,
    balance_verdict,
    check_deepseek_net,
    check_gpu,
    check_power_plan,
    check_scheduled_task,
    exit_code,
    run_checks,
    sleep_timeouts,
)
from twin.ops.taskscheduler import build_task_xml

SPEC = TASK_SPEC


def settings() -> Settings:
    return Settings()


def context(
    secret_store: SecretStore,
    handler: object = None,
    *,
    runner: ScriptedRunner | None = None,
    platform: str = "linux",
    conf: Settings | None = None,
) -> DoctorContext:
    transport = httpx.MockTransport(handler) if handler is not None else None  # type: ignore[arg-type]
    return DoctorContext(
        conf or settings(),
        secrets=secret_store,
        platform=platform,
        http_transport=transport,
        runner=runner,
    )


# ------------------------------------------------------------------- the balance answer


@pytest.mark.parametrize(
    ("status", "body", "level", "text"),
    [
        (401, None, CheckStatus.FAIL, "rejected the API key"),
        (403, {}, CheckStatus.FAIL, "rejected the API key"),
        (402, None, CheckStatus.FAIL, "no balance"),
        (200, {"is_available": False}, CheckStatus.FAIL, "cannot be used"),
        (
            200,
            {"is_available": True, "balance_infos": [{"currency": "CNY", "total_balance": "9.50"}]},
            CheckStatus.OK,
            "balance 9.50 CNY",
        ),
        (200, {"is_available": True, "balance_infos": []}, CheckStatus.OK, "account is available"),
        (200, ["not", "a", "dict"], CheckStatus.OK, "balance not shown"),
        (404, None, CheckStatus.OK, "HTTP 404"),
        (503, None, CheckStatus.WARN, "HTTP 503"),
    ],
)
def test_the_balance_answer_is_read_for_what_it_means(
    status: int, body: object, level: CheckStatus, text: str
) -> None:
    verdict, detail, hint = balance_verdict(status, body)
    assert verdict is level and text in detail
    assert bool(hint) == (level is CheckStatus.FAIL or status >= 500)


def test_deepseek_is_asked_for_the_balance_with_the_stored_key(secret_store: SecretStore) -> None:
    secret_store.set("deepseek_api_key", "sk-synthetic")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "is_available": True,
                "balance_infos": [{"currency": "USD", "total_balance": "3"}],
            },
        )

    result = check_deepseek_net(context(secret_store, handler))
    assert result.status is CheckStatus.OK and "balance 3 USD" in result.detail
    (request,) = seen
    assert request.url.path == "/user/balance" and request.method == "GET"
    assert request.headers["authorization"] == "Bearer sk-synthetic"
    assert "sk-synthetic" not in result.detail + result.hint


def test_a_refused_key_and_an_empty_account_fail_the_check(secret_store: SecretStore) -> None:
    secret_store.set("deepseek_api_key", "sk-synthetic")
    assert (
        check_deepseek_net(context(secret_store, lambda r: httpx.Response(401))).status
        is CheckStatus.FAIL
    )
    empty = check_deepseek_net(context(secret_store, lambda r: httpx.Response(402)))
    assert empty.status is CheckStatus.FAIL and "top up" in empty.hint


def test_a_missing_key_is_a_warning_with_the_command_to_fix_it(secret_store: SecretStore) -> None:
    result = check_deepseek_net(context(secret_store, lambda r: httpx.Response(401)))
    assert result.status is CheckStatus.WARN and "twin secrets set deepseek_api_key" in result.hint


def test_no_network_is_a_warning_not_a_failure(secret_store: SecretStore) -> None:
    secret_store.set("deepseek_api_key", "sk-synthetic")

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    slow = check_deepseek_net(context(secret_store, timeout))
    assert slow.status is CheckStatus.WARN and "timed out" in slow.detail
    gone = check_deepseek_net(context(secret_store, refused))
    assert gone.status is CheckStatus.WARN and "ConnectError" in gone.detail
    skipped = check_deepseek_net(DoctorContext(None, "bad config"))
    assert skipped.status is CheckStatus.WARN and "skipped" in skipped.detail


# ------------------------------------------------------------------------------- the GPU


def test_a_gpu_is_reported_with_its_driver() -> None:
    runner = ScriptedRunner({"nvidia-smi": ok("NVIDIA GeForce RTX 5090, 570.86, 32607 MiB\n")})
    result = check_gpu(DoctorContext(settings(), runner=runner))
    assert result.status is CheckStatus.OK and "RTX 5090" in result.detail
    assert runner.calls[0][0] == "nvidia-smi"
    empty = check_gpu(DoctorContext(settings(), runner=ScriptedRunner({"nvidia-smi": ok("")})))
    assert empty.status is CheckStatus.OK and "found" in empty.detail


def test_no_gpu_matters_only_when_the_local_style_model_is_wanted() -> None:
    absent = ScriptedRunner({})  # nvidia-smi is not installed
    plain = check_gpu(DoctorContext(settings(), runner=absent))
    assert plain.status is CheckStatus.OK and "only the local style model needs one" in plain.detail
    wanted = settings()
    wanted.backend.active = "style"
    wanted.style_model.mode = "llamacpp_completion"
    needing = check_gpu(DoctorContext(wanted, runner=ScriptedRunner({"nvidia-smi": failed()})))
    assert needing.status is CheckStatus.WARN and "vllm_completion" in needing.hint
    remote = settings()
    remote.backend.active = "style"
    remote.style_model.mode = "vllm_completion"
    assert (
        check_gpu(DoctorContext(remote, runner=absent)).status is CheckStatus.OK
    )  # the card is on the rented machine


# ----------------------------------------------------------------------- scheduled task


def task_runner(xml: str) -> ScriptedRunner:
    return ScriptedRunner({"schtasks.exe /Query": ok(xml)})


def test_the_task_check_applies_to_windows_only(secret_store: SecretStore) -> None:
    result = check_scheduled_task(context(secret_store))
    assert result.status is CheckStatus.OK and "Windows only" in result.detail


def test_a_task_as_installed_is_fine(secret_store: SecretStore) -> None:
    result = check_scheduled_task(
        context(secret_store, runner=task_runner(build_task_xml(SPEC)), platform="win32")
    )
    assert result.status is CheckStatus.OK and "InteractiveToken" in result.detail


@pytest.mark.parametrize(
    ("change", "level", "text"),
    [
        (("InteractiveToken", "S4U"), CheckStatus.FAIL, "must be InteractiveToken"),
        (("PT0S", "PT72H"), CheckStatus.WARN, "time limit"),
        (("IgnoreNew", "Parallel"), CheckStatus.WARN, "second instance"),
        (
            ("<DisallowStartIfOnBatteries>false", "<DisallowStartIfOnBatteries>true"),
            CheckStatus.WARN,
            "battery",
        ),
        (
            ("<Enabled>true</Enabled>\n    <Hidden>", "<Enabled>false</Enabled>\n    <Hidden>"),
            CheckStatus.WARN,
            "disabled",
        ),
        (
            (
                "<Command>C:\\repo\\.venv\\Scripts\\twin.exe</Command>\n"
                "      <Arguments>supervise --from-task</Arguments>",
                "<Command>C:\\tools\\uv.exe</Command>\n      <Arguments>sync --frozen</Arguments>",
            ),
            CheckStatus.FAIL,
            "uv sync",
        ),
    ],
)
def test_a_task_that_is_not_as_installed_is_found_out(
    secret_store: SecretStore, change: tuple[str, str], level: CheckStatus, text: str
) -> None:
    xml = build_task_xml(SPEC)
    assert change[0] in xml
    result = check_scheduled_task(
        context(secret_store, runner=task_runner(xml.replace(*change)), platform="win32")
    )
    assert result.status is level and text in result.detail
    assert "twin service install" in result.hint


def test_a_missing_or_unreadable_task_is_a_warning(secret_store: SecretStore) -> None:
    missing = check_scheduled_task(
        context(secret_store, runner=ScriptedRunner({"schtasks.exe": failed()}), platform="win32")
    )
    assert missing.status is CheckStatus.WARN and "not registered" in missing.detail
    no_tool = check_scheduled_task(
        context(secret_store, runner=ScriptedRunner({}), platform="win32")
    )
    assert no_tool.status is CheckStatus.WARN and "cannot run" in no_tool.detail
    garbled = check_scheduled_task(
        context(secret_store, runner=task_runner("<not xml"), platform="win32")
    )
    assert garbled.status is CheckStatus.WARN and "cannot be read" in garbled.detail


# -------------------------------------------------------------------------- power plan


def test_the_sleep_timeouts_are_the_last_two_numbers_whatever_the_language() -> None:
    assert sleep_timeouts(POWERCFG.format(ac=0, dc=900)) == (0, 900)
    chinese = POWERCFG.replace("Current AC Power Setting Index", "当前交流电源设置索引").replace(
        "Current DC Power Setting Index", "当前直流电源设置索引"
    )
    assert sleep_timeouts(chinese.format(ac=1800, dc=300)) == (1800, 300)
    assert sleep_timeouts("nothing useful") is None
    assert sleep_timeouts("0x00000000") is None


@pytest.mark.parametrize(
    ("ac", "dc", "level", "text"),
    [
        (0, 0, CheckStatus.OK, "never"),
        (1800, 0, CheckStatus.WARN, "on mains power the computer sleeps after 30 min"),
        (0, 900, CheckStatus.WARN, "on battery the computer sleeps after 15 min"),
        (600, 600, CheckStatus.WARN, "after 10 min"),
    ],
)
def test_a_power_plan_that_sleeps_is_a_warning(
    secret_store: SecretStore, ac: int, dc: int, level: CheckStatus, text: str
) -> None:
    runner = ScriptedRunner({"powercfg": ok(POWERCFG.format(ac=ac, dc=dc))})
    result = check_power_plan(context(secret_store, runner=runner, platform="win32"))
    assert result.status is level and text in result.detail
    assert runner.calls[0][:3] == ["powercfg", "/query", "SCHEME_CURRENT"]
    if level is CheckStatus.WARN:
        assert "Never" in result.hint


def test_an_unreadable_power_plan_and_other_platforms(secret_store: SecretStore) -> None:
    unreadable = check_power_plan(
        context(secret_store, runner=ScriptedRunner({"powercfg": failed()}), platform="win32")
    )
    assert unreadable.status is CheckStatus.WARN and "powercfg" in unreadable.detail
    absent = check_power_plan(context(secret_store, runner=ScriptedRunner({}), platform="win32"))
    assert absent.status is CheckStatus.WARN
    assert check_power_plan(context(secret_store)).status is CheckStatus.OK


# -------------------------------------------------------------------------- the runner


def test_a_crashing_check_is_reported_and_the_exit_code_follows_failures(
    secret_store: SecretStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(ctx: DoctorContext) -> CheckResult:
        raise RuntimeError("synthetic")

    monkeypatch.setattr(doctor, "_CHECKS", [check_power_plan, boom])
    results = run_checks(context(secret_store))
    assert [r.name for r in results] == ["power-plan", "boom"]
    assert results[1].status is CheckStatus.FAIL and "synthetic" in results[1].detail
    assert exit_code(results) == 1 and exit_code(results[:1]) == 0


def test_the_context_runs_programs_without_a_shell_and_none_when_they_are_missing() -> None:
    ctx = DoctorContext(None)
    assert ctx.run(["/definitely/not/installed"]) is None
    answered = DoctorContext(None, runner=ScriptedRunner({"echo": ok("hi")})).run(["echo", "x"])
    assert answered is not None and answered.returncode == 0
