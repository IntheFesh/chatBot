"""The scheduled task, ``twin service`` and ``twin supervise`` (R-OPS-001, R-ARCH-006)."""

from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.support.ops import ScriptedRunner, failed, ok
from tests.support.win32 import FakeWin32
from twin.cli import app
from twin.config.loader import parse_overrides
from twin.config.secrets import SecretStore
from twin.ops.instance_lock import (
    LOCK_RUN,
    LOCK_SUPERVISOR,
    InstanceLock,
    locks_held_elsewhere,
)
from twin.ops.process_model import ExitCode
from twin.ops.service import (
    StopOutcome,
    default_waiter,
    read_status,
    running,
    stop_file_path,
    stop_service,
    summarize_restarts,
)
from twin.ops.service_cli import (
    AUTOLOGON_NOTE,
    ServiceEnv,
    describe_task,
    global_options,
    task_spec,
    use_service_env,
)
from twin.ops.supervise import Restart, RestartLog
from twin.ops.taskscheduler import (
    TASK_NAME,
    TASK_NS,
    CommandResult,
    SubprocessRunner,
    TaskInfo,
    TaskScheduler,
    TaskSchedulerError,
    TaskSpec,
    build_task_xml,
    decode_output,
    parse_task_xml,
)
from twin.services import CliContext, Services, get_cli_context, set_cli_context

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "ops" / "task.xml"
SPEC = TaskSpec(
    command=r"C:\Users\小明\wechat-twin\.venv\Scripts\twin.exe",
    arguments="supervise --from-task",
    working_dir=r"C:\Users\小明\wechat-twin & 备份",
    user_id=r"DESKTOP-TWIN\小明",
)
runner = CliRunner()


def tag(name: str) -> str:
    return TASK_NS + name


# ----------------------------------------------------------------------------- task XML


def test_the_task_definition_is_the_reviewed_snapshot() -> None:
    assert build_task_xml(SPEC) == FIXTURE.read_text(encoding="utf-8")


def test_the_task_runs_at_logon_as_the_interactive_user_without_limits() -> None:
    root = ET.fromstring(build_task_xml(SPEC).split("?>", 1)[1])  # noqa: S314 - our own text

    def text(*path: str) -> str | None:
        node: ET.Element | None = root
        for name in path:
            node = node.find(tag(name)) if node is not None else None
        return None if node is None else node.text

    assert root.find(f"{tag('Triggers')}/{tag('LogonTrigger')}") is not None
    assert text("Principals", "Principal", "LogonType") == "InteractiveToken"
    assert text("Principals", "Principal", "RunLevel") == "LeastPrivilege"
    assert text("Principals", "Principal", "UserId") == r"DESKTOP-TWIN\小明"
    assert text("Settings", "ExecutionTimeLimit") == "PT0S"
    assert text("Settings", "DisallowStartIfOnBatteries") == "false"
    assert text("Settings", "StopIfGoingOnBatteries") == "false"
    assert text("Settings", "MultipleInstancesPolicy") == "IgnoreNew"
    assert text("Settings", "RestartOnFailure", "Interval") == "PT1M"
    assert text("Settings", "RestartOnFailure", "Count") == "999"
    assert text("Actions", "Exec", "Command") == SPEC.command
    assert text("Actions", "Exec", "Arguments") == "supervise --from-task"
    assert text("Actions", "Exec", "WorkingDirectory") == SPEC.working_dir
    document = build_task_xml(SPEC)
    for forbidden in ("S4U", "Password", "<Service", "uv sync", "SYSTEM"):
        assert forbidden not in document, forbidden


def test_special_characters_in_the_paths_are_escaped() -> None:
    spec = TaskSpec("C:\\a<b>.exe", 'x "y" & z', "C:\\d&e", "PC\\me<1>", description="a&b")
    root = ET.fromstring(build_task_xml(spec).split("?>", 1)[1])  # noqa: S314 - our own text
    assert root.findtext(f"{tag('Actions')}/{tag('Exec')}/{tag('Command')}") == "C:\\a<b>.exe"
    assert root.findtext(f"{tag('Actions')}/{tag('Exec')}/{tag('Arguments')}") == 'x "y" & z'
    assert root.findtext(f"{tag('RegistrationInfo')}/{tag('Description')}") == "a&b"


def test_the_registered_definition_is_read_back_fact_by_fact() -> None:
    info = parse_task_xml(build_task_xml(SPEC))
    assert info == TaskInfo(
        registered=True,
        logon_type="InteractiveToken",
        command=SPEC.command,
        arguments="supervise --from-task",
        working_dir=SPEC.working_dir,
        enabled=True,
        time_limit="PT0S",
        multiple_instances="IgnoreNew",
        on_battery_allowed=True,
        restart_interval="PT1M",
        runs_uv_sync=False,
    )
    assert info.interactive
    assert describe_task(info)[2] == "  logon type: InteractiveToken"


def test_a_task_without_an_enabled_setting_is_an_enabled_task() -> None:
    """Task Scheduler leaves out settings that have their default value when it exports a task."""
    full = build_task_xml(SPEC)
    exported = full.replace("    <Enabled>true</Enabled>\n    <Hidden>", "    <Hidden>")
    assert exported != full and "<Enabled>false" not in exported
    info = parse_task_xml(exported)
    assert info.enabled is True
    assert "disabled" not in "\n".join(describe_task(info))
    disabled = parse_task_xml(
        full.replace(
            "<Enabled>true</Enabled>\n    <Hidden>", "<Enabled>false</Enabled>\n    <Hidden>"
        )
    )
    assert disabled.enabled is False and "disabled" in "\n".join(describe_task(disabled))
    assert TaskInfo(registered=False).enabled is None  # no task, no answer


def test_a_byte_order_mark_and_a_utf16_declaration_do_not_matter() -> None:
    text = "﻿" + build_task_xml(SPEC)
    assert parse_task_xml(text).interactive
    with pytest.raises(TaskSchedulerError, match="cannot be read"):
        parse_task_xml("<Task><oops></Task>")


def test_a_task_that_runs_uv_sync_is_recognised_and_one_that_does_not_sync_is_not() -> None:
    def info_for(command: str, arguments: str, logon: str = "S4U") -> TaskInfo:
        xml = (
            f'<Task xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">'
            f"<Principals><Principal><LogonType>{logon}</LogonType></Principal></Principals>"
            f"<Settings><Enabled>false</Enabled><DisallowStartIfOnBatteries>true"
            f"</DisallowStartIfOnBatteries></Settings>"
            f"<Actions><Exec><Command>{command}</Command><Arguments>{arguments}</Arguments></Exec>"
            f"</Actions></Task>"
        )
        return parse_task_xml(xml)

    assert info_for(r"C:\Users\me\.local\bin\uv.exe", "sync --frozen").runs_uv_sync
    assert info_for("uv", "run sync").runs_uv_sync
    assert not info_for(r"C:\repo\.venv\Scripts\twin.exe", "supervise").runs_uv_sync
    assert not info_for("uv.exe", "run --frozen --no-sync twin supervise").runs_uv_sync
    bad = info_for("uv.exe", "sync")
    assert not bad.interactive and bad.enabled is False and bad.on_battery_allowed is False
    lines = "\n".join(describe_task(bad))
    assert "must be InteractiveToken" in lines and "runs `uv sync`" in lines
    assert "disabled" in lines and "runs on battery: NO" in lines and "must be none" in lines


def test_the_output_of_console_programs_is_decoded_whatever_its_encoding() -> None:
    assert decode_output("任务".encode("utf-16")) == "任务"  # with a byte order mark
    assert decode_output("任务".encode()) == "任务"
    assert decode_output(b"caf\xe9") == "caf\xe9"  # not UTF-8: the legacy page


# ----------------------------------------------------------------------------- schtasks


def registered_runner(
    extra: dict[str, CommandResult | Callable[[list[str]], CommandResult]],
) -> ScriptedRunner:
    answers: dict[str, CommandResult | Callable[[list[str]], CommandResult]] = {
        "schtasks.exe /Query /TN wechat-twin /XML": ok(build_task_xml(SPEC)),
    }
    answers.update(extra)
    return ScriptedRunner(answers)


def test_install_registers_the_xml_with_force_in_utf16() -> None:
    seen: list[str] = []

    def create(args: list[str]) -> CommandResult:
        path = Path(args[args.index("/XML") + 1])
        raw = path.read_bytes()
        assert raw[:2] == b"\xff\xfe"
        seen.append(raw.decode("utf-16"))
        return ok("SUCCESS")

    scripted = ScriptedRunner({"schtasks.exe /Create": create})
    TaskScheduler(scripted).install(SPEC)
    (call,) = scripted.calls
    assert call[:4] == ["schtasks.exe", "/Create", "/TN", TASK_NAME] and call[-1] == "/F"
    assert seen == [build_task_xml(SPEC)]
    assert not Path(call[call.index("/XML") + 1]).exists()  # the temporary file is gone


def test_a_failed_registration_says_why_in_one_line() -> None:
    scripted = ScriptedRunner({"schtasks.exe /Create": failed("ERROR: Access is denied.\n")})
    with pytest.raises(TaskSchedulerError, match="registering the scheduled task failed: ERROR"):
        TaskScheduler(scripted).install(SPEC)
    empty = ScriptedRunner({"schtasks.exe /Create": CommandResult(5, b"", b"")})
    with pytest.raises(TaskSchedulerError, match="exit code 5"):
        TaskScheduler(empty).install(SPEC)


def test_uninstall_removes_a_registered_task_and_says_so_when_there_is_none() -> None:
    scripted = registered_runner({"schtasks.exe /Delete": ok()})
    assert TaskScheduler(scripted).uninstall() is True
    assert ["schtasks.exe", "/Delete", "/TN", TASK_NAME, "/F"] in scripted.calls
    none = ScriptedRunner({"schtasks.exe /Query": failed()})
    assert TaskScheduler(none).uninstall() is False
    assert not any(call[1] == "/Delete" for call in none.calls)
    stuck = registered_runner({"schtasks.exe /Delete": failed("ERROR: Access is denied.")})
    with pytest.raises(TaskSchedulerError, match="removing"):
        TaskScheduler(stuck).uninstall()


def test_start_and_end_run_the_task_and_report_failures() -> None:
    scripted = ScriptedRunner({"schtasks.exe /Run": ok(), "schtasks.exe /End": ok()})
    scheduler = TaskScheduler(scripted)
    scheduler.start()
    scheduler.end()
    assert scripted.calls == [
        ["schtasks.exe", "/Run", "/TN", TASK_NAME],
        ["schtasks.exe", "/End", "/TN", TASK_NAME],
    ]
    broken = ScriptedRunner({"schtasks.exe /Run": failed("no"), "schtasks.exe /End": failed("no")})
    with pytest.raises(TaskSchedulerError, match="starting"):
        TaskScheduler(broken).start()
    with pytest.raises(TaskSchedulerError, match="stopping"):
        TaskScheduler(broken).end()


def test_info_and_state_read_what_schtasks_prints() -> None:
    csv_row = '"PC","\\wechat-twin","N/A","Running","Interactive only"\r\n'
    scripted = registered_runner({"schtasks.exe /Query /TN wechat-twin /FO": ok(csv_row)})
    scheduler = TaskScheduler(scripted)
    assert scheduler.info().interactive and scheduler.state() == "Running"
    gone = TaskScheduler(ScriptedRunner({"schtasks.exe /Query": failed()}))
    assert gone.info() == TaskInfo(registered=False) and gone.state() is None
    short = TaskScheduler(ScriptedRunner({"schtasks.exe /Query": ok('"a","b"\r\n')}))
    assert short.state() is None


def test_the_real_runner_runs_a_program_without_a_shell_and_reports_what_it_cannot() -> None:
    real = SubprocessRunner()
    result = real.run([sys.executable, "-c", "import sys; print('out'); sys.exit(3)"])
    assert result.returncode == 3 and result.stdout.strip() == b"out"
    with pytest.raises(TaskSchedulerError, match="cannot run"):
        real.run(["/definitely/not/a/program"])
    with pytest.raises(TaskSchedulerError, match="TimeoutExpired"):
        real.run([sys.executable, "-c", "import time; time.sleep(30)"], timeout_s=0.2)


# --------------------------------------------------------- the command line of the child


def test_the_options_before_the_command_are_repeated_for_the_child(tmp_path: Path) -> None:
    config = (
        tmp_path / "twin.yaml"
    )  # a path of this platform: "/etc/..." is "\\etc\\..." on Windows
    context = CliContext(
        config_path=config,
        overrides={"paths": {"data_dir": "D:\\数据"}, "ops": {"supervise": {"backoff_start_s": 2}}},
        log_level="DEBUG",
    )
    options = global_options(context)
    assert options[:2] == ["--config", str(config)] and options[-2:] == [
        "--log-level",
        "DEBUG",
    ]
    sets = [options[i + 1] for i, item in enumerate(options) if item == "--set"]
    assert parse_overrides(sets) == context.overrides  # what the child parses is what was given
    assert global_options(CliContext()) == []


def test_the_task_runs_twin_exe_of_this_environment_when_there_is_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts = tmp_path / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "python.exe").write_bytes(b"")
    (scripts / "twin.exe").write_bytes(b"")
    monkeypatch.setenv("USERDOMAIN", "PC")
    monkeypatch.setenv("USERNAME", "me")
    spec = task_spec(tmp_path, ["--set", "a=1"], python=scripts / "python.exe")
    assert spec.command == str(scripts / "twin.exe")
    assert spec.arguments == "--set a=1 supervise --from-task"
    assert spec.working_dir == str(tmp_path) and spec.user_id == "PC\\me"
    (scripts / "twin.exe").unlink()
    fallback = task_spec(tmp_path, [], python=scripts / "python.exe")
    assert fallback.command == str(scripts / "python.exe")
    assert fallback.arguments == "-X utf8 -m twin supervise --from-task"
    monkeypatch.delenv("USERDOMAIN")
    assert task_spec(tmp_path, [], python=scripts / "python.exe").user_id == "me"


# ----------------------------------------------------------------------------- stopping


def hold(locks_dir: Path, name: str, win32: FakeWin32 | None = None) -> InstanceLock:
    lock = InstanceLock(name, locks_dir=locks_dir, platform="win32" if win32 else None, win32=win32)
    assert lock.acquire()
    return lock


def test_nothing_running_means_nothing_to_stop_and_a_stale_request_is_removed(
    tmp_path: Path,
) -> None:
    stale = stop_file_path(tmp_path)
    stale.write_text("stop\n", encoding="utf-8")
    assert stop_service(tmp_path, scheduler=None, grace_s=1) == StopOutcome(was_running=False)
    assert not stale.exists()
    assert running(tmp_path) == (False, False)


def test_a_run_without_a_supervisor_cannot_be_asked_to_stop(tmp_path: Path) -> None:
    lock = hold(tmp_path, LOCK_RUN)
    try:
        outcome = stop_service(tmp_path, scheduler=None, grace_s=1)
    finally:
        lock.release()
    assert outcome == StopOutcome(was_running=True, manual_run=True, still_running=True)
    assert not stop_file_path(tmp_path).exists()  # nobody was asked


def test_the_supervisor_is_asked_with_a_file_and_stops_in_its_own_time(tmp_path: Path) -> None:
    supervisor, child = hold(tmp_path, LOCK_SUPERVISOR), hold(tmp_path, LOCK_RUN)
    waits: list[float] = []

    def waiter(seconds: float) -> None:
        waits.append(seconds)
        assert stop_file_path(tmp_path).read_text(encoding="utf-8") == "stop\n"  # asked
        if len(waits) == 3:  # the supervisor saw the file, the child saved its state and ended
            child.release()
            supervisor.release()

    outcome = stop_service(tmp_path, scheduler=None, grace_s=40, waiter=waiter)
    assert outcome == StopOutcome(was_running=True, graceful=True)
    assert len(waits) == 3 and not stop_file_path(tmp_path).exists()


def test_a_supervisor_that_does_not_stop_is_ended_through_the_task(tmp_path: Path) -> None:
    supervisor = hold(tmp_path, LOCK_SUPERVISOR)
    ended: list[bool] = []

    def end(args: list[str]) -> CommandResult:
        ended.append(True)
        supervisor.release()  # Windows killed the process: the lock goes with it
        return ok()

    scheduler = TaskScheduler(ScriptedRunner({"schtasks.exe /End": end}))
    calls: list[float] = []
    outcome = stop_service(tmp_path, scheduler=scheduler, grace_s=1, waiter=calls.append)
    assert outcome == StopOutcome(was_running=True, forced=True)
    assert ended == [True] and len(calls) == int((1 + 15) / 0.5)  # all of the grace time first
    assert not stop_file_path(tmp_path).exists()


def test_when_even_the_task_cannot_be_ended_the_process_is_reported_as_still_running(
    tmp_path: Path,
) -> None:
    supervisor = hold(tmp_path, LOCK_SUPERVISOR)
    try:
        scheduler = TaskScheduler(ScriptedRunner({"schtasks.exe /End": failed("denied")}))
        outcome = stop_service(tmp_path, scheduler=scheduler, grace_s=0.5, waiter=lambda s: None)
        assert outcome.still_running and not outcome.forced and outcome.was_running
        none = stop_service(tmp_path, scheduler=None, grace_s=0.5, waiter=lambda s: None)
        assert none.still_running
    finally:
        supervisor.release()


def test_the_default_waiter_really_waits() -> None:
    import time

    start = time.monotonic()
    default_waiter(0.05)
    assert time.monotonic() - start >= 0.04


# ----------------------------------------------------------------------------- status


def test_the_restart_summary_counts_the_last_day(services: Services) -> None:
    now = datetime(2026, 10, 9, 12, tzinfo=UTC)
    log = RestartLog(services.db, services.clock)
    assert summarize_restarts(log, now).total == 0 and summarize_restarts(log, now).last_at is None
    for hours_ago, code in ((30, 1), (5, 2), (1, 3)):
        log.add(Restart(now - timedelta(hours=hours_ago), code, 10.0, 5.0, f"exit code {code}"))
    summary = summarize_restarts(log, now)
    assert (summary.last_24h, summary.total) == (2, 3)
    assert summary.last_at == now - timedelta(hours=1) and summary.last_reason == "exit code 3"


def test_the_status_combines_locks_task_and_history(services: Services, tmp_path: Path) -> None:
    log = RestartLog(services.db, services.clock)
    log.start_session("task")
    log.add(Restart(services.clock.now_utc(), 1, 3.0, 5.0, "exit code 1"))
    scripted = registered_runner(
        {"schtasks.exe /Query /TN wechat-twin /FO": ok('"PC","\\wechat-twin","N/A","Ready"\r\n')}
    )
    lock = hold(tmp_path, LOCK_SUPERVISOR)
    try:
        status = read_status(tmp_path, TaskScheduler(scripted), log, services.clock.now_utc())
    finally:
        lock.release()
    assert status.supervisor_running and not status.run_running
    assert status.task is not None and status.task.interactive and status.task_state == "Ready"
    assert status.restarts is not None and status.restarts.total == 1
    assert status.session is not None and status.session["launch"] == "task"
    broken = read_status(
        tmp_path,
        TaskScheduler(ScriptedRunner({})),  # nobody scripted schtasks: it cannot run
        None,
        services.clock.now_utc(),
    )
    assert broken.task is None and "cannot run" in str(broken.task_error)
    assert broken.restarts is None and broken.session is None
    unregistered = read_status(
        tmp_path,
        TaskScheduler(ScriptedRunner({"schtasks.exe /Query": failed()})),
        None,
        services.clock.now_utc(),
    )
    assert unregistered.task == TaskInfo(registered=False) and unregistered.task_state is None


# ----------------------------------------------------------------------------- the commands


@pytest.fixture
def home(services: Services, secret_store: SecretStore) -> Services:
    set_cli_context(CliContext(secrets=secret_store))
    return services


def twin(home: Services, *args: str) -> tuple[int, str]:
    options = ["--set", f"paths.data_dir={home.paths.data_dir}"]
    context = get_cli_context()
    previous_clock, context.clock = context.clock, home.clock  # the command reads the test's time
    try:
        result = runner.invoke(app, [*options, *args])
    finally:
        get_cli_context().clock = previous_clock
    return result.exit_code, result.output


def windows_env(
    scripted: ScriptedRunner,
    fake: FakeWin32 | None = None,
    waiter: Callable[[float], None] | None = None,
) -> ServiceEnv:
    return ServiceEnv(
        runner=scripted,
        platform="win32",
        waiter=waiter or (lambda seconds: None),
        win32=fake or FakeWin32(),
    )


def test_install_can_print_the_definition_on_any_platform(home: Services) -> None:
    code, out = twin(home, "service", "install", "--print-xml")
    assert code == 0 and "<LogonType>InteractiveToken</LogonType>" in out
    assert "supervise --from-task" in out and "paths.data_dir=" in out
    # The child gets the same data folder.  The option value is JSON (a backslash is written
    # twice) and the whole command line follows the Windows quoting rules (a quote is written
    # \" ), so on Windows the folder does not appear letter for letter; what must hold is that the
    # argument is the one the child parses back into this folder.
    data_dir = str(home.paths.data_dir)
    argument = f"paths.data_dir={json.dumps(data_dir, ensure_ascii=False)}"
    root = ET.fromstring(out.split("?>", 1)[1])  # noqa: S314 - our own text
    arguments = root.findtext(f"{tag('Actions')}/{tag('Exec')}/{tag('Arguments')}")
    assert arguments is not None and subprocess.list2cmdline(["--set", argument]) in arguments
    assert parse_overrides([argument]) == {"paths": {"data_dir": data_dir}}
    assert "uv sync" not in out


def test_outside_windows_the_task_commands_say_so(home: Services) -> None:
    with use_service_env(ServiceEnv(platform="linux")):
        for command in ("install", "uninstall", "start"):
            code, out = twin(home, "service", command)
            assert code != 0 and "Windows only" in out, command
        code, out = twin(home, "service", "status")
        assert code == 0 and "scheduled task: not available" in out
        assert "nothing is running" in out
        code, out = twin(home, "service", "stop")
        assert code == 0 and "nothing was running" in out


def test_install_registers_checks_and_explains(home: Services) -> None:
    scripted = registered_runner({"schtasks.exe /Create": ok("SUCCESS")})
    with use_service_env(windows_env(scripted)):
        code, out = twin(home, "service", "install")
    assert code == 0, out
    assert f"scheduled task '{TASK_NAME}' registered" in out
    assert "logon type: InteractiveToken" in out and "time limit: PT0S" in out
    assert "twin service start" in out and AUTOLOGON_NOTE.splitlines()[0] in out
    assert "自动登录" in out and "BitLocker" in out
    assert any(call[1] == "/Create" for call in scripted.calls)


def test_install_refuses_a_task_that_is_not_interactive(home: Services) -> None:
    s4u = build_task_xml(SPEC).replace("InteractiveToken", "S4U")
    scripted = ScriptedRunner({"schtasks.exe /Create": ok(), "schtasks.exe /Query": ok(s4u)})
    with use_service_env(windows_env(scripted)):
        code, out = twin(home, "service", "install")
    assert code != 0 and "S4U" in out and "InteractiveToken" in out


def test_install_reports_a_registration_that_windows_refused(home: Services) -> None:
    scripted = ScriptedRunner({"schtasks.exe /Create": failed("ERROR: Access is denied.")})
    with use_service_env(windows_env(scripted)):
        code, out = twin(home, "service", "install")
    assert code != 0 and "Access is denied" in out


def test_install_and_uninstall_are_exclusive_commands(home: Services) -> None:
    lock = hold(home.paths.locks_dir, LOCK_RUN)
    try:
        with use_service_env(windows_env(registered_runner({"schtasks.exe /Create": ok()}))):
            code, out = twin(home, "service", "install")
            code2, _ = twin(home, "service", "uninstall")
    finally:
        lock.release()
    assert code == ExitCode.BUSY == code2 and "twin service stop" in out


def test_uninstall_removes_the_task_and_leaves_the_data(home: Services) -> None:
    scripted = registered_runner({"schtasks.exe /Delete": ok()})
    with use_service_env(windows_env(scripted)):
        code, out = twin(home, "service", "uninstall")
    assert code == 0 and "scheduled task removed" in out and home.paths.db_path.is_file()
    with use_service_env(windows_env(ScriptedRunner({"schtasks.exe /Query": failed()}))):
        code, out = twin(home, "service", "uninstall")
    assert code == 0 and "was not registered" in out
    scripted = registered_runner({"schtasks.exe /Delete": failed("ERROR: denied")})
    with use_service_env(windows_env(scripted)):
        code, out = twin(home, "service", "uninstall")
    assert code != 0 and "removing the scheduled task failed" in out


def test_start_runs_the_task_unless_it_is_missing_or_already_running(home: Services) -> None:
    scripted = registered_runner({"schtasks.exe /Run": ok()})
    with use_service_env(windows_env(scripted)):
        code, out = twin(home, "service", "start")
    assert (
        code == 0
        and "started" in out
        and ["schtasks.exe", "/Run", "/TN", TASK_NAME] in scripted.calls
    )
    with use_service_env(windows_env(ScriptedRunner({"schtasks.exe /Query": failed()}))):
        code, out = twin(home, "service", "start")
    assert code != 0 and "not registered" in out
    fake = FakeWin32()
    lock = hold(home.paths.locks_dir, LOCK_SUPERVISOR, fake)
    try:
        with use_service_env(windows_env(registered_runner({}), fake)):
            code, out = twin(home, "service", "start")
    finally:
        lock.release()
    assert code == 0 and "already running" in out
    with use_service_env(windows_env(registered_runner({"schtasks.exe /Run": failed("no")}))):
        code, out = twin(home, "service", "start")
    assert code != 0 and "starting the scheduled task failed" in out


def test_stop_is_graceful_when_the_supervisor_ends_after_the_request(home: Services) -> None:
    fake = FakeWin32()
    supervisor = hold(home.paths.locks_dir, LOCK_SUPERVISOR, fake)
    child = hold(home.paths.locks_dir, LOCK_RUN, fake)
    asked: list[bool] = []

    def waiter(seconds: float) -> None:
        asked.append(stop_file_path(home.paths.locks_dir).exists())
        child.release()
        supervisor.release()

    scripted = ScriptedRunner({})
    with use_service_env(windows_env(scripted, fake, waiter)):
        code, out = twin(home, "service", "stop")
    assert code == 0 and "stopped gracefully" in out and asked == [True]
    assert scripted.calls == []  # the task was not ended: nobody had to be killed


def test_stop_reports_a_forced_end_a_manual_run_and_a_process_that_stays(home: Services) -> None:
    fake = FakeWin32()
    supervisor = hold(home.paths.locks_dir, LOCK_SUPERVISOR, fake)

    def end(args: list[str]) -> CommandResult:
        supervisor.release()
        return ok()

    with use_service_env(windows_env(ScriptedRunner({"schtasks.exe /End": end}), fake)):
        code, out = twin(home, "service", "stop")
    assert code == 0 and "forced (the task was ended)" in out
    stubborn = hold(home.paths.locks_dir, LOCK_SUPERVISOR, fake)
    try:
        with use_service_env(
            windows_env(ScriptedRunner({"schtasks.exe /End": failed("no")}), fake)
        ):
            code, out = twin(home, "service", "stop")
    finally:
        stubborn.release()
    assert code != 0 and "did not stop in time" in out
    manual = hold(home.paths.locks_dir, LOCK_RUN, fake)
    try:
        with use_service_env(windows_env(ScriptedRunner({}), fake)):
            code, out = twin(home, "service", "stop")
    finally:
        manual.release()
    assert code == ExitCode.BUSY and "Ctrl+C" in out


def test_status_shows_the_task_the_processes_and_the_restarts(home: Services) -> None:
    log = RestartLog(home.db, home.clock)
    log.start_session("task")
    log.add(Restart(home.clock.now_utc(), 1, 3.0, 5.0, "exit code 1"))
    fake = FakeWin32()
    supervisor = hold(home.paths.locks_dir, LOCK_SUPERVISOR, fake)
    try:
        scripted = registered_runner(
            {
                "schtasks.exe /Query /TN wechat-twin /FO": ok(
                    '"PC","\\wechat-twin","N/A","Running"\r\n'
                )
            }
        )
        with use_service_env(windows_env(scripted, fake)):
            code, out = twin(home, "service", "status")
    finally:
        supervisor.release()
    assert code == 0, out
    assert "scheduled task: registered, status Running" in out and "InteractiveToken" in out
    assert "supervisor (twin supervise): running" in out
    assert "application (twin run):      not running" in out
    assert "(task)" in out and "restarts: 1 in the last 24 hours, 1 recorded in all" in out
    assert "last restart" in out and "exit code 1" in out
    with use_service_env(windows_env(ScriptedRunner({"schtasks.exe /Query": failed()}))):
        code, out = twin(home, "service", "status")
    assert "NOT REGISTERED" in out and "nothing is running" in out
    with use_service_env(windows_env(ScriptedRunner({}))):
        code, out = twin(home, "service", "status")
    assert "scheduled task: not available (cannot run" in out


# ------------------------------------------------------------------- twin supervise itself


def fast_child(monkeypatch: pytest.MonkeyPatch, script: str, *arguments: str) -> None:
    monkeypatch.setattr(
        "twin.ops.service_cli.child_arguments",
        lambda options: [sys.executable, "-c", script, *arguments],
    )


def test_supervise_runs_the_child_to_a_normal_end_and_notes_how_it_was_started(
    home: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    fast_child(monkeypatch, "pass")
    code, out = twin(home, "supervise", "--from-task")
    assert code == 0, out
    session = RestartLog(home.db, home.clock).session()
    assert session is not None and session["launch"] == "task"
    assert not InstanceLock(LOCK_SUPERVISOR, locks_dir=home.paths.locks_dir).is_held_elsewhere()
    code, out = twin(home, "supervise")
    session = RestartLog(home.db, home.clock).session()
    assert code == 0 and session is not None and session["launch"] == "manual"
    log = (home.paths.logs_dir / "twin-supervise.log").read_text(encoding="utf-8")
    assert "supervisor_starting" in log and "child_exited_normally" in log  # its own log file
    assert '"launch": "task"' in log or '"launch":"task"' in log


def test_supervise_restarts_a_child_that_fails_and_writes_the_history(
    home: Services, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = (
        "import pathlib, sys\n"
        "p = pathlib.Path(sys.argv[1])\n"
        "n = int(p.read_text()) if p.exists() else 0\n"
        "p.write_text(str(n + 1))\n"
        "sys.exit(1 if n == 0 else 0)\n"
    )
    counter = tmp_path / "runs"
    fast_child(monkeypatch, script, str(counter))
    result = runner.invoke(
        app,
        [
            "--set",
            f"paths.data_dir={home.paths.data_dir}",
            "--set",
            "ops.supervise.backoff_start_s=0.05",
            "supervise",
        ],
    )
    assert result.exit_code == 0, result.output
    assert counter.read_text() == "2"
    (entry,) = RestartLog(home.db, home.clock).entries()
    assert entry["exit_code"] == 1 and entry["reason"] == "exit code 1"


def test_the_supervisor_lock_is_held_while_it_waits_so_exclusive_commands_are_refused(
    home: Services, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R-OPS-001: between two children (the backoff) the lock is still the supervisor's."""
    from twin.ops import supervise as supervise_module

    script = "import sys; sys.exit(1)"
    fast_child(monkeypatch, script)
    seen: list[tuple[list[str], int, str]] = []
    original = supervise_module.SubprocessLauncher.start
    options = ["--set", f"paths.data_dir={home.paths.data_dir}"]

    async def spying_start(self: object, env: object) -> object:
        held = locks_held_elsewhere(home.paths.locks_dir, (LOCK_SUPERVISOR, LOCK_RUN))
        if len(seen) == 1:  # the second child: the supervisor has waited out its backoff
            refused = runner.invoke(app, [*options, "db", "upgrade"])
            seen.append((held, refused.exit_code, refused.output))
        else:
            seen.append((held, -1, ""))
        if len(seen) >= 3:
            stop_file_path(home.paths.locks_dir).write_text("stop\n", encoding="utf-8")
        return await original(self, env)  # type: ignore[arg-type]

    monkeypatch.setattr(supervise_module.SubprocessLauncher, "start", spying_start)
    result = runner.invoke(
        app, [*options, "--set", "ops.supervise.backoff_start_s=0.05", "supervise"]
    )
    assert result.exit_code == 0, result.output
    assert all(
        held == [LOCK_SUPERVISOR] for held, _, _ in seen
    )  # the child's `run` lock is not ours
    refused = seen[1]
    assert refused[1] == ExitCode.BUSY and "twin service stop" in refused[2]


def test_supervise_tolerates_the_run_lock_of_its_child_but_not_a_second_supervisor(
    home: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    fast_child(monkeypatch, "pass")
    child = hold(home.paths.locks_dir, LOCK_RUN)
    try:
        code, out = twin(home, "supervise")
    finally:
        child.release()
    assert code == 0, out
    other = hold(home.paths.locks_dir, LOCK_SUPERVISOR)
    try:
        code, out = twin(home, "supervise")
    finally:
        other.release()
    assert code == ExitCode.BUSY and "supervisor" in out


def test_supervise_works_when_the_database_needs_a_migration(
    home: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The child will say what is wrong; the supervisor must not be the one that fails."""
    from twin.storage import migrate

    home.db.dispose()
    migrate.downgrade(home.paths.db_path, "-1")
    fast_child(monkeypatch, "pass")
    code, out = twin(home, "supervise")
    assert code == 0, out
