"""The scheduled task that keeps the bot running (R-OPS-001).

**Why ``schtasks`` and XML, not the Task Scheduler COM interface**: the task needs a handful of
exact settings (logon type ``InteractiveToken``, no time limit, runs on battery, one instance,
restart every minute after a failure) and an XML definition states every one of them; the
command-line switches of ``schtasks`` cannot express most of them.  ``schtasks.exe`` is part of
Windows, so no extra package (``pywin32``, ``comtypes``) is needed, ``subprocess`` with an
argument list and ``shell=False`` is enough, and the very same XML can be read back
(``schtasks /Query /XML``) for ``twin doctor`` to check what is registered.  The XML is generated
by :func:`build_task_xml` from a :class:`TaskSpec`, so the registration is reproducible and the
text is tested on every platform.

The task, in the words of the Task Scheduler:

* **trigger** - when the user logs on;
* **principal** - that user, logon type ``InteractiveToken``: it runs only while the user is
  logged on, which is what the credential manager, the notifications and the QR window need.
  Not ``S4U``, not a stored password, and not a Windows service (a service account cannot read
  the user's credential manager);
* **action** - ``<venv>\\Scripts\\twin.exe supervise --from-task`` in the repository folder.  It
  never runs ``uv sync``: starting up must not change the environment;
* **settings** - no execution time limit, also on battery, a second instance is not started, and
  "restart on failure" every minute (999 times) as the second line of defence behind
  ``twin supervise``'s own restarts.

Everything that touches ``schtasks`` goes through a :class:`CommandRunner`, so the logic is tested
with a recording runner; the real one is :class:`SubprocessRunner`.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Protocol
from xml.sax.saxutils import escape

TASK_NAME = "wechat-twin"
TASK_NS = "{http://schemas.microsoft.com/windows/2004/02/mit/task}"
SCHTASKS = "schtasks.exe"
RESTART_INTERVAL = "PT1M"
RESTART_COUNT = 999
SUPERVISE_ARGUMENTS = ("supervise", "--from-task")


class TaskSchedulerError(Exception):
    """``schtasks`` failed or the task is not what it should be; the message says what."""


@dataclass(frozen=True)
class TaskSpec:
    """What the task runs and as whom."""

    command: str
    arguments: str
    working_dir: str
    user_id: str
    description: str = "wechat-twin: keeps the bot running (twin supervise)"


def build_task_xml(spec: TaskSpec) -> str:
    """The Task Scheduler XML for ``spec`` (UTF-16 declaration; see the module description)."""
    user = escape(spec.user_id)
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>{escape(spec.description)}</Description>
    <URI>\\{TASK_NAME}</URI>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <UserId>{user}</UserId>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{user}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority>
    <RestartOnFailure>
      <Interval>{RESTART_INTERVAL}</Interval>
      <Count>{RESTART_COUNT}</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{escape(spec.command)}</Command>
      <Arguments>{escape(spec.arguments)}</Arguments>
      <WorkingDirectory>{escape(spec.working_dir)}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


@dataclass(frozen=True)
class TaskInfo:
    """What the registered task says about itself (read from its XML)."""

    registered: bool
    logon_type: str | None = None
    command: str | None = None
    arguments: str | None = None
    working_dir: str | None = None
    enabled: bool | None = None
    time_limit: str | None = None
    multiple_instances: str | None = None
    on_battery_allowed: bool | None = None
    restart_interval: str | None = None
    runs_uv_sync: bool = False

    @property
    def interactive(self) -> bool:
        return self.logon_type == "InteractiveToken"


def parse_task_xml(text: str) -> TaskInfo:
    """Read the facts the doctor checks out of a task's XML."""
    cleaned = text.lstrip("﻿")
    if cleaned.startswith("<?xml"):  # the declaration says UTF-16 although the text is a str
        cleaned = cleaned.split("?>", 1)[1]
    try:
        root = ET.fromstring(cleaned)  # noqa: S314 - XML printed by schtasks.exe on this machine
    except ET.ParseError as exc:
        raise TaskSchedulerError(f"the task definition cannot be read: {exc}") from None

    def text_of(*names: str) -> str | None:
        node: ET.Element | None = root
        for name in names:
            node = node.find(TASK_NS + name) if node is not None else None
        return node.text.strip() if node is not None and node.text else None

    command = text_of("Actions", "Exec", "Command")
    arguments = text_of("Actions", "Exec", "Arguments")
    program = PureWindowsPath(command or "").stem.lower()
    battery = text_of("Settings", "DisallowStartIfOnBatteries")
    enabled = text_of("Settings", "Enabled")
    return TaskInfo(
        registered=True,
        logon_type=text_of("Principals", "Principal", "LogonType"),
        command=command,
        arguments=arguments,
        working_dir=text_of("Actions", "Exec", "WorkingDirectory"),
        enabled=None if enabled is None else enabled.lower() == "true",
        time_limit=text_of("Settings", "ExecutionTimeLimit"),
        multiple_instances=text_of("Settings", "MultipleInstancesPolicy"),
        on_battery_allowed=None if battery is None else battery.lower() == "false",
        restart_interval=text_of("Settings", "RestartOnFailure", "Interval"),
        runs_uv_sync=program == "uv" and "sync" in (arguments or "").lower().split(),
    )


# ----------------------------------------------------------------------- running schtasks


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes


class CommandRunner(Protocol):
    """Runs a program with an argument list (never through a shell)."""

    def run(self, args: list[str], *, timeout_s: float = 30.0) -> CommandResult: ...


class SubprocessRunner:
    """The real runner: :func:`subprocess.run` with ``shell=False``."""

    def run(self, args: list[str], *, timeout_s: float = 30.0) -> CommandResult:
        try:
            done = subprocess.run(  # noqa: S603 - fixed program, argument list, no shell
                args, capture_output=True, timeout=timeout_s, check=False, shell=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TaskSchedulerError(f"cannot run {args[0]}: {type(exc).__name__}") from None
        return CommandResult(done.returncode, done.stdout, done.stderr)


def decode_output(raw: bytes) -> str:
    """Text printed by a console program: UTF-16 with a byte order mark, UTF-8, or the ANSI page."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("mbcs" if os.name == "nt" else "latin-1", errors="replace")


def _failure(result: CommandResult, what: str) -> TaskSchedulerError:
    detail = decode_output(result.stderr or result.stdout).strip().splitlines()
    tail = detail[-1][:200] if detail else f"exit code {result.returncode}"
    return TaskSchedulerError(f"{what} failed: {tail}")


class TaskScheduler:
    """Registers, starts, stops and reads the task (see the module description)."""

    def __init__(self, runner: CommandRunner, name: str = TASK_NAME) -> None:
        self._runner = runner
        self.name = name

    def _schtasks(self, *args: str) -> CommandResult:
        return self._runner.run([SCHTASKS, *args])

    def install(self, spec: TaskSpec) -> None:
        """Register (or replace) the task from ``spec``."""
        xml = build_task_xml(spec)
        with tempfile.TemporaryDirectory(prefix="twin-task-") as folder:
            path = Path(folder) / "task.xml"
            path.write_bytes(b"\xff\xfe" + xml.encode("utf-16-le"))
            result = self._schtasks("/Create", "/TN", self.name, "/XML", str(path), "/F")
        if result.returncode != 0:
            raise _failure(result, "registering the scheduled task")

    def uninstall(self) -> bool:
        """Remove the task; ``False`` if it was not registered."""
        if not self.info().registered:
            return False
        result = self._schtasks("/Delete", "/TN", self.name, "/F")
        if result.returncode != 0:
            raise _failure(result, "removing the scheduled task")
        return True

    def start(self) -> None:
        result = self._schtasks("/Run", "/TN", self.name)
        if result.returncode != 0:
            raise _failure(result, "starting the scheduled task")

    def end(self) -> None:
        """Ask Windows to stop the task's process (a hard stop; ``twin service stop`` asks nicely
        first)."""
        result = self._schtasks("/End", "/TN", self.name)
        if result.returncode != 0:
            raise _failure(result, "stopping the scheduled task")

    def info(self) -> TaskInfo:
        """The registered task, or ``TaskInfo(registered=False)``."""
        result = self._schtasks("/Query", "/TN", self.name, "/XML")
        if result.returncode != 0:
            return TaskInfo(registered=False)
        return parse_task_xml(decode_output(result.stdout))

    def state(self) -> str | None:
        """The status column of ``schtasks /Query`` (as printed in the language of Windows)."""
        result = self._schtasks("/Query", "/TN", self.name, "/FO", "CSV", "/NH", "/V")
        if result.returncode != 0:
            return None
        import csv
        import io

        rows = list(csv.reader(io.StringIO(decode_output(result.stdout))))
        return rows[0][3] if rows and len(rows[0]) > 3 else None
