"""``twin doctor``: environment and installation checks (R-OPS-009, base items).

Each check returns OK, WARN or FAIL with a one-line detail and, when the user can
act on it, a hint.  FAIL means the application cannot run correctly; WARN means
it can run but something is worth knowing (for example the encrypted-file
fallback is used instead of a system keyring).  Later rounds append checks with
:func:`doctor_check` (network, GPU, scheduled task, calendar coverage, ...).
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import re
import secrets as pysecrets
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from twin.channel.ilink.connectivity import EndpointProbe, probe_api, probe_cdn
from twin.clock import now_utc
from twin.config.loader import DataPaths, ensure_consent, resolve_paths
from twin.config.secrets import SecretStore, SecretStoreError
from twin.config.settings import Settings
from twin.llm.official import VISION_MODELS
from twin.llm.pricing import PeakCalendar
from twin.ops.instance_lock import ALL_LOCKS, locks_held_elsewhere
from twin.ops.power import describe_power_strategy
from twin.ops.taskscheduler import (
    CommandResult,
    CommandRunner,
    SubprocessRunner,
    TaskScheduler,
    TaskSchedulerError,
    decode_output,
)
from twin.retrieval.embedder import read_manifest
from twin.storage.keystore import KeyStore, KeyStoreError
from twin.storage.migrate import SchemaState, schema_status

MIN_FREE_FAIL_BYTES = 1 * 1024**3
MIN_FREE_WARN_BYTES = 5 * 1024**3

REQUIRED_MODULES = (
    "typer",
    "rich",
    "pydantic",
    "pydantic_settings",
    "yaml",
    "sqlalchemy",
    "alembic",
    "cryptography",
    "keyring",
    "httpx",
    "openai",
    "ijson",
    "tzdata",
    "chinese_calendar",
    "holidays",
    "orjson",
    "PIL",
)
_DISTRIBUTIONS = {"yaml": "PyYAML", "chinese_calendar": "chinese-calendar", "PIL": "pillow"}


class CheckStatus(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    detail: str
    hint: str = ""


@dataclass
class DoctorContext:
    """Inputs of the checks; ``settings`` is ``None`` if the configuration did not load."""

    settings: Settings | None
    settings_error: str | None = None
    secrets: SecretStore | None = None
    root: Path | None = None
    platform: str = field(default_factory=lambda: sys.platform)
    http_transport: httpx.BaseTransport | None = None  # network checks use it when given
    runner: CommandRunner | None = None  # nvidia-smi, powercfg and schtasks go through it
    calendar: PeakCalendar | None = (
        None  # the holiday library to check (default: the installed one)
    )

    def run(self, args: list[str]) -> CommandResult | None:
        """Run a program (no shell); ``None`` if it is not installed or does not answer."""
        runner = self.runner or SubprocessRunner()
        try:
            return runner.run(args, timeout_s=15.0)
        except TaskSchedulerError:
            return None

    def paths(self) -> DataPaths | None:
        return resolve_paths(self.settings, self.root) if self.settings else None


DoctorCheck = Callable[[DoctorContext], CheckResult]
_CHECKS: list[DoctorCheck] = []


def doctor_check(func: DoctorCheck) -> DoctorCheck:
    """Register a check (executed in registration order)."""
    _CHECKS.append(func)
    return func


@doctor_check
def check_python(ctx: DoctorContext) -> CheckResult:
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info[:2] == (3, 12):
        return CheckResult("python", CheckStatus.OK, f"Python {version}")
    return CheckResult(
        "python",
        CheckStatus.FAIL,
        f"Python {version} (this project requires >=3.12,<3.13)",
        "install Python 3.12 (`uv python install 3.12`) and run `uv sync`",
    )


@doctor_check
def check_dependencies(ctx: DoctorContext) -> CheckResult:
    missing: list[str] = []
    versions: list[str] = []
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:
            missing.append(f"{module} ({type(exc).__name__})")
            continue
        try:
            versions.append(
                f"{module}={importlib.metadata.version(_DISTRIBUTIONS.get(module, module))}"
            )
        except importlib.metadata.PackageNotFoundError:
            versions.append(module)
    if missing:
        return CheckResult(
            "dependencies",
            CheckStatus.FAIL,
            "cannot import: " + ", ".join(missing),
            "run `uv sync`",
        )
    return CheckResult("dependencies", CheckStatus.OK, f"{len(versions)} packages import")


@doctor_check
def check_tzdata(ctx: DoctorContext) -> CheckResult:
    try:
        importlib.import_module("tzdata")
        zones = [ZoneInfo("America/Chicago"), ZoneInfo("Asia/Shanghai")]
    except Exception as exc:
        return CheckResult(
            "tzdata",
            CheckStatus.FAIL,
            f"time zone database unavailable: {type(exc).__name__}: {exc}",
            "install the `tzdata` package (Windows has no system IANA database)",
        )
    return CheckResult("tzdata", CheckStatus.OK, ", ".join(zone.key for zone in zones))


@doctor_check
def check_config(ctx: DoctorContext) -> CheckResult:
    if ctx.settings is None:
        return CheckResult(
            "config", CheckStatus.FAIL, ctx.settings_error or "configuration did not load"
        )
    try:
        confirmed = ensure_consent(ctx.settings)
    except Exception as exc:
        return CheckResult(
            "config",
            CheckStatus.FAIL,
            str(exc),
            "set consent.confirmed_at (YYYY-MM-DD) in config/config.yaml",
        )
    return CheckResult("config", CheckStatus.OK, f"loaded; consent confirmed {confirmed}")


@doctor_check
def check_keyring(ctx: DoctorContext) -> CheckResult:
    try:
        store = ctx.secrets or SecretStore.default()
    except SecretStoreError as exc:
        return CheckResult("keyring", CheckStatus.FAIL, str(exc))
    info = store.info
    if not info.usable:
        hint = (
            "unlock/enable Windows Credential Manager for this user"
            if ctx.platform == "win32"
            else "install a Secret Service keyring, or fix the encrypted file store location"
            " (TWIN_SECRETS_DIR) and permissions"
        )
        return CheckResult("keyring", CheckStatus.FAIL, f"{info.name}: {info.detail}", hint)
    probe = "doctor-probe-" + pysecrets.token_hex(4)
    try:
        store.set(probe, "ok")
        readback = store.get(probe)
        store.delete(probe)
    except Exception as exc:
        return CheckResult(
            "keyring", CheckStatus.FAIL, f"{info.name}: read/write test failed: {exc}"
        )
    if readback != "ok":
        return CheckResult("keyring", CheckStatus.FAIL, f"{info.name}: read-back mismatch")
    if info.kind == "file":
        return CheckResult(
            "keyring",
            CheckStatus.WARN,
            f"read/write ok via encrypted file fallback: {info.detail}",
            "expected on Linux without a desktop keyring; Windows uses Credential Manager. "
            "Set TWIN_KEYRING_PASSPHRASE to bind the file to a passphrase instead of a key file",
        )
    return CheckResult("keyring", CheckStatus.OK, f"{info.name}: read/write ok")


@doctor_check
def check_database_key(ctx: DoctorContext) -> CheckResult:
    if ctx.settings is None:
        return CheckResult("db-key", CheckStatus.WARN, "skipped (configuration did not load)")
    try:
        store = ctx.secrets or SecretStore.default()
        keystore = KeyStore(store)
        if not keystore.exists():
            return CheckResult(
                "db-key", CheckStatus.OK, "no master key yet (created on first start)"
            )
        ring = keystore.load()
    except (SecretStoreError, KeyStoreError) as exc:
        return CheckResult("db-key", CheckStatus.FAIL, str(exc))
    retired = len(ring.retired_ids)
    return CheckResult(
        "db-key",
        CheckStatus.OK,
        f"master key id {ring.current_id} (AES-256-GCM), {retired} retired key(s) kept",
    )


@doctor_check
def check_data_dir(ctx: DoctorContext) -> CheckResult:
    paths = ctx.paths()
    if paths is None:
        return CheckResult("data-dir", CheckStatus.WARN, "skipped (configuration did not load)")
    try:
        paths.data_dir.mkdir(parents=True, exist_ok=True)
        probe = paths.data_dir / f".doctor-{pysecrets.token_hex(4)}"
        probe.write_bytes(b"ok")
        probe.unlink()
        free = shutil.disk_usage(paths.data_dir).free
    except OSError as exc:
        return CheckResult(
            "data-dir",
            CheckStatus.FAIL,
            f"{paths.data_dir} is not writable: {exc}",
            "choose another paths.data_dir or fix permissions",
        )
    free_gb = free / 1024**3
    if free < MIN_FREE_FAIL_BYTES:
        return CheckResult(
            "data-dir",
            CheckStatus.FAIL,
            f"{paths.data_dir}: only {free_gb:.1f} GB free",
            "free disk space (database, media and backups live here)",
        )
    if free < MIN_FREE_WARN_BYTES:
        return CheckResult(
            "data-dir", CheckStatus.WARN, f"{paths.data_dir}: {free_gb:.1f} GB free (low)"
        )
    return CheckResult(
        "data-dir", CheckStatus.OK, f"{paths.data_dir} writable, {free_gb:.0f} GB free"
    )


@doctor_check
def check_database(ctx: DoctorContext) -> CheckResult:
    paths = ctx.paths()
    if paths is None:
        return CheckResult("database", CheckStatus.WARN, "skipped (configuration did not load)")
    status = schema_status(paths.db_path)
    match status.state:
        case SchemaState.CURRENT:
            return CheckResult("database", CheckStatus.OK, f"schema {status.current} (up to date)")
        case SchemaState.MISSING | SchemaState.EMPTY:
            return CheckResult(
                "database",
                CheckStatus.WARN,
                "not initialised yet",
                "run `twin db upgrade` before the first start",
            )
        case SchemaState.OUTDATED | SchemaState.AHEAD:
            return CheckResult("database", CheckStatus.FAIL, status.hint(), status.hint())


@doctor_check
def check_instances(ctx: DoctorContext) -> CheckResult:
    paths = ctx.paths()
    if paths is None:
        return CheckResult("instances", CheckStatus.WARN, "skipped (configuration did not load)")
    held = locks_held_elsewhere(paths.locks_dir, ALL_LOCKS, platform=ctx.platform)
    if held:
        return CheckResult("instances", CheckStatus.OK, "running: " + ", ".join(held))
    return CheckResult("instances", CheckStatus.OK, "no instance is running")


@doctor_check
def check_power(ctx: DoctorContext) -> CheckResult:
    return CheckResult("power", CheckStatus.OK, describe_power_strategy(ctx.platform))


@doctor_check
def check_holiday_calendar(ctx: DoctorContext) -> CheckResult:
    """The holiday library must cover this year and the next (R-LLM-007, R-OPS-009)."""
    today = now_utc().date()
    coverage = (ctx.calendar or PeakCalendar()).coverage(today)
    if coverage.complete:
        years = ", ".join(str(year) for year in coverage.covered_years)
        return CheckResult("holiday-calendar", CheckStatus.OK, f"chinese-calendar covers {years}")
    missing = ", ".join(str(year) for year in coverage.missing_years)
    return CheckResult(
        "holiday-calendar",
        CheckStatus.WARN,
        f"chinese-calendar has no data for {missing}: peak hours there assume Monday to Friday",
        "upgrade the dependency when a new release appears (`uv lock --upgrade-package "
        "chinese-calendar`); until then list exceptions in pricing.extra_offpeak_dates and "
        "pricing.extra_peak_dates",
    )


VECTOR_MODULES = ("numpy", "pyarrow", "lancedb", "torch", "sentence_transformers")


@doctor_check
def check_vector_model(ctx: DoctorContext) -> CheckResult:
    """The vector libraries are installed and the embedding model is (or will be) available.

    Nothing is imported: ``torch`` takes seconds to load and the model is loaded lazily on first
    use (R-NFR-003); a missing model is downloaded then, so it is only a note here (R-RET-002).
    """
    missing = [name for name in VECTOR_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        return CheckResult(
            "vector-model",
            CheckStatus.FAIL,
            "not installed: " + ", ".join(missing),
            "run `uv sync`",
        )
    paths = ctx.paths()
    if ctx.settings is None or paths is None:
        return CheckResult("vector-model", CheckStatus.OK, "libraries are installed")
    model = ctx.settings.retrieval.model
    record = read_manifest(paths.embeddings_dir, model)
    if record is None:
        return CheckResult(
            "vector-model",
            CheckStatus.OK,
            f"{model} is downloaded on first use (about 100 MB; Hugging Face must be reachable "
            "once)",
        )
    revision = str(record.get("revision", ""))[:8]
    return CheckResult("vector-model", CheckStatus.OK, f"{model} downloaded (revision {revision})")


@doctor_check
def check_llm_config(ctx: DoctorContext) -> CheckResult:
    """The configured DeepSeek models have prices and the vision model can see (R-LLM-004)."""
    if ctx.settings is None:
        return CheckResult("llm-config", CheckStatus.WARN, "skipped (configuration did not load)")
    config = ctx.settings.deepseek
    prices = ctx.settings.pricing_usd_per_mtok
    problems: list[str] = []
    for role, model in (
        ("chat_model", config.chat_model),
        ("offline_model", config.offline_model),
        ("vision_model", config.vision_model),
    ):
        canonical = {"deepseek-v4-flash": "deepseek-flash"}.get(model, model)
        if canonical not in prices:
            problems.append(f"deepseek.{role} {model!r} has no entry in pricing_usd_per_mtok")
    if config.vision_model not in VISION_MODELS and config.vision_model != "deepseek-v4-flash":
        problems.append(f"deepseek.vision_model {config.vision_model!r} cannot read images")
    if problems:
        return CheckResult(
            "llm-config",
            CheckStatus.FAIL,
            "; ".join(problems),
            "fix the deepseek section of the configuration",
        )
    return CheckResult(
        "llm-config",
        CheckStatus.OK,
        f"chat={config.chat_model} offline={config.offline_model} vision={config.vision_model}",
    )


@doctor_check
def check_deepseek_key(ctx: DoctorContext) -> CheckResult:
    """Whether the DeepSeek API key is stored (a missing key is the normal state before M0)."""
    store = ctx.secrets
    try:
        if store is None:
            store = SecretStore.default()
        present = store.exists("deepseek_api_key")
    except Exception as exc:  # a locked or broken credential store must not stop diagnostics
        return CheckResult(
            "deepseek-key", CheckStatus.WARN, f"cannot read the key: {type(exc).__name__}: {exc}"
        )
    if present:
        return CheckResult("deepseek-key", CheckStatus.OK, "deepseek_api_key is set")
    return CheckResult(
        "deepseek-key",
        CheckStatus.WARN,
        "deepseek_api_key is not set",
        "run `twin secrets set deepseek_api_key`, then `twin llm probe`",
    )


def _channel_host_check(
    ctx: DoctorContext, name: str, probe: Callable[[httpx.BaseTransport | None], EndpointProbe]
) -> CheckResult:
    if ctx.settings is not None and ctx.settings.channel.kind != "ilink":
        return CheckResult(name, CheckStatus.OK, "not needed (the console channel is selected)")
    result = probe(ctx.http_transport)
    if result.reachable:
        return CheckResult(name, CheckStatus.OK, f"{result.url} {result.detail}")
    return CheckResult(
        name,
        CheckStatus.WARN,
        f"{result.url} {result.detail}",
        "the WeChat channel cannot work from this network: check the internet connection, a "
        "proxy or VPN, and whether this host is blocked (docs/ILINK_PROTOCOL.md section 13, "
        "item 15)",
    )


@doctor_check
def check_ilink_api(ctx: DoctorContext) -> CheckResult:
    """The iLink API host answers (R-CH-002)."""
    return _channel_host_check(ctx, "ilink-api", probe_api)


@doctor_check
def check_ilink_cdn(ctx: DoctorContext) -> CheckResult:
    """The WeChat media CDN host answers (R-CH-005, R-CH-006)."""
    return _channel_host_check(ctx, "ilink-cdn", probe_cdn)


# ----------------------------------------------------------- round 12: operations checks

DEEPSEEK_BALANCE_PATH = "/user/balance"
NVIDIA_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,driver_version,memory.total",
    "--format=csv,noheader",
]
POWERCFG_QUERY = ["powercfg", "/query", "SCHEME_CURRENT", "SUB_SLEEP", "STANDBYIDLE"]
_HEX = re.compile(r"0x[0-9a-fA-F]{8}")


def balance_verdict(status: int, body: object) -> tuple[CheckStatus, str, str]:
    """What the answer of the balance request means: ``(status, detail, hint)``."""
    if status in (401, 403):
        return (
            CheckStatus.FAIL,
            f"DeepSeek rejected the API key (HTTP {status})",
            "store a valid key: `twin secrets set deepseek_api_key`",
        )
    if status == 402:
        return (
            CheckStatus.FAIL,
            "the DeepSeek account has no balance (HTTP 402)",
            "top up the account on the DeepSeek platform",
        )
    if status == 200 and isinstance(body, dict):
        if body.get("is_available") is False:
            return (
                CheckStatus.FAIL,
                "the DeepSeek account reports that it cannot be used (no balance)",
                "top up the account on the DeepSeek platform",
            )
        infos = body.get("balance_infos")
        if isinstance(infos, list) and infos and isinstance(infos[0], dict):
            first = infos[0]
            total = f"{first.get('total_balance', '?')} {first.get('currency', '')}".strip()
            return CheckStatus.OK, f"DeepSeek answers; balance {total}", ""
        return CheckStatus.OK, "DeepSeek answers; the account is available", ""
    if status < 500:
        return CheckStatus.OK, f"DeepSeek answers (HTTP {status}); balance not shown", ""
    return CheckStatus.WARN, f"DeepSeek answered HTTP {status}", "try again later"


@doctor_check
def check_deepseek_net(ctx: DoctorContext) -> CheckResult:
    """DeepSeek is reachable with the stored key, and the account has balance (R-OPS-009)."""
    if ctx.settings is None:
        return CheckResult("deepseek-net", CheckStatus.WARN, "skipped (configuration did not load)")
    try:
        key = (ctx.secrets or SecretStore.default()).get("deepseek_api_key")
    except Exception as exc:
        return CheckResult(
            "deepseek-net", CheckStatus.WARN, f"cannot read the key: {type(exc).__name__}"
        )
    base = ctx.settings.deepseek.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        with httpx.Client(transport=ctx.http_transport, timeout=10.0) as client:
            response = client.get(base + DEEPSEEK_BALANCE_PATH, headers=headers)
    except httpx.TimeoutException:
        return CheckResult(
            "deepseek-net", CheckStatus.WARN, f"{base} timed out", "check the connection"
        )
    except httpx.HTTPError as exc:
        return CheckResult(
            "deepseek-net",
            CheckStatus.WARN,
            f"{base} cannot be reached ({type(exc).__name__})",
            "check the internet connection, a proxy or a VPN",
        )
    if not key and response.status_code in (401, 403):
        return CheckResult(
            "deepseek-net",
            CheckStatus.WARN,
            "DeepSeek is reachable, but there is no API key to try",
            "run `twin secrets set deepseek_api_key`",
        )
    try:
        body: object = response.json()
    except ValueError:
        body = None
    status, detail, hint = balance_verdict(response.status_code, body)
    return CheckResult("deepseek-net", status, detail, hint)


@doctor_check
def check_gpu(ctx: DoctorContext) -> CheckResult:
    """An NVIDIA card and its driver, for the local style model (R-SRV-002)."""
    wanted = (
        ctx.settings is not None
        and ctx.settings.backend.active in ("style", "hybrid")
        and ctx.settings.style_model.mode == "llamacpp_completion"
    )
    result = ctx.run(NVIDIA_QUERY)
    if result is None or result.returncode != 0:
        detail = "nvidia-smi was not found or failed: no NVIDIA GPU is usable"
        if wanted:
            return CheckResult(
                "gpu",
                CheckStatus.WARN,
                detail,
                "the local style model needs an NVIDIA card (or style_model.mode vllm_completion)",
            )
        return CheckResult(
            "gpu", CheckStatus.OK, detail + " (only the local style model needs one)"
        )
    first = decode_output(result.stdout).strip().splitlines()[:1]
    return CheckResult("gpu", CheckStatus.OK, first[0].strip() if first else "NVIDIA GPU found")


@doctor_check
def check_scheduled_task(ctx: DoctorContext) -> CheckResult:
    """The scheduled task is registered, interactive, unlimited and does not run ``uv sync``."""
    if ctx.platform != "win32":
        return CheckResult("scheduled-task", CheckStatus.OK, "not applicable (Windows only)")
    runner = ctx.runner or SubprocessRunner()
    try:
        info = TaskScheduler(runner).info()
    except TaskSchedulerError as exc:
        return CheckResult("scheduled-task", CheckStatus.WARN, str(exc))
    hint = "run `twin service install`"
    if not info.registered:
        return CheckResult(
            "scheduled-task", CheckStatus.WARN, "the scheduled task is not registered", hint
        )
    problems: list[str] = []
    if not info.interactive:
        problems.append(f"logon type is {info.logon_type}, it must be InteractiveToken")
    if info.runs_uv_sync:
        problems.append("the action runs `uv sync` (it must not change the environment)")
    if info.time_limit != "PT0S":
        problems.append(f"it has a time limit ({info.time_limit})")
    if info.multiple_instances != "IgnoreNew":
        problems.append("a second instance may start")
    if info.on_battery_allowed is False:
        problems.append("it does not run on battery")
    if info.enabled is False:
        problems.append("the task is disabled")
    if problems:
        return CheckResult(
            "scheduled-task",
            CheckStatus.FAIL if not info.interactive or info.runs_uv_sync else CheckStatus.WARN,
            "; ".join(problems),
            "register it again: `twin service uninstall` then `twin service install`",
        )
    return CheckResult("scheduled-task", CheckStatus.OK, "registered, InteractiveToken, no limit")


def sleep_timeouts(output: str) -> tuple[int, int] | None:
    """``(AC, DC)`` idle sleep timeout in seconds from ``powercfg /query``; 0 means never.

    The labels of the output are in the language of Windows, the two hexadecimal numbers at its
    end (the current AC and DC setting) are not.
    """
    found = _HEX.findall(output)
    if len(found) < 2:
        return None
    return int(found[-2], 16), int(found[-1], 16)


@doctor_check
def check_power_plan(ctx: DoctorContext) -> CheckResult:
    """The power plan must not put the computer to sleep while the bot runs (R-OPS-002)."""
    if ctx.platform != "win32":
        return CheckResult("power-plan", CheckStatus.OK, "not applicable (Windows only)")
    result = ctx.run(POWERCFG_QUERY)
    timeouts = sleep_timeouts(decode_output(result.stdout)) if result else None
    if timeouts is None:
        return CheckResult(
            "power-plan", CheckStatus.WARN, "the sleep setting cannot be read (powercfg)"
        )
    ac, dc = timeouts
    hint = (
        "Settings > System > Power: set 'sleep' to Never (the bot also asks Windows to stay awake)"
    )
    if ac:
        return CheckResult(
            "power-plan",
            CheckStatus.WARN,
            f"on mains power the computer sleeps after {ac // 60} min idle",
            hint,
        )
    if dc:
        return CheckResult(
            "power-plan",
            CheckStatus.WARN,
            f"on battery the computer sleeps after {dc // 60} min idle (mains: never)",
            hint,
        )
    return CheckResult("power-plan", CheckStatus.OK, "sleep is set to never")


def run_checks(ctx: DoctorContext) -> list[CheckResult]:
    """Run every registered check; a crashing check is reported as FAIL."""
    results: list[CheckResult] = []
    for check in _CHECKS:
        try:
            results.append(check(ctx))
        except Exception as exc:
            results.append(
                CheckResult(
                    check.__name__.removeprefix("check_"),
                    CheckStatus.FAIL,
                    f"check crashed: {type(exc).__name__}: {exc}",
                )
            )
    return results


def exit_code(results: list[CheckResult]) -> int:
    return 1 if any(result.status is CheckStatus.FAIL for result in results) else 0
