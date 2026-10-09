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
    coverage = PeakCalendar().coverage(today)
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
