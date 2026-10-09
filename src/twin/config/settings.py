"""Typed application configuration (R-CFG-001, R-CFG-004).

The model tree mirrors the YAML block in ``docs/SPEC.md`` R-CFG-004 key for key.
Three copies must stay identical (R-CFG-005): this module's defaults,
``config/config.example.yaml`` and the SPEC block; a test compares them.

Source priority (highest first): explicit overrides (command line) >
environment variables (``TWIN_`` prefix, nested keys joined by ``__``) >
``config/config.yaml`` > defaults.  Unknown keys are errors at every level.
Secrets never appear here; they live in the credential store.
"""

from __future__ import annotations

from contextvars import ContextVar
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True, populate_by_name=True)


class ConsentConfig(_Section):
    confirmed_at: str | None = "2026-10-08"

    @field_validator("confirmed_at", mode="before")
    @classmethod
    def _stringify_date(cls, value: Any) -> Any:
        # YAML parses an unquoted 2026-10-08 as a date object
        if isinstance(value, date):
            return value.isoformat()
        return value


class PathsConfig(_Section):
    data_dir: str = "./data"
    export_dir: str | None = None


class TargetConfig(_Section):
    username: str | None = None


class TzRange(_Section):
    from_: date = Field(alias="from")
    to: date
    tz: str


class TimeConfig(_Section):
    bot_timezone: str = "America/Chicago"
    source_timezone: str = "America/Chicago"
    source_timezone_ranges: list[TzRange] = Field(default_factory=list)


class TimeoutConfig(_Section):
    non_thinking: int = 60
    thinking: int = 180


class DeepSeekConfig(_Section):
    base_url: str = "https://api.deepseek.com"
    chat_model: str = "deepseek-flash"
    offline_model: str = "deepseek-flash"
    vision_model: str = "deepseek-flash"
    timeout_s: TimeoutConfig = Field(default_factory=TimeoutConfig)
    max_concurrency: int = Field(default=4, ge=1)
    reasoning_effort: Literal["low", "high", "max"] = "high"


class ModelPrice(_Section):
    cache_hit: float
    cache_miss: float
    output: float


def _default_prices() -> dict[str, ModelPrice]:
    return {
        "deepseek-flash": ModelPrice(cache_hit=0.006, cache_miss=0.30, output=1.20),
        "deepseek-v4-pro": ModelPrice(cache_hit=0.044, cache_miss=1.32, output=3.96),
    }


class PricingConfig(_Section):
    offpeak_multiplier: float = Field(default=0.5, gt=0, le=1)
    extra_offpeak_dates: list[date] = Field(default_factory=list)
    extra_peak_dates: list[date] = Field(default_factory=list)


def _default_degrade_ratios() -> list[float]:
    return [1.0, 1.25, 1.5, 2.0]


class BudgetConfig(_Section):
    daily_usd: float = Field(default=1.00, ge=0)
    monthly_usd: float = Field(default=15.00, ge=0)
    alert_ratio: float = Field(default=0.8, gt=0, le=1)
    one_time_usd: float = Field(default=30.00, ge=0)
    # share of the budget at which degradation levels 1..4 begin (R-LLM-008)
    degrade_ratios: list[float] = Field(default_factory=_default_degrade_ratios)

    @field_validator("degrade_ratios")
    @classmethod
    def _four_increasing_ratios(cls, value: list[float]) -> list[float]:
        if len(value) != 4:
            raise ValueError("degrade_ratios needs exactly four values (levels 1 to 4)")
        if any(ratio <= 0 for ratio in value) or any(a >= b for a, b in pairwise(value)):
            raise ValueError("degrade_ratios must be positive and strictly increasing")
        return value


class ThinkingConfig(_Section):
    chat: Literal["off", "on", "auto"] = "off"
    proactive_planner: Literal["off", "on", "auto"] = "on"
    auto_rules: bool = True


class BackendConfig(_Section):
    active: Literal["deepseek", "style", "hybrid"] = "deepseek"


class JobsConfig(_Section):
    concurrency: int = Field(default=2, ge=1, le=32)


class StickerDownloadConfig(_Section):
    concurrency: int = Field(default=4, ge=1, le=16)
    per_second: float = Field(default=4, gt=0, le=50)
    retries: int = Field(default=3, ge=0, le=10)
    timeout_s: float = Field(default=30, gt=0)


class IngestConfig(_Section):
    batch_size: int = Field(default=2000, ge=1)
    caption_recent_days: int = Field(default=90, ge=1)
    caption_wait_timeout_s: float = Field(default=30, gt=0)
    sticker_download: StickerDownloadConfig = Field(default_factory=StickerDownloadConfig)


class EngineConfig(_Section):
    quiet_window_s: int = Field(default=15, ge=0)
    quiet_window_adaptive: bool = False
    quiet_window_max_s: int = Field(default=45, ge=0)
    max_wait_s: int = Field(default=90, ge=0)
    history_turns_min: int = Field(default=30, ge=1)
    history_turns_max: int = Field(default=40, ge=1)
    examples_k: int = Field(default=8, ge=0)
    max_bubbles: int = Field(default=8, ge=1)
    ai_phrases_file: str = "config/lists/ai_phrases.txt"
    commitment_patterns_file: str = "config/lists/commitment_patterns.txt"


class ProfileConfig(_Section):
    burst_gap_s: int = Field(default=120, ge=1)
    segment_gap_min: int = Field(default=60, ge=1)
    recent_days: int = Field(default=90, ge=1)
    recency_weight: float = Field(default=0.6, ge=0, le=1)
    rules_file: str = "config/lists/style_rules.yaml"
    emoji_codes_file: str = "config/lists/wechat_emoji_codes.txt"


class ActivityConfig(_Section):
    sleep_min_hours: float = Field(default=3.0, gt=0)
    edge_minutes: int = Field(default=30, ge=0)
    smoothing_sigma_slots: float = Field(default=2, ge=0)
    min_valid_days: int = Field(default=14, ge=1)
    sleep_rate_ratio: float = Field(default=0.25, gt=0, le=1)
    busy_rate_ratio: float = Field(default=0.5, gt=0, le=1)
    busy_latency_ratio: float = Field(default=3.0, gt=1)


class ProactiveConfig(_Section):
    daily_min: int = Field(default=1, ge=0)
    daily_max: int = Field(default=6, ge=0)
    min_spacing_min: int = Field(default=60, ge=0)
    max_chase: int = Field(default=1, ge=0)
    edge_of_sleep_weekly_max: int = Field(default=2, ge=0)
    tick_minutes: int = Field(default=5, ge=1)


class ChannelConfig(_Section):
    kind: Literal["ilink", "console"] = "ilink"
    proactive_window_safe_h: float = Field(default=22, gt=0)
    outbound_quota_safe: int = Field(default=8, ge=0)
    proactive_reserve: int = Field(default=2, ge=0)


class RetrievalConfig(_Section):
    model: str = "BAAI/bge-small-zh-v1.5"
    device: str = "auto"
    holdout_ratio: float = Field(default=0.10, gt=0, lt=1)


class MemoryConfig(_Section):
    daily_summary: bool = True
    fact_extraction: bool = True


class SmtpConfig(_Section):
    host: str | None = None
    port: int = Field(default=465, ge=1, le=65535)
    user: str | None = None
    to: str | None = None


class OpsConfig(_Section):
    backup_hour_local: int = Field(default=4, ge=0, le=23)
    backup_keep_daily: int = Field(default=14, ge=0)
    backup_keep_weekly: int = Field(default=8, ge=0)
    backup_mirror_dir: str | None = None
    smtp: SmtpConfig = Field(default_factory=SmtpConfig)


class TunnelConfig(_Section):
    local_port: int = Field(default=8082, ge=1, le=65535)
    remote_port: int = Field(default=8000, ge=1, le=65535)


class StyleModelConfig(_Section):
    mode: Literal["llamacpp_completion", "vllm_completion"] = "llamacpp_completion"
    endpoint: str = "http://127.0.0.1:8081"
    model_id: str | None = None
    tunnel: TunnelConfig = Field(default_factory=TunnelConfig)


class AutoDlConfig(_Section):
    host: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    user: str = "root"
    auth: Literal["password", "key"] = "password"
    key_path: str | None = None
    workdir: str = "/root/autodl-tmp/twin"


class TrainingConfig(_Section):
    hybrid_plan_ratio: float = Field(default=0.30, ge=0, le=1)
    dpo_min_pairs: int = Field(default=200, ge=1)
    retrain_new_ratio: float = Field(default=0.10, gt=0)


class EvalConfig(_Section):
    blind_n: int = Field(default=50, ge=1)
    memory_questions: int = Field(default=20, ge=1)


class EmergencyContactConfig(_Section):
    enabled: bool = False
    email: str | None = None


def _default_hotlines() -> dict[str, str]:
    return {
        "US": "988（美国心理危机热线，电话或短信）",
        "CN": "12356（全国心理援助热线）",
    }


def _default_timezone_country() -> dict[str, str]:
    return {
        "America/Chicago": "US",
        "America/New_York": "US",
        "America/Denver": "US",
        "America/Los_Angeles": "US",
        "Asia/Shanghai": "CN",
    }


class SafetyConfig(_Section):
    crisis_keywords_file: str = "config/lists/crisis_keywords.txt"
    hotlines: dict[str, str] = Field(default_factory=_default_hotlines)
    timezone_country: dict[str, str] = Field(default_factory=_default_timezone_country)
    emergency_contact: EmergencyContactConfig = Field(default_factory=EmergencyContactConfig)


# ------------------------------------------------------------------ sources

_yaml_path: ContextVar[Path | None] = ContextVar("twin_yaml_path", default=None)


class ConfigFileError(ValueError):
    """The YAML configuration file is unreadable or not a mapping."""


class _YamlSource(PydanticBaseSettingsSource):
    """Reads ``config.yaml`` (path supplied through :func:`yaml_config_path`)."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        self._data = self._read(_yaml_path.get())

    @staticmethod
    def _read(path: Path | None) -> dict[str, Any]:
        if path is None or not path.exists():
            return {}
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigFileError(f"cannot read configuration file {path}: {exc}") from exc
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise ConfigFileError(f"{path} must contain a YAML mapping at the top level")
        return loaded

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return dict(self._data)


class yaml_config_path:
    """Context manager selecting which YAML file :class:`Settings` reads."""

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._token: Any = None

    def __enter__(self) -> None:
        self._token = _yaml_path.set(self._path)

    def __exit__(self, *exc_info: object) -> None:
        _yaml_path.reset(self._token)


class Settings(BaseSettings):
    """The complete configuration tree (R-CFG-004)."""

    model_config = SettingsConfigDict(
        env_prefix="TWIN_",
        env_nested_delimiter="__",
        extra="forbid",
        validate_default=True,
        populate_by_name=True,
    )

    consent: ConsentConfig = Field(default_factory=ConsentConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    time: TimeConfig = Field(default_factory=TimeConfig)
    deepseek: DeepSeekConfig = Field(default_factory=DeepSeekConfig)
    pricing_usd_per_mtok: dict[str, ModelPrice] = Field(default_factory=_default_prices)
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    thinking: ThinkingConfig = Field(default_factory=ThinkingConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    jobs: JobsConfig = Field(default_factory=JobsConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    profile: ProfileConfig = Field(default_factory=ProfileConfig)
    activity: ActivityConfig = Field(default_factory=ActivityConfig)
    proactive: ProactiveConfig = Field(default_factory=ProactiveConfig)
    channel: ChannelConfig = Field(default_factory=ChannelConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    ops: OpsConfig = Field(default_factory=OpsConfig)
    style_model: StyleModelConfig = Field(default_factory=StyleModelConfig)
    autodl: AutoDlConfig = Field(default_factory=AutoDlConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, env_settings, _YamlSource(settings_cls))

    def to_plain(self) -> dict[str, Any]:
        """JSON-compatible dict using the YAML key names (``from`` not ``from_``)."""
        return self.model_dump(mode="json", by_alias=True)
