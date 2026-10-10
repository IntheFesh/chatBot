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
    health_check_s: float = Field(
        default=30, gt=0
    )  # how often the style model is looked at (R-SRV-004)
    fallback_violations: int = Field(
        default=3, ge=1
    )  # hard violations in a row that cause a fallback
    recover_after_min: float = Field(default=10, ge=0)  # healthy this long before switching back


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
    """Proactive messages (round 10; R-PRO-001 to R-PRO-008)."""

    daily_min: int = Field(default=1, ge=0)
    daily_max: int = Field(default=6, ge=0)
    min_spacing_min: int = Field(default=60, ge=0)
    max_chase: int = Field(default=1, ge=0)
    edge_of_sleep_weekly_max: int = Field(default=2, ge=0)
    tick_minutes: int = Field(default=5, ge=1)
    user_active_min: int = Field(default=10, ge=0)  # no candidate this soon after an exchange
    unanswered_after_min: int = Field(default=30, ge=1)  # silence that makes a message unanswered
    meal_window_min: int = Field(
        default=30, ge=0
    )  # a meal message goes out +- this around her meal
    bedtime_lead_min: list[int] = Field(default_factory=lambda: [15, 60])  # before she falls asleep

    @field_validator("bedtime_lead_min")
    @classmethod
    def _two_increasing_leads(cls, value: list[int]) -> list[int]:
        if len(value) != 2 or value[0] < 0 or value[0] >= value[1]:
            raise ValueError(
                "bedtime_lead_min needs [shortest, longest] minutes, shortest < longest"
            )
        return value


class ScheduleConfig(_Section):
    """Time service, day plan and wake-up handling (round 08; R-SCH-001 to R-SCH-005)."""

    plan_minute: int = Field(default=5, ge=0, le=59)  # the day plan is made at local 00:MM
    tick_s: float = Field(default=30, gt=0)  # how often the scheduler looks for due work
    greeting_min_gap_h: float = Field(default=18, ge=0)  # hours between two wake-up greetings
    greeting_window_min: list[int] = Field(default_factory=lambda: [5, 40])  # after waking
    min_awake_h: float = Field(default=6, gt=0)  # awake time between a wake-up and the next bed
    meal_jitter_min: float = Field(default=15, ge=0)  # spread of a meal around her usual time
    summary_lead_min: int = Field(default=60, ge=0)  # the daily summary is queued this long
    summary_latest_after_wake_h: float = Field(default=2, ge=0)  # ... but not later than this
    power_tick_s: float = Field(default=5, gt=0)  # interval of the wake-from-sleep detector
    power_jump_ticks: int = Field(default=2, ge=1)  # a clock jump of more than this many ticks

    @field_validator("greeting_window_min")
    @classmethod
    def _two_increasing_minutes(cls, value: list[int]) -> list[int]:
        if len(value) != 2 or value[0] < 0 or value[0] >= value[1]:
            raise ValueError(
                "greeting_window_min needs [earliest, latest] minutes, earliest < latest"
            )
        return value


class ChannelConfig(_Section):
    kind: Literal["ilink", "console"] = "ilink"
    proactive_window_safe_h: float = Field(default=22, gt=0)
    outbound_quota_safe: int = Field(default=8, ge=0)
    proactive_reserve: int = Field(default=2, ge=0)


class RetrievalConfig(_Section):
    model: str = "BAAI/bge-small-zh-v1.5"
    device: str = "auto"
    holdout_ratio: float = Field(default=0.10, gt=0, lt=1)
    context_turns: int = Field(default=6, ge=1, le=20)  # merged turns before a reply (R-RET-001)
    candidates: int = Field(default=50, ge=1)  # nearest windows taken before re-ranking (R-RET-005)
    mmr_lambda: float = Field(default=0.7, ge=0, le=1)  # 1 = pure relevance, 0 = pure diversity
    slot_weight: float = Field(default=0.06, ge=0)  # bonus for the same time of day
    slot_sigma_slots: float = Field(default=8, gt=0)  # width of that bonus in 15-minute slots
    recency_weight: float = Field(default=0.04, ge=0)  # bonus for recent windows
    recency_half_life_days: float = Field(default=180, gt=0)
    dedup_similarity: float = Field(default=0.9, gt=0, le=1)  # reply texts alike above this
    event_only_factor: float = Field(default=0.6, ge=0, le=1)  # score factor of event-only replies
    batch_size: int = Field(default=64, ge=1, le=4096)  # texts per encoding call


class PersonaConfig(_Section):
    sample_segments: int = Field(default=60, ge=1)  # conversation segments sampled (R-PERS-001)
    batch_segments: int = Field(default=10, ge=1)  # segments per map call
    segment_max_messages: int = Field(default=40, ge=4)  # longest stretch shown per segment
    regen_ratio: float = Field(default=0.10, gt=0, le=1)  # new messages that re-queue the text
    full_max_tokens: int = Field(default=1500, ge=100)  # render_full budget (R-PERS-004)
    compact_max_tokens: int = Field(default=400, ge=50)  # render_compact budget (R-PERS-004)


class StickersConfig(_Section):
    tags_file: str = "config/lists/sticker_tags.txt"  # the closed emotion vocabulary (R-STK-003)
    neighbors_file: str = "config/lists/sticker_tag_neighbors.yaml"  # nearby tags (R-STK-004)
    tag_job_size: int = Field(default=20, ge=1)  # stickers per tagging job
    context_min_uses: int = Field(default=3, ge=1)  # her uses before the cutoff that allow it
    context_max_samples: int = Field(default=5, ge=1)  # uses whose surroundings are shown
    no_repeat_window: int = Field(default=10, ge=0)  # bubbles in which a sticker is not repeated
    repeat_rate_threshold: float = Field(
        default=0.30, ge=0, le=1
    )  # her repeat rate that relaxes it
    recency_half_life_days: float = Field(default=90, gt=0)  # decay of the recency factor
    rate_window: int = Field(default=200, ge=1)  # bubbles the sticker share is measured over
    rate_tolerance: float = Field(default=0.20, ge=0, lt=1)  # allowed excess over her share
    describe_timeout_s: float = Field(default=15, gt=0)  # describing a sticker the user sent


class MemoryWeights(_Section):
    """How the five signals of R-MEM-008 are weighed when memory items are scored."""

    similarity: float = Field(default=0.45, ge=0)  # closeness to the current topic
    importance: float = Field(default=0.15, ge=0)  # the importance 1-5 given when extracted
    recency: float = Field(default=0.10, ge=0)  # newer is better (half-life below)
    source: float = Field(default=0.10, ge=0)  # real record > user said > the bot invented
    date: float = Field(default=0.20, ge=0)  # an anniversary / due date near today


class MemoryConfig(_Section):
    daily_summary: bool = True
    fact_extraction: bool = True
    block_tokens: int = Field(default=800, ge=50)  # memory block budget before R-LLM-008 cuts it
    recall_facts: int = Field(default=12, ge=1)  # facts taken by meaning and by keyword each
    recent_summary_days: int = Field(default=3, ge=0)  # the last days whose summary is always in
    recall_summaries: int = Field(default=2, ge=0)  # older days taken because they fit the topic
    followup_lookahead_h: float = Field(
        default=36, gt=0
    )  # how far ahead a follow-up counts as near
    recall_min_similarity: float = Field(default=0.35, ge=0, le=1)  # weaker vector hits are noise
    conflict_candidates: int = Field(default=6, ge=1)  # older facts shown to the conflict judge
    conflict_min_similarity: float = Field(default=0.45, ge=0, le=1)  # below: not a candidate
    recency_half_life_days: float = Field(default=90, gt=0)  # decay of the recency signal
    quiet_minutes: int = Field(default=30, ge=1)  # silence that ends a bot conversation (R-MEM-007)
    replay_job_days: int = Field(default=7, ge=1)  # local days one replay job works through
    replay_chunk_lines: int = Field(default=400, ge=20)  # dialogue lines shown to one extraction
    replay_auto_approve_ratio: float = Field(default=0.10, ge=0, le=1)  # of budget.one_time_usd
    weights: MemoryWeights = Field(default_factory=MemoryWeights)


class SmtpConfig(_Section):
    host: str | None = None
    port: int = Field(default=465, ge=1, le=65535)
    user: str | None = None
    to: str | None = None
    # auto: implicit TLS on port 465, STARTTLS on every other port (round 12, R-OPS-004)
    security: Literal["auto", "ssl", "starttls"] = "auto"


class HealthConfig(_Section):
    """Thresholds of the health check that runs every minute (round 12, R-OPS-003)."""

    interval_s: float = Field(default=60, gt=0)  # how often the monitor looks
    poll_stale_min: float = Field(default=5, gt=0)  # no successful long poll for this long: bad
    disk_min_gb: float = Field(default=5, gt=0)  # free disk space below this: alert
    queue_max: int = Field(default=500, ge=1)  # waiting jobs above this: alert
    queue_oldest_h: float = Field(default=24, gt=0)  # a waiting job older than this: alert
    backup_stale_h: float = Field(default=36, gt=0)  # newest backup older than this: alert
    keep_days: int = Field(default=30, ge=1)  # how long health snapshots are kept
    llm_window_min: float = Field(default=15, gt=0)  # the span DeepSeek errors are counted over
    llm_error_rate: float = Field(default=0.5, gt=0, le=1)  # failed share of calls that is bad
    llm_min_calls: int = Field(default=10, ge=1)  # fewer calls than this say nothing
    remind_h: float = Field(default=6, gt=0)  # a problem that goes on is announced again after this


class SuperviseConfig(_Section):
    """``twin supervise``: restarting ``twin run`` (round 12, R-OPS-001)."""

    backoff_start_s: float = Field(default=5, gt=0)  # first wait before a restart
    backoff_max_s: float = Field(default=300, gt=0)  # the wait doubles up to this
    stable_after_min: float = Field(default=30, gt=0)  # running this long resets the wait
    stop_grace_s: float = Field(
        default=40, gt=0
    )  # time `twin run` gets to stop before it is killed


class OpsConfig(_Section):
    backup_hour_local: int = Field(default=4, ge=0, le=23)
    backup_keep_daily: int = Field(default=14, ge=0)
    backup_keep_weekly: int = Field(default=8, ge=0)
    backup_mirror_dir: str | None = None
    backup_postpone_max_h: float = Field(default=2, ge=0)  # waiting for a quiet moment, at most
    smtp: SmtpConfig = Field(default_factory=SmtpConfig)
    alert_cooldown_min: float = Field(default=60, ge=0)  # at most one notice per kind in this time
    health: HealthConfig = Field(default_factory=HealthConfig)
    supervise: SuperviseConfig = Field(default_factory=SuperviseConfig)


class TunnelConfig(_Section):
    local_port: int = Field(default=8082, ge=1, le=65535)
    remote_port: int = Field(default=8000, ge=1, le=65535)
    backoff_start_s: float = Field(default=2.0, gt=0)  # first wait before the tunnel reconnects
    backoff_max_s: float = Field(default=60.0, gt=0)  # the wait doubles up to this
    remind_remote: bool = True  # tell the user once a day that the rented instance is running


class ServeConfig(_Section):
    """The local ``llama-server`` (round 14, R-SRV-002); it listens on this computer only."""

    binary: str | None = None  # llama-server executable; null: the newest below tools/llama.cpp
    context: int = Field(default=4096, ge=512)  # -c
    gpu_layers: int = Field(default=999, ge=0)  # -ngl (999: every layer on the GPU)
    parallel: int = Field(default=1, ge=1)  # --parallel
    start_timeout_s: float = Field(default=180.0, gt=0)  # loading the model, at most
    backoff_start_s: float = Field(default=2.0, gt=0)  # first wait before a crashed server restarts
    backoff_max_s: float = Field(default=60.0, gt=0)  # the wait doubles up to this
    stable_after_s: float = Field(default=120.0, gt=0)  # running this long resets the wait
    warm_standby: bool = True  # keep a gate-passed model loaded for budget level 4
    eval_port: int = Field(default=8083, ge=1, le=65535)  # a model under evaluation


class StyleModelConfig(_Section):
    mode: Literal["llamacpp_completion", "vllm_completion"] = "llamacpp_completion"
    endpoint: str = "http://127.0.0.1:8081"
    model_id: str | None = None
    tunnel: TunnelConfig = Field(default_factory=TunnelConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)
    memory_tokens: int = Field(default=300, ge=0)  # memory block in the system segment (R-TRN-002)
    n_predict: int = Field(default=200, ge=1)  # the most tokens one reply may have
    temperature: float = Field(default=0.7, ge=0)
    top_p: float = Field(default=0.9, gt=0, le=1)


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
    # the weekly consistency audit (round 15, R-EVAL-004): how many days it looks back, how many
    # characters of the bot's replies and how many facts it hands to DeepSeek
    consistency_days: int = Field(default=7, ge=1, le=60)
    consistency_reply_chars: int = Field(default=12000, ge=500)
    consistency_facts: int = Field(default=40, ge=0)


class CommandsConfig(_Section):
    """The in-chat commands (round 11): confirmation window, pause limit, import report."""

    confirm_window_min: int = Field(default=60, ge=1)
    pause_max_h: int = Field(default=168, ge=1)
    morning_hour: int = Field(default=8, ge=0, le=23)
    import_poll_s: float = Field(default=5.0, gt=0)


class LearningConfig(_Section):
    """Learning from the conversation with the bot (round 11): rules of ``[不要这样]``."""

    rules_max: int = Field(default=30, ge=1)
    rules_interval_days: int = Field(default=7, ge=1)
    rule_max_chars: int = Field(default=40, ge=8)
    detect_corrections: bool = True
    check_interval_s: float = Field(default=3600.0, gt=0)


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
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    channel: ChannelConfig = Field(default_factory=ChannelConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    stickers: StickersConfig = Field(default_factory=StickersConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    ops: OpsConfig = Field(default_factory=OpsConfig)
    style_model: StyleModelConfig = Field(default_factory=StyleModelConfig)
    autodl: AutoDlConfig = Field(default_factory=AutoDlConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    commands: CommandsConfig = Field(default_factory=CommandsConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
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
