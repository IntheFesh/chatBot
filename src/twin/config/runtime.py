"""Runtime-mutable settings (R-CFG-003).

Settings that the user changes while the bot runs (time zone, thinking mode,
backend, proactive range, pause flag, ...) live in the ``settings`` table, not
in the YAML file.  The YAML file only supplies the *initial* value, copied into
the database the first time the application starts (:meth:`RuntimeSettings.initialize`).

Each setting is declared once as a typed :class:`SettingSpec`; reads and writes
are validated through a pydantic ``TypeAdapter`` and every change keeps a
history record (who, when, old value, new value; values sealed on disk).  A write
bumps ``settings.state_version`` in the same transaction so a running
application notices it (R-ARCH-006).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AfterValidator, AwareDatetime, NonNegativeInt, TypeAdapter, ValidationError

from twin.clock import Clock
from twin.config.settings import Settings
from twin.storage.db import Database
from twin.storage.settings_store import get_history, get_setting, put_setting
from twin.storage.state import bump_state_version


class SettingValueError(ValueError):
    """A runtime setting value failed validation."""


@dataclass(frozen=True)
class SettingSpec[T]:
    """A declared runtime setting."""

    key: str
    adapter: TypeAdapter[T]
    initial: Callable[[Settings], T]
    description: str


@dataclass(frozen=True)
class SettingChange:
    at: str
    by: str
    old: Any
    new: Any
    created: bool


_REGISTRY: dict[str, SettingSpec[Any]] = {}
_ABSENT: Any = object()


def register_setting[T](spec: SettingSpec[T]) -> SettingSpec[T]:
    """Register ``spec`` so it is initialised from config and listed by tooling."""
    existing = _REGISTRY.get(spec.key)
    if existing is not None and existing is not spec:
        raise ValueError(f"runtime setting {spec.key!r} is already registered")
    _REGISTRY[spec.key] = spec
    return spec


def registered_settings() -> dict[str, SettingSpec[Any]]:
    return dict(_REGISTRY)


def _valid_timezone(name: str) -> str:
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
        raise ValueError(f"unknown IANA time zone {name!r}") from exc
    return name


TimezoneName = Annotated[str, AfterValidator(_valid_timezone)]
ThinkingMode = Literal["off", "on", "auto"]
BackendName = Literal["deepseek", "style", "hybrid"]

BOT_TIMEZONE = register_setting(
    SettingSpec(
        "time.bot_timezone",
        TypeAdapter(TimezoneName),
        lambda s: s.time.bot_timezone,
        "Time zone the bot lives in (/时区)",
    )
)
THINKING_CHAT = register_setting(
    SettingSpec(
        "thinking.chat",
        TypeAdapter(ThinkingMode),
        lambda s: s.thinking.chat,
        "Thinking mode for chat replies (/思考)",
    )
)
THINKING_PROACTIVE = register_setting(
    SettingSpec(
        "thinking.proactive_planner",
        TypeAdapter(ThinkingMode),
        lambda s: s.thinking.proactive_planner,
        "Thinking mode for the proactive-message planner",
    )
)
BACKEND_ACTIVE = register_setting(
    SettingSpec(
        "backend.active",
        TypeAdapter(BackendName),
        lambda s: s.backend.active,
        "Generation backend",
    )
)
BACKEND_FALLBACK = register_setting(
    SettingSpec(
        "backend.fallback",
        TypeAdapter(dict[str, Any] | None),
        lambda s: None,
        "Why the style model is not in use although backend.active asks for it (R-SRV-004)",
    )
)
PROACTIVE_DAILY_MIN = register_setting(
    SettingSpec(
        "proactive.daily_min",
        TypeAdapter(NonNegativeInt),
        lambda s: s.proactive.daily_min,
        "Minimum proactive messages per day (/主动)",
    )
)
PROACTIVE_DAILY_MAX = register_setting(
    SettingSpec(
        "proactive.daily_max",
        TypeAdapter(NonNegativeInt),
        lambda s: s.proactive.daily_max,
        "Maximum proactive messages per day (/主动)",
    )
)
PROACTIVE_ENABLED = register_setting(
    SettingSpec(
        "proactive.enabled",
        TypeAdapter(bool),
        lambda s: True,
        "Proactive messages on (/主动 开|关); off means a quota of zero (R-PRO-002)",
    )
)
TARGET_USERNAME = register_setting(
    SettingSpec(
        "target.username",
        TypeAdapter(str | None),
        lambda s: s.target.username,
        "WeChat id of the conversation the bot imitates (chosen at the first import)",
    )
)
PAUSED = register_setting(
    SettingSpec("paused", TypeAdapter(bool), lambda s: False, "Bot paused (/暂停)")
)
SHOW_THINKING = register_setting(
    SettingSpec(
        "show_thinking",
        TypeAdapter(bool),
        lambda s: False,
        "Show reasoning to the user (/显示思考)",
    )
)
ENGINE_PAUSED_UNTIL = register_setting(
    SettingSpec(
        "engine.paused_until",
        TypeAdapter(AwareDatetime | None),
        lambda s: None,
        "The bot does not reply before this UTC time (/暂停 <时长>, /恢复); empty: not paused",
    )
)


class RuntimeSettings:
    """Typed, validated, history-keeping access to the ``settings`` table."""

    def __init__(self, db: Database, settings: Settings, clock: Clock) -> None:
        self._db = db
        self._settings = settings
        self._clock = clock

    def initialize(self) -> list[str]:
        """Seed every registered setting that has no stored value; returns the keys seeded."""
        seeded: list[str] = []
        with self._db.transaction(bump_state=False) as session:
            for key, spec in _REGISTRY.items():
                if get_setting(session, key, _ABSENT) is _ABSENT:
                    put_setting(
                        session,
                        key,
                        spec.adapter.dump_python(spec.initial(self._settings), mode="json"),
                        clock=self._clock,
                        by="config",
                        record_history=False,
                    )
                    seeded.append(key)
        return seeded

    def get[T](self, spec: SettingSpec[T]) -> T:
        with self._db.session() as session:
            raw = get_setting(session, spec.key, _ABSENT)
        if raw is _ABSENT:
            return spec.initial(self._settings)
        try:
            return spec.adapter.validate_python(raw)
        except ValidationError as exc:
            raise SettingValueError(
                f"stored value of {spec.key} is invalid ({exc.errors()[0]['msg']}); "
                "reset it with the matching command"
            ) from exc

    def set[T](self, spec: SettingSpec[T], value: T, *, by: str = "user") -> bool:
        """Validate and store ``value``.  Returns ``True`` if it changed."""
        try:
            checked = spec.adapter.validate_python(value)
        except ValidationError as exc:
            raise SettingValueError(
                f"invalid value for {spec.key}: {exc.errors()[0]['msg']}"
            ) from exc
        encoded = spec.adapter.dump_python(checked, mode="json")
        with self._db.transaction(bump_state=False) as session:
            changed = put_setting(session, spec.key, encoded, clock=self._clock, by=by)
            if changed:  # lets a running application notice (R-ARCH-006.2)
                bump_state_version(session, self._clock)
            return changed

    def history(self, spec: SettingSpec[Any]) -> list[SettingChange]:
        with self._db.session() as session:
            records = get_history(session, spec.key)
        return [
            SettingChange(
                at=str(r["at"]),
                by=str(r["by"]),
                old=r.get("old"),
                new=r.get("new"),
                created=bool(r.get("created")),
            )
            for r in records
        ]

    def snapshot(self) -> dict[str, Any]:
        """All registered settings with their current values."""
        return {key: self.get(spec) for key, spec in _REGISTRY.items()}
