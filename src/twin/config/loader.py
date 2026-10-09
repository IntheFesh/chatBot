"""Loading configuration, resolving paths and the consent gate.

* :func:`load_settings` applies the priority CLI > environment > YAML > defaults;
* :func:`parse_overrides` turns repeated ``--set a.b.c=value`` options into the
  nested mapping handed to :class:`~twin.config.settings.Settings`;
* :func:`resolve_paths` anchors relative paths to the project root;
* :func:`ensure_consent` refuses to start without a valid
  ``consent.confirmed_at`` (R-SCOPE-003).
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from twin.config.settings import Settings, yaml_config_path

ENV_HOME = "TWIN_HOME"
ENV_CONFIG = "TWIN_CONFIG"


class ConsentError(RuntimeError):
    """The recorded consent is missing or not a valid date (R-SCOPE-003)."""


class ConfigError(ValueError):
    """The configuration could not be loaded; the message is user-facing."""


def project_root(cwd: Path | None = None, env: Mapping[str, str] | None = None) -> Path:
    """Directory that relative configuration paths are resolved against.

    ``TWIN_HOME`` if set; otherwise the nearest ancestor of the working directory
    that contains both ``pyproject.toml`` and ``config/``; otherwise the working
    directory itself.
    """
    environ = os.environ if env is None else env
    home = environ.get(ENV_HOME)
    if home:
        return Path(home).resolve()
    start = (cwd or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "config").is_dir():
            return candidate
    return start


def default_config_path(root: Path | None = None, env: Mapping[str, str] | None = None) -> Path:
    environ = os.environ if env is None else env
    explicit = environ.get(ENV_CONFIG)
    if explicit:
        return Path(explicit)
    return (root or project_root(env=environ)) / "config" / "config.yaml"


def parse_overrides(items: Iterable[str]) -> dict[str, Any]:
    """Parse ``["time.bot_timezone=Asia/Shanghai", "budget.daily_usd=2"]``.

    Values are interpreted as YAML scalars (so ``2`` is an integer, ``null`` is
    ``None``, ``[a, b]`` is a list).
    """
    result: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"invalid --set value {item!r}; expected key.path=value")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ConfigError(f"cannot parse value in --set {item!r}: {exc}") from exc
        node = result
        parts = [part.strip() for part in key.split(".")]
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ConfigError(f"--set {item!r} conflicts with another override")
            node = child
        node[parts[-1]] = value
    return result


def load_settings(
    config_path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Settings:
    """Load settings: ``overrides`` (CLI) > environment > YAML file > defaults."""
    path = config_path if config_path is not None else default_config_path()
    if config_path is not None and not path.exists():
        raise ConfigError(f"configuration file not found: {path}")
    try:
        with yaml_config_path(path):
            return Settings(**dict(overrides or {}))
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc


def _format_validation_error(exc: ValidationError) -> str:
    lines = ["invalid configuration:"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)


def ensure_consent(settings: Settings) -> date:
    """Return the consent date or raise :class:`ConsentError` (R-SCOPE-003)."""
    raw = settings.consent.confirmed_at
    if raw is None or not str(raw).strip():
        raise ConsentError(
            "consent.confirmed_at is missing. This bot imitates a real person and may only run "
            "with her knowledge and permission. Set consent.confirmed_at to the date she "
            "agreed (YYYY-MM-DD) in config/config.yaml."
        )
    try:
        return date.fromisoformat(str(raw).strip())
    except ValueError:
        raise ConsentError(
            f"consent.confirmed_at={raw!r} is not a valid date. Use the form YYYY-MM-DD "
            "(for example 2026-10-08) in config/config.yaml."
        ) from None


@dataclass(frozen=True)
class DataPaths:
    """Resolved filesystem locations (everything below ``data_dir`` is gitignored)."""

    root: Path
    data_dir: Path
    export_dir: Path | None

    @property
    def db_path(self) -> Path:
        return self.data_dir / "twin.db"

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def locks_dir(self) -> Path:
        return self.data_dir / "locks"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def backups_dir(self) -> Path:
        return self.data_dir / "backups"

    def ensure(self) -> None:
        """Create the data directories (owner-only on POSIX)."""
        for directory in (self.data_dir, self.logs_dir, self.locks_dir):
            directory.mkdir(parents=True, exist_ok=True)


def resolve_paths(settings: Settings, root: Path | None = None) -> DataPaths:
    base = root or project_root()

    def anchor(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (base / path).resolve()

    export = settings.paths.export_dir
    return DataPaths(
        root=base,
        data_dir=anchor(settings.paths.data_dir),
        export_dir=anchor(export) if export else None,
    )
