"""Structured JSON logging with content protection (R-OPS-007, CLAUDE.md rule 6).

Choice: standard :mod:`logging` plus a JSON formatter instead of ``structlog``.
The project needs one file handler with rotation, one console handler and a
single rule about message bodies; the standard library covers that without a
dependency, and third-party libraries already log through it.

The rule: **message content never appears in INFO-or-above logs.**  Callers use
:class:`StructLogger`, whose keyword fields named like content (``content``,
``text``, ``body``, ...) are dropped at INFO and above and passed through
:func:`twin.llm.redaction.redact_text` at DEBUG.  Exceptions are logged as type plus a
redacted, truncated message and bare stack locations, never with local values.

Files: ``<logs_dir>/twin.log`` for the application, ``twin-supervise.log`` for the supervisor
(``twin supervise``) and ``twin-cli.log`` for CLI invocations (separate files so two processes
never rotate the same file), each 10 MB x 10 rotated copies, one JSON object per line.
"""

from __future__ import annotations

import contextlib
import logging
import logging.handlers
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import orjson

from twin.llm.redaction import redact_text

CONTENT_FIELDS = frozenset(
    {
        "content",
        "text",
        "body",
        "prompt",
        "reply",
        "completion",
        "caption",
        "transcript",
        "quote",
        "reasoning",
        "utterance",
        "raw",
        "payload",
        "messages",
        "memory_text",
        "summary_text",
    }
)
_RESERVED = frozenset({"ts", "level", "logger", "event", "pid", "exc", "stack"})
FIELDS_ATTR = "twin_fields"
OMITTED_KEY = "omitted"
MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 10
_HANDLER_TAG = "_twin_handler"
_EXC_MESSAGE_LIMIT = 300


def sanitize_fields(level: int, fields: dict[str, Any]) -> dict[str, Any]:
    """Apply the content policy for a record of ``level``.

    INFO and above: content-named fields are removed (their names are listed
    under ``omitted``).  Below INFO: they are kept but redacted.
    """
    clean: dict[str, Any] = {}
    omitted: list[str] = []
    for key, value in fields.items():
        name = f"f_{key}" if key in _RESERVED else key
        if key.lower() in CONTENT_FIELDS:
            if level >= logging.INFO:
                omitted.append(key)
                continue
            clean[name] = redact_text(value if isinstance(value, str) else repr(value))
        else:
            clean[name] = value
    if omitted:
        clean[OMITTED_KEY] = sorted(omitted)
    return clean


class StructLogger:
    """Keyword-field logger; the only logging API used by the application."""

    def __init__(self, logger: logging.Logger, bound: dict[str, Any] | None = None) -> None:
        self._logger = logger
        self._bound = bound or {}

    @property
    def name(self) -> str:
        return self._logger.name

    def bind(self, **fields: Any) -> StructLogger:
        return StructLogger(self._logger, {**self._bound, **fields})

    def _log(self, level: int, event: str, fields: dict[str, Any], *, exc: bool = False) -> None:
        if not self._logger.isEnabledFor(level):
            return
        merged = sanitize_fields(level, {**self._bound, **fields})
        self._logger.log(
            level,
            event,
            extra={FIELDS_ATTR: merged},
            exc_info=exc,
            stacklevel=3,
        )

    def debug(self, event: str, /, **fields: Any) -> None:
        self._log(logging.DEBUG, event, fields)

    def info(self, event: str, /, **fields: Any) -> None:
        self._log(logging.INFO, event, fields)

    def warning(self, event: str, /, **fields: Any) -> None:
        self._log(logging.WARNING, event, fields)

    def error(self, event: str, /, **fields: Any) -> None:
        self._log(logging.ERROR, event, fields)

    def critical(self, event: str, /, **fields: Any) -> None:
        self._log(logging.CRITICAL, event, fields)

    def exception(self, event: str, /, **fields: Any) -> None:
        self._log(logging.ERROR, event, fields, exc=True)


def get_logger(name: str) -> StructLogger:
    """Return a logger under the ``twin`` namespace."""
    full = name if name == "twin" or name.startswith("twin.") else f"twin.{name}"
    return StructLogger(logging.getLogger(full))


def describe_exception(exc_info: Any) -> tuple[str, str]:
    """``(summary, stack)`` for an exception: no local values, message redacted."""
    exc_type, exc_value, tb = exc_info
    message = redact_text(str(exc_value))[:_EXC_MESSAGE_LIMIT]
    summary = f"{exc_type.__name__}: {message}"
    frames = traceback.extract_tb(tb)
    stack = " <- ".join(
        f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
        for frame in reversed(frames[-8:])
    )
    return summary, stack


class JsonFormatter(logging.Formatter):
    """One JSON object per line, UTC timestamps."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "pid": record.process,
        }
        extra = getattr(record, FIELDS_ATTR, None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"], payload["stack"] = describe_exception(record.exc_info)
        return orjson.dumps(payload, default=str).decode("utf-8")


class ConsoleFormatter(logging.Formatter):
    """Compact human-readable single line."""

    def format(self, record: logging.LogRecord) -> str:
        stamp = datetime.fromtimestamp(record.created, UTC).strftime("%H:%M:%S")
        name = record.name.removeprefix("twin.")
        line = f"{stamp} {record.levelname:<7} {name}: {record.getMessage()}"
        extra = getattr(record, FIELDS_ATTR, None)
        if isinstance(extra, dict) and extra:
            line += " " + " ".join(f"{k}={v}" for k, v in extra.items())
        if record.exc_info:
            summary, stack = describe_exception(record.exc_info)
            line += f"\n    {summary}\n    at {stack}"
        return line


Role = Literal["run", "cli", "supervise"]
_LOG_FILES = {"run": "twin.log", "cli": "twin-cli.log", "supervise": "twin-supervise.log"}


def configure_logging(
    logs_dir: Path | None,
    *,
    level: str | int = "INFO",
    role: Role = "run",
    console: bool = True,
    console_level: int = logging.INFO,
    max_bytes: int = MAX_BYTES,
    backup_count: int = BACKUP_COUNT,
) -> Path | None:
    """Install the file (JSON, rotating) and console handlers; returns the log file path.

    ``logs_dir=None`` installs the console handler only (used when the data directory is
    unusable, so that diagnostics such as ``twin doctor`` still work).  Safe to call
    repeatedly: handlers installed earlier by this function are replaced.
    """
    log_path: Path | None = None
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / _LOG_FILES[role]
    numeric = level if isinstance(level, int) else logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        raise ValueError(f"unknown log level {level!r}")

    for name in ("twin", ""):
        target = logging.getLogger(name)
        for handler in list(target.handlers):
            if getattr(handler, _HANDLER_TAG, False):
                target.removeHandler(handler)
                _close_quietly(handler)

    handlers: list[logging.Handler] = []
    if log_path is not None:
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8", delay=True
        )
        file_handler.setFormatter(JsonFormatter())
        handlers.append(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(ConsoleFormatter())
        handlers.append(console_handler)

    twin_logger = logging.getLogger("twin")
    twin_logger.setLevel(numeric)
    twin_logger.propagate = False
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    for handler in handlers:
        setattr(handler, _HANDLER_TAG, True)
        twin_logger.addHandler(handler)
        root.addHandler(handler)  # third-party warnings and errors land in the same sinks
    for noisy in ("openai", "httpx", "httpcore", "asyncio", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_path


def _close_quietly(handler: logging.Handler) -> None:
    """Flush and close; a stream that is already closed (e.g. a captured stderr) is fine."""
    with contextlib.suppress(ValueError, OSError):
        handler.flush()
    handler.close()


def shutdown_logging() -> None:
    """Flush and close the handlers installed by :func:`configure_logging`."""
    for name in ("twin", ""):
        target = logging.getLogger(name)
        for handler in list(target.handlers):
            if getattr(handler, _HANDLER_TAG, False):
                _close_quietly(handler)
                target.removeHandler(handler)
