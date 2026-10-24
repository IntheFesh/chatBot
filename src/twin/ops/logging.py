"""Structured JSON logging with content protection (R-OPS-007, CLAUDE.md rule 6).

Choice: standard :mod:`logging` plus a JSON formatter instead of ``structlog``.
The project needs one file handler with rotation, one console handler and a
single rule about message bodies; the standard library covers that without a
dependency, and third-party libraries already log through it.

The rule: **message content never appears in INFO-or-above logs.**  Callers use
:class:`StructLogger`, whose keyword fields named like content (``content``,
``text``, ``body``, ...) are dropped at INFO and above and passed through
:func:`twin.llm.redaction.redact_text` at DEBUG.  Exceptions are logged as their **type**, a few
structured facts (an HTTP status, an error code, a retry flag - :func:`exception_facts`) and bare
stack locations (``file:line:function``), never with local values and **never with the message**:
the message of an exception can quote a sentence of the conversation, and redaction masks
identifiers, not sentences (round 17 item 44, D-101, D-714).  The redacted message exists in two
places only: a separate DEBUG record ``<event>_detail`` (written only when the log level is
DEBUG) and, for failed jobs, the sealed column ``jobs.last_error_detail``.

Files: ``<logs_dir>/twin.log`` for the application, ``twin-supervise.log`` for the supervisor
(``twin supervise``) and ``twin-cli.log`` for CLI invocations (separate files so two processes
never rotate the same file), each 10 MB x 10 rotated copies, one JSON object per line.
"""

from __future__ import annotations

import contextlib
import enum
import logging
import logging.handlers
import re
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
EXC_DETAIL_LIMIT = 2000  # characters of a redacted message kept for the DEBUG record
MAX_STACK_FRAMES = 8
MAX_CAUSES = 3

# What an exception may tell a log without its message (see :func:`exception_facts`): attributes
# that our own code fills with numbers, flags and fixed words, or that the HTTP libraries fill with
# the status of the answer - never a text that was written for a person.
_INT_FACTS = ("status", "status_code", "errcode", "ret", "attempts", "level", "errno", "winerror")
_FLAG_FACTS = ("retryable", "timeout", "request_sent")
_WORD_FACTS = ("kind", "purpose", "endpoint", "code", "sqlite_errorname")
_WORD = re.compile(r"^[A-Za-z][A-Za-z0-9_./:\-]{0,63}$")  # one token: no space, no sentence


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
        caught = sys.exc_info()[1] if exc else None
        if caught is not None and self._logger.isEnabledFor(logging.DEBUG):
            # the message of the exception, redacted, in a record of its own at DEBUG
            detail = sanitize_fields(
                logging.DEBUG, {**self._bound, "exc_detail": exception_detail(caught)}
            )
            self._logger.log(
                logging.DEBUG, f"{event}_detail", extra={FIELDS_ATTR: detail}, stacklevel=3
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


def _fact_value(value: object) -> str | None:
    if isinstance(
        value, enum.Enum
    ):  # before int: an IntEnum such as ExitCode is named, not numbered
        return value.name.lower() if _WORD.fullmatch(value.name) else None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and _WORD.fullmatch(value):
        return value
    return None


def _attribute(source: object, name: str) -> object:
    try:
        return getattr(source, name, None)
    except Exception:  # a property that raises (httpx.RequestError.request) is no fact
        return None


def exception_facts(exc: BaseException) -> dict[str, str]:
    """The facts of an exception that are not its message, as ``{name: text}``.

    Only attributes on a short list are read, and only values that cannot be a sentence: integers
    (``status``, ``status_code``, ``errcode``, ``ret``, ``attempts``, ``errno``, ...), flags
    (``retryable``, ``timeout``, ``request_sent``) and single words (``kind``, ``purpose``,
    ``endpoint``, a ``code`` or an enum member such as :class:`~twin.ops.process_model.ExitCode`).
    The status of an HTTP answer is found on the exception (``openai``, our own errors) or on its
    ``response`` (``httpx``).  Anything else - ``reason``, ``args``, the text - is not read.
    """
    facts: dict[str, str] = {}
    sources: list[object] = [exc]
    response = _attribute(exc, "response")
    if response is not None:
        sources.append(response)
    for source in sources:
        for name in (*_INT_FACTS, *_FLAG_FACTS, *_WORD_FACTS):
            if name in facts:
                continue
            value = _attribute(source, name)
            if name in _INT_FACTS and (isinstance(value, bool) or not isinstance(value, int)):
                continue
            if name in _FLAG_FACTS and not isinstance(value, bool):
                continue
            text = _fact_value(value)
            if text is not None:
                facts[name] = text
    return facts


def _label(exc: BaseException) -> str:
    facts = " ".join(f"{name}={value}" for name, value in exception_facts(exc).items())
    return f"{type(exc).__name__} {facts}" if facts else type(exc).__name__


def exception_summary(exc: BaseException) -> str:
    """The type of ``exc`` with its facts, and the same for what caused it - no message.

    ``ApiError status=503 retryable=true <- caused by APIStatusError status_code=503``.
    """
    parts = [_label(exc)]
    seen = {id(exc)}
    cause = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
    while cause is not None and id(cause) not in seen and len(parts) <= MAX_CAUSES:
        seen.add(id(cause))
        parts.append(f"caused by {_label(cause)}")
        cause = cause.__cause__ or (None if cause.__suppress_context__ else cause.__context__)
    return " <- ".join(parts)


def exception_stack(tb: Any) -> str:
    """``file:line:function`` of the innermost frames, innermost first; no values."""
    frames = traceback.extract_tb(tb)
    return " <- ".join(
        f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
        for frame in reversed(frames[-MAX_STACK_FRAMES:])
    )


def exception_detail(exc: BaseException, limit: int = EXC_DETAIL_LIMIT) -> str:
    """The message of ``exc`` redacted and shortened: for the DEBUG log and sealed columns only."""
    return redact_text(str(exc))[:limit]


def describe_exception(exc_info: Any) -> tuple[str, str]:
    """``(summary, stack)`` for an exception: its type and facts, bare locations, no message."""
    _exc_type, exc_value, tb = exc_info
    return exception_summary(exc_value), exception_stack(tb)


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
