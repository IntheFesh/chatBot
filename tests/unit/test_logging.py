"""JSON logging, rotation settings and content protection (R-OPS-007)."""

from __future__ import annotations

import json
import logging
import logging.handlers
from pathlib import Path

import pytest

from twin.ops.logging import (
    BACKUP_COUNT,
    MAX_BYTES,
    configure_logging,
    describe_exception,
    get_logger,
    sanitize_fields,
    shutdown_logging,
)

BODY = "这是一段绝对不能出现在日志里的聊天正文"
PHONE = "13800138000"


def read_lines(path: Path) -> list[dict[str, object]]:
    for handler in logging.getLogger("twin").handlers:
        handler.flush()
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_info_logs_never_contain_content(tmp_path: Path) -> None:
    path = configure_logging(tmp_path, level="DEBUG", console=False)
    log = get_logger("test.privacy")
    log.info("message_received", content=BODY, text=BODY, user="u1", length=len(BODY))
    log.warning("odd", body=BODY, prompt=BODY, reply=BODY)
    log.error("bad", transcript=BODY, payload={"x": BODY})
    raw = path.read_text(encoding="utf-8")
    assert BODY not in raw
    lines = read_lines(path)
    assert lines[0]["event"] == "message_received"
    assert lines[0]["omitted"] == ["content", "text"]
    assert lines[0]["user"] == "u1" and lines[0]["length"] == len(BODY)
    assert "content" not in lines[0] and "text" not in lines[0]


def test_debug_logs_keep_content_but_redacted(tmp_path: Path) -> None:
    path = configure_logging(tmp_path, level="DEBUG", console=False)
    get_logger("test.debug").debug("generation", prompt=f"打这个电话 {PHONE} 或发 a@example.org")
    entry = read_lines(path)[0]
    assert PHONE not in str(entry["prompt"]) and "a@example.org" not in str(entry["prompt"])
    assert "[手机号]" in str(entry["prompt"]) and "[邮箱]" in str(entry["prompt"])
    assert "omitted" not in entry


def test_non_string_content_fields_are_redacted_via_repr(tmp_path: Path) -> None:
    path = configure_logging(tmp_path, level="DEBUG", console=False)
    get_logger("test.debug").debug("x", messages=[{"role": "user", "content": PHONE}])
    assert PHONE not in path.read_text(encoding="utf-8")


def test_below_the_configured_level_nothing_is_written(tmp_path: Path) -> None:
    path = configure_logging(tmp_path, level="INFO", console=False)
    get_logger("test.level").debug("hidden", prompt=BODY)
    assert not path.exists() or path.read_text(encoding="utf-8") == ""


def test_exceptions_are_logged_as_type_redacted_message_and_stack(tmp_path: Path) -> None:
    path = configure_logging(tmp_path, level="INFO", console=False)
    log = get_logger("test.exc")

    def failing() -> None:
        local_secret = BODY  # noqa: F841 - must never be logged
        raise ValueError(f"cannot send to {PHONE}")

    try:
        failing()
    except ValueError:
        log.exception("send_failed", attempt=2)
    entry = read_lines(path)[0]
    assert entry["level"] == "ERROR"
    assert str(entry["exc"]).startswith("ValueError: cannot send to [手机号]")
    assert "failing" in str(entry["stack"])
    assert BODY not in path.read_text(encoding="utf-8") and PHONE not in path.read_text(encoding="utf-8")


def test_long_exception_messages_are_truncated() -> None:
    try:
        raise RuntimeError("x" * 5000)
    except RuntimeError:
        import sys

        summary, _stack = describe_exception(sys.exc_info())
    assert len(summary) < 400


def test_records_are_json_lines_with_utc_timestamp_and_bound_fields(tmp_path: Path) -> None:
    path = configure_logging(tmp_path, level="INFO", console=False)
    log = get_logger("component.x").bind(component="x")
    log.info("hello", n=1, ts="clash", level="clash", event="clash")
    entry = read_lines(path)[0]
    assert entry["logger"] == "twin.component.x"
    assert entry["component"] == "x" and entry["n"] == 1
    assert str(entry["ts"]).endswith("+00:00")
    assert entry["level"] == "INFO" and entry["event"] == "hello"
    assert entry["f_ts"] == "clash" and entry["f_level"] == "clash" and entry["f_event"] == "clash"


def test_sanitize_fields_directly() -> None:
    assert sanitize_fields(logging.INFO, {"content": "a", "n": 1}) == {"n": 1, "omitted": ["content"]}
    assert sanitize_fields(logging.DEBUG, {"Content": PHONE})["Content"] == "[手机号]"


def test_rotation_is_ten_megabytes_times_ten_files(tmp_path: Path) -> None:
    configure_logging(tmp_path, console=False)
    handlers = [
        h for h in logging.getLogger("twin").handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
    ]
    assert handlers and handlers[0].maxBytes == MAX_BYTES == 10 * 1024 * 1024
    assert handlers[0].backupCount == BACKUP_COUNT == 10


def test_file_rotation_actually_happens(tmp_path: Path) -> None:
    path = configure_logging(tmp_path, console=False, max_bytes=2000, backup_count=3)
    log = get_logger("rotation")
    for index in range(200):
        log.info("tick", index=index, padding="p" * 40)
    shutdown_logging()
    files = sorted(tmp_path.glob("twin.log*"))
    assert len(files) == 4  # active file + 3 backups, older ones dropped
    assert path.exists()


def test_cli_and_run_roles_use_separate_files(tmp_path: Path) -> None:
    assert configure_logging(tmp_path, role="run", console=False).name == "twin.log"
    assert configure_logging(tmp_path, role="cli", console=False).name == "twin-cli.log"


def test_reconfiguring_replaces_handlers_instead_of_duplicating(tmp_path: Path) -> None:
    for _ in range(3):
        path = configure_logging(tmp_path, console=False)
    get_logger("once").info("single")
    assert len([e for e in read_lines(path) if e["event"] == "single"]) == 1


def test_console_output_is_readable_and_filtered_by_console_level(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging(tmp_path, level="INFO", console=True, console_level=logging.WARNING)
    log = get_logger("console")
    log.info("quiet", n=1)
    log.warning("loud", n=2, content=BODY)
    err = capsys.readouterr().err
    assert "quiet" not in err
    assert "WARNING" in err and "loud" in err and "n=2" in err
    assert BODY not in err


def test_invalid_level_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        configure_logging(tmp_path, level="LOUD")


def test_get_logger_namespaces_names() -> None:
    assert get_logger("jobs").name == "twin.jobs"
    assert get_logger("twin.jobs").name == "twin.jobs"
    assert get_logger("twin").name == "twin"
