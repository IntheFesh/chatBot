"""JSON logging, rotation settings and content protection (R-OPS-007)."""

from __future__ import annotations

import json
import logging
import logging.handlers
from pathlib import Path

import httpx
import openai
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.support.synthetic import mobile
from twin.channel.ilink.http import IlinkHttpError
from twin.llm.errors import ApiError, BudgetDeniedError, StyleModelError
from twin.ops.logging import (
    BACKUP_COUNT,
    EXC_DETAIL_LIMIT,
    MAX_BYTES,
    configure_logging,
    describe_exception,
    exception_detail,
    exception_summary,
    get_logger,
    sanitize_fields,
    shutdown_logging,
)
from twin.ops.process_model import CliError, ExitCode

BODY = "这是一段绝对不能出现在日志里的聊天正文"
PHONE = mobile()


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


def failing(message: str) -> None:
    local_secret = BODY  # noqa: F841 - must never be logged
    raise ValueError(message)


def test_exceptions_are_logged_as_type_and_stack_never_the_message(tmp_path: Path) -> None:
    """R17-44 (D-714): the message of an exception can quote the conversation; redaction masks
    identifiers, not sentences - so no message is logged at INFO and above, at any log level."""
    path = configure_logging(tmp_path, level="INFO", console=False)
    log = get_logger("test.exc")
    try:
        failing(f"cannot send {BODY} to {PHONE}")
    except ValueError:
        log.exception("send_failed", attempt=2)
    entry = read_lines(path)[0]
    assert entry["level"] == "ERROR" and entry["event"] == "send_failed" and entry["attempt"] == 2
    assert entry["exc"] == "ValueError"  # the type, nothing else
    assert "failing" in str(entry["stack"]) and "test_logging.py" in str(entry["stack"])
    written = path.read_text(encoding="utf-8")
    assert BODY not in written and PHONE not in written and "[手机号]" not in written
    assert "exc_detail" not in written  # no record of the message exists at INFO


def test_a_debug_log_keeps_the_redacted_message_in_a_record_of_its_own(tmp_path: Path) -> None:
    path = configure_logging(tmp_path, level="DEBUG", console=False)
    log = get_logger("test.exc")
    try:
        failing(f"cannot send {BODY} to {PHONE}")
    except ValueError:
        log.exception("send_failed")
    entries = read_lines(path)
    assert [(e["level"], e["event"]) for e in entries] == [
        ("ERROR", "send_failed"),
        ("DEBUG", "send_failed_detail"),
    ]
    error, detail = entries
    assert error["exc"] == "ValueError" and "exc_detail" not in error
    assert BODY not in json.dumps(error, ensure_ascii=False)  # nothing above DEBUG has it
    assert PHONE not in str(detail["exc_detail"]) and "[手机号]" in str(detail["exc_detail"])
    assert BODY in str(detail["exc_detail"])  # a sentence is not an identifier: DEBUG only


def test_the_console_shows_the_type_and_the_stack_not_the_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging(tmp_path, level="INFO", console=True, console_level=logging.INFO)
    try:
        failing(f"cannot send {BODY}")
    except ValueError:
        get_logger("test.exc").exception("send_failed")
    shutdown_logging()
    shown = capsys.readouterr().err
    assert "ValueError" in shown and "failing" in shown and BODY not in shown


class WithFacts(Exception):
    """An exception whose attributes are what the log may say about it."""

    def __init__(self, text: str, **attributes: object) -> None:
        super().__init__(text)
        for name, value in attributes.items():
            setattr(self, name, value)


def caught(exc: BaseException) -> BaseException:
    try:
        raise exc
    except BaseException as found:
        return found


def test_the_facts_of_the_project_errors_and_of_the_http_libraries_are_kept() -> None:
    request = httpx.Request("GET", "https://example.invalid/x")
    response = httpx.Response(429, request=request)
    unauthorized = openai.AuthenticationError(
        f"bad key {BODY}", response=httpx.Response(401, request=request), body=None
    )
    expected = [
        (ApiError(f"{BODY}", status=503, retryable=True), "ApiError status=503 retryable=true"),
        (IlinkHttpError("getupdates", 500), "IlinkHttpError status=500 endpoint=getupdates"),
        (StyleModelError(BODY, kind="timeout"), "StyleModelError kind=timeout"),
        (CliError(BODY, ExitCode.BUSY), "CliError code=busy"),
        (
            httpx.HTTPStatusError(BODY, request=request, response=response),
            "HTTPStatusError status_code=429",
        ),
        (unauthorized, "AuthenticationError status_code=401"),
        (FileNotFoundError(2, BODY), "FileNotFoundError errno=2"),
        (BudgetDeniedError("reply", 3), "BudgetDeniedError level=3 purpose=reply"),
        (httpx.ConnectError(BODY), "ConnectError"),
    ]
    for exc, summary in expected:
        assert exception_summary(exc) == summary
        assert BODY not in summary


def test_an_attribute_that_could_be_a_sentence_is_not_a_fact() -> None:
    sentence = WithFacts(BODY, kind=BODY, purpose="two words", endpoint=f"{BODY}/x", code=BODY)
    assert exception_summary(sentence) == "WithFacts"
    odd_types = WithFacts("m", status="503", attempts=True, retryable="yes", timeout=1)
    assert exception_summary(odd_types) == "WithFacts"  # a flag is a bool, a status an int
    raising = WithFacts("m")

    class Broken(Exception):
        @property
        def response(self) -> object:
            raise RuntimeError("no response")

    assert exception_summary(Broken(BODY)) == "Broken" and exception_summary(raising) == "WithFacts"


def chained() -> BaseException:
    try:
        try:
            raise ApiError(BODY, status=502)
        except ApiError as inner:
            raise RuntimeError(BODY) from inner
    except RuntimeError as outer:
        return outer


def suppressed() -> BaseException:
    try:
        try:
            raise ValueError(BODY)
        except ValueError:
            raise KeyError(BODY) from None  # the cause is hidden on purpose: not named
    except KeyError as outer:
        return outer


def test_the_causes_are_named_by_their_types_and_facts_only() -> None:
    summary = "RuntimeError <- caused by ApiError status=502 retryable=false"
    assert exception_summary(chained()) == summary
    assert exception_summary(suppressed()) == "KeyError"
    loop = ValueError("a")
    loop.__cause__ = loop  # a cycle must not hang the logger
    assert exception_summary(loop) == "ValueError"


@given(
    message=st.text(max_size=200),
    kind=st.sampled_from([ValueError, RuntimeError, KeyError, OSError, LookupError, TypeError]),
)
def test_no_message_ever_reaches_the_summary_or_the_stack(
    message: str, kind: type[Exception]
) -> None:
    """R17-44: whatever the text, the description of a plain exception is its type name."""
    found = caught(kind(message))
    summary, stack = describe_exception((type(found), found, found.__traceback__))
    assert summary == kind.__name__
    assert "\n" not in stack and "(" not in stack  # file:line:function only


def test_the_detail_is_the_redacted_message_shortened() -> None:
    detail = exception_detail(RuntimeError(f"{PHONE} " + "x" * 5000))
    assert detail.startswith("[手机号] x") and len(detail) == EXC_DETAIL_LIMIT
    assert exception_detail(RuntimeError("")) == ""
    assert BODY in exception_detail(ValueError(BODY))


def test_long_exception_messages_do_not_make_long_summaries() -> None:
    found = caught(RuntimeError("x" * 5000))
    summary, stack = describe_exception((type(found), found, found.__traceback__))
    assert summary == "RuntimeError" and len(stack) < 400


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
    assert sanitize_fields(logging.INFO, {"content": "a", "n": 1}) == {
        "n": 1,
        "omitted": ["content"],
    }
    assert sanitize_fields(logging.DEBUG, {"Content": PHONE})["Content"] == "[手机号]"


def test_rotation_is_ten_megabytes_times_ten_files(tmp_path: Path) -> None:
    configure_logging(tmp_path, console=False)
    handlers = [
        h
        for h in logging.getLogger("twin").handlers
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
