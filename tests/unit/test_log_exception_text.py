"""R-OPS-007, CLAUDE.md rule 6 (round 17 item 44, D-714): logs never carry an exception's text.

``StructLogger`` drops fields named like content and masks identifiers at DEBUG, but the message
of an exception is neither: it is whatever the code that raised it formatted, and it can quote a
sentence of the conversation.  So a log call at INFO and above does not take the exception (or a
piece of its text) as a field value; it takes :func:`~twin.ops.logging.exception_summary`, the
type's name, or nothing - ``log.exception`` adds the type and the stack by itself.

The test reads every ``log.<level>(...)`` call of the application that sits inside an
``except ... as name`` block and fails for each keyword value that uses ``name`` other than as the
argument of ``type``, ``exception_summary``, ``describe_failure`` or ``exception_facts``, or as an
attribute that holds a number or a code (``name.code``, ``name.kind``, ``name.status``, ...).  A
call that is allowed to keep its text is listed below with the reason; the list may only shrink.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "twin"
LEVELS = {"info", "warning", "error", "critical", "exception"}  # debug is redacted and DEBUG-only
SAFE_CALLS = {"type", "exception_summary", "describe_failure", "exception_facts"}
# attributes our exceptions fill with a number, a flag or a word of a closed vocabulary
STRUCTURED = {"code", "kind", "status", "status_code", "retry_in_s", "lock_name", "endpoint"}

# (file relative to src/twin, event name) -> why what it logs is not chat
ALLOWED = {
    ("llm/deepseek.py", "ledger_write_failed"): (
        "safe_message(): the error text of the cost ledger's own database write (SQL, numbers), "
        "keys and identifiers removed; no chat reaches it"
    ),
    ("llm/deepseek.py", "calibration_save_failed"): (
        "safe_message(): the error text of the token calibration's own database write"
    ),
    ("channel/ilink/inbound.py", "inbound_media_failed"): (
        "MediaFetchError.reason is one of a few codes (download_failed, bad_key, ...)"
    ),
    ("channel/ilink/outbound.py", "image_upload_failed"): (
        "the reason of an upload error is a code of the upload steps, not a text"
    ),
    ("serving/server.py", "server_blocked"): (
        "ServerBlocked.reason is a code (start_failed, model_changed, ...)"
    ),
}


def _uses(node: ast.AST, name: str) -> bool:
    """Does ``node`` use ``name`` other than as the argument of one of the safe calls?"""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name | ast.Attribute):
        called = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        if called in SAFE_CALLS:
            return False
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == name
        and node.attr in STRUCTURED
    ):
        return False
    if isinstance(node, ast.Name):
        return node.id == name
    return any(_uses(child, name) for child in ast.iter_child_nodes(node))


def _log_calls(handler: ast.ExceptHandler) -> list[ast.Call]:
    found = []
    for node in ast.walk(handler):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in LEVELS
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and _is_logger(node.func.value)
        ):
            found.append(node)
    return found


def _is_logger(node: ast.expr) -> bool:
    name = node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", "")
    return name in {"log", "_log", "logger", "_logger"}


def offenders() -> list[tuple[str, int, str]]:
    found: list[tuple[str, int, str]] = []
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for handler in (n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler) and n.name):
            for call in _log_calls(handler):
                event = str(call.args[0].value)  # type: ignore[attr-defined]
                if (
                    any(_uses(keyword.value, str(handler.name)) for keyword in call.keywords)
                    and (relative, event) not in ALLOWED
                ):
                    found.append((relative, call.lineno, event))
    return found


def test_no_log_call_at_info_or_above_takes_the_text_of_the_exception_it_handles() -> None:
    assert offenders() == []


def test_the_allowed_calls_still_exist() -> None:
    """An entry that no longer matches a call is removed, so the list only shrinks."""
    present: set[tuple[str, str]] = set()
    for relative, event in ALLOWED:
        text = (SRC / relative).read_text(encoding="utf-8")
        if f'"{event}"' in text:
            present.add((relative, event))
    assert present == set(ALLOWED)


@pytest.mark.parametrize(
    ("source", "flagged"),
    [
        ('try:\n    f()\nexcept E as exc:\n    log.warning("x", error=str(exc))\n', True),
        ('try:\n    f()\nexcept E as exc:\n    log.info("x", reason=str(exc)[:80])\n', True),
        ('try:\n    f()\nexcept E as exc:\n    log.error("x", reason=f"bad {exc}")\n', True),
        ('try:\n    f()\nexcept E as exc:\n    log.error("x", reason=exc.args)\n', True),
        ('try:\n    f()\nexcept E as exc:\n    log.error("x", reason=exc.reason)\n', True),
        ('try:\n    f()\nexcept E as exc:\n    log.error("x", reason=exc.code)\n', False),
        (
            'try:\n    f()\nexcept E as exc:\n    log.warning("x", error=type(exc).__name__)\n',
            False,
        ),
        (
            'try:\n    f()\nexcept E as exc:\n    log.warning("x", error=exception_summary(exc))\n',
            False,
        ),
        ('try:\n    f()\nexcept E as exc:\n    log.debug("x", error=str(exc))\n', False),
        ('try:\n    f()\nexcept E:\n    log.exception("x")\n', False),
    ],
)
def test_the_scan_sees_what_it_is_meant_to_see(source: str, flagged: bool) -> None:
    tree = ast.parse(source)
    handler = next(n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler))
    calls = _log_calls(handler)
    hit = bool(handler.name) and any(
        _uses(keyword.value, str(handler.name)) for call in calls for keyword in call.keywords
    )
    assert hit is flagged
