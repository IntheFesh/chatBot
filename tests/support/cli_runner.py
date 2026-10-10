"""Running the ``twin`` command line in a test without losing the reason for a crash."""

from __future__ import annotations

from typer.testing import CliRunner

from twin.cli import app
from twin.services import get_cli_context


def invoke(runner: CliRunner, args: list[str], *, answer: str | None = None) -> tuple[int, str]:
    """``(exit code, output)`` of one invocation.

    Click turns an exception that escapes a command into exit code 1 and an empty explanation,
    which is indistinguishable from a command that failed on purpose.  An unexpected exception is
    raised here instead, with its traceback, so that a failure on another platform says why.

    A real invocation is a process of its own: when it ends, so do its database connections.
    Here they would live on in the test process - and Windows refuses to replace or delete a
    database file that a connection still has open, in the next invocation of the same test - so
    the context of the invocation is closed explicitly.
    """
    try:
        result = runner.invoke(app, args, input=answer)
    finally:
        get_cli_context().reset()
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result.exit_code, result.output
