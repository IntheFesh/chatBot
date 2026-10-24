"""Re-labelling the calls of a stretch of code: purpose and budget account (R-EVAL-009, R-LLM-014).

The evaluation sandbox runs the reply pipeline - the same code the running bot uses - and the
pipeline's backends call DeepSeek with the purposes ``reply`` and ``plan``.  Those calls must not
be booked as the bot's own replies: they are evaluation, paid from a one-time batch (R-LLM-014).
Instead of copying the backends, the sandbox wraps the run in :func:`call_scope`::

    with call_scope(Purpose.EVAL, LedgerTag("one_time", batch_id)):
        draft = await pipeline.run(context, data_view)

Inside the scope :meth:`~twin.llm.deepseek.DeepSeekClient.chat` writes every ledger row with the
scope's purpose and account (and a batch call is checked against the batch's cap), but still
picks the **model** of the purpose the caller asked for: an evaluated reply is written by the
chat model, exactly like a real one.  The scope is a context variable, so it follows the task and
the worker threads started from it and never leaks into other tasks of the same process.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from twin.llm.types import LedgerTag, Purpose


@dataclass(frozen=True)
class CallScope:
    """How the model calls inside the scope are booked."""

    purpose: Purpose
    tag: LedgerTag


_scope: ContextVar[CallScope | None] = ContextVar("twin_llm_call_scope", default=None)


def current_call_scope() -> CallScope | None:
    """The scope the running code is inside of, or ``None``."""
    return _scope.get()


@contextmanager
def call_scope(purpose: Purpose, tag: LedgerTag) -> Iterator[CallScope]:
    """Book every DeepSeek call made inside the block under ``purpose`` and ``tag``."""
    scope = CallScope(purpose, tag)
    token = _scope.set(scope)
    try:
        yield scope
    finally:
        _scope.reset(token)
