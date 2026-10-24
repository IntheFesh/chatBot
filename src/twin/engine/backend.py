"""The generation backends of the reply pipeline (R-ENG-006, R-SCOPE-007).

A backend turns the material of one round into the **raw text** of a reply; everything after that
- parsing, post-processing, stickers, violations, retries - is the pipeline's and is the same for
every backend.  ``deepseek`` (this round), ``style`` and ``hybrid`` (round 09 step 2) all
implement :class:`ReplyBackend`.

A backend does not catch its own failures: a timeout or an API error is raised and the pipeline
turns it into the signal that the engine must fall back (R-ENG-010).  What a backend *can* know
and the pipeline cannot is that the model refused (``refused``) or - for the hybrid backend,
whose planner may decide not to answer - that the plan says so (``skip_reply``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from twin.engine.dataview import ReplyDataView
from twin.engine.types import ReplyContext, ReplyMaterial, UsageSummary


@dataclass(frozen=True)
class BackendRequest:
    """One call of a backend."""

    context: ReplyContext
    data: ReplyDataView
    material: ReplyMaterial
    notes: tuple[str, ...] = ()  # what the previous attempt got wrong (R-ENG-008)
    attempt: int = 1
    non_thinking: bool = False  # the last-resort retry: no thinking whatever the setting says


@dataclass(frozen=True)
class BackendResult:
    """What a backend returns."""

    text: str
    backend: str
    thinking: bool
    cost_usd: float
    usage: UsageSummary
    latency_ms: int
    reasoning: str | None = None  # never put back into a prompt, never sent to the user
    plan: dict[str, Any] | None = None  # the hybrid planner's JSON
    refused: bool = False  # the model refused to produce a reply (R-SAFE-005)
    skip_reply: bool = False  # the plan says: do not answer this
    skip_reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class ReplyBackend(Protocol):
    """Anything that can write the raw text of a reply."""

    name: str

    async def generate(self, request: BackendRequest) -> BackendResult:
        """Write the reply; raise on a failure (the pipeline turns it into a fallback)."""
        ...
