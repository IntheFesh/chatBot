"""Registered style models and a scripted style-model server for the round 09 tests.

``register_model`` writes a row of ``model_registry`` the way ``twin model register`` would (the
versions it is locked to included) and optionally marks it active, as ``twin model activate``
will in round 14.  ``ScriptedStyleClient`` is a :class:`twin.llm.style_client.StyleModelClient`
whose answers and health a test controls and which keeps every prompt it was sent; the HTTP
clients themselves are tested against ``tests/support/style_server.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from twin.llm.errors import StyleModelError
from twin.llm.style_client import RenderedPrompt, StyleHealth, StyleOutput, StyleParams
from twin.services import Services
from twin.storage.training_models import ModelRegistryEntry
from twin.training.lf_template import TEMPLATE_VERSION


def register_model(
    services: Services,
    *,
    run_id: str = "r-test-run",
    quant: str = "Q5_K_M",
    kind: str = "gguf",
    active: bool = True,
    gate_passed: bool | None = True,
    template_version: str = TEMPLATE_VERSION,
    persona_version: str = "v1",
    profile_version: str = "p1",
    dataset_version: str = "ds-test-01",
) -> str:
    """A model file in the registry; returns its id (``<run_id>-<quant>``)."""
    now = services.clock.now_utc()
    row_id = f"{run_id}-{quant}"
    with services.db.transaction(bump_state=False) as session:
        session.add(
            ModelRegistryEntry(
                id=row_id,
                run_id=run_id,
                kind=kind,
                profile="5090-8b",
                base_model="Qwen/Qwen3-8B",
                quant=quant,
                path=f"{run_id}/{quant}.gguf",
                sha256="0" * 64,
                size=1000,
                template_version=template_version,
                persona_version=persona_version,
                profile_version=profile_version,
                dataset_version=dataset_version,
                eval={},
                enabled=active,
                active=active,
                gate_passed=gate_passed,
                created_at=now,
                updated_at=now,
            )
        )
    return row_id


@dataclass
class ScriptedStyleClient:
    """A style model whose replies and health the test decides."""

    replies: list[str | Exception] = field(default_factory=lambda: ["好呀"])
    healthy: bool = True
    detail: str = "ok"
    prompts: list[RenderedPrompt] = field(default_factory=list)
    params: list[StyleParams] = field(default_factory=list)
    health_calls: int = 0
    closed: bool = False
    truncated: bool = False
    on_generate: Callable[[], None] | None = None

    async def generate(self, prompt: RenderedPrompt, params: StyleParams) -> StyleOutput:
        self.prompts.append(prompt)
        self.params.append(params)
        if self.on_generate is not None:
            self.on_generate()
        reply = self.replies[min(len(self.prompts), len(self.replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return StyleOutput(
            text=reply,
            stop_reason="length" if self.truncated else "stop",
            prompt_tokens=120,
            completion_tokens=8,
            latency_ms=25,
        )

    async def health(self) -> StyleHealth:
        self.health_calls += 1
        return StyleHealth(self.healthy, self.detail if not self.healthy else "ok", 3)

    async def tokenize(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    async def aclose(self) -> None:
        self.closed = True


def down(detail: str = "connection refused") -> StyleModelError:
    """The error a style client raises when the server is not there."""
    return StyleModelError(detail, kind="unavailable")
