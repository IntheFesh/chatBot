"""The style model's side of the running application: client, backends and selector, wired once.

::

    llm = build_llm_runtime(services)
    style = StyleRuntime.from_services(services, llm)
    llm.budget.set_style_status(style.selector)                  # R-LLM-008, level 4
    pipeline = ReplyPipeline.from_services(services, llm, extra_backends=style.backends)
    ...
    choice = await style.selector.choose()                       # per reply
    draft = await pipeline.run(replace(context, backend=choice.name), data)
    style.selector.record(choice.name, draft)
    ...
    monitor = style.monitor(services)                            # a component of the application
    await style.aclose()

:class:`StyleRuntime` builds nothing that talks to a server when it is created: the client is made
on first use (:class:`ConfiguredStyleClient`), so a configuration that cannot work - a vLLM server
without the name of its LoRA adapter, say - shows up as an unavailable style model and a fallback
to DeepSeek, not as a program that does not start.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from twin.clock import Clock
from twin.config.settings import StyleModelConfig
from twin.engine.backend import ReplyBackend
from twin.engine.backend_select import BackendMonitorComponent, BackendSelector
from twin.engine.hybrid_backend import HybridBackend, PlanPromptBuilder
from twin.engine.style_backend import StyleBackend, StyleSampling, StyleWriter
from twin.engine.style_models import StyleModels
from twin.engine.style_prompt import StylePromptBuilder
from twin.llm.errors import StyleModelError
from twin.llm.style_client import (
    RenderedPrompt,
    StyleHealth,
    StyleModelClient,
    StyleOutput,
    StyleParams,
    style_client_from_config,
)
from twin.stickers.tags import load_vocabulary
from twin.training.registry import LockedVersions

if TYPE_CHECKING:
    from twin.llm.runtime import LlmRuntime
    from twin.services import Services


class ConfiguredStyleClient:
    """The client ``style_model.*`` describes, built on first use (see the module description)."""

    def __init__(
        self, config: StyleModelConfig, clock: Clock, *, api_key: str | None = None
    ) -> None:
        self._config = config
        self._clock = clock
        self._api_key = api_key
        self._client: StyleModelClient | None = None

    def _real(self) -> StyleModelClient:
        if self._client is None:
            try:
                self._client = style_client_from_config(
                    self._config, self._clock, api_key=self._api_key
                )
            except ValueError as exc:
                raise StyleModelError(str(exc), kind="unavailable") from exc
        return self._client

    async def generate(self, prompt: RenderedPrompt, params: StyleParams) -> StyleOutput:
        return await self._real().generate(prompt, params)

    async def health(self) -> StyleHealth:
        try:
            client = self._real()
        except StyleModelError as exc:
            return StyleHealth(False, str(exc))
        return await client.health()

    async def tokenize(self, text: str) -> list[int]:
        return await self._real().tokenize(text)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


@dataclass
class StyleRuntime:
    """Everything of the style model that the application holds on to."""

    client: StyleModelClient
    models: StyleModels
    selector: BackendSelector
    style: StyleBackend
    hybrid: HybridBackend

    @property
    def backends(self) -> dict[str, ReplyBackend]:
        """The backends to register with the pipeline (``extra_backends=``)."""
        return {self.style.name: self.style, self.hybrid.name: self.hybrid}

    @classmethod
    def from_services(
        cls,
        services: Services,
        llm: LlmRuntime,
        *,
        client: StyleModelClient | None = None,
    ) -> StyleRuntime:
        """The style model of the running application (``client``: another transport, for tests)."""
        settings = services.settings
        chosen = client or ConfiguredStyleClient(settings.style_model, services.clock)
        models = StyleModels(services.db, mode=settings.style_model.mode)

        def builder(locked: LockedVersions) -> StylePromptBuilder:
            return StylePromptBuilder.from_services(services, locked=locked)

        writer = StyleWriter(
            client=chosen,
            models=models,
            builders=builder,
            sampling=StyleSampling.from_config(settings.style_model),
        )
        tags = load_vocabulary(settings, services.paths.root).tags

        def thinking_allowed() -> bool:
            return llm.budget.limits().chat_thinking_allowed

        hybrid = HybridBackend(
            writer,
            llm.client,
            PlanPromptBuilder.from_services(services, tags),
            thinking_allowed=thinking_allowed,
            auto_rules=settings.thinking.auto_rules,
        )
        style = StyleBackend(
            writer,
            planner=hybrid,
            thinking_allowed=thinking_allowed,
            auto_rules=settings.thinking.auto_rules,
        )
        selector = BackendSelector(
            runtime=services.runtime,
            models=models,
            client=chosen,
            config=settings.backend,
            clock=services.clock,
            alerts=services.alerts,
            limits=llm.budget.limits,
        )
        return cls(chosen, models, selector, style, hybrid)

    def monitor(self, services: Services) -> BackendMonitorComponent:
        """The component that keeps looking at the style model (register it with the app)."""
        return BackendMonitorComponent(
            self.selector,
            services.clock,
            services.alerts,
            interval_s=services.settings.backend.health_check_s,
        )

    async def aclose(self) -> None:
        await self.client.aclose()
