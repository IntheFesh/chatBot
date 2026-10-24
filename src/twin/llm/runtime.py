"""Wiring of the LLM layer from the application's services.

:func:`build_llm_runtime` creates the pieces that belong together - prices and peak calendar,
ledger, budget manager, token estimator, one-time batches and the DeepSeek client - from a
:class:`~twin.services.Services` container, and registers the production off-peak policy with
the job queue (R-ARCH-003, R-LLM-007).  :func:`activate_offpeak_policy` does only that last step
and needs no API key: the job worker calls it before it starts.

There is **one runtime per container** (D-701, round 17): the engine, every job handler, the
sticker description, the commands and the health check all ask for it and get the same object,
so the circuit breaker, the concurrency limit (``deepseek.max_concurrency`` is the limit of the
process, not of each caller), the token calibration, the budget manager and its events are
shared.  It is kept in ``services.extras``; whoever ends the process or the command calls
:func:`close_llm_runtime` - a job never closes it, the others are using it.
"""

from __future__ import annotations

import threading
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from twin.llm.budget import BudgetManager, StyleBackendStatus
from twin.llm.capabilities import CapabilityStore, LlmCapabilities
from twin.llm.deepseek import DeepSeekClient
from twin.llm.health import llm_health_of
from twin.llm.layout import CacheMonitor
from twin.llm.ledger import LedgerStore
from twin.llm.onetime import OneTimeBatches
from twin.llm.pricing import Pricing, install_offpeak_policy
from twin.llm.reliability import CircuitBreaker
from twin.llm.tokens import TokenEstimator
from twin.ops.jobs import JobQueue
from twin.ops.logging import get_logger
from twin.schedule.service import peak_calendar_for, time_service_for
from twin.schedule.time_service import TimeService

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.llm.runtime")

DEEPSEEK_SECRET = "deepseek_api_key"  # noqa: S105 - the credential store entry name, not a secret
EXTRAS_KEY = "llm_runtime"
# The health check builds the runtime from a worker thread while the application starts it on
# the loop; one lock for the process is enough, building takes milliseconds.
_BUILD_LOCK = threading.Lock()


@dataclass
class LlmRuntime:
    """Everything the rest of the application needs to talk to DeepSeek."""

    pricing: Pricing
    time_service: TimeService
    ledger: LedgerStore
    estimator: TokenEstimator
    cache: CacheMonitor
    capabilities: CapabilityStore
    budget: BudgetManager
    batches: OneTimeBatches
    client: DeepSeekClient

    async def aclose(self) -> None:
        """Release the connections.  The runtime stays usable: the client opens them again."""
        await self.client.aclose()

    async def on_state_change(self, old_version: int, new_version: int) -> None:
        """A state-watcher listener: notice a replaced or deleted API key (D-702)."""
        await self.client.on_state_change(old_version, new_version)


def _pricing(services: Services) -> Pricing:
    """The price table on the container's one holiday calendar (the day types use it too)."""
    settings = services.settings
    return Pricing(
        settings.pricing_usd_per_mtok,
        offpeak_multiplier=settings.pricing.offpeak_multiplier,
        calendar=peak_calendar_for(services),
    )


def activate_offpeak_policy(services: Services) -> Pricing:
    """Register the production off-peak policy with the job queue; returns the price table."""
    pricing = _pricing(services)
    install_offpeak_policy(pricing)
    return pricing


def build_llm_runtime(services: Services, *, style: StyleBackendStatus | None = None) -> LlmRuntime:
    """The LLM layer of this process: made on the first call, the same one after that.

    The API key is read when the first request goes out, not here.  ``style`` (who answers "may
    the style model take over?") is applied to the shared budget manager whenever it is given.
    """
    with _BUILD_LOCK:
        found = services.extras.get(EXTRAS_KEY)
        if isinstance(found, LlmRuntime):
            if style is not None:
                found.budget.set_style_status(style)
            return found
        runtime = _make_runtime(services, style)
        services.extras[EXTRAS_KEY] = runtime
        return runtime


async def close_llm_runtime(services: Services) -> None:
    """End the shared runtime: close its connections and forget it (the next ask builds anew).

    Called where the process or a command ends (``twin run``, the one-shot commands); calling it
    when there is none, or twice, does nothing.
    """
    with _BUILD_LOCK:
        found = services.extras.pop(EXTRAS_KEY, None)
    if isinstance(found, LlmRuntime):
        await found.aclose()


async def closing_llm_runtime[T](services: Services, work: Awaitable[T]) -> T:
    """Await ``work``, then close the shared runtime: the coroutine of a one-shot command."""
    try:
        return await work
    finally:
        await close_llm_runtime(services)


def _make_runtime(services: Services, style: StyleBackendStatus | None) -> LlmRuntime:
    settings = services.settings
    clock = services.clock
    pricing = activate_offpeak_policy(services)
    time_service = time_service_for(services)
    ledger = LedgerStore(services.db, clock, time_service)
    capabilities = CapabilityStore(services.db, clock)
    estimator = TokenEstimator(image_tokens=capabilities.get().image_table())
    with services.db.session() as session:
        estimator.load(session)
    applied = {"measured_at": capabilities.get().measured_at}

    def current_capabilities() -> LlmCapabilities:
        """The stored capabilities; a new probe result also updates the image estimates."""
        caps = capabilities.get()
        if caps.measured_at != applied["measured_at"]:
            estimator.set_image_table(caps.image_table())
            applied["measured_at"] = caps.measured_at
        return caps

    def save_calibration(est: TokenEstimator) -> None:
        with services.db.transaction(bump_state=False) as session:
            est.save(session, clock)

    budget = BudgetManager(
        settings.budget,
        examples_k=settings.engine.examples_k,
        ledger=ledger,
        time_service=time_service,
        clock=clock,
        db=services.db,
        alerts=services.alerts,
        style=style,
    )
    batches = OneTimeBatches(
        queue=JobQueue(services.db, clock),
        ledger=ledger,
        pricing=pricing,
        estimator=estimator,
        budget=settings.budget,
        db=services.db,
        clock=clock,
        alerts=services.alerts,
    )
    cache = CacheMonitor(estimator)
    client = DeepSeekClient(
        config=settings.deepseek,
        pricing=pricing,
        clock=clock,
        api_key=lambda: services.secrets.require(DEEPSEEK_SECRET),
        ledger=ledger,
        alerts=services.alerts,
        capabilities=current_capabilities,
        breaker=CircuitBreaker(clock),
        estimator=estimator,
        cache_monitor=cache,
        budget=budget,
        batches=batches,
        save_calibration=save_calibration,
        health=llm_health_of(services),
    )
    return LlmRuntime(
        pricing=pricing,
        time_service=time_service,
        ledger=ledger,
        estimator=estimator,
        cache=cache,
        capabilities=capabilities,
        budget=budget,
        batches=batches,
        client=client,
    )
