"""Wiring of the LLM layer from the application's services.

:func:`build_llm_runtime` creates the pieces that belong together - prices and peak calendar,
ledger, budget manager, token estimator, one-time batches and the DeepSeek client - from a
:class:`~twin.services.Services` container, and registers the production off-peak policy with
the job queue (R-ARCH-003, R-LLM-007).  :func:`activate_offpeak_policy` does only that last step
and needs no API key: the job worker calls it before it starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from twin.config.runtime import BOT_TIMEZONE
from twin.llm.budget import BudgetManager, StyleBackendStatus
from twin.llm.capabilities import CapabilityStore, LlmCapabilities
from twin.llm.deepseek import DeepSeekClient
from twin.llm.layout import CacheMonitor
from twin.llm.ledger import LedgerStore
from twin.llm.onetime import OneTimeBatches
from twin.llm.pricing import Pricing, install_offpeak_policy
from twin.llm.reliability import CircuitBreaker
from twin.llm.tokens import TokenEstimator
from twin.ops.jobs import JobQueue
from twin.ops.logging import get_logger
from twin.schedule.time_service import ConfiguredTimeService, TimeService

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.llm.runtime")

DEEPSEEK_SECRET = "deepseek_api_key"  # noqa: S105 - the credential store entry name, not a secret


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


def _pricing(services: Services) -> Pricing:
    def calendar_fallback(year: int) -> None:
        services.alerts.raise_alert(
            "calendar_out_of_range",
            f"the holiday calendar does not cover {year}; peak hours assume Monday to Friday",
            severity="warning",
            detail={"year": year},
            dedup_key=f"calendar_out_of_range:{year}",
        )

    return Pricing.from_settings(services.settings, on_fallback=calendar_fallback)


def activate_offpeak_policy(services: Services) -> Pricing:
    """Register the production off-peak policy with the job queue; returns the price table."""
    pricing = _pricing(services)
    install_offpeak_policy(pricing)
    return pricing


def build_llm_runtime(services: Services, *, style: StyleBackendStatus | None = None) -> LlmRuntime:
    """Create the LLM layer for this process (the API key is read on the first call)."""
    settings = services.settings
    clock = services.clock
    pricing = activate_offpeak_policy(services)
    time_service = ConfiguredTimeService(clock, lambda: services.runtime.get(BOT_TIMEZONE))
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
