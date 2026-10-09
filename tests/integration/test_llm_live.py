"""Live checks against the real DeepSeek API.  Skipped unless ``TWIN_LIVE=1``.

Set ``TWIN_LIVE_DEEPSEEK_KEY`` to a key as well.  These tests send a few dozen tiny requests with
synthetic content (cost: well under US$0.10) and need network access to api.deepseek.com.  They
are the automated form of ``twin llm probe``; the CLI additionally stores the result and writes
``docs/LLM_REPORT.md``.
"""

from __future__ import annotations

import os

import pytest

from twin.clock import SystemClock
from twin.config.loader import load_settings
from twin.llm.deepseek import DeepSeekClient
from twin.llm.pricing import Pricing
from twin.llm.probe import LlmProbe

pytestmark = pytest.mark.live


@pytest.fixture
def client() -> DeepSeekClient:
    key = os.environ.get("TWIN_LIVE_DEEPSEEK_KEY")
    if not key:
        pytest.skip("set TWIN_LIVE_DEEPSEEK_KEY to run the live DeepSeek checks")
    settings = load_settings()
    return DeepSeekClient(
        config=settings.deepseek,
        pricing=Pricing.from_settings(settings),
        clock=SystemClock(),
        api_key=lambda: key,
    )


async def test_the_m0_probe_passes_against_the_real_api(client: DeepSeekClient) -> None:
    report = await LlmProbe(client, clock=SystemClock()).run()
    await client.aclose()
    assert report.fatal is None, report.fatal
    failed = [check.id for check in report.checks if check.gate and not check.passed]
    assert not failed, f"M0 checks failed: {failed}"
    assert report.m0_passed
    assert report.total_cost_usd < 0.5
