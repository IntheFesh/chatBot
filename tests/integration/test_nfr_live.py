"""R-NFR-001 against the real DeepSeek: how long does a reply take (``live``, skipped by default).

Run it with ``TWIN_LIVE=1`` and ``TWIN_LIVE_DEEPSEEK_KEY=<key>`` set:
``uv run pytest tests/integration/test_nfr_live.py -s``.
It asks the real API for 20 replies without thinking and 5 with thinking, with prompts that have
the shape of a reply request (a persona's rules, a few turns of synthetic chat, the newest message;
no real chat content), and checks the 95th percentile of the time each took - the time the client
measures, which does not include the engine's deliberate delay - against the limits of the
specification: below 20 seconds, and below 60 with thinking.  A request that fails does not enter
the sample; if too few succeed the test fails instead of judging.  Nothing is invented: the numbers
printed are the ones measured, and ``TWIN_LIVE_REPORT=<file>`` writes them as JSON so that they can
be copied into ``docs/PERFORMANCE.md``.  Cost: well under US$0.05.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.support.scripts import load_script
from twin.clock import SystemClock
from twin.config.loader import load_settings
from twin.llm.deepseek import DeepSeekClient
from twin.llm.errors import LlmError
from twin.llm.pricing import Pricing
from twin.llm.types import ChatMessage, Purpose

pytestmark = pytest.mark.live

percentile = load_script("bench_runtime").percentile
SAMPLES = 20
THINKING_SAMPLES = 5
LIMIT_S = 20.0
LIMIT_THINKING_S = 60.0
MIN_SUCCESSFUL = 0.8  # of the requests; below that the network, not the model, is what was measured

RULES = (
    "你在微信上和对方聊天，用很短的句子，像朋友一样自然地说话，一次回一到三条，不要列清单，"
    "不要说自己是AI，不要答应打电话、视频或见面。每条消息单独一行。"
)
TURNS = (
    ("user", "今天好累啊"),
    ("assistant", "怎么啦"),
    ("user", "刚开完会，一直在说话"),
    ("assistant", "辛苦啦 快去喝点水"),
    ("user", "晚上想吃点好的"),
    ("assistant", "吃什么呀"),
)
QUESTIONS = (
    "火锅怎么样",
    "你今天做什么了",
    "周末有什么安排吗",
    "我明天要早起",
    "帮我想想晚饭吃什么",
)


@pytest.fixture
async def client() -> DeepSeekClient:
    key = os.environ.get("TWIN_LIVE_DEEPSEEK_KEY")
    if not key:
        pytest.skip("set TWIN_LIVE_DEEPSEEK_KEY to run the live DeepSeek checks")
    settings = load_settings()
    live = DeepSeekClient(
        config=settings.deepseek,
        pricing=Pricing.from_settings(settings),
        clock=SystemClock(),
        api_key=lambda: key,
    )
    yield live
    await live.aclose()


def reply_prompt(number: int) -> list[ChatMessage]:
    turns: list[ChatMessage] = [{"role": role, "content": text} for role, text in TURNS]  # type: ignore[typeddict-item]
    asked = QUESTIONS[number % len(QUESTIONS)]
    return [
        {"role": "system", "content": RULES},
        *turns,
        {"role": "user", "content": asked},
    ]


async def sample(client: DeepSeekClient, count: int, *, thinking: bool) -> list[float]:
    """The seconds each of ``count`` replies took (the requests that failed are left out)."""
    seconds: list[float] = []
    for number in range(count):
        try:
            result = await client.chat(
                reply_prompt(number),
                purpose=Purpose.REPLY,
                thinking=thinking,
                temperature=1.0,
                max_tokens=4000 if thinking else 600,
            )
        except LlmError:
            continue
        seconds.append(result.latency_ms / 1000.0)
    return seconds


async def test_a_real_reply_takes_less_than_the_limits_of_the_specification(
    client: DeepSeekClient,
) -> None:
    plain = await sample(client, SAMPLES, thinking=False)
    thoughtful = await sample(client, THINKING_SAMPLES, thinking=True)
    found = {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": f"{platform.system()} {platform.release()}",
        "plain": {
            "requested": SAMPLES,
            "answered": len(plain),
            "p50_s": percentile(plain, 50),
            "p95_s": percentile(plain, 95),
            "max_s": max(plain, default=0.0),
            "limit_s": LIMIT_S,
        },
        "thinking": {
            "requested": THINKING_SAMPLES,
            "answered": len(thoughtful),
            "p50_s": percentile(thoughtful, 50),
            "p95_s": percentile(thoughtful, 95),
            "max_s": max(thoughtful, default=0.0),
            "limit_s": LIMIT_THINKING_S,
        },
    }
    print(json.dumps(found, ensure_ascii=False, indent=2))
    target = os.environ.get("TWIN_LIVE_REPORT")
    if target:
        text = json.dumps(found, ensure_ascii=False, indent=2)
        await asyncio.to_thread(Path(target).write_text, text, encoding="utf-8")
    assert len(plain) >= SAMPLES * MIN_SUCCESSFUL, "too few requests were answered to judge"
    assert len(thoughtful) >= int(THINKING_SAMPLES * MIN_SUCCESSFUL)
    assert percentile(plain, 95) < LIMIT_S  # R-NFR-001, without thinking
    assert percentile(thoughtful, 95) < LIMIT_THINKING_S  # and with it
