"""Response builders for the DeepSeek chat API, for tests that intercept HTTP with respx."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from tests.support.clock import ManualClock
from tests.support.waiting import wait_until

API = "https://api.deepseek.com/chat/completions"
TEST_KEY = "synthetic-test-key-0001"


def completion(
    content: str = "好的",
    *,
    reasoning: str | None = None,
    prompt: int = 100,
    completion_tokens: int = 10,
    hit: int | None = 0,
    miss: int | None = None,
    reasoning_tokens: int = 0,
    finish: str = "stop",
    model: str = "deepseek-flash",
    request_id: str = "req-1",
) -> dict[str, Any]:
    """A chat completion body with DeepSeek's cache fields in ``usage``."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    usage: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt + completion_tokens,
    }
    if hit is not None:
        usage["prompt_cache_hit_tokens"] = hit
        usage["prompt_cache_miss_tokens"] = prompt - hit if miss is None else miss
    if reasoning_tokens:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": 1_790_000_000,
        "model": model,
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        "usage": usage,
    }


def ok(**kwargs: Any) -> httpx.Response:
    return httpx.Response(200, json=completion(**kwargs))


def error(status: int, message: str = "boom", **headers: str) -> httpx.Response:
    return httpx.Response(
        status,
        json={"error": {"message": message, "type": "error", "param": None, "code": None}},
        headers=headers,
    )


def request_json(request: httpx.Request) -> dict[str, Any]:
    """The decoded JSON body of a request the client sent."""
    body = json.loads(request.content)
    assert isinstance(body, dict)
    return body


# ------------------------------------------------- slow answers on the manual clock (R18-D6)

Effect = Callable[[httpx.Request], httpx.Response]


def timing_out(clock: ManualClock, seconds: float) -> Effect:
    """A respx side effect: the request takes ``seconds`` of the clock, then times out.

    This is how a request that never answers ends on the client side: the SDK's own timeout
    (``timeout_s``) gives up and raises ``ReadTimeout``.  The clock moves, nobody sleeps.
    """

    def effect(request: httpx.Request) -> httpx.Response:
        clock.tick(seconds)
        raise httpx.ReadTimeout("no answer", request=request)

    return effect


def slow(clock: ManualClock, seconds: float, response: httpx.Response) -> Effect:
    """A respx side effect: ``response`` arrives after ``seconds`` of the clock."""

    def effect(request: httpx.Request) -> httpx.Response:
        clock.tick(seconds)
        return response

    return effect


async def on_clock[T](clock: ManualClock, work: Awaitable[T], *, limit_s: float = 5.0) -> T:
    """Run ``work``; whenever it sleeps on the clock, move the clock to the end of that sleep.

    The clock then says exactly how long the work took in the time it believes in (back-off
    pauses included), and code that would wait forever on a clock nobody moves does not hang.
    """
    task = asyncio.ensure_future(work)
    try:
        while True:
            await wait_until(lambda: task.done() or clock.pending_sleepers >= 1, limit_s=limit_s)
            if task.done():
                return await task
            await clock.advance(clock.sleeps[-1])
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
