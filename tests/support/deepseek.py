"""Response builders for the DeepSeek chat API, for tests that intercept HTTP with respx."""

from __future__ import annotations

import json
from typing import Any

import httpx

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
