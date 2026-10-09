"""A stand-in for the DeepSeek chat endpoint that behaves like the documented API.

Use an instance as the ``respx`` side effect.  The knobs switch on the failures the M0 probe has
to notice.  Only the structure of the answers matters here; the content is made up.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from typing import Any

import httpx
from PIL import Image

from tests.support.deepseek import error, ok, request_json


@dataclass
class SimulatedDeepSeek:
    reject_gif: bool = False
    reject_jpeg: bool = False
    reject_detail: bool = False
    json_valid_with_thinking: bool = True
    json_valid_without_thinking: bool = True
    json_server_error: int | None = None
    detail_server_error: int | None = None
    reasoning_when_thinking: bool = True
    reasoning_when_not_thinking: bool = False
    cache_works: bool = True
    unauthorized: bool = False
    out_of_balance_after: int | None = None
    requests: list[dict[str, Any]] = field(default_factory=list)
    _seen: set[str] = field(default_factory=set)

    @staticmethod
    def image_tokens(width: int, height: int) -> int:
        """Tokens this simulation bills for an image: grows with size, capped at 1024."""
        return min(1024, max(16, width * height // 900))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = request_json(request)
        self.requests.append(body)
        if self.unauthorized:
            return error(401, "Authentication Fails")
        if self.out_of_balance_after is not None and len(self.requests) > self.out_of_balance_after:
            return error(402, "Insufficient Balance")
        thinking = body.get("thinking", {}).get("type") == "enabled"
        text_chars = 0
        image_tokens = 0
        for message in body["messages"]:
            content = message["content"]
            parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
            for part in parts:
                if part["type"] == "text":
                    text_chars += len(part["text"])
                    continue
                url = part["image_url"]["url"]
                mime = url.split(";", 1)[0].removeprefix("data:")
                if "detail" in part["image_url"] and self.detail_server_error is not None:
                    return error(self.detail_server_error, "temporary trouble")
                if (
                    (mime == "image/gif" and self.reject_gif)
                    or (mime == "image/jpeg" and self.reject_jpeg)
                    or ("detail" in part["image_url"] and self.reject_detail)
                ):
                    return error(400, "invalid image input")
                with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
                    image_tokens += self.image_tokens(*image.size)
        prompt = text_chars // 4 + 10 + image_tokens
        key = json.dumps(body["messages"], sort_keys=True)
        hit = prompt - 3 if (self.cache_works and key in self._seen) else 0
        self._seen.add(key)
        if "response_format" in body:
            if self.json_server_error is not None:
                return error(self.json_server_error, "temporary trouble")
            valid = self.json_valid_with_thinking if thinking else self.json_valid_without_thinking
            text = '{"city": "Paris", "population_millions": 2.1}' if valid else "Paris!"
        elif image_tokens:
            text = "A red circle on a blue background."
        else:
            text = "391"
        reasoning = None
        if thinking and self.reasoning_when_thinking:
            reasoning = "Work it out step by step."
        elif not thinking and self.reasoning_when_not_thinking:
            reasoning = "stray reasoning"
        return ok(
            content=text,
            reasoning=reasoning,
            prompt=prompt,
            hit=hit,
            completion_tokens=12,
            reasoning_tokens=8 if reasoning else 0,
        )
