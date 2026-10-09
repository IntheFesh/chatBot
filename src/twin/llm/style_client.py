"""Clients for the fine-tuned style model (R-LLM-011).

Both clients send **only the prompt string** that ``StylePromptBuilder`` (round 09) rendered:
the training template (ChatML, ``qwen3_nothink``) is part of that string, so the server must not
apply a chat template of its own.  Hence the raw completion endpoints and never a chat endpoint:

* ``llamacpp_completion`` - llama.cpp ``llama-server`` ``POST /completion`` (``prompt``,
  ``n_predict``, ``temperature``, ``top_p``, ``stop``; reply ``content``, ``stop_type``,
  ``tokens_evaluated``, ``tokens_predicted``, ``truncated``), ``GET /health`` (200 with
  ``{"status": "ok"}``, 503 while the model loads) and ``POST /tokenize`` (``content``,
  ``add_special``, ``parse_special``; reply ``tokens``);
* ``vllm_completion`` - vLLM ``POST /v1/completions`` (``model`` is the LoRA name, ``prompt`` the
  rendered string, ``max_tokens``, ``stop``, ``add_special_tokens``), ``GET /health`` and
  ``POST /tokenize`` (``prompt``, ``add_special_tokens``; reply ``tokens`` and ``count``).

Endpoint and field names were checked against the llama.cpp server README (master) and the
vLLM online serving documentation and protocol source on 2026-10-09.

Sampling is neutral and identical on both servers: temperature and top-p come from
:class:`StyleParams`, while top-k, min-p and repetition penalties are switched off explicitly
(llama.cpp would otherwise apply its own defaults of top-k 40, min-p 0.05 and repeat penalty
1.1), so the model samples from the distribution it was trained on.  ``<|im_end|>`` is always a
stop string.  Failures raise :class:`~twin.llm.errors.StyleModelError`; there is no retry here,
the engine decides whether to fall back to DeepSeek.

The vLLM client talks to AutoDL through an SSH tunnel and therefore redacts the prompt first
(CLAUDE.md rule 6); the llama.cpp client talks to a process on this computer and does not.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx

from twin.clock import Clock
from twin.config.settings import StyleModelConfig
from twin.llm.errors import StyleModelError
from twin.llm.redaction import redact_text

IM_END = "<|im_end|>"
DEFAULT_TIMEOUT_S = 60.0
HEALTH_TIMEOUT_S = 5.0
TOKENIZE_TIMEOUT_S = 30.0

StopReason = Literal["stop", "length"]


@dataclass(frozen=True)
class RenderedPrompt:
    """A prompt rendered by ``StylePromptBuilder``; the string is sent exactly as it is."""

    text: str
    template: str = "qwen3_nothink"

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("a rendered prompt must not be empty")


@dataclass(frozen=True)
class StyleParams:
    """Sampling settings for one generation."""

    n_predict: int = 200
    temperature: float = 0.7
    top_p: float = 0.9
    stop: tuple[str, ...] = (IM_END,)
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_predict < 1:
            raise ValueError("n_predict must be at least 1")
        if self.temperature < 0 or not 0 < self.top_p <= 1:
            raise ValueError("temperature must be >= 0 and top_p in (0, 1]")

    def stop_strings(self) -> list[str]:
        """The stop strings, always including ``<|im_end|>``."""
        return list(self.stop) if IM_END in self.stop else [*self.stop, IM_END]


@dataclass(frozen=True)
class StyleOutput:
    text: str
    stop_reason: StopReason
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int

    @property
    def truncated(self) -> bool:
        """True if generation stopped because ``n_predict`` ran out, not at a stop string."""
        return self.stop_reason == "length"


@dataclass(frozen=True)
class StyleHealth:
    ok: bool
    detail: str
    latency_ms: int = 0


class StyleModelClient(Protocol):
    """What the engine and the deployment code need from a style model server."""

    async def generate(self, prompt: RenderedPrompt, params: StyleParams) -> StyleOutput: ...

    async def health(self) -> StyleHealth: ...

    async def tokenize(self, text: str) -> list[int]: ...

    async def aclose(self) -> None: ...


class _HttpStyleClient:
    """Shared HTTP plumbing."""

    def __init__(
        self,
        endpoint: str,
        *,
        clock: Clock,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._clock = clock
        self._timeout_s = timeout_s
        self._client = client
        self._owns_client = client is None
        self._headers = headers or {}

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_s, headers=self._headers)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def _call(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        expect_json: bool = True,
    ) -> Any:
        url = f"{self._endpoint}{path}"
        try:
            response = await self._http().request(
                method, url, json=payload, timeout=timeout_s or self._timeout_s
            )
        except httpx.TimeoutException as exc:
            raise StyleModelError(f"style model {path} timed out", kind="timeout") from exc
        except httpx.HTTPError as exc:
            raise StyleModelError(
                f"style model {path} is unreachable: {type(exc).__name__}", kind="unavailable"
            ) from exc
        if response.status_code >= 400:
            raise StyleModelError(
                f"style model {path} answered {response.status_code}",
                status=response.status_code,
                kind="status",
            )
        if not expect_json:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise StyleModelError(
                f"style model {path} did not return JSON", kind="malformed"
            ) from exc

    def _elapsed_ms(self, started: float) -> int:
        return max(0, round((self._clock.monotonic() - started) * 1000))


def _field(data: Any, name: str, kind: type) -> Any:
    if not isinstance(data, dict) or not isinstance(data.get(name), kind):
        raise StyleModelError(f"style model reply lacks a valid {name!r}", kind="malformed")
    return data[name]


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


class LlamaCppCompletionClient(_HttpStyleClient):
    """llama.cpp ``llama-server`` on this computer, through ``/completion``."""

    async def generate(self, prompt: RenderedPrompt, params: StyleParams) -> StyleOutput:
        payload: dict[str, Any] = {
            "prompt": prompt.text,
            "n_predict": params.n_predict,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": 0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "repeat_last_n": 0,
            "stop": params.stop_strings(),
            "stream": False,
            "cache_prompt": True,
        }
        if params.seed is not None:
            payload["seed"] = params.seed
        started = self._clock.monotonic()
        data = await self._call("POST", "/completion", payload=payload)
        text = _field(data, "content", str)
        stop_type = data.get("stop_type")
        reason: StopReason = "length" if stop_type == "limit" else "stop"
        return StyleOutput(
            text=text,
            stop_reason=reason,
            prompt_tokens=_optional_int(data.get("tokens_evaluated")),
            completion_tokens=_optional_int(data.get("tokens_predicted")),
            latency_ms=self._elapsed_ms(started),
        )

    async def health(self) -> StyleHealth:
        started = self._clock.monotonic()
        try:
            data = await self._call("GET", "/health", timeout_s=HEALTH_TIMEOUT_S)
        except StyleModelError as exc:
            detail = "model is still loading" if exc.status == 503 else str(exc)
            return StyleHealth(False, detail, self._elapsed_ms(started))
        if isinstance(data, dict) and data.get("status") == "ok":
            return StyleHealth(True, "ok", self._elapsed_ms(started))
        return StyleHealth(False, "unexpected health reply", self._elapsed_ms(started))

    async def tokenize(self, text: str) -> list[int]:
        """Token ids exactly as the server sees ``text`` (special tokens parsed, no BOS added)."""
        data = await self._call(
            "POST",
            "/tokenize",
            payload={"content": text, "add_special": False, "parse_special": True},
            timeout_s=TOKENIZE_TIMEOUT_S,
        )
        return _token_ids(_field(data, "tokens", list))


class VllmCompletionClient(_HttpStyleClient):
    """vLLM on the AutoDL instance (through the SSH tunnel), through ``/v1/completions``."""

    def __init__(
        self,
        endpoint: str,
        *,
        model: str,
        clock: Clock,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
        api_key: str | None = None,
        outbound: Callable[[str], str] | None = redact_text,
    ) -> None:
        if not model:
            raise ValueError("the vLLM client needs the LoRA model name")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        super().__init__(endpoint, clock=clock, timeout_s=timeout_s, client=client, headers=headers)
        self._model = model
        self._outbound = outbound

    async def generate(self, prompt: RenderedPrompt, params: StyleParams) -> StyleOutput:
        text = self._outbound(prompt.text) if self._outbound else prompt.text
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": text,
            "max_tokens": params.n_predict,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "stop": params.stop_strings(),
            "add_special_tokens": False,
            "stream": False,
        }
        if params.seed is not None:
            payload["seed"] = params.seed
        started = self._clock.monotonic()
        data = await self._call("POST", "/v1/completions", payload=payload)
        choices = _field(data, "choices", list)
        if (
            not choices
            or not isinstance(choices[0], dict)
            or not isinstance(choices[0].get("text"), str)
        ):
            raise StyleModelError("style model reply has no completion text", kind="malformed")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        reason: StopReason = "length" if choices[0].get("finish_reason") == "length" else "stop"
        return StyleOutput(
            text=choices[0]["text"],
            stop_reason=reason,
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_int(usage.get("completion_tokens")),
            latency_ms=self._elapsed_ms(started),
        )

    async def health(self) -> StyleHealth:
        started = self._clock.monotonic()
        try:
            await self._call("GET", "/health", timeout_s=HEALTH_TIMEOUT_S, expect_json=False)
            models = await self._call("GET", "/v1/models", timeout_s=HEALTH_TIMEOUT_S)
        except StyleModelError as exc:
            return StyleHealth(False, str(exc), self._elapsed_ms(started))
        listed = models.get("data") if isinstance(models, dict) else None
        names = (
            [m.get("id") for m in listed if isinstance(m, dict)] if isinstance(listed, list) else []
        )
        if self._model not in names:
            return StyleHealth(
                False, f"model {self._model!r} is not served", self._elapsed_ms(started)
            )
        return StyleHealth(True, "ok", self._elapsed_ms(started))

    async def tokenize(self, text: str) -> list[int]:
        """Token ids exactly as the server sees ``text`` (no BOS added)."""
        data = await self._call(
            "POST",
            "/tokenize",
            payload={"model": self._model, "prompt": text, "add_special_tokens": False},
            timeout_s=TOKENIZE_TIMEOUT_S,
        )
        return _token_ids(_field(data, "tokens", list))


def _token_ids(raw: list[Any]) -> list[int]:
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in raw):
        raise StyleModelError(
            "style model returned token ids that are not integers", kind="malformed"
        )
    return [int(item) for item in raw]


def style_client_from_config(
    config: StyleModelConfig,
    clock: Clock,
    *,
    api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> StyleModelClient:
    """Build the client the configuration asks for.

    In ``vllm_completion`` mode the endpoint is the local end of the SSH tunnel to AutoDL
    (``style_model.tunnel.local_port``); ``style_model.model_id`` is the LoRA name.
    """
    if config.mode == "llamacpp_completion":
        return LlamaCppCompletionClient(config.endpoint, clock=clock, client=client)
    if not config.model_id:
        raise ValueError("style_model.model_id (the LoRA name) must be set for vllm_completion")
    return VllmCompletionClient(
        f"http://127.0.0.1:{config.tunnel.local_port}",
        model=config.model_id,
        clock=clock,
        api_key=api_key,
        client=client,
    )
