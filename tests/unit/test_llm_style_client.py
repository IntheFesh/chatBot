"""The style model clients against a local server with real server answers (R-LLM-011)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests.support.clock import ManualClock
from tests.support.style_server import (
    StyleServer,
    llama_defaults,
    running_server,
    vllm_defaults,
)
from tests.support.synthetic import mobile
from twin.config.loader import load_settings
from twin.llm.errors import StyleModelError
from twin.llm.style_client import (
    IM_END,
    LlamaCppCompletionClient,
    RenderedPrompt,
    StyleParams,
    VllmCompletionClient,
    style_client_from_config,
)

PROMPT = RenderedPrompt(
    "<|im_start|>system\n你是她。<|im_end|>\n<|im_start|>user\n在吗<|im_end|>\n<|im_start|>assistant\n"
)


@pytest.fixture
def server() -> Iterator[StyleServer]:
    with running_server() as running:
        yield running


@pytest.fixture
def llama(server: StyleServer, clock: ManualClock) -> LlamaCppCompletionClient:
    llama_defaults(server)
    return LlamaCppCompletionClient(server.url, clock=clock, timeout_s=5)


@pytest.fixture
def vllm(server: StyleServer, clock: ManualClock) -> VllmCompletionClient:
    vllm_defaults(server)
    return VllmCompletionClient(server.url, model="lora-a", clock=clock, timeout_s=5)


# ------------------------------------------------------------------ data classes


def test_prompts_and_parameters_are_validated() -> None:
    with pytest.raises(ValueError, match="empty"):
        RenderedPrompt("")
    assert PROMPT.template == "qwen3_nothink"
    with pytest.raises(ValueError, match="n_predict"):
        StyleParams(n_predict=0)
    with pytest.raises(ValueError, match="top_p"):
        StyleParams(top_p=0)
    with pytest.raises(ValueError, match="temperature"):
        StyleParams(temperature=-1)


def test_im_end_is_always_a_stop_string() -> None:
    assert StyleParams().stop_strings() == [IM_END]
    assert StyleParams(stop=("\n\n",)).stop_strings() == ["\n\n", IM_END]
    assert StyleParams(stop=(IM_END, "x")).stop_strings() == [IM_END, "x"]


# --------------------------------------------------------------------- llama.cpp


async def test_llama_sends_the_rendered_string_to_the_completion_endpoint(
    server: StyleServer, llama: LlamaCppCompletionClient
) -> None:
    output = await llama.generate(PROMPT, StyleParams(n_predict=64, temperature=0.6, top_p=0.8))
    assert server.paths() == ["/completion"]  # never a chat endpoint
    sent = server.last("/completion").json
    assert sent["prompt"] == PROMPT.text  # byte for byte, special tokens included
    assert (sent["n_predict"], sent["temperature"], sent["top_p"]) == (64, 0.6, 0.8)
    assert sent["stop"] == [IM_END] and sent["stream"] is False
    assert (sent["top_k"], sent["min_p"], sent["repeat_penalty"]) == (0, 0.0, 1.0)
    assert "seed" not in sent and "messages" not in sent
    assert output.text == "好呀" and output.stop_reason == "stop" and not output.truncated
    assert (output.prompt_tokens, output.completion_tokens) == (42, 3)
    assert output.latency_ms >= 0


async def test_llama_reports_truncation_and_passes_the_seed(
    server: StyleServer, llama: LlamaCppCompletionClient
) -> None:
    server.set(
        "POST",
        "/completion",
        body={
            "content": "好呀好呀",
            "stop_type": "limit",
            "tokens_evaluated": 5,
            "tokens_predicted": 9,
        },
    )
    output = await llama.generate(PROMPT, StyleParams(seed=7, stop=("。",)))
    assert output.truncated and output.stop_reason == "length"
    sent = server.last("/completion").json
    assert sent["seed"] == 7 and sent["stop"] == ["。", IM_END]


async def test_llama_health_distinguishes_ready_loading_and_down(
    server: StyleServer, llama: LlamaCppCompletionClient
) -> None:
    ready = await llama.health()
    assert ready.ok and ready.detail == "ok"
    server.set(
        "GET",
        "/health",
        status=503,
        body={"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}},
    )
    loading = await llama.health()
    assert not loading.ok and "loading" in loading.detail
    assert loading.loading  # the engine does not treat this as a failure (round 14)
    server.set("GET", "/health", body={"status": "no slot available"})
    assert not (await llama.health()).ok
    server.set("GET", "/health", status=500, body={})
    failing = await llama.health()
    assert not failing.ok and "500" in failing.detail and not failing.loading


async def test_llama_health_never_raises_when_the_server_is_gone(clock: ManualClock) -> None:
    with running_server() as gone:
        url = gone.url
    client = LlamaCppCompletionClient(url, clock=clock, timeout_s=2)
    health = await client.health()
    assert not health.ok and "unreachable" in health.detail and not health.loading
    await client.aclose()


async def test_llama_reports_the_timings_of_the_server_for_the_warm_up(
    server: StyleServer, llama: LlamaCppCompletionClient
) -> None:
    server.set(
        "POST",
        "/completion",
        body={
            "content": "好呀",
            "stop_type": "eos",
            "tokens_evaluated": 5,
            "tokens_predicted": 9,
            "timings": {
                "prompt_n": 5,
                "prompt_ms": 31.5,
                "predicted_n": 9,
                "predicted_ms": 120,
                "predicted_per_second": 75.0,
                "cache_n": True,  # a flag is not a number
                "other": "x",
            },
        },
    )
    output = await llama.generate(PROMPT, StyleParams())
    assert output.timings == {
        "prompt_n": 5.0,
        "prompt_ms": 31.5,
        "predicted_n": 9.0,
        "predicted_ms": 120.0,
        "predicted_per_second": 75.0,
    }
    server.set("POST", "/completion", body={"content": "好", "timings": "broken"})
    assert (await llama.generate(PROMPT, StyleParams())).timings is None
    server.set("POST", "/completion", body={"content": "好", "timings": {"x": 1}})
    assert (await llama.generate(PROMPT, StyleParams())).timings is None


async def test_llama_tokenize_asks_the_way_completion_tokenizes_a_string_prompt(
    server: StyleServer, llama: LlamaCppCompletionClient
) -> None:
    # /completion tokenizes a string prompt with add_special=true and parse_special=true, so a
    # GGUF that asks for a BOS gets one there; the comparison has to see exactly that (round 14)
    tokens = await llama.tokenize("<|im_start|>你")
    sent = server.last("/tokenize").json
    assert sent == {"content": "<|im_start|>你", "add_special": True, "parse_special": True}
    assert tokens == [ord(c) for c in "<|im_start|>你"]


async def test_llama_errors_are_style_model_errors(
    server: StyleServer, llama: LlamaCppCompletionClient, clock: ManualClock
) -> None:
    server.set("POST", "/completion", status=500, body={"error": "boom"})
    with pytest.raises(StyleModelError) as server_error:
        await llama.generate(PROMPT, StyleParams())
    assert server_error.value.status == 500 and server_error.value.kind == "status"

    server.set("POST", "/completion", status=503, body={"error": "Loading model"})
    with pytest.raises(StyleModelError) as loading:
        await llama.generate(PROMPT, StyleParams())
    assert loading.value.status == 503

    server.set("POST", "/completion", body=b"<html>not json</html>")
    with pytest.raises(StyleModelError) as not_json:
        await llama.generate(PROMPT, StyleParams())
    assert not_json.value.kind == "malformed"

    server.set("POST", "/completion", body={"stop_type": "word"})
    with pytest.raises(StyleModelError, match="content"):
        await llama.generate(PROMPT, StyleParams())

    server.set("POST", "/tokenize", body={"tokens": ["a", "b"]})
    with pytest.raises(StyleModelError, match="integers"):
        await llama.tokenize("ab")
    server.set("POST", "/tokenize", body={"tokens": [True]})
    with pytest.raises(StyleModelError, match="integers"):
        await llama.tokenize("ab")


async def test_llama_timeouts_are_reported_as_timeouts(
    server: StyleServer, clock: ManualClock
) -> None:
    llama_defaults(server)
    server.set("POST", "/completion", body={"content": "x"}, delay_s=0.8)
    slow = LlamaCppCompletionClient(server.url, clock=clock, timeout_s=0.2)
    with pytest.raises(StyleModelError) as info:
        await slow.generate(PROMPT, StyleParams())
    assert info.value.kind == "timeout"
    await slow.aclose()


async def test_llama_does_not_redact_because_the_server_is_local(
    server: StyleServer, llama: LlamaCppCompletionClient
) -> None:
    text = f"<|im_start|>user\n我的号码 {mobile()}<|im_end|>\n<|im_start|>assistant\n"
    await llama.generate(RenderedPrompt(text), StyleParams())
    assert server.last("/completion").json["prompt"] == text


# -------------------------------------------------------------------------- vLLM


async def test_vllm_uses_the_completions_endpoint_with_the_lora_name(
    server: StyleServer, vllm: VllmCompletionClient
) -> None:
    output = await vllm.generate(
        PROMPT, StyleParams(n_predict=50, temperature=0.5, top_p=0.7, seed=3)
    )
    assert server.paths() == ["/v1/completions"]  # no chat endpoint, no server-side template
    sent = server.last("/v1/completions").json
    assert sent["model"] == "lora-a" and sent["prompt"] == PROMPT.text
    assert (sent["max_tokens"], sent["temperature"], sent["top_p"]) == (50, 0.5, 0.7)
    assert sent["stop"] == [IM_END] and sent["add_special_tokens"] is False
    assert (sent["top_k"], sent["min_p"], sent["repetition_penalty"]) == (-1, 0.0, 1.0)
    assert sent["seed"] == 3 and "messages" not in sent
    assert output.text == "好呀" and not output.truncated
    assert (output.prompt_tokens, output.completion_tokens) == (40, 4)


async def test_vllm_marks_length_stops_as_truncated(
    server: StyleServer, vllm: VllmCompletionClient
) -> None:
    server.set(
        "POST",
        "/v1/completions",
        body={"choices": [{"text": "好", "finish_reason": "length"}]},
    )
    output = await vllm.generate(PROMPT, StyleParams())
    assert output.truncated and output.prompt_tokens is None


async def test_vllm_redacts_the_prompt_because_it_leaves_the_machine(
    server: StyleServer, vllm: VllmCompletionClient
) -> None:
    text = f"<|im_start|>user\n我的号码 {mobile()}<|im_end|>\n<|im_start|>assistant\n"
    await vllm.generate(RenderedPrompt(text), StyleParams())
    sent = server.last("/v1/completions").json["prompt"]
    assert mobile() not in sent and "[手机号]" in sent
    assert sent.startswith("<|im_start|>user\n") and sent.endswith("<|im_start|>assistant\n")


async def test_vllm_health_needs_the_server_and_the_model(
    server: StyleServer, vllm: VllmCompletionClient
) -> None:
    assert (await vllm.health()).ok
    server.set("GET", "/v1/models", body={"data": [{"id": "base"}]})
    missing = await vllm.health()
    assert not missing.ok and "lora-a" in missing.detail
    server.set("GET", "/v1/models", body={"data": "nope"})
    assert not (await vllm.health()).ok
    server.set("GET", "/health", status=503, body=b"")
    down = await vllm.health()
    assert not down.ok and "503" in down.detail


async def test_vllm_tokenize_uses_the_documented_fields(
    server: StyleServer, vllm: VllmCompletionClient
) -> None:
    tokens = await vllm.tokenize("<|im_start|>你")
    sent = server.last("/tokenize").json
    assert sent == {"model": "lora-a", "prompt": "<|im_start|>你", "add_special_tokens": False}
    assert tokens == [ord(c) for c in "<|im_start|>你"]
    server.set("POST", "/tokenize", body={"count": 1})
    with pytest.raises(StyleModelError, match="tokens"):
        await vllm.tokenize("x")


async def test_vllm_tokenize_sends_the_same_redacted_text_as_a_prompt(
    server: StyleServer, vllm: VllmCompletionClient
) -> None:
    """What is compared is what generate() would send: the instance never sees a phone number."""
    text = f"<|im_start|>user\n我的号码 {mobile()}<|im_end|>\n<|im_start|>assistant\n"
    await vllm.tokenize(text)
    sent = server.last("/tokenize").json["prompt"]
    assert mobile() not in sent and "[手机号]" in sent


async def test_vllm_errors_and_authentication(server: StyleServer, clock: ManualClock) -> None:
    vllm_defaults(server)
    client = VllmCompletionClient(
        server.url, model="lora-a", clock=clock, timeout_s=5, api_key="tunnel-token"
    )
    await client.generate(PROMPT, StyleParams())
    assert server.last("/v1/completions").headers["authorization"] == "Bearer tunnel-token"
    server.set("POST", "/v1/completions", status=500, body={"error": "x"})
    with pytest.raises(StyleModelError) as failed:
        await client.generate(PROMPT, StyleParams())
    assert failed.value.status == 500
    for bad in ({"choices": []}, {"choices": [{"text": 5}]}, {"nothing": 1}):
        server.set("POST", "/v1/completions", body=bad)
        with pytest.raises(StyleModelError):
            await client.generate(PROMPT, StyleParams())
    server.set("POST", "/v1/completions", body={"choices": [{"text": "x"}]}, delay_s=0.8)
    slow = VllmCompletionClient(server.url, model="lora-a", clock=clock, timeout_s=0.2)
    with pytest.raises(StyleModelError) as timed_out:
        await slow.generate(PROMPT, StyleParams())
    assert timed_out.value.kind == "timeout"
    await client.aclose()
    await slow.aclose()


def test_the_vllm_client_needs_a_model_name(clock: ManualClock) -> None:
    with pytest.raises(ValueError, match="LoRA"):
        VllmCompletionClient("http://127.0.0.1:1", model="", clock=clock)


# ------------------------------------------------------------------------ factory


def test_the_factory_follows_the_configuration(clock: ManualClock) -> None:
    local = style_client_from_config(load_settings().style_model, clock)
    assert isinstance(local, LlamaCppCompletionClient)
    settings = load_settings(
        None,
        {
            "style_model": {
                "mode": "vllm_completion",
                "model_id": "lora-a",
                "tunnel": {"local_port": 9123, "remote_port": 8000},
            }
        },
    )
    remote = style_client_from_config(settings.style_model, clock)
    assert isinstance(remote, VllmCompletionClient)
    assert remote._endpoint == "http://127.0.0.1:9123"  # the local end of the tunnel
    broken = load_settings(None, {"style_model": {"mode": "vllm_completion"}})
    with pytest.raises(ValueError, match="model_id"):
        style_client_from_config(broken.style_model, clock)
