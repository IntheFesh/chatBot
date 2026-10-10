"""What every owner of a style-model server does: files, clients, comparison, warm-up.

The running application (:mod:`twin.serving.component`), ``twin model serve`` and the evaluation
(:mod:`twin.serving.evaluation`) each own a server for a while; the steps that make a server
trustworthy are the same and live here, once:

* :func:`verify_model_file` - the registered file is there and has the registered sha256 (a model
  file that changed since it was registered is refused, R-SRV-001);
* :func:`server_spec_for` - the command line for a model (:mod:`twin.serving.llamacpp`);
* :func:`client_for` - the client that talks to an endpoint in the configured mode;
* :class:`TokenizerSource` and :func:`token_gate` - the comparison of R-TRN-011.4, run each time a
  server comes up, recorded in the registry (the selector refuses a model whose last comparison
  failed), with the alert ``style_tokenize_mismatch`` and a server that is stopped and stays
  stopped when the tokens differ;
* :func:`warm_up` - one short request to load the caches and measure the first-token latency and
  the generation speed, which ``/状态`` shows.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from twin.clock import Clock
from twin.config.settings import StyleModelConfig
from twin.llm.errors import StyleModelError
from twin.llm.style_client import (
    LlamaCppCompletionClient,
    RenderedPrompt,
    StyleModelClient,
    StyleParams,
    VllmCompletionClient,
)
from twin.ops.jobobject import ProcessJob
from twin.ops.logging import get_logger
from twin.serving.llamacpp import ServeError, ServerSpec, find_install, loopback_endpoint
from twin.serving.server import LlamaServerManager, ServerBlocked, ServerSnapshot, ServerTimings
from twin.serving.tokencheck import TokenizationReport, check_tokenization, tokenize_record
from twin.training import lf_template
from twin.training.lf_template import Turn
from twin.training.registry import (
    EVAL_TOKENIZE_CHECK,
    ModelView,
    record_eval,
    resolve_model_path,
    sha256_file,
)
from twin.training.tokenizer import QwenTokenizer, ensure_tokenizer
from twin.training.versions import LORA_NAME

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.serving.runtime")

WARMUP_SYSTEM: Final = "你是她。"
WARMUP_USER: Final = "在吗"
WARMUP_TOKENS: Final = 32
SHA_BLOCK: Final = 1 << 20


# --------------------------------------------------------------------- the files


def model_path(services: Services, model: ModelView) -> Path:
    """Where the file of a registered model is."""
    return resolve_model_path(services.paths.models_dir, model)


def verify_model_file(model: ModelView, path: Path, *, full: bool = True) -> None:
    """The file exists and is the one that was registered (size first, then sha256).

    ``full=False`` stops after the size: hashing a 9 GB file takes a while, and the running
    application checks it on every start (``twin model serve`` and the evaluation hash it).
    """
    if not path.is_file():
        raise ServeError(f"the model file {path} does not exist (registered as {model.id})")
    if path.stat().st_size != model.size:
        raise ServeError(f"{path.name} has another size than registered; it changed or is damaged")
    if full and sha256_file(path) != model.sha256:
        raise ServeError(f"{path.name} does not match its registered sha256; it was changed")


@dataclass(frozen=True)
class LocalProgram:
    """The program that serves a GGUF: its command (without arguments) and where it runs."""

    prefix: tuple[str, ...]
    cwd: Path | None = None
    label: str = ""


def resolve_program(services: Services) -> LocalProgram:
    """The installed ``llama-server`` (``ServeError`` says what to do when there is none)."""
    install = find_install(services.paths.root, services.settings.style_model.serve.binary)
    if install is None:
        raise ServeError(
            "llama.cpp is not installed: run scripts\\windows\\get_llamacpp.ps1 "
            "(or set style_model.serve.binary)"
        )
    if not install.complete:
        raise ServeError(
            f"llama.cpp in {install.binary.parent} is incomplete: {', '.join(install.missing)}; "
            "run scripts\\windows\\get_llamacpp.ps1 -Force"
        )
    return LocalProgram(
        (str(install.binary),), install.binary.parent, f"{install.version} {install.kind}"
    )


def server_spec_for(
    services: Services, model: ModelView, program: LocalProgram, *, port: int
) -> ServerSpec:
    """The command line of ``llama-server`` for ``model`` on ``port`` (127.0.0.1 only)."""
    serve = services.settings.style_model.serve
    return ServerSpec(
        prefix=program.prefix,
        model=model_path(services, model),
        port=port,
        context=serve.context,
        gpu_layers=serve.gpu_layers,
        parallel=serve.parallel,
    )


def primary_port(config: StyleModelConfig) -> int:
    """The port of ``style_model.endpoint`` (refused when it is not on this computer)."""
    return loopback_endpoint(config.endpoint)[1]


def client_for(
    config: StyleModelConfig, clock: Clock, *, endpoint: str | None = None
) -> StyleModelClient:
    """The client of the configured mode; ``endpoint`` replaces the configured one (evaluation).

    ``vllm_completion`` always talks to the local end of the tunnel and names the LoRA adapter
    (``style_model.model_id``, ``twin-style`` for the adapter ``serve_vllm.sh`` serves).
    """
    if config.mode == "llamacpp_completion":
        return LlamaCppCompletionClient(endpoint or config.endpoint, clock=clock)
    if not config.model_id:
        raise ServeError(
            f"style_model.model_id (the LoRA name) must be set for vllm_completion: {LORA_NAME}"
        )
    return VllmCompletionClient(
        endpoint or f"http://127.0.0.1:{config.tunnel.local_port}",
        model=config.model_id,
        clock=clock,
    )


# ---------------------------------------------------------------- the comparison


class TokenizerSource:
    """The pinned tokenizer, loaded once (and downloaded once, verified) on first use."""

    def __init__(
        self, services: Services, loader: Callable[[], QwenTokenizer] | None = None
    ) -> None:
        self._services = services
        self._loader = loader
        self._tokenizer: QwenTokenizer | None = None

    def _load(self) -> QwenTokenizer:
        if self._loader is not None:
            return self._loader()
        return ensure_tokenizer(self._services.paths.data_dir / "training" / "tokenizer")

    async def get(self) -> QwenTokenizer:
        if self._tokenizer is None:
            self._tokenizer = await asyncio.to_thread(self._load)
        return self._tokenizer


async def compare_tokens(
    services: Services,
    model: ModelView,
    client: StyleModelClient,
    tokenizers: TokenizerSource,
    *,
    bump_state: bool | None = False,
) -> TokenizationReport:
    """Compare the server's tokens with the tokenizer's and record the verdict on the model.

    The application records it without waking itself (``bump_state=False``); a command that
    wants the application to look again passes ``None`` (see :func:`record_eval`).
    """
    tokenizer = await tokenizers.get()
    report = await check_tokenization(client, tokenizer)
    now = services.clock.now_utc().isoformat()
    await asyncio.to_thread(
        record_eval,
        services.db,
        model.id,
        {EVAL_TOKENIZE_CHECK: tokenize_record(report, model, at=now)},
        bump_state=bump_state,
    )
    if not report.ok:
        services.alerts.raise_alert(
            "style_tokenize_mismatch",
            f"the tokens of {model.id} differ from the training tokenizer",
            severity="warning",
            detail={"model": model.id, "differences": len(report.differences)},
            dedup_key=f"style_tokenize_mismatch:{model.id}",
        )
        log.warning("tokenize_mismatch", model=model.id, differences=len(report.differences))
    return report


def token_gate(
    services: Services,
    model: ModelView,
    client: StyleModelClient,
    tokenizers: TokenizerSource,
) -> Callable[[], Awaitable[None]]:
    """The ``on_ready`` hook of a server: a model whose tokens differ is not served."""

    async def gate() -> None:
        # a server that cannot tokenize raises TokenizeCheckError: no verdict on the model, the
        # supervisor starts the server again; a verdict that says "different" blocks it
        report = await compare_tokens(services, model, client, tokenizers)
        if not report.ok:
            raise ServerBlocked("tokenizer", report.lines()[1])

    return gate


# ----------------------------------------------------------------------- warm-up


@dataclass(frozen=True)
class Warmup:
    """What the warm-up request measured."""

    first_token_ms: int
    tokens_per_s: float | None
    prompt_tokens: int | None
    at: str
    measured_by: str  # "server" (llama.cpp's own timings) or "client" (wall clock)

    def to_json(self) -> dict[str, object]:
        return {
            "first_token_ms": self.first_token_ms,
            "tokens_per_s": self.tokens_per_s,
            "prompt_tokens": self.prompt_tokens,
            "at": self.at,
            "measured_by": self.measured_by,
        }

    def line(self) -> str:
        speed = f"{self.tokens_per_s:.1f}" if self.tokens_per_s is not None else "?"
        return f"first token {self.first_token_ms} ms, {speed} tokens/s ({self.measured_by})"


def warmup_prompt() -> RenderedPrompt:
    """A short ChatML prompt rendered with the training template."""
    return RenderedPrompt(lf_template.render_prompt(WARMUP_SYSTEM, [Turn("user", WARMUP_USER)]))


async def warm_up(client: StyleModelClient, clock: Clock) -> Warmup | None:
    """One-token and 32-token requests; ``None`` when the server does not answer them."""
    prompt = warmup_prompt()
    try:
        first = await client.generate(prompt, StyleParams(n_predict=1, temperature=0.0))
        longer = await client.generate(
            prompt, StyleParams(n_predict=WARMUP_TOKENS, temperature=0.0)
        )
    except StyleModelError as exc:
        log.warning("warmup_failed", reason=exc.kind or type(exc).__name__)
        return None
    at = clock.now_utc().isoformat()
    timings = first.timings or {}
    from_server = "prompt_ms" in timings
    if from_server:
        first_ms = round(timings["prompt_ms"] + timings.get("predicted_ms", 0.0))
    else:
        first_ms = first.latency_ms
    speed = (longer.timings or {}).get("predicted_per_second")
    if speed is None:
        from_server = False
        if longer.completion_tokens and longer.completion_tokens > 1:
            spent = longer.latency_ms - first.latency_ms
            if spent > 0:
                speed = (longer.completion_tokens - 1) * 1000.0 / spent
    return Warmup(
        first_ms,
        round(speed, 1) if speed is not None else None,
        first.prompt_tokens,
        at,
        "server" if from_server else "client",
    )


def parse_size(text: str) -> int | None:
    """``"4096 MiB"`` or ``"4096"`` -> bytes (used by ``twin model recommend --vram``)."""
    found = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(GiB|GB|MiB|MB)?\s*", text)
    if not found:
        return None
    number = float(found.group(1))
    unit = found.group(2) or "MiB"
    factor = {"GiB": 1024**3, "GB": 10**9, "MiB": 1024**2, "MB": 10**6}[unit]
    return round(number * factor)


# ------------------------------------------------------------------ a local server


@dataclass
class LocalServer:
    """A managed ``llama-server`` with the client that talks to it and its last warm-up."""

    manager: LlamaServerManager
    client: StyleModelClient
    endpoint: str
    warmup: Warmup | None = None


def default_timings(services: Services) -> ServerTimings:
    """The waits of the supervision loop, from ``style_model.serve`` and ``backend``."""
    serve = services.settings.style_model.serve
    return ServerTimings(
        start_timeout_s=serve.start_timeout_s,
        backoff_start_s=serve.backoff_start_s,
        backoff_max_s=serve.backoff_max_s,
        stable_after_s=serve.stable_after_s,
        health_interval_s=services.settings.backend.health_check_s,
    )


def build_local_server(
    services: Services,
    model: ModelView,
    program: LocalProgram,
    *,
    port: int,
    tokenizers: TokenizerSource,
    job: ProcessJob | None = None,
    timings: ServerTimings | None = None,
    on_change: Callable[[ServerSnapshot], None] | None = None,
    log_name: str = "llama-server.log",
    warm: bool = True,
    gate: bool = True,
) -> LocalServer:
    """The manager of the server of ``model`` on ``port``: not started yet.

    When the process has loaded the model the comparison with the training tokenizer runs (a
    model whose tokens differ is stopped and kept stopped; ``gate=False`` leaves that to the
    caller, who wants the whole report) and, with ``warm``, one warm-up.
    """
    endpoint = f"http://127.0.0.1:{port}"
    config = services.settings.style_model
    client = client_for(config, services.clock, endpoint=endpoint)
    check = token_gate(services, model, client, tokenizers)
    holder: list[LocalServer] = []

    async def ready() -> None:
        if gate:
            await check()
        if warm:
            holder[0].warmup = await warm_up(client, services.clock)

    manager = LlamaServerManager(
        server_spec_for(services, model, program, port=port),
        clock=services.clock,
        client=client,
        log_path=services.paths.logs_dir / log_name,
        timings=timings or default_timings(services),
        alerts=services.alerts,
        job=job,
        on_ready=ready,
        on_change=on_change,
        cwd=program.cwd,
    )
    server = LocalServer(manager, client, endpoint)
    holder.append(server)
    return server
