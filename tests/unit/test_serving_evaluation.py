"""``twin model evaluate``: the plan, the model's own server, the pairs it writes (R-SRV-005, D).

The model is registered but **not active**; its server is the simulated llama-server (a real child
process on the evaluation port) or, for an adapter, a vLLM behind a local SSH server.  DeepSeek is
the scripted one of the evaluation tests.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import respx

from tests.support.eval_world import DeepSeekScript
from tests.support.export_world import LIVE_STYLE, PAST_MARKER, World
from tests.support.llama_sim import free_port, sim_prefix
from tests.support.serving_world import ServingWorld, build_serving_world
from tests.support.ssh_server import PASSWORD, USER, LocalSSHServer
from tests.support.style_server import running_server, vllm_defaults
from tests.support.tiny_tokenizer import tiny_qwen_tokenizer
from tests.support.waiting import wait_until
from twin.config.runtime import TUNNEL_WANTED
from twin.eval.blind import EVAL_GENERATE_JOB, generate_items
from twin.eval.store import EvalStore
from twin.llm.runtime import build_llm_runtime
from twin.llm.style_client import LlamaCppCompletionClient, VllmCompletionClient
from twin.serving.evaluation import (
    EvaluationError,
    EvaluationServers,
    evaluation_style,
    install_servers,
    installed_servers,
    plan_model_evaluation,
    preflight,
    tunnel_known_hosts,
)
from twin.serving.gate_m5 import MODEL_PARAM
from twin.serving.server import ServerTimings
from twin.serving.tunnel import TunnelManager, TunnelState, TunnelTimings
from twin.storage.models import Job
from twin.training.registry import get_model
from twin.training.remote.connection import RemoteTarget, remember_host

PLAN = (
    '{"reply": true, "intent": "答应", "facts_to_use": [], "tone": "轻松", '
    '"bubble_hint": "两条", "sticker_hint": ""}'
)


def no_file_check(model: object, path: Path) -> None:
    """The registered sha256 of the dummy model file is made up: only its existence counts."""
    if not path.is_file():
        raise AssertionError(f"{path} does not exist")


@pytest.fixture(autouse=True)
def real_local_servers(api: respx.MockRouter) -> None:
    """The scripted DeepSeek intercepts HTTP; the servers on 127.0.0.1 are real and get through."""
    api.route(host="127.0.0.1").pass_through()


@pytest.fixture
def candidate(world: World, tmp_path: Path) -> ServingWorld:
    """A registered model that is not active, with the simulated server as its program."""
    return build_serving_world(
        world.services, tmp_path, active=False, gate_passed=None, backend="deepseek"
    )


def pool_for(candidate: ServingWorld, *options: str, **kwargs: object) -> EvaluationServers:
    return EvaluationServers(
        candidate.services,
        tokenizers=candidate.tokenizers(),
        program=candidate.program(*options),
        check_file=no_file_check,
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------- preflight


def test_a_model_without_an_installation_is_refused_before_anything_is_planned(
    candidate: ServingWorld,
) -> None:
    model = get_model(candidate.services.db, candidate.model_id)
    with pytest.raises(EvaluationError, match=r"llama\.cpp is not installed"):
        preflight(candidate.services, model)
    preflight(candidate.services, model, candidate.program())  # with a program it is fine


def test_a_model_file_that_is_gone_is_refused(candidate: ServingWorld) -> None:
    candidate.model_file.unlink()
    model = get_model(candidate.services.db, candidate.model_id)
    with pytest.raises(EvaluationError, match="does not exist"):
        preflight(candidate.services, model, candidate.program())


def test_a_model_of_another_template_or_kind_is_refused(world: World, tmp_path: Path) -> None:
    odd = build_serving_world(
        world.services, tmp_path, active=False, gate_passed=None, template_version="x@1"
    )
    with pytest.raises(EvaluationError, match="x@1"):
        preflight(odd.services, get_model(odd.services.db, odd.model_id), odd.program())
    adapter = build_serving_world(
        world.services,
        tmp_path / "a",
        run_id="r-lora",
        quant="lora",
        kind="adapter",
        active=False,
        gate_passed=None,
    )
    with pytest.raises(EvaluationError, match="llamacpp_completion"):
        preflight(adapter.services, get_model(adapter.services.db, adapter.model_id))


def test_an_adapter_needs_the_login_of_the_instance(world: World, tmp_path: Path) -> None:
    adapter = build_serving_world(
        world.services,
        tmp_path,
        run_id="r-lora",
        quant="lora",
        kind="adapter",
        active=False,
        gate_passed=None,
    )
    adapter.services.settings.style_model.mode = "vllm_completion"
    with pytest.raises(EvaluationError, match=r"autodl\.host"):
        preflight(adapter.services, get_model(adapter.services.db, adapter.model_id))


# ------------------------------------------------------------------------- plan


async def test_the_plan_draws_new_contexts_for_the_three_backends_and_names_the_model(
    candidate: ServingWorld, api: respx.MockRouter
) -> None:
    model, plan = await plan_model_evaluation(
        candidate.services, candidate.model_id, 3, seed=11, program=candidate.program()
    )
    assert model.id == candidate.model_id and plan.samples == 3 and plan.pairs == 9
    run = plan.run
    assert run.backends == ("deepseek", "style", "hybrid")
    assert (
        run.params[MODEL_PARAM] == candidate.model_id and run.params["model_sha256"] == model.sha256
    )
    assert not get_model(candidate.services.db, candidate.model_id).active  # nothing activated
    assert plan.batch_ids and plan.estimated_usd > 0  # DeepSeek's replies and the planner cost
    store = EvalStore(candidate.services.db, candidate.services.clock)
    assert {item.backend for item in store.items(run.id)} == {"deepseek", "style", "hybrid"}
    # the contexts of the first evaluation are not drawn again
    _, second = await plan_model_evaluation(
        candidate.services, candidate.model_id, 3, seed=12, program=candidate.program()
    )
    first = {i.sample_key for i in store.items(run.id)}
    again = {i.sample_key for i in store.items(second.run.id)}
    assert first.isdisjoint(again)


async def test_the_plan_does_not_start_any_server(
    candidate: ServingWorld, api: respx.MockRouter
) -> None:
    await plan_model_evaluation(
        candidate.services, candidate.model_id, 2, program=candidate.program()
    )
    assert installed_servers() is None


# ----------------------------------------------------------------------- servers


async def test_the_pool_starts_the_server_of_the_candidate_on_the_evaluation_port(
    candidate: ServingWorld, api: respx.MockRouter
) -> None:
    pool = pool_for(candidate)
    llm = build_llm_runtime(candidate.services)
    try:
        style, pinned = await pool.style_runtime(llm, candidate.model_id)
        active = pinned.active()
        assert active is not None and active.id == candidate.model_id  # pinned, not active
        health = await style.client.health()
        assert health.ok
        port = candidate.services.settings.style_model.serve.eval_port
        assert port != candidate.port  # the server of the application is not touched
        again, _ = await pool.style_runtime(llm, candidate.model_id)  # the same server
        assert (await again.client.health()).ok
        await again.aclose()
        await style.aclose()
    finally:
        await pool.aclose()
        await llm.client.aclose()
    from twin.llm.style_client import LlamaCppCompletionClient

    gone = LlamaCppCompletionClient(
        f"http://127.0.0.1:{candidate.services.settings.style_model.serve.eval_port}",
        clock=candidate.services.clock,
    )
    assert not (await gone.health()).ok
    await gone.aclose()
    model = get_model(candidate.services.db, candidate.model_id)
    assert model.eval["tokenize_check"]["ok"] is True  # compared before the first prompt


async def test_a_candidate_whose_tokens_differ_is_refused(
    candidate: ServingWorld, api: respx.MockRouter
) -> None:
    pool = pool_for(candidate, "--sim-add-bos", "7")
    llm = build_llm_runtime(candidate.services)
    try:
        with pytest.raises(EvaluationError, match="differ from the training tokenizer"):
            await pool.style_runtime(llm, candidate.model_id)
    finally:
        await pool.aclose()
        await llm.client.aclose()
    model = get_model(candidate.services.db, candidate.model_id)
    assert model.eval["tokenize_check"]["ok"] is False


async def test_a_server_that_does_not_come_up_is_an_error_not_a_hang(
    candidate: ServingWorld, api: respx.MockRouter
) -> None:
    pool = EvaluationServers(
        candidate.services,
        tokenizers=candidate.tokenizers(),
        program=candidate.program("--sim-load-s", "600"),
        check_file=no_file_check,
        timings=ServerTimings(
            start_timeout_s=0.6,
            backoff_start_s=0.05,
            backoff_max_s=0.1,
            stable_after_s=60.0,
            health_interval_s=0.1,
            unhealthy_limit=3,
            stop_grace_s=5.0,
        ),
    )
    llm = build_llm_runtime(candidate.services)
    try:
        with pytest.raises(EvaluationError):
            await pool.style_runtime(llm, candidate.model_id)
    finally:
        await pool.aclose()
        await llm.client.aclose()


async def test_the_server_of_the_application_is_used_when_it_serves_this_very_file(
    candidate: ServingWorld, api: respx.MockRouter
) -> None:
    outside = await asyncio.create_subprocess_exec(
        *[
            *sim_prefix("--sim-tokenizer", str(candidate.tokenizer_json)),
            "-m",
            str(candidate.model_file),
            "--port",
            str(candidate.port),
        ],
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    client = LlamaCppCompletionClient(
        f"http://127.0.0.1:{candidate.port}", clock=candidate.services.clock
    )
    pool = pool_for(candidate)
    llm = build_llm_runtime(candidate.services)
    try:
        for _ in range(600):
            if (await client.health()).ok:
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError("the simulated server did not come up")
        style, _ = await pool.style_runtime(llm, candidate.model_id)
        assert (await style.client.health()).ok  # the pool uses the running server
        await style.aclose()
    finally:
        await pool.aclose()
        await llm.client.aclose()
        await client.aclose()
        outside.kill()
        await outside.wait()


async def test_the_evaluation_port_must_differ_from_the_port_of_the_application(
    candidate: ServingWorld, api: respx.MockRouter
) -> None:
    candidate.services.settings.style_model.serve.eval_port = candidate.port
    pool = pool_for(candidate)
    llm = build_llm_runtime(candidate.services)
    try:
        with pytest.raises(EvaluationError, match="eval_port"):
            await pool.style_runtime(llm, candidate.model_id)
    finally:
        await pool.aclose()
        await llm.client.aclose()


async def test_without_a_pool_the_job_says_what_to_do(
    candidate: ServingWorld, api: respx.MockRouter
) -> None:
    install_servers(None)
    llm = build_llm_runtime(candidate.services)
    try:
        with pytest.raises(EvaluationError, match="--foreground"):
            await evaluation_style(candidate.services, llm, candidate.model_id)
    finally:
        await llm.client.aclose()


# -------------------------------------------------------------- the pairs of a run


async def test_the_pairs_of_a_model_run_are_written_by_the_models_server(
    candidate: ServingWorld, world: World, api: respx.MockRouter, script: DeepSeekScript
) -> None:
    script.reply = lambda body: PLAN if body.get("response_format") else "好呀\n哈哈[拥抱]"
    requests_file = candidate.tmp / "requests.jsonl"
    program_options = ("--sim-requests", str(requests_file), "--sim-reply", "嗯嗯\n好哒")
    pool = pool_for(candidate, *program_options)
    install_servers(pool)
    services = candidate.services
    try:
        _, plan = await plan_model_evaluation(
            services, candidate.model_id, 2, seed=5, program=candidate.program()
        )
        store = EvalStore(services.db, services.clock)
        ids = [item.id for item in store.items(plan.run.id)]
        summary = await generate_items(services, plan.run.id, ids, plan.batch_ids[0])
    finally:
        install_servers(None)
        await pool.aclose()
    assert summary.failed == 0 and summary.generated == 6, summary
    items = store.items(plan.run.id)
    by_backend = {b: [i for i in items if i.backend == b] for b in ("deepseek", "style", "hybrid")}
    assert all(i.status == "generated" for i in items)

    def texts(backend: str) -> list[str]:
        return [
            line["t"]
            for i in by_backend[backend]
            for line in i.payload["bot"]["lines"]
            if line["k"] == "text"
        ]

    style_text = texts("style")
    assert set(style_text) == {"嗯嗯", "好哒"}  # what the model's server wrote
    deepseek_text = set(texts("deepseek"))
    assert "嗯嗯" not in deepseek_text
    hybrid_text = set(texts("hybrid"))
    assert hybrid_text == {"嗯嗯", "好哒"}  # DeepSeek planned, the model wrote
    # the model's server was asked with the prompt of the hold-out: the pre-holdout card, ChatML
    asked = [json.loads(line) for line in requests_file.read_text(encoding="utf-8").splitlines()]
    prompts = [r["body"]["prompt"] for r in asked if r["path"] == "/completion"]
    assert len(prompts) >= 4  # 2 style + 2 hybrid pairs (+ the warm-up is not made here)
    assert all(p.endswith("<|im_start|>assistant\n") for p in prompts)
    assert all(PAST_MARKER in p and LIVE_STYLE not in p for p in prompts)
    assert any("【规划】" in p for p in prompts)  # the hybrid ones
    assert not get_model(services.db, candidate.model_id).active  # still not the default


async def test_a_job_for_a_deepseek_only_batch_needs_no_server(
    candidate: ServingWorld, world: World, api: respx.MockRouter, script: DeepSeekScript
) -> None:
    install_servers(None)
    services = candidate.services
    _, plan = await plan_model_evaluation(
        services, candidate.model_id, 2, seed=6, program=candidate.program()
    )
    store = EvalStore(services.db, services.clock)
    only = [i.id for i in store.items(plan.run.id) if i.backend == "deepseek"]
    summary = await generate_items(services, plan.run.id, only, plan.batch_ids[0])
    assert summary.generated == len(only) and summary.failed == 0


async def test_the_jobs_of_the_run_are_the_ordinary_evaluation_jobs(
    candidate: ServingWorld, world: World, api: respx.MockRouter
) -> None:
    _, plan = await plan_model_evaluation(
        candidate.services, candidate.model_id, 2, seed=7, program=candidate.program()
    )
    with candidate.services.db.session() as session:
        from sqlalchemy import select

        types = {job.type for job in session.scalars(select(Job))}
    assert EVAL_GENERATE_JOB in types and plan.batch_ids


# ----------------------------------------------------------------------- remote


@contextlib.asynccontextmanager
async def remote_candidate(world: World, tmp_path: Path) -> AsyncIterator[ServingWorld]:
    candidate = build_serving_world(
        world.services,
        tmp_path,
        run_id="r-lora",
        quant="lora",
        kind="adapter",
        active=False,
        gate_passed=None,
        backend="deepseek",
    )
    tokenizer = tiny_qwen_tokenizer()
    with running_server() as vllm:
        vllm_defaults(vllm, model="twin-style")
        vllm.set(
            "POST",
            "/tokenize",
            handler=lambda payload: {
                "count": 0,
                "max_model_len": 4096,
                "tokens": tokenizer.encode(payload["prompt"]),
            },
        )
        async with LocalSSHServer(tmp_path / "ssh") as ssh:
            ssh.state.allow_forwarding = True
            config = candidate.services.settings.style_model
            config.mode = "vllm_completion"
            config.model_id = "twin-style"
            config.tunnel.local_port = free_port()
            config.tunnel.remote_port = vllm.port
            config.tunnel.backoff_start_s = 0.05
            config.tunnel.backoff_max_s = 0.2
            auto = candidate.services.settings.autodl
            auto.host, auto.port, auto.user = "127.0.0.1", ssh.port, USER
            candidate.services.secrets.set("autodl_password", PASSWORD)
            remember_host(
                tunnel_known_hosts(candidate.services), "127.0.0.1", ssh.port, ssh.host_key
            )
            yield candidate


async def test_an_adapter_is_evaluated_through_the_tunnel_that_the_pool_opens(
    world: World, api: respx.MockRouter, tmp_path: Path
) -> None:
    async with remote_candidate(world, tmp_path) as candidate:
        pool = EvaluationServers(candidate.services, tokenizers=candidate.tokenizers())
        llm = build_llm_runtime(candidate.services)
        try:
            style, pinned = await pool.style_runtime(llm, candidate.model_id)
            assert (await style.client.health()).ok  # vLLM behind the tunnel, model twin-style
            assert pinned.active() is not None
            again, _ = await pool.style_runtime(llm, candidate.model_id)
            assert (await again.client.health()).ok  # the same tunnel
            await again.aclose()
            await style.aclose()
        finally:
            await pool.aclose()
            await llm.client.aclose()
        model = get_model(candidate.services.db, candidate.model_id)
        assert model.eval["tokenize_check"]["ok"] is True
        port = candidate.services.settings.style_model.tunnel.local_port
        gone = VllmCompletionClient(
            f"http://127.0.0.1:{port}", model="twin-style", clock=candidate.services.clock
        )
        assert not (await gone.health()).ok  # the pool closed its tunnel
        await gone.aclose()


async def test_an_existing_tunnel_is_used_and_left_open(
    world: World, api: respx.MockRouter, tmp_path: Path
) -> None:
    async with remote_candidate(world, tmp_path) as candidate:
        services = candidate.services
        config = services.settings.style_model
        auto = services.settings.autodl
        assert auto.port is not None
        target = RemoteTarget(
            "127.0.0.1",
            auto.port,
            USER,
            "password",
            tunnel_known_hosts(services),
            password=PASSWORD,
        )
        tunnel = TunnelManager(
            target,
            local_port=config.tunnel.local_port,
            remote_port=config.tunnel.remote_port,
            clock=services.clock,
            timings=TunnelTimings(0.05, 0.2),
        )
        task = asyncio.create_task(tunnel.run())
        try:
            await wait_until(lambda: tunnel.state is TunnelState.UP, limit_s=30)
            pool = EvaluationServers(services, tokenizers=candidate.tokenizers())
            llm = build_llm_runtime(services)
            try:
                style, _ = await pool.style_runtime(llm, candidate.model_id)
                assert (await style.client.health()).ok
                await style.aclose()
            finally:
                await pool.aclose()
                await llm.client.aclose()
            assert tunnel.state is TunnelState.UP  # not closed by the pool
        finally:
            await tunnel.stop()
            await asyncio.wait_for(task, 20)
        assert services.runtime.get(TUNNEL_WANTED) is False
