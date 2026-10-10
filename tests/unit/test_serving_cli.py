"""``twin model recommend|activate|disable|verify|serve|evaluate|tunnel`` through the real CLI.

R-SRV-001 (activate, disable), R-SRV-002 (recommend, serve), R-SRV-003 (tunnel), R-SRV-004 (verify),
R-SRV-005 (evaluate).  The commands run against a database in the test's data directory and find
``llama-server`` the way they do on the user's computer - through ``style_model.serve.binary`` -
but the program is a small executable that runs the simulated server (POSIX).  The tokenizer is
the tiny test one, DeepSeek is a scripted ``respx`` route and the rented instance is a real SSH
server on localhost.
"""

from __future__ import annotations

import io
import os
import re
import signal
import socket
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest
import respx
from rich.console import Console
from sqlalchemy import select
from typer.testing import CliRunner

import twin.serving.cli as serving_cli
from tests.support.embedding import HashingBackend
from tests.support.eval_world import DeepSeekScript, build_eval_world, mock_deepseek
from tests.support.export_world import World
from tests.support.llama_sim import free_port
from tests.support.serving_world import ServingWorld, build_serving_world, install_wrapper
from tests.support.ssh_server import PASSWORD, USER, ThreadedSSHServer
from tests.support.style_server import running_server, vllm_defaults
from tests.support.tiny_tokenizer import tiny_qwen_tokenizer
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.config.runtime import BACKEND_ACTIVE, TUNNEL_WANTED
from twin.config.secrets import SecretStore
from twin.eval.cli import use_interaction
from twin.eval.store import EvalStore
from twin.eval.ui import LineKeys
from twin.services import Services, build_services
from twin.serving.evaluation import tunnel_known_hosts
from twin.serving.runtime import TokenizerSource
from twin.serving.state import ServingStateStore
from twin.storage.models import Alert
from twin.training.registry import get_model
from twin.training.remote.connection import remember_host

runner = CliRunner()

posix_only = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the program the commands start is an executable shell script that runs the simulation",
)


def open_services() -> Services:
    settings = load_settings()
    return build_services(settings, root=resolve_paths(settings).root)


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    monkeypatch.setenv("TWIN_RETRIEVAL__MODEL", "test/hashing-bigram")
    assert runner.invoke(app, ["db", "upgrade"]).exit_code == 0
    return path


@pytest.fixture(autouse=True)
def tiny_tokenizers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The commands compare against the tiny tokenizer (the real one is downloaded)."""
    monkeypatch.setattr(
        serving_cli,
        "TokenizerSource",
        lambda services: TokenizerSource(services, loader=tiny_qwen_tokenizer),
    )


def point_commands_at(
    monkeypatch: pytest.MonkeyPatch, world: ServingWorld, *sim_options: str
) -> None:
    """The settings the commands read: where the model is served and by which program."""
    program = install_wrapper(world.tmp / "bin", *sim_options, tokenizer_json=world.tokenizer_json)
    serve = world.services.settings.style_model.serve
    monkeypatch.setenv("TWIN_STYLE_MODEL__ENDPOINT", f"http://127.0.0.1:{world.port}")
    monkeypatch.setenv("TWIN_STYLE_MODEL__MODEL_ID", "twin-style")
    monkeypatch.setenv("TWIN_STYLE_MODEL__SERVE__BINARY", str(program))
    monkeypatch.setenv("TWIN_STYLE_MODEL__SERVE__EVAL_PORT", str(serve.eval_port))
    monkeypatch.setenv("TWIN_STYLE_MODEL__SERVE__START_TIMEOUT_S", "30")
    monkeypatch.setenv("TWIN_STYLE_MODEL__SERVE__BACKOFF_START_S", "0.05")
    monkeypatch.setenv("TWIN_STYLE_MODEL__SERVE__BACKOFF_MAX_S", "0.2")


@pytest.fixture
def served(
    data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[ServingWorld]:
    """A registered model (not active, no gate) whose file matches the registry."""
    services = open_services()
    try:
        world = build_serving_world(
            services, tmp_path, active=False, gate_passed=None, backend="deepseek", real_sha256=True
        )
        point_commands_at(monkeypatch, world)
        yield world
    finally:
        services.close()


def invoke(*args: str) -> tuple[int, str]:
    result = runner.invoke(app, ["model", *args])
    return result.exit_code, result.output


# ------------------------------------------------------------------ recommend


def test_recommend_without_a_registered_model_estimates_from_the_profiles(data_dir: Path) -> None:
    code, out = invoke("recommend", "--vram", "24GiB")
    assert code == 0 and "graphics card: 24.0 GiB (--vram)" in out
    assert "5090-8b" in out and "pro6000-14b" in out and "no GGUF registered yet" in out
    assert "Q8_0" in out or "Q5_K_M" in out or "Q4_K_M" in out


def test_recommend_for_a_computer_without_a_card_says_q4_and_the_latency(data_dir: Path) -> None:
    code, out = invoke("recommend", "--cpu")
    assert code == 0 and "graphics card: none (--cpu)" in out and "Q4_K_M" in out


def test_recommend_reads_the_card_when_no_size_is_given(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(serving_cli, "detect_gpu", lambda runner: None)
    code, out = invoke("recommend")
    assert code == 0 and "none found (nvidia-smi)" in out and "llama.cpp build for it: cpu" in out


def test_recommend_names_the_registered_file_to_evaluate(served: ServingWorld) -> None:
    code, out = invoke("recommend", "--vram", "32GiB")
    assert code == 0 and f"run {served.model_id.split('-Q')[0]}" in out
    assert f"twin model evaluate {served.model_id}" in out


def test_recommend_refuses_a_size_it_cannot_read(data_dir: Path) -> None:
    code, out = invoke("recommend", "--vram", "lots")
    assert code == 2 and "24GiB" in out


# ------------------------------------------------------------ activate, disable


@posix_only
def test_activate_without_a_passed_gate_is_refused_and_changes_nothing(
    served: ServingWorld,
) -> None:
    code, out = invoke("activate", served.model_id)
    assert code == 1 and "did not pass the release gate (M5)" in out
    model = get_model(served.services.db, served.model_id)
    assert model.active is False and model.gate_passed is None
    assert served.services.runtime.get(BACKEND_ACTIVE) == "deepseek"


@posix_only
def test_activate_force_is_on_record_marks_the_gate_failed_and_follows_the_choice(
    served: ServingWorld,
) -> None:
    code, out = invoke("activate", served.model_id, "--force", "--backend", "hybrid")
    assert code == 0, out
    assert "activated" in out and "backend.active = hybrid" in out and "FORCED" in out
    model = get_model(served.services.db, served.model_id)
    assert model.active is True and model.gate_passed is False
    assert served.services.runtime.get(BACKEND_ACTIVE) == "hybrid"
    log = model.eval["activation_log"]
    assert log[-1]["forced"] is True and log[-1]["action"] == "activate"
    # the tokens were compared on the way: the record names the file it is valid for
    assert model.eval["tokenize_check"]["ok"] is True


@posix_only
def test_activate_refuses_a_model_whose_tokens_differ_even_with_force(
    served: ServingWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    point_commands_at(monkeypatch, served, "--sim-add-bos", "5")
    code, out = invoke("activate", served.model_id, "--force")
    assert code == 1 and "differ from the training tokenizer" in out
    assert "extra token(s) in front" in out and "first difference at token 0" in out
    assert get_model(served.services.db, served.model_id).active is False
    with served.services.db.session() as session:
        categories = list(session.scalars(select(Alert.category)))
    assert "style_tokenize_mismatch" in categories


@posix_only
def test_activate_an_unknown_model_and_a_bad_backend_are_errors(served: ServingWorld) -> None:
    code, out = invoke("activate", "r-nope")
    assert code == 1 and "r-nope" in out
    code, out = invoke("activate", served.model_id, "--force", "--backend", "gpt")
    assert code == 1 and "--backend is style or hybrid" in out


@posix_only
def test_disable_ends_the_use_and_goes_back_to_deepseek(served: ServingWorld) -> None:
    assert invoke("activate", served.model_id, "--force")[0] == 0
    assert served.services.runtime.get(BACKEND_ACTIVE) == "style"
    code, out = invoke("disable", served.model_id)
    assert code == 0 and f"disabled {served.model_id}" in out and "deepseek again" in out
    assert get_model(served.services.db, served.model_id).active is False
    assert served.services.runtime.get(BACKEND_ACTIVE) == "deepseek"
    code, out = invoke("disable", "r-nope")
    assert code == 1


# ---------------------------------------------------------------------- verify


@posix_only
def test_verify_starts_the_server_and_compares_the_tokens(served: ServingWorld) -> None:
    code, out = invoke("verify", served.model_id)
    assert code == 0, out
    assert "tokenization matches" in out
    check = get_model(served.services.db, served.model_id).eval["tokenize_check"]
    assert check["ok"] is True


@posix_only
def test_verify_reports_the_position_of_a_difference_and_raises_the_alert(
    served: ServingWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    point_commands_at(monkeypatch, served, "--sim-add-bos", "5")
    code, out = invoke("verify", served.model_id)
    assert code == 1 and "tokenization differs" in out
    check = get_model(served.services.db, served.model_id).eval["tokenize_check"]
    assert check["ok"] is False
    with served.services.db.session() as session:
        categories = list(session.scalars(select(Alert.category)))
    assert "style_tokenize_mismatch" in categories


def test_verify_without_an_installation_says_how_to_get_one(
    served: ServingWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TWIN_STYLE_MODEL__SERVE__BINARY")
    code, out = invoke("verify", served.model_id)
    assert code == 1 and "get_llamacpp.ps1" in out


# ----------------------------------------------------------------------- serve


def interrupt_when_warm(services: Services, model_id: str, done: threading.Event) -> None:
    """Ctrl+C for the command in the main thread once it has recorded the warm-up of the server."""
    import time

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not done.is_set():
        if "warmup" in get_model(services.db, model_id).eval:
            os.kill(os.getpid(), signal.SIGINT)
            return
        time.sleep(0.1)


@posix_only
def test_serve_runs_the_server_warms_it_up_and_ends_it_with_ctrl_c(
    served: ServingWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    done = threading.Event()
    thread = threading.Thread(
        target=interrupt_when_warm, args=(served.services, served.model_id, done), daemon=True
    )
    thread.start()
    try:
        code, out = invoke("serve", served.model_id)
    finally:
        done.set()
        thread.join(10)
    assert code == 0, out
    assert f"serving {served.model_id} on http://127.0.0.1:{served.port}" in out
    assert "warm-up: first token" in out
    warmup = get_model(served.services.db, served.model_id).eval["warmup"]
    assert warmup["measured_by"] == "server" and warmup["first_token_ms"] >= 0  # llama.cpp times
    with pytest.raises((urllib.error.URLError, OSError)):  # the server ended with the command
        urllib.request.urlopen(f"http://127.0.0.1:{served.port}/health", timeout=1)


def test_serve_does_not_compete_with_the_running_application(
    served: ServingWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(serving_cli, "app_is_running", lambda services: True)
    code, out = invoke("serve", served.model_id)
    assert code == 1 and "starts llama-server by itself" in out


def test_serve_of_a_remote_model_points_to_the_tunnel(
    served: ServingWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TWIN_STYLE_MODEL__MODE", "vllm_completion")
    code, out = invoke("serve", served.model_id)
    assert code == 1 and "twin model tunnel start" in out


def test_serve_refuses_an_endpoint_that_is_not_on_this_computer(
    served: ServingWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TWIN_STYLE_MODEL__ENDPOINT", "http://192.168.1.20:8081")
    code, out = invoke("serve", served.model_id)
    assert code == 1 and "not on this computer" in out


def test_serve_without_a_model_says_which_ones_there_are(data_dir: Path) -> None:
    code, out = invoke("serve")
    assert code == 1 and "no model is active" in out


# -------------------------------------------------------------------- evaluate


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    script = DeepSeekScript(
        lambda body: (
            '{"reply": true, "intent": "答应", "facts_to_use": [], "tone": "轻松", '
            '"bubble_hint": "两条", "sticker_hint": ""}'
            if body.get("response_format")
            else "好呀\n哈哈[拥抱]"
        )
    )
    yield from mock_deepseek(script)


@pytest.fixture
def evaluated(
    data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: HashingBackend,
    api: respx.MockRouter,
) -> Iterator[tuple[World, ServingWorld]]:
    """The synthetic conversation of the evaluation tests and a model that is not active."""
    api.route(host="127.0.0.1").pass_through()
    services = open_services()
    try:
        world = build_eval_world(services, embedder, days=30)
        model = build_serving_world(
            world.services,
            tmp_path,
            active=False,
            gate_passed=None,
            backend="deepseek",
            real_sha256=True,
        )
        point_commands_at(monkeypatch, model)
        yield world, model
    finally:
        services.close()


def evaluate(*args: str, keys: str = "") -> tuple[int, str, str]:
    """Invoke ``twin model evaluate ...``: (exit code, printed lines, the screen)."""
    screen = io.StringIO()
    console = Console(file=screen, width=120, color_system=None, highlight=False)
    with use_interaction(console, LineKeys(io.StringIO(keys))):
        result = runner.invoke(app, ["model", "evaluate", *args])
    return result.exit_code, result.output, screen.getvalue()


@posix_only
def test_evaluate_plans_generates_with_the_models_server_judges_and_previews_the_gate(
    evaluated: tuple[World, ServingWorld],
) -> None:
    world, model = evaluated
    code, out, screen = evaluate(model.model_id, "--n", "2", "--seed", "3")
    assert code == 0, out + screen
    assert "盲测" in screen and "twin jobs approve" in screen and model.model_id in screen
    store = EvalStore(world.services.db, world.services.clock)
    (run,) = store.list_runs()
    assert run.params["model_id"] == model.model_id
    for batch in re.findall(r"批次 (\S+?)：", screen):
        approved = runner.invoke(app, ["jobs", "approve", batch, "--yes"])
        assert approved.exit_code == 0, approved.output
    # without --foreground nothing is generated; the model's server is not needed to say so
    code, _, screen = evaluate(model.model_id, "--resume", run.id)
    assert code == 0 and "还有 6 对没有生成" in screen
    # with it the jobs run here, on the model's own server, then the pairs are judged
    code, out, screen = evaluate(model.model_id, "--resume", run.id, "--foreground", keys="1\n" * 6)
    assert code == 0, out + screen
    assert "本次判了 6 对" in screen
    assert "M5" in screen and "twin eval gate M5" in screen
    assert store.get_run(run.id).status == "done"
    # the server of the evaluation ended with the command
    with pytest.raises((urllib.error.URLError, OSError)):
        urllib.request.urlopen(
            f"http://127.0.0.1:{model.services.settings.style_model.serve.eval_port}/health",
            timeout=1,
        )


def test_evaluate_refuses_a_run_that_was_drawn_for_another_model(
    evaluated: tuple[World, ServingWorld],
) -> None:
    world, model = evaluated
    store = EvalStore(world.services.db, world.services.clock)
    other = store.create_run("blind", status="running", params={"model_id": "r-other"})
    code, out, _ = evaluate(model.model_id, "--resume", other.id)
    assert code == 1 and f"was not drawn for {model.model_id}" in out
    code, out, _ = evaluate(model.model_id, "--resume", "nothere")
    assert code == 1 and "no evaluation run" in out


def test_evaluate_of_a_model_that_cannot_be_served_says_why(
    evaluated: tuple[World, ServingWorld], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, model = evaluated
    model.model_file.unlink()
    code, out, _ = evaluate(model.model_id, "--n", "2")
    assert code == 1 and "does not exist" in out
    code, out, _ = evaluate("r-nope", "--n", "2")
    assert code == 1 and "r-nope" in out


# ---------------------------------------------------------------------- tunnel


@pytest.fixture
def instance(
    data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Services, ThreadedSSHServer, int]]:
    """A rented instance (SSH, forwarding allowed, vLLM behind it) and the settings for it."""
    with running_server() as vllm, ThreadedSSHServer(tmp_path / "srv") as ssh:
        vllm_defaults(vllm, model="twin-style")
        ssh.server.state.allow_forwarding = True
        local_port = free_port()
        for key, value in {
            "TWIN_STYLE_MODEL__MODE": "vllm_completion",
            "TWIN_STYLE_MODEL__MODEL_ID": "twin-style",
            "TWIN_STYLE_MODEL__ENDPOINT": f"http://127.0.0.1:{local_port}",
            "TWIN_STYLE_MODEL__TUNNEL__LOCAL_PORT": str(local_port),
            "TWIN_STYLE_MODEL__TUNNEL__REMOTE_PORT": str(vllm.port),
            "TWIN_STYLE_MODEL__TUNNEL__BACKOFF_START_S": "0.05",
            "TWIN_STYLE_MODEL__TUNNEL__BACKOFF_MAX_S": "0.2",
            "TWIN_AUTODL__HOST": "127.0.0.1",
            "TWIN_AUTODL__PORT": str(ssh.port),
            "TWIN_AUTODL__USER": USER,
        }.items():
            monkeypatch.setenv(key, value)
        SecretStore.default().set("autodl_password", PASSWORD)
        services = open_services()
        try:
            remember_host(tunnel_known_hosts(services), "127.0.0.1", ssh.port, ssh.server.host_key)
            yield services, ssh, local_port
        finally:
            services.close()


def test_tunnel_status_without_a_tunnel_says_so(data_dir: Path) -> None:
    code, out = invoke("tunnel", "status")
    assert code == 0 and "wanted" in out and "no tunnel is held by the application" in out


def test_tunnel_start_needs_the_remote_mode(data_dir: Path) -> None:
    code, out = invoke("tunnel", "start")
    assert code == 1 and "there is no remote model to reach" in out


def test_tunnel_start_without_the_login_data_says_what_is_missing(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TWIN_STYLE_MODEL__MODE", "vllm_completion")
    code, out = invoke("tunnel", "start")
    assert code == 1 and "autodl" in out.lower()


def test_tunnel_start_while_the_application_runs_only_asks_for_it(
    instance: tuple[Services, ThreadedSSHServer, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    services, _, _ = instance
    monkeypatch.setattr(serving_cli, "app_is_running", lambda services: True)
    code, out = invoke("tunnel", "start")
    assert code == 0 and "brings the tunnel up" in out
    assert services.runtime.get(TUNNEL_WANTED) is True
    code, out = invoke("tunnel", "stop")
    assert code == 0 and "the tunnel is closed" in out and "billed by the hour" in out
    assert services.runtime.get(TUNNEL_WANTED) is False
    code, out = invoke("tunnel", "stop")
    assert code == 0 and "was not wanted" in out


def test_tunnel_stop_says_that_replies_fall_back_while_a_style_backend_is_asked_for(
    instance: tuple[Services, ThreadedSSHServer, int],
) -> None:
    services, _, _ = instance
    services.runtime.set(BACKEND_ACTIVE, "style", by="test")
    code, out = invoke("tunnel", "stop")
    assert code == 0 and "backend.active is style" in out and "/后端 deepseek" in out


def connect_when_listening(port: int, done: threading.Event, then: threading.Event) -> None:
    """Wait until the tunnel's local end accepts a connection, then let the test end it."""
    import time

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not done.is_set():
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                then.set()
                return
        except OSError:
            time.sleep(0.1)


def test_tunnel_start_holds_the_tunnel_until_it_is_stopped_from_another_window(
    instance: tuple[Services, ThreadedSSHServer, int],
) -> None:
    services, _, local_port = instance
    done, up = threading.Event(), threading.Event()
    threading.Thread(
        target=connect_when_listening, args=(local_port, done, up), daemon=True
    ).start()

    def stop_from_another_window() -> None:
        if up.wait(60):
            other = open_services()  # `twin model tunnel stop` is another process
            try:
                other.runtime.set(TUNNEL_WANTED, False, by="command")
            finally:
                other.close()

    stopper = threading.Thread(target=stop_from_another_window, daemon=True)
    stopper.start()
    try:
        code, out = invoke("tunnel", "start")
    finally:
        done.set()
        stopper.join(10)
    assert code == 0, out
    assert f"127.0.0.1:{local_port} -> the instance's 127.0.0.1:" in out
    assert "tunnel closed" in out
    assert services.runtime.get(TUNNEL_WANTED) is False


def test_tunnel_status_shows_the_record_of_the_application_and_whether_the_port_answers(
    instance: tuple[Services, ThreadedSSHServer, int],
) -> None:
    services, _, local_port = instance
    store = ServingStateStore(services.db, services.clock)
    store.update(
        {
            "tunnel": {
                "state": "up",
                "up_since": "2026-10-10T08:00:00+00:00",
                "reconnects": 2,
                "instance_uptime_s": 7200,
                "detail": "",
            }
        }
    )
    services.runtime.set(TUNNEL_WANTED, True, by="test")
    code, out = invoke("tunnel", "status")
    assert code == 0, out
    assert "up" in out and "reconnects" in out and "2.0 h" in out
    assert "no application is running: this record may be old" in out
    assert f"127.0.0.1:{local_port}" in out and "does not answer" in out
