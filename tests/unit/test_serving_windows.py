"""The local server on Windows ends with the application, however that ends (R-SRV-002).

These run only on Windows (``pytest.mark.windows``; CI skips them elsewhere).  A parent process
starts the simulated llama-server through :class:`~twin.serving.server.LlamaServerManager` - the
class ``twin run`` uses - and the test kills the parent the hard way, as a crash or the task
manager would; the job object must take the child down with it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import httpx
import pytest

from tests.support.llama_sim import free_port

pytestmark = pytest.mark.windows

ROOT = Path(__file__).resolve().parents[2]

PARENT = textwrap.dedent(
    """
    import asyncio
    import sys
    from pathlib import Path

    from tests.support.llama_sim import make_model_file, spec_for
    from twin.clock import SystemClock
    from twin.llm.style_client import LlamaCppCompletionClient
    from twin.ops.jobobject import ProcessJob
    from twin.serving.server import LlamaServerManager, ServerState, ServerTimings

    TIMINGS = ServerTimings(
        start_timeout_s=60.0,
        backoff_start_s=0.1,
        backoff_max_s=0.5,
        stable_after_s=60.0,
        health_interval_s=0.2,
        unhealthy_limit=3,
        stop_grace_s=5.0,
    )


    async def main(port: int, folder: Path, how: str) -> None:
        job = ProcessJob()
        if how == "adopt":  # twin run: the whole process is in the job
            assert job.adopt_current_process()
        model = make_model_file(folder / "models")
        client = LlamaCppCompletionClient(f"http://127.0.0.1:{port}", clock=SystemClock())
        manager = LlamaServerManager(
            spec_for(model, port),
            clock=SystemClock(),
            client=client,
            log_path=folder / "logs" / "llama-server.log",
            timings=TIMINGS,
            job=job if how == "assign" else None,  # the manager puts the child in the job
        )
        task = asyncio.create_task(manager.run())
        while manager.state is not ServerState.READY:
            await asyncio.sleep(0.1)
        print("ready", flush=True)
        await asyncio.sleep(300)
        await manager.stop()
        await task


    asyncio.run(main(int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]))
    """
)


def answers(port: int) -> bool:
    try:
        return httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


def wait_for(condition: object, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if callable(condition) and condition():
            return True
        time.sleep(0.1)
    return False


@pytest.mark.parametrize("how", ["adopt", "assign"])
def test_when_the_application_dies_the_server_dies_with_it(tmp_path: Path, how: str) -> None:
    port = free_port()
    parent = subprocess.Popen(
        [sys.executable, "-c", PARENT, str(port), str(tmp_path), how],
        stdout=subprocess.PIPE,
        text=True,
        cwd=ROOT,
    )
    try:
        assert parent.stdout is not None and parent.stdout.readline().strip() == "ready"
        assert answers(port)  # the server is up and answering
        parent.kill()  # no clean-up code runs: a crash, or the task manager
        parent.wait(timeout=10)
        assert wait_for(lambda: not answers(port), 15), "the server outlived its parent"
    finally:
        parent.kill()
