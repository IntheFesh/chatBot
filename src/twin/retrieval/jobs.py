"""The ``retrieval_index`` job: sync the example windows and encode the new ones.

The work is synchronous (SQLite reads, the embedding model, LanceDB writes), so it runs in a
worker thread; cancelling the job asks the thread to stop after the chunk it is encoding.  An
index made with another model is not touched: the job raises an alert that names the command to
run instead of retrying (retrying cannot help).
"""

from __future__ import annotations

import asyncio
import threading

from twin.ops.jobs import JobContext, JobDeferred, job_handler
from twin.ops.logging import get_logger
from twin.retrieval.indexer import INDEX_JOB, IndexBusyError, IndexMode, run_index
from twin.retrieval.vector_store import IndexMismatchError

log = get_logger("twin.retrieval")
MISMATCH_ALERT = "retrieval_index"


@job_handler(INDEX_JOB)
async def handle_retrieval_index(ctx: JobContext) -> None:
    services = ctx.services
    if services is None:
        raise RuntimeError("the retrieval job needs the services container")
    payload = ctx.job.payload
    mode: IndexMode = "rebuild" if payload.get("mode") == "rebuild" else "update"
    stop = threading.Event()
    try:
        result = await asyncio.to_thread(
            run_index, services, mode=mode, full=bool(payload.get("full", False)), stop=stop
        )
    except asyncio.CancelledError:
        stop.set()
        raise
    except IndexBusyError as exc:
        raise JobDeferred(str(exc), retry_in_s=120.0) from exc
    except IndexMismatchError as exc:
        services.alerts.raise_alert(
            MISMATCH_ALERT,
            "检索库的向量模型与当前配置不一致，请运行 twin retrieval rebuild",
            detail={"error": str(exc)},
            dedup_key="retrieval-index-mismatch",
        )
        log.warning("retrieval_index_mismatch")
        return
    log.info(
        "retrieval_indexed",
        encoded=result.encoded,
        pending=result.pending,
        reset=result.reset,
        seconds=round(result.seconds, 1),
    )
