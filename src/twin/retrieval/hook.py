"""The post-import hook of the retrieval library (R-IMP-011, R-RET-006).

After an import that brought messages, an index run is queued: it adds the windows of the new
messages (only those are encoded) and leaves the rest alone.  An index made with a different
model is reported instead of being mixed with new vectors.  The backfill command is
``twin retrieval rebuild``.
"""

from __future__ import annotations

from twin.ingest.hooks import HookContext, HookResult, post_import_hook
from twin.retrieval.indexer import window_table
from twin.retrieval.queue import queue_retrieval_index


@post_import_hook(
    "retrieval",
    backfill_command="retrieval rebuild",
    description="add the windows of the new messages to the example library (hold-out excluded)",
)
def queue_retrieval(context: HookContext) -> HookResult:
    services = context.services
    if context.inserted == 0 and context.changed == 0 and not context.first_import:
        return HookResult("skipped", "no new messages, so no new example windows")
    meta = window_table(services).read_meta()
    configured = services.settings.retrieval.model
    if meta is not None and meta.model != configured:
        return HookResult(
            "failed",
            f"the example library was made with {meta.model} but retrieval.model is "
            f"{configured}; run `twin retrieval rebuild`",
        )
    queued = queue_retrieval_index(services, mode="update", reason="import", run_id=context.run_id)
    if queued.already_queued:
        return HookResult("queued", "an index update is already waiting", jobs=0)
    return HookResult("queued", "example windows of the new messages will be encoded", jobs=1)
