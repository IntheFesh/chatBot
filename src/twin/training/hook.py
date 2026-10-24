"""After an import: is it time to train again? (R-IMP-011, R-TRN-012).

Importing adds her messages.  The hook compares them with the messages the last training covered
(:mod:`twin.training.retrain`) and raises the ``retrain_suggested`` alert once the new share
reaches ``training.retrain_new_ratio``.  It trains nothing and costs nothing.  The backfill
command is ``twin train retrain-check``.
"""

from __future__ import annotations

from twin.ingest.hooks import HookContext, HookResult, post_import_hook
from twin.training.retrain import NO_TRAINING, check_retrain


@post_import_hook(
    "retrain",
    backfill_command="train retrain-check",
    description="suggest training the style model again when her new messages reach 10 %",
)
def check_retraining(context: HookContext) -> HookResult:
    status = check_retrain(context.services)
    if status.reason == NO_TRAINING:
        return HookResult("skipped", status.describe())
    return HookResult("done", status.describe())
