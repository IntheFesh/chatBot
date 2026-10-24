"""The post-import hook of the profile and routine recomputation (R-IMP-011).

After an import that brought new messages from her (or the first import), the recomputation
of both scopes is queued; the job writes new versions and the import report gets its
"风格变化" section.  The backfill command is ``twin profile rebuild``.
"""

from __future__ import annotations

from sqlalchemy import func, select

from twin.ingest.hooks import HookContext, HookResult, post_import_hook
from twin.profile.queue import queue_profile_rebuild
from twin.profile.store import VersionStore
from twin.storage.chat_models import Message


def her_message_count(context: HookContext) -> int:
    with context.services.db.session() as session:
        found = session.scalar(
            select(func.count())
            .select_from(Message)
            .where(Message.is_sent.is_(False), Message.kind != "system")
        )
    return int(found or 0)


@post_import_hook(
    "profile",
    backfill_command="profile rebuild",
    description="recompute the style profile and the routine model (live and pre_holdout)",
)
def queue_profile(context: HookContext) -> HookResult:
    services = context.services
    store = VersionStore(services.db, services.clock)
    latest = store.latest_profile("live")
    count = her_message_count(context)
    if count == 0:
        return HookResult("skipped", "no messages from her yet")
    if (
        latest is not None
        and latest.her_messages == count
        and context.changed == 0
        and not context.first_import
    ):
        return HookResult("skipped", "no new messages from her since the last profile")
    queued = queue_profile_rebuild(services, scope="all", reason="import", run_id=context.run_id)
    if queued.already_queued:
        return HookResult("queued", "a profile recomputation is already waiting", jobs=0)
    return HookResult("queued", f"profile and routine recomputation for {count:,} messages", jobs=1)
