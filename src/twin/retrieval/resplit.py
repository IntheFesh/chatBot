"""What the example library does when the hold-out is re-split (R-RET-003, R-TRN-013).

``twin retrieval resplit`` calls :func:`~twin.profile.holdout.resplit_holdout`, which stores the
new cutoff and tells every dependant.  The library's answer:

* the ``holdout`` flags of the stored windows follow the new cutoff at once (a database update,
  no message is read);
* windows that are now held out leave the index immediately - the evaluation must never find
  them - and windows that were released are queued for encoding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from twin.clock import to_epoch
from twin.profile.holdout import Holdout, on_holdout_change
from twin.retrieval.indexer import count_windows, window_table
from twin.retrieval.queue import queue_retrieval_index
from twin.retrieval.windows import apply_holdout

if TYPE_CHECKING:
    from twin.services import Services


@on_holdout_change("retrieval")
def retrieval_after_resplit(services: Services, previous: Holdout | None, current: Holdout) -> str:
    if count_windows(services) == 0:
        return "the example library has not been built yet; `twin retrieval rebuild` builds it"
    entering, released = apply_holdout(services, current.cutoff)
    table = window_table(services)
    table.delete_ids(list(entering))
    table.delete_from(int(to_epoch(current.cutoff)) + 1)
    queued = queue_retrieval_index(services, mode="update", reason="resplit")
    waiting = "already queued" if queued.already_queued else "queued"
    return (
        f"{len(entering):,} window(s) are now held out and left the index, "
        f"{released:,} were released; encoding the released ones is {waiting}"
    )
