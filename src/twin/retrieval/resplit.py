"""What the example library does when the hold-out is re-split (R-RET-003, R-TRN-013).

``twin retrieval resplit`` and ``twin train export`` call
:func:`~twin.profile.holdout.resplit_holdout`, which stores the new cutoff and tells every
dependant.  The library's answer:

* the ``holdout`` flags of the stored windows follow the new cutoff at once (a database update,
  no message is read);
* no vector moves: the library holds every window, the held-out period included, and the
  evaluation is kept from the future by the ``before`` of ``AsOfView(t)`` (round 18, user
  decision, D-880);
* windows that still wait for a vector - an index made before round 18 left the held-out ones
  out - are queued for encoding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from twin.profile.holdout import Holdout, on_holdout_change
from twin.retrieval.indexer import awaiting, count_windows
from twin.retrieval.queue import queue_retrieval_index
from twin.retrieval.windows import apply_holdout

if TYPE_CHECKING:
    from twin.services import Services


@on_holdout_change("retrieval")
def retrieval_after_resplit(services: Services, previous: Holdout | None, current: Holdout) -> str:
    if count_windows(services) == 0:
        return "the example library has not been built yet; `twin retrieval rebuild` builds it"
    entering, released = apply_holdout(services, current.cutoff)
    note = (
        f"{len(entering):,} window(s) are now held out and {released:,} were released "
        "(the flags only: every window stays in the library)"
    )
    waiting = count_windows(services, *awaiting(None))
    if not waiting:
        return note
    queued = queue_retrieval_index(services, mode="update", reason="resplit")
    state = "already queued" if queued.already_queued else "queued"
    return f"{note}; encoding the {waiting:,} window(s) without a vector is {state}"
