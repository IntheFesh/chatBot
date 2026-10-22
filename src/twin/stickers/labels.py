"""The label of a sticker, from the library, for every place that shows one (R-STK-002, D-691).

A sticker the bot reads or writes is shown by one word: the first of its tags - ``[表情包:开心]``
in an example of a prompt, ``[表情包：开心] <path>`` on the terminal, in a training sample.  The
selector writes the line and the examples read it back, so they must agree; so must the training
export and the live bot (iron rule 8).  :class:`StickerLabelBook` is the one place the table is
read for the running program:

* ``label_md5(md5)`` is what an example of her real history needs (her messages name a sticker by
  its MD5); ``label(sha256)`` is what the channel has, the digest of the file it is about to show
  (the file of the library - or, once the setting ``stickers.gif_send_mode`` makes the bot send
  the still first frame of a GIF, that picture, D-697);
* the table is read once and kept, and read again when ``settings.state_version`` moves - the
  counter every change that the running application must hear about moves (R-ARCH-006.2); the
  check costs one tiny read of one setting, so a label asked for a hundred times costs one
  library read;
* the labels are the tags the training export and the evaluation already use
  (:meth:`~twin.stickers.catalog.StickerCatalog.tag_lookup`, from :func:`label_of`): the
  context tags were made from her messages before the hold-out cutoff only, the visual tags
  from the picture alone, so a label shown in the view of a past moment tells nothing that was
  not already known to the pre-hold-out data (R-TRN-013, D-691).

One book serves a container (:func:`sticker_labels`), so the terminal, the live view, the
as-of view and the opener examples all see the same table, and it is loaded once.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from twin.stickers.catalog import StickerCatalog, label_of
from twin.storage.state import current_state_version

if TYPE_CHECKING:
    from twin.services import Services

EXTRAS_KEY = "sticker_labels"


@dataclass(frozen=True)
class _Table:
    """The labels as they were read, with the state version they were read at."""

    version: int
    by_md5: dict[str, str]
    by_sha256: dict[str, str]


class StickerLabelBook:
    """``md5`` / ``sha256`` -> the label of a library sticker (see the module description)."""

    def __init__(self, services: Services, catalog: StickerCatalog | None = None) -> None:
        self._services = services
        self._catalog = catalog or StickerCatalog(services)
        self._table: _Table | None = None
        self._lock = threading.Lock()

    def label_md5(self, md5: str) -> str | None:
        """The label of the sticker with this MD5, or ``None`` (not in the library, no tags)."""
        return self._current().by_md5.get(md5)

    def label(self, sha256: str) -> str | None:
        """The label of the sticker whose picture has this SHA-256, or ``None``."""
        return self._current().by_sha256.get(sha256)

    def reload(self) -> None:
        """Forget the table; the next question reads the library again."""
        with self._lock:
            self._table = None

    def _current(self) -> _Table:
        version = current_state_version(self._services.db)
        table = self._table
        if table is not None and table.version == version:
            return table
        with self._lock:
            table = self._table
            if table is None or table.version != version:
                table = self._read(version)
                self._table = table
            return table

    def _read(self, version: int) -> _Table:
        by_md5: dict[str, str] = {}
        by_sha256: dict[str, str] = {}
        for record in self._catalog.records(tagged=True):
            label = label_of(record)
            if label is None:
                continue
            by_md5[record.md5] = label
            if record.sha256:
                by_sha256[record.sha256] = label
        return _Table(version, by_md5, by_sha256)


def sticker_labels(services: Services) -> StickerLabelBook:
    """The label book of the container (made on first use, shared by everything that shows one)."""
    found = services.extras.get(EXTRAS_KEY)
    if isinstance(found, StickerLabelBook):
        return found
    book = StickerLabelBook(services)
    services.extras[EXTRAS_KEY] = book
    return book
