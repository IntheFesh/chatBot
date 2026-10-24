"""Streaming access to the big ``messages.json`` (R-IMP-004).

``ijson`` with the C backend reads the file in small blocks; nothing here keeps more than
one message in memory.  Numbers are decoded as floats (not ``Decimal``) so a message can be
stored as JSON unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any, BinaryIO

import ijson

from twin.ingest.paths import long_path

UTF8_BOM = b"\xef\xbb\xbf"
MESSAGES_PREFIX = "messages.item"


def skip_bom(handle: BinaryIO) -> None:
    """Leave ``handle`` after a UTF-8 byte order mark, or at the start if there is none."""
    if handle.read(3) != UTF8_BOM:
        handle.seek(0)


def first_value(path: Path, prefix: str) -> Any:
    """The first value found at ``prefix`` (``None`` if absent), reading as little as possible."""
    with long_path(path).open("rb") as handle:
        skip_bom(handle)
        return next(iter(ijson.items(handle, prefix, use_float=True)), None)


def iter_messages(handle: BinaryIO, skip: int = 0) -> Iterator[Any]:
    """The elements of ``messages``, after skipping the first ``skip`` of them."""
    stream = ijson.items(handle, MESSAGES_PREFIX, use_float=True)
    return islice(stream, skip, None)
