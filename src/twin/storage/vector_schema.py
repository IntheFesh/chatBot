"""What the vector store may contain (R-STO-005).

The LanceDB index (round 05) stores only the embedding, the id of the source row
and non-sensitive metadata: a timestamp and a record kind.  Text never goes into
the vector store - the plaintext stays in the encrypted database and is looked up
by id.  Every record written to the index must pass :func:`validate_record`, which
rejects any column outside this schema, so a later change that tries to "cache the
text next to the vector" fails loudly.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from twin.clock import to_epoch

ALLOWED_COLUMNS = frozenset({"id", "vector", "at", "kind"})
_ID_RE = re.compile(r"^[0-9A-Za-z_-]{1,64}$")


class VectorKind(StrEnum):
    """Kinds of indexed rows (extended by the retrieval round if needed)."""

    MESSAGE = "message"
    WINDOW = "window"
    FACT = "fact"
    SUMMARY = "summary"


class VectorSchemaError(ValueError):
    """A record does not fit the vector store schema."""


@dataclass(frozen=True)
class VectorRow:
    """One indexed row: embedding, source id, timestamp, kind."""

    row_id: str
    vector: Sequence[float]
    at: datetime
    kind: VectorKind

    def to_record(self) -> dict[str, Any]:
        """The dictionary written to the index (epoch seconds for the timestamp)."""
        record = {
            "id": self.row_id,
            "vector": [float(x) for x in self.vector],
            "at": int(to_epoch(self.at)),
            "kind": self.kind.value,
        }
        validate_record(record)
        return record


def validate_record(record: Mapping[str, Any], *, dimension: int | None = None) -> None:
    """Raise :class:`VectorSchemaError` unless ``record`` holds only vector-store columns."""
    extra = set(record) - ALLOWED_COLUMNS
    if extra:
        raise VectorSchemaError(
            f"the vector store holds only {sorted(ALLOWED_COLUMNS)}; "
            f"refusing columns {sorted(extra)}"
        )
    missing = ALLOWED_COLUMNS - set(record)
    if missing:
        raise VectorSchemaError(f"record is missing columns {sorted(missing)}")
    row_id = record["id"]
    if not isinstance(row_id, str) or not _ID_RE.fullmatch(row_id):
        raise VectorSchemaError("id must be a short identifier of the source row, not text")
    vector = record["vector"]
    if (
        not isinstance(vector, Sequence)
        or isinstance(vector, str | bytes)
        or not vector
        or not all(isinstance(x, int | float) and math.isfinite(x) for x in vector)
    ):
        raise VectorSchemaError("vector must be a non-empty sequence of finite numbers")
    if dimension is not None and len(vector) != dimension:
        raise VectorSchemaError(f"vector has {len(vector)} dimensions, expected {dimension}")
    if not isinstance(record["at"], int) or isinstance(record["at"], bool):
        raise VectorSchemaError("at must be an integer epoch timestamp")
    if record["kind"] not in {kind.value for kind in VectorKind}:
        raise VectorSchemaError(f"unknown record kind {record['kind']!r}")
