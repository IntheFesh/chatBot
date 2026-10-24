"""What the vector store may contain (R-STO-005).

The LanceDB index stores only the embedding, the id of the source row and non-sensitive
metadata: a timestamp and a few closed-vocabulary or numeric columns.  Text never goes into
the vector store - the plaintext stays in the encrypted database and is looked up by id.
Every record written to a table must pass that table's :class:`VectorTableSchema`, which
rejects any column outside the schema, so a later change that tries to "cache the text next to
the vector" fails loudly.

Five schemas exist:

``GENERIC_SCHEMA``
    ``id``, ``vector``, ``at``, ``kind`` - the general layout;
``FACT_SCHEMA`` / ``SUMMARY_SCHEMA``
    the same four columns in the two tables of the memory (round 07), ``memory_facts`` and
    ``memory_summaries``; the row id is the id of the fact or summary, ``at`` the moment it
    became known (a summary: the end of its day), ``kind`` is fixed;
``STICKER_SCHEMA``
    the same four columns in a table of its own, ``sticker_descriptions`` (round 06): the vector
    of the description of each sticker, the row id being the sticker's MD5;
``WINDOW_SCHEMA``
    ``window_id``, ``vector``, ``reply_at_utc``, ``local_slot``, ``day_type`` - the real-reply
    example windows of the retrieval library (R-RET, round 05).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from twin.clock import to_epoch

VECTOR_COLUMN = "vector"
ALLOWED_COLUMNS = frozenset({"id", "vector", "at", "kind"})
SLOTS_PER_DAY = 96
DAY_TYPE_VALUES = ("workday", "weekend", "holiday")
_ID_RE = re.compile(r"^[0-9A-Za-z_-]{1,64}$")


class VectorKind(StrEnum):
    """Kinds of rows in the generic layout."""

    MESSAGE = "message"
    WINDOW = "window"
    FACT = "fact"
    SUMMARY = "summary"
    STICKER = "sticker"


class VectorSchemaError(ValueError):
    """A record does not fit the vector store schema."""


@dataclass(frozen=True)
class ExtraColumn:
    """A non-sensitive metadata column: its Arrow type name and a value check."""

    name: str
    arrow_type: str  # "int16" or "string"
    accepts: Callable[[Any], bool]
    expectation: str  # words for the error message


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class VectorTableSchema:
    """The columns one LanceDB table may hold."""

    name: str
    id_column: str
    time_column: str
    extras: tuple[ExtraColumn, ...] = field(default_factory=tuple)

    @property
    def columns(self) -> frozenset[str]:
        return frozenset(
            {self.id_column, VECTOR_COLUMN, self.time_column, *(e.name for e in self.extras)}
        )

    def validate(self, record: Mapping[str, Any], *, dimension: int | None = None) -> None:
        """Raise :class:`VectorSchemaError` unless ``record`` holds only this table's columns."""
        allowed = self.columns
        extra = set(record) - allowed
        if extra:
            raise VectorSchemaError(
                f"the vector store holds only {sorted(allowed)}; refusing columns {sorted(extra)}"
            )
        missing = allowed - set(record)
        if missing:
            raise VectorSchemaError(f"record is missing columns {sorted(missing)}")
        row_id = record[self.id_column]
        if not isinstance(row_id, str) or not _ID_RE.fullmatch(row_id):
            raise VectorSchemaError(
                f"{self.id_column} must be a short identifier of the source row, not text"
            )
        vector = record[VECTOR_COLUMN]
        if (
            not isinstance(vector, Sequence)
            or isinstance(vector, str | bytes)
            or not vector
            or not all(isinstance(x, int | float) and math.isfinite(x) for x in vector)
        ):
            raise VectorSchemaError("vector must be a non-empty sequence of finite numbers")
        if dimension is not None and len(vector) != dimension:
            raise VectorSchemaError(f"vector has {len(vector)} dimensions, expected {dimension}")
        if not _is_int(record[self.time_column]):
            raise VectorSchemaError(f"{self.time_column} must be an integer epoch timestamp")
        for column in self.extras:
            if not column.accepts(record[column.name]):
                raise VectorSchemaError(f"{column.name} must be {column.expectation}")


GENERIC_SCHEMA = VectorTableSchema(
    name="generic",
    id_column="id",
    time_column="at",
    extras=(
        ExtraColumn(
            "kind",
            "string",
            lambda value: value in {kind.value for kind in VectorKind},
            "a known record kind (unknown record kind refused)",
        ),
    ),
)

FACT_SCHEMA = VectorTableSchema(
    name="memory_facts",
    id_column="id",
    time_column="at",
    extras=(
        ExtraColumn(
            "kind",
            "string",
            lambda value: value == VectorKind.FACT.value,
            "the fact record kind",
        ),
    ),
)

SUMMARY_SCHEMA = VectorTableSchema(
    name="memory_summaries",
    id_column="id",
    time_column="at",
    extras=(
        ExtraColumn(
            "kind",
            "string",
            lambda value: value == VectorKind.SUMMARY.value,
            "the summary record kind",
        ),
    ),
)

STICKER_SCHEMA = VectorTableSchema(
    name="sticker_descriptions",
    id_column="id",
    time_column="at",
    extras=(
        ExtraColumn(
            "kind",
            "string",
            lambda value: value == VectorKind.STICKER.value,
            "the sticker record kind",
        ),
    ),
)

WINDOW_SCHEMA = VectorTableSchema(
    name="example_windows",
    id_column="window_id",
    time_column="reply_at_utc",
    extras=(
        ExtraColumn(
            "local_slot",
            "int16",
            lambda value: _is_int(value) and 0 <= value < SLOTS_PER_DAY,
            f"an integer 15-minute slot 0..{SLOTS_PER_DAY - 1}",
        ),
        ExtraColumn(
            "day_type",
            "string",
            lambda value: value in DAY_TYPE_VALUES,
            f"one of {', '.join(DAY_TYPE_VALUES)}",
        ),
    ),
)


@dataclass(frozen=True)
class VectorRow:
    """One row of the generic layout: embedding, source id, timestamp, kind."""

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
    """Raise :class:`VectorSchemaError` unless ``record`` is a valid generic-layout row."""
    GENERIC_SCHEMA.validate(record, dimension=dimension)
