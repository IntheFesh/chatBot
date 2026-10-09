"""The vector index: a thin layer over LanceDB (R-STO-005).

Every table of the project that needs nearest-neighbour search uses this layer: the example
windows of this round (:data:`~twin.storage.vector_schema.WINDOW_SCHEMA`), the memory of round 07
(:data:`~twin.storage.vector_schema.GENERIC_SCHEMA`) and the sticker descriptions of round 06
(:data:`~twin.storage.vector_schema.STICKER_SCHEMA`), each in its own table below
``data/vectors/``.

What a table holds is fixed by its :class:`~twin.storage.vector_schema.VectorTableSchema`: the
id of the source row, the vector, a timestamp and a few non-sensitive columns.  Every record
passes the schema before it is written, so text cannot reach the index.  Next to each table a
small ``<table>.meta.json`` records which encoding (model, size, weights hash) the vectors were
made with; a different encoding is refused instead of being mixed in (R-RET-006).

LanceDB itself is imported on first use, so building the application stays fast.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from twin.storage.vector_schema import VECTOR_COLUMN, VectorTableSchema

UPSERT_CHUNK = 2000
DELETE_CHUNK = 500
_SAFE_ID = re.compile(r"^[0-9A-Za-z_-]{1,64}$")


class IndexMismatchError(RuntimeError):
    """The index was made with another encoding than the one now configured."""


@dataclass(frozen=True)
class IndexMeta:
    """Which encoding the vectors of a table were made with."""

    table: str
    encoding: str  # the embed_version stored with every indexed row
    model: str
    dimension: int
    revision: str
    weights_sha256: str
    written_at: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> IndexMeta:
        return cls(
            table=str(data["table"]),
            encoding=str(data["encoding"]),
            model=str(data["model"]),
            dimension=int(data["dimension"]),
            revision=str(data.get("revision", "")),
            weights_sha256=str(data.get("weights_sha256", "")),
            written_at=str(data.get("written_at", "")),
        )


@dataclass(frozen=True)
class VectorHit:
    """One nearest neighbour: id, cosine similarity and the stored metadata."""

    id: str
    similarity: float
    at: int
    vector: NDArray[np.float32]
    extras: Mapping[str, Any]


def _quoted_ids(ids: Iterable[str]) -> str:
    cleaned = list(ids)
    for value in cleaned:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("only identifiers of source rows can be deleted by id")
    return ", ".join(f"'{value}'" for value in cleaned)


class VectorTable:
    """One LanceDB table with a fixed, text-free schema."""

    def __init__(self, directory: Path, schema: VectorTableSchema) -> None:
        self._directory = directory
        self._schema = schema
        self._connection: Any = None

    @property
    def schema(self) -> VectorTableSchema:
        return self._schema

    @property
    def meta_path(self) -> Path:
        return self._directory / f"{self._schema.name}.meta.json"

    # ---------------------------------------------------------------- plumbing

    def _db(self) -> Any:
        if self._connection is None:
            import lancedb

            self._directory.mkdir(parents=True, exist_ok=True)
            self._connection = lancedb.connect(str(self._directory))
        return self._connection

    def exists(self) -> bool:
        if not self._directory.is_dir():
            return False
        return self._schema.name in self._db().list_tables().tables

    def _open(self) -> Any:
        return self._db().open_table(self._schema.name)

    def _arrow_schema(self, dimension: int) -> Any:
        import pyarrow as pa

        arrow_types = {"int16": pa.int16(), "string": pa.string()}
        fields = [
            pa.field(self._schema.id_column, pa.string()),
            pa.field(VECTOR_COLUMN, pa.list_(pa.float32(), dimension)),
            pa.field(self._schema.time_column, pa.int64()),
        ]
        fields.extend(pa.field(c.name, arrow_types[c.arrow_type]) for c in self._schema.extras)
        return pa.schema(fields)

    def dimension(self) -> int | None:
        """Vector size of the existing table, or ``None`` before it was created."""
        if not self.exists():
            return None
        size = self._open().schema.field(VECTOR_COLUMN).type.list_size
        return int(size)

    def count(self) -> int:
        if not self.exists():
            return 0
        return int(self._open().count_rows())

    def columns(self) -> list[str]:
        """Names of the stored columns (always the schema's, and nothing else)."""
        return list(self._open().schema.names) if self.exists() else []

    def rows(self) -> list[dict[str, Any]]:
        """Every stored row as a dictionary (for checks and repairs; large tables are big)."""
        return list(self._open().to_arrow().to_pylist()) if self.exists() else []

    # ----------------------------------------------------------------- writing

    def create(self, dimension: int) -> None:
        """Create the empty table (nothing happens if it exists with this vector size)."""
        existing = self.dimension()
        if existing is not None:
            if existing != dimension:
                raise IndexMismatchError(
                    f"the {self._schema.name} index holds {existing}-dimensional vectors, "
                    f"not {dimension}; rebuild it"
                )
            return
        self._db().create_table(self._schema.name, schema=self._arrow_schema(dimension))

    def upsert(self, records: Sequence[Mapping[str, Any]], *, dimension: int) -> None:
        """Insert rows, replacing rows with the same id; every record passes the schema."""
        for record in records:
            self._schema.validate(record, dimension=dimension)
        if not records:
            return
        self.create(dimension)
        table = self._open()
        id_column = self._schema.id_column
        for start in range(0, len(records), UPSERT_CHUNK):
            chunk = [dict(record) for record in records[start : start + UPSERT_CHUNK]]
            (
                table.merge_insert(id_column)
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(chunk)
            )

    def delete_ids(self, ids: Sequence[str]) -> None:
        """Remove the rows of these source ids (missing ids are ignored)."""
        if not ids or not self.exists():
            return
        table = self._open()
        for start in range(0, len(ids), DELETE_CHUNK):
            part = ids[start : start + DELETE_CHUNK]
            table.delete(f"{self._schema.id_column} IN ({_quoted_ids(part)})")

    def delete_from(self, epoch: int) -> int:
        """Remove the rows whose timestamp is ``epoch`` or later; returns how many went."""
        if not self.exists():
            return 0
        table = self._open()
        before = int(table.count_rows())
        table.delete(f"{self._schema.time_column} >= {int(epoch)}")
        return before - int(table.count_rows())

    def optimize(self) -> None:
        """Merge small files and drop old versions (call after a long run of writes)."""
        if self.exists():
            self._open().optimize()

    def drop(self) -> None:
        if self.exists():
            self._db().drop_table(self._schema.name)

    # ----------------------------------------------------------------- reading

    def search(
        self, vector: NDArray[np.float32], limit: int, *, at_or_before: int | None = None
    ) -> list[VectorHit]:
        """The ``limit`` rows nearest to ``vector`` (cosine); ``at_or_before`` keeps older rows."""
        if limit <= 0 or not self.exists():
            return []
        table = self._open()
        query = table.search(vector.tolist(), vector_column_name=VECTOR_COLUMN).metric("cosine")
        if at_or_before is not None:
            bound = f"{self._schema.time_column} <= {int(at_or_before)}"
            query = query.where(bound, prefilter=True)
        rows = query.limit(limit).to_list()
        extras = [column.name for column in self._schema.extras]
        hits = [
            VectorHit(
                id=str(row[self._schema.id_column]),
                similarity=1.0 - float(row["_distance"]),
                at=int(row[self._schema.time_column]),
                vector=np.asarray(row[VECTOR_COLUMN], dtype=np.float32),
                extras={name: row[name] for name in extras},
            )
            for row in rows
        ]
        hits.sort(key=lambda hit: -hit.similarity)
        return hits

    # ---------------------------------------------------------------- metadata

    def read_meta(self) -> IndexMeta | None:
        try:
            loaded = json.loads(self.meta_path.read_text(encoding="utf-8"))
            return IndexMeta.from_json(loaded)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def write_meta(self, meta: IndexMeta) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(
            json.dumps(meta.to_json(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def clear_meta(self) -> None:
        self.meta_path.unlink(missing_ok=True)


class VectorStore:
    """The ``data/vectors/`` folder: one :class:`VectorTable` per schema."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    @property
    def directory(self) -> Path:
        return self._directory

    def table(self, schema: VectorTableSchema) -> VectorTable:
        return VectorTable(self._directory, schema)
