"""The vector store may hold vectors, ids and non-sensitive metadata only (R-STO-005)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from twin.storage.vector_schema import (
    ALLOWED_COLUMNS,
    VectorKind,
    VectorRow,
    VectorSchemaError,
    validate_record,
)

AT = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)


def good() -> dict[str, object]:
    return {
        "id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "vector": [0.1, 0.2, 0.3],
        "at": 1760011200,
        "kind": "message",
    }


def test_a_row_converts_to_a_valid_record() -> None:
    record = VectorRow("01ARZ3NDEKTSV4RRFFQ69G5FAV", [1, 2, 3], AT, VectorKind.WINDOW).to_record()
    assert set(record) == ALLOWED_COLUMNS == {"id", "vector", "at", "kind"}
    assert record["at"] == int(AT.timestamp()) and record["vector"] == [1.0, 2.0, 3.0]
    validate_record(record, dimension=3)


@pytest.mark.parametrize("column", ["text", "content", "body", "caption", "metadata", "payload"])
def test_any_extra_column_such_as_plaintext_is_refused(column: str) -> None:
    record = good()
    record[column] = "她说的话"
    with pytest.raises(VectorSchemaError, match="refusing columns"):
        validate_record(record)


def test_missing_columns_are_refused() -> None:
    record = good()
    del record["kind"]
    with pytest.raises(VectorSchemaError, match="missing columns"):
        validate_record(record)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", "this is a whole sentence of text", "identifier"),
        ("id", "", "identifier"),
        ("id", 5, "identifier"),
        ("vector", [], "finite numbers"),
        ("vector", [float("nan")], "finite numbers"),
        ("vector", "text", "finite numbers"),
        ("vector", ["a"], "finite numbers"),
        ("at", 1.5, "epoch"),
        ("at", True, "epoch"),
        ("kind", "chat", "unknown record kind"),
    ],
)
def test_malformed_fields_are_refused(field: str, value: object, message: str) -> None:
    record = good()
    record[field] = value
    with pytest.raises(VectorSchemaError, match=message):
        validate_record(record)


def test_dimension_is_enforced_when_given() -> None:
    validate_record(good(), dimension=3)
    with pytest.raises(VectorSchemaError, match="3 dimensions, expected 512"):
        validate_record(good(), dimension=512)


def test_a_row_with_text_as_its_id_cannot_be_converted() -> None:
    with pytest.raises(VectorSchemaError):
        VectorRow("hello world, how are you", [0.1], AT, VectorKind.MESSAGE).to_record()
