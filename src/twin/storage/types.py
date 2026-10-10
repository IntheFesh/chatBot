"""Custom column types and field descriptors for encrypted storage (R-STO-002).

Design
------
AES-GCM associated data must contain ``(table, primary key, column)``, but a
SQLAlchemy ``TypeDecorator`` never sees the row it is binding.  The encryption
layer therefore has two cooperating halves:

* ``EncryptedText`` / ``EncryptedJSON`` are the *column types*.  They store the
  sealed bytes and **refuse anything that is not a** :class:`SealedBlob`, so a
  plaintext value can never reach an encrypted column;
* ``sealed_text()`` / ``sealed_json()`` are *descriptors* on the model.  Reading
  ``row.payload`` decrypts ``row.payload_ct`` and assigning ``row.payload = x``
  seals ``x`` with the row's own ``(table, pk, column)`` as associated data.

Models declare an encrypted field as a pair::

    payload_ct = encrypted_column("payload", json=True)
    payload = sealed_json()
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, Protocol, Self, overload

import orjson
from sqlalchemy import DateTime, LargeBinary
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import MappedColumn, Mapper, mapped_column
from sqlalchemy.types import TypeDecorator

from twin.storage.crypto import SealedBlob, build_aad, get_keyring

# --------------------------------------------------------------- timestamps


class UTCDateTime(TypeDecorator[datetime]):
    """Timezone-aware datetime stored as naive UTC (CLAUDE.md rule 5).

    Binding a naive datetime is an error; values read back are aware UTC.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("naive datetime cannot be stored; attach a timezone first")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC)


# ------------------------------------------------------------ column types


class _SealedColumn(TypeDecorator[SealedBlob]):
    """Stores sealed bytes; refuses unsealed input."""

    impl = LargeBinary
    cache_ok = True

    def process_bind_param(self, value: SealedBlob | None, dialect: Dialect) -> bytes | None:
        if value is None:
            return None
        if not isinstance(value, SealedBlob):
            raise TypeError(
                "encrypted columns accept only sealed values; assign through the model's "
                "plaintext property (sealed_text()/sealed_json()) instead of the raw column"
            )
        return bytes(value)

    def process_result_value(self, value: bytes | None, dialect: Dialect) -> SealedBlob | None:
        if value is None:
            return None
        return SealedBlob(value)


class EncryptedText(_SealedColumn):
    """Column holding AES-256-GCM sealed UTF-8 text."""

    cache_ok = True


class EncryptedJSON(_SealedColumn):
    """Column holding AES-256-GCM sealed JSON (serialised with orjson)."""

    cache_ok = True


def encrypted_column(
    name: str, *, json: bool = False, nullable: bool = False
) -> MappedColumn[SealedBlob]:
    """Declare the physical column for an encrypted field (see module docstring)."""
    column_type = EncryptedJSON() if json else EncryptedText()
    return mapped_column(name, column_type, nullable=nullable)


# ------------------------------------------------------------- descriptors


class _Codec[T](Protocol):
    def encode(self, value: T) -> bytes: ...

    def decode(self, data: bytes) -> T: ...


class _TextCodec:
    def encode(self, value: str) -> bytes:
        return value.encode("utf-8")

    def decode(self, data: bytes) -> str:
        return data.decode("utf-8")


class _JsonCodec:
    def encode(self, value: Any) -> bytes:
        return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)

    def decode(self, data: bytes) -> Any:
        return orjson.loads(data)


_PK_NAMES: dict[type[Any], tuple[str, ...]] = {}


def _pk_attr_names(model: type[Any]) -> tuple[str, ...]:
    names = _PK_NAMES.get(model)
    if names is None:
        mapper: Mapper[Any] = sa_inspect(model)
        names = tuple(mapper.get_property_by_column(col).key for col in mapper.primary_key)
        _PK_NAMES[model] = names
    return names


def row_aad(instance: object, column: str) -> bytes:
    """Associated data for ``column`` of a mapped ``instance``."""
    model = type(instance)
    table = getattr(model, "__tablename__", None)
    if not isinstance(table, str):
        raise TypeError(f"{model.__name__} is not a mapped table")
    pk_values = [getattr(instance, name, None) for name in _pk_attr_names(model)]
    if any(value is None for value in pk_values):
        raise ValueError(
            f"{model.__name__}: the primary key must be assigned before encrypted fields "
            "(it is part of the authenticated data)"
        )
    return build_aad(table, pk_values, column)


class SealedAttr[T]:
    """Descriptor exposing an encrypted ``<name>_ct`` column as plaintext ``<name>``."""

    def __init__(self, codec: _Codec[Any], *, optional: bool) -> None:
        self._codec = codec
        self._optional = optional
        self.name = ""
        self.ct_attr = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.ct_attr = f"{name}_ct"

    @overload
    def __get__(self, obj: None, owner: type | None = None) -> Self: ...

    @overload
    def __get__(self, obj: object, owner: type | None = None) -> T: ...

    def __get__(self, obj: object | None, owner: type | None = None) -> Self | T:
        if obj is None:
            return self
        blob = getattr(obj, self.ct_attr)
        if blob is None:
            if self._optional:
                return None  # type: ignore[return-value]
            raise ValueError(f"{self.name} has not been set")
        plain = get_keyring().open(blob, row_aad(obj, self.name))
        result: T = self._codec.decode(plain)
        return result

    def __set__(self, obj: object, value: T) -> None:
        if value is None and self._optional:
            setattr(obj, self.ct_attr, None)
            return
        plain = self._codec.encode(value)
        setattr(obj, self.ct_attr, get_keyring().seal(plain, row_aad(obj, self.name)))


def sealed_text() -> SealedAttr[str]:
    """Descriptor for a required encrypted text field."""
    return SealedAttr(_TextCodec(), optional=False)


def sealed_optional_text() -> SealedAttr[str | None]:
    """Descriptor for an optional encrypted text field (``None`` stores SQL NULL)."""
    return SealedAttr(_TextCodec(), optional=True)


@overload
def sealed_json(*, optional: Literal[False] = False) -> SealedAttr[Any]: ...


@overload
def sealed_json(*, optional: Literal[True]) -> SealedAttr[Any | None]: ...


def sealed_json(*, optional: bool = False) -> SealedAttr[Any] | SealedAttr[Any | None]:
    """Descriptor for an encrypted JSON field (``None`` stores SQL NULL if optional)."""
    return SealedAttr(_JsonCodec(), optional=optional)
