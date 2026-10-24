"""Application-side primary keys (ULID).

Primary keys are generated in the application, not by the database, because
the encryption layer binds ``(table, primary key, column)`` into the AES-GCM
associated data (R-STO-002): the key must exist before the row is written.

ULIDs are 128-bit: 48-bit millisecond timestamp (from the active clock, so
tests with a manual clock get deterministic ordering) plus 80 bits of
randomness, encoded as 26 Crockford base32 characters.  Within one process the
generator is strictly monotonic.
"""

from __future__ import annotations

import os
import threading

from twin.clock import get_clock

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ULID_LEN = 26
_RANDOM_BITS = 80
_RANDOM_MAX = (1 << _RANDOM_BITS) - 1

_lock = threading.Lock()
_last_ms = -1
_last_random = 0


def _encode(value: int) -> str:
    chars = []
    for _ in range(_ULID_LEN):
        chars.append(_ALPHABET[value & 31])
        value >>= 5
    return "".join(reversed(chars))


def new_id() -> str:
    """Return a new, lexicographically time-ordered ULID string."""
    global _last_ms, _last_random
    now_ms = int(get_clock().now_utc().timestamp() * 1000)
    with _lock:
        if now_ms > _last_ms:
            _last_ms = now_ms
            _last_random = int.from_bytes(os.urandom(10), "big")
        else:
            # same millisecond, or the clock moved backwards: stay monotonic
            _last_random += 1
            if _last_random > _RANDOM_MAX:
                _last_ms += 1
                _last_random = int.from_bytes(os.urandom(10), "big")
        value = (_last_ms << _RANDOM_BITS) | _last_random
    return _encode(value)


def is_valid_id(value: str) -> bool:
    """True if ``value`` looks like a ULID produced by :func:`new_id`."""
    return len(value) == _ULID_LEN and value[0] in "01234567" and all(c in _ALPHABET for c in value)
