"""Outbound image allow list (R-SAFE-006, R-CH-006).

The only pictures the bot may send are stickers from the sticker library and the synthetic
test pictures the M0 probe generates.  The check is on the SHA-256 of the exact bytes about
to be sent, so a photo from ``media_assets`` (or anything else) cannot get through by being
renamed or relabelled.

* :class:`OutboundMediaPolicy` is the interface the channel asks;
* :class:`StickerAllowList` is the one-method interface the sticker library (round 03)
  implements;
* :class:`ProbeImageManifest` is the list of probe pictures, persisted in ``channel_state``;
* :class:`CompositeMediaPolicy` allows whatever any of its sources allows.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

from twin.channel.base import MediaNotAllowed
from twin.channel.state import ChannelStateStore

PROBE_IMAGES_KEY = "ilink.probe_images"
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class OutboundMediaPolicy(Protocol):
    """Decides whether bytes (identified by their SHA-256) may be sent as an image."""

    def allowed(self, sha256: str) -> bool: ...


class StickerAllowList(Protocol):
    """Implemented by the sticker library: is this hash one of its usable stickers?"""

    def allowed(self, sha256: str) -> bool: ...


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_media_allowed(policy: OutboundMediaPolicy, data: bytes) -> str:
    """Return the SHA-256 of ``data`` or raise :class:`MediaNotAllowed`."""
    digest = sha256_hex(data)
    if not policy.allowed(digest):
        raise MediaNotAllowed(
            "this image is neither a sticker from the library nor a probe test picture "
            f"(sha256 {digest[:12]}...)"
        )
    return digest


class ProbeImageManifest:
    """SHA-256 list of the pictures the channel probe generated (stored in ``channel_state``)."""

    def __init__(self, state: ChannelStateStore) -> None:
        self._state = state

    def register(self, sha256: str) -> None:
        """Add a probe picture; only a well-formed lowercase SHA-256 is accepted."""
        if not _SHA_RE.fullmatch(sha256):
            raise ValueError("a probe image is registered by its lowercase hex SHA-256")
        with self._state.transaction() as tx:
            current = list(tx.get(PROBE_IMAGES_KEY, []))
            if sha256 not in current:
                current.append(sha256)
                tx.put(PROBE_IMAGES_KEY, current)

    def register_bytes(self, data: bytes) -> str:
        digest = sha256_hex(data)
        self.register(digest)
        return digest

    def hashes(self) -> list[str]:
        return list(self._state.get(PROBE_IMAGES_KEY, []))

    def allowed(self, sha256: str) -> bool:
        return sha256 in self.hashes()


class CompositeMediaPolicy:
    """Allows an image if the sticker library or the probe manifest allows it."""

    def __init__(self, sources: Sequence[StickerAllowList]) -> None:
        self._sources = tuple(sources)

    def allowed(self, sha256: str) -> bool:
        return any(source.allowed(sha256) for source in self._sources)


class DenyAllMediaPolicy:
    """Allows nothing: the safe default when no sticker library is connected yet."""

    def allowed(self, sha256: str) -> bool:
        return False
