"""Where the backup files are and what they are called (R-OPS-006).

::

    data/backups/
        twin-2026-10-09.bak.enc          the archive of the day (local date of the bot)
        twin-2026-10-09.media.zst        its list of media files (sha256, size, key id) and keys
        pre-restore-20261009T120000Z.bak.enc   made by ``twin backup restore`` first (UTC)
        media-pool/<sha256>.enc          every media file any kept backup lists

    data/twin.db.replaced-20261009T120000      what a restore moved aside (see
    data/vectors.replaced-20261009T120000      :func:`replaced_path`); normally deleted at once

The list next to an archive is only an index (hashes and numbers, no content): it lets the
retention work out which pool files and which keys are still needed without decrypting the
archives.  The manifest inside the archive is the authority.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import zstandard

from twin.ops.backup.archive import Manifest

ARCHIVE_SUFFIX = ".bak.enc"
SIDECAR_SUFFIX = ".media.zst"
DAILY_PREFIX = "twin-"
PRE_RESTORE_PREFIX = "pre-restore-"
POOL_DIRNAME = "media-pool"
REPLACED_MARK = ".replaced-"


def replaced_path(path: Path, tag: str) -> Path:
    """Where ``twin backup restore`` moves ``path`` aside while it puts the restored data in place.

    The one spelling of that name (R18-A5, D-769): the restore deletes these after a success, and
    ``twin purge --all`` deletes the ones a restore could not (a file another program held open).
    """
    return path.with_name(f"{path.name}{REPLACED_MARK}{tag}")


def restore_targets(db_path: Path, vectors_dir: Path) -> list[Path]:
    """What a restore replaces: the database, its two journal files and the vector index."""
    return [db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm"), vectors_dir]


def moved_aside(originals: Iterable[Path]) -> list[Path]:
    """The :func:`replaced_path` copies of ``originals`` that are still there, in name order."""
    found: list[Path] = []
    for original in originals:
        folder = original.parent
        if not folder.is_dir():
            continue
        prefix = f"{original.name}{REPLACED_MARK}"
        found.extend(sorted(entry for entry in folder.iterdir() if entry.name.startswith(prefix)))
    return found


def archive_name(kind: str, local_date: str, at: datetime) -> str:
    """File name of a backup: one per local day, or one per restore."""
    if kind == "pre_restore":
        return f"{PRE_RESTORE_PREFIX}{at:%Y%m%dT%H%M%SZ}{ARCHIVE_SUFFIX}"
    return f"{DAILY_PREFIX}{local_date}{ARCHIVE_SUFFIX}"


def sidecar_name(archive: str) -> str:
    return archive.removesuffix(ARCHIVE_SUFFIX) + SIDECAR_SUFFIX


def is_archive(name: str) -> bool:
    return name.endswith(ARCHIVE_SUFFIX) and not name.endswith(".part")


def kind_of(name: str) -> str:
    return "pre_restore" if name.startswith(PRE_RESTORE_PREFIX) else "daily"


def date_of(name: str) -> str | None:
    """The local date in a ``twin-<date>.bak.enc`` name."""
    if not name.startswith(DAILY_PREFIX):
        return None
    stem = name.removeprefix(DAILY_PREFIX).removesuffix(ARCHIVE_SUFFIX)
    return stem if len(stem) == 10 else None


@dataclass(frozen=True)
class Sidecar:
    """The index next to an archive."""

    key_ids: tuple[int, ...]
    media: tuple[str, ...]


def write_sidecar(path: Path, manifest: Manifest) -> None:
    payload = {
        "key_ids": sorted(manifest.key_ids),
        "media": [entry.sha256 for entry in manifest.media],
    }
    blob = zstandard.ZstdCompressor(level=3).compress(json.dumps(payload).encode("utf-8"))
    part = path.with_name(path.name + ".part")
    part.write_bytes(blob)
    os.replace(part, path)


def read_sidecar(path: Path) -> Sidecar | None:
    """The index, or ``None`` if it is missing or unreadable."""
    try:
        data = json.loads(zstandard.ZstdDecompressor().decompress(path.read_bytes()))
        return Sidecar(
            tuple(int(item) for item in data["key_ids"]),
            tuple(str(item) for item in data["media"]),
        )
    except (OSError, ValueError, KeyError, TypeError, zstandard.ZstdError):
        return None
