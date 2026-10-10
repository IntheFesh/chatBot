"""The off-site copy of the backups (``ops.backup_mirror_dir``, R-OPS-006).

The folder is meant to be an external drive or a network location, so it must already exist: a
missing folder is a drive that is not plugged in, and creating it would only make a harmless-
looking directory on the wrong disk.  An unavailable mirror is reported with
:class:`MirrorUnavailableError` (the caller raises the ``backup_mirror_unavailable`` alert); it
never affects the local backup.

What goes there: every kept archive and its index, and the media-pool files the kept backups
need.  What is no longer kept locally is removed there too, so the mirror follows the same
policy.  ``pre-restore`` backups stay on this machine.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from twin.ops.backup.layout import (
    ARCHIVE_SUFFIX,
    PRE_RESTORE_PREFIX,
    SIDECAR_SUFFIX,
    is_archive,
    sidecar_name,
)

PROBE_NAME = ".twin-mirror-probe"


class MirrorUnavailableError(Exception):
    """The mirror folder cannot be used; ``code`` says why in one word."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class MirrorReport:
    copied: int
    removed: int


def check_mirror(directory: Path) -> None:
    """Raise :class:`MirrorUnavailableError` unless the folder exists and can be written to."""
    if not directory.is_dir():
        raise MirrorUnavailableError("missing")
    probe = directory / PROBE_NAME
    try:
        probe.write_bytes(b"ok")
        probe.unlink()
    except OSError:
        raise MirrorUnavailableError("not_writable") from None


def _copy(source: Path, target: Path) -> bool:
    """Copy unless the target already has the same size; atomic on the target side."""
    if target.is_file() and target.stat().st_size == source.stat().st_size:
        return False
    part = target.with_name(target.name + ".part")
    try:
        shutil.copyfile(source, part)
        os.replace(part, target)
    except OSError:
        part.unlink(missing_ok=True)
        raise
    return True


def sync_mirror(
    directory: Path,
    backups_dir: Path,
    pool_dir: Path,
    *,
    keep_archives: set[str],
    pool_shas: set[str],
) -> MirrorReport:
    """Make the mirror hold the kept archives and pool files, and nothing older."""
    check_mirror(directory)
    mirror_pool = directory / pool_dir.name
    mirror_pool.mkdir(exist_ok=True)
    copied = removed = 0
    try:
        for name in sorted(keep_archives):
            if name.startswith(PRE_RESTORE_PREFIX):
                continue
            for item in (name, sidecar_name(name)):
                source = backups_dir / item
                if source.is_file():
                    copied += int(_copy(source, directory / item))
        for sha in sorted(pool_shas):
            source = pool_dir / f"{sha}.enc"
            if source.is_file():
                copied += int(_copy(source, mirror_pool / source.name))
    except OSError:
        raise MirrorUnavailableError("copy_failed") from None
    keep_files = {n for n in keep_archives if not n.startswith(PRE_RESTORE_PREFIX)}
    keep_files |= {sidecar_name(n) for n in keep_files}
    for entry in directory.iterdir():
        indexed = is_archive(entry.name) or entry.name.endswith(SIDECAR_SUFFIX)
        if entry.is_file() and indexed and entry.name not in keep_files:
            entry.unlink(missing_ok=True)
            removed += 1
    wanted_pool = {f"{sha}.enc" for sha in pool_shas}
    for entry in mirror_pool.iterdir():
        if entry.is_file() and entry.suffix == ".enc" and entry.name not in wanted_pool:
            entry.unlink(missing_ok=True)
            removed += 1
    return MirrorReport(copied, removed)


def archive_names(directory: Path) -> set[str]:
    """The archive names a folder holds (tests and ``backup list`` use it)."""
    if not directory.is_dir():
        return set()
    return {
        entry.name
        for entry in directory.iterdir()
        if entry.is_file() and entry.name.endswith(ARCHIVE_SUFFIX)
    }
