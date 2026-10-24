"""Path handling for export directories (R-IMP-001, CLAUDE.md section 7).

Export folders carry Chinese nicknames, emoji and punctuation in their names, and on
Windows the full path of a media file can pass the classic 260 character limit.  This
module keeps all of that in one place:

* names are compared in Unicode NFC (a file system may hold the decomposed form);
* relative paths found *inside* the export (``offlineMedia[].path``) are checked so they
  can never leave the export root;
* very long Windows paths get the extended-length prefix.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import unicodedata
from pathlib import Path

LONG_PATH_THRESHOLD = 240
_DRIVE = re.compile(r"^[A-Za-z]:")
_SEQUENCE = re.compile(r"^(\d+)(?=_)")


def nfc(text: str) -> str:
    """Unicode normalisation form C."""
    return unicodedata.normalize("NFC", text)


def extended_length_path(text: str) -> str:
    """``text`` as a Windows extended-length path (``\\\\?\\`` prefix); other input is unchanged."""
    if text.startswith("\\\\?\\"):
        return text
    if text.startswith(("\\\\", "//")):
        return "\\\\?\\UNC\\" + text[2:].replace("/", "\\")
    if _DRIVE.match(text) and len(text) > 2 and text[2] in "\\/":
        return "\\\\?\\" + text.replace("/", "\\")
    return text


def long_path(path: Path, *, win32: bool | None = None) -> Path:
    """Make ``path`` usable beyond 260 characters on Windows; a no-op elsewhere."""
    on_windows = sys.platform == "win32" if win32 is None else win32
    if not on_windows:
        return path
    text = os.path.abspath(path)
    if len(text) < LONG_PATH_THRESHOLD:
        return path
    return Path(extended_length_path(text))


def split_relative(relative: str) -> list[str] | None:
    """Components of an export-relative path, or ``None`` if it is unsafe.

    Backslashes are treated as separators (the export may come from Windows).  Absolute
    paths, drive letters, NUL bytes and ``..`` components are refused.
    """
    if not relative or "\0" in relative:
        return None
    text = relative.replace("\\", "/")
    if text.startswith("/") or _DRIVE.match(text):
        return None
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        return None
    return parts


def _find_equivalent(directory: Path, name: str) -> Path | None:
    """The entry of ``directory`` whose NFC name equals NFC(``name``)."""
    wanted = nfc(name)
    try:
        with os.scandir(long_path(directory)) as entries:
            for entry in entries:
                if nfc(entry.name) == wanted:
                    return directory / entry.name
    except OSError:
        return None
    return None


def resolve_in_root(root: Path, relative: str) -> Path | None:
    """The existing file ``relative`` names below ``root``; ``None`` if absent or unsafe.

    Tries the path as written first, then component by component with Unicode
    normalisation differences ignored.  A result that resolves (through links) outside the
    root is treated as unsafe.
    """
    parts = split_relative(relative)
    if parts is None:
        return None
    candidate = root.joinpath(*parts)
    if not long_path(candidate).exists():
        current = root
        for part in parts:
            step = current / part
            if not long_path(step).exists():
                found = _find_equivalent(current, part)
                if found is None:
                    return None
                step = found
            current = step
        candidate = current
    try:
        resolved = Path(os.path.realpath(long_path(candidate)))
        base = Path(os.path.realpath(long_path(root)))
    except OSError:
        return None
    # on Windows realpath may return the extended-length form for one side only
    if not _is_inside(_strip_extended(resolved), _strip_extended(base)):
        return None
    return candidate if long_path(candidate).is_file() else None


def _strip_extended(path: Path) -> Path:
    text = str(path)
    if text.startswith("\\\\?\\UNC\\"):
        return Path("\\\\" + text[8:])
    if text.startswith("\\\\?\\"):
        return Path(text[4:])
    return path


def _is_inside(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def mask_dir_name(dir_name: str, nickname: str | None, position: int) -> str:
    """``<sequence>_<nickname length>_<first 4 hash characters>`` for a conversation folder.

    Used wherever a conversation folder has to be mentioned in a report: it identifies the
    folder for the person who owns the export without revealing the nickname or the id.
    """
    normalized = nfc(dir_name)
    match = _SEQUENCE.match(normalized)
    sequence = match.group(1) if match else f"{position:03d}"
    length = len(nfc(nickname)) if nickname else len(normalized)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:4]
    return f"{sequence}_{length}_{digest}"
