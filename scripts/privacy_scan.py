"""Privacy scan of tracked files (R-PRIV-001, CLAUDE.md rule 6).

Looks for traces of real data that must never be committed: WeChat ids, mainland
mobile numbers, ID card numbers (valid check digit), bank card numbers (Luhn),
fragments of a WeChat ``messages.json`` export, and API-key / private-key
shapes.  Findings are reported as ``file:line: kind`` - the matched text is never
printed.

Every file is read in pieces of at most :data:`PIECE_BYTES` (a longer line in pieces that
overlap), so a file of any size is scanned to its end with bounded memory.  Files over 5 MB used
to be skipped without a word - and a real ``messages.json`` export is that big (F9, D-770).

Usage::

    uv run python scripts/privacy_scan.py              # all files tracked by git
    uv run python scripts/privacy_scan.py a.py b.md    # pre-commit passes file names

Test data must be synthetic and generated at run time; fixtures that need a
pattern-shaped string build it from parts so that this scan stays clean.
"""

from __future__ import annotations

import argparse
import codecs
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

ROOT = Path(__file__).resolve().parent.parent
PIECE_BYTES = 1024 * 1024
"""The most of one line held at a time; a longer line is scanned in overlapping pieces."""
OVERLAP_CHARS = 256
"""How much of the previous piece of a long line is scanned again with the next one."""
SKIP_NAMES = {"uv.lock"}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".enc", ".db", ".lock"}

_B = r"(?<![0-9A-Za-z])"  # token boundaries: neighbours may not be alphanumeric
_E = r"(?![0-9A-Za-z])"
WXID = re.compile(r"\bwxid_[A-Za-z0-9_-]{6,}\b|\b[A-Za-z0-9_-]{6,}@chatroom\b")
CN_MOBILE = re.compile(_B + r"(?:\+?86[- ]?)?1[3-9]\d{9}" + _E)
ID_CARD = re.compile(
    _B + r"[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]" + _E
)
BANK_CARD = re.compile(_B + r"\d{16,19}" + _E)
SECRET_KEY = re.compile(r"\bsk-[A-Za-z0-9]{20,}\b|-----BEGIN [A-Z ]*PRIVATE KEY-----")
EXPORT_KEYS = ("createTimeText", "senderUsername", "isSent", "localId", "sortSeq", "renderType")
EXPORT_KEY_RE = {key: re.compile(rf'"{key}"\s*:') for key in EXPORT_KEYS}
EXPORT_ANY_KEY = re.compile(r'"(?:' + "|".join(EXPORT_KEYS) + r')"\s*:')
EXPORT_THRESHOLD = 3

_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_ID_CHECK = "10X98765432"


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    kind: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.kind}"


def valid_id_card(number: str) -> bool:
    if len(number) != 18 or not number[:17].isdigit():
        return False
    total = sum(int(d) * w for d, w in zip(number[:17], _ID_WEIGHTS, strict=True))
    return _ID_CHECK[total % 11] == number[17].upper()


def luhn(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


class BinaryFile(Exception):
    """The file holds a NUL byte: not text, not scanned (as before)."""


def line_kinds(line: str) -> list[str]:
    """The kinds of sensitive data in one line (or one piece of a long line)."""
    kinds: list[str] = []
    if WXID.search(line):
        kinds.append("wxid")
    if CN_MOBILE.search(line):
        kinds.append("mobile number")
    if any(valid_id_card(m.group(0)) for m in ID_CARD.finditer(line)):
        kinds.append("ID card number")
    if any(luhn(m.group(0)) for m in BANK_CARD.finditer(line)):
        kinds.append("bank card number (Luhn)")
    if SECRET_KEY.search(line):
        kinds.append("API key / private key")
    return kinds


def scan_lines(path: str, lines: Iterable[tuple[int, str]]) -> list[Finding]:
    """Findings for numbered lines; several pieces of one line may share its number."""
    findings: list[Finding] = []
    reported: set[tuple[int, str]] = set()
    export_keys: set[str] = set()
    first_export_line: int | None = None
    for number, line in lines:
        for kind in line_kinds(line):
            if (number, kind) not in reported:
                reported.add((number, kind))
                findings.append(Finding(path, number, kind))
        if not EXPORT_ANY_KEY.search(line):
            continue
        keys = {key for key, pattern in EXPORT_KEY_RE.items() if pattern.search(line)}
        if keys:
            export_keys |= keys
            if first_export_line is None:
                first_export_line = number
    if len(export_keys) >= EXPORT_THRESHOLD:
        findings.append(
            Finding(path, first_export_line or 1, "WeChat export (messages.json) fragment")
        )
    return findings


def scan_text(path: str, text: str) -> list[Finding]:
    """Findings for one file's text."""
    return scan_lines(path, enumerate(text.splitlines(), start=1))


def numbered_pieces(handle: BinaryIO) -> Iterator[tuple[int, str]]:
    """The lines of a file as ``(number, text)``, a very long line as several pieces.

    The file is read :data:`PIECE_BYTES` at a time.  A line that grows past that without ending
    is handed on in pieces, each further piece together with the last :data:`OVERLAP_CHARS`
    characters of the one before, so a match that straddles two reads is still seen and memory
    stays bounded.  Raises :class:`BinaryFile` at a NUL byte.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
    number = 1
    pending = ""  # the start of the current line, not handed on yet
    overlap = ""  # the end of the part of the current line that was handed on
    while block := handle.read(PIECE_BYTES):
        if b"\0" in block:
            raise BinaryFile
        *complete, pending = (pending + decoder.decode(block)).split("\n")
        for line in complete:
            yield number, overlap + line
            number += 1
            overlap = ""
        if len(pending) > PIECE_BYTES:  # a long line: scan what there is, keep its tail
            piece = overlap + pending
            yield number, piece
            overlap, pending = piece[max(len(piece) - OVERLAP_CHARS, 0) :], ""
    pending += decoder.decode(b"", final=True)
    if pending or overlap:
        yield number, overlap + pending


def tracked_files(root: Path) -> list[Path]:
    """Files git tracks plus new files that are not ignored (what a commit could include)."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return [p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]
    names = [name for name in result.stdout.decode("utf-8").split("\0") if name]
    return [root / name for name in names]


def scan_files(paths: Iterable[Path], root: Path) -> Iterator[Finding]:
    for path in paths:
        if path.name in SKIP_NAMES or path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
            continue
        try:
            display = str(path.relative_to(root))
        except ValueError:
            display = str(path)
        try:
            with path.open("rb") as handle:
                findings = scan_lines(display, numbered_pieces(handle))
        except (OSError, BinaryFile):
            continue
        yield from findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, help="files to scan (default: git ls-files)")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    paths = [p if p.is_absolute() else Path.cwd() / p for p in args.files] or tracked_files(
        args.root
    )
    findings = list(scan_files(paths, args.root))
    if findings:
        sys.stdout.write("privacy scan FAILED: possible real data in tracked files\n")
        for finding in findings:
            sys.stdout.write(f"  {finding.render()}\n")
        return 1
    sys.stdout.write(f"privacy scan passed ({len(paths)} files)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
