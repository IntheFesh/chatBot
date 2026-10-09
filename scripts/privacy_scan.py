"""Privacy scan of tracked files (R-PRIV-001, CLAUDE.md rule 6).

Looks for traces of real data that must never be committed: WeChat ids, mainland
mobile numbers, ID card numbers (valid check digit), bank card numbers (Luhn),
fragments of a WeChat ``messages.json`` export, and API-key / private-key
shapes.  Findings are reported as ``file:line: kind`` - the matched text is never
printed.

Usage::

    uv run python scripts/privacy_scan.py              # all files tracked by git
    uv run python scripts/privacy_scan.py a.py b.md    # pre-commit passes file names

Test data must be synthetic and generated at run time; fixtures that need a
pattern-shaped string build it from parts so that this scan stays clean.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 5 * 1024 * 1024
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


def scan_text(path: str, text: str) -> list[Finding]:
    """Findings for one file's text."""
    findings: list[Finding] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if WXID.search(line):
            findings.append(Finding(path, number, "wxid"))
        if CN_MOBILE.search(line):
            findings.append(Finding(path, number, "mobile number"))
        if any(valid_id_card(m.group(0)) for m in ID_CARD.finditer(line)):
            findings.append(Finding(path, number, "ID card number"))
        if any(luhn(m.group(0)) for m in BANK_CARD.finditer(line)):
            findings.append(Finding(path, number, "bank card number (Luhn)"))
        if SECRET_KEY.search(line):
            findings.append(Finding(path, number, "API key / private key"))
    present = [key for key, pattern in EXPORT_KEY_RE.items() if pattern.search(text)]
    if len(present) >= EXPORT_THRESHOLD:
        first = next(
            (
                number
                for number, line in enumerate(text.splitlines(), start=1)
                if any(EXPORT_KEY_RE[key].search(line) for key in present)
            ),
            1,
        )
        findings.append(Finding(path, first, "WeChat export (messages.json) fragment"))
    return findings


def tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=False
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
            if path.stat().st_size > MAX_BYTES:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data:
            continue
        try:
            display = str(path.relative_to(root))
        except ValueError:
            display = str(path)
        yield from scan_text(display, data.decode("utf-8", errors="ignore"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, help="files to scan (default: git ls-files)")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    paths = [p if p.is_absolute() else Path.cwd() / p for p in args.files] or tracked_files(args.root)
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
