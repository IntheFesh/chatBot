#!/usr/bin/env python3
"""Size and sha256 of a file, or of its first bytes (standard library only).

    file_hash.py --path FILE [--length N]

Prints one JSON object ``{"size": ..., "sha256": ...}``; ``size`` is the size of the file and the
hash covers the first ``N`` bytes (the whole file without ``--length``).  ``twin train remote``
uses it to check a partly uploaded file before it continues, and a finished upload before it
moves the file into place.  A file that does not exist is reported as ``{"size": null,
"sha256": null}``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BLOCK = 1 << 20


def hash_file(path: Path, length: int | None) -> dict[str, object]:
    if not path.is_file():
        return {"size": None, "sha256": None}
    size = path.stat().st_size
    remaining = size if length is None else min(length, size)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while remaining > 0:
            block = stream.read(min(BLOCK, remaining))
            if not block:
                break
            digest.update(block)
            remaining -= len(block)
    return {"size": size, "sha256": digest.hexdigest()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--length", type=int)
    args = parser.parse_args(argv)
    sys.stdout.write(json.dumps(hash_file(args.path, args.length)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
