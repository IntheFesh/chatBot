#!/usr/bin/env python3
"""Check an unpacked training package against its manifest (standard library only).

    verify_manifest.py --root DIR [--scripts DIR]

Every file listed in ``manifest.json`` must exist with the recorded size and sha256, and no other
file may be present.  With ``--scripts`` the ``autodl/`` files of the package must also be
identical to the scripts that were uploaded in plain (the ones that are about to run).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BLOCK = 1 << 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(BLOCK):
            digest.update(block)
    return digest.hexdigest()


def check(root: Path, scripts: Path | None) -> list[str]:
    problems: list[str] = []
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return ["manifest.json is missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    listed = {entry["path"]: entry for entry in manifest["files"]}
    present = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    for name in sorted(present - set(listed)):
        problems.append(f"{name} is not listed in the manifest")
    for name, entry in sorted(listed.items()):
        path = root / name
        if not path.is_file():
            problems.append(f"{name} is missing")
        elif path.stat().st_size != entry["size"] or sha256_of(path) != entry["sha256"]:
            problems.append(f"{name} does not match the manifest")
    if scripts is not None:
        for name, entry in sorted(listed.items()):
            if not name.startswith("autodl/"):
                continue
            uploaded = scripts / name.removeprefix("autodl/")
            if not uploaded.is_file() or sha256_of(uploaded) != entry["sha256"]:
                problems.append(f"{name} differs from the uploaded script")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--scripts", type=Path)
    args = parser.parse_args(argv)
    problems = check(args.root, args.scripts)
    for problem in problems:
        sys.stderr.write(f"error: {problem}\n")
    if problems:
        return 1
    sys.stdout.write("manifest verified\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
