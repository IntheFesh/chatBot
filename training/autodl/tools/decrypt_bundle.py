#!/usr/bin/env python3
"""Decrypt the training package (standard library + cryptography only).

    decrypt_bundle.py --in bundle.enc --out bundle.tar.zst
    decrypt_bundle.py --in bundle.enc --out - | zstd -d -c | tar -x

The passphrase is read from the first line of standard input (never from an argument or the
environment, so it appears in no process list and no file).  With ``--out -`` the decrypted
archive goes to standard output, so decrypt.sh can unpack it without ever writing the archive
to disk.  The format and the key derivation are in ``bundle_crypto.py`` next to this script.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bundle_crypto


def decrypt_to_file(source: Path, target: Path, passphrase: str) -> int:
    partial = target.with_name(target.name + ".part")
    try:
        with source.open("rb") as stream, partial.open("wb") as out:
            size = bundle_crypto.decrypt_stream(stream, out, passphrase)
        os.replace(partial, target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", required=True, type=Path)
    parser.add_argument("--out", dest="target", required=True)
    args = parser.parse_args(argv)

    passphrase = sys.stdin.readline().rstrip("\r\n")
    if not passphrase:
        sys.stderr.write("error: the passphrase must be given on standard input\n")
        return 2
    try:
        if args.target == "-":
            with args.source.open("rb") as stream:
                bundle_crypto.decrypt_stream(stream, sys.stdout.buffer, passphrase)
            sys.stdout.buffer.flush()
        else:
            size = decrypt_to_file(args.source, Path(args.target), passphrase)
            sys.stdout.write(f"decrypted {size} bytes to {args.target}\n")
    except (bundle_crypto.BundleDecryptionError, OSError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
