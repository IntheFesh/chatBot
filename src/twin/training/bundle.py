"""The encrypted training package (R-TRN-008, R-PRIV-003).

``build_bundle`` turns a validated dataset directory and a profile into one file that is safe to
put on a rented machine: a tar archive, compressed with zstd and sealed with a key derived from a
passphrase by scrypt (AES-256-GCM, :mod:`twin.training.bundle_crypto`).  The passphrase is an
argument, asked for on the terminal by the command and never stored.  Nothing readable is written
to disk on the way: the archive is compressed and encrypted while it is produced.

Contents of the archive::

    manifest.json               profile, versions (dataset, template, persona card, profile),
                                counts, epochs and the sha256 of every other file
    data/                       the ShareGPT splits, dataset_meta.json and dataset_info.json
    config/                     the five LLaMA-Factory YAML files (SFT, DPO when the bundle has
                                pairs, evaluation, loss, export) and profile.env
    autodl/                     the shell scripts and tools (the same files that are uploaded in
                                plain, so that the instance can verify them after decrypting)
    pylib/twin/...              the few ``twin`` modules the instance imports for the template
                                check

``plain_upload_files`` is what ``twin train remote upload`` puts on the instance next to the
encrypted file: the scripts, the pinned versions, ``profile.env`` and the decryption tool.  None
of it contains data, and it is needed before anything can be decrypted.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import zstandard

from twin.training import bundle_crypto, lf_template, versions
from twin.training.dataset_dir import DatasetDir, file_sha256
from twin.training.layout import (
    BUNDLE_MANIFEST,
    FILE_DATASET_INFO,
    FILE_DATASET_META,
    RemoteLayout,
)
from twin.training.profiles import TrainingProfile, epochs_for, profile_env
from twin.training.yaml_render import dataset_info, render_all, training_assets_dir

BUNDLE_SCHEMA: Final = 1
ZSTD_LEVEL: Final = 9
BUNDLE_SUFFIX: Final = ".bundle.enc"
EXECUTABLE_SUFFIXES: Final = (".sh", ".py")

# modules of ``twin`` the instance imports: the template constants and the template check
# (``setup.sh verify`` runs ``python -m twin.training.parity_check``, R-TRN-011)
PYLIB_MODULES: Final = ("lf_template", "parity_check")

Source = bytes | Path


class BundleError(ValueError):
    """The package cannot be built."""


@dataclass(frozen=True)
class BundleResult:
    """What ``build_bundle`` produced."""

    path: Path
    sha256: str
    size: int
    manifest: dict[str, Any]
    decrypt_script: Path


def _text(path: Path) -> bytes:
    """A script or module as bytes with LF line endings (a Windows checkout may have CRLF)."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def _script_files() -> dict[str, Source]:
    root = training_assets_dir() / "autodl"
    files: dict[str, Source] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        files[f"autodl/{path.relative_to(root).as_posix()}"] = _text(path)
    files["autodl/tools/bundle_crypto.py"] = _text(Path(bundle_crypto.__file__))
    return files


def plain_upload_files(profile: TrainingProfile, *, dpo_min_pairs: int = 200) -> dict[str, Source]:
    """``{path below the work directory: content}`` of the files uploaded without encryption."""
    files = _script_files()
    files["autodl/profile.env"] = profile_env(profile, dpo_min_pairs=dpo_min_pairs).encode("utf-8")
    return files


def _pylib_files() -> dict[str, Source]:
    files: dict[str, Source] = {
        "pylib/twin/__init__.py": b'"""Modules of the twin package that the instance imports."""\n',
        "pylib/twin/training/__init__.py": b'"""Template check modules."""\n',
    }
    package = Path(lf_template.__file__).parent
    for module in PYLIB_MODULES:
        files[f"pylib/twin/training/{module}.py"] = _text(package / f"{module}.py")
    return files


def _read(source: Source) -> bytes:
    return source if isinstance(source, bytes) else source.read_bytes()


def _entries(
    dataset: DatasetDir,
    profile: TrainingProfile,
    layout: RemoteLayout,
    train_samples: int,
    dpo_min_pairs: int,
) -> dict[str, Source]:
    entries: dict[str, Source] = {}
    for name in dataset.data_files():
        entries[f"data/{name}"] = dataset.file_path(name)
    entries[f"data/{FILE_DATASET_META}"] = dataset.file_path(FILE_DATASET_META)
    entries[f"data/{FILE_DATASET_INFO}"] = dataset_info(with_dpo=dataset.has_dpo).encode("utf-8")
    configs = render_all(
        profile, train_samples=train_samples, layout=layout, with_dpo=dataset.has_dpo
    )
    for name, text in configs.items():
        entries[f"config/{name}"] = text.encode("utf-8")
    entries.update(plain_upload_files(profile, dpo_min_pairs=dpo_min_pairs))
    entries["config/profile.env"] = profile_env(profile, dpo_min_pairs=dpo_min_pairs).encode(
        "utf-8"
    )
    entries.update(_pylib_files())
    return entries


def _manifest(
    dataset: DatasetDir,
    profile: TrainingProfile,
    layout: RemoteLayout,
    created_at: datetime,
    entries: Mapping[str, Source],
) -> dict[str, Any]:
    meta = dataset.meta
    files = []
    for path in sorted(entries):
        content = _read(entries[path])
        files.append(
            {"path": path, "sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
        )
    return {
        "schema": BUNDLE_SCHEMA,
        "created_at": created_at.isoformat(),
        "profile": profile.name,
        "base_model": profile.base_model,
        "gpu": profile.gpu,
        "method": profile.method,
        "dataset_version": meta.dataset_version,
        "template": lf_template.LF_TEMPLATE_NAME,
        "template_version": meta.template_version,
        "persona_version": meta.persona_version,
        "profile_version": meta.profile_version,
        "holdout_cutoff": meta.holdout_cutoff,
        "llamafactory_version": lf_template.LLAMAFACTORY_VERSION,
        "llama_cpp_tag": versions.LLAMA_CPP_TAG,
        "counts": meta.counts.model_dump(),
        "epochs": epochs_for(meta.counts.train),
        "has_dpo": dataset.has_dpo,
        "workdir": layout.workdir,
        "files": files,
    }


def _tar_info(name: str, size: int, created_at: datetime) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = int(created_at.timestamp())
    info.mode = (
        0o755 if name.endswith(EXECUTABLE_SUFFIXES) and name.startswith("autodl/") else 0o644
    )
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def _iter_members(
    entries: Mapping[str, Source], manifest: bytes, created_at: datetime
) -> Iterator[tuple[tarfile.TarInfo, bytes]]:
    yield _tar_info(BUNDLE_MANIFEST, len(manifest), created_at), manifest
    for name in sorted(entries):
        content = _read(entries[name])
        yield _tar_info(name, len(content), created_at), content


def build_bundle(
    dataset: DatasetDir,
    profile: TrainingProfile,
    *,
    passphrase: str,
    out_dir: Path,
    created_at: datetime,
    layout: RemoteLayout | None = None,
    dpo_min_pairs: int = 200,
    kdf_log_n: int = bundle_crypto.DEFAULT_LOG_N,
) -> BundleResult:
    """Write ``<out_dir>/<profile>-<dataset version>.bundle.enc`` and the decryption tool."""
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise BundleError("created_at needs a time zone")
    paths = layout or RemoteLayout()
    entries = _entries(dataset, profile, paths, dataset.meta.counts.train, dpo_min_pairs)
    manifest = _manifest(dataset, profile, paths, created_at, entries)
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{profile.name}-{dataset.meta.dataset_version}{BUNDLE_SUFFIX}"
    partial = target.with_name(target.name + ".part")
    try:
        with partial.open("wb") as raw:
            sealed = bundle_crypto.EncryptWriter(raw, passphrase, log_n=kdf_log_n)
            compressor = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
            with (
                compressor.stream_writer(sealed, closefd=False) as compressed,  # type: ignore[arg-type]
                tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as tar,
            ):
                for info, content in _iter_members(entries, manifest_bytes, created_at):
                    tar.addfile(info, io.BytesIO(content))
            sealed.close()
        os.replace(partial, target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    decrypt_script = out_dir / "decrypt_bundle.py"
    decrypt_script.write_bytes(
        (training_assets_dir() / "autodl" / "tools" / "decrypt_bundle.py").read_bytes()
    )
    (out_dir / "bundle_crypto.py").write_bytes(Path(bundle_crypto.__file__).read_bytes())
    return BundleResult(
        path=target,
        sha256=file_sha256(target),
        size=target.stat().st_size,
        manifest=manifest,
        decrypt_script=decrypt_script,
    )


def iter_bundle(path: Path, passphrase: str) -> Iterator[tuple[str, bytes]]:
    """Decrypt and unpack a package, member by member, without writing anything to disk."""
    with path.open("rb") as source:
        reader = bundle_crypto.DecryptReader(source, passphrase)
        decompressor = zstandard.ZstdDecompressor()
        with (
            decompressor.stream_reader(reader) as plain,  # type: ignore[arg-type]
            tarfile.open(fileobj=plain, mode="r|") as tar,
        ):
            for member in tar:
                extracted = tar.extractfile(member)
                if extracted is not None:
                    yield member.name, extracted.read()


def verify_bundle(path: Path, passphrase: str) -> dict[str, Any]:
    """Decrypt a package and check every file against its manifest; returns the manifest."""
    members = dict(iter_bundle(path, passphrase))
    if BUNDLE_MANIFEST not in members:
        raise BundleError("the package has no manifest")
    manifest: dict[str, Any] = json.loads(members[BUNDLE_MANIFEST].decode("utf-8"))
    listed = {entry["path"]: entry for entry in manifest["files"]}
    if set(listed) != set(members) - {BUNDLE_MANIFEST}:
        raise BundleError("the files of the package do not match its manifest")
    for name, entry in listed.items():
        content = members[name]
        if hashlib.sha256(content).hexdigest() != entry["sha256"] or len(content) != entry["size"]:
            raise BundleError(f"{name} does not match the manifest")
    return manifest
