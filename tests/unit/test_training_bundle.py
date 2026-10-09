"""R-TRN-008, R-PRIV-003: the dataset directory and the encrypted training package."""

from __future__ import annotations

import ast
import io
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from tests.support.training_data import sample, write_synthetic_dataset
from twin.training import bundle, bundle_crypto, lf_template
from twin.training.dataset_dir import (
    DatasetError,
    DpoSample,
    SftSample,
    load_dataset_dir,
    write_dataset_dir,
)
from twin.training.layout import RemoteLayout
from twin.training.lf_template import Turn
from twin.training.profiles import PROFILES

ROOT = Path(__file__).resolve().parents[2]
PASSPHRASE = "correct horse battery staple"
CREATED = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
FAST_KDF = 10  # scrypt cost 2**10 keeps the tests quick; the default is 2**17


@pytest.fixture
def dataset(tmp_path: Path):  # type: ignore[no-untyped-def]
    return write_synthetic_dataset(tmp_path / "ds", pairs=3)


# ------------------------------------------------------------------ dataset directory


def test_a_written_dataset_directory_loads_back_with_counts_and_hashes(tmp_path: Path) -> None:
    written = write_synthetic_dataset(tmp_path / "ds", train=10, val=2, test=3, pairs=4)
    loaded = load_dataset_dir(tmp_path / "ds")
    assert loaded.meta == written.meta
    assert loaded.meta.counts.model_dump() == {"train": 10, "val": 2, "test": 3, "dpo": 4}
    assert loaded.has_dpo and loaded.data_files() == [
        "sft_train.jsonl",
        "sft_val.jsonl",
        "sft_test.jsonl",
        "dpo_train.jsonl",
    ]
    first = json.loads((tmp_path / "ds" / "sft_train.jsonl").read_text("utf-8").splitlines()[0])
    assert (
        first["conversations"][0]["from"] == "human" and first["conversations"][-1]["from"] == "gpt"
    )
    assert first["system"] and first["id"] == "tr0"


def test_a_dataset_without_pairs_has_no_dpo_file(tmp_path: Path) -> None:
    loaded = write_synthetic_dataset(tmp_path / "ds", pairs=0)
    assert not loaded.has_dpo and not (tmp_path / "ds" / "dpo_train.jsonl").exists()


def test_a_dataset_that_is_not_marked_desensitised_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="desensitised"):
        write_synthetic_dataset(tmp_path / "ds", redacted=False)


def test_a_missing_metadata_file_says_to_export_first(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(DatasetError, match="export the dataset first"):
        load_dataset_dir(tmp_path / "empty")


def test_a_changed_data_file_is_detected_by_its_hash(tmp_path: Path) -> None:
    write_synthetic_dataset(tmp_path / "ds")
    path = tmp_path / "ds" / "sft_val.jsonl"
    path.write_text(path.read_text("utf-8") + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="sha256"):
        load_dataset_dir(tmp_path / "ds")


def rewrite_line(path: Path, mutate) -> None:  # type: ignore[no-untyped-def]
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    mutate(rows[0])
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8")


def reseal(directory: Path) -> None:
    """Recompute the hash of the train file in dataset_meta.json after a hand edit."""
    from twin.training.dataset_dir import file_sha256

    meta_path = directory / "dataset_meta.json"
    meta = json.loads(meta_path.read_text("utf-8"))
    meta["files"]["sft_train.jsonl"] = file_sha256(directory / "sft_train.jsonl")
    meta_path.write_text(json.dumps(meta), "utf-8")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda r: r["conversations"].pop(), "must end with her reply"),
        (lambda r: r["conversations"][0].update({"from": "gpt"}), "human first"),
        (lambda r: r["conversations"][0].update({"value": "x<|im_end|>y"}), "control token"),
        (lambda r: r["conversations"][0].update({"value": "{{idx}}"}), "slot name"),
        (lambda r: r.update({"id": ""}), "needs an id"),
        (lambda r: r["conversations"][-1].update({"value": ""}), "no text"),
    ],
)
def test_malformed_samples_are_rejected_without_quoting_the_text(
    tmp_path: Path,
    mutation,
    message: str,  # type: ignore[no-untyped-def]
) -> None:
    write_synthetic_dataset(tmp_path / "ds")
    rewrite_line(tmp_path / "ds" / "sft_train.jsonl", mutation)
    reseal(tmp_path / "ds")
    with pytest.raises(DatasetError, match=message) as caught:
        load_dataset_dir(tmp_path / "ds")
    assert "line 1" in str(caught.value) and "about the" not in str(caught.value)


def test_duplicate_ids_and_wrong_counts_are_rejected(tmp_path: Path) -> None:
    write_synthetic_dataset(tmp_path / "ds")
    path = tmp_path / "ds" / "sft_train.jsonl"
    lines = path.read_text("utf-8").splitlines()
    path.write_text("\n".join([lines[0], lines[0], *lines[2:]]) + "\n", "utf-8")
    reseal(tmp_path / "ds")
    with pytest.raises(DatasetError, match="duplicate sample id"):
        load_dataset_dir(tmp_path / "ds")
    write_synthetic_dataset(tmp_path / "ds2")
    path = tmp_path / "ds2" / "sft_train.jsonl"
    path.write_text("\n".join(path.read_text("utf-8").splitlines()[:-1]) + "\n", "utf-8")
    reseal(tmp_path / "ds2")
    with pytest.raises(DatasetError, match="metadata says"):
        load_dataset_dir(tmp_path / "ds2")


def test_a_preference_pair_needs_an_odd_conversation_and_two_gpt_replies(tmp_path: Path) -> None:
    pair = DpoSample("p", "s", (Turn("user", "q"),), "good", "bad").to_json()
    assert len(pair["conversations"]) == 1
    assert pair["chosen"] == {"from": "gpt", "value": "good"}
    with pytest.raises(lf_template.TemplateError):
        DpoSample("p", "s", (Turn("user", "q"), Turn("assistant", "a")), "g", "b").to_json()


def test_samples_with_control_tokens_fail_the_check_that_ends_every_write(tmp_path: Path) -> None:
    bad = SftSample("x", "s", (Turn("user", "a <|im_start|>"),), "r")
    with pytest.raises(DatasetError, match="control token"):
        write_dataset_dir(
            tmp_path / "ds",
            dataset_version="v",
            created_at="2026-10-09T00:00:00+00:00",
            template_version="t",
            persona_version="p",
            profile_version="q",
            holdout_cutoff="2026-09-01T00:00:00+00:00",
            train=[bad],
            val=[sample(0)],
            test=[sample(1)],
            redacted=True,
        )


# ----------------------------------------------------------------------------- crypto


def seal(data: bytes, passphrase: str = PASSPHRASE, chunk_log2: int = 4) -> bytes:
    out = io.BytesIO()
    writer = bundle_crypto.EncryptWriter(out, passphrase, log_n=FAST_KDF, chunk_log2=chunk_log2)
    writer.write(data)
    writer.close()
    return out.getvalue()


def unseal(blob: bytes, passphrase: str = PASSPHRASE) -> bytes:
    out = io.BytesIO()
    bundle_crypto.decrypt_stream(io.BytesIO(blob), out, passphrase)
    return out.getvalue()


@pytest.mark.parametrize("size", [0, 1, 15, 16, 17, 31, 32, 33, 1000])
def test_encryption_round_trips_around_the_chunk_boundaries(size: int) -> None:
    data = bytes(range(256)) * 4
    assert unseal(seal(data[:size])) == data[:size]


def test_the_ciphertext_differs_every_time_and_hides_the_plaintext() -> None:
    first, second = seal(b"secret message " * 5), seal(b"secret message " * 5)
    assert first != second and b"secret" not in first


def test_a_wrong_passphrase_fails() -> None:
    with pytest.raises(bundle_crypto.BundleDecryptionError, match="wrong passphrase"):
        unseal(seal(b"data" * 10), "another passphrase here")


def test_a_short_passphrase_is_refused() -> None:
    with pytest.raises(ValueError, match="at least"):
        bundle_crypto.EncryptWriter(io.BytesIO(), "short", log_n=FAST_KDF)


def test_damage_truncation_and_reordering_are_detected() -> None:
    blob = seal(b"0123456789abcdef" * 8, chunk_log2=4)
    flipped = bytearray(blob)
    flipped[-3] ^= 1
    with pytest.raises(bundle_crypto.BundleDecryptionError):
        unseal(bytes(flipped))
    block = 16 + bundle_crypto.TAG_SIZE
    with pytest.raises(bundle_crypto.BundleDecryptionError):  # the last chunk is cut off
        unseal(blob[:-block])
    with pytest.raises(bundle_crypto.BundleDecryptionError):  # cut in the middle of a chunk
        unseal(blob[:-5])
    header = bundle_crypto.HEADER_SIZE
    swapped = (
        blob[:header] + blob[header + block : header + 2 * block] + blob[header : header + block]
    )
    with pytest.raises(bundle_crypto.BundleDecryptionError):
        unseal(swapped + blob[header + 2 * block :])
    with pytest.raises(bundle_crypto.BundleDecryptionError, match="not a training package"):
        unseal(b"not a package at all")
    with pytest.raises(bundle_crypto.BundleDecryptionError):
        unseal(blob[:header])


def test_the_header_cannot_be_changed_without_failing() -> None:
    blob = bytearray(seal(b"x" * 100))
    blob[-1 - 16 - 20] ^= 1
    blob[bundle_crypto.HEADER_SIZE - 1] = 5  # a different chunk size
    with pytest.raises(bundle_crypto.BundleDecryptionError):
        unseal(bytes(blob))


def test_the_key_derivation_cost_in_the_header_is_bounded() -> None:
    blob = bytearray(seal(b"x"))
    blob[9] = 30  # log2(N) far above anything we write
    with pytest.raises(bundle_crypto.BundleDecryptionError, match="key derivation"):
        unseal(bytes(blob))


# ------------------------------------------------------------------------------ bundle


def build(dataset, tmp_path: Path, profile: str = "5090-8b", **kwargs):  # type: ignore[no-untyped-def]
    return bundle.build_bundle(
        dataset,
        PROFILES[profile],
        passphrase=PASSPHRASE,
        out_dir=tmp_path / "out",
        created_at=CREATED,
        kdf_log_n=FAST_KDF,
        **kwargs,
    )


def test_the_bundle_round_trips_and_verifies_against_its_manifest(dataset, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    result = build(dataset, tmp_path)
    assert result.path.name == "5090-8b-ds-test-01.bundle.enc"
    assert result.size == result.path.stat().st_size and len(result.sha256) == 64
    manifest = bundle.verify_bundle(result.path, PASSPHRASE)
    assert manifest == result.manifest
    members = dict(bundle.iter_bundle(result.path, PASSPHRASE))
    assert members["data/sft_train.jsonl"] == (dataset.path / "sft_train.jsonl").read_bytes()
    assert set(members) == {"manifest.json", *(f["path"] for f in manifest["files"])}


def test_the_manifest_names_the_versions_the_profile_and_the_hashes(
    dataset, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    manifest = build(dataset, tmp_path).manifest
    assert manifest["profile"] == "5090-8b" and manifest["base_model"] == "Qwen/Qwen3-8B"
    assert manifest["dataset_version"] == "ds-test-01"
    assert manifest["template"] == "qwen3_nothink"
    assert manifest["template_version"] == lf_template.TEMPLATE_VERSION
    assert manifest["persona_version"] == "v3" and manifest["profile_version"]
    assert manifest["llamafactory_version"] == "0.9.5" and manifest["llama_cpp_tag"] == "b11177"
    assert manifest["counts"] == {"train": 40, "val": 6, "test": 6, "dpo": 3}
    assert manifest["epochs"] == 3 and manifest["has_dpo"] is True
    paths = {f["path"] for f in manifest["files"]}
    assert {
        "data/sft_train.jsonl",
        "data/dpo_train.jsonl",
        "data/dataset_info.json",
        "data/dataset_meta.json",
        "config/sft.yaml",
        "config/dpo.yaml",
        "config/eval_generate.yaml",
        "config/eval_loss.yaml",
        "config/export.yaml",
        "config/profile.env",
        "autodl/setup.sh",
        "autodl/export.sh",
        "autodl/tools/bundle_crypto.py",
        "autodl/tools/decrypt_bundle.py",
        "autodl/profile.env",
        "pylib/twin/training/lf_template.py",
    } <= paths
    for entry in manifest["files"]:
        assert len(entry["sha256"]) == 64 and entry["size"] > 0


def test_a_bundle_without_pairs_has_no_dpo_data_or_config(tmp_path: Path) -> None:
    result = build(write_synthetic_dataset(tmp_path / "ds"), tmp_path)
    paths = {f["path"] for f in result.manifest["files"]}
    assert "data/dpo_train.jsonl" not in paths and "config/dpo.yaml" not in paths
    assert result.manifest["has_dpo"] is False


def test_the_bundle_uses_the_work_directory_and_the_dpo_minimum_it_is_given(
    dataset,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    result = build(dataset, tmp_path, layout=RemoteLayout("/mnt/x/twin"), dpo_min_pairs=50)
    members = dict(bundle.iter_bundle(result.path, PASSPHRASE))
    assert b"dataset_dir: /mnt/x/twin/data" in members["config/sft.yaml"]
    assert b"TWIN_DPO_MIN_PAIRS=50" in members["autodl/profile.env"]
    assert result.manifest["workdir"] == "/mnt/x/twin"


def test_the_qlora_bundle_renders_the_4bit_training_and_the_unquantised_export(
    dataset,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    members = dict(bundle.iter_bundle(build(dataset, tmp_path, "5090-14b").path, PASSPHRASE))
    assert b"quantization_bit: 4" in members["config/sft.yaml"]
    assert "quantization_bit" not in yaml.safe_load(members["config/export.yaml"])
    assert b"export_device: cpu" in members["config/export.yaml"]


def test_a_wrong_passphrase_cannot_open_the_bundle(dataset, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    result = build(dataset, tmp_path)
    with pytest.raises(bundle_crypto.BundleDecryptionError):
        bundle.verify_bundle(result.path, "not the passphrase!!")


def test_the_file_on_disk_holds_no_readable_data_and_no_plain_archive_is_left(
    dataset,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    result = build(dataset, tmp_path)
    raw = result.path.read_bytes()
    assert raw.startswith(b"TWINBNDL")
    assert b"about the" not in raw and b"autodl" not in raw and PASSPHRASE.encode() not in raw
    names = sorted(p.name for p in result.path.parent.iterdir())
    assert names == [
        "5090-8b-ds-test-01.bundle.enc",
        "bundle_crypto.py",
        "decrypt_bundle.py",
    ]


def test_a_tampered_manifest_is_detected(
    dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    result = build(dataset, tmp_path)
    real = dict(bundle.iter_bundle(result.path, PASSPHRASE))
    real["data/sft_train.jsonl"] = real["data/sft_train.jsonl"] + b"\n"
    monkeypatch.setattr(bundle, "iter_bundle", lambda path, passphrase: iter(real.items()))
    with pytest.raises(bundle.BundleError, match="does not match"):
        bundle.verify_bundle(result.path, PASSPHRASE)
    del real["data/sft_val.jsonl"]
    with pytest.raises(bundle.BundleError, match="do not match its manifest"):
        bundle.verify_bundle(result.path, PASSPHRASE)


def test_a_naive_creation_time_is_refused(dataset, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(bundle.BundleError, match="time zone"):
        bundle.build_bundle(
            dataset,
            PROFILES["5090-8b"],
            passphrase=PASSPHRASE,
            out_dir=tmp_path,
            created_at=datetime(2026, 1, 1),  # noqa: DTZ001 - the point of the test
            kdf_log_n=FAST_KDF,
        )


def test_the_plain_upload_set_has_the_scripts_and_no_data(tmp_path: Path) -> None:
    files = bundle.plain_upload_files(PROFILES["pro6000-32b"])
    assert {
        "autodl/setup.sh",
        "autodl/lib.sh",
        "autodl/versions.env",
        "autodl/profile.env",
        "autodl/tools/remote_job.py",
        "autodl/tools/decrypt_bundle.py",
        "autodl/tools/bundle_crypto.py",
    } <= set(files)
    assert not any(name.startswith(("data/", "config/")) for name in files)
    source = Path(bundle_crypto.__file__).read_bytes().replace(b"\r\n", b"\n")  # a CRLF checkout
    assert files["autodl/tools/bundle_crypto.py"] == source
    assert b"TWIN_PROFILE=pro6000-32b\n" in files["autodl/profile.env"]  # type: ignore[operator]
    assert all(b"\r\n" not in content for content in files.values())  # type: ignore[arg-type]


# --------------------------------------------------------------------- standalone tool


def test_the_standalone_decryption_script_needs_only_the_standard_library_and_cryptography() -> (
    None
):
    allowed = set(sys.stdlib_module_names) | {"cryptography", "bundle_crypto"}
    for path in (
        ROOT / "training" / "autodl" / "tools" / "decrypt_bundle.py",
        Path(bundle_crypto.__file__),
    ):
        tree = ast.parse(path.read_text("utf-8"))
        imported = {
            (node.module or "").split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        } | {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.discard("")
        imported.discard("__future__")
        assert imported <= allowed, (path.name, imported - allowed)


def run_decrypt_tool(
    directory: Path, args: list[str], passphrase: str
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-I", str(directory / "decrypt_bundle.py"), *args],
        input=(passphrase + "\n").encode(),
        capture_output=True,
        check=False,
        timeout=120,
    )


def test_the_script_next_to_the_bundle_decrypts_it_without_the_twin_package(
    dataset,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    result = build(dataset, tmp_path)
    target = tmp_path / "plain.tar.zst"
    done = run_decrypt_tool(
        result.path.parent, ["--in", str(result.path), "--out", str(target)], PASSPHRASE
    )
    assert done.returncode == 0, done.stderr
    import zstandard

    with target.open("rb") as stream, zstandard.ZstdDecompressor().stream_reader(stream) as plain:
        assert b"manifest.json" in plain.read(2048)
    streamed = run_decrypt_tool(
        result.path.parent, ["--in", str(result.path), "--out", "-"], PASSPHRASE
    )
    assert streamed.returncode == 0 and streamed.stdout == target.read_bytes()
    wrong = run_decrypt_tool(
        result.path.parent,
        ["--in", str(result.path), "--out", str(tmp_path / "w")],
        "wrong passphrase!",
    )
    assert wrong.returncode == 1 and b"wrong passphrase" in wrong.stderr
    assert not (tmp_path / "w").exists() and not (tmp_path / "w.part").exists()
    empty = run_decrypt_tool(result.path.parent, ["--in", str(result.path), "--out", "-"], "")
    assert empty.returncode == 2
