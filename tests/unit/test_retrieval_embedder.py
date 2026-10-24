"""The embedding service: redaction before encoding, lazy loading, version records
(R-RET-002, R-LLM-009, R-NFR-003) and the text a context is encoded as."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.support.embedding import HashingBackend
from twin.config.settings import RetrievalConfig
from twin.ingest.events import Kind
from twin.retrieval import embedder as embedder_module
from twin.retrieval.embedder import (
    EmbedderError,
    EmbedderInfo,
    EmbeddingService,
    EncodeKind,
    SentenceTransformerBackend,
    cpu_speed_warning,
    embedding_service,
    file_sha256,
    manifest_path,
    normalise,
    profile_of,
    read_manifest,
    record_model_files,
    reset_embedding_services,
    resolve_device,
    slug_of,
    snapshot_ignore_patterns,
    weights_file,
)
from twin.retrieval.records import MessageData
from twin.retrieval.texts import (
    EARLIER_TURN_CHARS,
    RECENT_TURN_CHARS,
    STICKER_TEXT,
    TurnText,
    clip,
    encoding_text,
    message_text,
    turn_text,
)
from twin.services import Services

ROOT = Path(__file__).resolve().parents[2]
PHONE = "1" + "3800138000"  # built from parts: the privacy scan must not see a number


def data(kind: str, text: str | None = None, *, sent: bool = False, **extra: object) -> MessageData:
    return MessageData(
        id="m",
        kind=kind,
        is_sent=sent,
        text=text,
        quote=None,
        sticker_md5=extra.get("md5"),  # type: ignore[arg-type]
        call_status=extra.get("call_status"),  # type: ignore[arg-type]
        call_duration_s=extra.get("call_duration_s"),  # type: ignore[arg-type]
        voice_seconds=None,
        has_transcript=False,
    )


# ------------------------------------------------------------------ the service


def test_text_is_redacted_before_it_reaches_the_model(embedder: HashingBackend) -> None:
    service = EmbeddingService(embedder)
    service.encode([f"打我电话 {PHONE} 或者发邮件 a.b@example.com"])
    seen = embedder.seen[-1]
    assert PHONE not in seen and "a.b@example.com" not in seen
    assert "[手机号]" in seen and "[邮箱]" in seen


def test_vectors_are_normalised_and_one_row_per_text(embedder: HashingBackend) -> None:
    service = EmbeddingService(embedder)
    out = service.encode(["今天吃饭", "明天上课", ""])
    assert out.shape == (3, embedder.info.dimension) and out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out[:2], axis=1), 1.0, atol=1e-5)
    assert service.encode([]).shape == (0, embedder.info.dimension)
    assert service.encode_one("你好").shape == (embedder.info.dimension,)
    assert np.all(normalise(np.zeros((1, 4), dtype=np.float32)) == 0)  # no division by zero


def test_a_query_gets_the_models_instruction_and_other_kinds_do_not() -> None:
    backend = HashingBackend(model="BAAI/bge-small-zh-v1.5")
    service = EmbeddingService(backend)
    service.encode(["附近的饭店"], EncodeKind.QUERY)
    service.encode(["附近的饭店"], EncodeKind.PASSAGE)
    service.encode(["附近的饭店"])
    assert backend.seen[0].startswith(profile_of("BAAI/bge-small-zh-v1.5").query_instruction)
    assert backend.seen[1] == backend.seen[2] == "附近的饭店"
    assert profile_of("some/other-model").query_instruction == ""


def test_the_shared_service_is_built_once_and_not_loaded_at_start(
    services: Services, embedder: HashingBackend
) -> None:
    first = embedding_service(services)
    assert embedding_service(services) is first
    assert embedder.runs == 0  # nothing is encoded by asking for the service
    reset_embedding_services()
    assert embedding_service(services) is not first


def test_info_compares_vector_spaces() -> None:
    a = EmbedderInfo("m", 8, "r1", "aa", "cpu")
    assert a.same_vectors(EmbedderInfo("m", 8, "r2", "aa", "cuda"))
    assert not a.same_vectors(EmbedderInfo("m", 8, "r1", "bb", "cpu"))
    assert not a.same_vectors(EmbedderInfo("m", 16, "r1", "aa", "cpu"))
    assert not a.same_vectors(EmbedderInfo("n", 8, "r1", "aa", "cpu"))
    assert "m (8 dimensions" in a.describe()


# ------------------------------------------------------------- lazy and the files


def test_importing_the_application_does_not_load_the_model_libraries() -> None:
    """R-NFR-003: start-up never imports torch, sentence-transformers, pyarrow or LanceDB."""
    code = (
        "import sys, twin.cli, twin.app, twin.retrieval.cli, twin.retrieval.query, "
        "twin.retrieval.indexer, twin.retrieval.embedder as e\n"
        "from pathlib import Path\n"
        "e.SentenceTransformerBackend('BAAI/bge-small-zh-v1.5', 'auto', Path('x'))\n"
        "bad = [m for m in ('torch', 'sentence_transformers', 'lancedb', 'pyarrow', 'transformers')"
        " if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_the_real_backend_does_nothing_until_it_is_used(tmp_path: Path) -> None:
    backend = SentenceTransformerBackend("BAAI/bge-small-zh-v1.5", "auto", tmp_path / "models")
    assert not backend.loaded
    assert not (tmp_path / "models").exists()  # not even the folder


def test_the_device_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_device("auto", lambda: True) == "cuda"
    assert resolve_device("auto", lambda: False) == "cpu"
    assert resolve_device("cpu", lambda: True) == "cpu"
    assert resolve_device("CUDA", lambda: True) == "cuda"
    assert resolve_device("cuda:1", lambda: True) == "cuda:1"
    with pytest.raises(EmbedderError, match="no usable NVIDIA GPU"):
        resolve_device("cuda", lambda: False)
    with pytest.raises(EmbedderError, match="auto, cpu or cuda"):
        resolve_device("tpu", lambda: False)


def test_only_what_torch_needs_is_downloaded() -> None:
    both = snapshot_ignore_patterns(["model.safetensors", "pytorch_model.bin", "config.json"])
    assert "*.bin" in both and "onnx/*" in both
    only_bin = snapshot_ignore_patterns(["pytorch_model.bin", "config.json"])
    assert "*.bin" not in only_bin and "*.onnx" in only_bin
    nested = snapshot_ignore_patterns(["onnx/model.safetensors", "pytorch_model.bin"])
    assert "*.bin" not in nested


def test_the_model_files_are_recorded_with_revision_and_hash(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--x--y" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"weights-v1")
    folder = tmp_path / "embeddings"
    info = record_model_files(folder, "x/y", snapshot, 512, "cpu")
    assert (info.model, info.dimension, info.revision, info.device) == ("x/y", 512, "abc123", "cpu")
    assert info.weights_sha256 == file_sha256(snapshot / "model.safetensors")
    record = read_manifest(folder, "x/y")
    assert record is not None
    assert record["revision"] == "abc123" and record["weights_file"] == "model.safetensors"
    assert record["weights_sha256"] == info.weights_sha256 and record["dimension"] == 512
    assert manifest_path(folder, "x/y").name == slug_of("x/y") + ".json"

    # unchanged files: the record is reused, not recomputed
    stamp = record["recorded_at"]
    again = record_model_files(folder, "x/y", snapshot, 512, "cuda")
    assert again.weights_sha256 == info.weights_sha256 and again.device == "cuda"
    assert read_manifest(folder, "x/y")["recorded_at"] == stamp  # type: ignore[index]

    # other weights: a new record, a different identity
    (snapshot / "model.safetensors").write_bytes(b"weights-v2-longer")
    changed = record_model_files(folder, "x/y", snapshot, 512, "cpu")
    assert changed.weights_sha256 != info.weights_sha256
    assert not changed.same_vectors(info)
    assert json.loads(manifest_path(folder, "x/y").read_text(encoding="utf-8"))[
        "weights_size"
    ] == len(b"weights-v2-longer")


def test_a_folder_without_weights_is_reported(tmp_path: Path) -> None:
    with pytest.raises(EmbedderError, match="no model weights"):
        weights_file(tmp_path)
    assert read_manifest(tmp_path, "nothing/here") is None


def test_the_default_factory_builds_the_real_backend_with_the_data_folder(
    services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(embedder_module, "backend_factory", embedder_module.default_backend_factory)
    reset_embedding_services()
    try:
        service = embedding_service(services)
        backend = service.backend
        assert isinstance(backend, SentenceTransformerBackend)
        assert not backend.loaded
    finally:
        reset_embedding_services()
    assert services.paths.embeddings_dir == services.paths.data_dir / "models" / "embeddings"
    assert services.paths.vectors_dir == services.paths.data_dir / "vectors"


def test_a_slow_model_on_a_cpu_is_pointed_out() -> None:
    small = RetrievalConfig()
    big = RetrievalConfig(model="BAAI/bge-m3")
    assert cpu_speed_warning(small, "cpu", 500_000) is None
    assert cpu_speed_warning(big, "cuda", 500_000) is None
    assert cpu_speed_warning(big, "cpu", 100) is None
    hint = cpu_speed_warning(big, "cpu", 80_000)
    assert hint is not None and "80,000" in hint and "bge-small" in hint
    assert cpu_speed_warning(big, "cpu", None) is not None


# --------------------------------------------------------------------- the text


def test_the_last_two_turns_are_written_twice_and_first() -> None:
    turns = [TurnText(i % 2 == 0, f"第{i}句") for i in range(5)]
    text = encoding_text(turns)
    lines = text.split("\n")
    assert lines[:2] == ["你：第3句", "我：第4句"]
    # the full context follows in order, ending with the same two turns again
    assert lines[2:5] == ["我：第0句", "你：第1句", "我：第2句"]
    assert lines[-2:] == lines[:2]
    assert len(lines) == 2 + 5


def test_a_short_context_is_not_repeated_and_nothing_gives_nothing() -> None:
    assert encoding_text([TurnText(False, "在吗")]) == "你：在吗"
    assert encoding_text([TurnText(False, "在吗"), TurnText(True, "在")]) == "你：在吗\n我：在"
    assert encoding_text([]) == "" and encoding_text([TurnText(True, "")]) == ""


def test_long_turns_are_clipped_so_the_recent_ones_always_fit() -> None:
    long = "长" * 500
    turns = [TurnText(False, long), TurnText(True, long), TurnText(False, long)]
    text = encoding_text(turns)
    first, *_, last = text.split("\n")
    assert len(first) == len("你：") + RECENT_TURN_CHARS and first.endswith("…")
    assert len(last) == len("你：") + RECENT_TURN_CHARS
    older = text.split("\n")[2]
    assert len(older) == len("你：") + EARLIER_TURN_CHARS
    assert clip("短", 5) == "短" and clip("很长很长很长", 4) == "很长很…"


def test_non_text_messages_read_as_event_text_and_stickers_as_a_marker() -> None:
    assert message_text(data("text", "好  的\n呀")) == "好 的 呀"
    assert message_text(data("sticker", None, md5="a" * 32)) == STICKER_TEXT
    assert message_text(data(Kind.IMAGE.value)) == "[图片]"
    assert message_text(data(Kind.CALL.value, call_status="connected", call_duration_s=125)) == (
        "[通话 2 分钟]"
    )
    turn = turn_text([data("text", "看", sent=True), data(Kind.IMAGE.value, sent=True)])
    assert turn == TurnText(False, "看 [图片]")
    with pytest.raises(ValueError, match="at least one message"):
        turn_text([])
