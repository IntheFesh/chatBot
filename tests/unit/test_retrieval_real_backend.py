"""The production embedding backend (sentence-transformers + torch) on a tiny model built here.

No network and no download: a very small BERT with random weights and a character vocabulary is
written to a folder in the layout of a Hugging Face model and loaded through the same code the
real ``BAAI/bge-small-zh-v1.5`` goes through (R-RET-002, R-NFR-003).  The real model itself is
exercised by ``tests/integration/test_retrieval_live.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from twin.retrieval import embedder as embedder_module
from twin.retrieval.embedder import (
    EmbedderError,
    EmbeddingService,
    SentenceTransformerBackend,
    read_manifest,
)

CHARS = "今天吃饭了吗你我他她好的是不在有这个那什么火锅论文考试"
DIMENSION = 16


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real (random, tiny) sentence-transformers model saved like a downloaded one."""
    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
    from transformers import BertConfig, BertModel, BertTokenizerFast

    folder = tmp_path_factory.mktemp("tiny-model")
    backbone = folder / "backbone"
    backbone.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *CHARS]
    (backbone / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(backbone / "vocab.txt"))
    tokenizer.save_pretrained(backbone)
    torch.manual_seed(7)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=DIMENSION,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    BertModel(config).save_pretrained(backbone)
    transformer = Transformer(str(backbone), max_seq_length=32)
    pooling = Pooling(DIMENSION, pooling_mode="cls")
    snapshot = folder / "snapshots" / "rev0123456789abcdef"
    SentenceTransformer(modules=[transformer, pooling], device="cpu").save(str(snapshot))
    return snapshot


def backend_for(
    tiny_model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str = "auto"
) -> SentenceTransformerBackend:
    backend = SentenceTransformerBackend("test/tiny-bert", device, tmp_path / "embeddings")
    monkeypatch.setattr(backend, "_snapshot", lambda: tiny_model)
    return backend


def test_the_real_backend_loads_lazily_encodes_and_records_the_model_files(
    tiny_model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = backend_for(tiny_model, tmp_path, monkeypatch)
    assert not backend.loaded
    info = backend.info  # the first use loads the model
    assert backend.loaded
    assert (info.model, info.dimension, info.device) == ("test/tiny-bert", DIMENSION, "cpu")
    assert info.revision == "rev0123456789abcdef" and len(info.weights_sha256) == 64
    record = read_manifest(tmp_path / "embeddings", "test/tiny-bert")
    assert record is not None and record["weights_sha256"] == info.weights_sha256
    assert record["weights_file"] in ("model.safetensors", "pytorch_model.bin")
    assert backend.info is info  # loaded once

    vectors = backend.encode(["今天吃饭了吗", "你好", "火锅"], batch_size=2)
    assert vectors.shape == (3, DIMENSION) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    again = backend.encode(["今天吃饭了吗"], batch_size=8)
    assert np.allclose(again[0], vectors[0], atol=1e-5)  # deterministic
    assert backend.encode([], batch_size=8).shape == (0, DIMENSION)


def test_the_service_redacts_before_the_real_model_and_keeps_vectors_normalised(
    tiny_model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = backend_for(tiny_model, tmp_path, monkeypatch, device="cpu")
    service = EmbeddingService(backend, batch_size=4)
    phone = "1" + "3800138000"
    with_number = service.encode_one(f"今天吃饭了吗 {phone}")
    placeholder = service.encode_one("今天吃饭了吗 [手机号]")
    assert np.allclose(with_number, placeholder, atol=1e-5)  # the model saw the same text
    assert service.dimension == DIMENSION


def test_cuda_is_refused_when_torch_has_no_gpu(
    tiny_model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    if torch.cuda.is_available():
        pytest.skip("this machine has a usable GPU")
    backend = backend_for(tiny_model, tmp_path, monkeypatch, device="cuda")
    with pytest.raises(EmbedderError, match="no usable NVIDIA GPU"):
        _ = backend.info


def test_a_broken_model_folder_is_reported_with_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = tmp_path / "broken"
    broken.mkdir()
    backend = SentenceTransformerBackend("test/broken", "cpu", tmp_path / "embeddings")
    monkeypatch.setattr(backend, "_snapshot", lambda: broken)
    with pytest.raises(EmbedderError, match="delete that folder"):
        _ = backend.info
    assert not backend.loaded


# ------------------------------------------------------------- the download step


def test_the_first_use_downloads_only_what_is_needed_and_later_uses_the_local_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    calls: list[dict[str, object]] = []
    folder = tmp_path / "snapshot-folder"
    folder.mkdir()

    downloaded = {"done": False}

    def snapshot_download(repo_id: str, **kwargs: object) -> str:
        calls.append({"repo": repo_id, **kwargs})
        if kwargs.get("local_files_only"):
            if not downloaded["done"]:
                raise LocalEntryNotFoundError("not downloaded yet")
            return str(folder)
        downloaded["done"] = True
        return str(folder)

    class Api:
        def list_repo_files(self, repo_id: str) -> list[str]:
            return ["config.json", "model.safetensors", "pytorch_model.bin", "onnx/model.onnx"]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    backend = SentenceTransformerBackend("org/model", "cpu", tmp_path / "embeddings")
    assert backend._snapshot() == folder
    assert [bool(c.get("local_files_only")) for c in calls] == [True, False]
    ignored = calls[1]["ignore_patterns"]
    assert "*.bin" in ignored and "onnx/*" in ignored  # type: ignore[operator]
    assert calls[1]["cache_dir"] == str(tmp_path / "embeddings")

    calls.clear()
    assert backend._snapshot() == folder  # present: the first, local-only call answers
    assert [bool(c.get("local_files_only")) for c in calls] == [True]


def test_a_failed_download_says_what_to_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    def snapshot_download(repo_id: str, **kwargs: object) -> str:
        if kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError("not downloaded yet")
        raise ConnectionError("no route to host")

    class Api:
        def list_repo_files(self, repo_id: str) -> list[str]:
            return ["config.json"]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    backend = SentenceTransformerBackend("org/model", "cpu", tmp_path / "embeddings")
    with pytest.raises(EmbedderError, match="Hugging Face must be reachable once"):
        backend._snapshot()
    assert embedder_module.profile_of("org/model").relative_cost == 1.0
