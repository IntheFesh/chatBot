"""The text-embedding service (R-RET-002, R-NFR-003, R-LLM-009).

One service encodes text for everything that needs vectors: the example windows of this round,
the memory of round 07 and the sticker descriptions of round 06.

* **The model** is a real ``sentence-transformers`` model (``retrieval.model``, default
  ``BAAI/bge-small-zh-v1.5``, alternative ``BAAI/bge-m3``).  It is loaded on first use, never at
  start-up: importing this module does not import ``torch`` and building the application does
  not touch the model (R-NFR-003).
* **Files.**  The first use downloads the model repository into ``data/models/embeddings/`` (the
  Hugging Face cache layout); later uses read the downloaded snapshot directly and need no
  network.  A record next to it (``<model>.json``) keeps the repository revision, the weights file
  and its SHA-256, so a changed model is noticed (:class:`EmbedderInfo`).
* **Device.**  ``retrieval.device`` is ``auto`` (CUDA when torch can use it, else CPU), ``cpu`` or
  ``cuda``.
* **Redaction.**  Vectors can leak what the text said, so :class:`EmbeddingService` passes every
  text through :func:`twin.llm.redaction.redact_text` (the one redaction function, R-LLM-009)
  *before* it reaches the model.  Nothing encodes raw text.
* **Normalised output.**  Vectors have length 1, so cosine similarity is a dot product.

The model sits behind :class:`EmbeddingBackend`; the tests inject a tiny offline backend through
:data:`backend_factory`.  No substitute model exists in production code.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

from twin.clock import now_utc
from twin.llm.redaction import redact_text
from twin.ops.logging import get_logger

if TYPE_CHECKING:
    from twin.config.loader import DataPaths
    from twin.config.settings import RetrievalConfig
    from twin.services import Services

log = get_logger("twin.retrieval.embedder")

Vectors = NDArray[np.float32]
HASH_CHUNK = 1 << 20
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
# files of a model repository that are never needed to run it with torch
UNUSED_FILES = (
    "onnx/*",
    "openvino/*",
    "*.onnx",
    "*.onnx_data",
    "*.msgpack",
    "*.h5",
    "*.ot",
    "*.tflite",
    "*.mlmodel",
    "*.gguf",
    "*.png",
    "*.jpg",
    "imgs/*",
    "flax_model*",
    "tf_model*",
)
WEIGHT_PATTERNS = ("*.bin", "*.pt")  # skipped when safetensors weights exist


class EmbedderError(RuntimeError):
    """The embedding model cannot be loaded or used (the message says what to do)."""


class EncodeKind(StrEnum):
    """What the text is, which decides the model-specific instruction (if the model has one)."""

    SYMMETRIC = "symmetric"  # compared with texts of its own kind (conversation contexts)
    QUERY = "query"  # a short request that searches longer passages
    PASSAGE = "passage"  # the passages such a query searches


@dataclass(frozen=True)
class ModelProfile:
    """What is known about a model family."""

    query_instruction: str = ""  # prepended to QUERY texts
    relative_cost: float = 1.0  # CPU time per text compared with bge-small


MODEL_PROFILES: dict[str, ModelProfile] = {
    "BAAI/bge-small-zh-v1.5": ModelProfile("为这个句子生成表示以用于检索相关文章：", 1.0),
    "BAAI/bge-m3": ModelProfile("", 20.0),  # measured: 181 s against 9.0 s per 1,000 windows
}


def profile_of(model: str) -> ModelProfile:
    return MODEL_PROFILES.get(model, ModelProfile())


@dataclass(frozen=True)
class EmbedderInfo:
    """Which model produced the vectors: the identity an index is tied to."""

    model: str
    dimension: int
    revision: str  # commit of the model repository ("" if the backend has no such notion)
    weights_sha256: str
    device: str

    def same_vectors(self, other: EmbedderInfo) -> bool:
        """True if vectors of the two are comparable (same model, size and weights)."""
        return (
            self.model == other.model
            and self.dimension == other.dimension
            and self.weights_sha256 == other.weights_sha256
        )

    def describe(self) -> str:
        weights = self.weights_sha256[:12] or "-"
        return f"{self.model} ({self.dimension} dimensions, weights {weights}, {self.device})"


class EmbeddingBackend(Protocol):
    """A text-embedding model: the real one below, a small offline one in the tests."""

    @property
    def info(self) -> EmbedderInfo:
        """Identity of the model; may load it."""
        ...

    def encode(self, texts: Sequence[str], *, batch_size: int) -> Vectors:
        """One row per text, length-1 vectors."""
        ...


# ---------------------------------------------------------------- the real model


def slug_of(model: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "--", model).strip("-")


def manifest_path(embeddings_dir: Path, model: str) -> Path:
    return embeddings_dir / f"{slug_of(model)}.json"


def read_manifest(embeddings_dir: Path, model: str) -> dict[str, Any] | None:
    """The version record written when the model was first used, or ``None``."""
    path = manifest_path(embeddings_dir, model)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def weights_file(snapshot: Path) -> Path:
    for name in WEIGHT_FILES:
        candidate = snapshot / name
        if candidate.is_file():
            return candidate
    raise EmbedderError(f"no model weights found in {snapshot}; delete the folder and retry")


def resolve_device(setting: str, cuda_available: Callable[[], bool]) -> str:
    """``auto`` -> ``cuda`` if usable else ``cpu``; ``cpu`` and ``cuda[:n]`` as given."""
    wanted = setting.strip().lower()
    if wanted == "auto":
        return "cuda" if cuda_available() else "cpu"
    if wanted == "cpu":
        return "cpu"
    if wanted == "cuda" or re.fullmatch(r"cuda:\d+", wanted):
        if not cuda_available():
            raise EmbedderError(
                "retrieval.device is cuda, but torch finds no usable NVIDIA GPU; use 'auto' or "
                "'cpu' (docs/PENDING_USER_ACTIONS.md, round 05, explains the CUDA build of torch)"
            )
        return wanted
    raise EmbedderError(f"retrieval.device must be auto, cpu or cuda, not {setting!r}")


def snapshot_ignore_patterns(files: Sequence[str]) -> list[str]:
    """What not to download: ONNX/TF copies, and ``.bin`` weights if safetensors exist."""
    ignore = list(UNUSED_FILES)
    if any(name.endswith(".safetensors") and "/" not in name for name in files):
        ignore.extend(WEIGHT_PATTERNS)
    return ignore


def record_model_files(
    embeddings_dir: Path, model: str, snapshot: Path, dimension: int, device: str
) -> EmbedderInfo:
    """Write (or confirm) the version record of a model's files and return its identity.

    The record keeps the repository revision (the snapshot folder name), the weights file, its
    size and SHA-256 (R-RET-002).  The hash is computed once and reused while revision and size
    are unchanged.
    """
    weights = weights_file(snapshot)
    size = weights.stat().st_size
    revision = snapshot.name
    known = read_manifest(embeddings_dir, model)
    if (
        known
        and known.get("revision") == revision
        and known.get("weights_size") == size
        and isinstance(known.get("weights_sha256"), str)
    ):
        digest = str(known["weights_sha256"])
    else:
        digest = file_sha256(weights)
        record = {
            "model": model,
            "revision": revision,
            "weights_file": weights.name,
            "weights_size": size,
            "weights_sha256": digest,
            "dimension": dimension,
            "recorded_at": now_utc().isoformat(),
        }
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        manifest_path(embeddings_dir, model).write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return EmbedderInfo(model, dimension, revision, digest, device)


class SentenceTransformerBackend:
    """``sentence-transformers`` on the downloaded model, loaded on first use."""

    def __init__(self, model: str, device: str, embeddings_dir: Path) -> None:
        self._model_name = model
        self._device_setting = device
        self._dir = embeddings_dir
        self._lock = threading.Lock()
        self._model: Any = None
        self._info: EmbedderInfo | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------ files

    def _snapshot(self) -> Path:
        """The local snapshot folder of the model, downloading it the first time."""
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        self._dir.mkdir(parents=True, exist_ok=True)
        try:
            return Path(
                snapshot_download(self._model_name, cache_dir=str(self._dir), local_files_only=True)
            )
        except LocalEntryNotFoundError:
            pass
        log.info("embedding_model_download", model=self._model_name, folder=str(self._dir))
        try:
            files = HfApi().list_repo_files(self._model_name)
            return Path(
                snapshot_download(
                    self._model_name,
                    cache_dir=str(self._dir),
                    ignore_patterns=snapshot_ignore_patterns(files),
                )
            )
        except Exception as exc:
            raise EmbedderError(
                f"cannot download the embedding model {self._model_name} into {self._dir} "
                f"({type(exc).__name__}); check the internet connection (Hugging Face must be "
                "reachable once) or place the model folder there by hand"
            ) from exc

    # ---------------------------------------------------------------- loading

    def _load(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            import torch
            from sentence_transformers import SentenceTransformer

            device = resolve_device(self._device_setting, torch.cuda.is_available)
            snapshot = self._snapshot()
            log.info("embedding_model_loading", model=self._model_name, device=device)
            try:
                model = SentenceTransformer(str(snapshot), device=device)
            except Exception as exc:
                raise EmbedderError(
                    f"the embedding model in {snapshot} cannot be loaded ({type(exc).__name__}); "
                    "delete that folder to download it again"
                ) from exc
            size_of = getattr(model, "get_embedding_dimension", None) or (
                model.get_sentence_embedding_dimension
            )
            dimension = int(size_of() or 0)
            if dimension <= 0:
                raise EmbedderError(f"{self._model_name} does not report an embedding size")
            self._info = record_model_files(
                self._dir, self._model_name, snapshot, dimension, device
            )
            self._model = model
            return model

    @property
    def info(self) -> EmbedderInfo:
        self._load()
        if self._info is None:  # set together with the model
            raise EmbedderError("the embedding model is not loaded")
        return self._info

    def encode(self, texts: Sequence[str], *, batch_size: int) -> Vectors:
        model = self._load()
        if not texts:
            return np.zeros((0, self.info.dimension), dtype=np.float32)
        with self._lock:
            encoded = model.encode(
                list(texts),
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return np.asarray(encoded, dtype=np.float32)


# --------------------------------------------------------------------- service


def normalise(vectors: Vectors) -> Vectors:
    """Scale each row to length 1 (a zero row stays zero)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.asarray(vectors / norms, dtype=np.float32)


class EmbeddingService:
    """Redacts, prepares and encodes text; the only door to the embedding model."""

    def __init__(self, backend: EmbeddingBackend, *, batch_size: int = 64) -> None:
        self._backend = backend
        self._batch_size = batch_size

    @property
    def backend(self) -> EmbeddingBackend:
        return self._backend

    @property
    def info(self) -> EmbedderInfo:
        return self._backend.info

    @property
    def dimension(self) -> int:
        return self._backend.info.dimension

    def prepare(self, text: str, kind: EncodeKind) -> str:
        """What the model sees: the redacted text, with the query instruction if it has one."""
        clean = redact_text(text)
        if kind is EncodeKind.QUERY:
            instruction = profile_of(self._backend.info.model).query_instruction
            return instruction + clean
        return clean

    def encode(self, texts: Sequence[str], kind: EncodeKind = EncodeKind.SYMMETRIC) -> Vectors:
        """Length-1 vectors, one per text, computed from the redacted texts (R-LLM-009)."""
        if not texts:
            return np.zeros((0, self._backend.info.dimension), dtype=np.float32)
        prepared = [self.prepare(text, kind) for text in texts]
        return normalise(self._backend.encode(prepared, batch_size=self._batch_size))

    def encode_one(self, text: str, kind: EncodeKind = EncodeKind.SYMMETRIC) -> Vectors:
        return np.asarray(self.encode([text], kind)[0], dtype=np.float32)


BackendFactory = Callable[["RetrievalConfig", "DataPaths"], EmbeddingBackend]


def default_backend_factory(config: RetrievalConfig, paths: DataPaths) -> EmbeddingBackend:
    """The real model: ``sentence-transformers`` with files in ``data/models/embeddings/``."""
    return SentenceTransformerBackend(config.model, config.device, paths.embeddings_dir)


backend_factory: BackendFactory = default_backend_factory
"""How a backend is made.  The tests assign a small offline factory here; nothing else does."""

_services: dict[tuple[int, str, str, str, int], EmbeddingService] = {}
_services_lock = threading.Lock()


def embedding_service(services: Services) -> EmbeddingService:
    """The shared service for the configured model (one loaded model per process)."""
    config = services.settings.retrieval
    key = (
        id(backend_factory),
        config.model,
        config.device,
        str(services.paths.embeddings_dir),
        config.batch_size,
    )
    with _services_lock:
        found = _services.get(key)
        if found is None:
            backend = backend_factory(config, services.paths)
            found = EmbeddingService(backend, batch_size=config.batch_size)
            _services[key] = found
        return found


def reset_embedding_services() -> None:
    """Forget the shared services (the model is unloaded when nothing else holds it)."""
    with _services_lock:
        _services.clear()


def cpu_speed_warning(config: RetrievalConfig, device: str, texts: int | None) -> str | None:
    """A hint when the chosen model is slow on the CPU (R-RET-006); ``texts`` may be unknown."""
    cost = profile_of(config.model).relative_cost
    if device != "cpu" or cost < 5 or (texts is not None and texts < 2000):
        return None
    amount = f"encoding {texts:,} windows" if texts is not None else "encoding the library"
    return (
        f"{config.model} is about {cost:.0f} times slower than BAAI/bge-small-zh-v1.5 on a CPU; "
        f"{amount} may take hours.  Use the small model (retrieval.model), or run on a computer "
        "with an NVIDIA GPU (retrieval.device: cuda)"
    )


def info_to_json(info: EmbedderInfo) -> dict[str, Any]:
    return asdict(info)
