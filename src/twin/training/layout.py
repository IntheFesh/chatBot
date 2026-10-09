"""Names and locations shared by the bundle, the YAML templates and the AutoDL scripts.

Remote paths are POSIX strings (the instance runs Linux); they are never built with
:mod:`pathlib`, so they stay the same when the local machine is Windows.

Below ``workdir`` (``autodl.workdir``, default ``/root/autodl-tmp/twin``)::

    bundle.enc                  the encrypted training package (uploaded)
    autodl/                     the shell scripts and tools (uploaded as plain files, they hold
                                no data and are needed to decrypt the package)
    data/                       the decrypted dataset, ``dataset_info.json`` and the manifest
    config/                     the LLaMA-Factory YAML files rendered for the profile
    output/                     adapters (``sft``, ``dpo``), evaluation runs and the merged model
    artifacts/                  what is downloaded: GGUF files, adapter, metrics, manifest
    logs/                       one log per step
    jobs/                       state of the background jobs (pid, exit code, log offsets)
    pylib/                      the few ``twin`` modules the instance imports (template check)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from twin.training.profiles import DEFAULT_WORKDIR

DATASET_TRAIN: Final = "twin_sft_train"
DATASET_VAL: Final = "twin_sft_val"
DATASET_TEST: Final = "twin_sft_test"
DATASET_DPO: Final = "twin_dpo_train"

FILE_TRAIN: Final = "sft_train.jsonl"
FILE_VAL: Final = "sft_val.jsonl"
FILE_TEST: Final = "sft_test.jsonl"
FILE_DPO: Final = "dpo_train.jsonl"
FILE_PARITY: Final = "parity_cases.jsonl"
FILE_DATASET_META: Final = "dataset_meta.json"
FILE_DATASET_INFO: Final = "dataset_info.json"

BUNDLE_FILE_NAME: Final = "bundle.enc"
BUNDLE_MANIFEST: Final = "manifest.json"
ARTIFACT_MANIFEST: Final = "manifest.json"


@dataclass(frozen=True)
class RemoteLayout:
    """Where everything lives on the instance."""

    workdir: str = DEFAULT_WORKDIR

    def _join(self, name: str) -> str:
        return f"{self.workdir.rstrip('/')}/{name}"

    @property
    def bundle(self) -> str:
        return self._join(BUNDLE_FILE_NAME)

    @property
    def scripts(self) -> str:
        return self._join("autodl")

    @property
    def tools(self) -> str:
        return self._join("autodl/tools")

    @property
    def data(self) -> str:
        return self._join("data")

    @property
    def config(self) -> str:
        return self._join("config")

    @property
    def output(self) -> str:
        return self._join("output")

    @property
    def artifacts(self) -> str:
        return self._join("artifacts")

    @property
    def logs(self) -> str:
        return self._join("logs")

    @property
    def pylib(self) -> str:
        return self._join("pylib")

    @property
    def jobs(self) -> str:
        return self._join("jobs")

    @property
    def sft_adapter(self) -> str:
        return f"{self.output}/sft"

    @property
    def dpo_adapter(self) -> str:
        return f"{self.output}/dpo"

    @property
    def merged(self) -> str:
        return f"{self.output}/merged"
