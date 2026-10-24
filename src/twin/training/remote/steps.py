"""The steps of a training run on the instance and their records (R-TRN-010, R-PRIV-003).

Each public method of :class:`RemoteSteps` is one command of ``twin train remote``:

``connect``
    log in, show the GPU and the data disk;
``upload``
    put the scripts (plain) and the encrypted package on the instance, with resume and sha256;
``setup``
    ``setup.sh install`` -> ``decrypt.sh`` (the passphrase goes through standard input of that one
    command and is never stored) -> ``setup.sh verify``;
``train``, ``dpo``, ``evaluate``, ``export``
    one script each, as a background job that survives a dropped connection.  ``dpo`` copies how
    the reference model was chosen (``artifacts/dpo/dpo_reference.json``: the SFT model, or - for
    the 4-bit profile only - the base after a fall-back) into the run record (D-651); ``evaluate``
    runs both adapter stacks on the same test contexts and held-out preference pairs when a DPO
    adapter exists, compares them on this machine (:mod:`twin.training.stack_choice`) and writes
    the winner back to the instance, where ``export.sh`` reads it (D-655, D-862);
``download``
    the files named by ``artifacts/manifest.json`` into ``data/models/<run_id>/``, every one
    checked against its sha256;
``cleanup``
    ``cleanup.sh --yes`` once the download is confirmed; records ``cleaned_at``.

Every step writes its start, end and exit code to ``training_runs``.  A step whose job already ran
to a successful end on the instance (for example because this program was closed while it ran) is
not run again; ``again=True`` forces a fresh run.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from twin.clock import Clock
from twin.training import bundle as bundle_module
from twin.training.layout import ARTIFACT_MANIFEST, RemoteLayout
from twin.training.profiles import TrainingProfile
from twin.training.registry import RegistryError, load_manifest, verify_artifacts
from twin.training.remote.connection import RemoteError
from twin.training.remote.jobs import JobOutcome, RemoteJobs
from twin.training.remote.session import RemoteSession
from twin.training.runs import RunStore, RunView
from twin.training.stack_choice import (
    RESAMPLES,
    STACKS,
    StackChoiceError,
    StackFiles,
    StackJudge,
    StackVerdict,
)

PROGRESS_STEP_PERCENT = 10
DPO_REFERENCE_FILE = "dpo/dpo_reference.json"
PAIR_SCORES_FILE = "pair_logps.jsonl"  # eval.sh: the held-out pairs scored per stack (D-862)
STACK_CHOICE_JSON = "eval/stack_choice.json"
STACK_CHOICE_STATE = "state/stack_choice"


@dataclass(frozen=True)
class ConnectionReport:
    target: str
    gpu: str
    disk: str


@dataclass(frozen=True)
class StepReport:
    step: str
    ok: bool
    detail: str


class StepFailedError(RemoteError):
    """A script on the instance ended with an error."""


DPO_RECORD_KEYS = (
    "mode",
    "reference",
    "reference_quantization_bit",
    "fallback",
    "fallback_reason",
    "per_device_batch",
    "accumulation",
    "pairs",
    "held_out_pairs",
    "after_cutoff_pairs",
)


def dpo_record_fields(reference: Mapping[str, Any]) -> dict[str, Any]:
    """The part of ``dpo_reference.json`` that goes into the record of the ``dpo`` step."""
    return {key: reference.get(key) for key in DPO_RECORD_KEYS}


def dpo_pairs_line(reference: Mapping[str, Any]) -> str:
    """How many pairs the preference run trained on, held out and left for later (D-860/861)."""
    return (
        f"dpo: trained on {reference.get('pairs')} preference pairs; "
        f"{reference.get('held_out_pairs') or 0} held out for the stack choice; "
        f"{reference.get('after_cutoff_pairs') or 0} made after the hold-out cut-off wait for a "
        "later export"
    )


def dpo_reference_line(reference: Mapping[str, Any]) -> str:
    """One line about the reference model of the preference run; a fall-back is said plainly."""
    if reference.get("fallback"):
        return (
            "dpo: FALLBACK - the reference model is the BASE (adapter switched off), not the SFT "
            f"model: {reference.get('fallback_reason')}"
        )
    bits = reference.get("reference_quantization_bit")
    detail = f" (explicit, {bits}-bit)" if bits else " (the merged SFT model, no second copy)"
    return f"dpo: the reference model is the SFT model{detail}"


async def inspect_instance(session: RemoteSession, layout: RemoteLayout) -> ConnectionReport:
    """The GPU and the data disk of the instance."""
    query = (
        "nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader"
    )
    gpu = await session.run(query)
    disk_root = layout.workdir.rsplit("/", 1)[0] or "/"
    disk = await session.run(f"df -Pk {shlex.quote(disk_root)}")
    gpu_line = (
        gpu.stdout.strip().splitlines()[0] if gpu.ok and gpu.stdout.strip() else "no GPU found"
    )
    disk_line = (
        disk.stdout.strip().splitlines()[-1] if disk.ok and disk.stdout.strip() else "unknown"
    )
    return ConnectionReport(session.target.label, gpu_line, disk_line)


class RemoteSteps:
    """The remote procedure of one run."""

    def __init__(
        self,
        *,
        session: RemoteSession,
        layout: RemoteLayout,
        profile: TrainingProfile,
        store: RunStore,
        clock: Clock,
        models_dir: Path,
        echo: Callable[[str], None],
        passphrase: Callable[[], str] | None = None,
        stack_judge: StackJudge | None = None,
        shell: str = "bash",
        poll_seconds: float = 2.0,
    ) -> None:
        self._stack_judge = stack_judge
        self._session = session
        self._layout = layout
        self._profile = profile
        self._store = store
        self._clock = clock
        self._models_dir = models_dir
        self._echo = echo
        self._passphrase = passphrase
        self._shell = shell
        self._jobs = RemoteJobs(session, layout, clock, poll_seconds=poll_seconds)

    # ----------------------------------------------------------------------- helpers

    def _script(self, name: str, *args: str) -> list[str]:
        return [self._shell, f"{self._layout.scripts}/{name}", self._profile.name, *args]

    def _env(self, run_id: str) -> dict[str, str]:
        return {"TWIN_HOME": self._layout.workdir, "TWIN_RUN_ID": run_id}

    def _emit(self, text: str) -> None:
        self._echo(text.rstrip("\n"))

    async def _run_job(
        self, run_id: str, step: str, script: str, *args: str, again: bool = False
    ) -> JobOutcome:
        """Run ``script`` as a background job, re-attaching to or reusing an earlier run of it.

        The log offset of the earlier attempt is carried over only when that attempt is the job
        this call goes on with: one that still runs (re-attached) or one that already ended well
        (its result is used, nothing starts).  A job that failed, was lost or is run ``again`` is
        started anew, and ``remote_job.py start`` truncates the log of a finished job - an old
        offset would point past the end of the new log and nothing would be shown until the new
        log outgrew the old one (D-636).
        """
        name = f"{run_id}-{step}"
        state = await self._jobs.status(name)
        finished_well = state.state == "exited" and state.exit_code == 0 and not again
        continues = (state.state == "running" and not again) or finished_well
        previous = self._store.get(run_id).steps.get(step, {})
        offset = int(previous.get("log_offset", 0)) if continues else 0
        self._store.begin_step(run_id, step, self._clock.now_utc(), keep_offset=continues)
        if finished_well:
            self._echo(f"{step}: the job already finished on the instance; using its result")
            outcome = JobOutcome(name, "exited", 0, offset)
        else:
            if again and state.state == "running":
                await self._jobs.stop(name)
            outcome = await self._jobs.run(
                name,
                self._script(script, *args),
                env=self._env(run_id),
                offset=offset,
                emit=self._emit,
                on_offset=lambda value: self._store.set_step_fields(run_id, step, log_offset=value),
            )
        return outcome

    def _finish(
        self,
        run_id: str,
        step: str,
        outcome: JobOutcome,
        fields: Mapping[str, Any] | None = None,
    ) -> StepReport:
        error = None
        if not outcome.ok:
            error = (
                f"{step} was lost on the instance (restarted?)"
                if outcome.state == "lost"
                else f"{step} exited with code {outcome.exit_code}"
            )
        self._store.end_step(
            run_id,
            step,
            self._clock.now_utc(),
            exit_code=outcome.exit_code,
            error=error,
            fields=fields,
        )
        if error:
            raise StepFailedError(
                f"{error}; the log was printed above, fix it and run the step again"
            )
        return StepReport(step, True, "done")

    async def _read_json(self, remote: str) -> dict[str, Any] | None:
        async def action() -> str | None:
            files = await self._session.files()
            return await files.read_text(remote)

        text = await self._session.retrying(action, what=f"reading {remote}")
        if text is None:
            return None
        value = json.loads(text)
        return value if isinstance(value, dict) else None

    # ------------------------------------------------------------------------ steps

    async def connect(self) -> ConnectionReport:
        return await inspect_instance(self._session, self._layout)

    async def upload(self, run_id: str, bundle_path: Path) -> StepReport:
        """Scripts first (plain), then the encrypted package; both verified."""
        minimum = int(self._store.get(run_id).hyperparameters.get("dpo_min_pairs", 200))
        self._store.begin_step(run_id, "upload", self._clock.now_utc())
        try:
            # the same profile.env the package was built with, or the instance refuses the scripts
            plain = bundle_module.plain_upload_files(self._profile, dpo_min_pairs=minimum)
            for relative, content in plain.items():
                data = content if isinstance(content, bytes) else content.read_bytes()
                remote = f"{self._layout.workdir}/{relative}"

                async def put(
                    remote: str = remote, data: bytes = data, relative: str = relative
                ) -> None:
                    handle = await self._session.files()
                    await handle.write_bytes(
                        remote, data, executable=relative.endswith((".sh", ".py"))
                    )

                await self._session.retrying(put, what=f"uploading {relative}")
            self._session.use_hash_tool(f"{self._layout.tools}/file_hash.py")

            async def make_directories() -> None:
                handle = await self._session.files()
                for directory in (self._layout.jobs, self._layout.logs):
                    await handle.makedirs(directory)

            await self._session.retrying(make_directories, what="preparing the directories")
            last = -1

            def progress(done: int, total: int) -> None:
                nonlocal last
                percent = (
                    done * 100 // max(total, 1) // PROGRESS_STEP_PERCENT * PROGRESS_STEP_PERCENT
                )
                if percent != last:
                    last = percent
                    self._echo(
                        f"upload: {percent}% ({done // (1 << 20)} of {total // (1 << 20)} MiB)"
                    )

            async def send() -> Any:
                handle = await self._session.files()
                return await handle.upload(bundle_path, self._layout.bundle, progress=progress)

            result = await self._session.retrying(send, what="uploading the package")
        except RemoteError as exc:
            self._store.end_step(
                run_id, "upload", self._clock.now_utc(), exit_code=1, error=str(exc)
            )
            raise
        run = self._store.get(run_id)
        if run.bundle_sha256 and run.bundle_sha256 != result.sha256:
            error = "the uploaded package is not the one the run was created with"
            self._store.end_step(run_id, "upload", self._clock.now_utc(), exit_code=1, error=error)
            raise RemoteError(error)
        self._store.end_step(
            run_id,
            "upload",
            self._clock.now_utc(),
            exit_code=0,
            fields={"bundle_sha256": result.sha256, "bundle_size": result.size},
        )
        detail = (
            "already on the instance"
            if result.skipped
            else f"{result.bytes_sent} bytes sent"
            + (f", resumed from {result.resumed_from}" if result.resumed_from else "")
        )
        return StepReport("upload", True, detail)

    async def setup(self, run_id: str, *, again: bool = False) -> StepReport:
        """Install (``setup``), decrypt, verify; the passphrase is asked for only here."""
        if self._passphrase is None:
            raise RemoteError("no way to ask for the passphrase")
        install = await self._run_job(run_id, "setup", "setup.sh", "install", again=again)
        self._finish(run_id, "setup", install)
        passphrase = self._passphrase()
        decrypt = " ".join(shlex.quote(part) for part in self._script("decrypt.sh"))
        command = f"env TWIN_HOME={shlex.quote(self._layout.workdir)} {decrypt}"
        self._store.begin_step(run_id, "decrypt", self._clock.now_utc())

        async def unpack() -> Any:
            return await self._session.run(command, stdin=passphrase + "\n", limit_s=3600.0)

        result = await self._session.retrying(unpack, what="decrypting the package")
        self._emit(result.stdout)
        if not result.ok:
            self._emit(result.stderr)
            self._emit(f"decrypt.sh exited with code {result.status}")
            self._store.end_step(
                run_id,
                "decrypt",
                self._clock.now_utc(),
                exit_code=result.status,
                error=f"decrypt.sh exited with code {result.status}",
            )
            raise StepFailedError(
                "decrypting failed (a wrong passphrase gives 'wrong passphrase, or the package "
                "is damaged'); run `twin train remote setup` again"
            )
        self._store.end_step(run_id, "decrypt", self._clock.now_utc(), exit_code=0)
        verify = await self._run_job(run_id, "verify", "setup.sh", "verify", again=again)
        return self._finish(run_id, "verify", verify)

    async def train(self, run_id: str, *, again: bool = False) -> StepReport:
        # --again: train anew - LLaMA-Factory would otherwise resume the last checkpoint (D-813)
        outcome = await self._run_job(
            run_id, "train", "train.sh", *(["--again"] if again else []), again=again
        )
        fields: dict[str, Any] = {}
        if outcome.ok:
            metrics = await self._read_json(f"{self._layout.artifacts}/train/metrics.json") or {}
            fields = {
                "best_val_loss": metrics.get("best_val_loss"),
                "best_checkpoint": metrics.get("best_checkpoint"),
                "peak_vram_mib": metrics.get("peak_vram_mib"),
                "gpu_model": metrics.get("gpu_name"),
            }
        return self._finish(run_id, "train", outcome, fields)

    async def dpo(self, run_id: str, *, again: bool = False) -> StepReport:
        # --again: train anew; a plain rerun keeps a finished preference run (D-813)
        outcome = await self._run_job(
            run_id, "dpo", "dpo.sh", *(["--again"] if again else []), again=again
        )
        if outcome.ok:
            reference = await self._read_json(f"{self._layout.artifacts}/{DPO_REFERENCE_FILE}")
            if reference:  # no record: the step was skipped (too few pairs)
                self._store.set_step_fields(run_id, "dpo", **dpo_record_fields(reference))
                self._emit(dpo_reference_line(reference))
                self._emit(dpo_pairs_line(reference))
        return self._finish(run_id, "dpo", outcome)

    async def evaluate(self, run_id: str, *, again: bool = False) -> StepReport:
        outcome = await self._run_job(run_id, "eval", "eval.sh", "auto", again=again)
        if outcome.ok:
            try:
                verdict = await self._choose_stack()
            except (RemoteError, StackChoiceError) as exc:
                self._store.end_step(
                    run_id, "eval", self._clock.now_utc(), exit_code=1, error=str(exc)
                )
                raise StepFailedError(
                    f"the two adapter stacks could not be compared: {exc}; "
                    "fix it and run the evaluation again"
                ) from exc
            if verdict is not None:
                self._store.set_step_fields(
                    run_id,
                    "eval",
                    stack_kept=verdict.keep,
                    stack_reason=verdict.reason,
                    stack_verdict=verdict.to_json(),
                )
                for line in verdict.lines():
                    self._emit(line)
        return self._finish(run_id, "eval", outcome)

    async def _choose_stack(self) -> StackVerdict | None:
        """Compare the stacks ``eval.sh`` evaluated and tell the instance which one is kept.

        ``None`` when only the SFT stack was evaluated (there is no DPO adapter): nothing to
        choose, and ``export.sh`` takes the SFT stack by itself.
        """
        base = f"{self._layout.artifacts}/eval"
        dpo_metrics = await self._read_json(f"{base}/dpo/eval_metrics.json")
        if dpo_metrics is None:
            return None
        if self._stack_judge is None:
            raise RemoteError("no judge for the adapter stacks was given")
        texts: dict[str, str] = {}
        pair_scores: dict[str, str] = {}
        metrics: dict[str, dict[str, Any]] = {"dpo": dpo_metrics}
        for stack in STACKS:
            remote = f"{base}/{stack}/eval_generations.jsonl"
            scored = f"{base}/{stack}/{PAIR_SCORES_FILE}"

            async def read(remote: str = remote) -> str | None:
                return await self._read_text(remote)

            async def read_scores(scored: str = scored) -> str | None:
                return await self._read_text(scored)

            text = await self._session.retrying(
                read, what=f"reading the generations of the {stack} stack"
            )
            if text is None:
                raise RemoteError(f"the instance has no {remote}")
            texts[stack] = text
            # the held-out preference pairs scored under the stack: none in an older package
            scores = await self._session.retrying(
                read_scores, what=f"reading the held-out pair scores of the {stack} stack"
            )
            if scores is not None:
                pair_scores[stack] = scores
        sft_metrics = await self._read_json(f"{base}/sft/eval_metrics.json")
        metrics["sft"] = sft_metrics or {}
        reference = await self._read_json(f"{self._layout.artifacts}/{DPO_REFERENCE_FILE}")
        self._echo(
            "comparing the sft and the dpo stack: the held-out preference pairs and a paired "
            f"bootstrap over the test contexts ({RESAMPLES} draws; a few minutes for thousands "
            "of contexts)"
        )
        verdict = self._stack_judge(StackFiles(texts, metrics, pair_scores, reference))

        async def write() -> None:
            handle = await self._session.files()
            await handle.makedirs(f"{self._layout.workdir}/state")
            await handle.write_bytes(
                f"{self._layout.workdir}/{STACK_CHOICE_STATE}", f"{verdict.keep}\n".encode()
            )
            payload = json.dumps(verdict.to_json(), indent=2) + "\n"
            await handle.write_bytes(
                f"{self._layout.artifacts}/{STACK_CHOICE_JSON}", payload.encode("utf-8")
            )

        await self._session.retrying(write, what="writing the stack that was kept")
        return verdict

    async def export(self, run_id: str, *, again: bool = False) -> StepReport:
        outcome = await self._run_job(
            run_id, "export", "export.sh", *(["--redo"] if again else []), again=again
        )
        fields: dict[str, Any] = {}
        if outcome.ok:
            manifest = await self._read_json(f"{self._layout.artifacts}/{ARTIFACT_MANIFEST}") or {}
            fields = {
                "artifacts": {
                    entry["path"]: {"sha256": entry["sha256"], "size": entry["size"]}
                    for entry in manifest.get("files", [])
                }
            }
        return self._finish(run_id, "export", outcome, fields)

    async def download(self, run_id: str) -> tuple[StepReport, Path]:
        """Fetch every file of ``artifacts/manifest.json`` into ``data/models/<run_id>/``."""
        self._store.begin_step(run_id, "download", self._clock.now_utc())
        target = self._models_dir / run_id
        try:
            manifest = await self._read_json(f"{self._layout.artifacts}/{ARTIFACT_MANIFEST}")
            if manifest is None:
                raise RemoteError(
                    "the instance has no artifacts/manifest.json; run the export first"
                )
            entries = manifest.get("files", [])
            total = sum(int(entry["size"]) for entry in entries)
            self._echo(f"download: {len(entries)} files, {total // (1 << 20)} MiB, into {target}")
            for entry in entries:
                relative = str(entry["path"])
                if relative.startswith(("/", "..")) or ".." in relative.split("/"):
                    raise RemoteError(
                        f"the manifest names a path outside the artifacts: {relative}"
                    )
                remote = f"{self._layout.artifacts}/{relative}"

                async def fetch(
                    remote: str = remote, relative: str = relative, entry: Any = entry
                ) -> Any:
                    handle = await self._session.files()
                    return await handle.download(
                        remote,
                        target / relative,
                        size=int(entry["size"]),
                        sha256=str(entry["sha256"]),
                    )

                result = await self._session.retrying(fetch, what=f"downloading {relative}")
                self._echo(f"download: {relative} {'(already here)' if result.skipped else 'ok'}")
            (target / ARTIFACT_MANIFEST).write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
            )
            verify_artifacts(target, load_manifest(target))
        except (RemoteError, RegistryError) as exc:
            self._store.end_step(
                run_id, "download", self._clock.now_utc(), exit_code=1, error=str(exc)
            )
            raise
        self._store.end_step(
            run_id,
            "download",
            self._clock.now_utc(),
            exit_code=0,
            fields={
                "artifacts": {
                    entry["path"]: {"sha256": entry["sha256"], "size": entry["size"]}
                    for entry in manifest["files"]
                }
            },
        )
        return StepReport("download", True, f"{len(entries)} files verified"), target

    async def cleanup(self, run_id: str, *, force: bool = False) -> StepReport:
        """Erase the data on the instance; only after the download, unless ``force``."""
        run = self._store.get(run_id)
        if not force and not run.steps.get("download", {}).get("ok"):
            raise RemoteError(
                "the artifacts have not been downloaded yet; cleaning up now would lose them. "
                "Run `twin train remote download` first (or use --force to discard them)"
            )
        self._store.begin_step(run_id, "cleanup", self._clock.now_utc())
        script = " ".join(
            shlex.quote(part)
            for part in [self._shell, f"{self._layout.scripts}/cleanup.sh", "--yes"]
        )
        command = f"env TWIN_HOME={shlex.quote(self._layout.workdir)} {script}"

        async def erase() -> Any:
            return await self._session.run(command, limit_s=3600.0)

        result = await self._session.retrying(erase, what="cleaning up")
        self._emit(result.stdout)
        done = await self._session.retrying(
            lambda: self._read_text(f"{self._layout.workdir}/state/cleanup.done"),
            what="checking the cleanup",
        )
        if not result.ok or done is None:
            self._emit(result.stderr)
            self._store.end_step(
                run_id,
                "cleanup",
                self._clock.now_utc(),
                exit_code=result.status or 1,
                error="cleanup.sh did not finish; run the cleanup again",
            )
            raise StepFailedError(
                "the cleanup did not finish; run `twin train remote cleanup` again"
            )
        self._store.end_step(run_id, "cleanup", self._clock.now_utc(), exit_code=0)
        return StepReport("cleanup", True, "release the instance in the AutoDL console")

    async def _read_text(self, remote: str) -> str | None:
        handle = await self._session.files()
        return await handle.read_text(remote)

    async def remote_states(self, run: RunView) -> dict[str, str]:
        """``{step: job state}`` of the jobs of ``run`` on the instance."""
        states: dict[str, str] = {}
        for step in ("setup", "verify", "train", "dpo", "eval", "export"):
            status = await self._jobs.status(f"{run.id}-{step}")
            if status.state != "none":
                states[step] = status.state + (
                    f" (exit {status.exit_code})" if status.exit_code is not None else ""
                )
        return states
