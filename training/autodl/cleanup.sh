#!/usr/bin/env bash
# Remove everything that holds her messages from the instance (R-TRN-009, R-PRIV-003).
#
#   cleanup.sh --yes [--all]
#
# Overwrites and deletes the decrypted dataset, the unpack directory, the encrypted package, the
# logs and job logs (they contain training samples), the processed-dataset caches, the
# checkpoints, the copies of the longest samples that the trial runs of setup.sh verify and of
# dpo.sh train on and the rendered configuration.  It runs once the artifacts are downloaded
# (`twin train remote cleanup` makes sure of that), so of artifacts/ only the adapter the remote
# vLLM serves stays (artifacts/adapter, R-SRV-003): the GGUF files, the adapter layers, the
# manifest, the training and DPO reports and the evaluation - eval/*/eval_generations.jsonl are the
# model's replies to her held-out contexts - go (D-814).  --all removes that adapter as well.  The
# export state of the run and the stack the evaluation kept go too, so that a second run on this
# instance starts from nothing of this one (D-812).  Run it again at any time: it only removes
# what is still there.  The base model and the installed programs stay; the data disk is
# reclaimed when the instance is released in the AutoDL console.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "${SCRIPT_DIR}/lib.sh"

CONFIRMED=0
INCLUDE_ARTIFACTS=0
for argument in "$@"; do
    case "${argument}" in
        --yes) CONFIRMED=1 ;;
        --all) INCLUDE_ARTIFACTS=1 ;;
        *) die "unknown argument '${argument}'" ;;
    esac
done
[ "${CONFIRMED}" = "1" ] || die "this deletes the training data for good; run it with --yes once the artifacts are downloaded"

ARTIFACTS="${TWIN_HOME}/artifacts"
SERVED="${ARTIFACTS}/adapter"
TARGETS=(
    "${TWIN_HOME}/data"
    "${TWIN_HOME}/.unpack"
    "${TWIN_HOME}/bundle.enc"
    "${TWIN_HOME}/bundle.tar.zst"
    "${TWIN_HOME}/logs"
    "${TWIN_HOME}/jobs"
    "${TWIN_HOME}/output"
    "${TWIN_HOME}/config"
    "${TWIN_HOME}/pylib"
    "${TWIN_HOME}/manifest.json"
    "${TWIN_HOME}/gguf"
    "${TWIN_HOME}/trial_data"
    "${TWIN_HOME}/trial_data_dpo"
    "${TWIN_DISK_ROOT}/cache/huggingface/datasets"
    "${TWIN_DISK_ROOT}/cache/modelscope/datasets"
    "${TWIN_DISK_ROOT}/tmp"
)
# The quantized models hold no message text, and overwriting 20 to 80 GB would take longer than
# the step may: they are deleted without being overwritten (the disk goes with the instance).
if [ -d "${ARTIFACTS}/gguf" ]; then
    log "deleting ${ARTIFACTS}/gguf"
    rm -rf "${ARTIFACTS}/gguf"
fi
if [ "${INCLUDE_ARTIFACTS}" = "1" ]; then
    TARGETS+=("${ARTIFACTS}")
elif [ -d "${ARTIFACTS}" ]; then
    while IFS= read -r -d '' entry; do
        if [ "${entry}" != "${SERVED}" ]; then
            TARGETS+=("${entry}")
        fi
    done < <(find "${ARTIFACTS}" -mindepth 1 -maxdepth 1 -print0)
fi

for target in "${TARGETS[@]}"; do
    if [ -e "${target}" ]; then
        log "erasing ${target}"
        secure_delete "${target}"
    fi
done
mkdir -p "${TWIN_HOME}/state"
reset_export_state
rm -f "${TWIN_HOME}/state/stack_choice"
date -u +%Y-%m-%dT%H:%M:%SZ >"${TWIN_HOME}/state/cleanup.done"
if [ -d "${SERVED}" ]; then
    log "cleanup finished; kept: artifacts/adapter (the LoRA adapter the remote vLLM serves; cleanup.sh --yes --all removes it), the base model and the installed programs"
else
    log "cleanup finished; kept: nothing of the run (the base model and the installed programs stay until the instance is released)"
fi
log "请到 AutoDL 控制台释放实例 (release the instance in the AutoDL console; the data disk is only reclaimed after the release)"
