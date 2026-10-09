#!/usr/bin/env bash
# Remove everything that holds her messages from the instance (R-TRN-009, R-PRIV-003).
#
#   cleanup.sh --yes [--all]
#
# Overwrites and deletes the decrypted dataset, the unpack directory, the encrypted package, the
# logs and job logs (they contain training samples), the processed-dataset caches, the checkpoints and the
# rendered configuration.  artifacts/ is kept until the download has been confirmed; --all removes
# it as well.  Run it again at any time: it only removes what is still there.  The data disk is
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
    "${TWIN_DISK_ROOT}/cache/huggingface/datasets"
    "${TWIN_DISK_ROOT}/cache/modelscope/datasets"
    "${TWIN_DISK_ROOT}/tmp"
)
if [ "${INCLUDE_ARTIFACTS}" = "1" ]; then
    TARGETS+=("${TWIN_HOME}/artifacts")
fi

for target in "${TARGETS[@]}"; do
    if [ -e "${target}" ]; then
        log "erasing ${target}"
        secure_delete "${target}"
    fi
done
mkdir -p "${TWIN_HOME}/state"
date -u +%Y-%m-%dT%H:%M:%SZ >"${TWIN_HOME}/state/cleanup.done"
log "cleanup finished; the artifacts were $([ "${INCLUDE_ARTIFACTS}" = "1" ] && echo removed || echo kept)"
log "请到 AutoDL 控制台释放实例 (release the instance in the AutoDL console; the data disk is only reclaimed after the release)"
