#!/usr/bin/env bash
# Decrypt the training package into the work directory and verify it (R-TRN-009).
#
#   printf '%s\n' "$PASSPHRASE" | decrypt.sh <profile>
#
# The passphrase comes from standard input only.  The archive is decrypted, decompressed and
# unpacked in one pipeline, so no readable copy of it is ever written to disk.  Running it again
# replaces the previous data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "${SCRIPT_DIR}/lib.sh"

load_env_files "${SCRIPT_DIR}"
require_profile "${1:-}"
start_logging decrypt

BUNDLE="${TWIN_HOME}/bundle.enc"
STAGE="${TWIN_HOME}/.unpack"

[ -f "${BUNDLE}" ] || die "${BUNDLE} is missing; upload the package first"

secure_delete "${STAGE}"
mkdir -p "${STAGE}"
"${TWIN_PYTHON}" "${SCRIPT_DIR}/tools/decrypt_bundle.py" --in "${BUNDLE}" --out - |
    zstd_decompress |
    tar -x --no-same-owner -C "${STAGE}"
"${TWIN_PYTHON}" "${SCRIPT_DIR}/tools/verify_manifest.py" --root "${STAGE}" --scripts "${SCRIPT_DIR}"

for name in data config pylib; do
    secure_delete "${TWIN_HOME:?}/${name}"
    mv "${STAGE}/${name}" "${TWIN_HOME}/${name}"
done
mv -f "${STAGE}/manifest.json" "${TWIN_HOME}/manifest.json"
secure_delete "${STAGE}"
log "package decrypted and verified: $(wc -l <"${TWIN_HOME}/data/sft_train.jsonl") training samples"
