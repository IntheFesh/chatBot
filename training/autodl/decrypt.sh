#!/usr/bin/env bash
# Decrypt the training package into the work directory and verify it (R-TRN-009).
#
#   printf '%s\n' "$PASSPHRASE" | decrypt.sh <profile>
#
# The passphrase comes from standard input only.  The archive is decrypted, decompressed and
# unpacked in one pipeline, so no readable copy of it is ever written to disk.  Running it again
# replaces the previous data - of the same package: while output/ holds the training of another
# package (another dataset or profile; a run without a cleanup), the new one is refused, or the
# training would resume from the old checkpoints (D-812).  Nothing is deleted on a guess: clean up
# the earlier run after its download, then decrypt again.
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

NEW_PACKAGE="$(package_identity "${STAGE}/manifest.json")"
if training_output_present; then
    OLD_PACKAGE="an unknown package"
    if [ -f "${TWIN_HOME}/manifest.json" ]; then
        OLD_PACKAGE="$(package_identity "${TWIN_HOME}/manifest.json")"
    fi
    if [ "${OLD_PACKAGE}" != "${NEW_PACKAGE}" ]; then
        secure_delete "${STAGE}"
        die "output/ holds the training of ${OLD_PACKAGE}, this package is ${NEW_PACKAGE}: download what you need of that run, clean it up (twin train remote cleanup --run <its run>, or cleanup.sh --yes here), then run the setup again"
    fi
fi

for name in data config pylib; do
    secure_delete "${TWIN_HOME:?}/${name}"
    mv "${STAGE}/${name}" "${TWIN_HOME}/${name}"
done
mv -f "${STAGE}/manifest.json" "${TWIN_HOME}/manifest.json"
secure_delete "${STAGE}"
log "package decrypted and verified: $(wc -l <"${TWIN_HOME}/data/sft_train.jsonl") training samples"
