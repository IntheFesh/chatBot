#!/usr/bin/env bash
# Evaluate the trained adapter (R-TRN-009).
#
#   eval.sh <profile> [sft|dpo|auto]
#
# 1. the loss on the validation set;
# 2. one generated reply for every context of the test set, with the fixed temperature, top_p and
#    seed of config/eval_generate.yaml.  artifacts/eval/eval_generations.jsonl holds only
#    {"id", "text"} per line.
# "auto" (the default) evaluates the DPO adapter when there is one, otherwise the SFT adapter.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "${SCRIPT_DIR}/lib.sh"

load_env_files "${SCRIPT_DIR}"
require_profile "${1:-}"
start_logging eval
use_data_disk

WHICH="${2:-auto}"
ADAPTER="$(best_adapter "${WHICH}")" || die "unknown adapter '${WHICH}' (sft, dpo or auto)"
[ -f "${ADAPTER}/adapter_model.safetensors" ] || die "no adapter in ${ADAPTER}; run train.sh first"
EVAL_DIR="${TWIN_HOME}/artifacts/eval"
mkdir -p "${EVAL_DIR}"

for kind in eval_loss eval_generate; do
    run_config="${TWIN_HOME}/config/${kind}.run.yaml"
    cp "${TWIN_HOME}/config/${kind}.yaml" "${run_config}"
    yaml_set "${run_config}" adapter_name_or_path "${ADAPTER}"
    log "${kind} with the adapter ${ADAPTER##*/}"
    llamafactory-cli train "${run_config}"
done

"${TWIN_PYTHON}" "${SCRIPT_DIR}/tools/make_eval_generations.py" \
    --test "${TWIN_HOME}/data/sft_test.jsonl" \
    --predictions "${TWIN_HOME}/output/eval_generate/generated_predictions.jsonl" \
    --eval-results "${TWIN_HOME}/output/eval_loss/eval_results.json" \
    --adapter "${ADAPTER##*/}" \
    --out-dir "${EVAL_DIR}"
