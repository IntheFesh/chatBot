#!/usr/bin/env bash
# Evaluate the trained adapter stacks (R-TRN-009; D-650, D-655).
#
#   eval.sh <profile> [sft|dpo|auto]
#
# For every stack under evaluation (lib.sh, "adapter stacks": sft is the SFT adapter, dpo is what
# the preference run added to it - both adapters of a bf16 profile, the continued one of the 4-bit
# profile):
# 1. the loss on the validation set;
# 2. one generated reply for every context of the test set, with the fixed temperature, top_p and
#    seed of config/eval_generate.yaml.
# artifacts/eval/<stack>/eval_generations.jsonl holds only {"id", "text"} per line and
# eval_metrics.json the validation loss.
# "auto" (the default) evaluates sft and, when a preference run left an adapter, dpo as well: the
# same test contexts and validation set, so that `twin train remote eval` can compare the two and
# keep the better one (it writes state/stack_choice, which export.sh reads).  A verdict of an
# earlier evaluation is deleted here: it would describe other generations.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "${SCRIPT_DIR}/lib.sh"

load_env_files "${SCRIPT_DIR}"
require_profile "${1:-}"
start_logging eval
use_data_disk

WHICH="${2:-auto}"
case "${WHICH}" in
    sft | dpo) STACKS=("${WHICH}") ;;
    auto)
        STACKS=(sft)
        if dpo_adapter_exists; then STACKS+=(dpo); fi
        ;;
    *) die "unknown adapter '${WHICH}' (sft, dpo or auto)" ;;
esac
for stack in "${STACKS[@]}"; do
    require_adapters "$(stack_adapters "${stack}")"
done
EVAL_DIR="${TWIN_HOME}/artifacts/eval"
mkdir -p "${EVAL_DIR}" "${TWIN_HOME}/state"
rm -f "${TWIN_HOME}/state/stack_choice" "${EVAL_DIR}/stack_choice.json"

evaluate_stack() {
    local stack="$1" adapters kind run_config
    adapters="$(stack_adapters "${stack}")"
    for kind in eval_loss eval_generate; do
        run_config="${TWIN_HOME}/config/${kind}.run.yaml"
        cp "${TWIN_HOME}/config/${kind}.yaml" "${run_config}"
        yaml_set "${run_config}" adapter_name_or_path "${adapters}"
        yaml_set "${run_config}" output_dir "${TWIN_HOME}/output/${kind}_${stack}"
        log "${kind} with the adapter stack ${stack} (${adapters//${TWIN_HOME}\//})"
        llamafactory-cli train "${run_config}"
    done
    "${TWIN_PYTHON}" "${SCRIPT_DIR}/tools/make_eval_generations.py" \
        --test "${TWIN_HOME}/data/sft_test.jsonl" \
        --predictions "${TWIN_HOME}/output/eval_generate_${stack}/generated_predictions.jsonl" \
        --eval-results "${TWIN_HOME}/output/eval_loss_${stack}/eval_results.json" \
        --adapter "${stack}" \
        --out-dir "${EVAL_DIR}/${stack}"
}

for stack in "${STACKS[@]}"; do
    evaluate_stack "${stack}"
done
