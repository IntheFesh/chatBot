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
# 3. when a preference run left an adapter: the held-out preference pairs (data/dpo_holdout.jsonl,
#    the fifth `twin train export-dpo` kept out of the DPO training, D-861) scored by
#    tools/pair_logprobs.py - the base, the stack's adapters and the quantisation of
#    config/eval_loss.yaml, loaded as LLaMA-Factory loads them for the evaluation (D-865).
# artifacts/eval/<stack>/eval_generations.jsonl holds only {"id", "text"} per line,
# eval_metrics.json the validation loss and pair_logps.jsonl the pairs' ids and numbers (no text).
# "auto" (the default) evaluates sft and, when a preference run left an adapter, dpo as well: the
# same test contexts and validation set, so that `twin train remote eval` can compare the two and
# keep the better one by the rule of D-862 (it writes state/stack_choice, which export.sh reads).
# A verdict of an earlier evaluation is deleted here: it would describe other generations.  A
# package without held-out pairs (exported before round 18) leaves nothing to score: the stack
# choice then cannot show that DPO is better and keeps SFT.
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
HOLDOUT="${TWIN_HOME}/data/dpo_holdout.jsonl"
mkdir -p "${EVAL_DIR}" "${TWIN_HOME}/state"
rm -f "${TWIN_HOME}/state/stack_choice" "${EVAL_DIR}/stack_choice.json"

# score_holdout_pairs STACK: data/dpo_holdout.jsonl under the adapter stack STACK, into
# artifacts/eval/STACK/pair_logps.jsonl.  The model is the evaluation's: model_name_or_path and
# quantization_bit of config/eval_loss.yaml (bf16, as that file evaluates), the stack's adapters
# in the order lib.sh gives them; the template module comes from pylib/ (the one rendering of
# qwen3_nothink, D-865).
score_holdout_pairs() {
    local stack="$1" config="${TWIN_HOME}/config/eval_loss.yaml" bits
    local -a quantization=()
    [ "$(yaml_get "${config}" bf16)" = "true" ] ||
        die "${config} does not evaluate in bf16; the held-out pairs are scored the way it evaluates"
    bits="$(yaml_get "${config}" quantization_bit)"
    if [ -n "${bits}" ]; then quantization=(--quantization-bit "${bits}"); fi
    log "scoring the held-out preference pairs under the ${stack} stack"
    PYTHONPATH="${TWIN_HOME}/pylib" "${TWIN_PYTHON}" "${SCRIPT_DIR}/tools/pair_logprobs.py" \
        --pairs "${HOLDOUT}" \
        --base "$(yaml_get "${config}" model_name_or_path)" \
        --adapters "$(stack_adapters "${stack}")" \
        --dtype bfloat16 "${quantization[@]}" \
        --out "${EVAL_DIR}/${stack}/pair_logps.jsonl"
}

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
    rm -f "${EVAL_DIR}/${stack}/pair_logps.jsonl"
    if dpo_adapter_exists; then
        if [ -s "${HOLDOUT}" ]; then
            score_holdout_pairs "${stack}"
        else
            log "no held-out preference pairs in this package (data/dpo_holdout.jsonl): the stack choice will keep SFT, DPO cannot be shown better (export the pairs again with twin train export-dpo)"
        fi
    fi
}

for stack in "${STACKS[@]}"; do
    evaluate_stack "${stack}"
done
