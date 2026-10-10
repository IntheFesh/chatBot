# shellcheck shell=bash
# Shared functions of the AutoDL scripts (R-TRN-009).  Sourced by the other scripts, never run.
#
# Everything that decides something is a small function that takes its inputs as arguments and
# prints or returns the answer, so it can be tried out without a GPU: the disk and memory checks,
# the PyTorch/CUDA verdict, the batch size reduction, the YAML key edits and the order of the
# export steps.  The probes that read the machine (probe_*) are separate functions, so a test
# can replace them.

TWIN_HOME="${TWIN_HOME:-/root/autodl-tmp/twin}"
TWIN_MODELS="${TWIN_MODELS:-/root/autodl-tmp/models}"
TWIN_DISK_ROOT="${TWIN_DISK_ROOT:-/root/autodl-tmp}"
TWIN_SCRIPTS="${TWIN_SCRIPTS:-${TWIN_HOME}/autodl}"
TWIN_PYTHON="${TWIN_PYTHON:-python3}"

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

die() {
    log "ERROR: $*" >&2
    exit 1
}

# ---------------------------------------------------------------- configuration files

# load_env_files [dir]: source versions.env and profile.env (KEY=value lines written by twin).
load_env_files() {
    local dir="${1:-${TWIN_SCRIPTS}}"
    local file
    for file in versions.env profile.env; do
        [ -f "${dir}/${file}" ] || die "${dir}/${file} is missing; upload the scripts again"
        set -a
        # shellcheck source=/dev/null
        . "${dir}/${file}"
        set +a
    done
}

# require_profile NAME: the profile of profile.env must be the one asked for.
require_profile() {
    [ -n "${1:-}" ] || die "usage: $0 <profile> (5090-8b, 5090-14b, pro6000-14b or pro6000-32b)"
    [ "${TWIN_PROFILE:-}" = "$1" ] || die "profile.env is for '${TWIN_PROFILE:-?}', not '$1'"
}

# start_logging NAME: append everything this script prints to logs/NAME.log as well.
start_logging() {
    mkdir -p "${TWIN_HOME}/logs"
    if [ "${TWIN_LOG_TEE:-1}" = "1" ]; then
        exec > >(tee -a "${TWIN_HOME}/logs/$1.log") 2>&1
    fi
}

# ---------------------------------------------------------------------- pure checks

# version_ge A B: success when version A >= version B (dotted numbers; a +local suffix is ignored).
version_ge() {
    local a="${1%%+*}" b="${2%%+*}"
    [ "$(printf '%s\n%s\n' "${b}" "${a}" | sort -V | head -n1)" = "${b}" ]
}

# check_disk_space FREE_GB REQUIRED_GB SPEC_GB: success when the free space is enough.
check_disk_space() {
    local free="$1" required="$2" spec="$3"
    if [ "${free}" -ge "${required}" ]; then
        log "disk: ${free} GB available, ${required} GB needed - ok"
        return 0
    fi
    log "disk: only ${free} GB available but this profile needs ${required} GB (the data disk should be at least ${spec} GB)" >&2
    log "请到 AutoDL 控制台扩容数据盘/换实例 (expand the data disk or choose another instance), then run setup again" >&2
    return 1
}

# check_memory TOTAL_GB REQUIRED_GB: success when the instance has enough RAM.
check_memory() {
    local total="$1" required="$2"
    if [ "${total}" -ge "${required}" ]; then
        log "memory: ${total} GB available, ${required} GB needed - ok"
        return 0
    fi
    log "memory: ${total} GB available but this profile needs ${required} GB (merging the adapter loads the bf16 base model into RAM)" >&2
    log "请到 AutoDL 控制台换内存更大的实例 (choose an instance with more memory)" >&2
    return 1
}

# torch_stack_verdict TORCH_VERSION CUDA_VERSION ARCH_LIST CAPABILITY
# Prints "ok" or "reinstall: <reason>".  CUDA 13.x is never accepted.  On Blackwell (compute
# capability 12.0, sm_120) PyTorch must be at least TWIN_TORCH_MIN, built for CUDA TWIN_CUDA_SERIES
# and list sm_120 among its architectures.
torch_stack_verdict() {
    local torch_version="$1" cuda="$2" archs="$3" capability="$4"
    case "${torch_version}" in none | "") echo "reinstall: PyTorch is not installed"; return 0 ;; esac
    case "${cuda}" in
        none | "") echo "reinstall: PyTorch has no CUDA build"; return 0 ;;
        13 | 13.*) echo "reinstall: CUDA ${cuda} builds are not used (need ${TWIN_CUDA_SERIES})"; return 0 ;;
    esac
    if [ "${capability}" = "${TWIN_BLACKWELL_CAPABILITY}" ]; then
        if ! version_ge "${torch_version}" "${TWIN_TORCH_MIN}"; then
            echo "reinstall: Blackwell needs torch >= ${TWIN_TORCH_MIN}, found ${torch_version}"
            return 0
        fi
        case "${cuda}" in
            "${TWIN_CUDA_SERIES}" | "${TWIN_CUDA_SERIES}".*) ;;
            *) echo "reinstall: Blackwell needs a CUDA ${TWIN_CUDA_SERIES} build, found ${cuda}"; return 0 ;;
        esac
        case ",${archs}," in
            *",${TWIN_BLACKWELL_ARCH},"*) ;;
            *) echo "reinstall: this PyTorch build does not list ${TWIN_BLACKWELL_ARCH} (${archs})"; return 0 ;;
        esac
    elif ! version_ge "${torch_version}" "2.4"; then
        echo "reinstall: LLaMA-Factory needs torch >= 2.4, found ${torch_version}"
        return 0
    fi
    echo "ok"
}

# next_smaller_batch BATCH ACCUM: the next layout that keeps batch*accum the same with half the
# per-device batch ("4 4" -> "2 8" -> "1 16"); fails when the batch is already 1.
next_smaller_batch() {
    local batch="$1" accum="$2"
    if [ "${batch}" -le 1 ]; then
        return 1
    fi
    echo "$((batch / 2)) $((accum * 2))"
}

# yaml_get FILE KEY: the value of a top-level "key: value" line.
yaml_get() {
    sed -n "s/^$2:[[:space:]]*//p" "$1" | head -n1
}

# yaml_set FILE KEY VALUE: replace the top-level key, or append it when absent.
yaml_set() {
    local file="$1" key="$2" value="$3" tmp
    tmp="$(mktemp "${file}.XXXXXX")"
    if grep -q "^${key}:" "${file}"; then
        awk -v key="${key}" -v value="${value}" \
            'index($0, key ":") == 1 { print key ": " value; next } { print }' "${file}" >"${tmp}"
    else
        cat "${file}" >"${tmp}"
        printf '%s: %s\n' "${key}" "${value}" >>"${tmp}"
    fi
    mv "${tmp}" "${file}"
}

# yaml_drop FILE KEY: remove the top-level key.
yaml_drop() {
    local file="$1" key="$2" tmp
    tmp="$(mktemp "${file}.XXXXXX")"
    awk -v key="${key}" 'index($0, key ":") != 1 { print }' "${file}" >"${tmp}"
    mv "${tmp}" "${file}"
}

# oom_in_log FILE: success when the log shows a CUDA out-of-memory failure.
oom_in_log() {
    grep -qE 'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED' "$1"
}

# best_adapter [sft|dpo|auto]: the adapter directory an evaluation or an export uses.
best_adapter() {
    local which="${1:-auto}" dpo="${TWIN_HOME}/output/dpo" sft="${TWIN_HOME}/output/sft"
    case "${which}" in
        sft) echo "${sft}" ;;
        dpo) echo "${dpo}" ;;
        auto)
            if [ -f "${dpo}/adapter_model.safetensors" ]; then echo "${dpo}"; else echo "${sft}"; fi
            ;;
        *) return 1 ;;
    esac
}

# ------------------------------------------------------------------------- probes
# The probes below read the machine.  When one of them misreads a container, point
# TWIN_PROBE_OVERRIDES at a file that redefines it; the file is sourced after lib.sh.

# probe_free_gb PATH: whole GB free on the file system of PATH.
probe_free_gb() {
    df -Pk "$1" | awk 'NR == 2 { print int($4 / 1048576) }'
}

# probe_present_gb: GB the project already occupies (an interrupted setup is resumed, not doubled).
probe_present_gb() {
    local total=0 path kb
    for path in "${TWIN_HOME}" "${TWIN_MODELS}"; do
        if [ -d "${path}" ]; then
            kb="$(du -sk "${path}" 2>/dev/null | awk '{ print $1 }')"
            total=$((total + ${kb:-0}))
        fi
    done
    echo $((total / 1048576))
}

# probe_ram_gb: GB of memory this container may use (the smaller of the host and the cgroup limit).
probe_ram_gb() {
    local host_kb limit="" bytes
    host_kb="$(awk '/^MemTotal:/ { print $2 }' /proc/meminfo)"
    for bytes in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
        if [ -r "${bytes}" ]; then
            limit="$(cat "${bytes}")"
            break
        fi
    done
    case "${limit}" in "" | max | *[!0-9]*) limit="" ;; esac
    local host_gb=$((host_kb / 1048576))
    if [ -n "${limit}" ] && [ "$((limit / 1073741824))" -lt "${host_gb}" ]; then
        echo $((limit / 1073741824))
    else
        echo "${host_gb}"
    fi
}

# probe_gpu: "name,memory.total,driver_version,compute_cap" of the first GPU.
probe_gpu() {
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader | head -n1
}

# probe_torch: "version cuda arch,arch,..." of the installed PyTorch, or "none none ".
probe_torch() {
    "${TWIN_PYTHON}" - <<'PY' 2>/dev/null || echo "none none "
import torch

arch = ",".join(torch.cuda.get_arch_list()) if torch.version.cuda else ""
print(torch.__version__, torch.version.cuda or "none", arch)
PY
}

# ------------------------------------------------------------------ combined checks

# run_resource_checks: the disk and memory checks of the profile in profile.env.
run_resource_checks() {
    local free present
    free="$(probe_free_gb "${TWIN_DISK_ROOT}")"
    present="$(probe_present_gb)"
    check_disk_space "$((free + present))" "${TWIN_REQUIRED_DISK_GB}" "${TWIN_SPEC_DISK_GB}" || return 1
    check_memory "$(probe_ram_gb)" "${TWIN_REQUIRED_RAM_GB}"
}

# run_torch_check: report the verdict on the PyTorch build ("ok" or "reinstall: reason") on
# standard output; the GPU and PyTorch lines go to standard error.
run_torch_check() {
    local gpu torch_version cuda archs capability
    gpu="$(probe_gpu)" || die "nvidia-smi does not see a GPU"
    log "GPU: ${gpu}" >&2
    capability="${gpu##*, }"
    read -r torch_version cuda archs <<<"$(probe_torch)"
    log "PyTorch: ${torch_version}, CUDA ${cuda}, architectures: ${archs:-none}" >&2
    torch_stack_verdict "${torch_version}" "${cuda}" "${archs:-}" "${capability}"
}

# --------------------------------------------------------------------- environment

# use_data_disk: caches and temporary files go to the data disk (the system disk is small).
use_data_disk() {
    export HF_HOME="${TWIN_DISK_ROOT}/cache/huggingface"
    export MODELSCOPE_CACHE="${TWIN_DISK_ROOT}/cache/modelscope"
    export PIP_CACHE_DIR="${TWIN_DISK_ROOT}/cache/pip"
    export TMPDIR="${TWIN_DISK_ROOT}/tmp"
    export USE_MODELSCOPE_HUB=1
    mkdir -p "${HF_HOME}" "${MODELSCOPE_CACHE}" "${PIP_CACHE_DIR}" "${TMPDIR}"
}

# with_turbo COMMAND...: run COMMAND with AutoDL's academic acceleration (GitHub, Hugging Face).
# The proxy variables live in a subshell only, so they never stay set afterwards.
with_turbo() {
    (
        if [ -f /etc/network_turbo ]; then
            set +u
            # shellcheck source=/dev/null
            . /etc/network_turbo >/dev/null 2>&1 || true
            set -u
        fi
        "$@"
    )
}

turbo_off() {
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
}

# free_space_gate LABEL MIN_GB: stop when the data disk has less than MIN_GB free.
free_space_gate() {
    local label="$1" minimum="$2" free
    free="$(probe_free_gb "${TWIN_DISK_ROOT}")"
    if [ "${free}" -lt "${minimum}" ]; then
        die "not enough free space before '${label}': ${free} GB free, ${minimum} GB needed. 请到 AutoDL 控制台扩容数据盘/换实例"
    fi
    log "free space before ${label}: ${free} GB (need ${minimum} GB) - ok"
}

# secure_delete PATH...: overwrite files with shred (or random data) and remove them.
secure_delete() {
    local target file
    for target in "$@"; do
        [ -e "${target}" ] || continue
        if [ -d "${target}" ]; then
            while IFS= read -r -d '' file; do
                secure_delete_file "${file}"
            done < <(find "${target}" -type f -print0)
            rm -rf "${target}"
        else
            secure_delete_file "${target}"
        fi
    done
}

secure_delete_file() {
    if command -v shred >/dev/null 2>&1; then
        shred -f -n 2 -z -u "$1" 2>/dev/null || rm -f "$1"
    else
        local size
        size="$(stat -c %s "$1")"
        head -c "${size}" /dev/urandom >"$1" 2>/dev/null || true
        rm -f "$1"
    fi
}

# start_gpu_sampler FILE: record the GPU memory in use every 30 seconds (the peak goes into the metrics).
start_gpu_sampler() {
    (
        trap 'kill "${sleeper:-0}" 2>/dev/null; exit 0' TERM
        while true; do
            nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >>"$1" 2>/dev/null || true
            sleep 30 &
            sleeper=$!
            wait "${sleeper}" || true
        done
    ) &
    GPU_SAMPLER_PID=$!
}

stop_gpu_sampler() {
    if [ -n "${GPU_SAMPLER_PID:-}" ]; then
        kill "${GPU_SAMPLER_PID}" 2>/dev/null || true
        wait "${GPU_SAMPLER_PID}" 2>/dev/null || true
        GPU_SAMPLER_PID=""
    fi
}

# zstd_decompress: standard input to standard output, with the zstd program or, when the
# instance has none, the zstandard module (setup.sh installs one of the two).
zstd_decompress() {
    if command -v zstd >/dev/null 2>&1; then
        zstd -d -c
    else
        "${TWIN_PYTHON}" -c 'import sys, zstandard; zstandard.ZstdDecompressor().copy_stream(sys.stdin.buffer, sys.stdout.buffer)'
    fi
}

# --------------------------------------------------------------------- export plan

# The order in which export.sh works; the deletions come after the step that makes the next
# file, so there is always one good copy of the model on the disk.
EXPORT_STEPS=(
    merge_adapter
    prepare_llama_cpp
    convert_f16
    delete_merged_hf
    quantize
    delete_f16
    collect_artifacts
)

export_step_text() {
    case "$1" in
        merge_adapter) echo "merge the LoRA adapter into the bf16 base model (llamafactory-cli export)" ;;
        prepare_llama_cpp) echo "clone llama.cpp ${TWIN_LLAMA_CPP_TAG:-?} and build llama-quantize on the CPU" ;;
        convert_f16) echo "convert the merged model to an F16 GGUF (convert_hf_to_gguf.py)" ;;
        delete_merged_hf) echo "delete the merged HF weights" ;;
        quantize) echo "quantize the F16 GGUF to ${TWIN_GGUF_QUANTS:-Q4_K_M,Q5_K_M,Q8_0}" ;;
        delete_f16) echo "delete the F16 GGUF" ;;
        collect_artifacts) echo "copy the adapter, compute sha256 and write artifacts/manifest.json" ;;
        *) return 1 ;;
    esac
}

# print_export_plan: one line per step, in execution order.
print_export_plan() {
    local index=1 step
    for step in "${EXPORT_STEPS[@]}"; do
        printf 'step %d: %s - %s\n' "${index}" "${step}" "$(export_step_text "${step}")"
        index=$((index + 1))
    done
}

if [ -n "${TWIN_PROBE_OVERRIDES:-}" ]; then
    # shellcheck source=/dev/null
    . "${TWIN_PROBE_OVERRIDES}"
fi
