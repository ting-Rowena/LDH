#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H4_DIR="${ROOT}/deep_temporal_benchmark_compare"
LOG_DIR="${ROOT}/hamiltonian4_training_logs"
MIN_FREE_MIB="${MIN_FREE_MIB:-22000}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-10}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

wait_for_gpu() {
    local gpu=""
    while [[ -z "${gpu}" ]]; do
        gpu="$(
            nvidia-smi \
                --query-gpu=index,memory.free,utilization.gpu \
                --format=csv,noheader,nounits |
            awk -F',' -v min_free="${MIN_FREE_MIB}" -v max_util="${MAX_UTIL_PCT}" '
                {
                    gsub(/ /, "", $1);
                    gsub(/ /, "", $2);
                    gsub(/ /, "", $3);
                    if (($2 + 0) >= min_free && ($3 + 0) <= max_util) {
                        print $1;
                        exit;
                    }
                }
            '
        )"
        if [[ -z "${gpu}" ]]; then
            echo "WAITING_FOR_GPU free>=${MIN_FREE_MIB}MiB util<=${MAX_UTIL_PCT}%" >&2
            sleep "${POLL_SECONDS}"
        fi
    done
    printf '%s' "${gpu}"
}

run_dataset() {
    local dataset="$1"
    local output_dir="$2"
    shift 2
    local device_label
    if [[ "${FORCE_CPU:-0}" == "1" ]]; then
        device_label="cpu"
        echo "TRAINING_STARTED dataset=${dataset} device=cpu output=${output_dir}"
        CUDA_VISIBLE_DEVICES="" \
        OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" \
        MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}" \
            python run_training.py \
            --dataset "${dataset}" \
            --loss-recipe hamiltonian \
            --checkpoint-dir "${output_dir}" \
            "$@" 2>&1 | tee "${LOG_DIR}/${dataset}.log"
    else
        local gpu
        gpu="$(wait_for_gpu)"
        device_label="gpu${gpu}"
        echo "TRAINING_STARTED dataset=${dataset} gpu=${gpu} output=${output_dir}"
        CUDA_VISIBLE_DEVICES="${gpu}" python run_training.py \
            --dataset "${dataset}" \
            --loss-recipe hamiltonian \
            --checkpoint-dir "${output_dir}" \
            "$@" 2>&1 | tee "${LOG_DIR}/${dataset}.log"
    fi
    echo "TRAINING_FINISHED dataset=${dataset} device=${device_label} output=${output_dir}"
}

run_dataset \
    "GSE155622" \
    "${H4_DIR}/Hamiltonian4_GSE155622"

run_dataset \
    "GSE141259" \
    "${H4_DIR}/Hamiltonian4_GSE141259" \
    --profile l3_genes5000

run_dataset \
    "HGSOC" \
    "${H4_DIR}/Hamiltonian4_HGSOC"

echo "ALL_TRAINING_FINISHED"
