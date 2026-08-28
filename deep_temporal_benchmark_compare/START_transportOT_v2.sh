#!/usr/bin/env bash
# After GPU memory is free (reboot or nvidia-smi --gpu-reset), start transportOT_v2
# train + Energy/W1/MMD/OT benchmark.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PYTHON:-python}"
LOGDIR=deep_temporal_benchmark_compare/logs
mkdir -p "$LOGDIR"

free_gb=$("$PY" -c 'import torch; f,_=torch.cuda.mem_get_info(); print(f/1e9)')
echo "[START_transportOT_v2] GPU${CUDA_VISIBLE_DEVICES} free=${free_gb}GB"
awk -v f="$free_gb" 'BEGIN{exit !(f+0 > 8)}' || {
  echo "ERROR: need >8GB free. Free GPUs first:"
  echo "  sudo nvidia-smi --gpu-reset -i 0,1"
  echo "  # or reboot"
  exit 1
}

nohup bash deep_temporal_benchmark_compare/run_transport_train_and_benchmark.sh \
  >> "$LOGDIR/transportOT_v2_pipeline.log" 2>&1 &
echo "started pid=$!"
echo "tail -f $LOGDIR/transportOT_v2_pipeline.log"
