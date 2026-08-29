#!/usr/bin/env bash
# Train transportOT_v2 (L≈Lot+0.05Lrecon+1.5Llatent+3.0Llat_disp), then OT benchmark.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
LOGDIR=deep_temporal_benchmark_compare/logs
CKROOT=deep_temporal_benchmark_compare/checkpoints/transportOT_v2
mkdir -p "$LOGDIR" "$CKROOT"

export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log() { echo "[transportOT_v2 $(date -Is)] $*"; }

wait_gpu_free() {
  # Need ~8GB+ free for LDH training; ghost CUDA contexts can fake "busy".
  local tries=0
  while true; do
    if "$PY" - <<'PY'
import torch
free, total = torch.cuda.mem_get_info()
ok = free > 8e9
raise SystemExit(0 if ok else 1)
PY
    then
      log "GPU has >8GB free"
      return 0
    fi
    tries=$((tries + 1))
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES:-0}" 2>/dev/null | head -1 || echo '?')
    log "waiting for GPU >8GB free (try=$tries, nvidia_free_MiB=$free_mb)"
    sleep 120
  done
}

train_one() {
  local ds="$1"
  local ck="$CKROOT/$ds/best_model.pth"
  if [[ -f "$ck" ]]; then
    log "skip train $ds (found $ck)"
    return 0
  fi
  log "START train $ds"
  "$PY" deep_temporal_benchmark_compare/train_transport_priority.py --datasets "$ds" \
    > "$LOGDIR/train_${ds}_transportOT_v2.log" 2>&1
  [[ -f "$ck" ]] || { log "ERROR missing $ck"; exit 1; }
  log "DONE train $ds"
}

wait_gpu_free
train_one GSE155622
train_one GSE141259
train_one HGSOC

log "START benchmark transportOT_v2 (pca, seeds 0 1 2)"
"$PY" deep_temporal_benchmark_compare/run_deep_temporal_benchmark.py \
  --checkpoint-set transportOT_v2 \
  --datasets GSE155622 GSE141259 HGSOC \
  --seeds 0 1 2 \
  --score-space pca \
  --device cuda \
  > "$LOGDIR/benchmark_transportOT_v2_pca.log" 2>&1
log "DONE benchmark -> deep_temporal_benchmark_compare/results/transportOT_v2/pca/"
log "PIPELINE_DONE"
