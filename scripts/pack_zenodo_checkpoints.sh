#!/usr/bin/env bash
# Build lean Zenodo zips (weights only; no TCGA cache, no analysis_protocol dumps).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$ROOT/zenodo_staging}"
STAGE="$OUT/stage"
rm -rf "$STAGE"
mkdir -p "$STAGE/adopted" "$STAGE/h4"

copy_ckpt() {
  local src="$1" dest="$2"
  mkdir -p "$dest"
  local f
  for f in best_model.pth obs.csv Loss_epoch.csv training_summary.json \
           training_var_names.json training_umap.npz latent_embeddings.npz
  do
    if [[ -f "$src/$f" ]]; then
      cp -a "$src/$f" "$dest/"
    fi
  done
  if [[ -d "$src/methods_enhancement" ]]; then
    mkdir -p "$dest/methods_enhancement"
    find "$src/methods_enhancement" -maxdepth 1 -name '*summary.json' \
      -exec cp -a {} "$dest/methods_enhancement/" \;
  fi
}

ADOPTED=(
  "GSE155622_checkpoints_3000_3000_384_0.05_recon0.01_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
  "GSE141259_checkpoints_5000_5000_512_0.01_recon0.1_valD28_timeX_lossnorm_qp_d0p01_z0p1_k0p2"
  "HGSOC_checkpoints_3000_3000_512_0.05_recon0.01_nactpair_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
)
for D in "${ADOPTED[@]}"; do
  copy_ckpt "$ROOT/$D" "$STAGE/adopted/$D"
done
(cd "$STAGE/adopted" && zip -r -q "$OUT/ldh-scrna-adopted-landscape-checkpoints.zip" .)

for D in Hamiltonian4_GSE155622 Hamiltonian4_GSE141259 Hamiltonian4_HGSOC; do
  copy_ckpt \
    "$ROOT/deep_temporal_benchmark_compare/$D" \
    "$STAGE/h4/deep_temporal_benchmark_compare/$D"
done
(cd "$STAGE/h4" && zip -r -q "$OUT/ldh-scrna-hamiltonian4-checkpoints.zip" .)

echo "Wrote:"
ls -lh "$OUT"/ldh-scrna-adopted-landscape-checkpoints.zip \
      "$OUT"/ldh-scrna-hamiltonian4-checkpoints.zip
