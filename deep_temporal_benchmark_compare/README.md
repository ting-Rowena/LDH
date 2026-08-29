# Deep temporal method comparison

All code and default outputs for comparing **LDH-scRNA** vs **PRESCIENT-family** /
**MIOFlow-family** / **WOT-inspired** live in this directory only.

Baselines are controlled core-objective reimplementations, not official package runs.

## Scripts

| File | Role |
|------|------|
| `baseline_evaluation.py` | Population-transfer harness (Energy / W1 / MMD / OT) |
| `run_deep_temporal_benchmark.py` | Multi-dataset / multi-seed OT launcher |
| `transport_priority_config.py` | Transport-priority LDH loss recipe |
| `train_transport_priority.py` | Retrain LDH for Energy/W1/MMD/OT |
| `run_deep_temporal_pcc_benchmark.py` | Trajectory–time Markov PCC (Fig. 2d axis) |
| `run_deep_temporal_fig2_axes.py` | Markov PCC + matched temporal null / U–KDE |
| `diagnose_short_horizon.py` | Adjacent-step OT diagnostic for LDH |

## One-shot: train transport LDH + run OT benchmark

```bash
# transportOT_v2: L ≈ Lot + 0.05 Lrecon + 1.5 Llatent + 3.0 Llat_disp
bash deep_temporal_benchmark_compare/run_transport_train_and_benchmark.sh
# logs: deep_temporal_benchmark_compare/logs/transportOT_v2_pipeline.log
# ckpts: deep_temporal_benchmark_compare/checkpoints/transportOT_v2/<DATASET>/
# results: deep_temporal_benchmark_compare/results/transportOT_v2/pca/
```

This waits for GPU memory, trains GSE155622 → GSE141259 → HGSOC if needed,
then runs seeds 0–2 on `score-space pca`.

## Population transfer (Energy / W1 / MMD / OT)

### Recommended LDH settings for these metrics

Publication checkpoints select on `pcc_then_mse` and keep strong landscape
terms. That hurts pure population-transfer scores. For a **transport variant**:

| Knob | Paper-ish default | Transport priority (`transportOT_v1`) |
|------|-------------------|----------------------------------------|
| `checkpoint_metric` / `early_stop_metric` | `pcc_then_mse` | **`loss`** |
| `lambda_latent` | 0.5 (pain/HGSOC) / 0.1 (lung) | **1.0** |
| `lambda_lat_disp` + OT coupling | 1.0 + on (pain/HGSOC); often 0 (lung) | **1.0 + OT coupling** (batch OT; avoid full-pop OT on large cohorts — OOM risk) |
| `lambda_kinetic` | 0.2 | **0.05** |
| `lambda_energy` / HJB | ~0.05 | **0.01** |
| `lambda_density` | 0.01 | **0.0** (off) |
| `lambda_recon` | 0.01–0.1 | **0.01** |

Treat this as a separate “LDH-transport” model: better OT metrics may come at
the cost of landscape / null / PCC narrative used in the main text.

### Train transport-priority LDH

```bash
# from the repository root
python deep_temporal_benchmark_compare/train_transport_priority.py \
  --datasets GSE155622 GSE141259 HGSOC
```

Checkpoints: `checkpoints/transportOT_v1/<DATASET>/best_model.pth`

### Compare on Energy / W1 / MMD / OT

```bash
# Paper landscape checkpoints
python deep_temporal_benchmark_compare/run_deep_temporal_benchmark.py \
  --checkpoint-set adopted --score-space pca --device cuda
# → results/pca/

# Hamiltonian4 (Figure 2 / Supplementary_table2.csv four-method transport)
python deep_temporal_benchmark_compare/run_deep_temporal_benchmark.py \
  --checkpoint-set hamiltonian4 --score-space pca --device cuda
# weights: deep_temporal_benchmark_compare/Hamiltonian4_<DATASET>/

# Transport-priority LDH
python deep_temporal_benchmark_compare/run_deep_temporal_benchmark.py \
  --checkpoint-set transportOT_v1 --score-space pca --device cuda
# → results/transportOT_v1/pca/
```

Primary tables usually use **pca**. Lower is better for every OT metric.

## Fig. 2–aligned axes (Markov PCC / null)

```bash
python deep_temporal_benchmark_compare/run_deep_temporal_pcc_benchmark.py \
  --datasets GSE155622 GSE141259 HGSOC --device cuda
# → results/pcc/

python deep_temporal_benchmark_compare/run_deep_temporal_fig2_axes.py \
  --datasets GSE155622 GSE141259 HGSOC --device cuda
# → results/fig2_axes/
```

## Short-horizon diagnostic

```bash
python deep_temporal_benchmark_compare/diagnose_short_horizon.py \
  --dataset GSE141259 \
  --checkpoint-dir /path/to/ckpt \
  --save-dir deep_temporal_benchmark_compare/results/diagnostics/shorthorizon
```

## Layout

```text
deep_temporal_benchmark_compare/
  Hamiltonian4_<DATASET>/                # four-method transport LDH weights
  checkpoints/transportOT_v1/<DATASET>/
  results/<score_space>/                 # OT: adopted LDH
  results/hamiltonian4_official/         # OT: Hamiltonian4 vs WOT / PRESCIENT / MIOFlow
  results/transportOT_v1/<score_space>/  # OT: transport LDH
  results/pcc/
  results/fig2_axes/
  results/diagnostics/
  _cache/
```
