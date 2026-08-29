# Baseline comparison (mean over held-out transitions)

Lower is better for all metrics.

**Score space: `gene`** (raw gene expression).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `wot_official` calls Broad WOT (`wot.ot.OTModel.compute_transport_map`). `scvelo_official` calls `scvelo.pp.neighbors` (no unspliced layers, so `scv.tl.velocity` is not used). `prescient_potential_flow` / `mioflow_neural_ode` remain family reimplementations. `wot_barycentric` is the in-harness WOT-inspired map without growth rates.

         method  energy_distance  ot_sinkhorn      mmd  mean_marginal_w1  mean_shift_l2
scvelo_official         2.469169   882.245728 0.005102          0.134350      10.361727
   wot_official         6.454693   665.283691 0.050968          0.167773      11.924709
      our_model        27.888357   917.042969 0.511039          0.570381      24.953886

## Winner per metric

- `ot_sinkhorn`: **wot_official** _(training objective — reference only)_
- `mmd`: **scvelo_official**
- `energy_distance`: **scvelo_official**
- `mean_marginal_w1`: **scvelo_official**
- `mean_shift_l2`: **scvelo_official**

Best method by the primary metric (`energy_distance`): **scvelo_official**.