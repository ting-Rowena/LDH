# Baseline comparison (mean over held-out transitions)

Lower is better for all metrics.

**Score space: `pca`** (PCA on training cells; avoids gene-space AE reconstruction floor).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `wot_official` calls Broad WOT (`wot.ot.OTModel.compute_transport_map`). `scvelo_official` calls `scvelo.pp.neighbors` (no unspliced layers, so `scv.tl.velocity` is not used). `prescient_potential_flow` / `mioflow_neural_ode` remain family reimplementations. `wot_barycentric` is the in-harness WOT-inspired map without growth rates.

         method  energy_distance  energy_centered  gauss_w2_mean  gauss_w2_cov  mean_shift_l2  var_ratio  energy_minus_ae_floor  ot_sinkhorn      mmd  mean_marginal_w1
scvelo_official         2.569862         1.000977      52.689987    106.937755       7.258788   1.006553             -11.258944   165.923645 0.029450          0.795376
   wot_official         4.998403         2.111522      96.178233    113.943880       9.807052   0.622060              -8.830404   164.725067 0.063058          0.990279
      our_model        15.180486         8.520804     160.695071    364.104879      12.676558   0.140962               1.351680   264.846924 0.519963          2.264438

## Winner per metric

- `energy_distance`: **scvelo_official**
- `energy_centered`: **scvelo_official**
- `gauss_w2_mean`: **scvelo_official**
- `gauss_w2_cov`: **scvelo_official**
- `mean_shift_l2`: **scvelo_official**
- `var_ratio`: **our_model**
- `energy_minus_ae_floor`: **scvelo_official**
- `ot_sinkhorn`: **wot_official** _(training objective — reference only)_
- `mmd`: **scvelo_official**
- `mean_marginal_w1`: **scvelo_official**

Best method by the primary metric (`energy_distance`): **scvelo_official**.