# Baseline comparison (mean over held-out transitions)

Lower is better for all metrics.

**Score space: `pca`** (PCA on training cells; avoids gene-space AE reconstruction floor).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `wot_official` calls Broad WOT (`wot.ot.OTModel.compute_transport_map`). `scvelo_official` calls `scvelo.pp.neighbors` (no unspliced layers, so `scv.tl.velocity` is not used). `prescient_potential_flow` / `mioflow_neural_ode` remain family reimplementations. `wot_barycentric` is the in-harness WOT-inspired map without growth rates.

         method  energy_distance  energy_centered  gauss_w2_mean  gauss_w2_cov  mean_shift_l2  var_ratio  energy_minus_ae_floor  ot_sinkhorn      mmd  mean_marginal_w1
scvelo_official         2.486617         0.976463      50.211640    106.638893       7.086016   0.971722             -11.144286   162.103378 0.028995          0.765481
   wot_official         4.639942         2.162052      84.235382    109.730196       9.177981   0.640906              -8.990961   158.966156 0.064061          0.954672
      our_model        14.608503         8.387739     150.752980    365.335189      12.278150   0.144111               0.977599   258.778809 0.505230          2.214611

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