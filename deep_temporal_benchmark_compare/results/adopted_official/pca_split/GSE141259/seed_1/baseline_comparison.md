# Baseline comparison (mean over held-out transitions)

Lower is better for all metrics.

**Score space: `pca`** (PCA on training cells; avoids gene-space AE reconstruction floor).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `wot_official` calls Broad WOT (`wot.ot.OTModel.compute_transport_map`). `scvelo_official` calls `scvelo.pp.neighbors` (no unspliced layers, so `scv.tl.velocity` is not used). `prescient_potential_flow` / `mioflow_neural_ode` remain family reimplementations. `wot_barycentric` is the in-harness WOT-inspired map without growth rates.

         method  energy_distance  energy_centered  gauss_w2_mean  gauss_w2_cov  mean_shift_l2  var_ratio  energy_minus_ae_floor  ot_sinkhorn      mmd  mean_marginal_w1
scvelo_official         1.848375         0.282078      46.374661    120.886877       6.328127   1.622065              -4.295453   234.866020 0.014432          0.819539
   wot_official         2.255063         0.739443      36.703833     48.562565       5.941770   0.735136              -3.888765   130.533337 0.038370          0.740591
      our_model         5.737164         2.839159      68.302810    277.469157       8.262694   1.256836              -0.406664   194.641632 0.207025          1.452216

## Winner per metric

- `energy_distance`: **scvelo_official**
- `energy_centered`: **scvelo_official**
- `gauss_w2_mean`: **wot_official**
- `gauss_w2_cov`: **wot_official**
- `mean_shift_l2`: **wot_official**
- `var_ratio`: **wot_official**
- `energy_minus_ae_floor`: **scvelo_official**
- `ot_sinkhorn`: **wot_official** _(training objective — reference only)_
- `mmd`: **scvelo_official**
- `mean_marginal_w1`: **wot_official**

Best method by the primary metric (`energy_distance`): **scvelo_official**.