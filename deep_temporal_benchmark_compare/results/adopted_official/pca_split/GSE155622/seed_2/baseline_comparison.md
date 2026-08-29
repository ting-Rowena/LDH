# Baseline comparison (mean over held-out transitions)

Lower is better for all metrics.

**Score space: `pca`** (PCA on training cells; avoids gene-space AE reconstruction floor).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `wot_official` calls Broad WOT (`wot.ot.OTModel.compute_transport_map`). `scvelo_official` calls `scvelo.pp.neighbors` (no unspliced layers, so `scv.tl.velocity` is not used). `prescient_potential_flow` / `mioflow_neural_ode` remain family reimplementations. `wot_barycentric` is the in-harness WOT-inspired map without growth rates.

         method  energy_distance  energy_centered  gauss_w2_mean  gauss_w2_cov  mean_shift_l2  var_ratio  energy_minus_ae_floor  ot_sinkhorn      mmd  mean_marginal_w1
      our_model         3.080075         1.989026      29.241723    204.712802       5.407545   1.152212               0.358322   140.050285 0.148989          1.181129
   wot_official         6.601785         0.404959     171.698747     72.482322      12.742296   0.738150               3.880032   187.912224 0.051156          1.027651
scvelo_official        38.187372        17.855942    2830.637576  10998.341847      43.139345  32.126820              35.465619  7499.451752 0.049548          7.598010

## Winner per metric

- `energy_distance`: **our_model**
- `energy_centered`: **wot_official**
- `gauss_w2_mean`: **our_model**
- `gauss_w2_cov`: **wot_official**
- `mean_shift_l2`: **our_model**
- `var_ratio`: **wot_official**
- `energy_minus_ae_floor`: **our_model**
- `ot_sinkhorn`: **our_model** _(training objective — reference only)_
- `mmd`: **scvelo_official**
- `mean_marginal_w1`: **wot_official**

Best method by the primary metric (`energy_distance`): **our_model**.