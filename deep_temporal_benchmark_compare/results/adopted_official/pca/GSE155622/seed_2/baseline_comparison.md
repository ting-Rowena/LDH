# Baseline comparison (mean over held-out transitions)

Lower is better for all metrics.

**Score space: `pca`** (PCA on training cells; avoids gene-space AE reconstruction floor).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `wot_official` calls Broad WOT (`wot.ot.OTModel.compute_transport_map`). `scvelo_official` calls `scvelo.pp.neighbors` (no unspliced layers, so `scv.tl.velocity` is not used). `prescient_potential_flow` / `mioflow_neural_ode` remain family reimplementations. `wot_barycentric` is the in-harness WOT-inspired map without growth rates.

         method  energy_distance  ot_sinkhorn      mmd  mean_marginal_w1  mean_shift_l2
      our_model         3.080075   140.050285 0.148989          1.181129       5.407545
   wot_official         6.601785   187.912224 0.051156          1.027651      12.742296
scvelo_official        38.187372  7499.451752 0.049548          7.598010      43.139345

## Winner per metric

- `ot_sinkhorn`: **our_model** _(training objective — reference only)_
- `mmd`: **scvelo_official**
- `energy_distance`: **our_model**
- `mean_marginal_w1`: **wot_official**
- `mean_shift_l2`: **our_model**

Best method by the primary metric (`energy_distance`): **our_model**.