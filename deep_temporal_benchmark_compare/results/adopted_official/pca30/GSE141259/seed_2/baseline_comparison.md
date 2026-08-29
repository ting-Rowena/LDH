# Baseline comparison (mean over held-out transitions)

Lower is better for all metrics.

**Score space: `pca`** (PCA on training cells; avoids gene-space AE reconstruction floor).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `wot_official` calls Broad WOT (`wot.ot.OTModel.compute_transport_map`). `scvelo_official` calls `scvelo.pp.neighbors` (no unspliced layers, so `scv.tl.velocity` is not used). `prescient_potential_flow` / `mioflow_neural_ode` remain family reimplementations. `wot_barycentric` is the in-harness WOT-inspired map without growth rates.

         method  energy_distance  ot_sinkhorn      mmd  mean_marginal_w1  mean_shift_l2
scvelo_official         2.025880   138.457413 0.020294          0.919154       6.357526
   wot_official         2.150308    86.838554 0.039089          0.935776       5.804688
      our_model         6.667477   165.481049 0.238965          1.773416       8.770546

## Winner per metric

- `ot_sinkhorn`: **wot_official** _(training objective — reference only)_
- `mmd`: **scvelo_official**
- `energy_distance`: **scvelo_official**
- `mean_marginal_w1`: **scvelo_official**
- `mean_shift_l2`: **wot_official**

Best method by the primary metric (`energy_distance`): **scvelo_official**.