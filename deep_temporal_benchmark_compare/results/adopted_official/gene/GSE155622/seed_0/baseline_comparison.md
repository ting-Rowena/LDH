# Baseline comparison (mean over held-out transitions)

Lower is better for all metrics.

**Score space: `gene`** (raw gene expression).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `wot_official` calls Broad WOT (`wot.ot.OTModel.compute_transport_map`). `scvelo_official` calls `scvelo.pp.neighbors` (no unspliced layers, so `scv.tl.velocity` is not used). `prescient_potential_flow` / `mioflow_neural_ode` remain family reimplementations. `wot_barycentric` is the in-harness WOT-inspired map without growth rates.

         method  energy_distance   ot_sinkhorn      mmd  mean_marginal_w1  mean_shift_l2
   wot_official         3.056014   1231.541321 0.003784          0.207424      12.535610
      our_model         6.584849    697.299744 0.184619          0.285325       6.586712
scvelo_official        77.823747 109402.861328 0.003826          2.154830      66.362001

## Winner per metric

- `ot_sinkhorn`: **our_model** _(training objective — reference only)_
- `mmd`: **wot_official**
- `energy_distance`: **wot_official**
- `mean_marginal_w1`: **wot_official**
- `mean_shift_l2`: **our_model**

Best method by the primary metric (`energy_distance`): **wot_official**.