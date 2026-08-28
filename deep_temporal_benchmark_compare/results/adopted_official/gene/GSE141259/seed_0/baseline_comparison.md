# Baseline comparison (mean over held-out transitions)

Lower is better for all metrics.

**Score space: `gene`** (raw gene expression).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `wot_official` calls Broad WOT (`wot.ot.OTModel.compute_transport_map`). `scvelo_official` calls `scvelo.pp.neighbors` (no unspliced layers, so `scv.tl.velocity` is not used). `prescient_potential_flow` / `mioflow_neural_ode` remain family reimplementations. `wot_barycentric` is the in-harness WOT-inspired map without growth rates.

         method  energy_distance  ot_sinkhorn      mmd  mean_marginal_w1  mean_shift_l2
   wot_official         2.178061   702.753601 0.021441          0.097396       6.563406
scvelo_official         2.669905  1391.636597 0.004881          0.131195       9.106803
      our_model         7.832096   551.597961 0.233600          0.141391       9.428374

## Winner per metric

- `ot_sinkhorn`: **our_model** _(training objective — reference only)_
- `mmd`: **scvelo_official**
- `energy_distance`: **wot_official**
- `mean_marginal_w1`: **wot_official**
- `mean_shift_l2`: **wot_official**

Best method by the primary metric (`energy_distance`): **wot_official**.