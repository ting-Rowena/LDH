# Baseline comparison (mean over held-out transitions)

Lower is better for all metrics.

**Score space: `latent`** (LDH latent z; LDH uses native integrated latent, other methods are gene-predicted then encoded by the LDH encoder).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `wot_official` calls Broad WOT (`wot.ot.OTModel.compute_transport_map`). `scvelo_official` calls `scvelo.pp.neighbors` (no unspliced layers, so `scv.tl.velocity` is not used). `prescient_potential_flow` / `mioflow_neural_ode` remain family reimplementations. `wot_barycentric` is the in-harness WOT-inspired map without growth rates.

         method  energy_distance  ot_sinkhorn      mmd  mean_marginal_w1  mean_shift_l2
scvelo_official         3.104344    62.841931 0.089807          0.267639       6.334350
   wot_official         3.238248    50.790054 0.095450          0.264089       6.456522
      our_model         3.583862    50.592701 0.160974          0.282209       6.416840

## Winner per metric

- `ot_sinkhorn`: **our_model** _(training objective — reference only)_
- `mmd`: **scvelo_official**
- `energy_distance`: **scvelo_official**
- `mean_marginal_w1`: **wot_official**
- `mean_shift_l2`: **scvelo_official**

Best method by the primary metric (`energy_distance`): **scvelo_official**.