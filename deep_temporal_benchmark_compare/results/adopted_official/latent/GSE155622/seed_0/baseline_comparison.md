# Baseline comparison (mean over held-out transitions)

Lower is better for all metrics.

**Score space: `latent`** (LDH latent z; LDH uses native integrated latent, other methods are gene-predicted then encoded by the LDH encoder).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `wot_official` calls Broad WOT (`wot.ot.OTModel.compute_transport_map`). `scvelo_official` calls `scvelo.pp.neighbors` (no unspliced layers, so `scv.tl.velocity` is not used). `prescient_potential_flow` / `mioflow_neural_ode` remain family reimplementations. `wot_barycentric` is the in-harness WOT-inspired map without growth rates.

         method  energy_distance  ot_sinkhorn      mmd  mean_marginal_w1  mean_shift_l2
   wot_official         0.484069     9.674324 0.015087          0.125359       2.755691
scvelo_official         0.592304    30.393600 0.020385          0.151457       3.129883
      our_model         0.850919    10.671967 0.063383          0.151621       3.010096

## Winner per metric

- `ot_sinkhorn`: **wot_official** _(training objective — reference only)_
- `mmd`: **wot_official**
- `energy_distance`: **wot_official**
- `mean_marginal_w1`: **wot_official**
- `mean_shift_l2`: **wot_official**

Best method by the primary metric (`energy_distance`): **wot_official**.