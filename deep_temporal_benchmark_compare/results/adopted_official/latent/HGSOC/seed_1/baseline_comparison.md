# Baseline comparison (mean over held-out transitions)

Lower is better for all metrics.

**Score space: `latent`** (LDH latent z; LDH uses native integrated latent, other methods are gene-predicted then encoded by the LDH encoder).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `wot_official` calls Broad WOT (`wot.ot.OTModel.compute_transport_map`). `scvelo_official` calls `scvelo.pp.neighbors` (no unspliced layers, so `scv.tl.velocity` is not used). `prescient_potential_flow` / `mioflow_neural_ode` remain family reimplementations. `wot_barycentric` is the in-harness WOT-inspired map without growth rates.

         method  energy_distance  ot_sinkhorn      mmd  mean_marginal_w1  mean_shift_l2
scvelo_official         2.614595    33.548389 0.080546          0.246767       6.436305
   wot_official         2.773864    33.245804 0.109671          0.255109       6.417201
      our_model         4.066677    37.559273 0.248981          0.304485       7.011703

## Winner per metric

- `ot_sinkhorn`: **wot_official** _(training objective — reference only)_
- `mmd`: **scvelo_official**
- `energy_distance`: **scvelo_official**
- `mean_marginal_w1`: **scvelo_official**
- `mean_shift_l2`: **wot_official**

Best method by the primary metric (`energy_distance`): **scvelo_official**.