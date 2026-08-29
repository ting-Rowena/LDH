# Baseline comparison (mean over held-out transitions)

Lower is better for all metrics.

**Score space: `pca`** (PCA on training cells; avoids gene-space AE reconstruction floor).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `wot_official` calls Broad WOT (`wot.ot.OTModel.compute_transport_map`). `scvelo_official` calls `scvelo.pp.neighbors` (no unspliced layers, so `scv.tl.velocity` is not used). `prescient_potential_flow` / `mioflow_neural_ode` remain family reimplementations. `wot_barycentric` is the in-harness WOT-inspired map without growth rates.

         method  energy_distance  ot_sinkhorn      mmd  mean_marginal_w1  mean_shift_l2
      our_model         3.118662   121.707745 0.150772          1.437367       5.477594
   wot_official         5.955222   133.236725 0.061968          1.407119      11.405445
scvelo_official        36.553165  4931.213638 0.057347          8.615701      40.510229

## Winner per metric

- `ot_sinkhorn`: **our_model** _(training objective — reference only)_
- `mmd`: **scvelo_official**
- `energy_distance`: **our_model**
- `mean_marginal_w1`: **wot_official**
- `mean_shift_l2`: **our_model**

Best method by the primary metric (`energy_distance`): **our_model**.