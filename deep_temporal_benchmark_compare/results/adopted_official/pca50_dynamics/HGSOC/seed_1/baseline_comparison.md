# Baseline comparison (mean over held-out transitions)

Lower is better for distance metrics; higher is better for `mdc`, `vpa`, lifts, `tif`, `ttc_module_spearman`, `module_mdc`, `hec`.

**Score space: `pca`** (PCA on training cells; avoids gene-space AE reconstruction floor).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `wot_official` calls Broad WOT (`wot.ot.OTModel.compute_transport_map`). `scvelo_official` calls `scvelo.pp.neighbors` (no unspliced layers, so `scv.tl.velocity` is not used). `prescient_potential_flow` / `mioflow_neural_ode` remain family reimplementations. `wot_barycentric` is the in-harness WOT-inspired map without growth rates.

         method  energy_distance  energy_centered  gauss_w2_mean  gauss_w2_cov  mean_shift_l2  var_ratio  energy_minus_ae_floor  ot_sinkhorn      mmd  mean_marginal_w1       mdc      vpa  lift_energy   lift_w1   lift_l2  lift_ae_energy  lift_ae_w1  lift_ae_l2      tif  ttc_module_spearman  module_mdc      hec
scvelo_official         2.569862         1.000977      52.689987    106.937755       7.258788   1.006553             -11.258944   165.923645 0.029450          0.795376  0.553850 0.392695     0.332986  0.118877  0.672029       12.605624    1.467627    5.421820 0.964580                  1.0    0.394156 0.999865
   wot_official         4.998403         2.111522      96.178233    113.943880       9.807052   0.622060              -8.830404   164.725067 0.063058          0.990279 -0.087672 0.583333    -2.095555 -0.076027 -1.876235       10.177083    1.272724    2.873556 0.991985                 -0.5   -0.772903 0.999865
      our_model        15.180486         8.520804     160.695071    364.104879      12.676558   0.140962               1.351680   264.846924 0.519963          2.264438  0.260737 0.648619   -12.277638 -1.350186 -4.745740       -0.005000   -0.001435    0.004050 0.699957                 -0.5   -0.536736 0.999864

## Winner per metric

- `energy_distance`: **scvelo_official**
- `energy_centered`: **scvelo_official**
- `gauss_w2_mean`: **scvelo_official**
- `gauss_w2_cov`: **scvelo_official**
- `mean_shift_l2`: **scvelo_official**
- `var_ratio`: **our_model**
- `energy_minus_ae_floor`: **scvelo_official**
- `ot_sinkhorn`: **wot_official** _(training objective — reference only)_
- `mmd`: **scvelo_official**
- `mean_marginal_w1`: **scvelo_official**
- `mdc`: **scvelo_official** _(higher better)_
- `vpa`: **our_model** _(higher better)_
- `lift_energy`: **scvelo_official** _(higher better)_
- `lift_w1`: **scvelo_official** _(higher better)_
- `lift_l2`: **scvelo_official** _(higher better)_
- `lift_ae_energy`: **scvelo_official** _(higher better)_
- `lift_ae_w1`: **scvelo_official** _(higher better)_
- `lift_ae_l2`: **scvelo_official** _(higher better)_
- `tif`: **wot_official** _(higher better)_
- `ttc_module_spearman`: **scvelo_official** _(higher better)_
- `module_mdc`: **scvelo_official** _(higher better)_
- `hec`: **wot_official** _(higher better)_ _(LDH landscape; not a fair baseline score)_

Best method by the primary metric (`energy_distance`): **scvelo_official**.