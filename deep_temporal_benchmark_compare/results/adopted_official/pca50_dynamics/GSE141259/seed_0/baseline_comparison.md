# Baseline comparison (mean over held-out transitions)

Lower is better for distance metrics; higher is better for `mdc`, `vpa`, lifts, `tif`, `ttc_module_spearman`, `module_mdc`, `hec`.

**Score space: `pca`** (PCA on training cells; avoids gene-space AE reconstruction floor).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `wot_official` calls Broad WOT (`wot.ot.OTModel.compute_transport_map`). `scvelo_official` calls `scvelo.pp.neighbors` (no unspliced layers, so `scv.tl.velocity` is not used). `prescient_potential_flow` / `mioflow_neural_ode` remain family reimplementations. `wot_barycentric` is the in-harness WOT-inspired map without growth rates.

         method  energy_distance  energy_centered  gauss_w2_mean  gauss_w2_cov  mean_shift_l2  var_ratio  energy_minus_ae_floor  ot_sinkhorn      mmd  mean_marginal_w1      mdc      vpa  lift_energy   lift_w1   lift_l2  lift_ae_energy  lift_ae_w1  lift_ae_l2      tif  ttc_module_spearman  module_mdc      hec
scvelo_official         2.177356         0.247497      55.445700     88.302833       7.010933   1.345382              -3.565146   203.618805 0.016644          0.777607 0.307544 0.346414     0.615749 -0.177700  0.754729        4.633503    0.714264    2.722816 0.913673                  0.0    0.164264 0.999996
   wot_official         2.185563         0.777346      34.343205     51.759493       5.830736   0.739218              -3.556939   130.006626 0.038880          0.761041 0.564000 0.488294     0.607542 -0.161134  1.934927        4.625296    0.730830    3.903013 0.817338                  1.0    0.996143 0.999997
      our_model         6.215360         2.897341      78.467889    263.242048       8.854658   1.200075               0.472857   194.702332 0.217684          1.459583 0.174447 0.513759    -3.422254 -0.859676 -1.088995        0.595499    0.032288    0.879091 0.886594                  1.0    0.659287 0.999997

## Winner per metric

- `energy_distance`: **scvelo_official**
- `energy_centered`: **scvelo_official**
- `gauss_w2_mean`: **wot_official**
- `gauss_w2_cov`: **wot_official**
- `mean_shift_l2`: **wot_official**
- `var_ratio`: **wot_official**
- `energy_minus_ae_floor`: **scvelo_official**
- `ot_sinkhorn`: **wot_official** _(training objective — reference only)_
- `mmd`: **scvelo_official**
- `mean_marginal_w1`: **wot_official**
- `mdc`: **wot_official** _(higher better)_
- `vpa`: **our_model** _(higher better)_
- `lift_energy`: **scvelo_official** _(higher better)_
- `lift_w1`: **wot_official** _(higher better)_
- `lift_l2`: **wot_official** _(higher better)_
- `lift_ae_energy`: **scvelo_official** _(higher better)_
- `lift_ae_w1`: **wot_official** _(higher better)_
- `lift_ae_l2`: **wot_official** _(higher better)_
- `tif`: **scvelo_official** _(higher better)_
- `ttc_module_spearman`: **wot_official** _(higher better)_
- `module_mdc`: **wot_official** _(higher better)_
- `hec`: **wot_official** _(higher better)_ _(LDH landscape; not a fair baseline score)_

Best method by the primary metric (`energy_distance`): **scvelo_official**.