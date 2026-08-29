# Baseline comparison (mean over held-out transitions)

Lower is better for distance metrics; higher is better for `mdc`, `vpa`, lifts, `tif`, `ttc_module_spearman`, `module_mdc`, `hec`.

**Score space: `pca`** (PCA on training cells; avoids gene-space AE reconstruction floor).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `wot_official` calls Broad WOT (`wot.ot.OTModel.compute_transport_map`). `scvelo_official` calls `scvelo.pp.neighbors` (no unspliced layers, so `scv.tl.velocity` is not used). `prescient_potential_flow` / `mioflow_neural_ode` remain family reimplementations. `wot_barycentric` is the in-harness WOT-inspired map without growth rates.

         method  energy_distance  energy_centered  gauss_w2_mean  gauss_w2_cov  mean_shift_l2  var_ratio  energy_minus_ae_floor  ot_sinkhorn      mmd  mean_marginal_w1       mdc      vpa  lift_energy   lift_w1   lift_l2  lift_ae_energy  lift_ae_w1  lift_ae_l2      tif  ttc_module_spearman  module_mdc      hec
scvelo_official         2.829645         1.070888      59.438640    118.471129       7.709650   0.965423             -10.696313   174.440308 0.031274          0.838184  0.518380 0.405158     0.282516  0.107888  0.626007       12.316097    1.423984    4.809401 0.899868                  1.0    0.877764 0.999864
   wot_official         4.997638         2.196152      92.886266    110.104730       9.637757   0.639727              -8.528320   165.191040 0.065823          1.012770 -0.015342 0.605983    -1.885477 -0.066698 -1.302100       10.148104    1.249398    2.881293 0.970201                 -0.5   -0.856767 0.999865
      our_model        15.151538         8.713222     156.668390    369.738852      12.516726   0.135853               1.625581   265.983093 0.522649          2.263977  0.294214 0.656535   -12.039377 -1.317906 -4.181068       -0.005796   -0.001810    0.002325 0.704397                 -0.5   -0.705291 0.999864

## Winner per metric

- `energy_distance`: **scvelo_official**
- `energy_centered`: **scvelo_official**
- `gauss_w2_mean`: **scvelo_official**
- `gauss_w2_cov`: **wot_official**
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