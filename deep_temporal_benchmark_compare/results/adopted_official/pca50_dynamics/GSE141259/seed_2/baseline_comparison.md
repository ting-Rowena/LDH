# Baseline comparison (mean over held-out transitions)

Lower is better for distance metrics; higher is better for `mdc`, `vpa`, lifts, `tif`, `ttc_module_spearman`, `module_mdc`, `hec`.

**Score space: `pca`** (PCA on training cells; avoids gene-space AE reconstruction floor).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `wot_official` calls Broad WOT (`wot.ot.OTModel.compute_transport_map`). `scvelo_official` calls `scvelo.pp.neighbors` (no unspliced layers, so `scv.tl.velocity` is not used). `prescient_potential_flow` / `mioflow_neural_ode` remain family reimplementations. `wot_barycentric` is the in-harness WOT-inspired map without growth rates.

         method  energy_distance  energy_centered  gauss_w2_mean  gauss_w2_cov  mean_shift_l2  var_ratio  energy_minus_ae_floor  ot_sinkhorn      mmd  mean_marginal_w1      mdc      vpa  lift_energy   lift_w1   lift_l2  lift_ae_energy  lift_ae_w1  lift_ae_l2      tif  ttc_module_spearman  module_mdc      hec
scvelo_official         2.114057         0.256912      55.164329    115.678520       6.918492   1.487540              -3.931368   230.376793 0.016260          0.825491 0.289026 0.343250     0.636644 -0.239797  0.660923        5.069826    0.709773    2.897843 0.867991                  0.0    0.700306 0.999997
   wot_official         2.226160         0.766073      36.144604     52.724091       5.973993   0.725014              -3.819265   130.581009 0.037715          0.767045 0.524601 0.479043     0.524541 -0.181351  1.605422        4.957723    0.768218    3.842342 0.848351                  1.0    0.999872 0.999997
      our_model         6.611131         3.111550      80.885667    271.731626       8.972739   1.169649               0.565706   200.105461 0.235406          1.503248 0.153212 0.507311    -3.860430 -0.917554 -1.393324        0.572752    0.032015    0.843596 0.892210                  1.0    0.641645 0.999997

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
- `tif`: **our_model** _(higher better)_
- `ttc_module_spearman`: **wot_official** _(higher better)_
- `module_mdc`: **wot_official** _(higher better)_
- `hec`: **wot_official** _(higher better)_ _(LDH landscape; not a fair baseline score)_

Best method by the primary metric (`energy_distance`): **scvelo_official**.