# Baseline comparison (mean over held-out transitions)

Lower is better for distance metrics; higher is better for `mdc`, `vpa`, lifts, `tif`, `ttc_module_spearman`, `module_mdc`, `hec`.

**Score space: `pca`** (PCA on training cells; avoids gene-space AE reconstruction floor).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `wot_official` calls Broad WOT (`wot.ot.OTModel.compute_transport_map`). `scvelo_official` calls `scvelo.pp.neighbors` (no unspliced layers, so `scv.tl.velocity` is not used). `prescient_potential_flow` / `mioflow_neural_ode` remain family reimplementations. `wot_barycentric` is the in-harness WOT-inspired map without growth rates.

         method  energy_distance  energy_centered  gauss_w2_mean  gauss_w2_cov  mean_shift_l2  var_ratio  energy_minus_ae_floor  ot_sinkhorn      mmd  mean_marginal_w1      mdc      vpa  lift_energy   lift_w1    lift_l2  lift_ae_energy  lift_ae_w1  lift_ae_l2      tif  ttc_module_spearman  module_mdc      hec
      our_model         3.102468         1.936487      31.297880    199.820819       5.575802   1.168984               0.237474   138.066589 0.147310          1.163202 0.686994 0.638497    -0.723119 -0.554274   1.362552        0.044386    0.002559    0.054847 0.943354                 1.00    0.964498 0.999987
   wot_official         5.892353         0.315868     145.526841     55.482567      11.767872   0.763925               3.027358   165.433891 0.050179          0.964069 0.328469 0.392923    -3.513004 -0.355140  -4.829518       -2.745499    0.201693   -6.137223 0.975597                 0.25    0.337523 0.999987
scvelo_official        42.553312        17.034878    3132.607048   9889.946890      45.832966  29.793837              39.688318  7011.879013 0.053548          7.644325 0.121169 0.156938   -40.173963 -7.035397 -38.894612      -39.406459   -6.478564  -40.202317 0.882078                -0.25   -0.028093 0.999969

## Winner per metric

- `energy_distance`: **our_model**
- `energy_centered`: **wot_official**
- `gauss_w2_mean`: **our_model**
- `gauss_w2_cov`: **wot_official**
- `mean_shift_l2`: **our_model**
- `var_ratio`: **wot_official**
- `energy_minus_ae_floor`: **our_model**
- `ot_sinkhorn`: **our_model** _(training objective — reference only)_
- `mmd`: **wot_official**
- `mean_marginal_w1`: **wot_official**
- `mdc`: **our_model** _(higher better)_
- `vpa`: **our_model** _(higher better)_
- `lift_energy`: **our_model** _(higher better)_
- `lift_w1`: **wot_official** _(higher better)_
- `lift_l2`: **our_model** _(higher better)_
- `lift_ae_energy`: **our_model** _(higher better)_
- `lift_ae_w1`: **wot_official** _(higher better)_
- `lift_ae_l2`: **our_model** _(higher better)_
- `tif`: **wot_official** _(higher better)_
- `ttc_module_spearman`: **our_model** _(higher better)_
- `module_mdc`: **our_model** _(higher better)_
- `hec`: **our_model** _(higher better)_ _(LDH landscape; not a fair baseline score)_

Best method by the primary metric (`energy_distance`): **our_model**.