# Baseline comparison (mean over held-out transitions)

Lower is better for distance metrics; higher is better for `mdc`, `vpa`, lifts, `tif`, `ttc_module_spearman`, `module_mdc`, `hec`.

**Score space: `pca`** (PCA on training cells; avoids gene-space AE reconstruction floor).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `wot_official` calls Broad WOT (`wot.ot.OTModel.compute_transport_map`). `scvelo_official` calls `scvelo.pp.neighbors` (no unspliced layers, so `scv.tl.velocity` is not used). `prescient_potential_flow` / `mioflow_neural_ode` remain family reimplementations. `wot_barycentric` is the in-harness WOT-inspired map without growth rates.

         method  energy_distance  energy_centered  gauss_w2_mean  gauss_w2_cov  mean_shift_l2  var_ratio  energy_minus_ae_floor  ot_sinkhorn      mmd  mean_marginal_w1      mdc      vpa  lift_energy   lift_w1    lift_l2  lift_ae_energy  lift_ae_w1  lift_ae_l2      tif  ttc_module_spearman  module_mdc      hec
      our_model         3.066112         1.908920      30.451209    201.005409       5.515248   1.116958               0.266112   138.813194 0.149524          1.166653 0.641993 0.623754    -0.739290 -0.547542   1.431358        0.031122    0.001742    0.028696 0.942321                 1.00    0.959684 0.999987
   wot_official         5.643326         0.323325     140.478750     65.907973      11.611965   0.786296               2.843326   162.729462 0.047616          0.993351 0.339584 0.391365    -3.316504 -0.374240  -4.665359       -2.546092    0.175044   -6.068020 0.976254                 0.25    0.218979 0.999987
scvelo_official        42.989219        19.590150    3360.008522  12049.424772      46.583392  34.875226              40.189219  8304.930542 0.048208          8.145938 0.131288 0.172179   -40.662396 -7.526828 -39.636786      -39.891985   -6.977543  -41.039448 0.897633                 0.00   -0.024909 0.999975

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