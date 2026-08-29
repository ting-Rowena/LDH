# Baseline comparison (mean over held-out transitions)

Lower is better for all metrics.

**Score space: `pca`** (PCA on training cells; avoids gene-space AE reconstruction floor).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `prescient_potential_flow` and `mioflow_neural_ode` are controlled reimplementations of the corresponding method-family objectives, not official package runs. `wot_barycentric` is a WOT-inspired entropic-transport baseline without WOT's growth-rate model.

                  method  energy_distance  ot_sinkhorn      mmd  mean_marginal_w1  mean_shift_l2
prescient_potential_flow         2.440863   101.042465 0.022545          0.622768       7.161414
      mioflow_neural_ode         2.656800   105.349075 0.023519          0.666111       7.557287
               our_model         3.266654   107.269295 0.158367          1.254738       4.276676
         wot_barycentric         5.087714   182.714306 0.046412          0.996499      10.924747

## Winner per metric

- `ot_sinkhorn`: **prescient_potential_flow** _(training objective — reference only)_
- `mmd`: **prescient_potential_flow**
- `energy_distance`: **prescient_potential_flow**
- `mean_marginal_w1`: **prescient_potential_flow**
- `mean_shift_l2`: **our_model**

Best method by the primary metric (`energy_distance`): **prescient_potential_flow**.