# Baseline comparison (mean over held-out transitions)

Lower is better for all metrics.

**Score space: `pca`** (PCA on training cells; avoids gene-space AE reconstruction floor).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `prescient_potential_flow` and `mioflow_neural_ode` are controlled reimplementations of the corresponding method-family objectives, not official package runs. `wot_barycentric` is a WOT-inspired entropic-transport baseline without WOT's growth-rate model.

                  method  energy_distance  ot_sinkhorn      mmd  mean_marginal_w1  mean_shift_l2
prescient_potential_flow         2.554620   158.823685 0.019306          0.584588       7.228346
      mioflow_neural_ode         2.697099   161.418549 0.020110          0.616812       7.396126
         wot_barycentric         3.164856   169.736610 0.028300          0.746368       8.107434
               our_model         5.570737   127.120327 0.277683          1.518298       4.767144

## Winner per metric

- `ot_sinkhorn`: **our_model** _(training objective — reference only)_
- `mmd`: **prescient_potential_flow**
- `energy_distance`: **prescient_potential_flow**
- `mean_marginal_w1`: **prescient_potential_flow**
- `mean_shift_l2`: **our_model**

Best method by the primary metric (`energy_distance`): **prescient_potential_flow**.