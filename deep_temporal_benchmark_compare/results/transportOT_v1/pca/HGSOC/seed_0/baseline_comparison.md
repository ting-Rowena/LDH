# Baseline comparison (mean over held-out transitions)

Lower is better for all metrics.

**Score space: `pca`** (PCA on training cells; avoids gene-space AE reconstruction floor).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `prescient_potential_flow` and `mioflow_neural_ode` are controlled reimplementations of the corresponding method-family objectives, not official package runs. `wot_barycentric` is a WOT-inspired entropic-transport baseline without WOT's growth-rate model.

                  method  energy_distance  ot_sinkhorn      mmd  mean_marginal_w1  mean_shift_l2
         wot_barycentric         2.719969   180.798355 0.029872          0.857659       7.611233
prescient_potential_flow         2.721178   180.771759 0.029877          0.856871       7.613746
      mioflow_neural_ode         2.728139   181.404129 0.029911          0.854684       7.623886
               our_model        22.716963   382.371582 0.722345          2.651954      15.391665

## Winner per metric

- `ot_sinkhorn`: **prescient_potential_flow** _(training objective — reference only)_
- `mmd`: **wot_barycentric**
- `energy_distance`: **wot_barycentric**
- `mean_marginal_w1`: **mioflow_neural_ode**
- `mean_shift_l2`: **wot_barycentric**

Best method by the primary metric (`energy_distance`): **wot_barycentric**.