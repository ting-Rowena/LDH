# Baseline comparison (mean over held-out transitions)

Lower is better for all metrics.

**Score space: `pca`** (PCA on training cells; avoids gene-space AE reconstruction floor).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `prescient_potential_flow` and `mioflow_neural_ode` are controlled reimplementations of the corresponding method-family objectives, not official package runs. `wot_barycentric` is a WOT-inspired entropic-transport baseline without WOT's growth-rate model.

                  method  energy_distance  ot_sinkhorn      mmd  mean_marginal_w1  mean_shift_l2
         wot_barycentric         2.902848   185.525299 0.031199          0.914253       7.930817
prescient_potential_flow         2.905700   185.575165 0.031202          0.914758       7.937472
      mioflow_neural_ode         3.066888   190.785431 0.031822          0.941994       8.270393
               our_model        23.251585   386.983398 0.733251          2.693477      15.800371

## Winner per metric

- `ot_sinkhorn`: **wot_barycentric** _(training objective — reference only)_
- `mmd`: **wot_barycentric**
- `energy_distance`: **wot_barycentric**
- `mean_marginal_w1`: **wot_barycentric**
- `mean_shift_l2`: **wot_barycentric**

Best method by the primary metric (`energy_distance`): **wot_barycentric**.