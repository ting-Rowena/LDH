# Baseline comparison (mean over held-out transitions)

Lower is better for all metrics.

**Score space: `pca`** (PCA on training cells; avoids gene-space AE reconstruction floor).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `prescient_potential_flow` and `mioflow_neural_ode` are controlled reimplementations of the corresponding method-family objectives, not official package runs. `wot_barycentric` is a WOT-inspired entropic-transport baseline without WOT's growth-rate model.

                  method  energy_distance  ot_sinkhorn      mmd  mean_marginal_w1  mean_shift_l2
prescient_potential_flow         2.320772    96.417030 0.021609          0.617689       6.935970
      mioflow_neural_ode         2.367484    98.502792 0.021974          0.658737       7.058748
               our_model         4.615001   137.863670 0.170966          1.291557       7.340510
         wot_barycentric         4.892011   181.930622 0.044508          0.990984      10.714982

## Winner per metric

- `ot_sinkhorn`: **prescient_potential_flow** _(training objective — reference only)_
- `mmd`: **prescient_potential_flow**
- `energy_distance`: **prescient_potential_flow**
- `mean_marginal_w1`: **prescient_potential_flow**
- `mean_shift_l2`: **prescient_potential_flow**

Best method by the primary metric (`energy_distance`): **prescient_potential_flow**.