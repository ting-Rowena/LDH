# Baseline comparison (mean over held-out transitions)

Lower is better for all metrics.

**Score space: `pca`** (PCA on training cells; avoids gene-space AE reconstruction floor).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `prescient_potential_flow` and `mioflow_neural_ode` are controlled reimplementations of the corresponding method-family objectives, not official package runs. `wot_barycentric` is a WOT-inspired entropic-transport baseline without WOT's growth-rate model.

                  method  energy_distance  ot_sinkhorn      mmd  mean_marginal_w1  mean_shift_l2
prescient_potential_flow         2.554619   158.823666 0.019306          0.584587       7.228343
      mioflow_neural_ode         2.697097   161.418556 0.020110          0.616812       7.396123
         wot_barycentric         3.164855   169.736603 0.028300          0.746369       8.107433
               our_model        23.706627   311.744781 0.754937          2.432074      17.640429

## Winner per metric

- `ot_sinkhorn`: **prescient_potential_flow** _(training objective — reference only)_
- `mmd`: **prescient_potential_flow**
- `energy_distance`: **prescient_potential_flow**
- `mean_marginal_w1`: **prescient_potential_flow**
- `mean_shift_l2`: **prescient_potential_flow**

Best method by the primary metric (`energy_distance`): **prescient_potential_flow**.