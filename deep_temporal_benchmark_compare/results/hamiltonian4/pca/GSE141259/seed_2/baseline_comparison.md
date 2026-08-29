# Baseline comparison (mean over held-out transitions)

Lower is better for all metrics.

**Score space: `pca`** (PCA on training cells; avoids gene-space AE reconstruction floor).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `prescient_potential_flow` and `mioflow_neural_ode` are controlled reimplementations of the corresponding method-family objectives, not official package runs. `wot_barycentric` is a WOT-inspired entropic-transport baseline without WOT's growth-rate model.

                  method  energy_distance  ot_sinkhorn      mmd  mean_marginal_w1  mean_shift_l2
prescient_potential_flow         2.753909   159.851257 0.021008          0.586027       7.583259
      mioflow_neural_ode         2.777489   161.168800 0.021049          0.593218       7.639718
         wot_barycentric         3.357251   171.363693 0.029820          0.739821       8.380515
               our_model         5.967350   131.347263 0.306029          1.555597       4.810016

## Winner per metric

- `ot_sinkhorn`: **our_model** _(training objective — reference only)_
- `mmd`: **prescient_potential_flow**
- `energy_distance`: **prescient_potential_flow**
- `mean_marginal_w1`: **prescient_potential_flow**
- `mean_shift_l2`: **our_model**

Best method by the primary metric (`energy_distance`): **prescient_potential_flow**.