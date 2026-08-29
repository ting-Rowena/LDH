# Baseline comparison (mean over held-out transitions)

Lower is better for all metrics.

**Score space: `pca`** (PCA on training cells; avoids gene-space AE reconstruction floor).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `prescient_potential_flow` and `mioflow_neural_ode` are controlled reimplementations of the corresponding method-family objectives, not official package runs. `wot_barycentric` is a WOT-inspired entropic-transport baseline without WOT's growth-rate model.

                  method  energy_distance  ot_sinkhorn      mmd  mean_marginal_w1  mean_shift_l2
prescient_potential_flow         2.793859   162.272377 0.020806          0.599176       7.766605
      mioflow_neural_ode         2.867987   164.138115 0.021108          0.598984       7.858899
         wot_barycentric         3.451284   173.656876 0.030044          0.747799       8.624289
               our_model         5.831454   128.855576 0.289892          1.539808       4.945669

## Winner per metric

- `ot_sinkhorn`: **our_model** _(training objective — reference only)_
- `mmd`: **prescient_potential_flow**
- `energy_distance`: **prescient_potential_flow**
- `mean_marginal_w1`: **mioflow_neural_ode**
- `mean_shift_l2`: **our_model**

Best method by the primary metric (`energy_distance`): **prescient_potential_flow**.