# Baseline comparison (mean over held-out transitions)

Lower is better for all metrics.

**Score space: `pca`** (PCA on training cells; avoids gene-space AE reconstruction floor).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `prescient_potential_flow` and `mioflow_neural_ode` are controlled reimplementations of the corresponding method-family objectives, not official package runs. `wot_barycentric` is a WOT-inspired entropic-transport baseline without WOT's growth-rate model.

                  method  energy_distance  ot_sinkhorn      mmd  mean_marginal_w1  mean_shift_l2
         wot_barycentric         3.112161   194.191299 0.032043          0.946072       8.335657
prescient_potential_flow         3.113225   194.238113 0.032042          0.947192       8.338674
      mioflow_neural_ode         3.147593   195.370926 0.032447          0.962405       8.362564
               our_model         9.857197   198.264221 0.463952          1.841312       7.809997

## Winner per metric

- `ot_sinkhorn`: **wot_barycentric** _(training objective — reference only)_
- `mmd`: **prescient_potential_flow**
- `energy_distance`: **wot_barycentric**
- `mean_marginal_w1`: **wot_barycentric**
- `mean_shift_l2`: **our_model**

Best method by the primary metric (`energy_distance`): **wot_barycentric**.