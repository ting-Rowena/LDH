# Baseline comparison (mean over held-out transitions)

Lower is better for all metrics.

**Score space: `pca`** (PCA on training cells; avoids gene-space AE reconstruction floor).

**Primary ranking metric: `energy_distance`** (a non-training objective; the table is sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for reference only — comparing methods on it would be biased toward the model.

**Implementation fidelity:** `prescient_potential_flow` and `mioflow_neural_ode` are controlled reimplementations of the corresponding method-family objectives, not official package runs. `wot_barycentric` is a WOT-inspired entropic-transport baseline without WOT's growth-rate model.

                  method  energy_distance  ot_sinkhorn      mmd  mean_marginal_w1  mean_shift_l2
         wot_barycentric         3.112161   194.191299 0.032043          0.946070       8.335657
prescient_potential_flow         3.113225   194.238129 0.032042          0.947191       8.338675
      mioflow_neural_ode         3.147595   195.370941 0.032447          0.962403       8.362567
               our_model        23.056055   387.019836 0.734819          2.689501      15.562018

## Winner per metric

- `ot_sinkhorn`: **wot_barycentric** _(training objective — reference only)_
- `mmd`: **prescient_potential_flow**
- `energy_distance`: **wot_barycentric**
- `mean_marginal_w1`: **wot_barycentric**
- `mean_shift_l2`: **wot_barycentric**

Best method by the primary metric (`energy_distance`): **wot_barycentric**.