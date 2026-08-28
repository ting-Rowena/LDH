# Supplementary Table 2. LDH-scRNA versus Waddington-OT, PRESCIENT-family, and MIOFlow-family on PCA-50

This note documents the dataset-averaged table in

`deep_temporal_benchmark_compare/Supplementary_table2.csv`

(same numbers as `h4_pca50_wot_prescient_mioflow_l2_vpa_mdc_dataset_mean.csv`)

from protocol through metric definitions, method implementations, aggregation, and the numeric results. The table is a **location and direction** comparison. It is not an Energy / MMD / Sinkhorn ranking, and it does **not** include scVelo.

CSV columns: `centroid Euclidean distance`, `Mean Displacement Cosine`, `Velocity Projection Alignment`. Method label for this model: `LDH-scRNA`.

---

## 1. Purpose

Single-cell time-series methods are often scored with cloud-overlap distances (Energy, MMD, global Wasserstein). Those quantities mix (i) whether the predicted population moved to the right place and (ii) whether the predicted cloud has the same spread as the data. Deterministic latent integration plus a decoder can shrink variance (variance collapse). Overlap metrics then penalize a model whose **mean trajectory** is correct.

Supplementary Table 2 therefore reports three complementary functionals of the same predicted and observed clouds, all computed in a shared 50-dimensional PCA space:

| Column in the CSV | Asks | Range | Better |
| --- | --- | --- | --- |
| centroid Euclidean distance | Did the predicted cloud sit at the right location? | \([0, +\infty)\) | lower |
| Mean Displacement Cosine | Did the population barycenter move in the observed direction? | \([-1, 1]\) | higher |
| Velocity Projection Alignment | Did each source cell move along a plausible coupling to the hold-out target? | \([-1, 1]\) | higher |

centroid Euclidean distance cannot be negative. Mean Displacement Cosine and Velocity Projection Alignment are cosines and **can be negative** (predicted displacement opposite the reference).

---

## 2. Cohorts, splits, and prediction task

All four methods receive the **same** source subsample, the **same** hold-out target cells, and the **same** scoring map. Random seeds are \(0,1,2\).

| Label in the CSV | Accession / cohort | Train / hold-out | Transitions averaged |
| --- | --- | --- | --- |
| Pain | GSE155622 (nerve injury) | Train times \(0\)–\(7\,\mathrm{d}\); hold **\(14\,\mathrm{d}\)** | \((0\to 14)\), \((7\to 14)\) |
| Lung | GSE141259 (bleomycin) | Train D0–D21; hold **D28** | \((0\to 28)\), \((21\to 28)\) |
| HGSOC | high-grade serous ovarian carcinoma | `val_mode=patients`; \(\hat P\) from train patients at \(t=0\); \(P\) = hold-out patients **EOC136, EOC153** at \(t=1\) | \((0\to 1)\) |

For each transition \((t_{\mathrm{curr}}, t_{\mathrm{next}})\):

- **Source** \(X_{\mathrm{curr}}\): training-mask cells at \(t_{\mathrm{curr}}\), uniformly subsampled to at most **512** cells (shared across methods).
- **Target** \(X_{\mathrm{target}}\): validation-mask cells at \(t_{\mathrm{next}}\) (metrics further subsample the scored clouds to at most **1000** cells with a fixed seed).
- Each method maps \(X_{\mathrm{curr}}\) to a predicted next population \(\hat X\) in gene space (highly variable genes of the processed AnnData).
- \(\hat X\), \(X_{\mathrm{curr}}\), and \(X_{\mathrm{target}}\) are then projected with the **same** PCA (Section 3) before centroid Euclidean distance, Mean Displacement Cosine, and Velocity Projection Alignment.

---

## 3. Scoring space (PCA-50)

Let \(X_{\mathrm{train}}\) be all training-split cells (not only the source subsample). Fit sklearn `PCA` with `n_components=50` and `random_state=seed` on \(X_{\mathrm{train}}\). Write \(\Pi\) for the linear map to 50 principal coordinates.

Gene-space clouds are scored as

\[
Z = \Pi(X),\qquad
\hat Z = \Pi(\hat X),\qquad
Z_{\mathrm{curr}} = \Pi(X_{\mathrm{curr}}),\qquad
Z_{\mathrm{target}} = \Pi(X_{\mathrm{target}}).
\]

PCA is fit **only on training cells** so the hold-out does not define the axes. Fifty components (not 30) is the setting of this table. LDH-scRNA still **decodes to genes** before \(\Pi\); Waddington-OT and the two family flows also return gene-space \(\hat X\) then \(\Pi\).

The CSV column `score_space` is `pca50`.

---

## 4. Four methods

### 4.1 LDH-scRNA (CSV: `LDH-scRNA`)

LDH-scRNA evaluated at the Hamiltonian-flow checkpoints `deep_temporal_benchmark_compare/Hamiltonian4_GSE155622`, `Hamiltonian4_GSE141259`, and `Hamiltonian4_HGSOC` (directory names on disk). Architecture: encode expression and cell type to latent \(z\), integrate a **damped Hamiltonian / state-momentum** flow, decode to genes.

Training uses a four-term objective of the form

\[
L = L_{\mathrm{OT}}(z) + \lambda_{\mathrm{ae}} L_{\mathrm{ae}} + \lambda_{d} L_{\mathrm{density}} + \lambda_{H} L_{H},
\]

with OT matching against **OT-barycentric** latent targets (not a test-time Energy loss). At evaluation, \(\hat X\) is the decoded population after integrating from \(t_{\mathrm{curr}}\) to \(t_{\mathrm{next}}\). Code path: `our_model` in `baseline_evaluation.py`.

### 4.2 Waddington-OT (CSV: `Waddington-OT`)

**Official** Broad Waddington-OT: `wot.ot.OTModel.compute_transport_map`. On intervals that exist in the **training** time grid, the source cloud is pushed by the transport map (barycentric images), then transferred onto the query source cells by \(k\)-NN displacement. For hold-out times **beyond** the last training day (Pain \(14\,\mathrm{d}\), Lung D28), the last in-train map is **scaled in time** (linear extrapolation of the displacement). Growth-rate models in the full Waddington-OT tutorial are not used. Code path: `wot_official`.

### 4.3 PRESCIENT-family (CSV: `PRESCIENT-family`)

**Not** the published PRESCIENT Python package. In-harness reimplementation of the family idea: a first-order potential flow \(\mathrm{d}x = -\nabla\Psi(x)\,\mathrm{d}t\) in a **training-fit PCA** (up to 50 PCs, \(\le 160\) points per train hop), trained for 120 Adam steps with a Gaussian MMD kernel loss, then inverse-transformed to genes. This is a controlled core-objective baseline, not an official PRESCIENT run. Code path: `prescient_potential_flow`.

### 4.4 MIOFlow-family (CSV: `MIOFlow-family`)

**Not** the published MIOFlow package. In-harness time-conditioned neural ODE \(\mathrm{d}x = v(x,t)\,\mathrm{d}t\), same reduced-space protocol and MMD training as Section 4.3, Euler rollout, inverse-PCA to genes. Code path: `mioflow_neural_ode`.

On long extrapolation these two flows often stay near persistence (Mean Displacement Cosine / Velocity Projection Alignment near 0 or negative). That is an empirical outcome of this harness, not a claim about the original software.

---

## 5. Metric definitions (as implemented)

All formulas below use the PCA-50 coordinates \(Z_{\mathrm{curr}}\), \(\hat Z\), \(Z_{\mathrm{target}}\). Write \(N\) for the number of paired source cells (after the shared \(\le 512\) subsample). Predictions from Waddington-OT and the family flows are \(k\)-NN (or flow) displacements of those same cells, so \(\hat Z_i\) is aligned with \(Z_{\mathrm{curr},i}\).

### 5.1 centroid Euclidean distance

Let \(\hat\mu = N^{-1}\sum_i \hat Z_i\) and \(\mu = |Z_{\mathrm{target}}|^{-1}\sum_j Z_{\mathrm{target},j}\).

\[
\text{centroid Euclidean distance}
= \bigl\| \hat\mu - \mu \bigr\|_2
= \sqrt{\sum_{k=1}^{50} (\hat\mu_k - \mu_k)^2}.
\]

**Meaning.** Location of the predicted cloud versus the hold-out cloud. Independent of within-cloud covariance. Code: `mean_shift_l2` in `population_distribution_metrics`.

**Why it helps LDH-scRNA.** Momentum integration can keep long-horizon means from drifting (especially Pain \(0\to 14\)). It does **not** prove that the predicted shape matches the data.

### 5.2 Mean Displacement Cosine

\[
\text{Mean Displacement Cosine}
=
\frac{
\bigl\langle
\hat\mu - \bar Z_{\mathrm{curr}},\;
\mu - \bar Z_{\mathrm{curr}}
\bigr\rangle
}{
\bigl\| \hat\mu - \bar Z_{\mathrm{curr}} \bigr\|_2
\;
\bigl\| \mu - \bar Z_{\mathrm{curr}} \bigr\|_2
},
\]

with \(\bar Z_{\mathrm{curr}} = N^{-1}\sum_i Z_{\mathrm{curr},i}\). If either displacement has Euclidean norm \(< 10^{-9}\), the implementation returns NaN.

**Meaning.** Cosine of the **population** velocity. \(+1\): barycenter moves parallel to the observed mean shift; \(-1\): opposite. Mean Displacement Cosine **does not penalize step size**: a method can point the right way and still have large centroid Euclidean distance (overshoot). Code: `_cosine` on the two mean displacements.

### 5.3 Velocity Projection Alignment

Velocity Projection Alignment is the cell-wise OT-coupled velocity projection. The “true” per-cell destination is **not** an arbitrary matching. It is the entropic-OT barycentric image of each source point onto a subsample (\(\le 1000\)) of \(Z_{\mathrm{target}}\):

1. Sinkhorn plan \(P\) between \(\{Z_{\mathrm{curr},i}\}\) and the subsampled target, cost \(\|z-z'\|_2^2\) (normalized), blur \(0.05\), as in `train_model._sinkhorn_plan`.
2. Barycentric map \(y^*(Z_{\mathrm{curr},i}) = (P_{i\cdot} / \sum_{j} P_{ij})\, Z_{\mathrm{target},j}^{\mathrm{sub}}\).
3. For cells with both predicted and coupled displacements longer than \(10^{-8}\),

\[
\text{Velocity Projection Alignment}
=
\frac{1}{|\mathcal{I}|}
\sum_{i\in\mathcal{I}}
\frac{
\bigl\langle
\hat Z_i - Z_{\mathrm{curr},i},\;
y^*(Z_{\mathrm{curr},i}) - Z_{\mathrm{curr},i}
\bigr\rangle
}{
\bigl\| \hat Z_i - Z_{\mathrm{curr},i} \bigr\|_2
\;
\bigl\| y^*(Z_{\mathrm{curr},i}) - Z_{\mathrm{curr},i} \bigr\|_2
}.
\]

Require \(|\mathcal{I}|\ge 8\); otherwise NaN.

**Meaning.** Average cosine between the model’s per-cell tangent and a **hold-out** coupling. This coupling is **not** Waddington-OT’s training transport map. Waddington-OT can still lose Velocity Projection Alignment on later hops if its extrapolated map points away from the true hold-out coupling. Code: `_ot_barycentric_map`, `_vpa`.

---

## 6. What “dataset mean” is

Each seed \(\in\{0,1,2\}\) and each hold-out hop produces one centroid Euclidean distance, one Mean Displacement Cosine, and one Velocity Projection Alignment.

- Pain / Lung: two hops \(\times\) three seeds \(= 6\) numbers per method, then arithmetic mean.
- HGSOC: one hop \(\times\) three seeds \(= 3\) numbers, then mean.

That average is exactly `Supplementary_table2.csv`. It **mixes** long hops (\(0\to 14\), \(0\to 28\)) with shorter ones (\(7\to 14\), \(21\to 28\)). Hop-resolved means (not in this file) still show Waddington-OT with higher Mean Displacement Cosine / Velocity Projection Alignment on Pain \(0\to 14\) and Lung \(0\to 28\), while LDH-scRNA has lower centroid Euclidean distance on every hop against these four methods.

---

## 7. Results (dataset means)

Numbers copied from the CSV (rounded to three decimals in the table; the CSV stores full float precision). **Bold**: best among the four methods in that dataset (centroid Euclidean distance min; Mean Displacement Cosine and Velocity Projection Alignment max). Column order matches the CSV.

### Pain (GSE155622)

| Method | centroid Euclidean distance ↓ | Mean Displacement Cosine ↑ | Velocity Projection Alignment ↑ |
| --- | ---: | ---: | ---: |
| **LDH-scRNA** | **4.240** | **0.776** | **0.743** |
| Waddington-OT | 12.041 | 0.332 | 0.398 |
| PRESCIENT-family | 7.010 | 0.138 | 0.063 |
| MIOFlow-family | 7.211 | −0.064 | −0.035 |

### Lung (GSE141259)

| Method | centroid Euclidean distance ↓ | Mean Displacement Cosine ↑ | Velocity Projection Alignment ↑ |
| --- | ---: | ---: | ---: |
| **LDH-scRNA** | **4.841** | **0.580** | **0.816** |
| Waddington-OT | 5.915 | 0.528 | 0.485 |
| PRESCIENT-family | 7.526 | −0.088 | −0.043 |
| MIOFlow-family | 7.632 | −0.050 | −0.038 |

### HGSOC

| Method | centroid Euclidean distance ↓ | Mean Displacement Cosine ↑ | Velocity Projection Alignment ↑ |
| --- | ---: | ---: | ---: |
| **LDH-scRNA** | **7.554** | **0.394** | **0.723** |
| Waddington-OT | 9.541 | −0.059 | 0.586 |
| PRESCIENT-family | 7.963 | −0.070 | −0.016 |
| MIOFlow-family | 8.086 | −0.048 | −0.029 |

**Winner summary.** Against Waddington-OT and the two family flows, LDH-scRNA is best on **all three datasets** for centroid Euclidean distance, Mean Displacement Cosine, and Velocity Projection Alignment at this aggregation.

This statement is **conditional on the comparator set**. If official scVelo neighbors are added on the same PCA-50 protocol, HGSOC centroid Euclidean distance and Mean Displacement Cosine favor scVelo (distance \(7.35\) vs LDH-scRNA \(7.55\); Mean Displacement Cosine \(0.515\) vs \(0.394\)). That comparison is **not** part of this CSV.

---

## 8. How to read the three columns together

1. **centroid Euclidean distance** answers location. LDH-scRNA’s largest gap versus Waddington-OT is Pain (4.24 vs 12.04): Waddington-OT’s extrapolated mean overshoots.
2. **Mean Displacement Cosine** answers only the angle of that mean shift. It can rank a method well while centroid Euclidean distance is poor, or the reverse if the step is short but slightly off-angle.
3. **Velocity Projection Alignment** answers per-cell alignment to a hold-out OT map. Dataset means favor LDH-scRNA; the longest hops can still favor Waddington-OT’s local geometry.
4. PRESCIENT-family / MIOFlow-family Mean Displacement Cosine and Velocity Projection Alignment near zero indicate little coherent displacement under this training budget—not a shape win.

Do not interpret these columns as “the predicted distribution matches.” Shape diagnostics (centered Energy, AE reconstruction floor, variance ratio) live in other result directories and are omitted here on purpose.

---

## 9. Reproducibility

| Item | Location |
| --- | --- |
| This table (dataset means only) | `deep_temporal_benchmark_compare/Supplementary_table2.csv` |
| Same numbers (duplicate filename) | `deep_temporal_benchmark_compare/h4_pca50_wot_prescient_mioflow_l2_vpa_mdc_dataset_mean.csv` |
| Same numbers plus hop- and seed-level rows | `deep_temporal_benchmark_compare/h4_pca50_wot_prescient_mioflow_l2_vpa_mdc.csv` |
| Full harness dump (includes scVelo) | `deep_temporal_benchmark_compare/results/hamiltonian4_official/pca50_families/` |
| Metrics and official/family baselines | `deep_temporal_benchmark_compare/baseline_evaluation.py` |
| Launcher | `deep_temporal_benchmark_compare/run_deep_temporal_benchmark.py` |
| LDH-scRNA weights | `deep_temporal_benchmark_compare/Hamiltonian4_GSE155622`, `deep_temporal_benchmark_compare/Hamiltonian4_GSE141259`, `deep_temporal_benchmark_compare/Hamiltonian4_HGSOC` |

Command used to generate the families run (PCA-50, seeds \(0\)–\(2\)):

```text
python deep_temporal_benchmark_compare/run_deep_temporal_benchmark.py \
  --datasets GSE155622 GSE141259 HGSOC --seeds 0 1 2 \
  --score-space pca --n-pca 50 --device cpu \
  --methods our_model wot_official scvelo_official prescient_potential_flow mioflow_neural_ode \
  --checkpoint-set hamiltonian4 \
  --save-dir .../results/hamiltonian4_official/pca50_families
```

The dataset-mean CSV is the four-method subset of that run, averaged over hops and seeds as in Section 6. Display names in the CSV are LDH-scRNA and the three full metric names above.
