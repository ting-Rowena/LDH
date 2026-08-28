# Data and checkpoints

Weights and processed AnnData are **not** in git (see `.gitignore`). Place them
next to `run_training.py` using the **exact directory and file names** below.
`output_file/_adopted.py` hard-codes those names.

When the Zenodo record is published, put the **concept DOI** here (and in
[`REVIEWER.md`](REVIEWER.md)):

```
Zenodo:  https://doi.org/10.5281/zenodo.<ID>
```

**Code** is https://github.com/ting-Rowena/LDH (weights are not in git).

Build the two upload zips on the machine that has the checkpoints:

```bash
bash scripts/pack_zenodo_checkpoints.sh
# → zenodo_staging/ldh-scrna-adopted-landscape-checkpoints.zip
# → zenodo_staging/ldh-scrna-hamiltonian4-checkpoints.zip
```

Reviewers unzip those archives at the clone root. Until the DOI is filled in,
point them at a private share of the same zips.

## Two LDH weight families (do not mix)

| Family | Role | Location |
| --- | --- | --- |
| **Adopted landscape** | Main figures, supplementary figures/tables except the four-method OT numbers in Figure 2 | Repo root, names below |
| **Hamiltonian4** | Figure 2 / `deep_temporal_benchmark_compare/Supplementary_table2.csv` four-method transport | `deep_temporal_benchmark_compare/Hamiltonian4_<DATASET>/` |

Adopted weights select on `pcc_then_mse` with quasi-potential + density terms.
Hamiltonian4 uses `--loss-recipe hamiltonian` (OT + AE + density + Hamiltonian
penalty; no residual drift). Using one family to redraw the other panel is incorrect.

The committed CSV `deep_temporal_benchmark_compare/Supplementary_table2.csv` is
enough to **redraw** Figure 2. Regenerating it requires Hamiltonian4 weights
plus the in-harness WOT / PRESCIENT-family / MIOFlow-family runs
(`deep_temporal_benchmark_compare/README.md`).

## Adopted landscape checkpoints (repo root)

```
GSE155622_checkpoints_3000_3000_384_0.05_recon0.01_lossnorm_qp_d0p01_z0p5_k0p2_ld1/
GSE141259_checkpoints_5000_5000_512_0.01_recon0.1_valD28_timeX_lossnorm_qp_d0p01_z0p1_k0p2/
HGSOC_checkpoints_3000_3000_512_0.05_recon0.01_nactpair_lossnorm_qp_d0p01_z0p5_k0p2_ld1/
```

Each folder must contain at least:

| File | Used for |
| --- | --- |
| `best_model.pth` | Dynamics, KO, SOTA, 3D landscapes |
| `obs.csv` | Cell metadata / UMAP coordinates in many panels |
| `Loss_epoch.csv` | Supplementary Figure 3 |
| `training_summary.json` | Hyperparameters / Table S1 |
| `training_var_names.json` | Gene alignment |
| `training_umap.npz` | Supplementary Figure 1 |

`python output_file/reproduce.py --check` verifies these paths.

Git ignores `*.pth`, `*.npz`, and `*_checkpoints_*/`. Keep the downloaded
bundle **untracked** on disk; do not `git add` it.

## Processed AnnData (repo root)

`dataset_pipeline.py` looks for the first existing path:

| Cohort | Local filename | Accession |
| --- | --- | --- |
| Mouse DRG SNI | `GSE155622_raw_UMI_counts_3.h5ad` | [GSE155622](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE155622) |
| Mouse bleomycin lung | `GSE141259_WholeLung.h5ad` | [GSE141259](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE141259) |
| HGSOC NACT-paired | `HGSOC.h5ad` | [EGAS00001005010](https://ega-archive.org/studies/EGAS00001005010) (restricted) |

Optional fallbacks (`./HGSOC/HGSOC.h5ad`, `../Transformer_julie/…`) are local
layout leftovers; **put files at the repo root** for a clean clone.

These `.h5ad` objects are already annotated (time/stage, cell type, HGSOC
`patient_id` + `treatment_phase`). Raw GEO matrices are **not** a drop-in
replacement. Training and `--group all` jobs tagged `h5ad` need them.
`--group fast` can run from checkpoint `obs.csv` / cached tables without
AnnData for most panels.

PDVS overall survival (Figure 5e–f, Supplementary Table 8) downloads or caches
TCGA-OV (Xena/GDC) and [GSE26712](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE26712)
at runtime.

## Retrain adopted landscape (optional)

Defaults in `apply_train_config` already turn on loss normalization,
quasi-stationary potential (`qp`), density (`d0p01`), and latent-flow terms.
Do **not** pass `--total-drift-hjb` for these paper weights.

Pain and HGSOC match the paper directory names with:

```bash
python run_training.py --dataset GSE155622
python run_training.py --dataset HGSOC
```

Lung adopted weights use profile `l3_genes5000` (5000 HVGs) and **override**
energy / recon / latent / displacement relative to that profile’s spec:

```bash
python run_training.py --dataset GSE141259 --profile l3_genes5000 \
  --lambda-energy 0.01 --lambda-recon 0.1 \
  --lambda-latent 0.1 --lambda-lat-disp 0 \
  --lambda-residual-balance 0
```

Expected output directory names are the three folders in the table above.
If the name differs, point `output_file/_adopted.py` at the new path.

A full run is thousands of epochs on a GPU with enough memory for the cohort
(HGSOC is the largest). Stochastic training will not bit-match `best_model.pth`.

## Retrain Hamiltonian4 (optional)

```bash
python run_training.py --dataset GSE155622 --loss-recipe hamiltonian \
  --checkpoint-dir deep_temporal_benchmark_compare/Hamiltonian4_GSE155622
python run_training.py --dataset GSE141259 --profile l3_genes5000 --loss-recipe hamiltonian \
  --checkpoint-dir deep_temporal_benchmark_compare/Hamiltonian4_GSE141259
python run_training.py --dataset HGSOC --loss-recipe hamiltonian \
  --checkpoint-dir deep_temporal_benchmark_compare/Hamiltonian4_HGSOC
```

Or: `bash scripts/run_hamiltonian4_three_datasets.sh` (waits for a free GPU;
`FORCE_CPU=1` trains on CPU).

## Intermediate tables already in git

These let many supplementary panels run without re-deriving audits:

- `output_file/mac_landscape_audit/` — Supplementary Figure 7
- `output_file/robustness/` — P0–P2 / Atf3-OE tables
- `deep_temporal_benchmark_compare/Supplementary_table2.csv` — Figure 2 transport numbers

Expensive recomputes write `output_file/_cache/` (gitignored).
