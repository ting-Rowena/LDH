# LDH-scRNA analysis reproduction

**Repository:** https://github.com/ting-Rowena/LDH  
**Reviewer checklist:** [`REVIEWER.md`](REVIEWER.md)  
**Weights (Zenodo):** https://doi.org/10.5281/zenodo.22146979

This repository trains **LDH-scRNA** (latent damped-Hamiltonian / second-order
Langevin dynamics on scRNA-seq time series) and writes manuscript figures and
tables under [`output_file/`](output_file/).

Figure 1 is a model schematic drawn separately and is **not** generated here.

## What this package reproduces

From the repository root, after checkpoints (and, for some panels, AnnData)
are in place — see [DATA_AND_CHECKPOINTS.md](DATA_AND_CHECKPOINTS.md):

| Output | Script |
| --- | --- |
| `output_file/figure2.png` | `output_file/figure2.py` |
| `output_file/figure3_*.png` | `output_file/figure3_bc.py` … `figure3_hijk.py` |
| `output_file/figure4_*.png` / `.html` | `output_file/figure4_*.py` |
| `output_file/figure5_*.png` | `output_file/figure5_b.py`, `figure5_cd.py`, `figure5_ef.py` |
| `output_file/Supplementary_figure1.png`–`8.png` | `output_file/Supplementary_figure1.py`–`8.py` |
| `output_file/Supplementary_table1.xlsx` / `2–8` | `output_file/Supplementary_table*.py` |
| `Supplementary_table9/10` CSVs | written by `Supplementary_figure5.py` |

The four-method PCA-50 transport table used in Figure 2 lives in
[`deep_temporal_benchmark_compare/Supplementary_table2.csv`](deep_temporal_benchmark_compare/Supplementary_table2.csv)
(LDH-scRNA vs Waddington-OT / PRESCIENT-family / MIOFlow-family). Regenerating
that CSV from scratch is a separate protocol (see that folder’s README).

## Setup

```bash
git clone https://github.com/ting-Rowena/LDH.git
cd LDH
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python **3.10+** is recommended. Install a **PyTorch** wheel that matches your
CUDA or CPU setup from https://pytorch.org if the pip default is wrong.

Place **adopted checkpoints** and, for slow jobs, **processed `.h5ad`** as
described in [DATA_AND_CHECKPOINTS.md](DATA_AND_CHECKPOINTS.md). Weights are
on Zenodo ([10.5281/zenodo.22146979](https://doi.org/10.5281/zenodo.22146979)),
not on GitHub.

```bash
python output_file/reproduce.py --check
python output_file/reproduce.py --list --group fast
python output_file/reproduce.py --group fast      # obs.csv / Loss_epoch / cached tables
python output_file/reproduce.py --group all       # hybrid KO, 3D landscapes, PDVS, h5ad
```

Single job:

```bash
python output_file/reproduce.py --only supp_fig1 figure2
python output_file/Supplementary_figure6.py
```

`--group fast` is the reviewer default when the three adopted folders are
present. `--group all` needs GPU for several jobs, raw AnnData for panels
tagged `h5ad`, and network access for TCGA/GSE26712 (Figure 5 / Table S8).

## Training (optional)

`apply_train_config` already enables loss normalization, quasi-stationary
potential, and density alignment. **Do not** add `--total-drift-hjb` if you
want the paper landscape folders.

```bash
# Adopted landscape (directory names must match DATA_AND_CHECKPOINTS.md)
python run_training.py --dataset GSE155622
python run_training.py --dataset HGSOC
python run_training.py --dataset GSE141259 --profile l3_genes5000 \
  --lambda-energy 0.01 --lambda-recon 0.1 \
  --lambda-latent 0.1 --lambda-lat-disp 0 \
  --lambda-residual-balance 0

# Hamiltonian4 (Figure 2 transport LDH only — not the landscape weights)
python run_training.py --dataset GSE155622 --loss-recipe hamiltonian \
  --checkpoint-dir deep_temporal_benchmark_compare/Hamiltonian4_GSE155622
python run_training.py --dataset GSE141259 --profile l3_genes5000 --loss-recipe hamiltonian \
  --checkpoint-dir deep_temporal_benchmark_compare/Hamiltonian4_GSE141259
python run_training.py --dataset HGSOC --loss-recipe hamiltonian \
  --checkpoint-dir deep_temporal_benchmark_compare/Hamiltonian4_HGSOC
```

If a retrained folder name differs, edit `output_file/_adopted.py`.

## Public data

| Cohort | Accession | Role |
| --- | --- | --- |
| Mouse DRG SNI | [GSE155622](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE155622) | Pain time course |
| Mouse bleomycin lung | [GSE141259](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE141259) | Lung time course |
| HGSOC NACT-paired |  [GSE165897](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE165897)(Count matrix）/[EGAS00001005010](https://ega-archive.org/studies/EGAS00001005010)(FASTQ) | Patient-paired TN→PN |
| PDVS OS | TCGA-OV (Xena / GDC); [GSE26712](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE26712) | Figure 5 / Table S8 |

Processed files expected at the repo root: `GSE155622_raw_UMI_counts_3.h5ad`,
`GSE141259_WholeLung.h5ad`, `HGSOC.h5ad`.

## Layout

```
train_model.py              # LDH-scRNA architecture + trainer
hamiltonian_flow.py         # (z, p) integration
dataset_pipeline.py         # cohort specs and preprocessing
run_training.py             # CLI
DATA_AND_CHECKPOINTS.md     # weights, h5ad names, retrain commands
output_file/                # figure/table generators + reproduce.py
output_file/_adopted.py     # paths to the three paper landscape checkpoints
output_file/mac_landscape_audit/  # Supplementary Figure 7 intermediate tables
scripts/                    # KO, CCC, macrophage audits, PDVS eviction
deep_temporal_benchmark_compare/   # four-method transport + Hamiltonian4 weights
```

## Notes for reviewers

- **Discrete snapshots vs continuous trajectories:** experimental time is
  discrete; \(z(t)\) is the latent ODE/SDE solution between user-specified times.
- **HGSOC hold-out** is by **patient** (not a third time point): train patients
  at \(t=0\) (treatment-naive) → score against hold-out patients at \(t=1\)
  (post-NACT).
- PRESCIENT-family / MIOFlow-family in the transport table are **in-harness**
  baselines, not the official packages, unless you rerun with those packages
  yourself.
- Landscape LDH and Hamiltonian4 LDH are different objectives; Figure 2 OT
  numbers come from Hamiltonian4 + the committed transport CSV.

## License

No reuse license has been selected yet. The repository is public for peer
review and reproducibility inspection; contact the authors for permission to
reuse or redistribute the code.
