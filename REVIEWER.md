# Reviewer reproduction

**Code:** https://github.com/ting-Rowena/LDH  
**Weights:** https://doi.org/10.5281/zenodo.22146979

Two LDH weight families exist. **Do not mix them.**

| What you want | Weights | Command |
| --- | --- | --- |
| Main figures / most supplementary panels | **Adopted landscape** (three folders at repo root) | `python output_file/reproduce.py --group fast` |
| Redraw Figure 2 from the committed transport table | none extra (`deep_temporal_benchmark_compare/Supplementary_table2.csv` is in git) | `python output_file/figure2.py` |
| Recompute Figure 2 OT vs WOT / PRESCIENT-family / MIOFlow-family | **Hamiltonian4** under `deep_temporal_benchmark_compare/` | see that folder’s README |
| Hybrid KO, 3D landscapes, PDVS OS | Adopted + processed `.h5ad` (and GPU / network) | `python output_file/reproduce.py --group all` |

## 1. Code

```bash
git clone https://github.com/ting-Rowena/LDH.git
cd LDH
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Install a PyTorch wheel that matches your CUDA/CPU: https://pytorch.org
```

## 2. Weights (Zenodo)

Download the two archives from https://doi.org/10.5281/zenodo.22146979 and unzip
**inside the clone root** so paths match `output_file/_adopted.py`:

```bash
# from the LDH clone root
unzip ldh-scrna-adopted-landscape-checkpoints.zip
# optional, only to recompute Figure 2 OT:
unzip ldh-scrna-hamiltonian4-checkpoints.zip
python output_file/reproduce.py --check
```

`--check` must report the three adopted directories and
`deep_temporal_benchmark_compare/Supplementary_table2.csv`.

## 3. Figures

```bash
python output_file/reproduce.py --group fast
```

Processed GEO/EGA `.h5ad` files are **not** on GitHub or in the Zenodo weight
bundle. They are only required for `--group all` jobs tagged `h5ad`. See
[`DATA_AND_CHECKPOINTS.md`](DATA_AND_CHECKPOINTS.md).
