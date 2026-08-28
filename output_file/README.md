# output_file generators

All commands assume the repository root as the working directory.

```bash
python output_file/reproduce.py --check
python output_file/reproduce.py --group fast
```

## Main figures

| File | Command |
| --- | --- |
| figure2.png | `python output_file/figure2.py` |
| figure3_bc.png | `python output_file/figure3_bc.py` |
| figure3_de.png | `python output_file/figure3_de.py` |
| figure3_fg.png | `python output_file/figure3_fg.py` |
| figure3_hijk.png | `python output_file/figure3_hijk.py` |
| figure4_bc.png | `python output_file/figure4_bc.py` |
| figure4_d.html / topview PNG | `python output_file/figure4_d.py` |
| figure4_efghi.png | `python output_file/figure4_efghi.py` |
| figure4_j.png | `python output_file/figure4_j.py` |
| figure4_klm.png | `python output_file/figure4_klm.py` |
| figure5_b.png | `python output_file/figure5_b.py` |
| figure5_cd.png | `python output_file/figure5_cd.py` |
| figure5_ef.png | `python output_file/figure5_ef.py` |

## Supplementary figures and tables

| File | Command |
| --- | --- |
| Supplementary_figure1–8.png | `python output_file/Supplementary_figureN.py` |
| Supplementary_table1.xlsx | `python output_file/Supplementary_table1.py` |
| Supplementary_table2.csv | `python output_file/Supplementary_table2.py` (SOTA PCC; slow) |
| Supplementary_table3.csv | `python output_file/Supplementary_table3.py` |
| Supplementary_table4.csv | `python output_file/Supplementary_table4.py --no-rebuild` |
| Supplementary_table5–8 | `python output_file/Supplementary_tableN.py` |
| Supplementary_table9/10 csv | produced by `Supplementary_figure5.py` |

Supplementary Figure 7 expects macrophage triad / path-endorsement tables under
`output_file/mac_landscape_audit/`. The reproduce driver runs:

```bash
python scripts/analyze_mac_fn1_m2_resolution_triad.py
python scripts/analyze_mac_landscape_path_endorsement.py
```

before `Supplementary_figure7.py`.

P0–P2 / Atf3-OE intermediates used by several figures live in `output_file/robustness/`.

Checkpoint paths are centralized in `_adopted.py`. Place the three adopted
folders as documented in `DATA_AND_CHECKPOINTS.md` (repository root). Captions
for tables are in `Supplementary_tables_1-8_captions.md`,
`Supplementary_table1_xlsx_caption.md`, and `Supplementary_table9_10_captions.md`.
