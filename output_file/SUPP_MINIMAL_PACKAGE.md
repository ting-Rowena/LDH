# Minimal supplementary package (SFig1–8, STable5–10)

Main-figure numbering: `output_file/figureN*.py` = manuscript Figure N.  
Figure 1 is a separately made model schematic (not generated here).

Generated from adopted checkpoints (and ``output_file/mac_landscape_audit/`` for Supplementary Figure 7).

| Asset | Script | Content |
|-------|--------|---------|
| `Supplementary_figure1.png` | `Supplementary_figure1.py` | Three-dataset UMAP atlas |
| `Supplementary_figure2.png` | `Supplementary_figure2.py` | Stage-wise composition / HGSOC counts |
| `Supplementary_figure3.png` | `Supplementary_figure3.py` | Training PCC / MSE curves |
| `Supplementary_figure4.png` | `Supplementary_figure4.py` | *Atf3* honesty: *Egr1* KO, cross-type, OE |
| `Supplementary_figure5.png` | `Supplementary_figure5.py` | GSE155622 injury DEGs × 9 cell-type correlation + Neuron4 $U_0$/$U_{rel}$ (d–f) |
| `Supplementary_figure6.png` | `Supplementary_figure6.py` | Lung: AT2→ADI entry; ADI trapping vs weak AT1-directed geometry (AT1 rare); reject Fibro as an ADI fate; $U_{rel}$-motivated KO |
| `Supplementary_figure7.png` | `Supplementary_figure7.py` | Macrophage 3D arrow audit: reject a 7-subtype complete graph; M2–Resolution is the strong local pair; AM(PBS)→Resolution is a weak-mixing prior — not lineage, not the protocol coupling table |
| `Supplementary_figure8.png` | `Supplementary_figure8.py` | SOD2 eviction, random-gene null, *IFI27*, PDVS7 screen |
| `Supplementary_table5.csv` | `Supplementary_table5.py` | Perturbation scorecard (PARTIAL/FAIL) |
| `Supplementary_table6.csv` | `Supplementary_table6.py` | Deep-valley EOC DEG (top 500) |
| `Supplementary_table7.xlsx` | `Supplementary_table7.py` | CCC LR ranks + permutation/patient audits |
| `Supplementary_table8.xlsx` | `Supplementary_table8.py` | PDVS clinical summaries + cohort scores |
| `Supplementary_table9_*.csv` | `Supplementary_figure5.py` | Injury DEG–cell-type Pearson matrix |
| `Supplementary_table10_*.csv` | `Supplementary_figure5.py` | Neuron4 potential summary |

```bash
python output_file/Supplementary_figure4.py
python output_file/Supplementary_figure5.py
python output_file/Supplementary_figure6.py
python scripts/analyze_mac_fn1_m2_resolution_triad.py
python scripts/analyze_mac_landscape_path_endorsement.py
python output_file/Supplementary_figure7.py
python output_file/Supplementary_figure8.py
python output_file/Supplementary_table5.py
python output_file/Supplementary_table6.py
python output_file/Supplementary_table7.py
python output_file/Supplementary_table8.py
```
