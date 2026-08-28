# Appendix Table S9–S10 legends (GSE155622; from Supplementary Figure 5)

Files:
- `output_file/Supplementary_table9_GSE155622_injury_DEG_celltype_corr.csv`
- `output_file/Supplementary_table10_GSE155622_neuron4_potential.csv`

Generator: `output_file/Supplementary_figure5.py` (adopted checkpoint  
`GSE155622_checkpoints_3000_3000_384_0.05_recon0.01_lossnorm_qp_d0p01_z0p5_k0p2_ld1`).

---

## Appendix Table S9. Tissue-wide late-injury DEGs versus nine DRG cell types (GSE155622).

**Appendix Table S9. Pearson correlation of late-injury DEGs with nine major DRG cell-type indicators (GSE155622).**

Rows are the top 60 genes that differ between late SNI (7 d and 14 d pooled) and Control on the 3000-gene training panel, tested tissue-wide across the nine major types (Fibroblast, Immune, Neuron, RBC, Satellite, Schwann, VEC, VECC, VSMC). Test: two-sided Mann–Whitney *U* on `log1p` expression from the cohort h5ad (not `normalize_total`). A gene is called significant if BH-adjusted *p* < 0.05 and |mean_late − mean_control| ≥ 0.10 on the log1p scale, then ranked by adjusted *p*. Columns Fibroblast…VSMC are Pearson *r* between that gene’s log1p expression and a binary indicator for the named type, computed on **all** checkpoint cells (not only Control/late). `log1p_diff` = mean_late − mean_control (positive = higher in late SNI). `pval_adj` is BH-FDR from the Wilcoxon screen. `mean_late` / `mean_control` are tissue-wide log1p means in those two groups. `best_celltype` is the type with largest |*r*| (here 55/60 genes Neuron; remainder Schwann, VEC or Satellite). These *r* values are compositional: a gene enriched in neurons will correlate positively with the Neuron indicator and negatively with other types even if it is not neuron-restricted in a biological sense. This table is not RNA velocity and not a neuron-only DEG list.

**中文表注：** 训练 3000 基因面板上，全组织（九类）晚期 SNI（7 d+14 d）对 Control 的 Wilcoxon DEG，取显著基因前 60 个（BH *p*<0.05 且 |log1p 均值差|≥0.10）。表中九列为该基因 log1p 表达与细胞类型 0/1 指示的 Pearson *r*（全体细胞）。`log1p_diff` 为晚期减对照。`best_celltype` 为 |*r*| 最大的类型（60 个里 55 个是 Neuron）。*r* 有组成性偏倚，不能单独当成细胞类型特异表达。

---

## Appendix Table S10. Quasi-stationary potential \(U_0\) for four neuron subtypes (GSE155622).

**Appendix Table S10. Stationary potential \(U_0\) summary for four DRG neuron subtypes (GSE155622).**

Neuron cells from the adopted checkpoint whose `celltype_2` maps to one of four subtypes: Myelinated, Non_peptidergic, Peptidergic, SNI-induced (`GSE155622_NEURON_SUBTYPE`; SNI-induced pools Atf3/Gfra3/Gal, Atf3/Mrgprd and Atf3/S100b/Gal). \(U_0\) is `potential_stationary` from LDH-scRNA (time-invariant quasi-potential), not \(U(z,t)\) and not \(U_{\mathrm{rel}}\). One row per subtype, sorted by increasing mean \(U_0\). `n_cells`, `mean`, `median`, sample `std` (ddof=1), 25th/75th percentiles. `rank_deepest` = 1 for the lowest mean \(U_0\) (SNI-induced). This CSV does **not** include \(U_{\mathrm{rel}}\) or the per-condition time course (those sit in `output_file/robustness/gse155622_supp_injury_neuron4/`). Lower \(U_0\) is the model’s deeper basin on this landscape, not a proof of lineage or of a dynamical steady state at a given SNI day.

**中文表注：** 四种神经元亚型（有髓、非肽能、肽能、SNI 诱导；后者合并三条 Atf3 伤后模块）上准稳态势 \(U_0\)（`potential_stationary`）的均值、中位数、标准差与四分位。按平均 \(U_0\) 从低到高排序，`rank_deepest=1` 为最深（SNI-induced）。本 CSV 不含 \(U_{\mathrm{rel}}\) 和分时间点。低 \(U_0\) 是景观上的更深盆地，不是某一天的实验稳态，也不是谱系证明。
