# Appendix Table S1–S8 legends (Molecular Systems Biology)

Source files: `output_file/Supplementary_table{1–8}.*`.  
Numbering follows the `output_file` package (not the four-method PCA transport table in `deep_temporal_benchmark_compare/`).

---

## Appendix Table S1. Cohort inventory for the three adopted LDH-scRNA datasets.

**File:** `Supplementary_table1.csv` (expanded workbook: `Supplementary_table1.py` → `.xlsx`).

Cell counts, gene-panel size, latent dimension, temporal axis, biological units, cell-type key, pairing design, and validation mode for GSE155622 (mouse DRG, spared-nerve injury), GSE141259 (mouse whole lung, bleomycin), and HGSOC (human NACT-paired high-grade serous ovarian carcinoma). Counts are taken from the adopted checkpoint `obs.csv`, not from the raw GEO/EGA objects before processing. Time labels are experimental stages (injury day or treatment-naive vs post-NACT), not dynamical steady states. GSE155622 and GSE141259 are unpaired cross-sectional snapshots; HGSOC pairs 11 patients across two treatment phases. Validation recorded in this table is the checkpoint protocol (GSE155622: random cell hold-out; GSE141259: D28 time extrapolation; HGSOC: patient hold-out). `val_PCC_approx` is a stored training diagnostic, not the trajectory–time PCC in Appendix Table S2.

**中文表注：** 三套采用 checkpoint 的队列一览：细胞数、训练基因面板、潜空间维数、时间轴、生物学重复（小鼠/病人）、细胞类型字段、配对设计与验证方式。计数来自处理后的 `obs.csv`。时间标签是实验取样阶段，不是稳态。痛觉与肺为非配对截面；HGSOC 为 11 名患者治疗前/后配对。验证方案：GSE155622 随机细胞 hold-out，GSE141259 外推 D28，HGSOC 按病人 hold-out。

---

## Appendix Table S2. Trajectory–time Pearson correlation of LDH-scRNA, scVelo and CellRank.

**File:** `Supplementary_table2.csv`.

Pearson correlation between reconstructed temporal order and experimental time on a shared 10-PC latent (`X_latent_pca`). LDH-scRNA order is absorbing-Markov hitting time from the MomentumNetwork field (same fate-order protocol as CellRank; not the supervised pseudotime head). scVelo uses a k-nearest-neighbour, time-ordered velocity proxy and a graph-Laplacian order. CellRank uses the same kNN velocity with absorbing-Markov hitting time. Higher is better. GSE141259 is scored under the D28 hold-out setting of that cohort; HGSOC uses the NACT-paired two-phase axis (\(t=0\to 1\)). These values are not the centroid Euclidean / MDC / VPA transport scores of the four-method hold-out benchmark.

**中文表注：** 同一 10 维 PCA 潜空间上，三种方法重建的时间顺序与实验时间的 Pearson 相关。LDH-scRNA 与 CellRank 均为吸收 Markov 击中时间；scVelo 为图 Laplacian 顺序。LDH 用 MomentumNetwork 场，不用监督伪时间头。数值越大越好。此表不是四方法运输基准（质心距离、MDC、VPA）。

---

## Appendix Table S3. Dual-metric matched temporal null audit.

**File:** `Supplementary_table3.csv`.

For each cohort, the adopted model (“Real Model”) is compared with the median of four from-scratch retrains on time-shuffled labels (“Null Model (Retrained)”; 500 epochs, 5,000 cells, batch 128), scored on the unshuffled subset. Shuffle: GSE155622 and GSE141259 jointly permute temporal metadata (`temporal_matched`); HGSOC shuffles `treatment_phase` within patient (`pairing_matched`). Metrics: (i) Spearman correlation of quasi-stationary potential \(U_0\) with a kNN KDE (geometric consistency); (ii) hold-out forecasting PCC. Collapse ratio = null median / real. A ratio near 1 means the metric does not require true time labels; a ratio \(\ll 1\) (or a sign flip of Spearman) means the real-time fit is required. HGSOC \(U_0\)–KDE Spearman barely collapses (0.986), so that geometric score is not a time-sensitive control in this cohort; hold-out PCC does collapse (0.693).

**中文表注：** 真实时间模型对比时间标签打乱后从头重训的空模型（4 次重复中位数）。痛觉/肺：打乱时间元数据；HGSOC：在病人内打乱治疗阶段。两列指标为 \(U_0\)–KDE Spearman 与 hold-out PCC。坍缩比 = 空模型/真实模型。接近 1 表示该指标不依赖真实时间；远小于 1 表示依赖。HGSOC 的 \(U_0\)–KDE 几乎不坍缩，hold-out PCC 会坍缩。

---

## Appendix Table S4. Landscape metrics for injury-state DRG neurons (GSE155622).

**File:** `Supplementary_table4.csv`.

Neuron cells only (\(n \approx 6466\)). Relative potential \(U_{\mathrm{rel}}\) is `potential_relative_type`. Expression is `normalize_total(10^4)` then `log1p`. SNIIC modules are the mean of the listed genes (SNIIC1: *Atf3*/*Gfra3*/*Gal*; SNIIC2: *Atf3*/*Mrgprd*; SNIIC3: *Atf3*/*S100b*/*Gal*). Nav genes are single-gene log-expression (*Scn9a*, *Scn10a*, *Scn11a*). Spearman \(\rho\) is versus \(U_{\mathrm{rel}}\) (two-sided Spearman \(p\)). Deep-valley score is the mean feature value in the lowest \(U_{\mathrm{rel}}\) quartile. Slope score is the OLS slope of feature versus \(U_{\mathrm{rel}}\). Negative \(\rho\) for SNIIC modules means higher injury-module score in the low-\(U_{\mathrm{rel}}\) basin; positive \(\rho\) for Nav genes means the opposite tilt.

**中文表注：** 仅神经元。\(U_{\mathrm{rel}}\) 为类型内相对势。模块为所列基因均值。Spearman 针对 \(U_{\mathrm{rel}}\)；深谷分为最低四分位的模块/基因表达均值；斜率为对 \(U_{\mathrm{rel}}\) 的 OLS。SNIIC 为负相关（伤后模块在低势盆地更高），Nav 为正相关。

---

## Appendix Table S5. In silico perturbation scorecard.

**File:** `Supplementary_table5.csv`.

Qualitative verdicts from hybrid knockout / valley-eviction on the adopted checkpoints. PASS / PARTIAL / FAIL are protocol calls, not genome-wide screens. GSE155622: hybrid *Atf3* KO (SNIIC drift block, partner-selective, not neuron-restricted) versus *Egr1* (negative control) and out-of-panel *Cpeb1*. GSE141259: KO of high-\(U_{\mathrm{rel}}\) genes (*Lgals3*/*Cdkn1a*/*Spp1*) versus low-\(U_{\mathrm{rel}}\) controls (*Cbr2*/*Hc*/*Chi3l1*); readouts can move without proving ADI→AT1 or ruling out a fibroblast foil. HGSOC: hybrid eviction of the EOC deep valley for *BBC3* (PASS; 99% escape, above a 20-gene random-gene null), *SOD2* (PARTIAL; escape without random-gene specificity), and *IFI27* (FAIL; 0% escape). Escape fractions use the published / q15 valley cutoff of that eviction table.

**中文表注：** 采用 checkpoint 上的 hybrid 敲除/驱离判定，不是全基因组筛选。痛觉：Atf3 为 PARTIAL，Egr1/Cpeb1 为 FAIL。肺：高 \(U_{\mathrm{rel}}\) 基因相对低势对照为 PARTIAL，不能当作 ADI→AT1 的谱系证明。HGSOC：BBC3 PASS，SOD2 PARTIAL，IFI27 FAIL。

---

## Appendix Table S6. Differentially expressed genes in deep-valley EOC versus other EOC (HGSOC).

**File:** `Supplementary_table6.csv` (top 500 genes by |Wilcoxon score|).

Two-sided Wilcoxon rank-sum test on the training HVG panel, deep-valley EOC (\(n=1124\)) versus remaining EOC (\(n_{\mathrm{EOC}}=8806\)). Deep valley is the low-\(U_0\) attractor subset defined in the HGSOC analysis protocol, not a cluster from graph clustering. Columns: gene symbol; Wilcoxon `score`; `logfoldchange` (deep valley minus other EOC); raw `pval`; BH-adjusted `pval_adj`; comparison tag `deep_valley_vs_other_EOC`; `direction` (`up_in_group_1` = higher in the deep valley). Positive scores (e.g. BBC3, SOD2, WFDC2, FTL, CEBPD) mark valley-enriched genes used to nominate PDVS; they are not themselves the survival signature until restricted and z-scored as in Appendix Table S8.

**中文表注：** HGSOC 上皮癌细胞中，深谷（低 \(U_0\) 吸引子，1124 细胞）对其余 EOC 的 Wilcoxon DEG，按 |score| 取前 500。深谷由势函数协议定义，不是图聚类。`up_in_group_1` 表示深谷更高。BBC3 等为谷内上调，临床 PDVS 计分见 Table S8。

---

## Appendix Table S7. Targeted ligand–receptor scores between deep-valley EOC and stromal cells (HGSOC).

**File:** `Supplementary_table7.xlsx`.

Sheets from the adopted HGSOC CCC protocol: `Stromal_to_EOC`, `EOC_to_Stromal`, `paracrine_feedforward`, plus robustness sheets when present (`observed_focus_pairs`, `patient_summary`, `band_permutation_null`). Sender/receiver means are ligand (or receptor) expression in the indicated compartment; `lr_score` is the product of those means (not CellPhoneDB permutation \(p\)-values unless a null sheet is used). Bands `highU` / `lowU` split stromal cells by relative potential. This is a curated pair list between deep-valley EOC and high-\(U\) stroma, not an unbiased interactome. Patient-level and band-permutation sheets test whether highlighted pairs exceed a compartment-shuffle null; they do not prove physical ligand–receptor binding.

**中文表注：** 深谷 EOC 与基质之间的指定配体–受体打分（表达均值之积），可含高/低 \(U\) 基质分带及病人/置换检验表。不是全基因组细胞通讯组，也不能单独证明结合。

---

## Appendix Table S8. Patient-level PDVS overall-survival analyses (TCGA-OV and GSE26712).

**File:** `Supplementary_table8.xlsx`.

Sheets: `summary`, `TCGA_patients`, `AOCS_patients`, `Cox_TCGA`, `Cox_AOCS`, and score tables when present. PDVS is the mean of per-gene z-scored expression of BBC3, SOD2, WFDC2, FTL and CEBPD (PDVS5: the PDVS panel genes that lie on the HGSOC training HVG list; ZC3H12A and IRF1 are excluded). Bulk matrices are TCGA-OV STAR TPM and GSE26712 arrays; HVG restriction selects genes, it does not subset bulk profiles to the scRNA HVG panel. Cox models use a second, patient-level z-score of that mean (`PDVS_z`); Kaplan–Meier groups split at the median of raw PDVS. Primary endpoint is overall survival only. Associations are exploratory; the analysis pipeline does not claim an independent prognostic biomarker.

**中文表注：** 患者水平 PDVS 与总生存。PDVS = BBC3、SOD2、WFDC2、FTL、CEBPD 各基因在队列内 z-score 后的均值（HGSOC 训练 HVG 内的五基因；不含 ZC3H12A、IRF1）。Cox 再用患者水平 `PDVS_z`；KM 按 PDVS 中位数分组。仅 OS，探索性相关，不作独立预后结论。
