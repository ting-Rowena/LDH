# Appendix Table S2 (transport hold-out) — table legend

File: `deep_temporal_benchmark_compare/Supplementary_table2.csv`  
(This is **not** the SOTA PCC table written by `output_file/Supplementary_table2.py`.)

---

## Legend (English, manuscript paste)

**Appendix Table SX. Hold-out population transport of LDH-scRNA versus Waddington-OT, PRESCIENT-family and MIOFlow-family in PCA-50.**

Dataset-averaged centroid Euclidean distance (lower better), Mean Displacement Cosine (higher better) and Velocity Projection Alignment (higher better) for predicted versus observed hold-out clouds. All four methods use the same source subsample (≤512 training-mask cells at \(t_{\mathrm{curr}}\)), the same validation-mask target at \(t_{\mathrm{next}}\), and the same PCA map fit on training cells only (`n_components=50`). Predictions are decoded or inverse-transformed to gene space, then projected. Pain (GSE155622): train \(0\)–\(7\,\mathrm{d}\), hold \(14\,\mathrm{d}\); mean of hops \((0\to 14)\) and \((7\to 14)\). Lung (GSE141259): train D0–D21, hold D28; mean of \((0\to 28)\) and \((21\to 28)\). HGSOC: `val_mode=patients`; source = train patients at treatment-naive (\(t=0\)); target = hold-out patients EOC136 and EOC153 at post-NACT (\(t=1\)); single hop \((0\to 1)\). Values are arithmetic means over seeds \(0,1,2\) (Pain/Lung: 6 numbers per method; HGSOC: 3). Cosines may be negative. LDH-scRNA is damped-Hamiltonian integration from the Hamiltonian4 checkpoints. Waddington-OT is official `wot.ot` barycentric push; hold-out times beyond the last training day use linear time-scaling of the last in-train map (no growth-rate model). PRESCIENT-family and MIOFlow-family are in-harness first-order potential flow and time-conditioned neural ODE trained with MMD on reduced PCA, **not** the published packages. These columns score location and direction, not cloud shape (Energy/MMD/Sinkhorn omitted). scVelo is not in this CSV.

---

## 中文表注

**附表 SX. PCA-50 上 LDH-scRNA 与 Waddington-OT、PRESCIENT-family、MIOFlow-family 的 hold-out 群体运输比较。**

同一套 source（训练掩膜、\(t_{\mathrm{curr}}\)、最多 512 细胞）、同一套 hold-out target、同一套仅在训练集上拟合的 50 维 PCA。三列依次为：预测云与真实云的质心欧氏距离（越小越好）、群体位移余弦 MDC（越大越好）、细胞水平与 hold-out OT 重心耦合的速度投影对齐 VPA（越大越好）。余弦可为负。Pain：训练 0–7 天、验证 14 天，对 \((0\to 14)\)、\((7\to 14)\) 平均。Lung：训练 D0–D21、验证 D28，对 \((0\to 28)\)、\((21\to 28)\) 平均。HGSOC：按病人 hold-out；source 为训练病人治疗前 \(t=0\)，target 为 EOC136、EOC153 治疗后 \(t=1\)，仅 \((0\to 1)\)。种子 0、1、2 再取算术平均。LDH-scRNA 为 Hamiltonian4 checkpoint 的阻尼哈密顿积分。Waddington-OT 为官方运输映射；超出最后训练日后线性外推位移，未用生长率模型。PRESCIENT-family / MIOFlow-family 为本仓库内 MMD 训练的势流与神经 ODE，**不是**官方软件。本表只比位置与方向，不含分布重叠或 scVelo。
