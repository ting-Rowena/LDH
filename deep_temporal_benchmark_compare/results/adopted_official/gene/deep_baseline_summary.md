# Deep temporal baseline benchmark

Lower is better for every metric. Results use identical held-out transitions, source-cell subsamples, target populations, and scoring spaces within each run.

**Fidelity note:** `prescient_potential_flow` and `mioflow_neural_ode` are controlled core-objective reimplementations, not official package runs. `wot_barycentric` is WOT-inspired and omits WOT's growth-rate model.

  dataset          method  energy_distance_mean  energy_distance_std  mean_marginal_w1_mean  mean_marginal_w1_std  mmd_mean  mmd_std  mean_shift_l2_mean  mean_shift_l2_std  ot_sinkhorn_mean  ot_sinkhorn_std
GSE141259    wot_official              2.224984             0.042264               0.097440              0.001680  0.021334 0.000135            6.667003           0.098165        701.515361         3.982067
GSE141259 scvelo_official              2.856378             0.214567               0.137863              0.007930  0.004833 0.000184            9.125594           0.142442       1696.280243       271.860168
GSE141259       our_model              7.789407             0.517964               0.139814              0.001635  0.235685 0.014897            9.281397           0.374294        551.652995         6.242675
GSE155622    wot_official              3.311331             0.324200               0.212831              0.006332  0.004039 0.000221           12.972940           0.597615       1242.454610        13.794800
GSE155622       our_model              6.523151             0.055195               0.285848              0.000510  0.182373 0.002053            6.568391           0.063135        701.609599         3.996030
GSE155622 scvelo_official             75.425242             2.079790               2.084927              0.062634  0.003808 0.000019           64.407696           1.773626     103765.765381      7029.289330
    HGSOC scvelo_official              2.320436             0.139961               0.130754              0.005938  0.005028 0.000110           10.014469           0.348220        876.841675        14.010002
    HGSOC    wot_official              6.369224             0.192637               0.169017              0.004881  0.050837 0.001099           11.769312           0.360728        660.154948         7.692692
    HGSOC       our_model             27.674786             0.272802               0.569376              0.001236  0.505257 0.008399           24.879147           0.090843        912.364746         5.623991
