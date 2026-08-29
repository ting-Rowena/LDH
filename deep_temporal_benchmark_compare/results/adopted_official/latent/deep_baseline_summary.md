# Deep temporal baseline benchmark

Lower is better for every metric. Results use identical held-out transitions, source-cell subsamples, target populations, and scoring spaces within each run.

**Fidelity note:** `prescient_potential_flow` and `mioflow_neural_ode` are controlled core-objective reimplementations, not official package runs. `wot_barycentric` is WOT-inspired and omits WOT's growth-rate model.

  dataset          method  energy_distance_mean  energy_distance_std  mean_marginal_w1_mean  mean_marginal_w1_std  mmd_mean  mmd_std  mean_shift_l2_mean  mean_shift_l2_std  ot_sinkhorn_mean  ot_sinkhorn_std
GSE141259 scvelo_official              3.063418             0.179150               0.261610              0.008281  0.088321 0.005554            6.078285           0.357362         65.658419         5.663754
GSE141259    wot_official              3.174233             0.152636               0.254896              0.012205  0.093221 0.004425            6.190706           0.367621         49.757688         1.228705
GSE141259       our_model              3.507212             0.144193               0.273339              0.011531  0.156626 0.004203            6.173587           0.335192         49.535608         1.259946
GSE155622    wot_official              0.552837             0.086500               0.131240              0.012098  0.017247 0.002968            2.903323           0.223069         10.535740         1.549330
GSE155622 scvelo_official              0.758421             0.186972               0.169048              0.023298  0.024825 0.004869            3.537672           0.538166         40.738226        10.943344
GSE155622       our_model              0.912588             0.074614               0.156518              0.011474  0.064563 0.001231            3.135012           0.222503         11.518039         1.543235
    HGSOC scvelo_official              2.508334             0.425367               0.242123              0.019810  0.077372 0.012784            6.288026           0.548029         32.311589         5.364898
    HGSOC    wot_official              2.672333             0.427312               0.250764              0.019162  0.107153 0.013183            6.272579           0.547322         32.028301         5.357581
    HGSOC       our_model              3.948921             0.414280               0.300009              0.017812  0.243911 0.011452            6.874896           0.502335         36.310879         5.377474
