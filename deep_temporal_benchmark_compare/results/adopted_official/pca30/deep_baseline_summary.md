# Deep temporal baseline benchmark

Lower is better for every metric. Results use identical held-out transitions, source-cell subsamples, target populations, and scoring spaces within each run.

**Fidelity note:** `prescient_potential_flow` and `mioflow_neural_ode` are controlled core-objective reimplementations, not official package runs. `wot_barycentric` is WOT-inspired and omits WOT's growth-rate model.

  dataset          method  energy_distance_mean  energy_distance_std  mean_marginal_w1_mean  mean_marginal_w1_std  mmd_mean  mmd_std  mean_shift_l2_mean  mean_shift_l2_std  ot_sinkhorn_mean  ot_sinkhorn_std
GSE141259 scvelo_official              1.937220             0.172806               0.907298              0.014420  0.019371 0.001482            6.204965           0.336454        133.659861         8.567650
GSE141259    wot_official              2.163479             0.030808               0.925413              0.018014  0.040041 0.000971            5.765125           0.061454         86.650065         0.171024
GSE141259       our_model              6.256650             0.428478               1.734795              0.035014  0.223749 0.014143            8.492687           0.401543        162.036766         2.998999
GSE155622       our_model              3.149609             0.031790               1.432296              0.010174  0.150377 0.000821            5.453071           0.092279        121.707230         0.941713
GSE155622    wot_official              6.410714             0.523140               1.430180              0.044730  0.064957 0.002597           11.868797           0.619886        143.002182        13.834618
GSE155622 scvelo_official             34.885081             2.939014               8.314687              0.340754  0.060418 0.003601           39.055199           2.037843       4462.587690       405.980266
    HGSOC scvelo_official              2.662003             0.172763               1.067267              0.045675  0.035142 0.001416            7.100011           0.306133        128.020121         6.112774
    HGSOC    wot_official              4.700722             0.186701               1.204227              0.025987  0.065427 0.001657            9.336681           0.316011        134.295135         3.380642
    HGSOC       our_model             12.628490             0.326828               2.633826              0.035135  0.502504 0.009611            9.968771           0.217550        211.479899         4.086872
