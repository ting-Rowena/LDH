# Deep temporal baseline benchmark

Lower is better for every metric. Results use identical held-out transitions, source-cell subsamples, target populations, and scoring spaces within each run.

**Fidelity note:** `prescient_potential_flow` and `mioflow_neural_ode` are controlled core-objective reimplementations, not official package runs. `wot_barycentric` is WOT-inspired and omits WOT's growth-rate model.

  dataset          method  energy_distance_mean  energy_distance_std  mean_marginal_w1_mean  mean_marginal_w1_std  mmd_mean  mmd_std  mean_shift_l2_mean  mean_shift_l2_std  ot_sinkhorn_mean  ot_sinkhorn_std
GSE141259 scvelo_official              2.046596             0.174558               0.807545              0.026098  0.015778 0.001182            6.752517           0.370428        222.953873        16.894435
GSE141259    wot_official              2.222262             0.034913               0.756226              0.013869  0.038322 0.000584            5.915500           0.075155        130.373657         0.318751
GSE141259       our_model              6.187885             0.437631               1.471683              0.027584  0.220038 0.014336            8.696697           0.380466        196.483142         3.137168
GSE155622       our_model              3.082885             0.018340               1.170328              0.009512  0.148608 0.001155            5.499532           0.085223        138.976690         1.001903
GSE155622    wot_official              6.045821             0.497318               0.995024              0.031824  0.049650 0.001828           12.040711           0.612571        172.025192        13.824862
GSE155622 scvelo_official             41.243301             2.655472               7.796091              0.303860  0.050435 0.002778           45.185234           1.811086       7605.420436       653.006574
    HGSOC scvelo_official              2.628708             0.178925               0.799680              0.036542  0.029906 0.001206            7.351484           0.321985        167.489110         6.315692
    HGSOC    wot_official              4.878661             0.206737               0.985907              0.029294  0.064314 0.001400            9.540930           0.325521        162.960754         3.467260
    HGSOC       our_model             14.980176             0.322204               2.247675              0.028636  0.515947 0.009378           12.490478           0.200497        263.202942         3.873298
