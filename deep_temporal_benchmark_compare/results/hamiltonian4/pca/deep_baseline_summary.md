# Deep temporal baseline benchmark

Lower is better for every metric. Results use identical held-out transitions, source-cell subsamples, target populations, and scoring spaces within each run.

**Fidelity note:** `prescient_potential_flow` and `mioflow_neural_ode` are controlled core-objective reimplementations, not official package runs. `wot_barycentric` is WOT-inspired and omits WOT's growth-rate model.

  dataset                   method  energy_distance_mean  energy_distance_std  mean_marginal_w1_mean  mean_marginal_w1_std  mmd_mean  mmd_std  mean_shift_l2_mean  mean_shift_l2_std  ot_sinkhorn_mean  ot_sinkhorn_std
GSE141259 prescient_potential_flow              2.700796             0.128158               0.589930              0.008039  0.020373 0.000929            7.526070           0.273648        160.315773         1.770650
GSE141259       mioflow_neural_ode              2.780858             0.085494               0.603005              0.012300  0.020756 0.000560            7.631581           0.231493        162.241821         1.646979
GSE141259          wot_barycentric              3.324464             0.146002               0.744663              0.004254  0.029388 0.000949            8.370746           0.258566        171.585726         1.969542
GSE141259                our_model              5.789847             0.201553               1.537901              0.018722  0.291201 0.014218            4.840943           0.093194        129.107722         2.124719
GSE155622 prescient_potential_flow              2.378339             0.060197               0.616643              0.006709  0.022140 0.000480            7.009872           0.131252         98.483874         2.351572
GSE155622       mioflow_neural_ode              2.477562             0.156561               0.659834              0.005808  0.022662 0.000786            7.210892           0.300729        101.606709         3.467230
GSE155622                our_model              3.265429             0.037202               1.252966              0.008216  0.158853 0.002306            4.239869           0.046882        106.784452         1.485485
GSE155622          wot_barycentric              5.033124             0.123245               1.000754              0.012454  0.045735 0.001064           10.873886           0.140552        183.151356         1.488180
    HGSOC          wot_barycentric              2.911660             0.196244               0.905995              0.044780  0.031038 0.001094            7.959235           0.363048        186.838343         6.792318
    HGSOC prescient_potential_flow              2.913368             0.196135               0.906274              0.045753  0.031040 0.001092            7.963297           0.363154        186.861694         6.824737
    HGSOC       mioflow_neural_ode              2.980873             0.222563               0.919694              0.057217  0.031393 0.001322            8.085612           0.402516        189.186834         7.119317
    HGSOC                our_model              9.542153             0.361549               1.817884              0.024100  0.456065 0.010075            7.554290           0.306876        193.427719         4.971083
