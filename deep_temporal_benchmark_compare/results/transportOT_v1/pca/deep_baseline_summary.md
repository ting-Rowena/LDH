# Deep temporal baseline benchmark

Lower is better for every metric. Results use identical held-out transitions, source-cell subsamples, target populations, and scoring spaces within each run.

**Fidelity note:** `prescient_potential_flow` and `mioflow_neural_ode` are controlled core-objective reimplementations, not official package runs. `wot_barycentric` is WOT-inspired and omits WOT's growth-rate model.

  dataset                   method  energy_distance_mean  energy_distance_std  mean_marginal_w1_mean  mean_marginal_w1_std  mmd_mean  mmd_std  mean_shift_l2_mean  mean_shift_l2_std  ot_sinkhorn_mean  ot_sinkhorn_std
GSE141259 prescient_potential_flow              2.700795             0.128159               0.589930              0.008039  0.020373 0.000929            7.526068           0.273649        160.315754         1.770637
GSE141259       mioflow_neural_ode              2.780857             0.085495               0.603004              0.012300  0.020756 0.000560            7.631580           0.231496        162.241811         1.646955
GSE141259          wot_barycentric              3.324463             0.146002               0.744663              0.004254  0.029388 0.000949            8.370745           0.258566        171.585719         1.969537
GSE141259                our_model             23.513532             0.192817               2.425291              0.007695  0.757216 0.002075           17.470757           0.164467        311.304784         0.456507
GSE155622 prescient_potential_flow              2.378338             0.060199               0.616643              0.006709  0.022140 0.000480            7.009869           0.131252         98.483829         2.351582
GSE155622       mioflow_neural_ode              2.477579             0.156556               0.659835              0.005806  0.022662 0.000786            7.210910           0.300706        101.606767         3.467468
GSE155622                our_model              4.681046             0.061693               1.302379              0.010881  0.169921 0.001036            7.488471           0.145416        138.532669         0.816043
GSE155622          wot_barycentric              5.033124             0.123246               1.000753              0.012454  0.045735 0.001064           10.873885           0.140553        183.151333         1.488192
    HGSOC          wot_barycentric              2.911659             0.196244               0.905994              0.044780  0.031038 0.001094            7.959236           0.363047        186.838318         6.792330
    HGSOC prescient_potential_flow              2.913368             0.196136               0.906273              0.045753  0.031040 0.001092            7.963298           0.363154        186.861684         6.824744
    HGSOC       mioflow_neural_ode              2.980874             0.222564               0.919694              0.057218  0.031393 0.001322            8.085615           0.402517        189.186834         7.119311
    HGSOC                our_model             23.008201             0.270504               2.678311              0.022912  0.730139 0.006795           15.584685           0.205294        385.458272         2.673214
