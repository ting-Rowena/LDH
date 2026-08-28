#!/usr/bin/env python
"""Unified SDE model training entry point for all datasets."""

from __future__ import annotations

import argparse
import sys
import warnings

import pandas as pd
import scanpy as sc

from plot_utils import configure_headless, run_standard_training_figures, setup_scanpy_figdir

configure_headless()

from train_model import (
    INERTIA_MOMENTUM_MIX,
    KineticAnalyzer,
    SDETrainer,
    TemporalDataProcessor,
    TemporalSDENetwork,
    gene_specific_mse,
    plot_metrics,
    predict,
)
from dataset_pipeline import (
    GSE141259,
    GSE141259_PROFILES,
    GSE155622,
    GSE225948_BRAIN,
    GSE225948_BRAIN_PROFILES,
    HGSOC,
    apply_hamiltonian_loss_options,
    apply_train_config,
    build_training_checkpoint_dir,
    format_train_config_summary,
    get_dataset_interpretation_metadata,
    prepare_gse141259_for_training,
    prepare_gse155622_for_training,
    prepare_gse225948_for_training,
    prepare_hgsoc_for_training,
    resolve_data_path,
    save_training_summary,
)

DATASETS = {
    "HGSOC": (HGSOC, prepare_hgsoc_for_training),
    "GSE155622": (GSE155622, prepare_gse155622_for_training),
    "GSE225948_Brain": (GSE225948_BRAIN, prepare_gse225948_for_training),
    "GSE141259": (GSE141259, prepare_gse141259_for_training),
}

BRAIN_INTERPRETATION_WARNING = (
    "WARNING: Brain dataset has only three time points. Potential landscape and "
    "transition-region claims should be treated as exploratory."
)


def _apply_ablation_flags(config, args) -> None:
    config.ablation_flags = {
        "no_hjb": bool(args.no_hjb),
        "no_residual_drift": bool(args.no_residual_drift),
        "no_cell_type_embedding": bool(args.no_cell_type_embedding),
        "adjacent_only": bool(args.adjacent_only),
        "no_state_momentum": bool(args.no_state_momentum),
    }
    if args.no_hjb:
        config.lambda_hjb = 0.0
        print("Ablation: Hamilton-Jacobi-inspired regularization disabled", flush=True)
    if args.no_residual_drift:
        config.use_residual_drift = False
        config.residual_drift_mode = "none"
        print("Ablation: residual drift disabled, using potential-only drift", flush=True)
    if args.no_cell_type_embedding:
        config.use_cell_type_embedding = False
        print("Ablation: cell type embedding disabled", flush=True)
    if args.no_state_momentum:
        config.use_state_momentum = False
        print("Ablation: state-dependent momentum disabled, using global EMA momentum", flush=True)
    if args.adjacent_only:
        config.use_adjacent_transitions = True
        config.use_anchor_transitions = False
        print("Ablation: adjacent transitions only, anchor transitions disabled", flush=True)


def _print_interpretation_notice(dataset_name: str) -> None:
    meta = get_dataset_interpretation_metadata(dataset_name)
    print(
        f"Interpretation | level={meta['interpretation_level']} | "
        f"potential_claim_strength={meta['potential_claim_strength']}",
        flush=True,
    )
    if dataset_name == "GSE225948_Brain":
        print(BRAIN_INTERPRETATION_WARNING, flush=True)
        warnings.warn(BRAIN_INTERPRETATION_WARNING, UserWarning, stacklevel=2)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train temporal SDE model for a dataset")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS.keys()))
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override DatasetSpec.epochs (e.g. 50 for smoke tests)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Short run: skip final landscape evaluation and post-train analysis",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Hyperparameter profile: GSE141259 l3_genes5000 (paper lung); "
        "GSE225948_Brain B0|B1|B2 (not used in the paper)",
    )
    parser.add_argument(
        "--loss-recipe",
        choices=["legacy", "hamiltonian"],
        default=None,
        help="legacy: paper landscape objective (default from apply_train_config). "
        "hamiltonian: four-term transport recipe for Hamiltonian4 / Figure 2 OT "
        "(see apply_hamiltonian_loss_options).",
    )
    parser.add_argument("--hidden-dim", type=int, default=None, help="Latent width (default: 512)")
    parser.add_argument("--n-layers", type=int, default=None, help="Encoder/decoder depth (default: 3)")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate (default: 0.15)")
    parser.add_argument(
        "--checkpoint-metric",
        choices=["pcc", "mse", "loss", "pcc_then_mse"],
        default=None,
        help="Criterion for saving best_model.pth (default: DatasetSpec.checkpoint_metric)",
    )
    parser.add_argument(
        "--early-stop-metric",
        choices=["pcc", "mse", "loss", "pcc_then_mse"],
        default=None,
        help="Criterion for early-stopping patience (default: DatasetSpec.early_stop_metric)",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Epochs without improvement before early stop (default: DatasetSpec / Config)",
    )
    parser.add_argument(
        "--val-mode",
        choices=["cells", "time", "cell_type", "time_extrapolate"],
        default=None,
        help="Held-out validation split (default: DatasetSpec.val_mode)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Validation hold-out ratio (time_extrapolate: fraction of late timepoints held out)",
    )
    parser.add_argument(
        "--val-time-point",
        type=float,
        default=None,
        help="For val_mode=time: hold out this time value (default: last timepoint)",
    )
    parser.add_argument(
        "--no-hjb",
        action="store_true",
        help="Ablation: set lambda_hjb=0",
    )
    parser.add_argument(
        "--no-residual-drift",
        action="store_true",
        help="Ablation: disable residual drift network (potential-only drift)",
    )
    parser.add_argument(
        "--no-cell-type-embedding",
        action="store_true",
        help="Ablation: disable cell-type embedding in the encoder",
    )
    parser.add_argument(
        "--adjacent-only",
        action="store_true",
        help="Ablation: train adjacent time transitions only (no anchor pairs)",
    )
    parser.add_argument(
        "--loss-normalization",
        action="store_true",
        help="Enable EMA normalization for OT/recon/energy/momentum/density/delta/direction loss terms",
    )
    parser.add_argument(
        "--no-loss-normalization",
        action="store_true",
        help="Disable EMA loss normalization even when enabled by the dataset defaults",
    )
    parser.add_argument(
        "--total-drift-hjb",
        action="store_true",
        help="Use total-drift-aware HJ-inspired regularizer instead of gradient-only HJ regularizer",
    )
    parser.add_argument(
        "--residual-drift-mode",
        choices=["velocity", "force", "none"],
        default=None,
        help="How residual drift enters Hamiltonian dynamics",
    )
    parser.add_argument(
        "--momentum-loss-type",
        choices=["velocity", "force", "both"],
        default=None,
        help="Momentum regularization formulation",
    )
    parser.add_argument(
        "--no-state-momentum",
        action="store_true",
        help="Ablation: use the legacy global EMA momentum instead of the per-cell "
        "state-dependent momentum network p_theta(z, t)",
    )
    parser.add_argument(
        "--potential-time-mode",
        choices=["time_varying", "quasi_stationary"],
        default=None,
        help="Potential parameterization: quasi_stationary reports the time-invariant "
        "quasi-potential U0(z) as the landscape plus a small time-varying correction",
    )
    parser.add_argument(
        "--potential-time-correction-scale",
        type=float,
        default=None,
        help="eps weight of the time-varying correction phi(z,t) in quasi_stationary mode",
    )
    parser.add_argument(
        "--reconstruction-mode",
        choices=["mse", "mse_mmd", "mmd_only"],
        default=None,
    )
    parser.add_argument(
        "--lambda-recon",
        type=float,
        default=None,
        help="Weight for reconstruction loss (mse_mmd: L_recon = lambda_recon * (L_mse + 5*L_mmd))",
    )
    parser.add_argument(
        "--lambda-energy",
        type=float,
        default=None,
        help="Override lambda_energy (alias: Hamilton-Jacobi / energy regularizer weight)",
    )
    parser.add_argument("--lambda-hjb", type=float, default=None, help="Alias for --lambda-energy")
    parser.add_argument(
        "--lambda-latent",
        type=float,
        default=None,
        help="Weight of the latent-consistency loss: match integrate(encode(x_t))@t+1 to "
        "encode(x_{t+1}) (target detached). Prevents the latent flow from diverging.",
    )
    parser.add_argument(
        "--lambda-kinetic",
        type=float,
        default=None,
        help="Weight of combined inertia loss: INERTIA_MOMENTUM_MIX * L_momentum + L_kinetic.",
    )
    parser.add_argument(
        "--lambda-lat-disp",
        type=float,
        default=None,
        help="Weight of OT-coupled latent displacement loss (per cell type).",
    )
    # Deprecated aliases (mapped internally; emit warnings when used).
    parser.add_argument("--lambda-pair-mse", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--lambda-mmd", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--lambda-momentum", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--lambda-lat-dir", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--lambda-drift-align", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--lambda-residual-ratio", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--kinetic-terminal-beta",
        type=float,
        default=None,
        help="Relative weight (>1) of the FINAL momentum in the kinetic penalty; "
        "up-weighting the terminal momentum penalises coasting past the target and "
        "directly curbs the latent displacement overshoot.",
    )
    parser.add_argument(
        "--latent-disp-detach-potential",
        dest="latent_disp_detach_potential",
        action="store_true",
        default=None,
        help="Run a second potential-detached (z,p) integration whose prediction "
        "feeds only the population displacement/direction loss, so those gradients "
        "do not distort the U0 landscape.",
    )
    parser.add_argument(
        "--no-latent-disp-detach-potential",
        dest="latent_disp_detach_potential",
        action="store_false",
        help="Disable potential gating for the displacement loss.",
    )
    parser.add_argument(
        "--latent-disp-ot-coupling",
        dest="latent_disp_ot_coupling",
        action="store_true",
        default=None,
        help="Supervise each source cell against its entropic-OT barycentric image "
        "in z_next (per cell type) instead of the group centroid; constrains per-cell "
        "direction and magnitude to curb the displacement overshoot.",
    )
    parser.add_argument(
        "--no-latent-disp-ot-coupling",
        dest="latent_disp_ot_coupling",
        action="store_false",
        help="Use the centroid displacement target instead of OT coupling.",
    )
    parser.add_argument(
        "--latent-disp-ot-blur",
        type=float,
        default=None,
        help="Sinkhorn entropic scale (blur) for the OT-coupled displacement target.",
    )
    parser.add_argument(
        "--latent-disp-mag-ratio",
        dest="latent_disp_use_mag_ratio",
        action="store_true",
        default=None,
        help="Penalise step-length overshoot: mean(ReLU(||z_pred-z_curr||/||tgt-z_curr|| - 1)^2) "
        "instead of absolute position MSE on the OT barycentric target.",
    )
    parser.add_argument(
        "--no-latent-disp-mag-ratio",
        dest="latent_disp_use_mag_ratio",
        action="store_false",
        help="Use position MSE on the OT displacement target (legacy).",
    )
    parser.add_argument(
        "--latent-disp-exclude-ema",
        dest="latent_disp_exclude_ema",
        action="store_true",
        default=None,
        help="Add lat_disp at raw scale (exclude from EMA loss normalization).",
    )
    parser.add_argument(
        "--no-latent-disp-exclude-ema",
        dest="latent_disp_exclude_ema",
        action="store_false",
        help="Include lat_disp in EMA loss normalization (legacy).",
    )
    parser.add_argument(
        "--latent-disp-fullpop-ot",
        dest="latent_disp_fullpop_ot",
        action="store_true",
        default=None,
        help="Precompute entropic-OT barycentric targets on the full per-type train "
        "population for each transition (avoids batch-OT soft-centroid degeneracy).",
    )
    parser.add_argument(
        "--no-latent-disp-fullpop-ot",
        dest="latent_disp_fullpop_ot",
        action="store_false",
        help="Compute OT displacement targets within each mini-batch only.",
    )
    parser.add_argument(
        "--hamiltonian-damping-gamma",
        type=float,
        default=None,
        help="Override linear damping gamma in the Hamiltonian flow.",
    )
    parser.add_argument("--lambda-density", type=float, default=None)
    parser.add_argument("--use-density-regularization", action="store_true")
    parser.add_argument(
        "--no-density-regularization",
        action="store_true",
        help="Disable density alignment (overrides recommended landscape defaults).",
    )
    parser.add_argument(
        "--lambda-residual-balance",
        type=float,
        default=None,
        help="Weight for penalizing residual drift dominating gradient flow.",
    )
    parser.add_argument(
        "--residual-ratio-target",
        type=float,
        default=None,
        help="Target upper bound for mean residual_ratio during training.",
    )
    parser.add_argument(
        "--homeostasis-ref-time",
        type=float,
        default=None,
        help="Reference time for potential_deviation at inference (default: earliest time).",
    )
    parser.add_argument(
        "--no-latent-density-batch",
        action="store_true",
        help="Use precomputed PCA KDE density targets instead of latent k-NN batches.",
    )
    parser.add_argument(
        "--density-basis",
        choices=["X_pca", "X_latent_pca"],
        default="X_pca",
    )
    parser.add_argument("--density-n-pcs", type=int, default=20)
    parser.add_argument("--density-bandwidth", type=float, default=None)
    parser.add_argument(
        "--checkpoint-suffix",
        type=str,
        default=None,
        help="Optional suffix appended to checkpoint dir (e.g. ablation tag).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="If set, write all training outputs to this directory (overrides auto naming).",
    )
    args = parser.parse_args(argv)

    spec, prepare_fn = DATASETS[args.dataset]
    if args.dataset == "GSE225948_Brain" and args.profile:
        if args.profile not in GSE225948_BRAIN_PROFILES:
            raise SystemExit(f"Unknown Brain profile {args.profile!r}; choose {sorted(GSE225948_BRAIN_PROFILES)}")
        spec = GSE225948_BRAIN_PROFILES[args.profile]
    elif args.dataset == "GSE141259" and args.profile:
        if args.profile not in GSE141259_PROFILES:
            raise SystemExit(f"Unknown GSE141259 profile {args.profile!r}; choose {sorted(GSE141259_PROFILES)}")
        spec = GSE141259_PROFILES[args.profile]
    config = apply_train_config(spec)
    if args.loss_recipe == "hamiltonian":
        apply_hamiltonian_loss_options(config)
    if args.hidden_dim is not None:
        config.hidden_dim = int(args.hidden_dim)
    if args.n_layers is not None:
        config.n_layers = int(args.n_layers)
    if args.dropout is not None:
        config.dropout = float(args.dropout)
    if args.epochs is not None:
        config.epochs = int(args.epochs)
    if args.smoke_test:
        config.skip_final_evaluation = True
    if args.checkpoint_metric is not None:
        config.checkpoint_metric = args.checkpoint_metric
    if args.early_stop_metric is not None:
        config.early_stop_metric = args.early_stop_metric
    if args.early_stop_patience is not None:
        config.early_stop_patience = int(args.early_stop_patience)
    if args.val_mode is not None:
        config.val_mode = args.val_mode
    if args.val_ratio is not None:
        config.val_ratio = float(args.val_ratio)
    if args.val_time_point is not None:
        config.val_time_point = float(args.val_time_point)
    if args.loss_normalization:
        config.use_loss_normalization = True
    if args.no_loss_normalization:
        config.use_loss_normalization = False
    if args.total_drift_hjb:
        config.use_total_drift_hjb = True
    if args.residual_drift_mode is not None:
        config.residual_drift_mode = args.residual_drift_mode
    if args.momentum_loss_type is not None:
        config.momentum_loss_type = args.momentum_loss_type
    if args.potential_time_mode is not None:
        config.potential_time_mode = args.potential_time_mode
    if args.potential_time_correction_scale is not None:
        config.potential_time_correction_scale = float(args.potential_time_correction_scale)
    if args.lambda_pair_mse is not None or args.lambda_mmd is not None:
        warnings.warn(
            "--lambda-pair-mse / --lambda-mmd are deprecated; use --lambda-recon "
            "(mse_mmd uses fixed 1:5 MSE:MMD mix). Legacy weights applied for this run.",
            UserWarning,
            stacklevel=1,
        )
        config._legacy_lambda_pair_mse = (
            float(args.lambda_pair_mse) if args.lambda_pair_mse is not None else None
        )
        config._legacy_lambda_mmd = (
            float(args.lambda_mmd) if args.lambda_mmd is not None else None
        )
    if args.reconstruction_mode is not None:
        config.reconstruction_mode = args.reconstruction_mode
    if args.lambda_recon is not None:
        config.lambda_recon = float(args.lambda_recon)
    energy = args.lambda_energy if args.lambda_energy is not None else args.lambda_hjb
    if energy is not None:
        config.lambda_hjb = float(energy)
        config.lambda_reg = float(energy)
    if args.lambda_momentum is not None:
        warnings.warn(
            "--lambda-momentum is deprecated; momentum is merged into --lambda-kinetic "
            f"(fixed mix {INERTIA_MOMENTUM_MIX}). Legacy weight applied for this run.",
            UserWarning,
            stacklevel=1,
        )
        config._legacy_lambda_momentum = float(args.lambda_momentum)
    if args.lambda_latent is not None:
        config.lambda_latent = float(args.lambda_latent)
    if args.lambda_kinetic is not None:
        config.lambda_kinetic = float(args.lambda_kinetic)
    if args.lambda_lat_disp is not None:
        config.lambda_lat_disp = float(args.lambda_lat_disp)
    if args.lambda_lat_dir is not None:
        warnings.warn(
            "--lambda-lat-dir is deprecated; displacement MSE already covers direction.",
            UserWarning,
            stacklevel=1,
        )
    if args.hamiltonian_damping_gamma is not None:
        config.hamiltonian_damping_gamma = float(args.hamiltonian_damping_gamma)
    if args.kinetic_terminal_beta is not None:
        config.kinetic_terminal_beta = float(args.kinetic_terminal_beta)
    if args.latent_disp_detach_potential is not None:
        config.latent_disp_detach_potential = bool(args.latent_disp_detach_potential)
    if args.latent_disp_ot_coupling is not None:
        config.latent_disp_ot_coupling = bool(args.latent_disp_ot_coupling)
    if args.latent_disp_ot_blur is not None:
        config.latent_disp_ot_blur = float(args.latent_disp_ot_blur)
    if args.latent_disp_use_mag_ratio is not None:
        config.latent_disp_use_mag_ratio = bool(args.latent_disp_use_mag_ratio)
    if args.latent_disp_exclude_ema is not None:
        config.latent_disp_exclude_ema = bool(args.latent_disp_exclude_ema)
    if args.latent_disp_fullpop_ot is not None:
        config.latent_disp_fullpop_ot = bool(args.latent_disp_fullpop_ot)
    if args.no_density_regularization:
        config.use_density_regularization = False
        config.lambda_density = 0.0
    elif args.lambda_density is not None:
        config.lambda_density = float(args.lambda_density)
        if config.lambda_density > 0:
            config.use_density_regularization = True
    if args.use_density_regularization:
        config.use_density_regularization = True
    if args.no_latent_density_batch:
        config.density_use_latent_batch = False
    if args.lambda_residual_balance is not None:
        config.lambda_residual_balance = float(args.lambda_residual_balance)
    if args.residual_ratio_target is not None:
        config.residual_ratio_target = float(args.residual_ratio_target)
    if args.homeostasis_ref_time is not None:
        config.homeostasis_ref_time = float(args.homeostasis_ref_time)
    config.density_basis = args.density_basis
    config.density_n_pcs = int(args.density_n_pcs)
    config.density_bandwidth = args.density_bandwidth
    _apply_ablation_flags(config, args)

    config.data_path = resolve_data_path(spec)
    if args.checkpoint_dir:
        save_dir = str(args.checkpoint_dir)
    else:
        save_dir = build_training_checkpoint_dir(spec, config)
        if args.smoke_test and args.epochs is not None:
            save_dir = f"{save_dir}_smoke{args.epochs}"
        if args.checkpoint_suffix:
            save_dir = f"{save_dir}_{args.checkpoint_suffix}"

    print(format_train_config_summary(spec, config))
    _print_interpretation_notice(args.dataset)
    print(f"Checkpoint dir: {save_dir}")

    adata = sc.read(config.data_path)
    adata = prepare_fn(adata, config)
    print(adata.shape)

    processor = TemporalDataProcessor(adata)
    adata = processor.process()

    model = TemporalSDENetwork(config, adata)
    trainer = SDETrainer(model, adata, cfg=config, save_dir=save_dir)
    setup_scanpy_figdir(save_dir)
    trainer.train()
    plot_metrics(trainer, adata, save_dir=save_dir)

    summary_path = save_training_summary(
        save_dir,
        dataset=args.dataset,
        profile=args.profile,
        spec=spec,
        config=config,
        metrics=trainer.metrics,
        val_split_info=trainer.val_split.split_info,
    )
    print(f"Training summary: {summary_path}")

    if args.smoke_test:
        print(f"Smoke test done. See {save_dir}/Loss_epoch.csv and training_loss_curve.png")
        return

    if args.dataset == "GSE225948_Brain":
        print(BRAIN_INTERPRETATION_WARNING, flush=True)

    adata = predict(model=model, adata=adata, config=config, save_dir=save_dir)

    analyzer = KineticAnalyzer(model=model, adata=adata, config=config, save_dir=save_dir)
    analyzer.infer_kinetics()
    analyzer.plot_kinetics()
    analyzer.plot_gene_trends()
    analyzer.plot_potential_landscape()

    adata.obs.to_csv(f"{save_dir}/obs.csv")

    pc1_loadings = adata.varm["PCs"][:, 0]
    gene_loadings = pd.DataFrame({"gene": adata.var_names, "PC1_loading": pc1_loadings})
    top_10_genes = gene_loadings.nlargest(10, "PC1_loading")
    bottom_10_genes = gene_loadings.nsmallest(10, "PC1_loading")
    analyzer.plot_genes(
        gene_list=[top_10_genes.iloc[0, 0], top_10_genes.iloc[1, 0], bottom_10_genes.iloc[0, 0]]
    )
    analyzer.plot_gene_phase(
        top_10_genes.iloc[0, 0],
        top_10_genes.iloc[1, 0],
        color_by="annotation" if "annotation" in adata.obs.columns else "cell_type",
        add_trend=True,
        trend_style="lowess",
    )

    result = gene_specific_mse(adata, marker_genes=[top_10_genes.iloc[0, 0], top_10_genes.iloc[1, 0]])
    print(result)

    run_standard_training_figures(
        adata,
        save_dir,
        config=config,
        top_gene=top_10_genes.iloc[1, 0],
    )

    print(f"Checkpoint saved to: {save_dir}")
    print(adata)


if __name__ == "__main__":
    main()
