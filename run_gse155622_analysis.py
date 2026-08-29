#!/usr/bin/env python
"""
GSE155622 protocol analysis: within-type relative potential, deviation timeline,
and 24h→2d Hamiltonian momentum rollout for SNIIC state switching.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from plot_utils import (
    ACCENT_HI,
    PALETTE,
    configure_headless,
    style_axis,
)

configure_headless()

from analysis_protocol_utils import (
    NOCICEPTIVE_SODIUM_CHANNELS,
    PAIN_CHANNELS,
    PAIN_CHANNELS_PLOT,
    SNIIC_MODULES,
    SWITCH_FACTORS_155622,
    fast_wilcoxon_deg,
    fig_path,
    gene_expression,
    init_protocol_outdir,
    module_score,
    resolve_genes,
    result_path,
    spearman_safe,
    write_json,
    write_output_file_index,
)
from celltype_analysis import GSE155622_PROFILE, load_annotated_adata
from dataset_pipeline import PROJECT_ROOT
from deg_enrichment_workflow import run_pathway_enrichment
from hamiltonian_flow import integrate_hamiltonian_flow
from model_dynamics import load_model_dynamics_from_checkpoint

DEFAULT_CHECKPOINT = (
    "GSE155622_checkpoints_3000_3000_384_0.05_recon0.01_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
)

CONDITION_ORDER = ["Control", "SNI 6h", "SNI 24h", "SNI 2d", "SNI 7d", "SNI 14d"]
CONDITION_TIME_DAYS = {
    "Control": 0.0,
    "SNI 6h": 0.25,
    "SNI 24h": 1.0,
    "SNI 2d": 2.0,
    "SNI 7d": 7.0,
    "SNI 14d": 14.0,
}


def _ensure_time_column(adata) -> None:
    """AnnData may only have condition/stage after annotate_fn; synthesize numeric time."""
    if "time" in adata.obs.columns:
        return
    cond = None
    for key in ("condition", "stage"):
        if key in adata.obs.columns:
            cond = adata.obs[key].astype(str)
            break
    if cond is None:
        raise KeyError("Need condition or stage to build time")
    adata.obs["time"] = cond.map(CONDITION_TIME_DAYS).astype(float)


def _resolve_ckpt(override: Optional[str]) -> Path:
    if override:
        p = Path(override).expanduser()
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p.resolve()
    return (PROJECT_ROOT / DEFAULT_CHECKPOINT).resolve()


def _neuron(adata):
    col = "celltype" if "celltype" in adata.obs else "annotation"
    return adata[adata.obs[col].astype(str) == "Neuron"].copy()


def step_relative_potential_deg(adata_neu, out_dir: Path, *, organism: str = "Mouse") -> dict:
    """High vs low potential_relative_type DEG + SNIIC / channel correlations."""
    report: Dict[str, object] = {}
    if "potential_relative_type" not in adata_neu.obs:
        raise KeyError("potential_relative_type missing from obs")

    rel = adata_neu.obs["potential_relative_type"].astype(float).values
    hi = rel >= np.nanquantile(rel, 0.75)
    lo = rel <= np.nanquantile(rel, 0.25)
    adata_neu.obs["relU_bin"] = "mid"
    adata_neu.obs.loc[hi, "relU_bin"] = "high_relU"
    adata_neu.obs.loc[lo, "relU_bin"] = "low_relU"
    report["n_high"] = int(hi.sum())
    report["n_low"] = int(lo.sum())

    deg = fast_wilcoxon_deg(
        adata_neu,
        group_key="relU_bin",
        group_1="high_relU",
        group_2="low_relU",
        n_top_genes=2500,
    )
    if deg is not None and not deg.empty:
        deg.to_csv(
            result_path(out_dir, "relative_potential_neuron_high_vs_low_relU_DEG.csv"),
            index=False,
        )
        report["deg_top"] = deg.head(20).to_dict(orient="records")
        up = deg.loc[deg["logfoldchange"] > 0, "gene"].astype(str).tolist()[:200]
        enr, ewarn = run_pathway_enrichment(
            up, comparison="high_relU", direction="up", organism=organism, gene_sets=["GO_Biological_Process_2023"]
        )
        if ewarn:
            report["enrichment_warning"] = ewarn
        if enr is not None and not enr.empty:
            enr.to_csv(
                result_path(out_dir, "relative_potential_neuron_high_relU_enrichment.csv"),
                index=False,
            )
            report["enrichment_top"] = enr.head(15).to_dict(orient="records")

    # Module scores vs relative potential
    rows = []
    for name, genes in SNIIC_MODULES.items():
        scs = module_score(adata_neu, genes, obs_key=f"score_{name}")
        rows.append(
            {
                "module": name,
                "genes_resolved": ",".join(resolve_genes(adata_neu.var_names, genes)),
                "spearman_vs_relU": spearman_safe(scs, rel),
                "mean_high_relU": float(np.nanmean(scs[hi])),
                "mean_low_relU": float(np.nanmean(scs[lo])),
            }
        )
    for g in PAIN_CHANNELS_PLOT:
        resolved = resolve_genes(adata_neu.var_names, [g])
        if not resolved:
            continue
        expr = gene_expression(adata_neu, resolved[0])
        rows.append(
            {
                "module": f"channel:{resolved[0]}",
                "genes_resolved": resolved[0],
                "spearman_vs_relU": spearman_safe(expr, rel),
                "mean_high_relU": float(np.nanmean(expr[hi])),
                "mean_low_relU": float(np.nanmean(expr[lo])),
            }
        )
    # Classical nociceptive Nav triad (Scn9a/Scn10a/Scn11a) as a module score.
    nav_genes = resolve_genes(adata_neu.var_names, NOCICEPTIVE_SODIUM_CHANNELS)
    if nav_genes:
        nav_score = module_score(adata_neu, nav_genes, obs_key="score_nociceptive_Nav")
        rows.append(
            {
                "module": "nociceptive_Nav_triad",
                "genes_resolved": ",".join(nav_genes),
                "spearman_vs_relU": spearman_safe(nav_score, rel),
                "mean_high_relU": float(np.nanmean(nav_score[hi])),
                "mean_low_relU": float(np.nanmean(nav_score[lo])),
            }
        )
    mod_df = pd.DataFrame(rows)
    mod_df.to_csv(
        result_path(out_dir, "relative_potential_SNIIC_and_pain_channel_vs_relU.csv"),
        index=False,
    )
    report["module_correlations"] = mod_df.to_dict(orient="records")

    # Smoothed expression trend along the relative-potential axis
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    order = np.argsort(rel)
    x = np.linspace(0, 1, len(order))
    plot_genes = resolve_genes(adata_neu.var_names, PAIN_CHANNELS_PLOT)
    missing = [g for g in PAIN_CHANNELS_PLOT if g.lower() not in {p.lower() for p in plot_genes}]
    if missing:
        print(f"[warn] pain channels missing from var_names: {missing}", flush=True)
    for i, g in enumerate(plot_genes):
        y = gene_expression(adata_neu, g)[order]
        w = max(20, len(y) // 40)
        ker = np.ones(w) / w
        ys = np.convolve(y, ker, mode="same")
        ax.plot(x, ys, lw=2.2, color=PALETTE[i % len(PALETTE)], label=g, zorder=3)
    ax.set_xlabel("Cells ranked by potential_relative_type  (low → high U)")
    ax.set_ylabel("Expression (smoothed)")
    ax.set_title(
        r"Classical nociceptive $\mathrm{Na}_{v}$s (Scn9a/10a/11a) and Trpv1 vs $U_{\mathrm{rel}}$"
    )
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left", ncol=1, fontsize=8)
    style_axis(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(
        fig_path(out_dir, "relative_potential_pain_channels_vs_relU.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    write_json(report, result_path(out_dir, "relative_potential_summary.json"))
    return report


def step_deviation_timeline(adata_neu, out_dir: Path) -> dict:
    if "potential_deviation" not in adata_neu.obs or "condition" not in adata_neu.obs:
        raise KeyError("Need potential_deviation and condition")
    _ensure_time_column(adata_neu)
    df = adata_neu.obs[["condition", "potential_deviation", "time"]].copy()
    df["potential_deviation"] = df["potential_deviation"].astype(float)
    df["abs_deviation"] = df["potential_deviation"].abs()
    # Also report relative to Control mean (pathology magnitude)
    ctrl = df.loc[df["condition"].astype(str) == "Control", "potential_deviation"]
    ctrl_mean = float(ctrl.mean()) if len(ctrl) else 0.0
    df["dev_vs_control"] = df["potential_deviation"] - ctrl_mean

    rows = []
    for cond in CONDITION_ORDER:
        sub = df[df["condition"].astype(str) == cond]
        if sub.empty:
            continue
        rows.append(
            {
                "condition": cond,
                "n": int(len(sub)),
                "time": float(sub["time"].astype(float).median()),
                "mean_deviation": float(sub["potential_deviation"].mean()),
                "se_deviation": float(sub["potential_deviation"].std(ddof=1) / np.sqrt(len(sub))),
                "mean_abs_deviation": float(sub["abs_deviation"].mean()),
                "mean_dev_vs_control": float(sub["dev_vs_control"].mean()),
            }
        )
    tab = pd.DataFrame(rows).sort_values("time")
    tab.to_csv(result_path(out_dir, "deviation_timeline_neuron_by_condition.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axvspan(0.8, 2.2, color=ACCENT_HI, alpha=0.10, lw=0, label="24h–2d window", zorder=0)
    ax.errorbar(
        tab["time"],
        tab["mean_abs_deviation"],
        yerr=tab["se_deviation"],
        fmt="-o",
        color=PALETTE[0],
        ecolor=PALETTE[0],
        elinewidth=1.2,
        capsize=3,
        lw=2.2,
        ms=7,
        markerfacecolor="white",
        markeredgecolor=PALETTE[0],
        markeredgewidth=1.8,
        label="mean |potential_deviation|",
        zorder=3,
    )
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Mean |potential_deviation|")
    ax.set_title("Neuron homeostasis deviation over the SNI time course")
    ax.legend(loc="lower right")
    yspan = float(tab["mean_abs_deviation"].max() - tab["mean_abs_deviation"].min()) or 1.0
    annotation_layout = {
        "Control": dict(offset=(0, -10), ha="center", va="top"),
        "SNI 6h": dict(offset=(13, 0), ha="left", va="center"),
        "SNI 24h": dict(offset=(11, -9), ha="left", va="top"),
        "SNI 2d": dict(offset=(0, 13), ha="center", va="bottom"),
        "SNI 7d": dict(offset=(0, 12), ha="center", va="bottom"),
        "SNI 14d": dict(offset=(0, -13), ha="center", va="top"),
    }
    for _, r in tab.sort_values("time").iterrows():
        layout = annotation_layout[str(r["condition"])]
        ax.annotate(
            str(r["condition"]),
            xy=(r["time"], r["mean_abs_deviation"]),
            xytext=layout["offset"],
            textcoords="offset points",
            fontsize=7.5,
            ha=layout["ha"],
            va=layout["va"],
            color="#374151",
            annotation_clip=False,
        )
    ax.set_xlim(-0.55, 14.5)
    ax.set_ylim(tab["mean_abs_deviation"].min() - 0.18 * yspan,
               tab["mean_abs_deviation"].max() + 0.27 * yspan)
    style_axis(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(
        fig_path(out_dir, "deviation_timeline_neuron_homeostasis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Peak window check on abs deviation among injury times
    inj = tab[tab["condition"] != "Control"]
    peak_cond = str(inj.loc[inj["mean_abs_deviation"].idxmax(), "condition"]) if len(inj) else None
    peak_in_window = peak_cond in ("SNI 24h", "SNI 2d")
    summary = {
        "table": tab.to_dict(orient="records"),
        "peak_condition": peak_cond,
        "peak_in_24h_2d_window": peak_in_window,
    }
    write_json(summary, result_path(out_dir, "deviation_timeline_summary.json"))
    print(f"[deviation] peak={peak_cond} in_24h_2d_window={peak_in_window}")
    return summary


def _load_hamiltonian_bundle_from_checkpoint(checkpoint: Path, device: str = "cpu"):
    """
    Load MomentumNetwork + PotentialNetwork + HamiltonianFlowFunc from state_dict
    without reconstructing the full gene encoder (uses saved X_latent for z0).
    """
    from train_model import Config, MomentumNetwork, PotentialNetwork
    from hamiltonian_flow import HamiltonianFlowFunc

    ckpt_path = Path(checkpoint) / "best_model.pth"
    if not ckpt_path.is_file():
        return None
    state = torch.load(ckpt_path, map_location=device)
    if "momentum_net.net.0.weight" not in state or "potential_net.stationary_net.0.weight" not in state:
        return None
    latent_dim = int(state["momentum_net.net.0.weight"].shape[1]) - 1
    cfg = Config()
    cfg.hidden_dim = latent_dim
    cfg.use_hamiltonian_flow = True
    cfg.use_state_momentum = True
    cfg.use_residual_drift = False
    cfg.potential_time_mode = (
        "quasi_stationary"
        if any(k.startswith("potential_net.stationary_net") for k in state)
        else "time_varying"
    )
    pot = PotentialNetwork(latent_dim, cfg)
    mom = MomentumNetwork(latent_dim, cfg)
    flow = HamiltonianFlowFunc(pot, latent_dim, cfg, residual_net=None)
    pot_state = {k.replace("potential_net.", ""): v for k, v in state.items() if k.startswith("potential_net.")}
    mom_state = {k.replace("momentum_net.", ""): v for k, v in state.items() if k.startswith("momentum_net.")}
    pot.load_state_dict(pot_state, strict=False)
    mom.load_state_dict(mom_state, strict=False)
    pot.to(device).eval()
    mom.to(device).eval()
    flow.to(device).eval()

    class _Bundle:
        def initial_momentum(self, z, t):
            return mom(z, t)

        @property
        def flow_func(self):
            return flow

    return _Bundle()


def step_momentum_rollout(adata_neu, checkpoint: Path, out_dir: Path, *, n_seeds: int = 40) -> dict:
    """Integrate Hamiltonian flow from 24h (SNIIC2-high) seeds toward 2d horizon."""
    report: Dict[str, object] = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_hamiltonian_bundle_from_checkpoint(checkpoint, device=device)
    if model is None:
        # Fallback: try full TemporalSDENetwork loader
        evaluator = load_model_dynamics_from_checkpoint(str(checkpoint), adata_full=adata_neu, device=device)
        model = None if evaluator is None else evaluator.model
    if model is None or not hasattr(model, "flow_func") or not hasattr(model, "initial_momentum"):
        report["error"] = "Could not load Hamiltonian/momentum networks from checkpoint"
        write_json(report, result_path(out_dir, "momentum_24h_to_2d_summary.json"))
        return report

    from latent_embeddings import ensure_latent_embeddings

    # Prefer full latent (matches MomentumNetwork dim), not PCA slice
    key, _ = ensure_latent_embeddings(adata_neu, checkpoint_dir=str(checkpoint), warn=True)
    if "X_latent" in adata_neu.obsm:
        key = "X_latent"
    if key not in adata_neu.obsm:
        report["error"] = f"Missing latent key {key}"
        write_json(report, result_path(out_dir, "momentum_24h_to_2d_summary.json"))
        return report

    _ensure_time_column(adata_neu)
    z_all = np.asarray(adata_neu.obsm[key], dtype=float)
    cond = adata_neu.obs["condition"].astype(str).values
    mask_24 = cond == "SNI 24h"
    if mask_24.sum() < 5:
        report["error"] = "Too few SNI 24h neurons"
        write_json(report, result_path(out_dir, "momentum_24h_to_2d_summary.json"))
        return report

    s2 = module_score(adata_neu, SNIIC_MODULES["SNIIC2"])
    s1 = module_score(adata_neu, SNIIC_MODULES["SNIIC1"])
    adata_neu.obs["score_SNIIC1"] = s1
    adata_neu.obs["score_SNIIC2"] = s2

    # Seeds: top SNIIC2 within 24h
    idx24 = np.where(mask_24)[0]
    order = idx24[np.argsort(-s2[idx24])]
    seeds = order[: min(n_seeds, len(order))]
    z0 = z_all[seeds]
    t0 = float(adata_neu.obs["time"].astype(float).values[seeds].mean())
    t1 = 2.0  # SNI 2d

    z_t = torch.tensor(z0, dtype=torch.float32, device=device)
    t_in = torch.full((z_t.shape[0], 1), t0, dtype=torch.float32, device=device)
    with torch.no_grad():
        p0 = model.initial_momentum(z_t, t_in)
    ts = torch.linspace(t0, t1, steps=9, device=device)
    # Gradients required for ∇U inside Hamiltonian integration (inference only).
    with torch.enable_grad():
        traj, p_final = integrate_hamiltonian_flow(
            model.flow_func, z_t, p0, ts, dt=0.05, add_noise=False, detach_potential=True
        )
    traj_np = traj.detach().cpu().numpy()  # (T,B,D)

    # Map each step to nearest cells for module / gene readouts
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=5).fit(z_all)
    step_rows = []
    factor_rows = []
    for ti, tval in enumerate(ts.detach().cpu().numpy()):
        pts = traj_np[ti]
        _, nn = nbrs.kneighbors(pts)
        nn_flat = nn.ravel()
        step_rows.append(
            {
                "t": float(tval),
                "SNIIC1": float(np.nanmean(s1[nn_flat])),
                "SNIIC2": float(np.nanmean(s2[nn_flat])),
                "delta_SNIIC1_minus_SNIIC2": float(np.nanmean(s1[nn_flat] - s2[nn_flat])),
            }
        )
        for g in resolve_genes(adata_neu.var_names, SWITCH_FACTORS_155622):
            factor_rows.append(
                {
                    "t": float(tval),
                    "gene": g,
                    "mean_expr": float(np.nanmean(gene_expression(adata_neu, g)[nn_flat])),
                }
            )
    step_df = pd.DataFrame(step_rows)
    fac_df = pd.DataFrame(factor_rows)
    step_df.to_csv(result_path(out_dir, "momentum_24h_to_2d_SNIIC_module_track.csv"), index=False)
    fac_df.to_csv(result_path(out_dir, "momentum_24h_to_2d_switch_factors.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(step_df["t"], step_df["SNIIC2"], "-o", label="SNIIC2 (early/reactive)",
            color="#E0BFB8", lw=2.2, ms=7, markerfacecolor="white",
            markeredgecolor="#E0BFB8", markeredgewidth=1.8, zorder=3)
    ax.plot(step_df["t"], step_df["SNIIC1"], "-o", label="SNIIC1 (late/chronic)",
            color="#9EC1C0", lw=2.2, ms=7, markerfacecolor="white",
            markeredgecolor="#9EC1C0", markeredgewidth=1.8, zorder=3)
    ax.set_xlabel("Simulated time (days)")
    ax.set_ylabel("Module score (nearest cells)")
    ax.set_title("Hamiltonian momentum rollout 24h → 2d:\nSNIIC2 → SNIIC1 sub-state drift", fontsize=12)
    ax.legend(loc="center right")
    style_axis(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(
        fig_path(out_dir, "momentum_24h_to_2d_SNIIC_substate_drift.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    report.update(
        {
            "n_seeds": int(len(seeds)),
            "t0": t0,
            "t1": t1,
            "SNIIC_start_delta": float(step_df.iloc[0]["delta_SNIIC1_minus_SNIIC2"]),
            "SNIIC_end_delta": float(step_df.iloc[-1]["delta_SNIIC1_minus_SNIIC2"]),
            "lateral_drift_toward_SNIIC1": bool(
                step_df.iloc[-1]["delta_SNIIC1_minus_SNIIC2"] > step_df.iloc[0]["delta_SNIIC1_minus_SNIIC2"]
            ),
            "track": step_df.to_dict(orient="records"),
        }
    )
    write_json(report, result_path(out_dir, "momentum_24h_to_2d_summary.json"))
    print(
        f"[momentum] seeds={len(seeds)} drift_to_SNIIC1={report['lateral_drift_toward_SNIIC1']} "
        f"delta {report['SNIIC_start_delta']:.3f}→{report['SNIIC_end_delta']:.3f}"
    )
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="GSE155622 protocol analysis")
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--n-seeds", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    np.random.seed(args.seed)
    checkpoint = _resolve_ckpt(args.checkpoint_dir)
    out_root = init_protocol_outdir(checkpoint / "analysis_protocol_GSE155622")
    print(f"Checkpoint: {checkpoint}")
    print("Loading AnnData...")
    adata = load_annotated_adata(GSE155622_PROFILE, str(checkpoint))
    # Light preprocess for DEG on neurons
    neu = _neuron(adata)
    print(f"Neuron cells: {neu.n_obs}")
    sc.pp.normalize_total(neu, target_sum=1e4, inplace=True)
    sc.pp.log1p(neu)

    r1 = step_relative_potential_deg(neu, out_root)
    r2 = step_deviation_timeline(neu, out_root)
    r3 = step_momentum_rollout(neu, checkpoint, out_root, n_seeds=args.n_seeds)
    write_json(
        {
            "checkpoint": str(checkpoint),
            "relative_potential": r1,
            "deviation_timeline": r2,
            "momentum_24h_to_2d": r3,
        },
        result_path(out_root, "analysis_summary.json"),
    )
    write_output_file_index(
        out_root,
        [
            ("figures/relative_potential_pain_channels_vs_relU.png",
             "Classical nociceptive Navs (Scn9a/Scn10a/Scn11a=Nav1.7/1.8/1.9) + Trpv1 vs neuron potential_relative_type"),
            ("figures/deviation_timeline_neuron_homeostasis.png",
             "Neuron mean |potential_deviation| across SNI time course (24h–2d window shaded)"),
            ("figures/momentum_24h_to_2d_SNIIC_substate_drift.png",
             "Hamiltonian momentum rollout 24h→2d: SNIIC2 vs SNIIC1 module scores"),
            ("relative_potential_neuron_high_vs_low_relU_DEG.csv",
             "Wilcoxon DEG: high vs low potential_relative_type neurons"),
            ("relative_potential_neuron_high_relU_enrichment.csv",
             "Pathway enrichment of genes up in high-relU neurons"),
            ("relative_potential_SNIIC_and_pain_channel_vs_relU.csv",
             "SNIIC modules + Trpv1 + classical Nav triad (Scn9a/10a/11a) Spearman vs relU"),
            ("relative_potential_summary.json",
             "Summary metrics for relative-potential step"),
            ("deviation_timeline_neuron_by_condition.csv",
             "Per-condition mean/SE of potential_deviation"),
            ("deviation_timeline_summary.json",
             "Peak-deviation condition and 24h–2d window check"),
            ("momentum_24h_to_2d_SNIIC_module_track.csv",
             "Per-time-step SNIIC1/SNIIC2 scores along simulated trajectories"),
            ("momentum_24h_to_2d_switch_factors.csv",
             "Atf3/Egr1/Cpeb1 expression along simulated trajectories"),
            ("momentum_24h_to_2d_summary.json",
             "Momentum rollout summary (seeds, drift direction)"),
            ("analysis_summary.json",
             "Top-level summary aggregating all GSE155622 protocol steps"),
            ("OUTPUT_FILE_INDEX.md",
             "This file: human-readable description of every output"),
        ],
    )
    print(f"Done. Results under: {out_root}")


if __name__ == "__main__":
    main()
