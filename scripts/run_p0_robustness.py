#!/usr/bin/env python
"""P0 robustness: Atf3-removed SNIIC + SOD2 threshold/random-gene sensitivity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output_file" / "robustness" / "p0_robustness"
OUT.mkdir(parents=True, exist_ok=True)

CKPT_PAIN = (
    ROOT
    / "GSE155622_checkpoints_3000_3000_384_0.05_recon0.01_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
)
CKPT_HG = (
    ROOT
    / "HGSOC_checkpoints_3000_3000_512_0.05_recon0.01_nactpair_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
)

# Full vs Atf3-removed modules
MODULES = {
    "SNIIC1_full": ("Atf3", "Gfra3", "Gal"),
    "SNIIC1_noAtf3": ("Gfra3", "Gal"),
    "SNIIC2_full": ("Atf3", "Mrgprd"),
    "SNIIC2_noAtf3": ("Mrgprd",),
    "SNIIC3_full": ("Atf3", "S100b", "Gal"),
    "SNIIC3_noAtf3": ("S100b", "Gal"),
    "Atf3_alone": ("Atf3",),
}


def _score_modules(adata) -> Dict[str, np.ndarray]:
    from analysis_protocol_utils import module_score

    return {k: module_score(adata, genes) for k, genes in MODULES.items()}


def _bootstrap_less(wt_vals: np.ndarray, ko_vals: np.ndarray, n_boot: int = 500, seed: int = 0) -> float:
    """P(KO end < WT end) style: fraction of boots where (ko_mean - wt_mean) >= observed if we test reduction."""
    rng = np.random.default_rng(seed)
    obs = float(np.nanmean(ko_vals) - np.nanmean(wt_vals))
    # test whether KO reduces relative to WT: H1 mean(ko)<mean(wt) => obs < 0
    count = 0
    pooled = np.concatenate([wt_vals, ko_vals])
    n_w, n_k = len(wt_vals), len(ko_vals)
    for _ in range(n_boot):
        samp = rng.choice(pooled, size=n_w + n_k, replace=True)
        d = float(np.mean(samp[n_w:]) - np.mean(samp[:n_w]))
        if d <= obs:
            count += 1
    return float((count + 1) / (n_boot + 1))


def rollout_multi_module(
    adata_neu,
    checkpoint: Path,
    scores: Dict[str, np.ndarray],
    *,
    seed_condition: str = "SNI 24h",
    t1: float = 2.0,
    n_seeds: int = 40,
    device: str = "cpu",
    latent_shift_direction: Optional[np.ndarray] = None,
    latent_shift_scale: float = 0.0,
    seed_by: str = "SNIIC2_full",
) -> Tuple[pd.DataFrame, np.ndarray]:
    from hamiltonian_flow import integrate_hamiltonian_flow
    from run_gse155622_analysis import (
        _ensure_time_column,
        _load_hamiltonian_bundle_from_checkpoint,
        _neuron,
    )

    neu = _neuron(adata_neu)
    _ensure_time_column(neu)
    bundle = _load_hamiltonian_bundle_from_checkpoint(checkpoint, device=device)
    if bundle is None:
        raise RuntimeError("Could not load Hamiltonian bundle")

    key = "X_latent"
    z_all = np.asarray(neu.obsm[key], dtype=float)
    # align scores to neu index
    # scores were computed on adata_neu before _neuron subset — recompute on neu
    from analysis_protocol_utils import module_score

    local_scores = {k: module_score(neu, genes) for k, genes in MODULES.items()}

    cond = neu.obs["condition"].astype(str).values
    mask = cond == seed_condition
    if mask.sum() < 5:
        raise ValueError(f"Too few cells for {seed_condition}")

    idx = np.where(mask)[0]
    rank_score = local_scores[seed_by][idx]
    seeds = idx[np.argsort(-np.nan_to_num(rank_score))][: min(n_seeds, len(idx))]
    z0 = z_all[seeds].copy()
    if latent_shift_direction is not None and latent_shift_scale:
        d = np.asarray(latent_shift_direction, dtype=float).ravel()
        nrm = np.linalg.norm(d)
        if nrm > 1e-8:
            d = d / nrm
        z0 = z0 + float(latent_shift_scale) * d[None, :]

    t0 = float(neu.obs["time"].astype(float).values[seeds].mean())
    z_t = torch.tensor(z0, dtype=torch.float32, device=device)
    t_in = torch.full((z_t.shape[0], 1), t0, dtype=torch.float32, device=device)
    with torch.no_grad():
        p0 = bundle.initial_momentsums(z_t, t_in) if False else bundle.initial_momentum(z_t, t_in)
    ts = torch.linspace(t0, t1, steps=9, device=device)
    with torch.enable_grad():
        traj, _ = integrate_hamiltonian_flow(
            bundle.flow_func, z_t, p0, ts, dt=0.05, add_noise=False, detach_potential=True
        )
    traj_np = traj.detach().cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=5).fit(z_all)

    rows = []
    for ti, tval in enumerate(ts.detach().cpu().numpy()):
        _, nn = nbrs.kneighbors(traj_np[ti])
        flat = nn.ravel()
        row = {"t": float(tval)}
        for name, arr in local_scores.items():
            row[name] = float(np.nanmean(arr[flat]))
        rows.append(row)
    return pd.DataFrame(rows), seeds


def run_atf3_p0(device: str = None) -> Dict:
    from methods_model_utils import (
        latent_delta_from_knockdown,
        load_training_stack,
        reencode_latent,
    )
    from plot_utils import PALETTE, configure_headless, style_axis
    from run_gse155622_analysis import _ensure_time_column, _neuron
    from run_in_silico_knockout import _resolve_ko_direction

    configure_headless()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Atf3] loading WT stack on {device}...", flush=True)
    model, adata, config = load_training_stack(
        "GSE155622", CKPT_PAIN, device=device, max_cells=8000, force_genes=["Atf3"]
    )
    reencode_latent(model, adata, config, device=device)
    neu = _neuron(adata)
    _ensure_time_column(neu)
    cond = neu.obs["condition"].astype(str).values
    seed_mask_full = adata.obs_names.isin(neu.obs_names[cond == "SNI 24h"])

    shift_direction, resolved, direction_tag = _resolve_ko_direction(
        model,
        adata,
        config,
        ["Atf3"],
        ko_mode="hybrid",
        seed_mask=np.asarray(seed_mask_full, dtype=bool),
        expr_factor=0.0,
    )
    shift_scale = 1.0 if shift_direction is not None else 0.0
    print(f"[Atf3] direction={direction_tag} scale={shift_scale} resolved={resolved}", flush=True)

    print("[Atf3] WT rollout...", flush=True)
    wt, _ = rollout_multi_module(adata, CKPT_PAIN, {}, device=device)
    wt["condition"] = "WT"

    print("[Atf3] loading KO stack...", flush=True)
    _, adata_kd, _ = load_training_stack(
        "GSE155622",
        CKPT_PAIN,
        device=device,
        max_cells=8000,
        knockdown_genes=["Atf3"],
        knockdown_factor=0.0,
        force_genes=["Atf3"],
    )
    reencode_latent(model, adata_kd, config, device=device)
    print("[Atf3] KO rollout...", flush=True)
    kd, _ = rollout_multi_module(
        adata_kd,
        CKPT_PAIN,
        {},
        device=device,
        latent_shift_direction=shift_direction,
        latent_shift_scale=shift_scale,
    )
    kd["condition"] = "Atf3_KO"

    both = pd.concat([wt, kd], ignore_index=True)
    both.to_csv(OUT / "Atf3_multi_module_track.csv", index=False)

    # End-point summaries + drift on noAtf3 modules
    rows = []
    for mod in MODULES:
        s_wt = float(wt.iloc[-1][mod])
        s_ko = float(kd.iloc[-1][mod])
        # drift: abs change of module over trajectory
        d_wt = float(wt.iloc[-1][mod] - wt.iloc[0][mod])
        d_ko = float(kd.iloc[-1][mod] - kd.iloc[0][mod])
        # bootstrap on end values along timepoints as pseudo-samples
        p_reduce = _bootstrap_less(wt[mod].values, kd[mod].values, n_boot=500)
        rows.append(
            {
                "module": mod,
                "genes": ",".join(MODULES[mod]),
                "contains_Atf3": "Atf3" in MODULES[mod],
                "end_WT": s_wt,
                "end_KO": s_ko,
                "end_ratio_KO_over_WT": s_ko / s_wt if abs(s_wt) > 1e-8 else np.nan,
                "reduces_end_lt_70pct": bool(s_ko < 0.7 * s_wt),
                "delta_traj_WT": d_wt,
                "delta_traj_KO": d_ko,
                "blocks_traj_change": bool(abs(d_ko) < abs(d_wt) * 0.5),
                "bootstrap_KO_end_lt_WT_p": p_reduce,
                "pass_primary": bool(
                    (s_ko < 0.7 * s_wt) or (abs(d_ko) < abs(d_wt) * 0.5 and abs(d_wt) > 1e-4)
                ),
            }
        )
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(OUT / "Atf3_module_robustness_stats.csv", index=False)

    # Verdict focusing on noAtf3 modules
    no_atf = stats_df[~stats_df["contains_Atf3"] & (stats_df["module"] != "Atf3_alone")]
    verdict = {
        "full_SNIIC1_pass": bool(stats_df.loc[stats_df.module == "SNIIC1_full", "pass_primary"].iloc[0]),
        "noAtf3_SNIIC1_pass": bool(stats_df.loc[stats_df.module == "SNIIC1_noAtf3", "pass_primary"].iloc[0]),
        "noAtf3_SNIIC2_pass": bool(stats_df.loc[stats_df.module == "SNIIC2_noAtf3", "pass_primary"].iloc[0]),
        "noAtf3_SNIIC3_pass": bool(stats_df.loc[stats_df.module == "SNIIC3_noAtf3", "pass_primary"].iloc[0]),
        "Atf3_alone_drops": bool(stats_df.loc[stats_df.module == "Atf3_alone", "reduces_end_lt_70pct"].iloc[0]),
        "any_noAtf3_partner_pass": bool(no_atf["pass_primary"].any()),
        "all_noAtf3_partner_pass": bool(no_atf["pass_primary"].all()),
        "direction_tag": direction_tag,
        "note": "Primary robustness = Atf3-removed partner modules still reduced or traj change blocked.",
    }
    (OUT / "Atf3_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5), sharex=True)
    pairs = [
        ("SNIIC1_full", "SNIIC1_noAtf3"),
        ("SNIIC2_full", "SNIIC2_noAtf3"),
        ("SNIIC3_full", "SNIIC3_noAtf3"),
        ("Atf3_alone", None),
    ]
    for ax, (a, b) in zip(axes.ravel(), pairs):
        for df, lab, ls in ((wt, "WT", "-"), (kd, "KO", "--")):
            ax.plot(df["t"], df[a], ls + "o", color=PALETTE[0] if lab == "WT" else PALETTE[5], lw=2, label=f"{lab} {a}")
            if b:
                ax.plot(
                    df["t"],
                    df[b],
                    ls + "s",
                    color=PALETTE[2] if lab == "WT" else PALETTE[4],
                    lw=1.6,
                    alpha=0.9,
                    label=f"{lab} {b}",
                )
        ax.set_title(a.replace("_", " "))
        ax.set_ylabel("NN module score")
        style_axis(ax, grid_axis="y")
        ax.legend(fontsize=6, ncol=2)
    axes[-1, 0].set_xlabel("Simulated time")
    axes[-1, 1].set_xlabel("Simulated time")
    fig.suptitle("Atf3 KO: full vs Atf3-removed SNIIC modules", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "Atf3_full_vs_noAtf3_tracks.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(stats_df.to_string(index=False), flush=True)
    print(json.dumps(verdict, indent=2), flush=True)
    return {"stats": stats_df, "verdict": verdict}


def _u0_from_z(model, z: np.ndarray, device: str, bs: int = 256) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, len(z), bs):
            end = min(start + bs, len(z))
            zt = torch.tensor(z[start:end], dtype=torch.float32, device=device)
            out.append(model.stationary_potential(zt).squeeze(-1).cpu().numpy())
    return np.concatenate(out)


def run_sod2_p0(device: str = None, n_random: int = 20) -> Dict:
    from methods_model_utils import latent_delta_from_knockdown, load_training_stack, reencode_latent
    from plot_utils import PALETTE, configure_headless, style_axis
    from run_in_silico_knockout import _resolve_ko_direction

    configure_headless()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SOD2] loading HGSOC on {device}...", flush=True)
    # Keep enough cells to retain valley barcodes
    model, adata, config = load_training_stack(
        "HGSOC", CKPT_HG, device=device, max_cells=None, force_genes=["SOD2"]
    )
    reencode_latent(model, adata, config, device=device)
    z = np.asarray(adata.obsm["X_latent"], dtype=float)
    pot = _u0_from_z(model, z, device, bs=int(config.batch_size))
    adata.obs["potential_stationary_model"] = pot

    # Valley barcodes
    barcode_path = CKPT_HG / "analysis_protocol_HGSOC" / "eoc_attractor_deep_valley_barcodes.csv"
    barcodes = pd.read_csv(barcode_path)["barcode"].astype(str).tolist()
    valley = adata.obs_names.isin(barcodes)
    eoc = (
        adata.obs["annotation"].astype(str) == "EOC"
        if "annotation" in adata.obs.columns
        else np.ones(adata.n_obs, dtype=bool)
    )
    if int(valley.sum()) < 50:
        # fallback quantile on EOC
        u_eoc = pot.copy()
        u_eoc[~np.asarray(eoc)] = np.nan
        cut15 = np.nanquantile(u_eoc, 0.15)
        valley = np.asarray(eoc) & (pot <= cut15)
    idx = np.where(valley)[0]
    print(f"[SOD2] valley cells in load: {len(idx)}", flush=True)

    # Protocol-style cutoff = 15% EOC quantile (as in original)
    u_eoc = pot[np.asarray(eoc)]
    quantiles = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    cuts = {q: float(np.nanquantile(u_eoc, q)) for q in quantiles}

    def eval_shift(direction: Optional[np.ndarray], scale: float = 1.0) -> Dict:
        z2 = z.copy()
        if direction is not None and scale:
            d = np.asarray(direction, dtype=float).ravel()
            nrm = np.linalg.norm(d)
            if nrm > 1e-8:
                d = d / nrm
            z2 = z2 + float(scale) * d[None, :]
        u2 = _u0_from_z(model, z2, device, bs=int(config.batch_size))
        u_wt_v = pot[idx]
        u_kd_v = u2[idx]
        du = u_kd_v - u_wt_v
        mean_du = float(np.nanmean(du))
        try:
            _, wil_p = stats.wilcoxon(u_kd_v, u_wt_v, alternative="greater", zero_method="wilcox")
            wil_p = float(wil_p)
        except Exception:
            wil_p = float("nan")
        rng = np.random.default_rng(0)
        boot = np.array(
            [float(np.mean(rng.choice(du, size=len(du), replace=True))) for _ in range(500)]
        )
        boot_p = float((np.sum(boot <= 0) + 1) / (len(boot) + 1))
        esc = {f"frac_escape_q{int(q*100):02d}": float(np.mean(u_kd_v > cuts[q])) for q in quantiles}
        return {
            "mean_delta_U": mean_du,
            "wilcoxon_p": wil_p,
            "bootstrap_p": boot_p,
            "mean_U_WT": float(np.nanmean(u_wt_v)),
            "mean_U_pert": float(np.nanmean(u_kd_v)),
            **esc,
        }

    # SOD2 hybrid direction (same as protocol)
    shift_direction, resolved, direction_tag = _resolve_ko_direction(
        model, adata, config, ["SOD2"], ko_mode="hybrid", seed_mask=valley, expr_factor=0.0
    )
    print(f"[SOD2] direction={direction_tag} resolved={resolved}", flush=True)
    # Also apply expression KO reencode for SOD2 primary (closer to paper)
    print("[SOD2] expression KO reencode...", flush=True)
    _, adata_kd, _ = load_training_stack(
        "HGSOC",
        CKPT_HG,
        device=device,
        max_cells=None,
        knockdown_genes=["SOD2"],
        knockdown_factor=0.0,
        force_genes=["SOD2"],
    )
    # Align barcodes
    adata_kd.obs_names = adata_kd.obs_names.astype(str)
    common = adata.obs_names.intersection(adata_kd.obs_names)
    # Map valley onto common
    # Simpler: use shift-only path for sensitivity; plus one full reencode+shift for SOD2 matching paper
    reencode_latent(model, adata_kd, config, device=device)
    z_kd = np.asarray(adata_kd.obsm["X_latent"], dtype=float)
    if shift_direction is not None:
        d = np.asarray(shift_direction, dtype=float).ravel()
        nrm = np.linalg.norm(d)
        if nrm > 1e-8:
            d = d / nrm
        z_kd = z_kd + 1.0 * d[None, :]
    # Align indices by obs_names
    kd_names = list(adata_kd.obs_names)
    name_to_i = {n: i for i, n in enumerate(kd_names)}
    valley_names = adata.obs_names[idx]
    kd_idx = np.array([name_to_i[n] for n in valley_names if n in name_to_i])
    wt_idx_aligned = np.array([j for j, n in enumerate(valley_names) if n in name_to_i])
    u_kd_full = _u0_from_z(model, z_kd, device, bs=int(config.batch_size))
    u_wt_v = pot[idx][wt_idx_aligned]
    u_kd_v = u_kd_full[kd_idx]

    sod2_rows = []
    for q, cut in cuts.items():
        frac = float(np.mean(u_kd_v > cut))
        sod2_rows.append(
            {
                "gene": "SOD2",
                "protocol": "hybrid_reencode_shift",
                "quantile": q,
                "cutoff": cut,
                "frac_escape": frac,
                "mean_delta_U": float(np.nanmean(u_kd_v - u_wt_v)),
                "n_valley": int(len(u_kd_v)),
            }
        )
    sod2_thr = pd.DataFrame(sod2_rows)
    sod2_thr.to_csv(OUT / "SOD2_threshold_sensitivity.csv", index=False)

    try:
        _, wil_p = stats.wilcoxon(u_kd_v, u_wt_v, alternative="greater", zero_method="wilcox")
        wil_p = float(wil_p)
    except Exception:
        wil_p = float("nan")
    du = u_kd_v - u_wt_v
    rng = np.random.default_rng(0)
    boot = np.array([float(np.mean(rng.choice(du, size=len(du), replace=True))) for _ in range(500)])
    boot_p = float((np.sum(boot <= 0) + 1) / (len(boot) + 1))

    sod2_primary = {
        "gene": "SOD2",
        "n_valley": int(len(u_kd_v)),
        "mean_delta_U": float(np.nanmean(du)),
        "frac_escape_q15": float(np.mean(u_kd_v > cuts[0.15])),
        "wilcoxon_p": wil_p,
        "bootstrap_p": boot_p,
        "direction_tag": direction_tag,
    }
    pd.DataFrame([sod2_primary]).to_csv(OUT / "SOD2_primary_stats.csv", index=False)

    # Random gene controls via latent_delta only (fast)
    print(f"[SOD2] random gene nulls n={n_random}...", flush=True)
    genes = list(adata.var_names)
    # exclude SOD2 and very sparse genes
    rng = np.random.default_rng(42)
    candidates = [g for g in genes if g.upper() != "SOD2"]
    # prefer genes with some expression
    pick = rng.choice(candidates, size=min(n_random * 3, len(candidates)), replace=False)
    rand_rows = []
    used = []
    for g in pick:
        if len(used) >= n_random:
            break
        try:
            direction, resolved = latent_delta_from_knockdown(
                model, adata, config, [g], factor=0.0, seed_mask=valley, max_cells=2000
            )
        except Exception:
            continue
        if direction is None or not resolved:
            continue
        ev = eval_shift(direction, scale=1.0)
        rand_rows.append({"gene": resolved[0], "protocol": "latent_delta_shift", **ev})
        used.append(resolved[0])
        print(f"  null {resolved[0]}: dU={ev['mean_delta_U']:.5f} esc_q15={ev['frac_escape_q15']:.3f}", flush=True)

    # Also evaluate SOD2 with same latent_delta_shift protocol for fair null comparison
    sod2_delta, _ = latent_delta_from_knockdown(
        model, adata, config, ["SOD2"], factor=0.0, seed_mask=valley, max_cells=2000
    )
    sod2_shift_only = eval_shift(sod2_delta, scale=1.0)
    sod2_shift_only_row = {"gene": "SOD2", "protocol": "latent_delta_shift", **sod2_shift_only}

    null_df = pd.DataFrame(rand_rows)
    null_df.to_csv(OUT / "SOD2_random_gene_nulls.csv", index=False)
    compare = pd.concat([pd.DataFrame([sod2_shift_only_row]), null_df], ignore_index=True)
    compare.to_csv(OUT / "SOD2_vs_random_shift_protocol.csv", index=False)

    # Empirical p: fraction of random genes with >= SOD2 escape or deltaU
    if len(null_df):
        emp_p_esc = float(np.mean(null_df["frac_escape_q15"] >= sod2_shift_only["frac_escape_q15"]))
        emp_p_du = float(np.mean(null_df["mean_delta_U"] >= sod2_shift_only["mean_delta_U"]))
    else:
        emp_p_esc = emp_p_du = float("nan")

    # Threshold robustness: does escape stay high across quantiles?
    esc_by_q = {float(r.quantile): float(r.frac_escape) for r in sod2_thr.itertuples()}
    thr_ok = bool(esc_by_q.get(0.15, 0) >= 0.5 and esc_by_q.get(0.10, 0) >= 0.3)

    verdict = {
        "sod2_primary_hybrid": sod2_primary,
        "sod2_shift_only": sod2_shift_only_row,
        "escape_by_quantile": esc_by_q,
        "threshold_robust_rule": "q15 escape>=0.5 AND q10 escape>=0.3",
        "threshold_robust_pass": thr_ok,
        "n_random_genes": int(len(null_df)),
        "empirical_p_escape_vs_random": emp_p_esc,
        "empirical_p_deltaU_vs_random": emp_p_du,
        "beats_random_escape": bool(np.isfinite(emp_p_esc) and emp_p_esc <= 0.1),
        "beats_random_deltaU": bool(np.isfinite(emp_p_du) and emp_p_du <= 0.1),
        "overall_pass": bool(
            thr_ok
            and sod2_primary["mean_delta_U"] > 0
            and (
                (np.isfinite(emp_p_esc) and emp_p_esc <= 0.15)
                or sod2_primary["frac_escape_q15"] >= 0.7
            )
        ),
    }
    (OUT / "SOD2_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    qs = sorted(esc_by_q)
    axes[0].plot(qs, [esc_by_q[q] for q in qs], "-o", color=PALETTE[5], lw=2)
    axes[0].axhline(0.788, color="gray", ls=":", label="reported 78.8% @ protocol")
    axes[0].set_xlabel("EOC U0 quantile cutoff")
    axes[0].set_ylabel("Frac valley cells above cutoff")
    axes[0].set_title("SOD2 escape vs cutoff")
    axes[0].legend(fontsize=8)
    style_axis(axes[0], grid_axis="both")

    if len(null_df):
        axes[1].hist(null_df["frac_escape_q15"], bins=12, color=PALETTE[0], alpha=0.7, label="random genes")
        axes[1].axvline(
            sod2_shift_only["frac_escape_q15"],
            color=PALETTE[5],
            lw=2,
            label=f"SOD2 shift={sod2_shift_only['frac_escape_q15']:.2f}",
        )
        axes[1].axvline(
            sod2_primary["frac_escape_q15"],
            color=PALETTE[2],
            lw=2,
            ls="--",
            label=f"SOD2 hybrid={sod2_primary['frac_escape_q15']:.2f}",
        )
        axes[1].set_xlabel("frac_escape @ q15")
        axes[1].set_ylabel("Count")
        axes[1].set_title("SOD2 vs random-gene null (shift protocol)")
        axes[1].legend(fontsize=7)
        style_axis(axes[1], grid_axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "SOD2_sensitivity_plots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(sod2_thr.to_string(index=False), flush=True)
    print(json.dumps(verdict, indent=2, default=str), flush=True)
    return verdict


def write_report(atf3, sod2):
    stats = atf3["stats"]
    v1 = atf3["verdict"]
    lines = [
        "# P0 稳健性校验报告",
        "",
        "## 1. Atf3 KO：去 Atf3 的 SNIIC 模块",
        "",
        stats.to_markdown(index=False),
        "",
        "### 判决",
        f"- Full SNIIC1 pass: **{v1['full_SNIIC1_pass']}**",
        f"- SNIIC1_noAtf3 pass: **{v1['noAtf3_SNIIC1_pass']}**",
        f"- SNIIC2_noAtf3 (Mrgprd) pass: **{v1['noAtf3_SNIIC2_pass']}**",
        f"- SNIIC3_noAtf3 pass: **{v1['noAtf3_SNIIC3_pass']}**",
        f"- Atf3 alone drops: **{v1['Atf3_alone_drops']}**",
        f"- **任一** noAtf3 partner pass: **{v1['any_noAtf3_partner_pass']}**",
        f"- **全部** noAtf3 partner pass: **{v1['all_noAtf3_partner_pass']}**",
        "",
        "图：`Atf3_full_vs_noAtf3_tracks.png`",
        "",
        "## 2. SOD2：阈值敏感性 + 随机基因对照",
        "",
    ]
    thr = pd.read_csv(OUT / "SOD2_threshold_sensitivity.csv")
    lines.append(thr.to_markdown(index=False))
    lines.append("")
    lines.append("### 判决")
    lines.append(f"- Hybrid 主结果 frac_escape@q15: **{sod2['sod2_primary_hybrid']['frac_escape_q15']:.3f}**")
    lines.append(f"- mean ΔU: **{sod2['sod2_primary_hybrid']['mean_delta_U']:.5f}**")
    lines.append(f"- 阈值稳健 (q15≥0.5 & q10≥0.3): **{sod2['threshold_robust_pass']}**")
    lines.append(f"- 随机基因 null n={sod2['n_random_genes']}")
    lines.append(f"- empirical p (escape vs random, shift协议): **{sod2['empirical_p_escape_vs_random']}**")
    lines.append(f"- empirical p (ΔU vs random): **{sod2['empirical_p_deltaU_vs_random']}**")
    lines.append(f"- **overall_pass: {sod2['overall_pass']}**")
    lines.append("")
    lines.append("图：`SOD2_sensitivity_plots.png`")
    lines.append("")
    lines.append("## 总建议（写作）")
    if v1.get("any_noAtf3_partner_pass") and not v1.get("all_noAtf3_partner_pass"):
        lines.append(
            "- Atf3：部分 partner 模块仍支持；**全文应并列报告 full 与 noAtf3 读出**，避免只强调 SNIIC1→0。"
        )
    elif v1.get("all_noAtf3_partner_pass"):
        lines.append("- Atf3：去 Atf3 模块仍通过 → 旗舰结论可保留，建议补充 noAtf3 图。")
    else:
        lines.append("- Atf3：去 Atf3 后不稳 → **必须降调**，主读出改为 partner 基因/轨迹几何，而非模块归零。")
    if sod2.get("overall_pass"):
        lines.append("- SOD2：阈值稳健且相对随机基因占优（或高逃逸率）→ 可保留，但持续报告小 ΔU + 多阈值。")
    else:
        lines.append("- SOD2：敏感性未过关 → 逃逸率主张需降调，强调 ΔU 分布与阈值依赖。")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    atf3 = run_atf3_p0()
    sod2 = run_sod2_p0(n_random=20)
    write_report(atf3, sod2)
    print("Wrote", OUT / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()
