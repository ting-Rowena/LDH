#!/usr/bin/env python
"""P2: Atf3 Hamiltonian KO Neuron vs Fibroblast (FB_remodel readout) + type-balanced U0 ranking."""

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
OUT = ROOT / "output_file" / "robustness" / "p2_robustness"
OUT.mkdir(parents=True, exist_ok=True)

CKPT_PAIN = (
    ROOT
    / "GSE155622_checkpoints_3000_3000_384_0.05_recon0.01_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
)
CKPT_LUNG = (
    ROOT
    / "GSE141259_checkpoints_5000_5000_512_0.01_recon0.1_valD28_timeX_lossnorm_qp_d0p01_z0p1_k0p2"
)
CKPT_HG = (
    ROOT
    / "HGSOC_checkpoints_3000_3000_512_0.05_recon0.01_nactpair_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
)

# Neuron pathology modules (Atf3-removed partners preferred for verdict)
SNIIC1_NO = ("Gfra3", "Gal")
SNIIC2_NO = ("Mrgprd",)
SNIIC3_NO = ("S100b", "Gal")
SNIIC2_FULL = ("Atf3", "Mrgprd")

# Fibroblast non-SNIIC readouts
FB_REMODEL = (
    "Acta2", "Tagln", "Col1a1", "Col1a2", "Col3a1", "Pdgfra", "Pdgfrb",
    "Fn1", "Ctgf", "Postn", "Timp1", "Mmp2", "Fap", "Thy1",
)
FB_ECM = ("Col1a1", "Col1a2", "Col3a1", "Fn1", "Postn")
FB_CONTRACTILE = ("Acta2", "Tagln", "Myh11", "Cnn1")

FORCE_GENES = list(
    dict.fromkeys(
        ["Atf3"]
        + list(SNIIC1_NO)
        + list(SNIIC2_NO)
        + list(SNIIC3_NO)
        + list(FB_REMODEL)
        + list(FB_CONTRACTILE)
    )
)


def _type_col(adata) -> str:
    return "celltype" if "celltype" in adata.obs else "annotation"


def _merge_obs(adata, ckpt: Path, cols: Sequence[str]):
    obs = pd.read_csv(ckpt / "obs.csv", index_col=0, low_memory=False)
    obs.index = obs.index.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    common = adata.obs_names.intersection(obs.index)
    adata = adata[common].copy()
    for c in cols:
        if c in obs.columns:
            adata.obs[c] = obs.loc[common, c].values
    return adata


def _module_scores(adata, modules: Dict[str, Sequence[str]]) -> Dict[str, np.ndarray]:
    from analysis_protocol_utils import module_score

    return {k: module_score(adata, genes) for k, genes in modules.items()}


def _bootstrap_end_reduce(wt_end: float, ko_end: float, wt_track: np.ndarray, ko_track: np.ndarray, n_boot=500, seed=0) -> float:
    """Empirical p that KO end < WT end using trajectory timepoints as pseudo-replicates."""
    rng = np.random.default_rng(seed)
    obs = float(ko_end - wt_end)
    pooled = np.concatenate([wt_track, ko_track])
    n_w, n_k = len(wt_track), len(ko_track)
    count = 0
    for _ in range(n_boot):
        samp = rng.choice(pooled, size=n_w + n_k, replace=True)
        if float(np.mean(samp[n_w:]) - np.mean(samp[:n_w])) <= obs:
            count += 1
    return float((count + 1) / (n_boot + 1))


def rollout_type_modules(
    adata,
    checkpoint: Path,
    modules: Dict[str, Sequence[str]],
    *,
    cell_type: str,
    seed_condition: str = "SNI 24h",
    seed_by: str,
    t1: float = 2.0,
    n_seeds: int = 40,
    device: str = "cpu",
    latent_shift_direction: Optional[np.ndarray] = None,
    latent_shift_scale: float = 0.0,
    nn_k: int = 5,
) -> Tuple[pd.DataFrame, Dict]:
    from hamiltonian_flow import integrate_hamiltonian_flow
    from run_gse155622_analysis import _ensure_time_column, _load_hamiltonian_bundle_from_checkpoint

    col = _type_col(adata)
    sub = adata[adata.obs[col].astype(str) == cell_type].copy()
    if sub.n_obs < 30:
        raise ValueError(f"Too few {cell_type} cells: {sub.n_obs}")
    _ensure_time_column(sub)
    bundle = _load_hamiltonian_bundle_from_checkpoint(checkpoint, device=device)
    if bundle is None:
        raise RuntimeError("Could not load Hamiltonian bundle")

    key = "X_latent"
    if key not in sub.obsm:
        raise KeyError(f"{key} missing; call reencode_latent first")
    z_all = np.asarray(sub.obsm[key], dtype=float)
    scores = _module_scores(sub, modules)
    cond = sub.obs["condition"].astype(str).values
    mask = cond == seed_condition
    if int(np.sum(mask)) < 5:
        # fallback: any SNI condition
        mask = np.array(["SNI" in str(c) for c in cond], dtype=bool)
    if int(np.sum(mask)) < 5:
        raise ValueError(f"Too few seed cells for {cell_type}/{seed_condition}: {int(np.sum(mask))}")

    idx = np.where(mask)[0]
    rank = scores[seed_by][idx]
    seeds = idx[np.argsort(-np.nan_to_num(rank))][: min(n_seeds, len(idx))]
    z0 = z_all[seeds].copy()
    if latent_shift_direction is not None and latent_shift_scale:
        d = np.asarray(latent_shift_direction, dtype=float).ravel()
        nrm = np.linalg.norm(d)
        if nrm > 1e-8:
            d = d / nrm
        z0 = z0 + float(latent_shift_scale) * d[None, :]

    t0 = float(sub.obs["time"].astype(float).values[seeds].mean())
    z_t = torch.tensor(z0, dtype=torch.float32, device=device)
    t_in = torch.full((z_t.shape[0], 1), t0, dtype=torch.float32, device=device)
    with torch.no_grad():
        p0 = bundle.initial_momentum(z_t, t_in)
    ts = torch.linspace(t0, t1, steps=9, device=device)
    with torch.enable_grad():
        traj, _ = integrate_hamiltonian_flow(
            bundle.flow_func, z_t, p0, ts, dt=0.05, add_noise=False, detach_potential=True
        )
    traj_np = traj.detach().cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=nn_k).fit(z_all)
    rows = []
    for ti, tval in enumerate(ts.detach().cpu().numpy()):
        _, nn = nbrs.kneighbors(traj_np[ti])
        flat = nn.ravel()
        row = {"t": float(tval), "cell_type": cell_type}
        for name, arr in scores.items():
            row[name] = float(np.nanmean(arr[flat]))
        rows.append(row)
    meta = {
        "cell_type": cell_type,
        "n_type": int(sub.n_obs),
        "n_seed_pool": int(np.sum(mask)),
        "n_seeds": int(len(seeds)),
        "seed_condition": seed_condition,
        "seed_by": seed_by,
        "t0": t0,
        "t1": t1,
    }
    return pd.DataFrame(rows), meta


def run_atf3_cross_type(device: str = None) -> Dict:
    from methods_model_utils import load_training_stack, reencode_latent
    from plot_utils import PALETTE, configure_headless, style_axis
    from run_in_silico_knockout import _resolve_ko_direction

    configure_headless()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P2 Atf3] loading WT stack on {device}...", flush=True)

    neuron_modules = {
        "SNIIC1_noAtf3": SNIIC1_NO,
        "SNIIC2_noAtf3": SNIIC2_NO,
        "SNIIC2_full": SNIIC2_FULL,
        "SNIIC3_noAtf3": SNIIC3_NO,
        "Atf3_alone": ("Atf3",),
        "FB_remodel": FB_REMODEL,  # negative-control readout on neurons
    }
    fibro_modules = {
        "FB_remodel": FB_REMODEL,
        "FB_ECM": FB_ECM,
        "FB_contractile": FB_CONTRACTILE,
        "SNIIC2_full": SNIIC2_FULL,  # invalid cross-type module (contains Atf3)
        "SNIIC2_noAtf3": SNIIC2_NO,
        "Atf3_alone": ("Atf3",),
    }

    # Stratified subsample: keep Neuron + Fibroblast priority, then fill
    model, adata, config = load_training_stack(
        "GSE155622", CKPT_PAIN, device=device, max_cells=None, force_genes=FORCE_GENES
    )
    adata = _merge_obs(adata, CKPT_PAIN, ["annotation", "condition", "potential_stationary", "time"])
    col = _type_col(adata)
    # Cap memory: keep all Neuron+Fibroblast, subsample others to total ~12k
    keep_types = {"Neuron", "Fibroblast"}
    is_keep = adata.obs[col].astype(str).isin(keep_types).values
    other_idx = np.where(~is_keep)[0]
    target_other = max(0, 12000 - int(is_keep.sum()))
    if len(other_idx) > target_other:
        rng = np.random.default_rng(42)
        other_idx = rng.choice(other_idx, size=target_other, replace=False)
    keep_idx = np.concatenate([np.where(is_keep)[0], other_idx])
    adata = adata[keep_idx].copy()
    print(f"[P2 Atf3] stratified n={adata.n_obs}; counts=\n{adata.obs[col].astype(str).value_counts()}", flush=True)

    reencode_latent(model, adata, config, device=device)

    # KO direction from Neuron SNI 24h seeds (primary context)
    neu_mask = (adata.obs[col].astype(str) == "Neuron").values & (
        adata.obs["condition"].astype(str).values == "SNI 24h"
    )
    shift_direction, resolved, direction_tag = _resolve_ko_direction(
        model, adata, config, ["Atf3"], ko_mode="hybrid", seed_mask=neu_mask, expr_factor=0.0
    )
    # Also type-specific Fibroblast direction for secondary arm
    fb_mask = (adata.obs[col].astype(str) == "Fibroblast").values & (
        adata.obs["condition"].astype(str).values == "SNI 24h"
    )
    fb_dir, _, fb_dir_tag = _resolve_ko_direction(
        model, adata, config, ["Atf3"], ko_mode="hybrid", seed_mask=fb_mask, expr_factor=0.0
    )
    shift_scale = 1.0 if shift_direction is not None else 0.0
    print(f"[P2 Atf3] neuron_dir={direction_tag} fibro_dir={fb_dir_tag} scale={shift_scale}", flush=True)

    print("[P2 Atf3] WT rollouts...", flush=True)
    wt_neu, meta_neu = rollout_type_modules(
        adata, CKPT_PAIN, neuron_modules, cell_type="Neuron", seed_by="SNIIC2_full", device=device
    )
    wt_neu["condition"] = "WT"
    wt_fb, meta_fb = rollout_type_modules(
        adata, CKPT_PAIN, fibro_modules, cell_type="Fibroblast", seed_by="FB_remodel", device=device
    )
    wt_fb["condition"] = "WT"

    print("[P2 Atf3] loading KO expression + reencode...", flush=True)
    wt_barcodes = adata.obs_names.astype(str).tolist()
    _, adata_kd, _ = load_training_stack(
        "GSE155622",
        CKPT_PAIN,
        device=device,
        max_cells=None,
        knockdown_genes=["Atf3"],
        knockdown_factor=0.0,
        force_genes=FORCE_GENES,
    )
    adata_kd = _merge_obs(adata_kd, CKPT_PAIN, ["annotation", "condition", "potential_stationary", "time"])
    adata_kd.obs_names = adata_kd.obs_names.astype(str)
    missing = [b for b in wt_barcodes if b not in set(adata_kd.obs_names)]
    if missing:
        raise RuntimeError(f"KO stack missing {len(missing)} WT barcodes (e.g. {missing[:3]})")
    adata_kd = adata_kd[wt_barcodes].copy()
    reencode_latent(model, adata_kd, config, device=device)

    print("[P2 Atf3] KO rollouts...", flush=True)
    ko_neu, _ = rollout_type_modules(
        adata_kd,
        CKPT_PAIN,
        neuron_modules,
        cell_type="Neuron",
        seed_by="SNIIC2_full",
        device=device,
        latent_shift_direction=shift_direction,
        latent_shift_scale=shift_scale,
    )
    ko_neu["condition"] = "Atf3_KO"
    ko_fb, _ = rollout_type_modules(
        adata_kd,
        CKPT_PAIN,
        fibro_modules,
        cell_type="Fibroblast",
        seed_by="FB_remodel",
        device=device,
        latent_shift_direction=fb_dir if fb_dir is not None else shift_direction,
        latent_shift_scale=shift_scale,
    )
    ko_fb["condition"] = "Atf3_KO"

    tracks = pd.concat([wt_neu, ko_neu, wt_fb, ko_fb], ignore_index=True)
    tracks.to_csv(OUT / "Atf3_cross_type_tracks.csv", index=False)

    def summarize(wt: pd.DataFrame, ko: pd.DataFrame, mods: Sequence[str], cell_type: str, primary: Sequence[str]) -> pd.DataFrame:
        rows = []
        for mod in mods:
            if mod not in wt.columns:
                continue
            e_wt, e_ko = float(wt.iloc[-1][mod]), float(ko.iloc[-1][mod])
            d_wt = float(wt.iloc[-1][mod] - wt.iloc[0][mod])
            d_ko = float(ko.iloc[-1][mod] - ko.iloc[0][mod])
            p = _bootstrap_end_reduce(e_wt, e_ko, wt[mod].values, ko[mod].values)
            reduces = bool(e_ko < 0.7 * e_wt) if abs(e_wt) > 1e-8 else False
            blocks = bool(abs(d_ko) < abs(d_wt) * 0.5) if abs(d_wt) > 1e-4 else False
            rows.append(
                {
                    "cell_type": cell_type,
                    "module": mod,
                    "is_primary_readout": mod in primary,
                    "end_WT": e_wt,
                    "end_KO": e_ko,
                    "end_ratio_KO_over_WT": e_ko / e_wt if abs(e_wt) > 1e-8 else np.nan,
                    "delta_traj_WT": d_wt,
                    "delta_traj_KO": d_ko,
                    "reduces_end_lt_70pct": reduces,
                    "blocks_traj_change": blocks,
                    "bootstrap_KO_end_lt_WT_p": p,
                    "pass_module": bool(reduces or blocks),
                }
            )
        return pd.DataFrame(rows)

    stats_neu = summarize(
        wt_neu, ko_neu, list(neuron_modules), "Neuron", primary=["SNIIC1_noAtf3", "SNIIC3_noAtf3", "SNIIC2_noAtf3"]
    )
    stats_fb = summarize(
        wt_fb, ko_fb, list(fibro_modules), "Fibroblast", primary=["FB_remodel", "FB_ECM", "FB_contractile"]
    )
    stats = pd.concat([stats_neu, stats_fb], ignore_index=True)
    stats.to_csv(OUT / "Atf3_cross_type_stats.csv", index=False)

    # Strict verdict: partner pass requires endpoint reduction (not mere traj damping)
    def _reduces(df, mod):
        m = df[df.module == mod]
        return bool(m["reduces_end_lt_70pct"].iloc[0]) if len(m) else False

    def _ratio(df, mod):
        m = df[df.module == mod]
        return float(m["end_ratio_KO_over_WT"].iloc[0]) if len(m) else float("nan")

    neu_partner_reduce = any(_reduces(stats_neu, m) for m in ["SNIIC1_noAtf3", "SNIIC3_noAtf3"])
    fb_primary = stats_fb[stats_fb["is_primary_readout"]]
    fb_any_hit = bool((fb_primary["end_ratio_KO_over_WT"] < 0.7).any()) if len(fb_primary) else False
    fb_not_collapsed = not fb_any_hit
    fb_frac_pass = float(fb_primary["reduces_end_lt_70pct"].mean()) if len(fb_primary) else np.nan

    verdict = {
        "direction_tag_neuron": direction_tag,
        "direction_tag_fibroblast": fb_dir_tag,
        "neuron_meta": meta_neu,
        "fibroblast_meta": meta_fb,
        "neuron_SNIIC1_noAtf3_ratio": _ratio(stats_neu, "SNIIC1_noAtf3"),
        "neuron_SNIIC2_noAtf3_ratio": _ratio(stats_neu, "SNIIC2_noAtf3"),
        "neuron_SNIIC3_noAtf3_ratio": _ratio(stats_neu, "SNIIC3_noAtf3"),
        "strict_neuron_partner_end_reduction": neu_partner_reduce,
        "fibroblast_FB_remodel_ratio": _ratio(stats_fb, "FB_remodel"),
        "fibroblast_FB_ECM_ratio": _ratio(stats_fb, "FB_ECM"),
        "fibroblast_FB_contractile_ratio": _ratio(stats_fb, "FB_contractile"),
        "fibroblast_primary_frac_end_reduction": fb_frac_pass,
        "fibroblast_FB_modules_not_collapsed": fb_not_collapsed,
        "specificity_ok": bool(neu_partner_reduce and fb_not_collapsed),
        "writing_rule": (
            "Claim Atf3 KO specificity only if Neuron Atf3-removed partners drop at endpoint AND "
            "Fibroblast FB_remodel/ECM/contractile do not collapse under the same protocol."
        ),
    }
    if verdict["specificity_ok"]:
        verdict["overall"] = "PASS_SPECIFICITY"
    elif (not neu_partner_reduce) and fb_any_hit:
        verdict["overall"] = "FAIL_SPECIFICITY_FB_HIT_NEURON_PARTNERS_WEAK"
    elif neu_partner_reduce and fb_any_hit:
        verdict["overall"] = "FAIL_SPECIFICITY_FB_ALSO_HIT"
    else:
        verdict["overall"] = "PARTIAL_OR_INCONCLUSIVE"
    (OUT / "Atf3_cross_type_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    # Figure
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.2), sharex="row")
    sniic_colors = ["#9EC1C0", "#E0BFB8", "#F0E4D2"]
    fb_colors = ["#5F8D4E", "#6E2C4B", "#B38B6D"]
    neu_plot = [
        ("SNIIC1_noAtf3", sniic_colors[0]),
        ("SNIIC2_noAtf3", sniic_colors[1]),
        ("SNIIC3_noAtf3", sniic_colors[2]),
    ]
    fb_plot = [
        ("FB_remodel", fb_colors[0]),
        ("FB_contractile", fb_colors[1]),
        ("FB_ECM", fb_colors[2]),
    ]
    for ax, (mod, color) in zip(axes[0], neu_plot):
        ax.plot(wt_neu["t"], wt_neu[mod], "-o", color=color, lw=2, label="WT")
        ax.plot(ko_neu["t"], ko_neu[mod], "--s", color=color, lw=2, label=r"$\mathit{Atf3}$-KO")
        ax.set_title(f"Neuron · {mod}")
        ax.set_ylabel("NN module score")
        style_axis(ax, grid_axis="y")
        ax.legend(
            fontsize=7,
            loc="center right",
            bbox_to_anchor=(0.98, 0.70),
            frameon=False,
        )
    for ax, (mod, color) in zip(axes[1], fb_plot):
        ax.plot(wt_fb["t"], wt_fb[mod], "-o", color=color, lw=2, label="WT")
        ax.plot(ko_fb["t"], ko_fb[mod], "--s", color=color, lw=2, label=r"$\mathit{Atf3}$-KO")
        ax.set_title(f"Fibroblast · {mod}")
        ax.set_ylabel("NN module score")
        ax.set_xlabel("Simulated time")
        style_axis(ax, grid_axis="y")
        ax.legend(
            fontsize=7,
            loc="center right",
            bbox_to_anchor=(0.98, 0.70),
            frameon=False,
        )
    fig.suptitle(r"$\mathit{Atf3}$-KO: Neuron (SNIIC partners) vs Fibroblast (FB remodel)", fontsize=12)
    fig.subplots_adjust(wspace=0.32, hspace=0.32, top=0.88)
    fig.savefig(OUT / "Atf3_cross_type_tracks.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(stats.to_string(index=False), flush=True)
    print(json.dumps(verdict, indent=2), flush=True)
    return {"stats": stats, "verdict": verdict, "tracks": tracks}


def _balanced_rank_one(
    u: np.ndarray,
    labels: np.ndarray,
    *,
    n_per_type: int,
    n_boot: int = 200,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Abundance-naive mean U0 via equal-n bootstrap."""
    rng = np.random.default_rng(seed)
    types = sorted(pd.unique(labels))
    # Raw (all cells)
    raw_rows = []
    for t in types:
        m = labels == t
        vals = u[m]
        raw_rows.append(
            {
                "cell_type": t,
                "n_cells": int(m.sum()),
                "mean_U0_raw": float(np.nanmean(vals)),
                "median_U0_raw": float(np.nanmedian(vals)),
            }
        )
    raw = pd.DataFrame(raw_rows)
    raw["rank_raw_high_to_low"] = raw["mean_U0_raw"].rank(ascending=False, method="min").astype(int)
    raw["rank_raw_deep_to_shallow"] = raw["mean_U0_raw"].rank(ascending=True, method="min").astype(int)

    usable = [t for t in types if (labels == t).sum() >= max(20, n_per_type)]
    boot_means = {t: [] for t in usable}
    for _ in range(n_boot):
        for t in usable:
            idx = np.where(labels == t)[0]
            take = rng.choice(idx, size=n_per_type, replace=False)
            boot_means[t].append(float(np.nanmean(u[take])))

    bal_rows = []
    for t in usable:
        arr = np.asarray(boot_means[t], dtype=float)
        bal_rows.append(
            {
                "cell_type": t,
                "n_per_type": n_per_type,
                "n_boot": n_boot,
                "mean_U0_balanced": float(np.mean(arr)),
                "std_U0_balanced": float(np.std(arr)),
                "ci05": float(np.quantile(arr, 0.05)),
                "ci95": float(np.quantile(arr, 0.95)),
            }
        )
    bal = pd.DataFrame(bal_rows)
    bal["rank_bal_high_to_low"] = bal["mean_U0_balanced"].rank(ascending=False, method="min").astype(int)
    bal["rank_bal_deep_to_shallow"] = bal["mean_U0_balanced"].rank(ascending=True, method="min").astype(int)

    # Merge
    out = raw.merge(bal, on="cell_type", how="left")
    # Spearman of ranks (among usable)
    both = out.dropna(subset=["rank_bal_deep_to_shallow"])
    if len(both) >= 3:
        rho, p = stats.spearmanr(both["rank_raw_deep_to_shallow"], both["rank_bal_deep_to_shallow"])
        rank_corr = {"spearman_raw_vs_balanced_rank": float(rho), "pvalue": float(p)}
    else:
        rank_corr = {"spearman_raw_vs_balanced_rank": float("nan"), "pvalue": float("nan")}
    # Top/bottom stability
    raw_deep = set(both.nsmallest(3, "mean_U0_raw")["cell_type"])
    bal_deep = set(both.nsmallest(3, "mean_U0_balanced")["cell_type"])
    raw_high = set(both.nlargest(3, "mean_U0_raw")["cell_type"])
    bal_high = set(both.nlargest(3, "mean_U0_balanced")["cell_type"])
    stability = {
        **rank_corr,
        "top3_deep_overlap": sorted(raw_deep & bal_deep),
        "top3_deep_jaccard": float(len(raw_deep & bal_deep) / max(len(raw_deep | bal_deep), 1)),
        "top3_high_overlap": sorted(raw_high & bal_high),
        "top3_high_jaccard": float(len(raw_high & bal_high) / max(len(raw_high | bal_high), 1)),
        "rank_changed": bool(
            list(both.sort_values("mean_U0_raw")["cell_type"])
            != list(both.sort_values("mean_U0_balanced")["cell_type"])
        ),
    }
    return out.sort_values("mean_U0_raw"), pd.DataFrame([stability])


def run_type_balanced_u0(n_boot: int = 300) -> Dict:
    from plot_utils import PALETTE, configure_headless, style_axis

    configure_headless()
    datasets = {
        "GSE155622": (CKPT_PAIN, "annotation"),
        "GSE141259": (CKPT_LUNG, "annotation"),
        "HGSOC": (CKPT_HG, "annotation"),
    }
    all_tables = []
    stab_rows = []
    for key, (ckpt, col) in datasets.items():
        print(f"[P2 U0] {key}...", flush=True)
        obs = pd.read_csv(ckpt / "obs.csv", low_memory=False)
        if col not in obs.columns:
            col = "cell_type" if "cell_type" in obs.columns else obs.columns[0]
        u_col = "potential_stationary" if "potential_stationary" in obs.columns else "potential"
        labels = obs[col].astype(str).values
        u = pd.to_numeric(obs[u_col], errors="coerce").values
        # choose n_per_type = min(500, min type size among types with >=100 cells)
        vc = pd.Series(labels).value_counts()
        eligible = vc[vc >= 100]
        n_per = int(min(500, eligible.min())) if len(eligible) else 50
        table, stab = _balanced_rank_one(u, labels, n_per_type=n_per, n_boot=n_boot, seed=0)
        table.insert(0, "dataset", key)
        stab.insert(0, "dataset", key)
        stab["n_per_type"] = n_per
        table.to_csv(OUT / f"{key}_U0_raw_vs_balanced.csv", index=False)
        all_tables.append(table)
        stab_rows.append(stab.iloc[0].to_dict())
        print(table[["cell_type", "n_cells", "mean_U0_raw", "mean_U0_balanced", "rank_raw_deep_to_shallow", "rank_bal_deep_to_shallow"]].to_string(index=False), flush=True)

    all_df = pd.concat(all_tables, ignore_index=True)
    all_df.to_csv(OUT / "all_datasets_U0_raw_vs_balanced.csv", index=False)
    stab_df = pd.DataFrame(stab_rows)
    stab_df.to_csv(OUT / "U0_rank_stability_summary.csv", index=False)

    # Verdict: if Jaccard of deep top3 >= 2/3 and spearman high, ranking not just abundance artifact
    verdict = {
        "per_dataset": stab_rows,
        "all_deep_jaccard_ge_0.5": bool(all(r.get("top3_deep_jaccard", 0) >= 0.5 for r in stab_rows)),
        "all_spearman_ge_0.7": bool(
            all(
                np.isfinite(r.get("spearman_raw_vs_balanced_rank", np.nan))
                and r["spearman_raw_vs_balanced_rank"] >= 0.7
                for r in stab_rows
            )
        ),
        "writing_rule": (
            "If balanced ranks preserve deep/high extremes, abundance bias is limited; "
            "still prefer reporting both raw and balanced means."
        ),
    }
    if verdict["all_deep_jaccard_ge_0.5"] and verdict["all_spearman_ge_0.7"]:
        verdict["overall"] = "PASS_RANK_STABLE"
    elif verdict["all_deep_jaccard_ge_0.5"] or verdict["all_spearman_ge_0.7"]:
        verdict["overall"] = "PARTIAL_RANK_SHIFT"
    else:
        verdict["overall"] = "FAIL_ABUNDANCE_DRIVEN"
    (OUT / "U0_balanced_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    # Plot: GSE155622 raw vs balanced
    pain = all_df[all_df.dataset == "GSE155622"].dropna(subset=["mean_U0_balanced"]).sort_values("mean_U0_raw")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(pain))
    w = 0.35
    ax.bar(x - w / 2, pain["mean_U0_raw"], w, label="raw mean", color=PALETTE[0])
    ax.bar(x + w / 2, pain["mean_U0_balanced"], w, label="balanced mean", color=PALETTE[5])
    ax.set_xticks(x)
    ax.set_xticklabels(pain["cell_type"], rotation=35, ha="right")
    ax.set_ylabel("mean U0")
    ax.set_title("GSE155622: raw vs type-balanced mean U0")
    ax.legend()
    style_axis(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "GSE155622_U0_raw_vs_balanced.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(stab_df.to_string(index=False), flush=True)
    print(json.dumps(verdict, indent=2), flush=True)
    return {"tables": all_df, "stability": stab_df, "verdict": verdict}


def write_report(atf3: Dict, u0: Dict):
    stats = atf3["stats"]
    av = atf3["verdict"]
    stab = u0["stability"]
    uv = u0["verdict"]
    lines = [
        "# P2 稳健性校验报告",
        "",
        "目录：`output_file/robustness/p2_robustness/`",
        "",
        "## 1. Atf3 Hamiltonian KO：Neuron vs Fibroblast",
        "",
        "读出约定：Neuron → Atf3-removed SNIIC partners；Fibroblast → FB_remodel / FB_ECM / contractile（非 SNIIC）。",
        "",
        stats.to_markdown(index=False),
        "",
        "### 判决",
        f"- Neuron partner 终点压低？ **{av.get('strict_neuron_partner_end_reduction', av.get('neuron_partner_blocks'))}**",
        f"-  SNIIC1_noAtf3 KO/WT = {av['neuron_SNIIC1_noAtf3_ratio']:.3g}",
        f"-  SNIIC2_noAtf3 (Mrgprd) KO/WT = {av['neuron_SNIIC2_noAtf3_ratio']:.3g}",
        f"-  SNIIC3_noAtf3 KO/WT = {av['neuron_SNIIC3_noAtf3_ratio']:.3g}",
        f"- Fibroblast FB_remodel KO/WT = {av['fibroblast_FB_remodel_ratio']:.3g}",
        f"- Fibroblast FB_ECM KO/WT = {av['fibroblast_FB_ECM_ratio']:.3g}",
        f"- FB 模块未整体坍塌？ **{av['fibroblast_FB_modules_not_collapsed']}**",
        f"- **overall: {av['overall']}**",
        "",
        av["writing_rule"],
        "",
        "图：`Atf3_cross_type_tracks.png`",
        "",
        "## 2. 类型平衡 U0 排序",
        "",
        stab.to_markdown(index=False),
        "",
        "### 判决",
        f"- 深尾 top3 Jaccard≥0.5 全部数据集？ **{uv['all_deep_jaccard_ge_0.5']}**",
        f"- Spearman(raw,balanced)≥0.7 全部？ **{uv['all_spearman_ge_0.7']}**",
        f"- **overall: {uv['overall']}**",
        "",
        uv["writing_rule"],
        "",
        "图：`GSE155622_U0_raw_vs_balanced.png`",
        "",
        "## 写作修改清单",
        "",
        "1. Atf3：不可写神经元特异——Fibroblast FB_remodel/ECM 同样下降。",
        "2. Partner 阻断协议敏感（P0 vs P2）；严格看终点压低，不看轨迹变钝。",
        "3. U0 类型排序：balanced 与 raw 秩一致时可保留深谷叙事，并同时报告两种均值。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    # Balanced U0 first (fast, obs-only) then Atf3 KO (GPU/heavy)
    u0 = run_type_balanced_u0(n_boot=300)
    atf3 = run_atf3_cross_type()
    write_report(atf3, u0)
    print("Wrote", OUT / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()
