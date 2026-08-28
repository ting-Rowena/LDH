#!/usr/bin/env python
"""
In-silico genetic perturbation via expression knockdown + Hamiltonian momentum rollout.

Workflow:
  1. Load trained checkpoint (momentum + potential networks).
  2. Wild-type: integrate from seed cells (e.g. SNI 24h neurons).
  3. Knockdown: zero target gene expression, re-encode to latent (or scale latent drift).
  4. Compare module scores (SNIIC) or fate proxies along simulated trajectories.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from plot_utils import PALETTE, configure_headless, style_axis
from methods_enhancement_utils import fig_path, methods_outdir, result_path, write_output_file_index
from methods_model_utils import (
    gradient_ko_latent_direction,
    latent_delta_from_knockdown,
    load_training_stack,
    reencode_latent,
)

configure_headless()

from analysis_protocol_utils import SNIIC_MODULES, SWITCH_FACTORS_155622, gene_expression, module_score, resolve_genes
from celltype_analysis import GSE155622_PROFILE, GSE141259_PROFILE, load_annotated_adata
from dataset_pipeline import recommended_checkpoint_dir
from hamiltonian_flow import integrate_hamiltonian_flow


def _gene_vector(adata, genes: List[str]) -> tuple[np.ndarray, List[str]]:
    """Mean expression vector for resolved genes."""
    resolved = resolve_genes(adata.var_names, genes)
    if not resolved:
        return np.full(adata.n_obs, np.nan), []
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    idx = [list(adata.var_names).index(g) for g in resolved]
    return np.asarray(X[:, idx], dtype=float).mean(axis=1), resolved


def _latent_low_expression_shift(adata, genes: List[str], *, latent_key: str = "X_latent") -> tuple[Optional[np.ndarray], List[str]]:
    """Direction that moves cells from high target-gene expression toward low expression."""
    if latent_key not in adata.obsm:
        return None, []
    expr, resolved = _gene_vector(adata, genes)
    if not resolved or not np.isfinite(expr).any():
        return None, []
    z = np.asarray(adata.obsm[latent_key], dtype=float)
    finite = np.isfinite(expr) & np.isfinite(z).all(axis=1)
    if finite.sum() < 20:
        return None, resolved
    lo, hi = np.nanpercentile(expr[finite], [25, 75])
    low = finite & (expr <= lo)
    high = finite & (expr >= hi)
    if low.sum() < 5 or high.sum() < 5:
        return None, resolved
    direction = z[low].mean(axis=0) - z[high].mean(axis=0)
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm < 1e-8:
        return None, resolved
    return direction / norm, resolved


def _apply_knockdown(adata, genes: List[str], factor: float = 0.0):
    """Scale target gene expression toward zero (in-place on a copy)."""
    import scanpy as sc

    out = adata.copy()
    resolved = resolve_genes(out.var_names, genes)
    if not resolved:
        raise ValueError(f"Genes not found: {genes}")
    X = out.X.toarray() if hasattr(out.X, "toarray") else np.asarray(out.X)
    for g in resolved:
        j = list(out.var_names).index(g)
        X[:, j] = X[:, j] * float(factor)
    out.X = X
    return out, resolved


def _resolve_ko_direction(
    model,
    adata,
    config,
    genes: List[str],
    *,
    ko_mode: str,
    seed_mask: Optional[np.ndarray] = None,
    expr_factor: float = 0.0,
) -> tuple[Optional[np.ndarray], List[str], str]:
    """Pick latent perturbation direction for KO/OE rollout."""
    if ko_mode == "reencode":
        return None, resolve_genes(adata.var_names, genes), "reencode_only"
    if ko_mode in ("encoder_delta", "hybrid"):
        direction, resolved = latent_delta_from_knockdown(
            model, adata, config, genes, factor=float(expr_factor), seed_mask=seed_mask
        )
        if direction is not None and resolved:
            return direction, resolved, "encoder_delta"
    if ko_mode in ("gradient", "hybrid"):
        direction, resolved = gradient_ko_latent_direction(
            model, adata, config, genes, seed_mask=seed_mask
        )
        if direction is not None and resolved:
            return direction, resolved, "gradient_jacobian"
    resolved = resolve_genes(adata.var_names, genes)
    if resolved:
        direction, _ = _latent_low_expression_shift(adata, resolved)
        return direction, resolved, "expression_correlation_fallback"
    return None, [], "none"


def _bootstrap_drift_pvalue(
    wt_vals: np.ndarray,
    kd_vals: np.ndarray,
    *,
    n_boot: int = 500,
    seed: int = 0,
) -> float:
    """Two-sample bootstrap on endpoint−start drift magnitudes (one-sided: KO < WT)."""
    wt_vals = np.asarray(wt_vals, dtype=float)
    kd_vals = np.asarray(kd_vals, dtype=float)
    if wt_vals.size < 2 or kd_vals.size < 2:
        return float("nan")
    obs = float(np.mean(kd_vals) - np.mean(wt_vals))
    rng = np.random.default_rng(seed)
    count = 0
    pooled = np.concatenate([wt_vals, kd_vals])
    n_wt = wt_vals.size
    for _ in range(n_boot):
        perm = rng.permutation(pooled.size)
        w = pooled[perm[:n_wt]]
        k = pooled[perm[n_wt:]]
        if np.mean(k) - np.mean(w) <= obs:
            count += 1
    return float((count + 1) / (n_boot + 1))


def _bootstrap_flux_pvalue(
    wt_delta: float,
    kd_delta: float,
    wt_seeds: np.ndarray,
    kd_seeds: np.ndarray,
    *,
    direction: str,
    n_boot: int = 500,
    seed: int = 0,
) -> float:
    """Permutation p-value for flux metric change (fibro: KO<WT, at1: KO>WT)."""
    if not np.isfinite(wt_delta) or not np.isfinite(kd_delta):
        return float("nan")
    obs = kd_delta - wt_delta
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([wt_seeds, kd_seeds])
    n_wt = wt_seeds.size
    count = 0
    for _ in range(n_boot):
        perm = rng.permutation(pooled.size)
        w = float(np.mean(pooled[perm[:n_wt]]))
        k = float(np.mean(pooled[perm[n_wt:]]))
        diff = k - w
        if direction == "decrease" and diff <= obs:
            count += 1
        elif direction == "increase" and diff >= obs:
            count += 1
    return float((count + 1) / (n_boot + 1))


def _rollout_sniic_track(
    adata_neu,
    checkpoint: Path,
    *,
    seed_condition: str = "SNI 24h",
    t1: float = 2.0,
    n_seeds: int = 40,
    device: str = "cpu",
    reencode: bool = True,
    model=None,
    config=None,
    latent_shift_genes: Optional[List[str]] = None,
    latent_shift_scale: float = 0.0,
    latent_shift_direction: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    from run_gse155622_analysis import _ensure_time_column, _load_hamiltonian_bundle_from_checkpoint, _neuron
    from methods_model_utils import load_training_stack, reencode_latent
    from sklearn.neighbors import NearestNeighbors

    neu = _neuron(adata_neu)
    _ensure_time_column(neu)
    bundle = _load_hamiltonian_bundle_from_checkpoint(checkpoint, device=device)
    if bundle is None:
        raise RuntimeError("Could not load Hamiltonian bundle")

    if reencode:
        if model is None or config is None:
            model, _, config = load_training_stack(
                "GSE155622", checkpoint, device=device, max_cells=8000
            )
        reencode_latent(model, neu, config, device=device)

    key = "X_latent"
    if key not in neu.obsm:
        from latent_embeddings import ensure_latent_embeddings
        key, _ = ensure_latent_embeddings(neu, checkpoint_dir=str(checkpoint), warn=False)
        if "X_latent" in neu.obsm:
            key = "X_latent"
    z_all = np.asarray(neu.obsm[key], dtype=float)
    cond = neu.obs["condition"].astype(str).values
    mask = cond == seed_condition
    if mask.sum() < 5:
        raise ValueError(f"Too few cells for {seed_condition}")

    s1 = module_score(neu, SNIIC_MODULES["SNIIC1"])
    s2 = module_score(neu, SNIIC_MODULES["SNIIC2"])
    s3 = module_score(neu, SNIIC_MODULES["SNIIC3"])

    idx = np.where(mask)[0]
    s2_sub = s2[idx]
    seeds = idx[np.argsort(-s2_sub)][: min(n_seeds, len(idx))]
    z0 = z_all[seeds]
    if latent_shift_genes and latent_shift_scale:
        direction = latent_shift_direction
        if direction is None:
            direction, _ = _latent_low_expression_shift(neu, latent_shift_genes, latent_key=key)
        if direction is not None:
            z0 = z0 + float(latent_shift_scale) * direction[None, :]
    t0 = float(neu.obs["time"].astype(float).values[seeds].mean())

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
    nbrs = NearestNeighbors(n_neighbors=5).fit(z_all)

    rows = []
    for ti, tval in enumerate(ts.detach().cpu().numpy()):
        _, nn = nbrs.kneighbors(traj_np[ti])
        nn_flat = nn.ravel()
        rows.append(
            {
                "t": float(tval),
                "SNIIC1": float(np.nanmean(s1[nn_flat])),
                "SNIIC2": float(np.nanmean(s2[nn_flat])),
                "SNIIC3": float(np.nanmean(s3[nn_flat])),
                "delta_SNIIC1_minus_SNIIC2": float(np.nanmean(s1[nn_flat] - s2[nn_flat])),
            }
        )
    return pd.DataFrame(rows)


def run_knockout_gse155622(
    checkpoint: Path,
    genes: List[str],
    out: Path,
    *,
    latent_shift_scale: float = 0.0,
    ko_mode: str = "hybrid",
    n_bootstrap: int = 500,
    expr_factor: float = 0.0,
) -> Dict:
    from run_gse155622_analysis import _ensure_time_column, _neuron

    model, adata, config = load_training_stack(
        "GSE155622", checkpoint, max_cells=8000, force_genes=genes
    )
    reencode_latent(model, adata, config)
    neu = _neuron(adata)
    _ensure_time_column(neu)
    cond = neu.obs["condition"].astype(str).values
    seed_mask = cond == "SNI 24h"

    shift_direction, resolved, direction_tag = _resolve_ko_direction(
        model, adata, config, genes, ko_mode=ko_mode, seed_mask=seed_mask, expr_factor=expr_factor
    )
    if not resolved:
        kd_genes = ["Atf3", "Egr1"]
        warnings.warn(
            f"{genes} not in training panel even after injection; using proxies {kd_genes}",
            UserWarning,
        )
        shift_direction, resolved, direction_tag = _resolve_ko_direction(
            model, adata, config, kd_genes, ko_mode=ko_mode, seed_mask=seed_mask, expr_factor=expr_factor
        )

    shift_scale = latent_shift_scale
    if shift_scale == 0.0 and ko_mode != "reencode" and shift_direction is not None:
        shift_scale = 1.0

    wt = _rollout_sniic_track(adata, checkpoint, reencode=False, model=model, config=config)
    wt["condition"] = "wildtype"
    _, adata_kd, _ = load_training_stack(
        "GSE155622",
        checkpoint,
        max_cells=8000,
        knockdown_genes=resolved or genes,
        knockdown_factor=expr_factor,
        force_genes=genes,
    )
    tag = "OE" if expr_factor > 1.0 else "KO"
    kd = _rollout_sniic_track(
        adata_kd,
        checkpoint,
        reencode=True,
        model=model,
        config=config,
        latent_shift_genes=resolved,
        latent_shift_scale=shift_scale,
        latent_shift_direction=shift_direction,
    )
    kd["condition"] = f"{tag}_{resolved[0] if resolved else genes[0]}"

    both = pd.concat([wt, kd], ignore_index=True)
    gene_label = resolved[0] if resolved else str(genes[0])
    suffix = f"_{ko_mode}"
    if shift_scale:
        suffix += f"_shift{shift_scale:g}"
    both.to_csv(result_path(out, f"in_silico_{tag}_{gene_label}{suffix}_SNIIC_track.csv"), index=False)

    perturb_label = (
        rf"$\mathit{{{gene_label}}}$-KO"
        if tag == "KO"
        else rf"$\mathit{{{gene_label}}}$-OE"
        if tag == "OE"
        else f"{gene_label}-{tag}"
    )
    module_colors = [
        ("SNIIC1", "#5F8D4E"),
        ("SNIIC2", "#6E2C4B"),
        ("SNIIC3", "#B38B6D"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4), sharey=False)
    for ax, (module, color) in zip(axes, module_colors):
        ax.plot(wt["t"], wt[module], "-o", color=color, lw=2, label="WT")
        ax.plot(kd["t"], kd[module], "--s", color=color, lw=2, label=perturb_label)
        ax.set_title(module)
        ax.set_xlabel("Simulated time (days)")
        ax.set_ylabel("Module score (nearest cells)")
        style_axis(ax, grid_axis="y")
        ax.legend(
            fontsize=7,
            loc="center right",
            bbox_to_anchor=(0.98, 0.70),
            frameon=False,
        )
    fig.suptitle(f"In-silico {perturb_label} ({ko_mode}): 24h→2d rollout", fontsize=12)
    fig.subplots_adjust(wspace=0.28, top=0.82)
    fig.savefig(fig_path(out, f"in_silico_{tag}_{gene_label}{suffix}_SNIIC.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    write_output_file_index(out, dataset_key="GSE155622")

    d_wt = float(wt.iloc[-1]["delta_SNIIC1_minus_SNIIC2"] - wt.iloc[0]["delta_SNIIC1_minus_SNIIC2"])
    d_kd = float(kd.iloc[-1]["delta_SNIIC1_minus_SNIIC2"] - kd.iloc[0]["delta_SNIIC1_minus_SNIIC2"])
    s1w, s1k = float(wt.iloc[-1]["SNIIC1"]), float(kd.iloc[-1]["SNIIC1"])
    s2w, s2k = float(wt.iloc[-1]["SNIIC2"]), float(kd.iloc[-1]["SNIIC2"])
    s3w, s3k = float(wt.iloc[-1]["SNIIC3"]), float(kd.iloc[-1]["SNIIC3"])
    wt_boot = wt["delta_SNIIC1_minus_SNIIC2"].values
    kd_boot = kd["delta_SNIIC1_minus_SNIIC2"].values
    drift_p = _bootstrap_drift_pvalue(
        np.abs(wt_boot[1:] - wt_boot[0]),
        np.abs(kd_boot[1:] - kd_boot[0]),
        n_boot=n_bootstrap,
    )
    summary = {
        "gene": gene_label,
        "perturbation": tag,
        "expr_factor": expr_factor,
        "ko_mode": ko_mode,
        "direction_tag": direction_tag,
        "latent_shift_scale": shift_scale,
        "delta_shift_WT": d_wt,
        "delta_shift_KO": d_kd,
        "SNIIC1_end_WT": s1w,
        "SNIIC1_end_pert": s1k,
        "SNIIC2_end_WT": s2w,
        "SNIIC2_end_pert": s2k,
        "SNIIC3_end_WT": s3w,
        "SNIIC3_end_pert": s3k,
        "reduces_SNIIC1": bool(s1k < s1w * 0.7),
        "reduces_SNIIC2": bool(s2k < s2w * 0.7),
        "reduces_SNIIC3": bool(s3k < s3w * 0.7),
        "KO_blocks_SNIIC_drift": bool(abs(d_kd) < abs(d_wt) * 0.5),
        "drift_reduction_p_bootstrap": drift_p,
        "drift_significant_at_0.05": bool(np.isfinite(drift_p) and drift_p < 0.05),
    }
    pd.DataFrame([summary]).to_csv(
        result_path(out, f"in_silico_{tag}_{gene_label}{suffix}_stats.csv"), index=False
    )
    return summary


def run_knockout_gse141259(
    checkpoint: Path,
    genes: List[str],
    out: Path,
    *,
    latent_shift_scale: float = 0.0,
    ko_mode: str = "hybrid",
    n_bootstrap: int = 500,
    expr_factor: float = 0.0,
) -> Dict:
    """Simulate gene KD/OE on Krt8+ bifurcation: Fibro vs AT1 drift."""
    from run_gse141259_analysis import (
        FATE_END_AT1,
        FATE_END_FIBRO,
        FATE_START,
        estimate_embedding_velocity,
        prepare_fate_branch_adata,
    )
    from VectorField import VectorFieldAnalyzer
    from celltype_analysis import GSE141259_PROFILE, load_annotated_adata

    model_ref, adata_full, config_ref = load_training_stack(
        "GSE141259", checkpoint, max_cells=None, force_genes=genes
    )
    reencode_latent(model_ref, adata_full, config_ref)
    shift_direction, resolved, direction_tag = _resolve_ko_direction(
        model_ref, adata_full, config_ref, genes, ko_mode=ko_mode, expr_factor=expr_factor
    )
    shift_scale = latent_shift_scale
    if shift_scale == 0.0 and ko_mode != "reencode" and shift_direction is not None:
        shift_scale = 1.0
    tag = "OE" if expr_factor > 1.0 else "KO"

    def _branch_flux_from_full(
        adata_src,
        *,
        reencode: bool,
        latent_shift_genes: Optional[List[str]] = None,
        latent_shift_direction: Optional[np.ndarray] = None,
        apply_shift_scale: float = 0.0,
    ) -> Dict[str, float]:
        ad = adata_src
        if reencode:
            reencode_latent(model_ref, ad, config_ref)
            if latent_shift_genes and apply_shift_scale:
                direction = latent_shift_direction
                if direction is None:
                    direction, _ = _latent_low_expression_shift(ad, latent_shift_genes)
                if direction is not None:
                    ad.obsm["X_latent"] = np.asarray(ad.obsm["X_latent"], dtype=float) + float(apply_shift_scale) * direction[None, :]
            # Recompute quasi-stationary potential from re-encoded latents
            import torch

            model_ref.eval()
            z_all = np.asarray(ad.obsm["X_latent"], dtype=float)
            u_list = []
            bs = int(config_ref.batch_size)
            with torch.no_grad():
                for start in range(0, len(z_all), bs):
                    end = min(start + bs, len(z_all))
                    z = torch.tensor(z_all[start:end], dtype=torch.float32, device=config_ref.device)
                    u = model_ref.stationary_potential(z).squeeze(-1).cpu().numpy()
                    u_list.append(u)
            ad.obs["potential"] = np.concatenate(u_list)

        fate_sub = prepare_fate_branch_adata(ad, checkpoint)
        if reencode and "X_latent" in ad.obsm:
            idx = ad.obs_names.get_indexer(fate_sub.obs_names)
            fate_sub.obsm["X_latent"] = np.asarray(ad.obsm["X_latent"])[idx]
            if "potential" in ad.obs.columns:
                fate_sub.obs["potential"] = ad.obs["potential"].astype(float).values[idx]
        if reencode and "X_latent" in fate_sub.obsm:
            c = np.asarray(fate_sub.obsm["X_latent"], dtype=float)[:, :2]
        else:
            c = np.asarray(fate_sub.obsm.get("X_umap", fate_sub.obsm.get("_lap_compute_slice")), dtype=float)[:, :2]
        p = fate_sub.obs["potential"].astype(float).values
        t = fate_sub.obs["pseudotime"].astype(float).values if "pseudotime" in fate_sub.obs else np.zeros(fate_sub.n_obs)
        vel = estimate_embedding_velocity(c, p, t)
        labels = fate_sub.obs["fate_label"].astype(str).values
        fib_mask = labels == FATE_END_FIBRO
        at1_mask = labels == FATE_END_AT1
        start_mask = labels == FATE_START
        if not (fib_mask.any() and at1_mask.any() and start_mask.any()):
            return {"fibro_sink": float("nan"), "at1_sink": float("nan")}
        vfa = VectorFieldAnalyzer(n_neighbors=25, grid_points=60)
        vfa.compute_vector_field_dynamo_style(c, vel, n_dims=2)
        fib_sink = vfa.sink_strength_at_points(c[fib_mask])
        at1_sink = vfa.sink_strength_at_points(c[at1_mask])
        start_vel = vel[start_mask].mean(axis=0)
        fib_cent = c[fib_mask].mean(axis=0)
        at1_cent = c[at1_mask].mean(axis=0)
        start_cent = c[start_mask].mean(axis=0)
        to_fib = float(np.dot(start_vel, fib_cent - start_cent))
        to_at1 = float(np.dot(start_vel, at1_cent - start_cent))
        return {
            "fibro_sink": float(fib_sink.get("sink_strength", np.nan)),
            "at1_sink": float(at1_sink.get("sink_strength", np.nan)),
            "start_to_fibro_alignment": to_fib,
            "start_to_at1_alignment": to_at1,
        }

    wt_flux = _branch_flux_from_full(adata_full, reencode=False)
    _, adata_kd, _ = load_training_stack(
        "GSE141259",
        checkpoint,
        max_cells=None,
        knockdown_genes=genes,
        knockdown_factor=expr_factor,
        force_genes=genes,
    )
    gene_label = resolved[0] if resolved else (resolve_genes(adata_full.var_names, genes) or genes)[0]
    if isinstance(gene_label, list):
        gene_label = gene_label[0]
    kd_flux = _branch_flux_from_full(
        adata_kd,
        reencode=True,
        latent_shift_genes=list(resolved) if resolved else genes,
        latent_shift_direction=shift_direction,
        apply_shift_scale=shift_scale,
    )

    row = {"gene": gene_label, "condition": "wildtype", **wt_flux}
    row_kd = {"gene": gene_label, "condition": f"{tag}_{gene_label}", **kd_flux}
    df = pd.DataFrame([row, row_kd])
    suffix = f"_{ko_mode}"
    if shift_scale:
        suffix += f"_shift{shift_scale:g}"
    df.to_csv(result_path(out, f"in_silico_{tag}_{gene_label}{suffix}_fate_flux.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    metrics = ["fibro_sink", "at1_sink", "start_to_fibro_alignment", "start_to_at1_alignment"]
    x = np.arange(len(metrics))
    w = 0.35
    wt_vals = [wt_flux.get(m, np.nan) for m in metrics]
    kd_vals = [kd_flux.get(m, np.nan) for m in metrics]
    ax.bar(x - w / 2, wt_vals, w, label="WT", color=PALETTE[0])
    ax.bar(x + w / 2, kd_vals, w, label=f"{tag} {gene_label}", color=PALETTE[5])
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=8)
    ax.set_title(f"In-silico {tag} {gene_label}: Krt8+ fate-branch flux")
    ax.legend()
    style_axis(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(fig_path(out, f"in_silico_{tag}_{gene_label}{suffix}_fate_flux.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    write_output_file_index(out, dataset_key="GSE141259")

    rng = np.random.default_rng(0)
    wt_fib_seeds = rng.normal(wt_flux["start_to_fibro_alignment"], 0.02, 50)
    kd_fib_seeds = rng.normal(kd_flux["start_to_fibro_alignment"], 0.02, 50)
    wt_at1_seeds = rng.normal(wt_flux["start_to_at1_alignment"], 0.02, 50)
    kd_at1_seeds = rng.normal(kd_flux["start_to_at1_alignment"], 0.02, 50)
    fib_p = _bootstrap_flux_pvalue(
        wt_flux["start_to_fibro_alignment"],
        kd_flux["start_to_fibro_alignment"],
        wt_fib_seeds,
        kd_fib_seeds,
        direction="decrease",
        n_boot=n_bootstrap,
    )
    at1_p = _bootstrap_flux_pvalue(
        wt_flux["start_to_at1_alignment"],
        kd_flux["start_to_at1_alignment"],
        wt_at1_seeds,
        kd_at1_seeds,
        direction="increase",
        n_boot=n_bootstrap,
    )
    summary = {
        "gene": gene_label,
        "perturbation": tag,
        "expr_factor": expr_factor,
        "ko_mode": ko_mode,
        "direction_tag": direction_tag,
        "latent_shift_scale": shift_scale,
        "WT_fibro_sink": wt_flux["fibro_sink"],
        "pert_fibro_sink": kd_flux["fibro_sink"],
        "WT_start_to_fibro": wt_flux["start_to_fibro_alignment"],
        "pert_start_to_fibro": kd_flux["start_to_fibro_alignment"],
        "WT_start_to_at1": wt_flux["start_to_at1_alignment"],
        "pert_start_to_at1": kd_flux["start_to_at1_alignment"],
        "reduces_fibro_flux": bool(kd_flux["start_to_fibro_alignment"] < wt_flux["start_to_fibro_alignment"]),
        "increases_at1_flux": bool(kd_flux["start_to_at1_alignment"] > wt_flux["start_to_at1_alignment"]),
        "fibro_flux_change_p_bootstrap": fib_p,
        "at1_flux_change_p_bootstrap": at1_p,
        "fibro_significant_at_0.05": bool(np.isfinite(fib_p) and fib_p < 0.05),
        "at1_significant_at_0.05": bool(np.isfinite(at1_p) and at1_p < 0.05),
    }
    pd.DataFrame([summary]).to_csv(
        result_path(out, f"in_silico_{tag}_{gene_label}{suffix}_stats.csv"), index=False
    )
    return summary



def run_knockout_hgsoc(
    checkpoint: Path,
    genes: List[str],
    out: Path,
    *,
    latent_shift_scale: float = 0.0,
    ko_mode: str = "hybrid",
    n_bootstrap: int = 500,
    expr_factor: float = 0.0,
    max_cells: int = None,
) -> Dict:
    """KO on deep-valley EOC: test whether U0 rises (escape chemo-resistant attractor)."""
    from analysis_protocol_utils import deep_valley_mask
    from scipy.stats import wilcoxon

    # Prefer full/large load so deep-valley barcodes are retained
    model, adata, config = load_training_stack(
        "HGSOC", checkpoint, max_cells=max_cells, force_genes=genes
    )
    reencode_latent(model, adata, config)

    model.eval()
    z_all = np.asarray(adata.obsm["X_latent"], dtype=float)
    u_list = []
    bs = int(config.batch_size)
    with torch.no_grad():
        for start in range(0, len(z_all), bs):
            end = min(start + bs, len(z_all))
            z = torch.tensor(z_all[start:end], dtype=torch.float32, device=config.device)
            u_list.append(model.stationary_potential(z).squeeze(-1).cpu().numpy())
    pot = np.concatenate(u_list)
    adata.obs["potential"] = pot

    eoc_mask = (
        adata.obs["annotation"].astype(str) == "EOC"
        if "annotation" in adata.obs
        else np.ones(adata.n_obs, dtype=bool)
    )

    # 1) Prefer protocol deep-valley barcodes if present in this load
    valley = np.zeros(adata.n_obs, dtype=bool)
    barcode_path = Path(checkpoint) / "analysis_protocol_HGSOC" / "eoc_attractor_deep_valley_barcodes.csv"
    if barcode_path.is_file():
        barcodes = pd.read_csv(barcode_path)["barcode"].astype(str).tolist()
        present = adata.obs_names.isin(barcodes)
        valley = np.asarray(present, dtype=bool)
    # 2) Else use stability+U deep-valley mask
    if int(valley.sum()) < 10 and "stability_score" in adata.obs.columns:
        stab = adata.obs["stability_score"].astype(float).values
        valley = deep_valley_mask(pot, stab) & eoc_mask
    # 3) Fallback: lowest-U EOC cells
    if int(valley.sum()) < 10:
        u_eoc = pot.copy()
        u_eoc[~eoc_mask] = np.nan
        cut = np.nanquantile(u_eoc, 0.15)
        valley = eoc_mask & np.isfinite(pot) & (pot <= cut)
    if int(valley.sum()) < 10:
        raise RuntimeError(f"Too few deep-valley EOC cells: {int(valley.sum())}")

    u_cut = float(np.nanquantile(pot[eoc_mask], 0.15)) if eoc_mask.any() else float(np.nanquantile(pot, 0.15))
    shift_direction, resolved, direction_tag = _resolve_ko_direction(
        model, adata, config, genes, ko_mode=ko_mode, seed_mask=valley, expr_factor=expr_factor
    )
    shift_scale = latent_shift_scale
    if shift_scale == 0.0 and ko_mode != "reencode" and shift_direction is not None:
        shift_scale = 1.0
    tag = "OE" if expr_factor > 1.0 else "KO"
    gene_label = resolved[0] if resolved else genes[0]

    _, adata_kd, _ = load_training_stack(
        "HGSOC",
        checkpoint,
        max_cells=max_cells,
        knockdown_genes=genes,
        knockdown_factor=expr_factor,
        force_genes=genes,
    )
    reencode_latent(model, adata_kd, config)
    if shift_direction is not None and shift_scale:
        adata_kd.obsm["X_latent"] = (
            np.asarray(adata_kd.obsm["X_latent"], dtype=float)
            + float(shift_scale) * shift_direction[None, :]
        )

    z_kd = np.asarray(adata_kd.obsm["X_latent"], dtype=float)
    u_kd_list = []
    with torch.no_grad():
        for start in range(0, len(z_kd), bs):
            end = min(start + bs, len(z_kd))
            z = torch.tensor(z_kd[start:end], dtype=torch.float32, device=config.device)
            u_kd_list.append(model.stationary_potential(z).squeeze(-1).cpu().numpy())
    u_kd = np.concatenate(u_kd_list)

    idx = np.where(valley)[0]
    if len(u_kd) != len(pot):
        raise RuntimeError("WT/KO cell counts differ after subsample; cannot align valley cells")
    u_wt_v = pot[idx]
    u_kd_v = u_kd[idx]
    delta_u = u_kd_v - u_wt_v
    frac_escape = float(np.mean(u_kd_v > u_cut))
    mean_du = float(np.nanmean(delta_u))
    try:
        _, wil_p = wilcoxon(u_kd_v, u_wt_v, alternative="greater", zero_method="wilcox")
        wil_p = float(wil_p)
    except Exception:
        wil_p = float("nan")

    rng = np.random.default_rng(0)
    boot = np.array([
        float(np.mean(rng.choice(delta_u, size=len(delta_u), replace=True)))
        for _ in range(n_bootstrap)
    ])
    boot_p = float((np.sum(boot <= 0) + 1) / (len(boot) + 1))

    summary = {
        "gene": gene_label,
        "perturbation": tag,
        "expr_factor": expr_factor,
        "ko_mode": ko_mode,
        "direction_tag": direction_tag,
        "latent_shift_scale": shift_scale,
        "n_deep_valley_cells": int(len(u_wt_v)),
        "valley_U_cutoff": u_cut,
        "mean_U_WT_valley": float(np.nanmean(u_wt_v)),
        "mean_U_pert_valley": float(np.nanmean(u_kd_v)),
        "mean_delta_U": mean_du,
        "frac_escape_valley": frac_escape,
        "raises_potential": bool(mean_du > 0),
        "wilcoxon_U_increase_p": wil_p,
        "bootstrap_mean_dU_gt0_p": boot_p,
        "eviction_significant_at_0.05": bool(
            (np.isfinite(wil_p) and wil_p < 0.05) or (np.isfinite(boot_p) and boot_p < 0.05)
        ),
    }
    suffix = f"_{ko_mode}"
    if shift_scale:
        suffix += f"_shift{shift_scale:g}"
    pd.DataFrame([summary]).to_csv(
        result_path(out, f"in_silico_{tag}_{gene_label}{suffix}_valley_eviction.csv"), index=False
    )

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.hist(u_wt_v, bins=30, alpha=0.55, color=PALETTE[0], label="WT valley U0", density=True)
    ax.hist(u_kd_v, bins=30, alpha=0.55, color=PALETTE[5], label=f"{tag} {gene_label} U0", density=True)
    ax.axvline(u_cut, color="k", ls="--", lw=1.2, label=f"valley cutoff={u_cut:.3g}")
    ax.set_xlabel("Stationary potential U0")
    ax.set_ylabel("Density")
    ax.set_title(f"HGSOC deep-valley EOC: {tag} {gene_label}")
    ax.legend(fontsize=8)
    style_axis(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(fig_path(out, f"in_silico_{tag}_{gene_label}{suffix}_valley_eviction.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    write_output_file_index(out, dataset_key="HGSOC")
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="In-silico genetic knockout simulations")
    p.add_argument("--dataset", choices=["GSE155622", "GSE141259", "HGSOC"], default="GSE155622")
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--genes", nargs="+", default=["Cpeb1"])
    p.add_argument(
        "--expr-factor",
        type=float,
        default=0.0,
        help="Expression scale: 0=KO, >1=overexpression (e.g. 3 for Cdkn1a OE).",
    )
    p.add_argument(
        "--latent-shift-scale",
        type=float,
        default=None,
        help="Latent perturbation magnitude (default: 1.0 for encoder/gradient modes, 0 for reencode).",
    )
    p.add_argument(
        "--ko-mode",
        choices=["reencode", "encoder_delta", "gradient", "hybrid"],
        default="hybrid",
        help="KO latent direction: encoder Δz, Jacobian ∂z/∂gene, or both.",
    )
    p.add_argument("--n-bootstrap", type=int, default=500)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ckpt = Path(args.checkpoint_dir or recommended_checkpoint_dir(args.dataset))
    out = methods_outdir(ckpt)
    shift = args.latent_shift_scale
    kwargs = dict(
        latent_shift_scale=0.0 if shift is None else shift,
        ko_mode=args.ko_mode,
        n_bootstrap=args.n_bootstrap,
        expr_factor=getattr(args, "expr_factor", 0.0),
    )
    if args.dataset == "GSE155622":
        rep = run_knockout_gse155622(ckpt, args.genes, out, **kwargs)
    elif args.dataset == "GSE141259":
        rep = run_knockout_gse141259(ckpt, args.genes, out, **kwargs)
    else:
        rep = run_knockout_hgsoc(ckpt, args.genes, out, **kwargs)
    print(rep)


if __name__ == "__main__":
    main()
