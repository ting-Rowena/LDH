#!/usr/bin/env python3
"""Atf3 OE sufficiency (Control→injury) + path-cost (enter vs 7d→14d exit).

(3) Bidirectional perturbation: Atf3 overexpression on Control neuron seeds.
    If OE raises Atf3-removed SNIIC partners and lowers Nav → closer to sufficiency;
    if not → Atf3 is better described as a maintenance factor than an initiator.

(4) Path cost: injury slide-in (Control→SNI 2d deep) vs fall-back (SNI 7d→14d)
    via geodesic / light LAP action and ΔU0. Biology claims stay restrained;
    the increment vs heatmaps is the physical path quantities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / (
    "GSE155622_checkpoints_3000_3000_384_0.05_recon0.01_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
)
OUT = ROOT / "output_file" / "robustness" / "atf3_oe_and_path_cost"
OUT.mkdir(parents=True, exist_ok=True)
PANEL = OUT

FORCE_GENES = [
    "Atf3",
    "S100b",
    "Csf1",
    "Clcf1",
    "Scn9a",
    "Scn10a",
    "Scn11a",
    "Gfra3",
    "Gal",
    "Mrgprd",
]
MODULES: Dict[str, Sequence[str]] = {
    "SNIIC1": ("Atf3", "Gfra3", "Gal"),
    "SNIIC2": ("Atf3", "Mrgprd"),
    "SNIIC3": ("Atf3", "S100b", "Gal"),
    "SNIIC1_noAtf3": ("Gfra3", "Gal"),
    "SNIIC2_noAtf3": ("Mrgprd",),
    "SNIIC3_noAtf3": ("S100b", "Gal"),
    "Atf3_alone": ("Atf3",),
    "Nav_triad": ("Scn9a", "Scn10a", "Scn11a"),
    "Scn9a": ("Scn9a",),
    "Scn10a": ("Scn10a",),
    "Scn11a": ("Scn11a",),
    "Csf1": ("Csf1",),
    "Clcf1": ("Clcf1",),
}
COLORS = {
    "SNIIC1_noAtf3": "#9EC1C0",
    "SNIIC2_noAtf3": "#E0BFB8",
    "SNIIC3_noAtf3": "#F0E4D2",
    "Nav_triad": "#355C8A",
    "Csf1": "#C45C26",
    "Atf3_alone": "#8B4557",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _module_scores(adata, modules: Dict[str, Sequence[str]]) -> Dict[str, np.ndarray]:
    from analysis_protocol_utils import module_score

    return {k: module_score(adata, genes) for k, genes in modules.items()}


def _endpoint_row(wt: pd.DataFrame, pert: pd.DataFrame, mod: str, arm: str) -> Dict:
    wt_end = float(wt.iloc[-1][mod])
    p_end = float(pert.iloc[-1][mod])
    ratio = p_end / wt_end if abs(wt_end) > 1e-12 else np.nan
    return {
        "arm": arm,
        "module": mod,
        "end_WT": wt_end,
        "end_pert": p_end,
        "end_ratio_pert_over_WT": ratio,
        "delta_end_pert_minus_WT": p_end - wt_end,
        "start_WT": float(wt.iloc[0][mod]),
        "start_pert": float(pert.iloc[0][mod]),
    }


def _plot_tracks(tracks: pd.DataFrame, modules: List[str], *, title: str, outfile: Path, ncols: int = 3) -> None:
    from panel_style import LEGEND_SIZE, apply_panel_title_rc, set_panel_title
    from plot_utils import configure_headless, style_axis

    configure_headless()
    apply_panel_title_rc()
    n = len(modules)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.1 * nrows), squeeze=False)
    for ax, mod in zip(axes.ravel(), modules):
        color = COLORS.get(mod, "#355C8A")
        for cond, style in (("WT", dict(ls="-", marker="o")), ("Atf3_OE", dict(ls="--", marker="s"))):
            d = tracks[tracks["condition"] == cond]
            if d.empty:
                continue
            ax.plot(
                d["t"],
                d[mod],
                color=color,
                lw=2,
                label="WT" if cond == "WT" else r"$\mathit{Atf3}$-OE $\times3$",
                **style,
            )
        set_panel_title(ax, mod.replace("_noAtf3", " partners"))
        ax.set_xlabel("Simulated time (days)")
        ax.set_ylabel("NN module score")
        style_axis(ax, grid_axis="y")
        ax.legend(fontsize=LEGEND_SIZE, loc="best", frameon=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(wspace=0.30, hspace=0.45, top=0.88)
    fig.savefig(outfile, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# (3) Atf3 OE on Control → injury horizon
# ---------------------------------------------------------------------------

def rollout_control_modules(
    adata,
    checkpoint: Path,
    modules: Dict[str, Sequence[str]],
    *,
    seed_condition: str = "Control",
    seed_by: str = "SNIIC2",
    seed_low: bool = True,
    t1: float = 2.0,
    n_seeds: int = 40,
    device: str = "cpu",
    latent_shift_direction: Optional[np.ndarray] = None,
    latent_shift_scale: float = 0.0,
    nn_k: int = 5,
) -> Tuple[pd.DataFrame, Dict]:
    from hamiltonian_flow import integrate_hamiltonian_flow
    from run_gse155622_analysis import _ensure_time_column, _load_hamiltonian_bundle_from_checkpoint, _neuron

    neu = _neuron(adata)
    _ensure_time_column(neu)
    bundle = _load_hamiltonian_bundle_from_checkpoint(checkpoint, device=device)
    if bundle is None:
        raise RuntimeError("Could not load Hamiltonian bundle")
    if "X_latent" not in neu.obsm:
        raise KeyError("X_latent missing")

    z_all = np.asarray(neu.obsm["X_latent"], dtype=float)
    scores = _module_scores(neu, modules)
    cond = neu.obs["condition"].astype(str).values
    mask = cond == seed_condition
    if int(mask.sum()) < 5:
        raise ValueError(f"Too few {seed_condition} neurons")

    idx = np.where(mask)[0]
    rank = scores[seed_by][idx]
    finite = np.isfinite(rank)
    idx = idx[finite]
    rank = rank[finite]
    order = np.argsort(rank) if seed_low else np.argsort(-rank)
    seeds = idx[order][: min(n_seeds, len(idx))]

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
        row = {"t": float(tval), "seed_by": seed_by, "seed_low": bool(seed_low)}
        for name, arr in scores.items():
            row[name] = float(np.nanmean(arr[flat]))
        rows.append(row)
    meta = {
        "seed_condition": seed_condition,
        "seed_by": seed_by,
        "seed_low": bool(seed_low),
        "n_seeds": int(len(seeds)),
        "t0": t0,
        "t1": t1,
        "seed_mean_Atf3": float(np.nanmean(scores["Atf3_alone"][seeds])),
        "seed_mean_SNIIC2": float(np.nanmean(scores["SNIIC2"][seeds])),
    }
    return pd.DataFrame(rows), meta


def run_atf3_oe(device: Optional[str] = None, expr_factor: float = 3.0, t1: float = 2.0) -> Dict:
    from methods_model_utils import load_training_stack, reencode_latent
    from plot_utils import configure_headless
    from run_gse155622_analysis import _ensure_time_column, _neuron
    from run_in_silico_knockout import _resolve_ko_direction

    configure_headless()
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[OE] device={device} expr_factor={expr_factor} t1={t1}", flush=True)

    print("[OE] load WT stack...", flush=True)
    model, adata, config = load_training_stack(
        "GSE155622", CKPT, device=device, max_cells=8000, force_genes=FORCE_GENES
    )
    reencode_latent(model, adata, config, device=device)
    neu = _neuron(adata)
    _ensure_time_column(neu)
    ctrl = set(neu.obs_names[neu.obs["condition"].astype(str) == "Control"].astype(str))
    seed_mask = np.asarray([bn in ctrl for bn in adata.obs_names.astype(str)], dtype=bool)

    # OE direction: z(OE×factor) − z(WT)
    shift_direction, resolved, direction_tag = _resolve_ko_direction(
        model,
        adata,
        config,
        ["Atf3"],
        ko_mode="hybrid",
        seed_mask=seed_mask,
        expr_factor=expr_factor,
    )
    shift_scale = 1.0 if shift_direction is not None else 0.0
    print(f"[OE] direction={direction_tag} scale={shift_scale} resolved={resolved}", flush=True)

    print("[OE] WT Control→t1 rollout (low-SNIIC2 seeds)...", flush=True)
    wt, meta_wt = rollout_control_modules(
        adata, CKPT, MODULES, seed_low=True, t1=t1, device=device
    )
    wt["condition"] = "WT"

    print("[OE] load OE stack...", flush=True)
    _, adata_oe, _ = load_training_stack(
        "GSE155622",
        CKPT,
        device=device,
        max_cells=8000,
        knockdown_genes=["Atf3"],
        knockdown_factor=expr_factor,
        force_genes=FORCE_GENES,
    )
    common = [b for b in adata.obs_names.astype(str) if b in set(adata_oe.obs_names.astype(str))]
    adata_oe = adata_oe[common].copy() if len(common) >= 1000 else adata_oe
    reencode_latent(model, adata_oe, config, device=device)

    print("[OE] OE Control→t1 rollout...", flush=True)
    oe, meta_oe = rollout_control_modules(
        adata_oe,
        CKPT,
        MODULES,
        seed_low=True,
        t1=t1,
        device=device,
        latent_shift_direction=shift_direction,
        latent_shift_scale=shift_scale,
    )
    oe["condition"] = "Atf3_OE"
    both = pd.concat([wt, oe], ignore_index=True)
    both.to_csv(OUT / "Atf3_OE_Control_tracks.csv", index=False)

    # Empirical SNI 2d reference on WT expression space
    neu_wt = _neuron(adata)
    scores_wt = _module_scores(neu_wt, MODULES)
    cond_wt = neu_wt.obs["condition"].astype(str).values
    ref_rows = []
    for cond in ["Control", "SNI 24h", "SNI 2d", "SNI 7d", "SNI 14d"]:
        m = cond_wt == cond
        if m.sum() < 5:
            continue
        row = {"condition": cond, "n": int(m.sum())}
        for name, arr in scores_wt.items():
            row[name] = float(np.nanmean(arr[m]))
        ref_rows.append(row)
    ref = pd.DataFrame(ref_rows)
    ref.to_csv(OUT / "empirical_condition_module_means.csv", index=False)

    stats = pd.DataFrame([_endpoint_row(wt, oe, m, f"Control_lowSNIIC2_to_{t1:g}d") for m in MODULES])
    stats.to_csv(OUT / "Atf3_OE_endpoint_stats.csv", index=False)

    _plot_tracks(
        both,
        ["Atf3_alone", "SNIIC1_noAtf3", "SNIIC3_noAtf3", "Nav_triad", "Csf1", "SNIIC2_noAtf3"],
        title=rf"Control$\rightarrow${t1:g}d: $\mathit{{Atf3}}$-OE $\times{expr_factor:g}$ sufficiency test",
        outfile=OUT / "Atf3_OE_Control_tracks.png",
        ncols=3,
    )
    _plot_tracks(
        both,
        ["SNIIC1_noAtf3", "Nav_triad", "Csf1", "Atf3_alone"],
        title=rf"Does $\mathit{{Atf3}}$-OE push Control into the injury program?",
        outfile=PANEL / "Fig2_Atf3_OE_sufficiency.png",
        ncols=2,
    )

    s = stats.set_index("module")

    def rr(m: str) -> float:
        return float(s.loc[m, "end_ratio_pert_over_WT"])

    partners_up = bool(rr("SNIIC1_noAtf3") > 1.3 or rr("SNIIC3_noAtf3") > 1.3)
    nav_down = bool(rr("Nav_triad") < 0.9)
    csf_up = bool(rr("Csf1") > 1.3)
    # Fraction of gap closed toward SNI 2d empirical mean (partners)
    ref2 = ref.set_index("condition")
    gap_rows = {}
    for m in ["SNIIC1_noAtf3", "SNIIC3_noAtf3", "Nav_triad", "Csf1"]:
        ctrl_m = float(ref2.loc["Control", m])
        inj_m = float(ref2.loc["SNI 2d", m])
        oe_m = float(s.loc[m, "end_pert"])
        denom = inj_m - ctrl_m
        frac = (oe_m - ctrl_m) / denom if abs(denom) > 1e-8 else np.nan
        gap_rows[m] = {
            "Control_emp": ctrl_m,
            "SNI2d_emp": inj_m,
            "OE_end": oe_m,
            "WT_end": float(s.loc[m, "end_WT"]),
            "fraction_gap_closed": float(frac) if np.isfinite(frac) else np.nan,
        }
    if partners_up and nav_down:
        tag = "SUFFICIENT_TO_PUSH"
    elif partners_up or nav_down or csf_up:
        tag = "PARTIAL_SUFFICIENCY"
    else:
        tag = "MAINTENANCE_ONLY"

    verdict = {
        "perturbation": "Atf3_OE",
        "expr_factor": expr_factor,
        "ko_mode": "hybrid",
        "direction_tag": direction_tag,
        "latent_shift_scale": shift_scale,
        "t1": t1,
        "seed_meta_WT": meta_wt,
        "seed_meta_OE": meta_oe,
        "ratios": {
            "SNIIC1_partners": rr("SNIIC1_noAtf3"),
            "SNIIC2_partners": rr("SNIIC2_noAtf3"),
            "SNIIC3_partners": rr("SNIIC3_noAtf3"),
            "Nav_triad": rr("Nav_triad"),
            "Csf1": rr("Csf1"),
            "Atf3_alone": rr("Atf3_alone"),
        },
        "gap_to_SNI2d": gap_rows,
        "oe_verdict": tag,
        "caveats": [
            "Injury partners scored Atf3-removed; Atf3_alone is the direct OE readout.",
            "Hybrid remapping can dominate early; flat OE tracks imply state jump at t0.",
            "Sufficiency judged vs WT Control rollout, with empirical SNI 2d as reference ceiling.",
        ],
        "interpretation": {
            "SUFFICIENT_TO_PUSH": "OE raises injury partners and lowers Nav from Control → closer to sufficiency as an initiator.",
            "PARTIAL_SUFFICIENCY": "OE moves only part of the injury/excitability axis from Control.",
            "MAINTENANCE_ONLY": "OE fails to push Control into the injury program → prefer maintenance-factor wording over driver/initiator.",
        }.get(tag, ""),
    }
    (OUT / "Atf3_OE_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2), flush=True)
    return verdict


# ---------------------------------------------------------------------------
# (4) Path cost: enter vs exit
# ---------------------------------------------------------------------------

def _make_U0_func(checkpoint: Path, device: str = "cpu"):
    from train_model import Config, PotentialNetwork

    state = torch.load(checkpoint / "best_model.pth", map_location=device)
    latent_dim = int(state["momentum_net.net.0.weight"].shape[1]) - 1
    cfg = Config()
    cfg.hidden_dim = latent_dim
    cfg.potential_time_mode = "quasi_stationary"
    pot = PotentialNetwork(latent_dim, cfg)
    pot_state = {k.replace("potential_net.", ""): v for k, v in state.items() if k.startswith("potential_net.")}
    pot.load_state_dict(pot_state, strict=False)
    pot.to(device).eval()

    @torch.no_grad()
    def U0(z):
        z = np.asarray(z, dtype=float).reshape(1, -1)
        t = torch.tensor(z, dtype=torch.float32, device=device)
        return float(pot.stationary_potential(t).cpu().numpy().reshape(-1)[0])

    return U0, pot, device


def _path_metrics(path: np.ndarray, U_func) -> Dict:
    from hamiltonian_flow import hamiltonian_action_score

    path = np.asarray(path, dtype=float)
    u = np.array([float(U_func(p)) for p in path], dtype=float)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    path_len = float(seg.sum()) if len(seg) else 0.0
    action = float(hamiltonian_action_score(path, U_func))
    return {
        "n_points": int(len(path)),
        "path_length": path_len,
        "action": action,
        "delta_U": float(u[-1] - u[0]) if len(u) else np.nan,
        "U_start": float(u[0]) if len(u) else np.nan,
        "U_end": float(u[-1]) if len(u) else np.nan,
        "mean_U_along_path": float(np.mean(u)) if len(u) else np.nan,
        "max_U_along_path": float(np.max(u)) if len(u) else np.nan,
        "min_U_along_path": float(np.min(u)) if len(u) else np.nan,
        "action_per_length": float(action / path_len) if path_len > 1e-12 else np.nan,
        "barrier_above_start": float(np.max(u) - u[0]) if len(u) else np.nan,
    }


def _select_core(z: np.ndarray, scores: np.ndarray, *, low: bool, frac: float = 0.25, min_n: int = 20) -> np.ndarray:
    n = len(scores)
    k = max(min_n, int(np.ceil(frac * n)))
    k = min(k, n)
    order = np.argsort(scores) if low else np.argsort(-scores)
    return order[:k]


def run_path_cost(device: Optional[str] = None, n_boot: int = 40, *, run_lap: bool = False) -> Dict:
    from plot_utils import PALETTE, configure_headless, style_axis
    from panel_style import apply_panel_title_rc, set_panel_title

    configure_headless()
    device = device or ("cpu" if not torch.cuda.is_available() else "cpu")
    print(f"[path] device={device}", flush=True)

    lat = np.load(CKPT / "latent_embeddings.npz", allow_pickle=True)
    z_all = np.asarray(lat["X_latent"], dtype=float)
    index = np.asarray(lat["index"]).astype(str)
    obs = pd.read_csv(CKPT / "obs.csv", low_memory=False)
    # obs.csv uses integer row labels; barcodes live in Unnamed: 0 / barcode.
    if "barcode" in obs.columns:
        obs.index = obs["barcode"].astype(str)
    elif "Unnamed: 0" in obs.columns:
        obs.index = obs["Unnamed: 0"].astype(str)
    else:
        obs.index = obs.index.astype(str)
    common = [i for i in index if i in set(obs.index)]
    if len(common) < 1000:
        raise RuntimeError(f"Latent/obs barcode overlap too small: {len(common)}")
    idx_map = {b: i for i, b in enumerate(index)}
    keep = [idx_map[b] for b in common]
    z_all = z_all[keep]
    obs = obs.loc[common].copy()
    is_neu = obs["annotation"].astype(str).values == "Neuron"
    if int(is_neu.sum()) < 100:
        raise RuntimeError(f"Too few neurons after align: {int(is_neu.sum())}")
    neu = obs.loc[is_neu].copy()
    z_neu = z_all[is_neu]
    cond = neu["condition"].astype(str).values
    u0_obs = neu["potential_stationary"].astype(float).values
    relu = neu["potential_relative_type"].astype(float).values
    print(f"[path] aligned n={len(common)} neurons={int(is_neu.sum())}", flush=True)

    U0, _, _ = _make_U0_func(CKPT, device=device)

    # Endpoint definitions
    defs = {
        "enter_Control_to_SNI2d": {
            "start_cond": "Control",
            "end_cond": "SNI 2d",
            "start_prefer_high_U": True,  # shallow / high relU
            "end_prefer_low_U": True,  # deep
        },
        "exit_SNI7d_to_SNI14d": {
            "start_cond": "SNI 7d",
            "end_cond": "SNI 14d",
            "start_prefer_high_U": False,  # start in deep 7d
            "end_prefer_low_U": False,  # end toward recovered/shallower 14d
        },
    }

    rows = []
    path_store = {}
    rng = np.random.default_rng(0)

    for name, d in defs.items():
        m_s = cond == d["start_cond"]
        m_e = cond == d["end_cond"]
        # Core by relative U (injury-aligned): high relU = shallow, low = deep
        start_core = _select_core(z_neu[m_s], relu[m_s], low=not d["start_prefer_high_U"])
        end_core = _select_core(z_neu[m_e], relu[m_e], low=d["end_prefer_low_U"])
        z_s = z_neu[m_s][start_core]
        z_e = z_neu[m_e][end_core]
        start = z_s.mean(axis=0)
        end = z_e.mean(axis=0)

        # Geodesic
        geo = np.linspace(start, end, 25)
        geo_m = _path_metrics(geo, U0)
        row = {
            "path": name,
            "method": "geodesic",
            "n_start_core": int(len(start_core)),
            "n_end_core": int(len(end_core)),
            "obs_delta_U_mean": float(u0_obs[m_e].mean() - u0_obs[m_s].mean()),
            "obs_delta_relU_mean": float(relu[m_e].mean() - relu[m_s].mean()),
            **geo_m,
        }
        rows.append(row)
        path_store[name] = {"geodesic": geo, "U_geo": np.array([U0(p) for p in geo])}

        # Optional LAP (slow in 512-d; off by default — geodesic is primary)
        if run_lap:
            from hamiltonian_flow import optimize_hamiltonian_action_path

            print(f"[path] optimizing LAP for {name}...", flush=True)
            try:
                man_idx = rng.choice(len(z_neu), size=min(4000, len(z_neu)), replace=False)
                lap_path, lap_action, ok, meta = optimize_hamiltonian_action_path(
                    start,
                    end,
                    U0,
                    z_neu[man_idx],
                    n_points=16,
                    project_to_manifold=True,
                    max_iter=40,
                )
                lap_m = _path_metrics(lap_path, U0)
                rows.append(
                    {
                        "path": name,
                        "method": "LAP",
                        "n_start_core": int(len(start_core)),
                        "n_end_core": int(len(end_core)),
                        "obs_delta_U_mean": float(u0_obs[m_e].mean() - u0_obs[m_s].mean()),
                        "obs_delta_relU_mean": float(relu[m_e].mean() - relu[m_s].mean()),
                        "optimizer_success": bool(ok),
                        "unstable": bool(meta.get("hamiltonian_path_unstable", False)),
                        **lap_m,
                    }
                )
                path_store[name]["LAP"] = lap_path
                path_store[name]["U_LAP"] = np.array([U0(p) for p in lap_path])
            except Exception as exc:  # noqa: BLE001
                print(f"[path] LAP failed for {name}: {exc}", flush=True)
                rows.append({"path": name, "method": "LAP", "error": str(exc)})

        # Bootstrap random endpoint pairs (geodesic only)
        boot_du, boot_S, boot_len = [], [], []
        s_idx = np.where(m_s)[0]
        e_idx = np.where(m_e)[0]
        if len(s_idx) == 0 or len(e_idx) == 0:
            print(f"[path] skip bootstrap for {name}: empty pools", flush=True)
        else:
            for _ in range(n_boot):
                i = int(rng.choice(s_idx))
                j = int(rng.choice(e_idx))
                p = np.linspace(z_neu[i], z_neu[j], 15)
                m = _path_metrics(p, U0)
                boot_du.append(m["delta_U"])
                boot_S.append(m["action"])
                boot_len.append(m["path_length"])
            rows.append(
                {
                    "path": name,
                    "method": "bootstrap_geodesic_mean",
                    "n_boot": n_boot,
                    "delta_U": float(np.mean(boot_du)),
                    "delta_U_std": float(np.std(boot_du)),
                    "action": float(np.mean(boot_S)),
                    "action_std": float(np.std(boot_S)),
                    "path_length": float(np.mean(boot_len)),
                    "path_length_std": float(np.std(boot_len)),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "path_cost_metrics.csv", index=False)

    geo = df[df.method == "geodesic"].set_index("path")
    enter = geo.loc["enter_Control_to_SNI2d"]
    exit_ = geo.loc["exit_SNI7d_to_SNI14d"]

    # ΔU sign asymmetry is the primary physical claim; raw action magnitudes are similar here.
    enter_downhill = bool(enter["delta_U"] < 0)
    exit_uphill = bool(exit_["delta_U"] > 0)
    action_asym = float(exit_["action"] / enter["action"]) if abs(enter["action"]) > 1e-12 else np.nan
    len_asym = float(exit_["path_length"] / enter["path_length"]) if enter["path_length"] > 1e-12 else np.nan
    du_asym_ratio = float(abs(exit_["delta_U"]) / abs(enter["delta_U"])) if abs(enter["delta_U"]) > 1e-12 else np.nan
    if enter_downhill and exit_uphill:
        path_tag = "DELTA_U_ASYMMETRIC_ENTER_DOWN_EXIT_UP"
    elif enter_downhill and not exit_uphill:
        path_tag = "ENTER_DOWNHILL_EXIT_NOT_CLIMB"
    else:
        path_tag = "NO_CLEAR_ASYMMETRY"

    verdict = {
        "enter_delta_U": float(enter["delta_U"]),
        "exit_delta_U": float(exit_["delta_U"]),
        "enter_action": float(enter["action"]),
        "exit_action": float(exit_["action"]),
        "enter_path_length": float(enter["path_length"]),
        "exit_path_length": float(exit_["path_length"]),
        "enter_barrier_above_start": float(enter["barrier_above_start"]),
        "exit_barrier_above_start": float(exit_["barrier_above_start"]),
        "enter_obs_delta_relU": float(enter["obs_delta_relU_mean"]),
        "exit_obs_delta_relU": float(exit_["obs_delta_relU_mean"]),
        "action_exit_over_enter": action_asym,
        "length_exit_over_enter": len_asym,
        "abs_deltaU_exit_over_enter": du_asym_ratio,
        "path_verdict": path_tag,
        "honest_claim": (
            "Primary claim is ΔU0 sign asymmetry (enter downhill, exit uphill). "
            "Raw geodesic action and path length are nearly matched (exit/enter≈0.95) — "
            "do NOT cite action ratio as evidence of a hard chronic lock. "
            "relU changes are large; absolute U0 changes are ~1e-3."
        ),
        "interpretation": {
            "DELTA_U_ASYMMETRIC_ENTER_DOWN_EXIT_UP": (
                "Control→SNI2d descends U0 (no barrier); 7d→14d climbs U0. "
                "Quasi-potential supports easier energetic slide-in than climb-out, "
                "while action integrals remain comparable along matched-length geodesics."
            ),
            "ENTER_DOWNHILL_EXIT_NOT_CLIMB": "Enter is downhill but exit is not a clear climb — weak asymmetry.",
            "NO_CLEAR_ASYMMETRY": "No clear enter/exit energetic asymmetry under geodesic U0.",
        }.get(path_tag, ""),
    }
    (OUT / "path_cost_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    # Figure
    apply_panel_title_rc()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    labs = ["Enter\nControl→2d", "Exit\n7d→14d"]
    axes[0].bar(labs, [enter["delta_U"], exit_["delta_U"]], color=[PALETTE[5], PALETTE[2]])
    axes[0].axhline(0, color="0.5", lw=0.8)
    set_panel_title(axes[0], r"$\Delta U_0$ (geodesic ends)")
    axes[0].set_ylabel(r"$\Delta U_0$")
    style_axis(axes[0], grid_axis="y")

    axes[1].bar(labs, [enter["action"], exit_["action"]], color=[PALETTE[5], PALETTE[2]])
    set_panel_title(axes[1], "Hamiltonian action S")
    axes[1].set_ylabel("S")
    style_axis(axes[1], grid_axis="y")

    axes[2].bar(labs, [enter["path_length"], exit_["path_length"]], color=[PALETTE[5], PALETTE[2]])
    set_panel_title(axes[2], "Path length")
    axes[2].set_ylabel(r"$\|dz\|$ sum")
    style_axis(axes[2], grid_axis="y")
    fig.suptitle("Injury slide-in vs 7d→14d fall-back (neuron cores)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "path_cost_enter_vs_exit.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(PANEL / "Fig2_path_cost_enter_vs_exit.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # U profiles
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for name, color, lab in [
        ("enter_Control_to_SNI2d", PALETTE[5], "Enter Control→2d"),
        ("exit_SNI7d_to_SNI14d", PALETTE[2], "Exit 7d→14d"),
    ]:
        u = path_store[name]["U_geo"]
        x = np.linspace(0, 1, len(u))
        ax.plot(x, u, "-o", color=color, lw=2, ms=3, label=lab)
    ax.set_xlabel("Path fraction")
    ax.set_ylabel(r"$U_0(z)$")
    set_panel_title(ax, r"Geodesic $U_0$ profiles")
    ax.legend(fontsize=8, frameon=False)
    style_axis(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "path_cost_U0_profiles.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(PANEL / "Fig2_path_cost_U0_profiles.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(df.to_string(index=False), flush=True)
    print(json.dumps(verdict, indent=2), flush=True)
    return {"table": df, "verdict": verdict}


def write_combined_report(oe: Optional[Dict], path: Optional[Dict]) -> None:
    lines = [
        "# Atf3 OE sufficiency + path-cost (enter vs exit)",
        "",
    ]
    if oe:
        lines += [
            "## 3. Atf3 OE on Control (sufficiency)",
            "",
            f"Verdict: `{oe.get('oe_verdict')}`",
            "",
            oe.get("interpretation", ""),
            "",
            "| module | OE/WT |",
            "|---|---:|",
        ]
        for k, v in oe.get("ratios", {}).items():
            lines.append(f"| {k} | {v:.3g} |")
        lines += ["", "Gap closed toward empirical SNI 2d:", ""]
        for m, g in oe.get("gap_to_SNI2d", {}).items():
            lines.append(
                f"- **{m}**: fraction_gap_closed={g.get('fraction_gap_closed'):.3g} "
                f"(OE_end={g.get('OE_end'):.3g}, Control={g.get('Control_emp'):.3g}, SNI2d={g.get('SNI2d_emp'):.3g})"
            )
        lines.append("")
    if path:
        v = path.get("verdict", path) if isinstance(path, dict) and "verdict" in path else path
        # path may be {"table","verdict"}
        vv = path["verdict"] if isinstance(path, dict) and "verdict" in path else path
        lines += [
            "## 4. Path cost: enter vs 7d→14d exit",
            "",
            f"Verdict: `{vv.get('path_verdict')}`",
            "",
            vv.get("interpretation", ""),
            "",
            f"- Enter ΔU₀ = **{vv.get('enter_delta_U'):+.4g}**; Exit ΔU₀ = **{vv.get('exit_delta_U'):+.4g}**",
            f"- Enter action = **{vv.get('enter_action'):.4g}**; Exit action = **{vv.get('exit_action'):.4g}** "
            f"(exit/enter = {vv.get('action_exit_over_enter'):.3g})",
            f"- Enter length = **{vv.get('enter_path_length'):.4g}**; Exit length = **{vv.get('exit_path_length'):.4g}**",
            f"- Obs ΔrelU enter/exit = {vv.get('enter_obs_delta_relU'):+.3g} / {vv.get('exit_obs_delta_relU'):+.3g}",
            "",
            vv.get("honest_claim", ""),
            "",
        ]
    lines += [
        "## Figures",
        "",
        f"- `{OUT / 'Atf3_OE_Control_tracks.png'}`",
        f"- `{PANEL / 'Fig2_Atf3_OE_sufficiency.png'}`",
        f"- `{OUT / 'path_cost_enter_vs_exit.png'}`",
        f"- `{OUT / 'path_cost_U0_profiles.png'}`",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["oe", "path", "both"], default="both")
    p.add_argument("--device", default=None)
    p.add_argument("--oe-factor", type=float, default=3.0)
    p.add_argument("--oe-t1", type=float, default=2.0)
    args = p.parse_args()

    oe_v = path_v = None
    if args.only in ("path", "both"):
        path_v = run_path_cost(device="cpu")
    if args.only in ("oe", "both"):
        oe_v = run_atf3_oe(device=args.device, expr_factor=args.oe_factor, t1=args.oe_t1)
    write_combined_report(oe_v, path_v)
    print(f"[done] {OUT}", flush=True)


if __name__ == "__main__":
    main()
