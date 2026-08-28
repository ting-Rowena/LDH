#!/usr/bin/env python3
"""GSE141259 three-axis dynamical mining on the potential landscape.

1. Barrier / action matrix (forward + reverse LAPs)
2. Krt8+ bifurcation energy tilt / fate bias
3. U-covariate gene screen (expression vs U_rel)

Outputs → <CK>/analysis_protocol_GSE141259/figures/ and output_file/mac_landscape_audit/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import plot_mac_alv_3d_potential_landscape as L  # noqa: E402
from analyze_mac_alv_dynamics_first_paths import ALV_TYPES, MAC_TYPES  # noqa: E402
from dataset_pipeline import GSE141259, resolve_data_path  # noqa: E402
from panel_style import apply_panel_title_rc, apply_ygrid, set_panel_title  # noqa: E402
from plot_utils import INK, MUTED, configure_headless  # noqa: E402

configure_headless()
apply_panel_title_rc()
mpl.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

PANELS = L.PANELS
TABLES = ROOT / "output_file" / "mac_landscape_audit"
PROTO = L.PROTO
for p in (PANELS, TABLES, PROTO / "figures", PROTO / "tables"):
    p.mkdir(parents=True, exist_ok=True)

SHORT = dict(L.SHORT)
SHORT.update(
    {
        "Activated AT2 cells": "Activated AT2",
        "MHC-II+ Club cells": "MHC-II+ Club",
        "Club cells": "Club",
        "Ciliated cells": "Ciliated",
        "Goblet cells": "Goblet",
    }
)


def _short(t: str) -> str:
    return SHORT.get(t, t)


# ---------------------------------------------------------------------------
# 1) Barrier / action matrices
# ---------------------------------------------------------------------------

ALV_EDGES = [
    ("AT2 cells", "Activated AT2 cells"),
    ("AT2 cells", "Krt8 ADI"),
    ("Activated AT2 cells", "Krt8 ADI"),
    ("Krt8 ADI", "AT1 cells"),
]
MAC_EDGES = [
    ("AM (PBS)", "AM (Bleo)"),
    ("AM (Bleo)", "Resolution macrophages"),
    ("M2 macrophages", "Resolution macrophages"),
    ("Fn1+ macrophages", "Resolution macrophages"),
    ("Cd163-/Cd11c+ IMs", "Cd163+/Cd11c- IMs"),
    ("Cd163+/Cd11c- IMs", "Resolution macrophages"),
]
CLUB_EDGES = [
    ("MHC-II+ Club cells", "Krt8 ADI"),
    ("Krt8 ADI", "AT1 cells"),
    ("Club cells", "AT2 cells"),
    ("Club cells", "Ciliated cells"),
    ("Club cells", "Goblet cells"),
    ("MHC-II+ Club cells", "AT2 cells"),
]


def _wells_from_adata(adata, types: list[str], min_n: int = 5) -> dict:
    xy = np.asarray(adata.obsm["X_umap"], float)
    labels = adata.obs["cell.type"].astype(str).to_numpy()
    out = {}
    for t in types:
        m = labels == t
        if int(m.sum()) >= min_n:
            out[t] = L._spatial_median_centroid(xy, m)
    return out


def _path_metrics(field, a: np.ndarray, b: np.ndarray) -> dict:
    p = L._resolve_path(field, a, b, try_flow=False)
    U = np.asarray(p["U_path"], float)
    return {
        "method": p.get("method", ""),
        "U_start": float(U[0]),
        "U_end": float(U[-1]),
        "delta_U_end_start": float(U[-1] - U[0]),
        "barrier_height": float(p.get("barrier_height", np.nan)),
        "has_barrier": bool(p.get("has_barrier", False)),
        "is_strict_saddle": bool(p.get("is_strict_saddle", False)),
        "path_action": float(p.get("action", np.nan)),
        "n_points": int(len(U)),
    }


def build_barrier_panel(name: str, adata, edges: list[tuple[str, str]], types: list[str]):
    print(f"===== Barrier matrix: {name} =====", flush=True)
    field = L._build_field(adata, n_grid=110, max_fit=None, smooth_sigma=3.6)
    wells = _wells_from_adata(adata, types)
    rows = []
    for src, dst in edges:
        if src not in wells or dst not in wells:
            print(f"  skip {src}→{dst} (missing well)", flush=True)
            continue
        print(f"  {src} → {dst}", flush=True)
        fwd = _path_metrics(field, wells[src], wells[dst])
        rev = _path_metrics(field, wells[dst], wells[src])
        rows.append(
            {
                "panel": name,
                "src": src,
                "dst": dst,
                "direction": "forward",
                **fwd,
                "reverse_barrier_height": rev["barrier_height"],
                "reverse_path_action": rev["path_action"],
                "reverse_delta_U": rev["delta_U_end_start"],
                "asymmetry_barrier": float(rev["barrier_height"] - fwd["barrier_height"]),
                "asymmetry_action": float(rev["path_action"] - fwd["path_action"]),
            }
        )
        rows.append(
            {
                "panel": name,
                "src": dst,
                "dst": src,
                "direction": "reverse",
                **rev,
                "reverse_barrier_height": fwd["barrier_height"],
                "reverse_path_action": fwd["path_action"],
                "reverse_delta_U": fwd["delta_U_end_start"],
                "asymmetry_barrier": float(fwd["barrier_height"] - rev["barrier_height"]),
                "asymmetry_action": float(fwd["path_action"] - rev["path_action"]),
            }
        )
    return pd.DataFrame(rows), field, wells


def _style_journal_ax(ax):
    ax.tick_params(labelsize=7.5, length=2.2, width=0.55, colors="#475467")
    for sp in ax.spines.values():
        sp.set_color("#CBD5E1")
        sp.set_linewidth(0.7)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.55, color="0.88", zorder=0)
    ax.set_axisbelow(True)


def _black_axes(ax):
    """Force axis spines and ticks to black."""
    ax.tick_params(colors="black")
    for sp in ax.spines.values():
        if sp.get_visible():
            sp.set_color("black")


def _spines_above_bars(ax):
    """Replace buried spines with a single top-layer line (avoid double strokes)."""
    ax.set_axisbelow(False)
    for side, xy in (
        ("left", ([0, 0], [0, 1])),
        ("bottom", ([0, 1], [0, 0])),
        ("right", ([1, 1], [0, 1])),
        ("top", ([0, 1], [1, 1])),
    ):
        sp = ax.spines.get(side)
        if sp is None or not sp.get_visible():
            continue
        color = sp.get_edgecolor()
        lw = sp.get_linewidth()
        # Hide native spine so only the overlay remains (one visible line).
        sp.set_visible(False)
        ax.plot(
            xy[0],
            xy[1],
            transform=ax.transAxes,
            color=color,
            lw=lw,
            solid_capstyle="projecting",
            clip_on=False,
            zorder=100,
        )


def _panel_letter(ax, letter: str, *, x: float = -0.12, y: float = 1.08):
    if not str(letter).strip():
        return
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=INK,
        va="top",
        ha="left",
    )


def plot_barrier_heatmaps(df: pd.DataFrame, panel: str, out: Path):
    sub = df[(df["panel"] == panel) & (df["direction"] == "forward")].copy()
    if sub.empty:
        return
    labels = [f"{_short(r['src'])}→{_short(r['dst'])}" for _, r in sub.iterrows()]
    metrics = [
        ("barrier_height", r"Barrier height $\Delta U$", "#2F6FAD"),
        ("path_action", "Path action", "#B45309"),
        ("asymmetry_barrier", r"Reverse−forward $\Delta U$", None),
    ]
    n = len(sub)
    fig_h = max(3.2, 0.50 * n + 1.9)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.0, fig_h),
        sharey=True,
        gridspec_kw={"wspace": 0.28},
    )
    y = np.arange(n)
    letters = "ABC"
    for ax, (col, title, color), letter in zip(axes, metrics, letters):
        vals = sub[col].to_numpy(float)
        if color is None:
            colors = ["#2F6FAD" if v < 0 else "#C2410C" for v in vals]
            ax.barh(y, vals, color=colors, edgecolor="none", height=0.68, zorder=3)
            ax.axvline(0, color=INK, lw=0.75, zorder=2)
        else:
            ax.barh(y, vals, color=color, edgecolor="none", height=0.68, alpha=0.95, zorder=3)
            if np.nanmin(vals) < 0 < np.nanmax(vals):
                ax.axvline(0, color=INK, lw=0.75, zorder=2)

        # Tight data xlim only — value labels sit in axes-fraction space outside the spine
        # so they do not push path names away from the plot.
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        span = max(vmax - vmin, abs(vmin), abs(vmax), 1e-3)
        left = min(0.0, vmin) - 0.04 * span
        right = max(0.0, vmax) + 0.04 * span
        if vmin >= 0:
            left = -0.02 * span
        if vmax <= 0:
            right = 0.02 * span
        ax.set_xlim(left, right)

        ytrans = ax.get_yaxis_transform()  # x: axes fraction, y: data
        for i, v in enumerate(vals):
            ax.text(
                1.02,
                i,
                f"{v:.3f}",
                transform=ytrans,
                va="center",
                ha="left",
                fontsize=7.0,
                color="#111827",
                clip_on=False,
                zorder=4,
            )

        ax.set_yticks([])
        set_panel_title(ax, title)
        _style_journal_ax(ax)
        _panel_letter(ax, letter, x=-0.02, y=1.10)

    axes[0].invert_yaxis()
    # Native y-tick labels, flush to the left spine (pad in points).
    # Set ticks only on axes[0] after the loop so sharey siblings cannot clear labels.
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7.6)
    axes[0].tick_params(axis="y", which="major", pad=0.5, length=2.0, labelleft=True)
    for lab in axes[0].get_yticklabels():
        lab.set_ha("right")
        lab.set_color("#1F2937")
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False, length=0)

    fig.suptitle(
        f"GSE141259 · {panel} potential-barrier / action matrix (LAP)",
        fontsize=11,
        fontweight="bold",
        color=INK,
        y=1.02,
    )
    fig.subplots_adjust(left=0.01, right=0.92, top=0.86, bottom=0.12, wspace=0.30)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    fig.savefig(PROTO / "figures" / out.name, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote {out}", flush=True)


# ---------------------------------------------------------------------------
# 2) Bifurcation energy tilt / fate bias
# ---------------------------------------------------------------------------

def run_bifurcation_bias() -> pd.DataFrame:
    print("===== Krt8+ bifurcation energy tilt =====", flush=True)
    adata = L._load_parent("alv_epithelium", ALV_TYPES)
    # Include fibroblasts for pathology sink if present in obs — load by cell.type
    obs = pd.read_csv(L.CK / "obs.csv", index_col=0, low_memory=False)
    obs.index = obs.index.astype(str)
    fibro_types = ["Fibroblasts", "Myofibroblasts"]
    # Build joint alv+fibro field for ridge tilt
    types = ALV_TYPES + fibro_types
    adata_j = L._load_cell_types(types)
    field = L._build_field(adata_j, n_grid=120, max_fit=None, smooth_sigma=3.6)
    xy = field["xy"]
    U = field["U"]
    labels = adata_j.obs["cell.type"].astype(str).to_numpy()
    stages = (
        adata_j.obs["stage"].astype(str).to_numpy()
        if "stage" in adata_j.obs.columns
        else np.array(["NA"] * len(labels))
    )

    wells = {
        t: L._spatial_median_centroid(xy, labels == t)
        for t in ["Krt8 ADI", "AT1 cells", "AT2 cells", "Activated AT2 cells"]
        if (labels == t).sum() >= 5
    }
    fibro_mask = np.isin(labels, fibro_types)
    if fibro_mask.any():
        wells["Fibroblast_pathology"] = L._spatial_median_centroid(xy, fibro_mask)

    start = wells.get("Krt8 ADI")
    at1 = wells.get("AT1 cells")
    fibro = wells.get("Fibroblast_pathology")
    if start is None or at1 is None or fibro is None:
        raise RuntimeError("Missing ADI/AT1/Fibro wells for bifurcation bias")

    # Path barriers ADI→AT1 vs ADI→Fibro
    p_at1 = _path_metrics(field, start, at1)
    p_fib = _path_metrics(field, start, fibro)

    # Per-cell tilt among ADI cells: project −∇U (descent) onto fate axes
    adi_ix = np.where(labels == "Krt8 ADI")[0]
    # Finite-difference gradient on field grid via U_func neighbors
    U_func = field["U_func"]
    eps = 0.08 * max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1.0)
    grads = []
    for i in adi_ix:
        p = xy[i]
        dUx = (U_func(p + np.array([eps, 0.0])) - U_func(p - np.array([eps, 0.0]))) / (2 * eps)
        dUy = (U_func(p + np.array([0.0, eps])) - U_func(p - np.array([0.0, eps]))) / (2 * eps)
        grads.append([-float(dUx), -float(dUy)])  # descent direction
    grads = np.asarray(grads, float)
    v_at1 = at1 - start
    v_fib = fibro - start
    v_at1 = v_at1 / (np.linalg.norm(v_at1) + 1e-8)
    v_fib = v_fib / (np.linalg.norm(v_fib) + 1e-8)
    align_at1 = grads @ v_at1
    align_fib = grads @ v_fib
    tilt = align_at1 - align_fib  # >0 prefers AT1 descent

    # Also ΔU to wells from each ADI cell (direct potential drop)
    U_adi = U[adi_ix]
    dU_at1 = float(U_func(at1)) - U_adi
    dU_fib = float(U_func(fibro)) - U_adi
    prefer_at1_by_U = dU_at1 < dU_fib  # larger drop (more negative) to AT1

    cell_df = pd.DataFrame(
        {
            "barcode": adata_j.obs_names.astype(str).to_numpy()[adi_ix],
            "stage": stages[adi_ix],
            "U_rel": U_adi,
            "align_to_AT1": align_at1,
            "align_to_Fibro": align_fib,
            "tilt_AT1_minus_Fibro": tilt,
            "deltaU_to_AT1": dU_at1,
            "deltaU_to_Fibro": dU_fib,
            "prefer_AT1_by_deltaU": prefer_at1_by_U,
        }
    )

    summary = {
        "n_ADI": int(len(adi_ix)),
        "mean_tilt": float(np.nanmean(tilt)),
        "frac_tilt_AT1": float(np.nanmean(tilt > 0)),
        "frac_prefer_AT1_by_deltaU": float(np.nanmean(prefer_at1_by_U)),
        "ADI_to_AT1_barrier": p_at1["barrier_height"],
        "ADI_to_Fibro_barrier": p_fib["barrier_height"],
        "ADI_to_AT1_action": p_at1["path_action"],
        "ADI_to_Fibro_action": p_fib["path_action"],
        "barrier_ratio_Fibro_over_AT1": float(
            p_fib["barrier_height"] / (abs(p_at1["barrier_height"]) + 1e-8)
        ),
    }

    plot_bifurcation_figure(cell_df, summary)

    cell_df.to_csv(TABLES / "GSE141259_krt8_ADI_energy_tilt_cells.csv", index=False)
    pd.DataFrame([summary]).to_csv(TABLES / "GSE141259_krt8_bifurcation_bias_summary.csv", index=False)
    (PROTO / "tables").mkdir(parents=True, exist_ok=True)
    cell_df.to_csv(PROTO / "tables" / "GSE141259_krt8_ADI_energy_tilt_cells.csv", index=False)
    return cell_df, summary, p_at1, p_fib


def plot_bifurcation_figure(cell_df: pd.DataFrame, summary: dict | pd.Series):
    """Journal-style Krt8+ bifurcation panels (no overlapping labels)."""
    if isinstance(summary, pd.DataFrame):
        summary = summary.iloc[0]
    summary = dict(summary)
    AT1_C, FIB_C = "#1B6A7A", "#A65D2E"
    # Full cell-type names (match obs annotations; Fibro well = Fibroblasts ∪ Myofibroblasts)
    path_labels = [
        "Krt8 ADI → AT1 cells",
        "Krt8 ADI → Fibroblasts / Myofibroblasts",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.9), gridspec_kw={"wspace": 0.32})

    ax = axes[0]
    bars = [float(summary["ADI_to_AT1_barrier"]), float(summary["ADI_to_Fibro_barrier"])]
    ax.bar([0, 1], bars, color=[AT1_C, FIB_C], edgecolor="none", width=0.58, zorder=3)
    ax.axhline(0, color=INK, lw=0.7, zorder=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(path_labels, fontsize=6.2, rotation=0, ha="center")
    ax.set_ylabel(r"Barrier height $\Delta U$", fontsize=9)
    set_panel_title(ax, "Path barrier from Krt8 ADI")
    _style_journal_ax(ax)
    ymin, ymax = min(bars + [0.0]), max(bars + [0.0])
    yspan = max(ymax - ymin, 1e-3)
    # Leave headroom above 0 for outside labels on downward bars
    ax.set_ylim(ymin - 0.12 * yspan, 0.22 * yspan)
    for i, v in enumerate(bars):
        ax.text(
            i,
            0.03 * yspan,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#111827",
            clip_on=False,
        )
    _panel_letter(ax, "A", x=-0.16)

    ax = axes[1]
    vals = [float(summary["ADI_to_AT1_action"]), float(summary["ADI_to_Fibro_action"])]
    ax.bar([0, 1], vals, color=[AT1_C, FIB_C], edgecolor="none", width=0.58, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(path_labels, fontsize=6.2, rotation=0, ha="center")
    ax.set_ylabel("Path action", fontsize=9)
    set_panel_title(ax, "Least-action cost")
    _style_journal_ax(ax)
    ax.set_ylim(0, max(vals) * 1.18)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03 * max(vals), f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color="#334155")
    _panel_letter(ax, "B", x=-0.14)

    ax = axes[2]
    stage_order = [
        s
        for s in ["D0", "D2", "D3", "D5", "D7", "D10", "D14", "D21", "D28"]
        if s in set(cell_df["stage"])
    ]
    if not stage_order:
        stage_order = sorted(cell_df["stage"].unique())
    means, sems = [], []
    for s in stage_order:
        x = cell_df.loc[cell_df["stage"] == s, "tilt_AT1_minus_Fibro"].to_numpy(float)
        means.append(float(np.nanmean(x)) if len(x) else np.nan)
        sems.append(float(stats.sem(x, nan_policy="omit")) if len(x) > 1 else 0.0)
    xs = np.arange(len(stage_order))
    ax.axhline(0, color=INK, lw=0.7, zorder=2)
    ax.errorbar(
        xs,
        means,
        yerr=sems,
        fmt="-o",
        color=AT1_C,
        lw=1.6,
        ms=5.5,
        capsize=2.5,
        zorder=3,
    )
    ax.set_xticks(xs, stage_order, rotation=0, ha="center")
    ax.set_ylabel("Descent tilt\n(AT1 cells − Fibroblasts)", fontsize=8.5)
    set_panel_title(ax, "Krt8 ADI cell energy tilt by stage")
    _style_journal_ax(ax)
    ax.text(
        0.98,
        0.97,
        f"mean tilt = {float(summary['mean_tilt']):.3f}\nfrac tilt>0 = {float(summary['frac_tilt_AT1']):.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color="#475467",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#E2E8F0", linewidth=0.6),
    )
    _panel_letter(ax, "C", x=-0.14)

    fig.suptitle(
        "GSE141259 · Krt8$^+$ bifurcation energy tilt / fate bias",
        fontsize=11,
        fontweight="bold",
        color=INK,
        y=1.04,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.80, bottom=0.18, wspace=0.34)
    out = PANELS / "GSE141259_krt8_bifurcation_energy_tilt.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    fig.savefig(PROTO / "figures" / out.name, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print(f"Wrote {out}", flush=True)


# ---------------------------------------------------------------------------
# 2b) Club multi-fate bias (classical airway vs alveolar contribution)
# ---------------------------------------------------------------------------

def run_club_fate_bias():
    """Club → AT2 / Ciliated / Goblet fate costs + MHC-II+ regenerative contrast."""
    print("===== Club multi-fate energy bias =====", flush=True)
    adata = L.load_club_lineage_adata()
    field = L._build_field(adata, n_grid=120, max_fit=None, smooth_sigma=3.8)
    xy = field["xy"]
    U = field["U"]
    U_func = field["U_func"]
    labels = adata.obs["cell.type"].astype(str).to_numpy()
    stages = (
        adata.obs["stage"].astype(str).to_numpy()
        if "stage" in adata.obs.columns
        else np.array(["NA"] * len(labels))
    )

    need = [
        "Club cells",
        "MHC-II+ Club cells",
        "AT2 cells",
        "Ciliated cells",
        "Goblet cells",
        "Krt8 ADI",
        "AT1 cells",
    ]
    wells = {
        t: L._spatial_median_centroid(xy, labels == t)
        for t in need
        if (labels == t).sum() >= 5
    }
    for t in ("Club cells", "AT2 cells", "Ciliated cells", "Goblet cells"):
        if t not in wells:
            raise RuntimeError(f"Missing well for Club fate bias: {t}")

    # Club progenitor → three classical / alveolar fates
    fate_order = ["AT2 cells", "Ciliated cells", "Goblet cells"]
    path_rows = []
    path_metrics = {}
    for dst in fate_order:
        m = _path_metrics(field, wells["Club cells"], wells[dst])
        path_metrics[dst] = m
        path_rows.append(
            {
                "src": "Club cells",
                "dst": dst,
                "branch": "club_trifurcation",
                "barrier_height": m["barrier_height"],
                "path_action": m["path_action"],
                "delta_U_end_start": m["delta_U_end_start"],
            }
        )
        print(
            f"  Club → {dst}: barrier={m['barrier_height']:.3f} action={m['path_action']:.3f}",
            flush=True,
        )

    # MHC-II+ regenerative contrasts
    mhc_paths = {}
    if "MHC-II+ Club cells" in wells:
        for dst, key in (
            ("Krt8 ADI", "MHC_to_ADI"),
            ("AT2 cells", "MHC_to_AT2"),
        ):
            if dst not in wells:
                continue
            m = _path_metrics(field, wells["MHC-II+ Club cells"], wells[dst])
            mhc_paths[key] = m
            path_rows.append(
                {
                    "src": "MHC-II+ Club cells",
                    "dst": dst,
                    "branch": "mhc_regenerative",
                    "barrier_height": m["barrier_height"],
                    "path_action": m["path_action"],
                    "delta_U_end_start": m["delta_U_end_start"],
                }
            )
            print(
                f"  MHC-II+ Club → {dst}: barrier={m['barrier_height']:.3f} "
                f"action={m['path_action']:.3f}",
                flush=True,
            )
    if "Krt8 ADI" in wells and "AT1 cells" in wells:
        m = _path_metrics(field, wells["Krt8 ADI"], wells["AT1 cells"])
        mhc_paths["ADI_to_AT1"] = m
        path_rows.append(
            {
                "src": "Krt8 ADI",
                "dst": "AT1 cells",
                "branch": "mhc_regenerative",
                "barrier_height": m["barrier_height"],
                "path_action": m["path_action"],
                "delta_U_end_start": m["delta_U_end_start"],
            }
        )

    # Per-cell descent tilt among Club cells: AT2 vs airway (mean Ciliated/Goblet)
    club_ix = np.where(labels == "Club cells")[0]
    eps = 0.08 * max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1.0)
    grads = []
    for i in club_ix:
        p = xy[i]
        dUx = (U_func(p + np.array([eps, 0.0])) - U_func(p - np.array([eps, 0.0]))) / (2 * eps)
        dUy = (U_func(p + np.array([0.0, eps])) - U_func(p - np.array([0.0, eps]))) / (2 * eps)
        grads.append([-float(dUx), -float(dUy)])
    grads = np.asarray(grads, float)
    start = wells["Club cells"]

    def _unit(dst_name: str) -> np.ndarray:
        v = wells[dst_name] - start
        return v / (np.linalg.norm(v) + 1e-8)

    v_at2 = _unit("AT2 cells")
    v_cil = _unit("Ciliated cells")
    v_gob = _unit("Goblet cells")
    v_air = 0.5 * (v_cil + v_gob)
    v_air = v_air / (np.linalg.norm(v_air) + 1e-8)

    align_at2 = grads @ v_at2
    align_cil = grads @ v_cil
    align_gob = grads @ v_gob
    align_air = grads @ v_air
    tilt = align_at2 - align_air  # >0 prefers alveolar (AT2) over airway

    U_club = U[club_ix]
    dU_at2 = float(U_func(wells["AT2 cells"])) - U_club
    dU_cil = float(U_func(wells["Ciliated cells"])) - U_club
    dU_gob = float(U_func(wells["Goblet cells"])) - U_club
    dU_air = 0.5 * (dU_cil + dU_gob)
    prefer_at2_by_U = dU_at2 < dU_air

    cell_df = pd.DataFrame(
        {
            "barcode": adata.obs_names.astype(str).to_numpy()[club_ix],
            "stage": stages[club_ix],
            "U_rel": U_club,
            "align_to_AT2": align_at2,
            "align_to_Ciliated": align_cil,
            "align_to_Goblet": align_gob,
            "align_to_airway": align_air,
            "tilt_AT2_minus_airway": tilt,
            "deltaU_to_AT2": dU_at2,
            "deltaU_to_Ciliated": dU_cil,
            "deltaU_to_Goblet": dU_gob,
            "prefer_AT2_by_deltaU": prefer_at2_by_U,
        }
    )

    summary = {
        "n_Club": int(len(club_ix)),
        "mean_tilt_AT2_minus_airway": float(np.nanmean(tilt)),
        "frac_tilt_AT2": float(np.nanmean(tilt > 0)),
        "frac_prefer_AT2_by_deltaU": float(np.nanmean(prefer_at2_by_U)),
        "Club_to_AT2_barrier": path_metrics["AT2 cells"]["barrier_height"],
        "Club_to_Ciliated_barrier": path_metrics["Ciliated cells"]["barrier_height"],
        "Club_to_Goblet_barrier": path_metrics["Goblet cells"]["barrier_height"],
        "Club_to_AT2_action": path_metrics["AT2 cells"]["path_action"],
        "Club_to_Ciliated_action": path_metrics["Ciliated cells"]["path_action"],
        "Club_to_Goblet_action": path_metrics["Goblet cells"]["path_action"],
    }
    if "MHC_to_ADI" in mhc_paths:
        summary["MHC_to_ADI_barrier"] = mhc_paths["MHC_to_ADI"]["barrier_height"]
        summary["MHC_to_ADI_action"] = mhc_paths["MHC_to_ADI"]["path_action"]
    if "MHC_to_AT2" in mhc_paths:
        summary["MHC_to_AT2_barrier"] = mhc_paths["MHC_to_AT2"]["barrier_height"]
        summary["MHC_to_AT2_action"] = mhc_paths["MHC_to_AT2"]["path_action"]
    if "ADI_to_AT1" in mhc_paths:
        summary["ADI_to_AT1_barrier"] = mhc_paths["ADI_to_AT1"]["barrier_height"]
        summary["ADI_to_AT1_action"] = mhc_paths["ADI_to_AT1"]["path_action"]
        if "MHC_to_ADI" in mhc_paths:
            summary["MHC_ADI_AT1_action_sum"] = (
                mhc_paths["MHC_to_ADI"]["path_action"] + mhc_paths["ADI_to_AT1"]["path_action"]
            )

    path_df = pd.DataFrame(path_rows)
    plot_club_fate_bias_figure(cell_df, summary)

    cell_df.to_csv(TABLES / "GSE141259_club_fate_tilt_cells.csv", index=False)
    pd.DataFrame([summary]).to_csv(TABLES / "GSE141259_club_fate_bias_summary.csv", index=False)
    path_df.to_csv(TABLES / "GSE141259_club_fate_path_metrics.csv", index=False)
    (PROTO / "tables").mkdir(parents=True, exist_ok=True)
    cell_df.to_csv(PROTO / "tables" / "GSE141259_club_fate_tilt_cells.csv", index=False)
    pd.DataFrame([summary]).to_csv(PROTO / "tables" / "GSE141259_club_fate_bias_summary.csv", index=False)
    path_df.to_csv(PROTO / "tables" / "GSE141259_club_fate_path_metrics.csv", index=False)
    return cell_df, summary, path_df


def plot_club_fate_bias_figure(cell_df: pd.DataFrame, summary: dict | pd.Series):
    """Compact 3-panel Club fate figure (journal style).

    A · Club multipotent path costs (ΔU + action, shared fates)
    B · Stage-wise descent tilt (AT2 vs airway)
    C · MHC-II⁺ regenerative route costs
    """
    if isinstance(summary, pd.DataFrame):
        summary = summary.iloc[0]
    summary = dict(summary)

    # Match narrative combined row-1 bar color
    BAR_C = "#A0C7DB"
    fate_short = ["→ AT2", "→ Ciliated", "→ Goblet"]
    fate_colors = [BAR_C, BAR_C, BAR_C]
    barriers = [
        float(summary["Club_to_AT2_barrier"]),
        float(summary["Club_to_Ciliated_barrier"]),
        float(summary["Club_to_Goblet_barrier"]),
    ]
    actions = [
        float(summary["Club_to_AT2_action"]),
        float(summary["Club_to_Ciliated_action"]),
        float(summary["Club_to_Goblet_action"]),
    ]

    out_w, out_h, out_dpi = 3050, 800, 300
    fig = plt.figure(figsize=(out_w / out_dpi, out_h / out_dpi), facecolor="white", dpi=out_dpi)
    outer = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.28, 1.05, 1.22],
        wspace=0.22,
        left=0.07,
        right=0.985,
        top=0.88,
        bottom=0.18,
    )
    gs_a = outer[0].subgridspec(2, 1, hspace=0.12, height_ratios=[1.0, 1.15])
    ax_a0 = fig.add_subplot(gs_a[0])
    ax_a1 = fig.add_subplot(gs_a[1], sharex=ax_a0)
    ax_b = fig.add_subplot(outer[1])
    ax_c = fig.add_subplot(outer[2])

    xs = np.arange(3)
    # ---- A top: barriers ----
    ax_a0.axhline(0, color=INK, lw=0.7, zorder=2)
    ax_a0.bar(xs, barriers, color=fate_colors, edgecolor="none", width=0.62, zorder=3)
    ymin, ymax = min(barriers + [0.0]), max(barriers + [0.0])
    yspan = max(ymax - ymin, 1e-3)
    ax_a0.set_ylim(ymin - 0.18 * yspan, ymax + 0.18 * yspan)
    ax_a0.set_ylabel(r"Barrier $\Delta U$", fontsize=7.6)
    set_panel_title(ax_a0, "Club multipotent path costs", pad=4)
    _style_journal_ax(ax_a0)
    ax_a0.tick_params(axis="x", labelbottom=False, length=0)

    # ---- A bottom: actions ----
    ax_a1.bar(xs, actions, color=fate_colors, edgecolor="none", width=0.62, zorder=3)
    ax_a1.set_ylim(0, max(actions) * 1.12)
    ax_a1.set_xticks(xs, fate_short, fontsize=7.2)
    ax_a1.set_xlabel("Fate from Club cells", fontsize=7.6, labelpad=2)
    ax_a1.set_ylabel("Path action", fontsize=7.6)
    _style_journal_ax(ax_a1)

    # ---- B: stage tilt ----
    stage_order = [
        s
        for s in ["D0", "D2", "D3", "D5", "D7", "D10", "D14", "D21", "D28"]
        if s in set(cell_df["stage"])
    ]
    if not stage_order:
        stage_order = sorted(cell_df["stage"].unique())
    means, sems = [], []
    for s in stage_order:
        x = cell_df.loc[cell_df["stage"] == s, "tilt_AT2_minus_airway"].to_numpy(float)
        means.append(float(np.nanmean(x)) if len(x) else np.nan)
        sems.append(float(stats.sem(x, nan_policy="omit")) if len(x) > 1 else 0.0)
    xs_s = np.arange(len(stage_order))
    # Match narrative combined row-1 tilt curve color
    CURVE_C = "#EC7CBB"
    ax_b.axhline(0, color=INK, lw=0.7, zorder=2)
    ax_b.fill_between(
        xs_s,
        np.asarray(means) - np.asarray(sems),
        np.asarray(means) + np.asarray(sems),
        color=CURVE_C,
        alpha=0.12,
        linewidth=0,
        zorder=2,
    )
    ax_b.plot(
        xs_s,
        means,
        "-o",
        color=CURVE_C,
        lw=1.9,
        ms=5.6,
        markerfacecolor=CURVE_C,
        markeredgecolor=CURVE_C,
        markeredgewidth=0.8,
        zorder=3,
    )
    ax_b.set_xticks(xs_s, stage_order, rotation=0, ha="center", fontsize=6.8)
    ax_b.set_ylabel("Descent tilt (AT2 − airway)", fontsize=7.6)
    set_panel_title(ax_b, "Club local fate bias by stage", pad=4)
    _style_journal_ax(ax_b)
    ax_b.text(
        0.98,
        0.97,
        f"mean = {float(summary['mean_tilt_AT2_minus_airway']):.2f}\n"
        f"tilt>0 = {float(summary['frac_tilt_AT2']):.0%}",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color="#475467",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#E2E8F0", linewidth=0.6),
    )

    # ---- C: MHC regenerative routes (action) + barrier callouts ----
    route_labs, route_vals, route_cols, route_notes = [], [], [], []
    if "MHC_ADI_AT1_action_sum" in summary:
        note = ""
        if "MHC_to_ADI_barrier" in summary:
            note = rf"$\Delta U_{{\mathrm{{ADI}}}}$={float(summary['MHC_to_ADI_barrier']):+.2f}"
        route_labs.append("MHC-II⁺→ADI→AT1")
        route_vals.append(float(summary["MHC_ADI_AT1_action_sum"]))
        route_cols.append(BAR_C)
        route_notes.append(note)
    if "MHC_to_AT2_action" in summary:
        note = ""
        if "MHC_to_AT2_barrier" in summary:
            note = rf"$\Delta U$={float(summary['MHC_to_AT2_barrier']):+.2f}"
        route_labs.append("MHC-II⁺→AT2")
        route_vals.append(float(summary["MHC_to_AT2_action"]))
        route_cols.append(BAR_C)
        route_notes.append(note)
    route_labs.append("Club→AT2")
    route_vals.append(float(summary["Club_to_AT2_action"]))
    route_cols.append(BAR_C)
    route_notes.append(
        rf"$\Delta U$={float(summary['Club_to_AT2_barrier']):+.2f}"
    )

    xs_c = np.arange(len(route_vals))
    ax_c.bar(xs_c, route_vals, color=route_cols, edgecolor="none", width=0.64, zorder=3)
    ax_c.set_xticks(xs_c, route_labs, fontsize=6.4)
    ax_c.set_ylabel("Path action", fontsize=7.6)
    set_panel_title(ax_c, r"MHC-II$^+$ regenerative route costs", pad=4)
    _style_journal_ax(ax_c)
    ax_c.set_ylim(0, max(route_vals) * 1.28)
    for i, (v, note) in enumerate(zip(route_vals, route_notes)):
        if note:
            ax_c.text(
                i,
                v + 0.04 * max(route_vals),
                note,
                ha="center",
                va="bottom",
                fontsize=5.6,
                color="#667085",
            )

    out = PANELS / "GSE141259_club_fate_bias.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"savefig.bbox": None}):
        fig.savefig(out, dpi=out_dpi, facecolor="white")
        fig.savefig(PROTO / "figures" / out.name, dpi=out_dpi, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}", flush=True)
    return out


def plot_alv_barrier_bifurcation_combined(
    barriers: pd.DataFrame,
    cell_df: pd.DataFrame,
    summary: dict | pd.Series,
    *,
    out: Path | None = None,
):
    """Merge alveolar LAP barrier matrix + Krt8+ bifurcation tilt into one figure.

    Row 1 (A–C): alveolar forward-path barrier / action / reverse−forward ΔU
    Row 2 (D–F): Krt8 ADI→AT1 vs Fibro barrier, action, and stage-wise tilt
    """
    if isinstance(summary, pd.DataFrame):
        summary = summary.iloc[0]
    summary = dict(summary)
    AT1_C, FIB_C = "#1B6A7A", "#A65D2E"
    path_labels = [
        "Krt8 ADI → AT1 cells",
        "Krt8 ADI → Fibroblasts / Myofibroblasts",
    ]

    sub = barriers[(barriers["panel"] == "Alveolar") & (barriers["direction"] == "forward")].copy()
    if sub.empty:
        raise RuntimeError("No Alveolar forward barrier rows for combined figure")
    labels = [f"{_short(r['src'])}→{_short(r['dst'])}" for _, r in sub.iterrows()]
    metrics = [
        ("barrier_height", r"Barrier height $\Delta U$", "#2F6FAD"),
        ("path_action", "Path action", "#B45309"),
        ("asymmetry_barrier", r"Reverse−forward $\Delta U$", None),
    ]
    n = len(sub)
    y = np.arange(n)

    fig = plt.figure(figsize=(13.6, 7.6), facecolor="white")
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.05, 1.15],
        hspace=0.48,
        wspace=0.32,
        left=0.11,
        right=0.97,
        top=0.86,
        bottom=0.10,
    )
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axes_bot = [fig.add_subplot(gs[1, i]) for i in range(3)]
    # share y among top row
    for ax in axes_top[1:]:
        ax.sharey(axes_top[0])

    # ---- Row 1: alveolar barrier matrix ----
    for ax, (col, title, color), letter in zip(axes_top, metrics, "ABC"):
        vals = sub[col].to_numpy(float)
        if color is None:
            colors = ["#2F6FAD" if v < 0 else "#C2410C" for v in vals]
            ax.barh(y, vals, color=colors, edgecolor="none", height=0.68, zorder=3)
            ax.axvline(0, color=INK, lw=0.75, zorder=2)
        else:
            ax.barh(y, vals, color=color, edgecolor="none", height=0.68, alpha=0.95, zorder=3)
            if np.nanmin(vals) < 0 < np.nanmax(vals):
                ax.axvline(0, color=INK, lw=0.75, zorder=2)

        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        span = max(vmax - vmin, abs(vmin), abs(vmax), 1e-3)
        left = min(0.0, vmin) - 0.06 * span
        right = max(0.0, vmax) + 0.06 * span
        if vmin >= 0:
            left = -0.02 * span
        if vmax <= 0:
            right = 0.02 * span
        ax.set_xlim(left, right)

        ax.set_yticks([])
        set_panel_title(ax, title, pad=6)
        _style_journal_ax(ax)
        _panel_letter(ax, letter, x=-0.04, y=1.12)

    axes_top[0].invert_yaxis()
    axes_top[0].set_yticks(y)
    axes_top[0].set_yticklabels(labels, fontsize=7.2)
    axes_top[0].tick_params(axis="y", which="major", pad=0.5, length=2.0, labelleft=True)
    for lab in axes_top[0].get_yticklabels():
        lab.set_ha("right")
        lab.set_color("#1F2937")
    for ax in axes_top[1:]:
        ax.tick_params(axis="y", labelleft=False, length=0)

    # ---- Row 2: bifurcation ----
    ax = axes_bot[0]
    bars = [float(summary["ADI_to_AT1_barrier"]), float(summary["ADI_to_Fibro_barrier"])]
    ax.bar([0, 1], bars, color=[AT1_C, FIB_C], edgecolor="none", width=0.58, zorder=3)
    ax.axhline(0, color=INK, lw=0.7, zorder=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(path_labels, fontsize=5.8, rotation=0, ha="center")
    ax.set_ylabel(r"Barrier height $\Delta U$", fontsize=8.5)
    set_panel_title(ax, "Path barrier from Krt8 ADI", pad=6)
    _style_journal_ax(ax)
    ymin, ymax = min(bars + [0.0]), max(bars + [0.0])
    yspan = max(ymax - ymin, 1e-3)
    ax.set_ylim(ymin - 0.12 * yspan, 0.22 * yspan)
    for i, v in enumerate(bars):
        ax.text(i, 0.03 * yspan, f"{v:.3f}", ha="center", va="bottom", fontsize=7.0, color="#111827", clip_on=False)
    _panel_letter(ax, "D", x=-0.16, y=1.12)

    ax = axes_bot[1]
    vals = [float(summary["ADI_to_AT1_action"]), float(summary["ADI_to_Fibro_action"])]
    ax.bar([0, 1], vals, color=[AT1_C, FIB_C], edgecolor="none", width=0.58, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(path_labels, fontsize=5.8, rotation=0, ha="center")
    ax.set_ylabel("Path action", fontsize=8.5)
    set_panel_title(ax, "Least-action cost", pad=6)
    _style_journal_ax(ax)
    ax.set_ylim(0, max(vals) * 1.18)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03 * max(vals), f"{v:.2f}", ha="center", va="bottom", fontsize=7.0, color="#334155")
    _panel_letter(ax, "E", x=-0.14, y=1.12)

    ax = axes_bot[2]
    stage_order = [
        s
        for s in ["D0", "D2", "D3", "D5", "D7", "D10", "D14", "D21", "D28"]
        if s in set(cell_df["stage"])
    ]
    if not stage_order:
        stage_order = sorted(cell_df["stage"].unique())
    means, sems = [], []
    for s in stage_order:
        x = cell_df.loc[cell_df["stage"] == s, "tilt_AT1_minus_Fibro"].to_numpy(float)
        means.append(float(np.nanmean(x)) if len(x) else np.nan)
        sems.append(float(stats.sem(x, nan_policy="omit")) if len(x) > 1 else 0.0)
    xs = np.arange(len(stage_order))
    ax.axhline(0, color=INK, lw=0.7, zorder=2)
    ax.errorbar(
        xs,
        means,
        yerr=sems,
        fmt="-o",
        color=AT1_C,
        lw=1.55,
        ms=5.0,
        capsize=2.2,
        zorder=3,
    )
    ax.set_xticks(xs, stage_order, rotation=0, ha="center", fontsize=7.0)
    ax.set_ylabel("Descent tilt\n(AT1 − Fibroblasts)", fontsize=8.0)
    set_panel_title(ax, "Krt8 ADI energy tilt by stage", pad=6)
    _style_journal_ax(ax)
    ax.text(
        0.98,
        0.97,
        f"mean tilt = {float(summary['mean_tilt']):.3f}\nfrac tilt>0 = {float(summary['frac_tilt_AT1']):.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.4,
        color="#475467",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#E2E8F0", linewidth=0.6),
    )
    _panel_letter(ax, "F", x=-0.14, y=1.12)

    fig.suptitle(
        r"GSE141259 · Alveolar LAP barriers and Krt8$^+$ bifurcation energy tilt",
        fontsize=12,
        fontweight="bold",
        color=INK,
        y=0.97,
    )
    # Row captions
    fig.text(
        0.11,
        0.895,
        "Alveolar epithelium · forward LAP matrix",
        fontsize=8.0,
        color="#475467",
        fontweight="bold",
        ha="left",
    )
    fig.text(
        0.11,
        0.48,
        r"Krt8$^+$ ADI bifurcation · AT1 vs Fibroblast fate bias",
        fontsize=8.0,
        color="#475467",
        fontweight="bold",
        ha="left",
    )

    for ax in axes_top + axes_bot:
        _black_axes(ax)

    if out is None:
        out = PANELS / "GSE141259_alv_barrier_and_krt8_bifurcation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    fig.savefig(PROTO / "figures" / out.name, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print(f"Wrote combined figure: {out}", flush=True)
    return out


def plot_alv_dynamics_narrative_combined(
    barriers: pd.DataFrame,
    cell_df: pd.DataFrame,
    summary: dict | pd.Series,
    cov: pd.DataFrame,
    *,
    enrichment: pd.DataFrame | None = None,
    out: Path | None = None,
    gene_top_n: int = 10,
    pathway_top_n: int = 8,
):
    """Compact 4-panel alveolar narrative figure.

    A · Forward LAP costs (ΔU / action / reverse−forward; shared transitions)
    B · Krt8⁺ bifurcation (barrier + action + stage tilt)
    C · Bidirectional U_rel gene covariates
    D · Hallmark pathways (co- / anti-varying)
    """
    if isinstance(summary, pd.DataFrame):
        summary = summary.iloc[0]
    summary = dict(summary)
    # Unified journal palette
    ROW1_BAR = "#A0C7DB"
    AT1_C, FIB_C = ROW1_BAR, ROW1_BAR
    CURVE_C = "#EC7CBB"
    path_labels = ["→ AT1", "→ Fibro / Myofibro"]
    x_bar = np.array([0.0, 0.42])
    bar_w = 0.32
    bar_xlim = (-0.26, 0.68)

    sub = barriers[(barriers["panel"] == "Alveolar") & (barriers["direction"] == "forward")].copy()
    if sub.empty:
        raise RuntimeError("No Alveolar forward barrier rows for narrative figure")
    labels = [f"{_short(r['src'])}→{_short(r['dst'])}" for _, r in sub.iterrows()]
    metrics = [
        ("path_action", "Path action", ROW1_BAR),
        ("barrier_height", r"Barrier $\Delta U$", ROW1_BAR),
        ("asymmetry_barrier", r"Rev−fwd $\Delta U$", ROW1_BAR),
    ]
    n = len(sub)
    y = np.arange(n)

    out_w, out_h, out_dpi = 3050, 1800, 300
    fig = plt.figure(figsize=(out_w / out_dpi, out_h / out_dpi), facecolor="white", dpi=out_dpi)
    # Outer left is tight; row 1 adds its own label gutter, row 2 can start further left.
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.7, 1.0],
        hspace=0.30,
        left=0.035,
        right=0.965,
        top=0.95,
        bottom=0.07,
    )
    # Row 1: left gutter for Path-action y-tick labels, then A | B
    gs1_row = outer[0].subgridspec(1, 2, width_ratios=[0.11, 1.0], wspace=0.0)
    gs1 = gs1_row[1].subgridspec(1, 2, width_ratios=[1.25, 1.30], wspace=0.18)
    gs_a = gs1[0].subgridspec(1, 3, wspace=0.22)
    gs_b = gs1[1].subgridspec(1, 2, width_ratios=[0.78, 1.20], wspace=0.26)
    gs_b_left = gs_b[0].subgridspec(2, 1, hspace=0.18)

    axes_a = [fig.add_subplot(gs_a[0, i]) for i in range(3)]
    for ax in axes_a[1:]:
        ax.sharey(axes_a[0])
    ax_b0 = fig.add_subplot(gs_b_left[0])
    ax_b1 = fig.add_subplot(gs_b_left[1])
    ax_b2 = fig.add_subplot(gs_b[1])

    # Row 2: independent 3-column layout (not aligned to row 1).
    # Genes get a wide panel so names clear the center guide / frame.
    gs2 = outer[1].subgridspec(1, 3, width_ratios=[1.55, 1.0, 1.0], wspace=0.58)
    ax_gene = fig.add_subplot(gs2[0])
    ax_path_up = fig.add_subplot(gs2[1])
    ax_path_dn = fig.add_subplot(gs2[2])

    # ---- A: LAP matrix (compound) ----
    for i, (ax, (col, title, color)) in enumerate(zip(axes_a, metrics)):
        vals = sub[col].to_numpy(float)
        ax.barh(
            y,
            vals,
            color=color,
            edgecolor="none",
            height=0.68,
            alpha=0.95,
            zorder=2,
        )

        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        span = max(vmax - vmin, abs(vmin), abs(vmax), 1e-3)
        left = min(0.0, vmin) - 0.06 * span
        right = max(0.0, vmax) + 0.06 * span
        if vmin >= 0:
            left = 0.0  # flush to left spine / zero
        if vmax <= 0:
            # Zero sits on the right spine — do not also draw axvline(0).
            right = 0.0
        elif vmin < 0 < vmax:
            ax.axvline(0, color="#1F2937", lw=0.75, zorder=1)
        ax.set_xlim(left, right)

        ax.set_yticks([])
        set_panel_title(ax, title, pad=3)
        _style_journal_ax(ax)

    axes_a[0].invert_yaxis()
    axes_a[0].set_yticks(y)
    axes_a[0].set_yticklabels(labels, fontsize=6.2)
    axes_a[0].tick_params(axis="y", which="major", pad=0.5, length=2.0, labelleft=True)
    for lab in axes_a[0].get_yticklabels():
        lab.set_ha("right")
        lab.set_color("#1F2937")
    for ax in axes_a[1:]:
        ax.tick_params(axis="y", labelleft=False, length=0)
        # Barrier / Rev−fwd: drop left spine, keep right spine as the panel edge.
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(True)

    # ---- B: bifurcation compound ----
    bars = [float(summary["ADI_to_AT1_barrier"]), float(summary["ADI_to_Fibro_barrier"])]
    ax_b0.bar(x_bar, bars, color=[AT1_C, FIB_C], edgecolor="none", width=bar_w, zorder=2)
    ax_b0.axhline(0, color="#1F2937", lw=0.7, zorder=1)
    ax_b0.set_xticks([])
    ax_b0.set_xlim(*bar_xlim)
    ax_b0.set_ylabel(r"Barrier $\Delta U$", fontsize=7.0)
    set_panel_title(ax_b0, r"Krt8$^+$ ADI fate bias", pad=3)
    _style_journal_ax(ax_b0)
    ymin, ymax = min(bars + [0.0]), max(bars + [0.0])
    yspan = max(ymax - ymin, 1e-3)
    ax_b0.set_ylim(ymin - 0.14 * yspan, 0.28 * yspan)
    for xi, v in zip(x_bar, bars):
        ax_b0.text(xi, 0.04 * yspan, f"{v:.3f}", ha="center", va="bottom", fontsize=6.0, color="#111827")

    acts = [float(summary["ADI_to_AT1_action"]), float(summary["ADI_to_Fibro_action"])]
    ax_b1.bar(x_bar, acts, color=[AT1_C, FIB_C], edgecolor="none", width=bar_w, zorder=2)
    ax_b1.set_xticks(x_bar, path_labels, fontsize=5.8)
    ax_b1.set_xlim(*bar_xlim)
    ax_b1.set_ylabel("Path action", fontsize=7.0)
    _style_journal_ax(ax_b1)
    ax_b1.set_ylim(0, max(acts) * 1.22)
    for xi, v in zip(x_bar, acts):
        ax_b1.text(xi, v + 0.03 * max(acts), f"{v:.2f}", ha="center", va="bottom", fontsize=6.0, color="#334155")

    stage_order = [
        s
        for s in ["D0", "D2", "D3", "D5", "D7", "D10", "D14", "D21", "D28"]
        if s in set(cell_df["stage"])
    ]
    if not stage_order:
        stage_order = sorted(cell_df["stage"].unique())
    means, sems = [], []
    for s in stage_order:
        x = cell_df.loc[cell_df["stage"] == s, "tilt_AT1_minus_Fibro"].to_numpy(float)
        means.append(float(np.nanmean(x)) if len(x) else np.nan)
        sems.append(float(stats.sem(x, nan_policy="omit")) if len(x) > 1 else 0.0)
    xs = np.arange(len(stage_order))
    ax_b2.fill_between(
        xs,
        np.asarray(means) - np.asarray(sems),
        np.asarray(means) + np.asarray(sems),
        color=CURVE_C,
        alpha=0.14,
        linewidth=0,
        zorder=2,
    )
    ax_b2.plot(
        xs,
        means,
        "-o",
        color=CURVE_C,
        lw=1.7,
        ms=5.0,
        markerfacecolor=CURVE_C,
        markeredgecolor=CURVE_C,
        markeredgewidth=0.8,
        zorder=3,
    )
    ax_b2.set_xticks(xs, stage_order, fontsize=6.0)
    ax_b2.set_ylabel("Tilt (AT1 − Fibro)", fontsize=7.0)
    set_panel_title(ax_b2, "Stage-wise energy tilt", pad=3)
    _style_journal_ax(ax_b2)
    ax_b2.text(
        0.98,
        0.97,
        f"mean={float(summary['mean_tilt']):.2f}\ntilt>0={float(summary['frac_tilt_AT1']):.0%}",
        transform=ax_b2.transAxes,
        ha="right",
        va="top",
        fontsize=5.6,
        color="#475467",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="#E2E8F0", linewidth=0.6),
    )

    # ---- C–D: U covariates ----
    plot_u_covariate_figure(
        cov,
        enrichment=enrichment,
        top_n=gene_top_n,
        pathway_top_n=pathway_top_n,
        axes=(ax_gene, ax_path_up, ax_path_dn),
        letters=("", "", ""),
        letter_xy=(-0.10, 1.10),
        save=False,
        show_chrome=False,
        gene_layout="bidirectional",
        pathway_style="bars",
        co_color="#F9DBC4",
        anti_color="#B8D1CB",
        flat_pair_colors=True,
        pathway_order=("down", "up"),
        gene_name_size=8.2,
    )

    for ax in (axes_a + [ax_b0, ax_b1, ax_b2, ax_gene, ax_path_up, ax_path_dn]):
        _black_axes(ax)
        _spines_above_bars(ax)

    if out is None:
        out = PANELS / "GSE141259_alv_dynamics_narrative_combined.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Fixed pixel canvas; override global savefig.bbox='tight'.
    with mpl.rc_context({"savefig.bbox": None}):
        fig.savefig(out, dpi=out_dpi, facecolor="white")
        fig.savefig(PROTO / "figures" / out.name, dpi=out_dpi, facecolor="white")
    plt.close(fig)
    print(f"Wrote narrative combined figure: {out}", flush=True)
    return out


def _load_expr_for_barcodes(barcodes: list[str], gene_cap: int = 3000) -> tuple[np.ndarray, list[str], list[str]]:
    """Return log1p expression matrix aligned to barcodes (subset of training HVGs)."""
    gene_list = json.loads((L.CK / "training_var_names.json").read_text(encoding="utf-8"))
    gene_list = gene_list[:gene_cap]
    h5 = resolve_data_path(GSE141259)
    raw = ad.read_h5ad(h5, backed="r")
    name_to_i = {b: i for i, b in enumerate(raw.obs_names.astype(str))}
    idx = [name_to_i[b] for b in barcodes if b in name_to_i]
    keep_bc = [b for b in barcodes if b in name_to_i]
    present = [g for g in gene_list if g in set(map(str, raw.var_names))]
    # h5py backed AnnData allows only one fancy index at a time.
    sub = raw[idx].to_memory()
    raw.file.close()
    present = [g for g in present if g in set(map(str, sub.var_names))]
    sub = sub[:, present].copy()
    if "log1p" not in sub.uns:
        sc.pp.normalize_total(sub, target_sum=1e4, inplace=True)
        sc.pp.log1p(sub)
    X = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X, float)
    # map back to requested order
    fetched = {b: j for j, b in enumerate(np.asarray(sub.obs_names.astype(str)))}
    out = np.full((len(barcodes), len(present)), np.nan, dtype=float)
    for i, b in enumerate(barcodes):
        j = fetched.get(b)
        if j is not None:
            out[i] = X[j]
    return out, present, keep_bc


def run_u_covariate(top_n: int = 25) -> pd.DataFrame:
    print("===== U-covariate gene screen =====", flush=True)
    # Use alveolar epithelium cells (clearest U_rel biology for injury repair)
    adata = L._load_parent("alv_epithelium", ALV_TYPES)
    barcodes = adata.obs_names.astype(str).tolist()
    U = pd.to_numeric(adata.obs[L.POTENTIAL_KEY], errors="coerce").to_numpy(float)
    labels = adata.obs["cell.type"].astype(str).to_numpy()

    X, genes, _ = _load_expr_for_barcodes(barcodes)
    ok = np.isfinite(U) & np.isfinite(X).any(axis=1)
    U = U[ok]
    X = X[ok]
    labels = labels[ok]

    rows = []
    for j, g in enumerate(genes):
        x = X[:, j]
        m = np.isfinite(x)
        if m.sum() < 50 or np.nanstd(x[m]) < 1e-8:
            continue
        rho, p = stats.spearmanr(U[m], x[m])
        # also Pearson for magnitude
        r, p_p = stats.pearsonr(U[m], x[m])
        rows.append(
            {
                "gene": g,
                "spearman_rho": float(rho),
                "spearman_p": float(p),
                "pearson_r": float(r),
                "pearson_p": float(p_p),
                "mean_expr": float(np.nanmean(x)),
                "frac_expr": float(np.nanmean(x > 0)),
                "n": int(m.sum()),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # BH-ish simple FDR via rank
    df = df.sort_values("spearman_p")
    n = len(df)
    df["fdr_bh"] = np.minimum(1.0, df["spearman_p"].to_numpy() * n / np.arange(1, n + 1))
    df = df.sort_values("spearman_rho", ascending=False).reset_index(drop=True)
    df.to_csv(TABLES / "GSE141259_alv_U_covariate_genes.csv", index=False)
    df.to_csv(PROTO / "tables" / "GSE141259_alv_U_covariate_genes.csv", index=False)

    enr = run_u_covariate_pathway_enrichment(df)
    plot_u_covariate_figure(df, enrichment=enr, top_n=top_n)

    # Compact subtype mean U annotation strip
    meanU = (
        pd.DataFrame({"cell.type": labels, "U": U})
        .groupby("cell.type")["U"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    meanU.to_csv(TABLES / "GSE141259_alv_subtype_mean_U_for_covariate.csv", index=False)
    return df


def _short_go_term(term: str, *, max_len: int = 52) -> str:
    """Strip GO id suffix and truncate for axis labels."""
    t = str(term)
    if "(GO:" in t:
        t = t[: t.rfind("(GO:")].strip()
    t = t.rstrip(" .")
    if len(t) > max_len:
        t = t[: max_len - 1].rstrip() + "…"
    return t


def run_u_covariate_pathway_enrichment(
    df: pd.DataFrame,
    *,
    fdr_cut: float = 0.05,
    gene_sets: list[str] | None = None,
) -> pd.DataFrame:
    """Enrichr ORA on FDR-significant U_rel co-/anti-varying genes (Mouse)."""
    from deg_enrichment_workflow import run_pathway_enrichment

    if gene_sets is None:
        gene_sets = ["GO_Biological_Process_2023", "MSigDB_Hallmark_2020"]

    sig = df.loc[df["fdr_bh"] < fdr_cut].copy()
    up_genes = sig.loc[sig["spearman_rho"] > 0, "gene"].astype(str).tolist()
    down_genes = sig.loc[sig["spearman_rho"] < 0, "gene"].astype(str).tolist()
    print(
        f"  pathway ORA: FDR<{fdr_cut} → n_up={len(up_genes)} n_down={len(down_genes)}",
        flush=True,
    )

    frames = []
    for direction, genes in (("up", up_genes), ("down", down_genes)):
        enr, warn = run_pathway_enrichment(
            genes,
            comparison="alv_U_covariate",
            direction=direction,
            organism="Mouse",
            gene_sets=gene_sets,
        )
        if warn:
            print(f"  enrichment [{direction}]: {warn}", flush=True)
        if enr is None or enr.empty:
            continue
        enr = enr.copy()
        enr["n_query_genes"] = len(genes)
        enr["fdr_cut"] = fdr_cut
        frames.append(enr)
        top = enr.sort_values("adjusted_p_value").head(5)
        for _, r in top.iterrows():
            print(
                f"    [{direction}] {_short_go_term(r['term'], max_len=60)} "
                f"padj={r['adjusted_p_value']:.2e}",
                flush=True,
            )

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_path = TABLES / "GSE141259_alv_U_covariate_pathway_enrichment.csv"
    out.to_csv(out_path, index=False)
    (PROTO / "tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(PROTO / "tables" / out_path.name, index=False)
    print(f"  Wrote {out_path} (n={len(out)})", flush=True)
    return out


def plot_u_covariate_figure(
    df: pd.DataFrame,
    *,
    enrichment: pd.DataFrame | None = None,
    top_n: int = 20,
    pathway_top_n: int = 10,
    axes: tuple | list | None = None,
    letters: tuple[str, ...] = ("A", "B", "C", "D"),
    letter_xy: tuple[float, float] = (-0.22, 1.05),
    save: bool = True,
    out: Path | None = None,
    show_chrome: bool = True,
    gene_layout: str = "split",
    pathway_style: str = "bubbles",
    co_color: str = "#E07A5F",
    anti_color: str = "#3A6EA5",
    flat_pair_colors: bool = False,
    pathway_order: tuple[str, str] = ("up", "down"),
    gene_name_size: float = 6.4,
):
    """Premium journal figure: gene lollipops + enrichment panels.

    ``gene_layout``:
      - ``split``: two gene panels (co / anti) + two pathway panels → 4 axes / letters
      - ``bidirectional``: one left–right gene panel + two pathway panels → 3 axes / letters

    ``pathway_style``: ``bubbles`` (overlap-sized dots) or ``bars`` (horizontal bars).
    ``flat_pair_colors``: if True, use ``co_color`` / ``anti_color`` as solid fills
    (no magnitude/significance remapping).

    If ``axes`` is provided, draw into those axes (caller owns the figure).
    """
    from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    if gene_layout not in {"split", "bidirectional"}:
        raise ValueError("gene_layout must be 'split' or 'bidirectional'")
    if pathway_style not in {"bubbles", "bars"}:
        raise ValueError("pathway_style must be 'bubbles' or 'bars'")
    if tuple(pathway_order) not in {("up", "down"), ("down", "up")}:
        raise ValueError("pathway_order must be ('up','down') or ('down','up')")

    up = df.nlargest(top_n, "spearman_rho")  # strongest positive first
    down = df.nsmallest(top_n, "spearman_rho")  # strongest negative first
    # split panels: strongest at top after invert_yaxis via ascending y + invert
    up_split = up.iloc[::-1]
    down_split = down.iloc[::-1]

    if enrichment is None:
        enr_path = TABLES / "GSE141259_alv_U_covariate_pathway_enrichment.csv"
        enrichment = pd.read_csv(enr_path) if enr_path.exists() else pd.DataFrame()

    def _soft_cmap(hex_color: str, name: str) -> LinearSegmentedColormap:
        rgb = np.asarray(to_rgb(hex_color), float)
        light = 0.94 * np.ones(3) + 0.06 * rgb
        dark = np.clip(rgb * 0.72, 0, 1)
        return LinearSegmentedColormap.from_list(name, [light, rgb, dark])

    CMAP_UP = _soft_cmap(co_color, "urel_up")
    CMAP_DN = _soft_cmap(anti_color, "urel_dn")
    CO_C, ANTI_C = co_color, anti_color
    SPINE, TICK, LABEL = "#D0D5DD", "#667085", "#101828"

    own_fig = axes is None
    if own_fig:
        if gene_layout == "bidirectional":
            fig = plt.figure(figsize=(12.8, 10.0), facecolor="white")
            outer = fig.add_gridspec(
                2,
                1,
                height_ratios=[1.25, 1.0],
                hspace=0.30,
                left=0.06,
                right=0.98,
                top=0.90,
                bottom=0.07,
            )
            ax_gene = fig.add_subplot(outer[0, 0])
            gs_bot = outer[1].subgridspec(1, 2, wspace=0.42)
            ax_c = fig.add_subplot(gs_bot[0, 0])
            ax_d = fig.add_subplot(gs_bot[0, 1])
            ax_a = ax_b = None
        else:
            fig = plt.figure(figsize=(12.8, 10.4), facecolor="white")
            outer = fig.add_gridspec(
                2,
                1,
                height_ratios=[1.15, 1.0],
                hspace=0.28,
                left=0.01,
                right=0.99,
                top=0.90,
                bottom=0.07,
            )
            gs_top = outer[0].subgridspec(1, 2, wspace=0.42)
            gs_bot = outer[1].subgridspec(1, 2, wspace=0.42, width_ratios=[1.0, 1.0])
            ax_a = fig.add_subplot(gs_top[0, 0])
            ax_b = fig.add_subplot(gs_top[0, 1])
            ax_c = fig.add_subplot(gs_bot[0, 0])
            ax_d = fig.add_subplot(gs_bot[0, 1])
            ax_gene = None
    else:
        if gene_layout == "bidirectional":
            if len(axes) != 3:
                raise ValueError("bidirectional layout needs 3 axes (genes, path↑, path↓)")
            if len(letters) < 3:
                raise ValueError("bidirectional layout needs ≥3 panel letters")
            ax_gene, ax_c, ax_d = axes
            ax_a = ax_b = None
            fig = ax_gene.figure
        else:
            if len(axes) != 4:
                raise ValueError("split layout needs 4 axes (gene↑, gene↓, path↑, path↓)")
            if len(letters) < 4:
                raise ValueError("split layout needs ≥4 panel letters")
            ax_a, ax_b, ax_c, ax_d = axes
            ax_gene = None
            fig = ax_a.figure

    def _base(ax):
        ax.set_facecolor("#FCFCFD")
        ax.tick_params(labelsize=6.8, length=1.8, width=0.5, colors=TICK, pad=1.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(SPINE)
            ax.spines[sp].set_linewidth(0.65)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, ls=":", lw=0.5, color="#E4E7EC", zorder=0)
        ax.yaxis.grid(False)

    def _lollipop_genes(ax, sub, *, cmap, letter, title, positive: bool):
        genes = sub["gene"].astype(str).tolist()
        rhos = sub["spearman_rho"].to_numpy(float)
        y = np.arange(len(genes))
        strength = np.abs(rhos)
        norm = Normalize(vmin=strength.min() * 0.85, vmax=strength.max())
        colors = cmap(norm(strength))

        for yi, rho, c in zip(y, rhos, colors):
            ax.plot([0, rho], [yi, yi], color=c, lw=1.35, solid_capstyle="round", zorder=2)
            ax.scatter(
                [rho],
                [yi],
                s=38,
                c=[c],
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(genes, fontsize=7.2, fontstyle="italic", color=LABEL)
        ax.axvline(0, color="#98A2B3", lw=0.7, zorder=1)
        set_panel_title(ax, title, pad=7)
        ax.set_xlabel(
            r"Spearman $\rho\!\left(U_{\mathrm{rel}},\ \mathrm{expression}\right)$",
            fontsize=8.2,
            color="#344054",
            labelpad=3,
        )
        _base(ax)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        span = max(float(np.ptp(rhos)), 1e-3)
        if positive:
            ax.set_xlim(-0.01 * span, float(rhos.max()) + 0.28 * span)
            for yi, rho in zip(y, rhos):
                ax.text(
                    rho + 0.035 * span,
                    yi,
                    f"{rho:.3f}",
                    va="center",
                    ha="left",
                    fontsize=5.9,
                    color="#475467",
                    clip_on=False,
                )
        else:
            ax.set_xlim(float(rhos.min()) - 0.04 * span, 0.26 * span)
            for yi, rho in zip(y, rhos):
                ax.text(
                    0.03 * span,
                    yi,
                    f"{rho:.3f}",
                    va="center",
                    ha="left",
                    fontsize=5.9,
                    color="#475467",
                    clip_on=False,
                )
        _panel_letter(ax, letter, x=letter_xy[0], y=letter_xy[1])

    def _bidirectional_genes(ax, *, letter, title):
        """Left = anti-varying (ρ<0), right = co-varying (ρ>0); strongest at top."""
        n = top_n
        y = np.arange(n)
        rho_dn = down["spearman_rho"].to_numpy(float)
        rho_up = up["spearman_rho"].to_numpy(float)
        genes_dn = down["gene"].astype(str).tolist()
        genes_up = up["gene"].astype(str).tolist()

        all_abs = np.concatenate([np.abs(rho_dn), np.abs(rho_up)])
        if flat_pair_colors:
            cols_dn = [ANTI_C] * len(rho_dn)
            cols_up = [CO_C] * len(rho_up)
        else:
            norm = Normalize(vmin=float(all_abs.min()) * 0.85, vmax=float(all_abs.max()))
            cols_dn = CMAP_DN(norm(np.abs(rho_dn)))
            cols_up = CMAP_UP(norm(np.abs(rho_up)))

        # Stems first; center guide after ylim so it stays in the gene band.
        for yi, rho, c in zip(y, rho_dn, cols_dn):
            ax.plot([0, rho], [yi, yi], color=c, lw=1.55, solid_capstyle="round", zorder=2)
            ax.scatter([rho], [yi], s=40, c=[c], edgecolors="white", linewidths=0.7, zorder=3)
        for yi, rho, c in zip(y, rho_up, cols_up):
            ax.plot([0, rho], [yi, yi], color=c, lw=1.55, solid_capstyle="round", zorder=2)
            ax.scatter([rho], [yi], s=40, c=[c], edgecolors="white", linewidths=0.7, zorder=3)

        xmax = float(max(abs(rho_dn.min()), rho_up.max(), 1e-3))
        # Wide side gutters so gene names sit clear of the center guide.
        pad_l = pad_r = 1.05 * xmax
        ax.set_xlim(-xmax - pad_l, xmax + pad_r)
        # Extra headroom under the title for Anti-/Co-varying labels.
        if not own_fig:
            ax.set_ylim(-1.35, n - 0.35)
        else:
            ax.set_ylim(-0.55, n - 0.4)
        ax.invert_yaxis()
        ax.plot([0, 0], [-0.35, n - 0.55], color="#667085", lw=0.9, zorder=1, solid_capstyle="butt")
        ax.set_yticks([])
        _base(ax)
        # No left/right frame spines in embedded layout — names use that space;
        # the center ρ=0 guide is the reference axis.
        if not own_fig:
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel(
            r"Spearman $\rho\!\left(U_{\mathrm{rel}},\ \mathrm{expression}\right)$",
            fontsize=7.6,
            color="#344054",
            labelpad=3,
        )
        set_panel_title(ax, title, pad=6 if not own_fig else 7)

        if not own_fig:
            # Inside headroom, clearly below the panel title.
            ax.text(
                0.02,
                -0.72,
                "Anti-varying",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=7.2,
                color=ANTI_C,
                fontweight="bold",
                clip_on=False,
            )
            ax.text(
                0.98,
                -0.72,
                "Co-varying",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=7.2,
                color=CO_C,
                fontweight="bold",
                clip_on=False,
            )
        else:
            ax.text(
                0.02,
                1.015,
                "Anti-varying",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.4,
                color=ANTI_C,
                fontweight="bold",
                clip_on=False,
            )
            ax.text(
                0.98,
                1.015,
                "Co-varying",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.4,
                color=CO_C,
                fontweight="bold",
                clip_on=False,
            )

        # Place names in the side gutters (away from stems and center guide).
        for yi, gene, rho in zip(y, genes_dn, rho_dn):
            ax.text(
                -xmax - 0.06 * xmax,
                yi,
                gene,
                va="center",
                ha="right",
                fontsize=gene_name_size,
                fontstyle="italic",
                color=LABEL,
                clip_on=True,
            )
        for yi, gene, rho in zip(y, genes_up, rho_up):
            ax.text(
                xmax + 0.06 * xmax,
                yi,
                gene,
                va="center",
                ha="left",
                fontsize=gene_name_size,
                fontstyle="italic",
                color=LABEL,
                clip_on=True,
            )
        _panel_letter(ax, letter, x=letter_xy[0], y=letter_xy[1])

    if gene_layout == "bidirectional":
        _bidirectional_genes(
            ax_gene,
            letter=letters[0],
            title=r"Genes associated with $U_{\mathrm{rel}}$",
        )
        path_letters = (letters[1], letters[2])
    else:
        _lollipop_genes(
            ax_a,
            up_split,
            cmap=CMAP_UP,
            letter=letters[0],
            title=r"Genes co-varying with $U_{\mathrm{rel}}$",
            positive=True,
        )
        _lollipop_genes(
            ax_b,
            down_split,
            cmap=CMAP_DN,
            letter=letters[1],
            title=r"Genes anti-varying with $U_{\mathrm{rel}}$",
            positive=False,
        )
        path_letters = (letters[2], letters[3])

    def _select_terms(direction: str) -> pd.DataFrame:
        if enrichment is None or enrichment.empty:
            return pd.DataFrame()
        sub = enrichment.loc[enrichment["direction"] == direction].copy()
        if sub.empty:
            return sub
        hall = sub[
            sub["gene_set"].astype(str).str.contains("Hallmark", case=False, na=False)
            & (sub["adjusted_p_value"] < 0.05)
        ]
        go = sub[
            sub["gene_set"].astype(str).str.contains("GO_Biological", case=False, na=False)
            & (sub["adjusted_p_value"] < 0.05)
        ]
        use = hall if len(hall) >= 3 else (go if not go.empty else sub)
        use = use.loc[use["adjusted_p_value"] < 0.05] if (use["adjusted_p_value"] < 0.05).any() else use
        return (
            use.sort_values("adjusted_p_value")
            .drop_duplicates("term")
            .head(pathway_top_n)
            .iloc[::-1]
        )

    def _overlap_n(val) -> int:
        s = str(val)
        if "/" in s:
            try:
                return int(s.split("/")[0])
            except ValueError:
                return 1
        return 1

    def _pathway_bubbles(ax, direction: str, *, cmap, letter, title):
        use = _select_terms(direction)
        if use.empty:
            ax.text(0.5, 0.5, "No significant pathways", ha="center", va="center", color=MUTED)
            ax.set_axis_off()
            set_panel_title(ax, title, pad=7)
            _panel_letter(ax, letter, x=letter_xy[0], y=letter_xy[1])
            return

        y = np.arange(len(use))
        padj = use["adjusted_p_value"].to_numpy(float)
        neglog = -np.log10(np.clip(padj, 1e-300, None))
        counts = np.array([_overlap_n(v) for v in use.get("overlap", [1] * len(use))], float)
        # Size: encode gene overlap
        sizes = 55 + 14.0 * counts
        norm = Normalize(vmin=0, vmax=max(float(neglog.max()), 2.0))
        colors = cmap(0.35 + 0.60 * norm(neglog))

        thr = -np.log10(0.05)
        ax.axvline(thr, color="#98A2B3", lw=0.75, ls=(0, (3, 2.5)), zorder=1)
        for yi, nl, s, c in zip(y, neglog, sizes, colors):
            ax.plot([0, nl], [yi, yi], color=c, lw=1.15, alpha=0.85, zorder=2, solid_capstyle="round")
            ax.scatter(
                [nl],
                [yi],
                s=s,
                c=[c],
                edgecolors="white",
                linewidths=0.85,
                zorder=3,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(
            [_short_go_term(t, max_len=42) for t in use["term"]],
            fontsize=6.9,
            color=LABEL,
        )
        set_panel_title(ax, title, pad=7)
        ax.set_xlabel(r"$-\log_{10}(\mathrm{adjusted}\ P)$", fontsize=8.2, color="#344054", labelpad=3)
        _base(ax)
        xmax = float(neglog.max())
        ax.set_xlim(0, xmax * 1.22)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        # Compact size legend (once on left pathway panel)
        if letter == path_letters[0]:
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#98A2B3",
                    markeredgecolor="white",
                    markersize=ms,
                    label=lab,
                )
                for ms, lab in ((5.5, "5 genes"), (8.0, "15"), (10.5, "30+"))
            ]
            ax.legend(
                handles=handles,
                title="Overlap",
                loc="lower right",
                fontsize=5.8,
                title_fontsize=6.0,
                frameon=True,
                fancybox=False,
                edgecolor="#E4E7EC",
                framealpha=0.95,
                borderpad=0.35,
                handletextpad=0.35,
                labelspacing=0.25,
            )
        _panel_letter(ax, letter, x=letter_xy[0], y=letter_xy[1])

    def _pathway_bars(ax, direction: str, *, cmap, letter, title):
        use = _select_terms(direction)
        if use.empty:
            ax.text(0.5, 0.5, "No significant pathways", ha="center", va="center", color=MUTED)
            ax.set_axis_off()
            set_panel_title(ax, title, pad=7)
            _panel_letter(ax, letter, x=letter_xy[0], y=letter_xy[1])
            return

        y = np.arange(len(use))
        padj = use["adjusted_p_value"].to_numpy(float)
        neglog = -np.log10(np.clip(padj, 1e-300, None))
        counts = np.array([_overlap_n(v) for v in use.get("overlap", [1] * len(use))], float)
        if flat_pair_colors:
            # Exact solid pair colors: up→co, down→anti
            bar_color = CO_C if direction == "up" else ANTI_C
            colors = [bar_color] * len(use)
        else:
            norm = Normalize(vmin=0, vmax=max(float(neglog.max()), 2.0))
            colors = cmap(0.40 + 0.55 * norm(neglog))

        thr = -np.log10(0.05)
        ax.axvline(thr, color="#98A2B3", lw=0.75, ls=(0, (3, 2.5)), zorder=1)
        ax.barh(y, neglog, color=colors, edgecolor="none", height=0.72, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [_short_go_term(t, max_len=22 if not own_fig else 36) for t in use["term"]],
            fontsize=5.9 if not own_fig else 6.5,
            color=LABEL,
        )
        set_panel_title(ax, title, pad=7)
        ax.set_xlabel(r"$-\log_{10}(\mathrm{adjusted}\ P)$", fontsize=8.0, color="#344054", labelpad=3)
        _base(ax)
        xmax = float(neglog.max())
        ax.set_xlim(0, xmax * 1.22)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        # Overlap counts at bar tips (standalone figure only)
        if own_fig:
            for yi, nl, n_ov in zip(y, neglog, counts):
                ax.text(
                    nl + 0.015 * max(xmax, 1e-6),
                    yi,
                    f"n={int(n_ov)}",
                    va="center",
                    ha="left",
                    fontsize=5.4,
                    color="#667085",
                    clip_on=True,
                )
        _panel_letter(ax, letter, x=letter_xy[0], y=letter_xy[1])

    n_up = int(((df["fdr_bh"] < 0.05) & (df["spearman_rho"] > 0)).sum()) if "fdr_bh" in df else 0
    n_down = int(((df["fdr_bh"] < 0.05) & (df["spearman_rho"] < 0)).sum()) if "fdr_bh" in df else 0
    if enrichment is not None and not enrichment.empty and "n_query_genes" in enrichment.columns:
        nu = enrichment.loc[enrichment["direction"] == "up", "n_query_genes"]
        nd = enrichment.loc[enrichment["direction"] == "down", "n_query_genes"]
        if len(nu):
            n_up = int(nu.iloc[0])
        if len(nd):
            n_down = int(nd.iloc[0])

    _draw_pathways = _pathway_bars if pathway_style == "bars" else _pathway_bubbles
    for ax_p, direction, letter in zip((ax_c, ax_d), pathway_order, path_letters):
        is_up = direction == "up"
        _draw_pathways(
            ax_p,
            direction,
            cmap=CMAP_UP if is_up else CMAP_DN,
            letter=letter,
            title=(
                ("Co-varying" if is_up else "Anti-varying")
                if not own_fig
                else (
                    rf"Hallmark enrichment · co-varying ($n={n_up}$)"
                    if is_up
                    else rf"Hallmark enrichment · anti-varying ($n={n_down}$)"
                )
            ),
        )

    if show_chrome and own_fig:
        if gene_layout == "bidirectional":
            fig.suptitle(
                r"GSE141259 · Alveolar $U_{\mathrm{rel}}$ covariates and pathway programs",
                fontsize=12,
                fontweight="bold",
                color=INK,
                y=0.965,
            )
            fig.text(
                0.5,
                0.925,
                "Top: bidirectional gene–potential association.  "
                "Bottom: Enrichr Hallmark ORA on FDR$<0.05$ sets · point size ∝ overlap · dashed line FDR$=0.05$.",
                ha="center",
                va="center",
                fontsize=7.0,
                color="#667085",
            )
        else:
            fig.suptitle(
                r"GSE141259 · Alveolar $U_{\mathrm{rel}}$ covariates and pathway programs",
                fontsize=12,
                fontweight="bold",
                color=INK,
                y=0.965,
            )
            fig.text(
                0.5,
                0.925,
                "Top: gene–potential association (lollipop).  "
                "Bottom: Enrichr Hallmark ORA on FDR$<0.05$ sets · point size ∝ overlap · dashed line FDR$=0.05$.",
                ha="center",
                va="center",
                fontsize=7.0,
                color="#667085",
            )
        fig.text(
            0.01,
            0.012,
            "Spearman ρ among training HVGs in alveolar epithelium; pathway ORA via Enrichr (Mouse).",
            fontsize=5.9,
            color="#98A2B3",
            style="italic",
            ha="left",
        )
    if save and own_fig:
        if out is None:
            out = PANELS / "GSE141259_alv_U_covariate_genes.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=320, bbox_inches="tight", facecolor="white", pad_inches=0.16)
        fig.savefig(PROTO / "figures" / out.name, dpi=320, bbox_inches="tight", facecolor="white", pad_inches=0.16)
        plt.close(fig)
        print(f"Wrote {out}", flush=True)
    return fig


def plot_only_from_tables():
    """Re-render panels from saved CSVs (fast; no LAP recomputation)."""
    barriers = pd.read_csv(TABLES / "GSE141259_barrier_action_matrix.csv")
    for panel, name in (
        ("Alveolar", "GSE141259_alv_barrier_action_matrix.png"),
        ("Macrophages", "GSE141259_mac_barrier_action_matrix.png"),
        ("Club", "GSE141259_club_barrier_action_matrix.png"),
    ):
        plot_barrier_heatmaps(barriers, panel, PANELS / name)

    cell_df = pd.read_csv(TABLES / "GSE141259_krt8_ADI_energy_tilt_cells.csv")
    summary = pd.read_csv(TABLES / "GSE141259_krt8_bifurcation_bias_summary.csv")
    plot_bifurcation_figure(cell_df, summary)
    plot_alv_barrier_bifurcation_combined(barriers, cell_df, summary)

    cov = pd.read_csv(TABLES / "GSE141259_alv_U_covariate_genes.csv")
    enr_path = TABLES / "GSE141259_alv_U_covariate_pathway_enrichment.csv"
    if enr_path.exists():
        enr = pd.read_csv(enr_path)
    else:
        enr = run_u_covariate_pathway_enrichment(cov)
    plot_u_covariate_figure(cov, enrichment=enr, top_n=20)
    plot_alv_dynamics_narrative_combined(
        barriers, cell_df, summary, cov, enrichment=enr, gene_top_n=10, pathway_top_n=8
    )

    club_sum = TABLES / "GSE141259_club_fate_bias_summary.csv"
    club_cells = TABLES / "GSE141259_club_fate_tilt_cells.csv"
    if club_sum.exists() and club_cells.exists():
        plot_club_fate_bias_figure(pd.read_csv(club_cells), pd.read_csv(club_sum))
    else:
        print("[plot-only] Club fate tables missing — skip (run full pipeline once).", flush=True)
    print("\n[Done] Replotted panels from tables.", flush=True)


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help="Only re-render figures from output_file/mac_landscape_audit CSVs",
    )
    args = ap.parse_args()
    if args.plot_only:
        plot_only_from_tables()
        return

    # --- barriers ---
    barrier_frames = []
    adata_a = L._load_parent("alv_epithelium", ALV_TYPES)
    df_a, _, _ = build_barrier_panel("Alveolar", adata_a, ALV_EDGES, ALV_TYPES)
    barrier_frames.append(df_a)
    plot_barrier_heatmaps(df_a, "Alveolar", PANELS / "GSE141259_alv_barrier_action_matrix.png")

    adata_m = L._load_parent("macrophages", MAC_TYPES)
    df_m, _, _ = build_barrier_panel("Macrophages", adata_m, MAC_EDGES, MAC_TYPES)
    barrier_frames.append(df_m)
    plot_barrier_heatmaps(df_m, "Macrophages", PANELS / "GSE141259_mac_barrier_action_matrix.png")

    adata_c = L.load_club_lineage_adata()
    df_c, _, _ = build_barrier_panel("Club", adata_c, CLUB_EDGES, list(L.CLUB_LINEAGE_TYPES))
    barrier_frames.append(df_c)
    plot_barrier_heatmaps(df_c, "Club", PANELS / "GSE141259_club_barrier_action_matrix.png")

    barriers = pd.concat(barrier_frames, ignore_index=True)
    barriers.to_csv(TABLES / "GSE141259_barrier_action_matrix.csv", index=False)
    barriers.to_csv(PROTO / "tables" / "GSE141259_barrier_action_matrix.csv", index=False)

    # --- bifurcation ---
    cell_df, summary, _p_at1, _p_fib = run_bifurcation_bias()
    plot_alv_barrier_bifurcation_combined(barriers, cell_df, summary)

    # --- Club multi-fate bias ---
    run_club_fate_bias()

    # --- U covariates ---
    run_u_covariate()
    cov = pd.read_csv(TABLES / "GSE141259_alv_U_covariate_genes.csv")
    enr_path = TABLES / "GSE141259_alv_U_covariate_pathway_enrichment.csv"
    enr = pd.read_csv(enr_path) if enr_path.exists() else None
    plot_alv_dynamics_narrative_combined(
        barriers, cell_df, summary, cov, enrichment=enr, gene_top_n=10, pathway_top_n=8
    )

    print("\n[Done] Three-axis dynamical mining complete.", flush=True)
    print(f"  Tables: {TABLES}", flush=True)
    print(f"  Figures: {PANELS}", flush=True)


if __name__ == "__main__":
    main()
