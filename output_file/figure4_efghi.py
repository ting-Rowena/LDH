#!/usr/bin/env python3
"""Figure 4: GSE141259 alveolar dynamics narrative combined figure.

Self-contained script consolidated from:
  - scripts/run_gse141259_three_dynamics_axes.py
    (plot_alv_dynamics_narrative_combined + plot_u_covariate_figure subset)
  - panel_style.py / plot_utils.py (minimal helpers inlined)

Default output:
  output_file/figure4_efghi.png
  (= archived panel GSE141259_alv_dynamics_narrative_combined.png; manuscript Figure 4)

Usage:
  python output_file/figure4_efghi.py
  python output_file/figure4_efghi.py /path/to/out.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.ticker import MaxNLocator
from scipy import stats

if os.environ.get("MPLBACKEND") is None:
    mpl.use("Agg", force=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts"))
from _adopted import cache_file  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent / "figure4_efghi.png"

# ---------------------------------------------------------------------------
# Style (inlined)
# ---------------------------------------------------------------------------
PANEL_TITLE_SIZE = 10
PANEL_TITLE_WEIGHT = "bold"
PANEL_TITLE_LOC = "center"
PANEL_TITLE_PAD = 4
INK = "#1f2933"
MUTED = "#7b8794"

SHORT = {
    "AT2 cells": "AT2",
    "Activated AT2 cells": "Activated AT2",
    "Krt8 ADI": "Krt8+ ADI",
    "AT1 cells": "AT1",
    "AM (PBS)": "AM(PBS)",
    "AM (Bleo)": "AM(Bleo)",
    "M2 macrophages": "Arg1+ M2",
    "Resolution macrophages": "Mfge8+ Resol.",
    "Fn1+ macrophages": "Fn1+",
    "Cd163-/Cd11c+ IMs": "IM−",
    "Cd163+/Cd11c- IMs": "IM+",
    "Club cells": "Club",
    "MHC-II+ Club cells": "MHC-II+ Club",
    "Ciliated cells": "Ciliated",
    "Goblet cells": "Goblet",
}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "axes.titlesize": PANEL_TITLE_SIZE,
            "axes.titleweight": PANEL_TITLE_WEIGHT,
            "axes.titlelocation": PANEL_TITLE_LOC,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _set_panel_title(ax, title: str, *, pad: float | None = None, **kwargs) -> None:
    kw = {
        "loc": PANEL_TITLE_LOC,
        "fontweight": PANEL_TITLE_WEIGHT,
        "fontsize": PANEL_TITLE_SIZE,
        "pad": PANEL_TITLE_PAD if pad is None else pad,
    }
    kw.update(kwargs)
    ax.set_title(title, **kw)


def _short(t: str) -> str:
    return SHORT.get(t, t)


def _style_journal_ax(ax) -> None:
    ax.tick_params(labelsize=7.5, length=2.2, width=0.55, colors="#475467")
    for sp in ax.spines.values():
        sp.set_color("#CBD5E1")
        sp.set_linewidth(0.7)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.55, color="0.88", zorder=0)
    ax.set_axisbelow(True)


def _black_axes(ax) -> None:
    ax.tick_params(colors="black")
    for sp in ax.spines.values():
        if sp.get_visible():
            sp.set_color("black")


def _spines_above_bars(ax) -> None:
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


def _panel_letter(ax, letter: str, *, x: float = -0.12, y: float = 1.08) -> None:
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


def _short_go_term(term: str, *, max_len: int = 52) -> str:
    t = str(term)
    if "(GO:" in t:
        t = t[: t.rfind("(GO:")].strip()
    t = t.rstrip(" .")
    if len(t) > max_len:
        t = t[: max_len - 1].rstrip() + "…"
    return t


def _plot_lap_metric_lollipop(
    ax,
    vals: np.ndarray,
    title: str,
    *,
    signed: bool,
) -> None:
    """Single-metric lollipop column with its own x-axis."""
    pos_c = "#5B8FA8"
    neg_c = "#C0784A"
    zero_c = "#CBD5E1"
    span = max(float(np.max(np.abs(vals))), 1e-3)
    pad_scale = span * 1.12
    ax.axvline(0, color=zero_c, lw=0.75, zorder=1)
    for i, v in enumerate(vals):
        if abs(v) < 1e-6:
            color = "#94A3B8"
        else:
            color = pos_c if v > 0 else neg_c
        ax.plot([0, v], [i, i], color=color, lw=1.9, solid_capstyle="round", zorder=2)
        ax.scatter([v], [i], s=32, c=color, edgecolors="white", linewidths=0.9, zorder=3)
        ha = "left" if v >= 0 else "right"
        pad = 0.02 * pad_scale if v >= 0 else -0.02 * pad_scale
        ax.text(
            v + pad,
            i,
            f"{v:.1f}" if not signed else f"{v:.2f}",
            va="center",
            ha=ha,
            fontsize=5.4,
            color="#475467",
            zorder=4,
        )
    if signed:
        ax.set_xlim(-pad_scale, pad_scale)
    else:
        ax.set_xlim(0.0, pad_scale)

    ax.set_yticks([])
    _set_panel_title(ax, title, pad=3)
    ax.set_facecolor("#FCFCFD")
    ax.tick_params(labelsize=6.8, length=2.0, width=0.55, colors="#475467")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.spines["bottom"].set_linewidth(0.7)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))


def plot_u_covariate_figure(
    df: pd.DataFrame,
    *,
    enrichment: pd.DataFrame | None = None,
    top_n: int = 20,
    pathway_top_n: int = 10,
    axes: tuple | list | None = None,
    letters: tuple[str, ...] = ("", "", ""),
    letter_xy: tuple[float, float] = (-0.10, 1.10),
    gene_layout: str = "bidirectional",
    pathway_style: str = "bars",
    co_color: str = "#F9DBC4",
    anti_color: str = "#B8D1CB",
    flat_pair_colors: bool = True,
    pathway_order: tuple[str, str] = ("down", "up"),
    gene_name_size: float = 8.2,
) -> None:
    """Gene bidirectional panel + Hallmark pathway bars (narrative-figure subset)."""
    if gene_layout != "bidirectional" or pathway_style != "bars":
        raise ValueError("figure4_efghi only supports bidirectional genes + pathway bars")
    if axes is None or len(axes) != 3:
        raise ValueError("need 3 axes (genes, path↑, path↓)")

    up = df.nlargest(top_n, "spearman_rho")
    down = df.nsmallest(top_n, "spearman_rho")

    if enrichment is None:
        enrichment = pd.DataFrame()

    def _soft_cmap(hex_color: str, name: str) -> LinearSegmentedColormap:
        rgb = np.asarray(to_rgb(hex_color), float)
        light = 0.94 * np.ones(3) + 0.06 * rgb
        dark = np.clip(rgb * 0.72, 0, 1)
        return LinearSegmentedColormap.from_list(name, [light, rgb, dark])

    CMAP_UP = _soft_cmap(co_color, "urel_up")
    CMAP_DN = _soft_cmap(anti_color, "urel_dn")
    CO_C, ANTI_C = co_color, anti_color
    SPINE, TICK, LABEL = "#D0D5DD", "#667085", "#101828"

    ax_gene, ax_c, ax_d = axes

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

    # ---- bidirectional genes ----
    n = top_n
    y = np.arange(n)
    rho_dn = down["spearman_rho"].to_numpy(float)
    rho_up = up["spearman_rho"].to_numpy(float)
    genes_dn = down["gene"].astype(str).tolist()
    genes_up = up["gene"].astype(str).tolist()

    if flat_pair_colors:
        cols_dn = [ANTI_C] * len(rho_dn)
        cols_up = [CO_C] * len(rho_up)
    else:
        all_abs = np.concatenate([np.abs(rho_dn), np.abs(rho_up)])
        norm = Normalize(vmin=float(all_abs.min()) * 0.85, vmax=float(all_abs.max()))
        cols_dn = CMAP_DN(norm(np.abs(rho_dn)))
        cols_up = CMAP_UP(norm(np.abs(rho_up)))

    for yi, rho, c in zip(y, rho_dn, cols_dn):
        ax_gene.plot([0, rho], [yi, yi], color=c, lw=1.55, solid_capstyle="round", zorder=2)
        ax_gene.scatter([rho], [yi], s=40, c=[c], edgecolors="white", linewidths=0.7, zorder=3)
    for yi, rho, c in zip(y, rho_up, cols_up):
        ax_gene.plot([0, rho], [yi, yi], color=c, lw=1.55, solid_capstyle="round", zorder=2)
        ax_gene.scatter([rho], [yi], s=40, c=[c], edgecolors="white", linewidths=0.7, zorder=3)

    xmax = float(max(abs(rho_dn.min()), rho_up.max(), 1e-3))
    pad_l = pad_r = 1.05 * xmax
    ax_gene.set_xlim(-xmax - pad_l, xmax + pad_r)
    ax_gene.set_ylim(-1.35, n - 0.35)
    ax_gene.invert_yaxis()
    ax_gene.plot([0, 0], [-0.35, n - 0.55], color="#667085", lw=0.9, zorder=1, solid_capstyle="butt")
    ax_gene.set_yticks([])
    _base(ax_gene)
    ax_gene.spines["left"].set_visible(False)
    ax_gene.spines["right"].set_visible(False)
    ax_gene.xaxis.set_major_locator(MaxNLocator(5))
    ax_gene.set_xlabel(
        r"Spearman $\rho\!\left(U_{\mathrm{rel}},\ \mathrm{expression}\right)$",
        fontsize=7.6,
        color="#344054",
        labelpad=3,
    )
    _set_panel_title(ax_gene, r"Genes associated with $U_{\mathrm{rel}}$", pad=6)
    ax_gene.text(
        0.02,
        -0.72,
        "Anti-varying",
        transform=ax_gene.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=7.2,
        color=ANTI_C,
        fontweight="bold",
        clip_on=False,
    )
    ax_gene.text(
        0.98,
        -0.72,
        "Co-varying",
        transform=ax_gene.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=7.2,
        color=CO_C,
        fontweight="bold",
        clip_on=False,
    )
    for yi, gene, rho in zip(y, genes_dn, rho_dn):
        ax_gene.text(
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
        ax_gene.text(
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
    _panel_letter(ax_gene, letters[0], x=letter_xy[0], y=letter_xy[1])

    # ---- pathway bars ----
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
            use.sort_values("adjusted_p_value").drop_duplicates("term").head(pathway_top_n).iloc[::-1]
        )

    def _pathway_bars(ax, direction: str, *, cmap, letter, title):
        use = _select_terms(direction)
        if use.empty:
            ax.text(0.5, 0.5, "No significant pathways", ha="center", va="center", color=MUTED)
            ax.set_axis_off()
            _set_panel_title(ax, title, pad=7)
            _panel_letter(ax, letter, x=letter_xy[0], y=letter_xy[1])
            return

        yy = np.arange(len(use))
        padj = use["adjusted_p_value"].to_numpy(float)
        neglog = -np.log10(np.clip(padj, 1e-300, None))
        if flat_pair_colors:
            bar_color = CO_C if direction == "up" else ANTI_C
            colors = [bar_color] * len(use)
        else:
            norm = Normalize(vmin=0, vmax=max(float(neglog.max()), 2.0))
            colors = cmap(0.40 + 0.55 * norm(neglog))

        thr = -np.log10(0.05)
        ax.axvline(thr, color="#98A2B3", lw=0.75, ls=(0, (3, 2.5)), zorder=1)
        ax.barh(yy, neglog, color=colors, edgecolor="none", height=0.72, zorder=2)
        ax.set_yticks(yy)
        ax.set_yticklabels(
            [_short_go_term(t, max_len=22) for t in use["term"]],
            fontsize=5.9,
            color=LABEL,
        )
        _set_panel_title(ax, title, pad=7)
        ax.set_xlabel(r"$-\log_{10}(\mathrm{adjusted}\ P)$", fontsize=8.0, color="#344054", labelpad=3)
        _base(ax)
        xmax_p = float(neglog.max())
        ax.set_xlim(0, xmax_p * 1.22)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        _panel_letter(ax, letter, x=letter_xy[0], y=letter_xy[1])

    path_letters = (letters[1], letters[2])
    for ax_p, direction, letter in zip((ax_c, ax_d), pathway_order, path_letters):
        is_up = direction == "up"
        _pathway_bars(
            ax_p,
            direction,
            cmap=CMAP_UP if is_up else CMAP_DN,
            letter=letter,
            title=("Co-varying" if is_up else "Anti-varying"),
        )


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
) -> Path:
    """Compact alveolar constraint figure; entry support is stronger than exit support."""
    if isinstance(summary, pd.DataFrame):
        summary = summary.iloc[0]
    summary = dict(summary)

    ROW1_BAR = "#A0C7DB"
    AT1_C, FIB_C = ROW1_BAR, ROW1_BAR
    CURVE_C = "#EC7CBB"
    path_labels = ["AT1\n(weak)", "Fibro\n(not a fate)"]
    x_bar = np.array([0.0, 0.42])
    bar_w = 0.32
    bar_xlim = (-0.26, 0.68)

    sub = barriers[(barriers["panel"] == "Alveolar") & (barriers["direction"] == "forward")].copy()
    if sub.empty:
        raise RuntimeError("No Alveolar forward barrier rows for narrative figure")
    # Barrier height is symmetric on reversible geodesics; endpoint net ΔU is directional.
    labels = []
    for _, row in sub.iterrows():
        label = f"{_short(row['src'])}→{_short(row['dst'])}"
        if row["src"] == "Krt8 ADI" and row["dst"] == "AT1 cells":
            label += " (weak)"
        labels.append(label)
    metrics = [
        ("path_action", "Path action", ROW1_BAR),
        ("barrier_height", r"Barrier $\Delta U$", ROW1_BAR),
        ("delta_U_end_start", r"Net $\Delta U$", ROW1_BAR),
    ]
    n = len(sub)

    out_w, out_h, out_dpi = 3050, 1800, 300
    fig = plt.figure(figsize=(out_w / out_dpi, out_h / out_dpi), facecolor="white", dpi=out_dpi)
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
    gs1_row = outer[0].subgridspec(1, 2, width_ratios=[0.11, 1.0], wspace=0.0)
    gs1 = gs1_row[1].subgridspec(1, 2, width_ratios=[1.25, 1.30], wspace=0.18)
    gs_a = gs1[0].subgridspec(1, 3, wspace=0.22)
    gs_b = gs1[1].subgridspec(1, 2, width_ratios=[0.78, 1.20], wspace=0.32)
    gs_b_left = gs_b[0].subgridspec(2, 1, hspace=0.18)

    axes_a = [fig.add_subplot(gs_a[0, i]) for i in range(3)]
    for ax in axes_a[1:]:
        ax.sharey(axes_a[0])
    ax_b0 = fig.add_subplot(gs_b_left[0])
    ax_b1 = fig.add_subplot(gs_b_left[1])
    ax_b2 = fig.add_subplot(gs_b[1])

    gs2 = outer[1].subgridspec(1, 3, width_ratios=[1.55, 1.0, 1.0], wspace=0.58)
    ax_gene = fig.add_subplot(gs2[0])
    ax_path_up = fig.add_subplot(gs2[1])
    ax_path_dn = fig.add_subplot(gs2[2])

    # ---- A: transition metrics (three independent lollipop columns) ----
    y = np.arange(n, dtype=float)
    for ax, (col, title, _) in zip(axes_a, metrics):
        vals = sub[col].to_numpy(float)
        _plot_lap_metric_lollipop(ax, vals, title, signed=(col != "path_action"))

    axes_a[0].invert_yaxis()
    axes_a[0].set_yticks(y)
    axes_a[0].set_yticklabels(labels, fontsize=6.2)
    axes_a[0].tick_params(axis="y", which="major", pad=0.5, length=2.0, labelleft=True)
    for lab in axes_a[0].get_yticklabels():
        lab.set_ha("right")
        lab.set_color("#1F2937")
    for ax in axes_a[1:]:
        ax.tick_params(axis="y", labelleft=False, length=0)
        ax.spines["left"].set_visible(False)

    # ---- B: bifurcation ----
    bars = [float(summary["ADI_to_AT1_barrier"]), float(summary["ADI_to_Fibro_barrier"])]
    ax_b0.bar(x_bar, bars, color=[AT1_C, FIB_C], edgecolor="none", width=bar_w, zorder=2)
    ax_b0.axhline(0, color="#1F2937", lw=0.7, zorder=1)
    ax_b0.set_xticks([])
    ax_b0.set_xlim(*bar_xlim)
    ax_b0.set_ylabel(r"Barrier $\Delta U$", fontsize=7.0)
    _set_panel_title(ax_b0, r"Krt8$^+$ ADI exit controls", pad=3)
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
    _set_panel_title(ax_b2, "Stage-wise foil contrast", pad=3)
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
        gene_layout="bidirectional",
        pathway_style="bars",
        co_color="#F9DBC4",
        anti_color="#B8D1CB",
        flat_pair_colors=True,
        pathway_order=("down", "up"),
        gene_name_size=8.2,
    )

    for ax in axes_a + [ax_b0, ax_b1, ax_b2, ax_gene, ax_path_up, ax_path_dn]:
        _black_axes(ax)
        _spines_above_bars(ax)

    fig.canvas.draw()
    dx = -0.010
    width_scale = 0.86
    for ax in (ax_b0, ax_b1):
        pos = ax.get_position()
        ax.set_position([pos.x0 - dx, pos.y0, pos.width * width_scale, pos.height])

    if out is None:
        out = DEFAULT_OUT
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"savefig.bbox": None}):
        fig.savefig(out, dpi=out_dpi, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def _compute_alv_dynamics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Barrier / bifurcation / U-covariate from the adopted GSE141259 checkpoint."""
    b_path = cache_file("GSE141259_barrier_action_matrix.csv")
    c_path = cache_file("GSE141259_krt8_ADI_energy_tilt_cells.csv")
    s_path = cache_file("GSE141259_krt8_bifurcation_bias_summary.csv")
    g_path = cache_file("GSE141259_alv_U_covariate_genes.csv")
    e_path = cache_file("GSE141259_alv_U_covariate_pathway_enrichment.csv")
    if all(p.is_file() for p in (b_path, c_path, s_path, g_path)):
        enr = pd.read_csv(e_path) if e_path.is_file() else None
        return pd.read_csv(b_path), pd.read_csv(c_path), pd.read_csv(s_path), pd.read_csv(g_path), enr

    from analyze_mac_alv_dynamics_first_paths import ALV_TYPES
    from run_gse141259_three_dynamics_axes import (
        ALV_EDGES,
        build_barrier_panel,
        run_bifurcation_bias,
        run_u_covariate,
        run_u_covariate_pathway_enrichment,
    )
    import plot_mac_alv_3d_potential_landscape as L

    print("[figure4_efghi] computing alveolar barrier / bifurcation / U-covariate...", flush=True)
    adata_a = L._load_parent("alv_epithelium", ALV_TYPES)
    barriers, _, _ = build_barrier_panel("Alveolar", adata_a, ALV_EDGES, ALV_TYPES)
    cell_df, summary, _p_at1, _p_fib = run_bifurcation_bias()
    if isinstance(summary, dict):
        summary = pd.DataFrame([summary])
    cov = run_u_covariate()
    enrichment = run_u_covariate_pathway_enrichment(cov)
    barriers.to_csv(b_path, index=False)
    cell_df.to_csv(c_path, index=False)
    summary.to_csv(s_path, index=False)
    cov.to_csv(g_path, index=False)
    if enrichment is not None and len(enrichment):
        enrichment.to_csv(e_path, index=False)
    return barriers, cell_df, summary, cov, enrichment


def compose(out: Path | None = None) -> Path:
    _apply_style()
    barriers, cell_df, summary, cov, enrichment = _compute_alv_dynamics()
    return plot_alv_dynamics_narrative_combined(
        barriers,
        cell_df,
        summary,
        cov,
        enrichment=enrichment,
        out=out or DEFAULT_OUT,
        gene_top_n=10,
        pathway_top_n=8,
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out = Path(argv[0]) if argv else DEFAULT_OUT
    compose(out=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
