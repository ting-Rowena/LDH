#!/usr/bin/env python3
"""Journal figure: why focus Macrophages & Alveolar epithelium (GSE141259).

Outputs:
  <CK>/analysis_protocol_GSE141259/figures/GSE141259_why_mac_alv_focus.png
  output_file/mac_landscape_audit/GSE141259_why_mac_alv_focus_stats.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, LogLocator, MaxNLocator, NullFormatter

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_pipeline import recommended_checkpoint_dir  # noqa: E402
from panel_style import (  # noqa: E402
    ANNOT_SIZE,
    AXIS_LABEL_SIZE,
    LEGEND_SIZE,
    TICK_LABEL_SIZE,
    apply_panel_title_rc,
    set_panel_title,
)
from plot_utils import INK, MUTED, configure_headless  # noqa: E402

configure_headless()
apply_panel_title_rc()

plt.rcParams.update(
    {
        "axes.labelcolor": INK,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.color": "black",
        "ytick.color": "black",
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

_HALO = [pe.withStroke(linewidth=3.0, foreground="white")]

CK = Path(recommended_checkpoint_dir("GSE141259"))
TAB = _ROOT / "output_file" / "mac_landscape_audit"
OUT = CK / "analysis_protocol_GSE141259" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Match GSE141259_metacelltype_formal_celltype_umap.png names + palette.
_PAL = pd.read_csv(CK / "figures" / "GSE141259_umap_hierarchical_palette.csv")
_FMAP = pd.read_csv(CK / "figures" / "GSE141259_metacelltype_formal_label_mapping.csv")
FORMAL = dict(zip(_FMAP["metacelltype"].astype(str), _FMAP["formal_label"].astype(str)))
MAC_LAB = FORMAL["macrophages"]  # Macrophages
ALV_LAB = FORMAL["alv_epithelium"]  # Alveolar epithelium
MAC = str(_PAL.loc[_PAL.metacelltype == "macrophages", "parent_color"].iloc[0])
ALV = str(_PAL.loc[_PAL.metacelltype == "alv_epithelium", "parent_color"].iloc[0])
_SUB_COL = dict(zip(_PAL["cell.type"].astype(str), _PAL["subtype_color"].astype(str)))
OTHER = "#C8CDD1"
FOCUS = {"macrophages", "alv_epithelium"}


def _load() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    ranked = pd.read_csv(TAB / "GSE141259_15type_mean_U0_ranked.csv")
    obs = pd.read_csv(CK / "obs.csv", usecols=["annotation", "potential_stationary"], low_memory=False)
    u = obs["potential_stationary"].to_numpy(float)
    med = float(np.nanmedian(u))
    deep = obs.loc[u <= med]
    n_total = int(len(obs))
    n_mac = int((obs["annotation"] == "macrophages").sum())
    n_alv = int((obs["annotation"] == "alv_epithelium").sum())
    n_deep = int(len(deep))
    n_deep_mac = int((deep["annotation"] == "macrophages").sum())
    n_deep_alv = int((deep["annotation"] == "alv_epithelium").sum())
    n_deep_other = n_deep - n_deep_mac - n_deep_alv
    stats = {
        "n_total": n_total,
        "n_mac": n_mac,
        "n_alv": n_alv,
        "frac_mac_alv": (n_mac + n_alv) / n_total,
        "U0_median": med,
        "n_deep": n_deep,
        "n_deep_mac": n_deep_mac,
        "n_deep_alv": n_deep_alv,
        "n_deep_other": n_deep_other,
        "n_deep_mac_alv": n_deep_mac + n_deep_alv,
        "frac_deep_in_mac_alv": (n_deep_mac + n_deep_alv) / max(n_deep, 1),
        "mac_rank_U0": int(ranked.loc[ranked.annotation == "macrophages", "rank"].iloc[0]),
        "alv_rank_U0": int(ranked.loc[ranked.annotation == "alv_epithelium", "rank"].iloc[0]),
        "mean_U0_mac": float(ranked.loc[ranked.annotation == "macrophages", "mean_U0"].iloc[0]),
        "mean_U0_alv": float(ranked.loc[ranked.annotation == "alv_epithelium", "mean_U0"].iloc[0]),
        "u0_span": float(ranked["mean_U0"].max() - ranked["mean_U0"].min()),
    }
    subtypes = pd.read_csv(TAB / "GSE141259_mac_alv_subtype_mean_U0_Urel.csv")
    return ranked, stats, subtypes


def _urel_bars(ax, sub: pd.DataFrame) -> None:
    """Horizontal bars of mean U_rel, deepest (most negative) at the top."""
    sub = sub.sort_values("mean_Urel", ascending=True)
    names = [str(ct) for ct in sub["cell.type"]]
    vals = sub["mean_Urel"].to_numpy(float)
    cols = [_SUB_COL[ct] for ct in sub["cell.type"]]
    y = np.arange(len(sub))[::-1]
    ax.axvline(0.0, color="#B0B6BC", lw=0.8, zorder=1)
    ax.barh(y, vals, color=cols, edgecolor="black", linewidth=0.55, height=0.68, zorder=2)
    for yi, v in zip(y, vals):
        dx = 0.025 if v >= 0 else -0.025
        ax.text(
            v + dx,
            yi,
            f"{v:+.2f}",
            ha="left" if v >= 0 else "right",
            va="center",
            fontsize=6.6,
            color=INK,
            path_effects=_HALO,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.0, color=INK)
    ax.tick_params(axis="y", length=0, colors="black")
    ax.tick_params(axis="x", colors="black")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.55, color="#E3E8EE", zorder=0)
    ax.spines["left"].set_visible(False)
    for sp in ("bottom", "top", "right"):
        if ax.spines[sp].get_visible():
            ax.spines[sp].set_color("black")


def paint_landscape(
    ax,
    ranked: pd.DataFrame,
    stats: dict,
    *,
    panel_title: str | None = None,
    x_col: str = "n",
    xlabel: str = "Number of cells",
    xlim: tuple[float, float] | None = None,
    log_x: bool = True,
) -> None:
    """Left panel: mean U0 vs abundance (default) or another x metric (e.g. DEG count)."""
    u_lo = float(ranked["mean_U0"].min())
    u_hi = float(ranked["mean_U0"].max())
    pad = 0.28 * (u_hi - u_lo)
    n_mac = int(stats["n_mac"])
    n_alv = int(stats["n_alv"])
    x_mac = float(ranked.loc[ranked.annotation == "macrophages", x_col].iloc[0])
    x_alv = float(ranked.loc[ranked.annotation == "alv_epithelium", x_col].iloc[0])
    u_mac = float(stats["mean_U0_mac"])
    u_alv = float(stats["mean_U0_alv"])
    med = float(stats["U0_median"])

    other = ranked[~ranked.annotation.isin(FOCUS)]
    ax.axhspan(u_lo - pad, med, color="#EEF3F6", zorder=0)
    ax.scatter(other[x_col], other["mean_U0"], s=38, c=OTHER, edgecolors="white", linewidths=0.45, zorder=2)
    ax.scatter([x_mac], [u_mac], s=118, c=MAC, edgecolors="black", linewidths=0.9, zorder=4)
    ax.scatter([x_alv], [u_alv], s=118, c=ALV, edgecolors="black", linewidths=0.9, zorder=4)
    ax.axhline(med, color="#8B949E", ls="--", lw=0.75, zorder=1)

    ax.text(
        0.03,
        0.965,
        "deep basin",
        transform=ax.transAxes,
        fontsize=7.5,
        color=MUTED,
        style="italic",
        va="top",
        ha="left",
        zorder=5,
    )
    ax.annotate(
        f"{MAC_LAB}  #1",
        xy=(x_mac, u_mac),
        xytext=(36, 8),
        textcoords="offset points",
        fontsize=8.0,
        fontweight="bold",
        color="black",
        ha="center",
        va="bottom",
        path_effects=_HALO,
        zorder=5,
        annotation_clip=False,
    )
    ax.annotate(
        f"{ALV_LAB}  #2",
        xy=(x_alv, u_alv),
        xytext=(-42, 8),
        textcoords="offset points",
        fontsize=8.0,
        fontweight="bold",
        color="black",
        ha="center",
        va="bottom",
        path_effects=_HALO,
        zorder=5,
        annotation_clip=False,
    )

    x_all = ranked[x_col].to_numpy(float)
    x_all = x_all[np.isfinite(x_all) & (x_all > 0)]
    if xlim is not None:
        x0, x1 = xlim
    elif log_x and x_all.size:
        x0 = max(1.0, float(np.min(x_all)) / 1.6)
        x1 = float(np.max(x_all)) * 1.6
    elif x_all.size:
        x0 = 0.0
        x1 = float(np.max(x_all)) * 1.12
    else:
        x0, x1 = (70.0, 4.2e4) if log_x else (0.0, 1.0)

    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlim(x0, x1)
    ax.set_ylim(u_hi + pad, u_lo - pad)
    ax.figure.canvas.draw()
    # Place median label near the left of the data area (works for log/linear).
    x_lab = 10 ** (0.08 * np.log10(x1 / x0) + np.log10(x0)) if log_x and x0 > 0 else x0 + 0.04 * (x1 - x0)
    _, y_ax = ax.transAxes.inverted().transform(ax.transData.transform((x_lab, med)))
    ax.text(
        0.035,
        float(y_ax) - 0.032,
        r"median $U_0$",
        transform=ax.transAxes,
        fontsize=7.5,
        color=MUTED,
        va="top",
        ha="left",
        path_effects=_HALO,
        zorder=5,
        clip_on=False,
    )
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, colors="black")
    for sp in ax.spines.values():
        sp.set_color("black")
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE, color=INK)
    ax.set_ylabel(r"Mean $U_0$", fontsize=AXIS_LABEL_SIZE, color=INK)
    set_panel_title(ax, panel_title or "Landscape position versus abundance")

    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=OTHER,
                markeredgecolor="white",
                markersize=6.0,
                label="Other types (n=13)",
            )
        ],
        loc="lower right",
        bbox_to_anchor=(0.99, 0.13),
        fontsize=LEGEND_SIZE,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    ax.text(
        0.98,
        0.03,
        f"{MAC_LAB} + {ALV_LAB}:  {100 * stats['frac_mac_alv']:.1f}%\n({n_mac + n_alv:,} / {stats['n_total']:,})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=ANNOT_SIZE,
        color=MUTED,
        linespacing=1.25,
        path_effects=_HALO,
        zorder=5,
    )


def paint_subtype_urel(
    ax_alv,
    ax_mac,
    subtypes: pd.DataFrame,
    *,
    panel_title: str | None = None,
) -> None:
    """Right panel of why_mac_alv_focus: Alv + Mac subtype mean U_rel bars."""
    alv_sub = subtypes.loc[subtypes.annotation == "alv_epithelium"]
    mac_sub = subtypes.loc[subtypes.annotation == "macrophages"]
    _urel_bars(ax_alv, alv_sub)
    _urel_bars(ax_mac, mac_sub)
    urel_lo = float(subtypes["mean_Urel"].min())
    urel_hi = float(subtypes["mean_Urel"].max())
    ax_mac.set_xlim(urel_lo - 0.14, urel_hi + 0.16)
    ax_alv.tick_params(axis="x", labelbottom=False, colors="black")
    ax_mac.tick_params(axis="x", colors="black")
    for ax in (ax_alv, ax_mac):
        for sp in ax.spines.values():
            if sp.get_visible():
                sp.set_color("black")
    ax_mac.set_xlabel(r"Mean $U_{rel}$", fontsize=AXIS_LABEL_SIZE, color=INK)
    ax_alv.set_ylabel(ALV_LAB, fontsize=AXIS_LABEL_SIZE, color=INK)
    ax_mac.set_ylabel(MAC_LAB, fontsize=AXIS_LABEL_SIZE, color=INK)
    set_panel_title(ax_alv, panel_title or "Subtype mean relative potential")


def draw() -> Path:
    ranked, stats, subtypes = _load()

    fig = plt.figure(figsize=(11.4, 4.65))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.18, 1.0],
        height_ratios=[4.0, 7.0],
        wspace=0.55,
        hspace=0.22,
        left=0.09,
        right=0.98,
        top=0.88,
        bottom=0.12,
    )
    ax = fig.add_subplot(gs[:, 0])
    paint_landscape(ax, ranked, stats)

    ax_alv = fig.add_subplot(gs[0, 1])
    ax_mac = fig.add_subplot(gs[1, 1], sharex=ax_alv)
    paint_subtype_urel(ax_alv, ax_mac, subtypes)

    out = OUT / "GSE141259_why_mac_alv_focus.png"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)

    (TAB / "GSE141259_why_mac_alv_focus_stats.json").write_text(
        json.dumps({"checkpoint": str(CK), **stats}, indent=2),
        encoding="utf-8",
    )
    return out


if __name__ == "__main__":
    print("wrote", draw())
