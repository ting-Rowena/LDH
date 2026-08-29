#!/usr/bin/env python3
"""Supplementary Figure 4: Atf3 honesty controls (Egr1 / cross-type / OE).

Fully redrawn from CSV for a unified journal style:
  - Egr1 KO SNIIC tracks
  - Cross-type Atf3-KO dumbbell
  - Atf3-OE sufficiency tracks (1×4)

Default output:
  output_file/Supplementary_figure4.png

Usage:
  python output_file/Supplementary_figure4.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

if os.environ.get("MPLBACKEND") is None:
    mpl.use("Agg", force=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "output_file"))
from _supp_compose import save_fig  # noqa: E402

CK = ROOT / (
    "GSE155622_checkpoints_3000_3000_384_0.05_recon0.01_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
)
EGR1_TRACK = CK / "methods_enhancement" / "in_silico_KO_Egr1_hybrid_shift1_SNIIC_track.csv"
OE_TRACK = ROOT / "output_file" / "robustness" / "atf3_oe_and_path_cost" / "Atf3_OE_Control_tracks.csv"
STATS = ROOT / "output_file" / "robustness" / "p2_robustness" / "Atf3_cross_type_stats.csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "Supplementary_figure4.png"

# Unified academic palette
INK = "#1F2933"
MUTED = "#6B7280"
GRID = "#E9EEF3"
BAND = "#F5F7FA"
WT_DOT = "#4F7FA0"
KO_DOT = "#C56B63"

SNIIC_COLORS = {
    "SNIIC1": "#4F9A94",
    "SNIIC2": "#C9897C",
    "SNIIC3": "#B8945F",
}
OE_SPECS = [
    ("SNIIC1_noAtf3", "SNIIC1 partners", "#4F9A94"),
    ("Nav_triad", "Nav triad", "#3D6F8E"),
    ("Csf1", r"$\boldsymbol{Csf1}$", "#C47A4A"),
    ("Atf3_alone", r"$\boldsymbol{Atf3}$ alone", "#8B4B5C"),
]


def _bold_title(ax, text: str, *, fontsize: float = 9.0, pad: float = 4, loc: str = "center") -> None:
    """Set axis title: all bold; gene names already marked with \\boldsymbol{} stay bold-italic."""
    ax.set_title(
        text,
        fontsize=fontsize,
        fontweight="bold",
        color=INK,
        pad=pad,
        loc=loc,
    )


def _style_axis(ax, *, grid: str = "y") -> None:
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#D0D7DE")
    ax.spines["bottom"].set_color("#D0D7DE")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=7.2, length=2.4, width=0.6)
    if grid == "y":
        ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    elif grid == "x":
        ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def _letter(ax, letter: str, *, x: float = -0.02, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        str(letter).upper(),
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=INK,
        clip_on=False,
    )


def _plot_pair(
    ax,
    t_wt,
    y_wt,
    t_ko,
    y_ko,
    *,
    color: str,
    wt_label: str,
    ko_label: str,
    show_legend: bool = False,
) -> None:
    ax.plot(
        t_wt,
        y_wt,
        "-",
        color=color,
        lw=2.0,
        marker="o",
        ms=4.2,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.35,
        label=wt_label,
        zorder=3,
    )
    ax.plot(
        t_ko,
        y_ko,
        "--",
        color=color,
        lw=1.85,
        marker="s",
        ms=4.0,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=0.6,
        alpha=0.92,
        label=ko_label,
        zorder=3,
    )
    if show_legend:
        ax.legend(
            frameon=False,
            fontsize=7.0,
            loc="best",
            handlelength=1.5,
            handletextpad=0.35,
            borderaxespad=0.15,
            labelcolor=INK,
        )


def draw_egr1(gs_slot, fig) -> None:
    track = pd.read_csv(EGR1_TRACK)
    wt = track[track["condition"] == "wildtype"]
    ko = track[track["condition"] == "KO_Egr1"]
    modules = ["SNIIC1", "SNIIC2", "SNIIC3"]

    # Three equal panels; shared legend tucked into the rightmost plot
    inner = gs_slot.subgridspec(1, 3, wspace=0.20)
    axes = [fig.add_subplot(inner[0, i]) for i in range(3)]

    ymax = 0.0
    for ax, mod in zip(axes, modules):
        color = SNIIC_COLORS[mod]
        _plot_pair(
            ax,
            wt["t"],
            wt[mod],
            ko["t"],
            ko[mod],
            color=color,
            wt_label="WT",
            ko_label=r"$\boldsymbol{Egr1}$-KO",
        )
        _bold_title(ax, mod, fontsize=9.2, pad=3)
        _style_axis(ax)
        ymax = max(ymax, float(wt[mod].max()), float(ko[mod].max()))

    axes[0].set_ylabel("Module score", fontsize=8.0, color=INK, fontweight="bold")
    for i, ax in enumerate(axes):
        ax.set_ylim(0, ymax * 1.12)
        ax.set_xlim(float(wt["t"].min()), float(wt["t"].max()))
        if i != 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Simulated time (days)", fontsize=7.5, color=INK, fontweight="bold")

    handles = [
        Line2D(
            [0],
            [0],
            color=INK,
            lw=2.0,
            ls="-",
            marker="o",
            ms=4.5,
            markerfacecolor="white",
            markeredgecolor=INK,
            markeredgewidth=1.3,
            label="WT",
        ),
        Line2D(
            [0],
            [0],
            color=INK,
            lw=1.85,
            ls="--",
            marker="s",
            ms=4.2,
            markerfacecolor=INK,
            markeredgewidth=0,
            label=r"$\boldsymbol{Egr1}$-KO",
        ),
    ]
    axes[2].legend(
        handles=handles,
        loc="center right",
        frameon=False,
        fontsize=7.0,
        handlelength=1.4,
        borderaxespad=0.2,
        labelcolor=INK,
    )
    _letter(axes[0], "A", x=-0.14, y=1.06)
    axes[0].text(
        0.04,
        1.06,
        r"$\boldsymbol{Egr1}$-KO negative control",
        transform=axes[0].transAxes,
        fontsize=9.0,
        fontweight="bold",
        color=INK,
        va="bottom",
        ha="left",
        clip_on=False,
    )


def draw_crosstype(ax) -> None:
    ct = pd.read_csv(STATS)
    focus = [
        ("Neuron", "Atf3_alone", "Neuron", r"$\boldsymbol{Atf3}$"),
        ("Fibroblast", "FB_remodel", "Fibroblast", "FB_remodel"),
        ("Fibroblast", "FB_ECM", "Fibroblast", "FB_ECM"),
    ]
    rows = []
    for ctype, mod, major, minor in focus:
        r = ct.loc[(ct.cell_type == ctype) & (ct.module == mod)].iloc[0]
        rows.append(
            {
                "major": major,
                "minor": minor,
                "WT": float(r["end_WT"]),
                "KO": float(r["end_KO"]),
                "ratio": float(r["end_ratio_KO_over_WT"]),
            }
        )
    df = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)

    for i, row in df.iterrows():
        if row["major"] == "Fibroblast":
            ax.axhspan(i - 0.38, i + 0.38, color=BAND, zorder=0, lw=0)
        ax.plot(
            [row["KO"], row["WT"]],
            [i, i],
            color="#CBD5E1",
            lw=2.4,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            row["WT"],
            i,
            s=64,
            color=WT_DOT,
            edgecolors="white",
            linewidths=1.0,
            zorder=4,
            label="WT" if i == len(df) - 1 else None,
        )
        ax.scatter(
            row["KO"],
            i,
            s=64,
            color=KO_DOT,
            edgecolors="white",
            linewidths=1.0,
            zorder=4,
            label=r"$\boldsymbol{Atf3}$-KO" if i == len(df) - 1 else None,
        )

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([])
    for i, row in df.iterrows():
        ax.text(
            -0.06,
            i + 0.13,
            row["minor"],
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8.0,
            color=INK,
            fontweight="bold",
            clip_on=False,
        )
        ax.text(
            -0.06,
            i - 0.15,
            row["major"],
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6.6,
            color=MUTED,
            clip_on=False,
        )

    # Data range stops before ratio column so values are never covered by dots/legend
    x_max_data = float(max(df["WT"].max(), df["KO"].max()))
    ax.set_xlim(-0.15, x_max_data * 1.08)
    ax.set_ylim(-0.55, len(df) - 0.35)
    ax.set_xlabel("Module score (endpoint)", fontsize=7.8, color=INK, labelpad=2, fontweight="bold")
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    _style_axis(ax, grid="x")
    ax.spines["left"].set_visible(False)

    # KO/WT ratios in right margin; header sits above top row (not into subplot title)
    top_i = float(len(df) - 1)
    ax.text(
        1.02,
        top_i + 0.42,
        "KO/WT",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=MUTED,
        fontweight="bold",
        clip_on=False,
    )
    for i, row in df.iterrows():
        ax.text(
            1.02,
            i,
            f"{row['ratio']:.2f}×",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=8.0,
            color=MUTED,
            fontweight="bold",
            clip_on=False,
            zorder=5,
        )

    ax.set_title(
        r"Cross-type $\boldsymbol{Atf3}$-KO · not neuron-exclusive",
        fontsize=9.0,
        fontweight="bold",
        color=INK,
        pad=10,
        loc="left",
    )
    ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=7.0,
        handlelength=0.9,
        markerscale=0.9,
        borderaxespad=0.15,
        labelcolor=INK,
    )
    _letter(ax, "B", x=-0.28, y=1.08)


def draw_oe(gs_slot, fig) -> None:
    tracks = pd.read_csv(OE_TRACK)
    wt = tracks[tracks["condition"] == "WT"]
    oe = tracks[tracks["condition"] == "Atf3_OE"]
    axes = [fig.add_subplot(gs_slot[0, i]) for i in range(4)]

    for i, (ax, (col, title, color)) in enumerate(zip(axes, OE_SPECS)):
        _plot_pair(
            ax,
            wt["t"],
            wt[col],
            oe["t"],
            oe[col],
            color=color,
            wt_label="WT",
            ko_label=r"$\boldsymbol{Atf3}$-OE ×3",
            show_legend=(i == 0),
        )
        _bold_title(ax, title, fontsize=8.8, pad=3)
        ax.set_xlabel("Simulated time (days)", fontsize=7.3, color=INK, fontweight="bold")
        if i == 0:
            ax.set_ylabel("NN module score", fontsize=8.0, color=INK, fontweight="bold")
        _style_axis(ax)
        y_all = np.concatenate([wt[col].to_numpy(), oe[col].to_numpy()])
        ax.set_ylim(0, float(np.nanmax(y_all)) * 1.15)
        ax.set_xlim(float(wt["t"].min()), float(wt["t"].max()))

    _letter(axes[0], "C", x=-0.12, y=1.08)
    axes[0].text(
        0.04,
        1.08,
        r"$\boldsymbol{Atf3}$-OE (×3) sufficiency readouts",
        transform=axes[0].transAxes,
        fontsize=9.0,
        fontweight="bold",
        color=INK,
        va="bottom",
        ha="left",
        clip_on=False,
    )


def _ensure_inputs() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    if not EGR1_TRACK.is_file():
        print("[supp fig4] computing Egr1 hybrid KO tracks...", flush=True)
        from run_in_silico_knockout import run_knockout_gse155622

        out = CK / "methods_enhancement"
        run_knockout_gse155622(CK, ["Egr1"], out, ko_mode="hybrid", latent_shift_scale=1.0)
    if not OE_TRACK.is_file():
        print("[supp fig4] computing Atf3 OE tracks...", flush=True)
        from run_atf3_oe_and_path_cost import main as _oe

        _oe()
    if not STATS.is_file():
        print("[supp fig4] computing Atf3 cross-type stats...", flush=True)
        from run_p2_cross_type_and_balanced_U import main as _p2

        _p2()
    for p in (EGR1_TRACK, OE_TRACK, STATS):
        if not p.is_file():
            raise FileNotFoundError(p)


def compose(out: Path | None = None) -> Path:
    out = Path(out or DEFAULT_OUT)
    _ensure_inputs()

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "axes.unicode_minus": False,
        }
    )

    fig = plt.figure(figsize=(15.4, 9.4), facecolor="white")
    # Extra top margin so the figure title does not collide with panel-a title
    outer = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1.12, 1.0],
        hspace=0.42,
        left=0.05,
        right=0.97,
        top=0.93,
        bottom=0.07,
    )

    # Wider gap between a and b so b's left labels do not cover a
    top = outer[0].subgridspec(1, 2, width_ratios=[3.2, 1.0], wspace=0.22)
    draw_egr1(top[0], fig)
    ax_b = fig.add_subplot(top[1])
    draw_crosstype(ax_b)
    # Left pad for y-labels; right pad for KO/WT ratio column
    pos = ax_b.get_position()
    ax_b.set_position([pos.x0 + 0.018, pos.y0, pos.width - 0.055, pos.height])

    bottom = outer[1].subgridspec(1, 4, wspace=0.22)
    draw_oe(bottom, fig)

    return save_fig(fig, out)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    compose(out=Path(argv[0]) if argv else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
