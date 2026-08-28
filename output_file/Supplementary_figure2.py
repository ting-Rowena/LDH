#!/usr/bin/env python3
"""Supplementary Figure 2: stage-wise cell-type composition.

Three rows, one column (adopted checkpoints' ``obs.csv``):
  (a) GSE155622 — 6 injury stages × cell-type fractions
  (b) GSE141259 — 7 time points × major cell-type fractions
  (c) HGSOC — per-patient TN vs PN cell counts

Default output:
  output_file/Supplementary_figure2.png

Usage:
  python output_file/Supplementary_figure2.py
  python output_file/Supplementary_figure2.py /path/to/out.png
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
from matplotlib.patches import Patch

if os.environ.get("MPLBACKEND") is None:
    mpl.use("Agg", force=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset_pipeline import (  # noqa: E402
    GSE141259_CELLTYPE_PALETTE,
    GSE155622_CELLTYPE_PALETTE,
    HGSOC_CELLTYPE_PALETTE,
)

CK_PAIN = ROOT / (
    "GSE155622_checkpoints_3000_3000_384_0.05_recon0.01_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
)
CK_LUNG = ROOT / (
    "GSE141259_checkpoints_5000_5000_512_0.01_recon0.1_valD28_timeX_lossnorm_qp_d0p01_z0p1_k0p2"
)
CK_HG = ROOT / (
    "HGSOC_checkpoints_3000_3000_512_0.05_recon0.01_nactpair_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
)
LUNG_MAP = CK_LUNG / "figures" / "GSE141259_metacelltype_formal_label_mapping.csv"

DEFAULT_OUT = Path(__file__).resolve().parent / "Supplementary_figure2.png"

INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#ececec"
TN_EDGE = "#1B4F72"
PN_EDGE = "#C45C26"

PAIN_STAGES = ["Control", "SNI 6h", "SNI 24h", "SNI 2d", "SNI 7d", "SNI 14d"]
LUNG_STAGES = ["D0", "D3", "D7", "D10", "D14", "D21", "D28"]
HG_TYPES = ["EOC", "Immune", "Stromal"]
HG_PHASES = ["treatment-naive", "post-NACT"]
HG_PHASE_LAB = {"treatment-naive": "TN", "post-NACT": "PN"}


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "font.size": 8.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "center",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _spine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.85)
    ax.tick_params(colors=INK, width=0.7, length=3)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.55, color=GRID, zorder=0)
    ax.set_axisbelow(True)


def _panel_letter(ax, letter: str, *, x: float = -0.04, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        str(letter).upper(),
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
        clip_on=False,
    )


def _frac_table(df: pd.DataFrame, stage_col: str, type_col: str, stages: list[str], types: list[str]) -> pd.DataFrame:
    """Rows = stages, columns = types, values = within-stage fraction."""
    sub = df[df[stage_col].isin(stages) & df[type_col].isin(types)].copy()
    counts = (
        sub.groupby([stage_col, type_col], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=stages, columns=types, fill_value=0)
    )
    tot = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(tot, axis=0).fillna(0.0)


def _order_types_by_abundance(df: pd.DataFrame, type_col: str, candidates: list[str]) -> list[str]:
    counts = df[type_col].astype(str).value_counts()
    present = [t for t in candidates if t in counts.index]
    present.sort(key=lambda t: -int(counts.get(t, 0)))
    return present


def _stacked_stage_frac(
    ax,
    frac: pd.DataFrame,
    colors: dict[str, str],
    *,
    display: dict[str, str] | None = None,
) -> list[str]:
    """Draw 100% stacked bars; return type order used (bottom→top)."""
    stages = list(frac.index)
    types = list(frac.columns)
    x = np.arange(len(stages))
    bottom = np.zeros(len(stages))
    for t in types:
        y = frac[t].to_numpy(dtype=float) * 100.0
        ax.bar(
            x,
            y,
            width=0.72,
            bottom=bottom,
            color=colors.get(t, "#BDBDBD"),
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        bottom = bottom + y

    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=0)
    ax.set_xlim(-0.6, len(stages) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Composition (%)")
    _spine(ax)
    return types if display is None else [display.get(t, t) for t in types]


def _plot_pain(ax) -> list[Patch]:
    obs = pd.read_csv(CK_PAIN / "obs.csv", usecols=["condition", "annotation"], low_memory=False)
    types = _order_types_by_abundance(obs, "annotation", list(GSE155622_CELLTYPE_PALETTE))
    frac = _frac_table(obs, "condition", "annotation", PAIN_STAGES, types)
    _stacked_stage_frac(ax, frac, GSE155622_CELLTYPE_PALETTE)
    ax.set_title("GSE155622 · cell-type composition by stage", loc="center", pad=8)
    _panel_letter(ax, "A")
    return [Patch(facecolor=GSE155622_CELLTYPE_PALETTE[t], edgecolor="white", label=t) for t in types]


def _plot_lung(ax) -> list[Patch]:
    obs = pd.read_csv(CK_LUNG / "obs.csv", usecols=["stage", "annotation"], low_memory=False)
    map_df = pd.read_csv(LUNG_MAP)
    formal = dict(zip(map_df["metacelltype"].astype(str), map_df["formal_label"].astype(str)))
    # Prefer mapping order (abundance), fall back to palette keys
    types = [t for t in map_df["metacelltype"].astype(str).tolist() if t in set(obs["annotation"].astype(str))]
    if not types:
        types = _order_types_by_abundance(obs, "annotation", list(GSE141259_CELLTYPE_PALETTE))
    frac = _frac_table(obs, "stage", "annotation", LUNG_STAGES, types)
    _stacked_stage_frac(ax, frac, GSE141259_CELLTYPE_PALETTE)
    ax.set_title("GSE141259 · major cell-type composition by stage", loc="center", pad=8)
    _panel_letter(ax, "B")
    return [
        Patch(
            facecolor=GSE141259_CELLTYPE_PALETTE.get(t, "#BDBDBD"),
            edgecolor="white",
            label=formal.get(t, t),
        )
        for t in types
    ]


def _plot_hgsoc(ax) -> list[Patch]:
    obs = pd.read_csv(
        CK_HG / "obs.csv",
        usecols=["patient_id", "annotation", "treatment_phase"],
        low_memory=False,
    )
    obs = obs[obs["annotation"].isin(HG_TYPES)].copy()
    patients = sorted(
        obs["patient_id"].astype(str).unique(),
        key=lambda p: int("".join(ch for ch in p if ch.isdigit()) or 0),
    )
    n_pat = len(patients)
    width = 0.85
    x_tn = np.arange(n_pat) * 3.0
    x_pn = x_tn + 1.0
    bottoms_tn = np.zeros(n_pat)
    bottoms_pn = np.zeros(n_pat)
    n_tot_tn = np.zeros(n_pat)
    n_tot_pn = np.zeros(n_pat)

    for t in HG_TYPES:
        y_tn, y_pn = [], []
        for pid in patients:
            for ph, bucket in (("treatment-naive", y_tn), ("post-NACT", y_pn)):
                sub = obs[(obs.patient_id == pid) & (obs.treatment_phase == ph)]
                bucket.append(float((sub.annotation == t).sum()))
        y_tn_a = np.asarray(y_tn, dtype=float)
        y_pn_a = np.asarray(y_pn, dtype=float)
        ax.bar(
            x_tn,
            y_tn_a,
            width=width,
            bottom=bottoms_tn,
            color=HGSOC_CELLTYPE_PALETTE[t],
            edgecolor=TN_EDGE,
            linewidth=0.7,
            zorder=3,
        )
        ax.bar(
            x_pn,
            y_pn_a,
            width=width,
            bottom=bottoms_pn,
            color=HGSOC_CELLTYPE_PALETTE[t],
            edgecolor=PN_EDGE,
            linewidth=0.7,
            zorder=3,
        )
        bottoms_tn = bottoms_tn + y_tn_a
        bottoms_pn = bottoms_pn + y_pn_a
        n_tot_tn = bottoms_tn
        n_tot_pn = bottoms_pn

    ymax = float(max(n_tot_tn.max(), n_tot_pn.max()))
    for i in range(n_pat):
        ax.text(
            x_tn[i],
            n_tot_tn[i] + ymax * 0.015,
            f"{int(n_tot_tn[i])}",
            ha="center",
            va="bottom",
            fontsize=5.5,
            color=MUTED,
            zorder=4,
        )
        ax.text(
            x_pn[i],
            n_tot_pn[i] + ymax * 0.015,
            f"{int(n_tot_pn[i])}",
            ha="center",
            va="bottom",
            fontsize=5.5,
            color=MUTED,
            zorder=4,
        )
        ax.text(
            x_tn[i],
            -0.02,
            "TN",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=5.5,
            color=TN_EDGE,
            fontweight="bold",
            clip_on=False,
        )
        ax.text(
            x_pn[i],
            -0.02,
            "PN",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=5.5,
            color=PN_EDGE,
            fontweight="bold",
            clip_on=False,
        )

    ax.set_xticks(x_tn + 0.5)
    ax.set_xticklabels(patients, rotation=35, ha="right")
    ax.set_xlim(-0.8, x_pn[-1] + 1.0)
    ax.set_ylim(0, ymax * 1.14)
    ax.set_ylabel("Cell count")
    ax.set_xlabel("Patient ID")
    ax.set_title("HGSOC · per-patient cell counts (TN vs PN)", loc="center", pad=8)
    _panel_letter(ax, "C")
    _spine(ax)

    handles = [Patch(facecolor=HGSOC_CELLTYPE_PALETTE[t], edgecolor=INK, label=t) for t in HG_TYPES]
    handles += [
        Patch(facecolor="white", edgecolor=TN_EDGE, linewidth=1.4, label="TN (chemo-naive)"),
        Patch(facecolor="white", edgecolor=PN_EDGE, linewidth=1.4, label="PN (post-NACT)"),
    ]
    return handles


def _place_legend(ax_leg, handles: list[Patch], *, ncol: int = 1, fontsize: float = 7.5) -> None:
    ax_leg.set_axis_off()
    ax_leg.legend(
        handles=handles,
        loc="center left",
        fontsize=fontsize,
        frameon=False,
        handlelength=1.2,
        handleheight=0.9,
        borderaxespad=0.0,
        labelspacing=0.35,
        ncol=ncol,
    )


def compose(out: Path | None = None) -> Path:
    out = Path(out or DEFAULT_OUT)
    _apply_style()

    fig = plt.figure(figsize=(11.5, 12.2), facecolor="white")
    gs = GridSpec(
        3,
        2,
        figure=fig,
        width_ratios=[1.0, 0.28],
        height_ratios=[1.0, 1.0, 1.15],
        hspace=0.38,
        wspace=0.04,
        left=0.08,
        right=0.98,
        top=0.97,
        bottom=0.06,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[2, 0])
    handles_a = _plot_pain(ax_a)
    handles_b = _plot_lung(ax_b)
    handles_c = _plot_hgsoc(ax_c)

    _place_legend(fig.add_subplot(gs[0, 1]), handles_a, fontsize=7.2)
    _place_legend(fig.add_subplot(gs[1, 1]), handles_b, fontsize=6.6)
    _place_legend(fig.add_subplot(gs[2, 1]), handles_c, fontsize=7.5)

    out.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"savefig.bbox": None}):
        fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out = Path(argv[0]) if argv else DEFAULT_OUT
    compose(out=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
