#!/usr/bin/env python3
"""Generate manuscript Figure 2 (framework validation).

Figure 1 is a separately made model schematic (not generated here).

Self-contained visualization script (former assemble_figure1 / cohort panels inlined).

Output (default):
  output_file/figure2.png

Usage (from repo root or any cwd):
  python output_file/figure2.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR))
from _adopted import CK_HG, CK_LUNG, CK_PAIN, load_null_summary  # noqa: E402

TABLE2_CSV = ROOT / "deep_temporal_benchmark_compare" / "Supplementary_table2.csv"

# ---------------------------------------------------------------------------
# Style (inlined from panel_style.py)
# ---------------------------------------------------------------------------
PANEL_TITLE_SIZE = 10
PANEL_TITLE_WEIGHT = "bold"
PANEL_TITLE_LOC = "center"
PANEL_TITLE_PAD = 4
AXIS_LABEL_SIZE = 9
TICK_LABEL_SIZE = 8.5
LEGEND_SIZE = 7.5
ANNOT_SIZE = 7
YGRID_KW = dict(linestyle=":", linewidth=0.6, color="0.85", zorder=0)

PAL = ["#1f4e79", "#2e75b6", "#c45911", "#548235", "#7030a0", "#a9d08e"]
DS_LABELS = ["Neuropathic Pain", "Bleomycin Lung Injury", "HGSOC"]
BAR_TIME = "#35568a"
BAR_NAIVE = "#7d5434"
BAR_POST = "#ae9b88"
INK = "#1a1a1a"
XTICK_SIZE = TICK_LABEL_SIZE


def apply_panel_title_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "font.size": TICK_LABEL_SIZE,
            "axes.titlesize": PANEL_TITLE_SIZE,
            "axes.titleweight": PANEL_TITLE_WEIGHT,
            "axes.titlelocation": PANEL_TITLE_LOC,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
        }
    )


def set_panel_title(ax, title: str, *, pad: float | None = None, color=None, **kwargs) -> None:
    kw = {
        "loc": PANEL_TITLE_LOC,
        "fontweight": PANEL_TITLE_WEIGHT,
        "fontsize": PANEL_TITLE_SIZE,
        "pad": PANEL_TITLE_PAD if pad is None else pad,
    }
    if color is not None:
        kw["color"] = color
    kw.update(kwargs)
    ax.set_title(title, **kw)


def apply_ygrid(ax) -> None:
    ax.yaxis.grid(True, **YGRID_KW)
    ax.set_axisbelow(True)


def _style() -> None:
    apply_panel_title_rc()
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.dpi": 300,
            "axes.titlelocation": "center",
            "font.size": 9,
        }
    )


def _spine(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#444")
    ax.spines["bottom"].set_color("#444")
    ax.tick_params(colors=INK, length=3, width=0.7, labelsize=XTICK_SIZE)


def _set_xticklabels(ax, labels) -> None:
    ax.set_xticklabels(
        list(labels),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=XTICK_SIZE,
        fontfamily="sans-serif",
    )
    ax.tick_params(axis="x", labelsize=XTICK_SIZE, pad=2)


# ---------------------------------------------------------------------------
# Cohort / pairing panels (from plot_dataset_cohort_pairing.py)
# ---------------------------------------------------------------------------
def _load_counts() -> dict:
    pain = pd.read_csv(CK_PAIN / "obs.csv", usecols=["condition"], low_memory=False)
    lung = pd.read_csv(CK_LUNG / "obs.csv", usecols=["stage", "orig.ident"], low_memory=False)
    hg = pd.read_csv(
        CK_HG / "obs.csv",
        usecols=["patient_id", "treatment_phase", "stage"],
        low_memory=False,
    )

    pain_order = ["Control", "SNI 6h", "SNI 24h", "SNI 2d", "SNI 7d", "SNI 14d"]
    pain_n = pain["condition"].astype(str).value_counts().reindex(pain_order).fillna(0).astype(int)

    lung_order = ["D0", "D3", "D7", "D10", "D14", "D21", "D28"]
    lung_n = lung["stage"].astype(str).value_counts().reindex(lung_order).fillna(0).astype(int)
    lung_mice = (
        lung.groupby(lung["stage"].astype(str))["orig.ident"].nunique().reindex(lung_order).fillna(0).astype(int)
    )

    phase_map = {
        "treatment-naive": "naive",
        "treatment_naive": "naive",
        "post-NACT": "post",
        "post_NACT": "post",
    }
    hg = hg.assign(
        phase=hg["treatment_phase"].astype(str).map(phase_map).fillna(hg["treatment_phase"].astype(str)),
        stage=hg["stage"].astype(str),
        patient_id=hg["patient_id"].astype(str),
    )
    patients = sorted(hg["patient_id"].unique())
    phase_by_patient = (
        hg.groupby(["patient_id", "phase"]).size().unstack(fill_value=0).reindex(patients).fillna(0).astype(int)
    )
    for col in ("naive", "post"):
        if col not in phase_by_patient.columns:
            phase_by_patient[col] = 0
    phase_by_patient = phase_by_patient[["naive", "post"]]

    return {
        "pain_n": pain_n,
        "lung_n": lung_n,
        "lung_mice": lung_mice,
        "hg_patient": phase_by_patient,
    }


def _draw_pain(ax, counts: pd.Series) -> None:
    x = np.arange(len(counts))
    ymax = float(counts.max())
    ax.bar(x, counts.values, color=BAR_TIME, edgecolor="white", width=0.72, zorder=3)
    for b, v in zip(ax.patches, counts.values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + ymax * 0.015,
            f"{v:,}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=INK,
            zorder=4,
        )
    ax.set_xticks(x)
    _set_xticklabels(ax, counts.index)
    ax.set_ylabel("Cells", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(0, ymax * 1.18)
    apply_ygrid(ax)
    _spine(ax)
    ax.tick_params(axis="x", labelsize=XTICK_SIZE)


def _draw_lung(ax, counts: pd.Series, mice: pd.Series) -> None:
    x = np.arange(len(counts))
    ymax = float(counts.max())
    ax.bar(x, counts.values, color=BAR_TIME, edgecolor="white", width=0.72, zorder=3)
    for b, v, m in zip(ax.patches, counts.values, mice.values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + ymax * 0.015,
            f"{v:,}\n({m} mice)",
            ha="center",
            va="bottom",
            fontsize=6,
            color=INK,
            linespacing=1.05,
            zorder=4,
        )
    ax.set_xticks(x)
    _set_xticklabels(ax, counts.index)
    ax.set_ylabel("Cells", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(0, ymax * 1.28)
    apply_ygrid(ax)
    _spine(ax)
    ax.tick_params(axis="x", labelsize=XTICK_SIZE)


def _draw_hgsoc_patients(ax, by_patient: pd.DataFrame) -> None:
    patients = sorted(
        by_patient.index,
        key=lambda p: int("".join(ch for ch in str(p) if ch.isdigit()) or 0),
    )
    mat = by_patient.reindex(patients)
    x = np.arange(len(patients))
    w = 0.36
    ymax = float(max(mat["naive"].max(), mat["post"].max()))
    bars_n = ax.bar(
        x - w / 2, mat["naive"], w, color=BAR_NAIVE, label="treatment-naive", edgecolor="white", zorder=3
    )
    bars_p = ax.bar(
        x + w / 2, mat["post"], w, color=BAR_POST, label="post-NACT", edgecolor="white", zorder=3
    )
    for bars in (bars_n, bars_p):
        for b in bars:
            v = b.get_height()
            if v <= 0:
                continue
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + ymax * 0.012,
                f"{int(v):,}",
                ha="center",
                va="bottom",
                fontsize=5.5,
                color=INK,
                zorder=4,
            )
    ax.set_xticks(x)
    _set_xticklabels(ax, [str(p) for p in patients])
    ax.set_xlim(-0.6, len(patients) - 0.4)
    ax.set_ylabel("Cells", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(0, ymax * 1.18)
    ax.legend(fontsize=LEGEND_SIZE, frameon=False, loc="upper right")
    apply_ygrid(ax)
    _spine(ax)
    ax.tick_params(axis="x", labelsize=XTICK_SIZE)


# ---------------------------------------------------------------------------
# Supplementary Table 2 (PCA-50 hold-out vs Waddington-OT / PRESCIENT / MIOFlow)
# ---------------------------------------------------------------------------
TABLE2_DATASETS = ["Pain", "Lung", "HGSOC"]
TABLE2_METHODS = [
    ("LDH-scRNA", "LDH-scRNA", PAL[0]),
    ("Waddington-OT", "Waddington-OT", "#7aa2c4"),
    ("PRESCIENT-family", "PRESCIENT", "#c48a6a"),  # soft copper
    ("MIOFlow-family", "MIOFlow", "#3d7a78"),  # deep teal
]


def _load_table2() -> pd.DataFrame:
    df = pd.read_csv(TABLE2_CSV)
    missing = set(TABLE2_DATASETS) - set(df["dataset"].astype(str))
    if missing:
        raise RuntimeError(f"{TABLE2_CSV} missing datasets: {sorted(missing)}")
    return df


def _paint_table2_metric(
    ax,
    df: pd.DataFrame,
    column: str,
    *,
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
    fmt: str = "{:.2f}",
    zero_line: bool = False,
) -> None:
    x = np.arange(len(TABLE2_DATASETS))
    n = len(TABLE2_METHODS)
    w = 0.18
    offsets = (np.arange(n) - (n - 1) / 2.0) * w
    y_span = ylim[1] - ylim[0]
    for i, (method, label, color) in enumerate(TABLE2_METHODS):
        vals = [
            float(df.loc[(df["dataset"] == d) & (df["method"] == method), column].iloc[0])
            for d in TABLE2_DATASETS
        ]
        ax.bar(
            x + offsets[i],
            vals,
            w,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        for xi, v in zip(x + offsets[i], vals):
            above = v >= 0
            ax.text(
                xi,
                v + (0.012 * y_span if above else -0.012 * y_span),
                fmt.format(v),
                ha="center",
                va="bottom" if above else "top",
                fontsize=5.4,
                zorder=3,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(DS_LABELS, fontsize=TICK_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="both", length=3, width=0.8, labelsize=TICK_LABEL_SIZE)
    set_panel_title(ax, title)
    if zero_line:
        ax.axhline(0, color="0.8", lw=0.8, zorder=1)
    ax.legend(
        fontsize=LEGEND_SIZE,
        frameon=False,
        loc="upper right",
        handlelength=1.2,
        handletextpad=0.4,
        borderaxespad=0.2,
        labelspacing=0.3,
    )
    apply_ygrid(ax)
    _spine(ax)


# ---------------------------------------------------------------------------
# Row-2 validation panels
# ---------------------------------------------------------------------------
def _paint_fig1b(ax, df: pd.DataFrame) -> None:
    datasets = ["GSE155622", "GSE141259", "HGSOC"]
    methods = [
        ("MomentumNetwork_markov", "LDH-scRNA", PAL[0]),
        ("CellRank", "CellRank", "#9aa8b5"),
        ("scVelo", "scVelo", "#c9c0b6"),
    ]
    # Fall back if older cache lacks the Markov-on-momentum column.
    if "MomentumNetwork_markov" not in df.columns:
        methods[0] = ("MomentumNetwork", "LDH-scRNA", PAL[0])
    x = np.arange(len(datasets))
    w = 0.25
    for i, (col, label, c) in enumerate(methods):
        vals = [float(df.loc[df.dataset == d, col].iloc[0]) for d in datasets]
        ax.bar(x + (i - 1) * w, vals, w, label=label, color=c, edgecolor="white", linewidth=0.4, zorder=2)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=ANNOT_SIZE, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["Neuropathic Pain", "Bleomycin Lung Injury", "HGSOC"], fontsize=TICK_LABEL_SIZE)
    ax.set_ylabel("Trajectory–time Pearson r", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis="both", length=3, width=0.8, labelsize=TICK_LABEL_SIZE)
    set_panel_title(ax, "Trajectory–time benchmark")
    ax.legend(fontsize=LEGEND_SIZE, frameon=False, loc="upper right")
    ax.axhline(0, color="0.8", lw=0.8, zorder=1)
    apply_ygrid(ax)


def _paint_fig1c(ax, summ: pd.DataFrame) -> None:
    x_labels = ["Neuropathic Pain", "Bleomycin Lung Injury", "HGSOC"]
    x = np.arange(len(summ))
    w = 0.35
    ax.bar(x - w / 2, summ["real_spearman"], w, label="Real", color=PAL[0], zorder=2)
    ax.bar(x + w / 2, summ["null_median_spearman"], w, label="Null", color=PAL[2], zorder=2)
    for i, r in enumerate(summ["collapse_ratio"]):
        y = max(summ.loc[i, "real_spearman"], summ.loc[i, "null_median_spearman"]) + 0.05
        ax.text(i, y, f"ratio={r:.2f}", ha="center", fontsize=ANNOT_SIZE, zorder=3)
    ax.axhline(0, color="0.75", lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=TICK_LABEL_SIZE)
    ax.set_ylabel(r"Spearman($U_0$, −log KDE)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(-1.2, 1.25)
    ax.tick_params(axis="both", length=3, width=0.8, labelsize=TICK_LABEL_SIZE)
    set_panel_title(ax, "Matched temporal null (U–KDE)")
    ax.legend(fontsize=LEGEND_SIZE, frameon=False, loc="upper right")
    apply_ygrid(ax)


def _paint_fig1d(ax, summ: pd.DataFrame) -> None:
    x_labels = ["Neuropathic Pain", "Bleomycin Lung Injury", "HGSOC"]
    x = np.arange(len(summ))
    w = 0.35
    bars_real = ax.bar(
        x - w / 2,
        summ["real_holdout_pcc"],
        w,
        label="Real",
        color=PAL[0],
        edgecolor="white",
        linewidth=0.4,
        zorder=2,
    )
    bars_null = ax.bar(
        x + w / 2,
        summ["null_median_holdout_pcc"],
        w,
        label="Null",
        color=PAL[2],
        edgecolor="white",
        linewidth=0.4,
        zorder=2,
    )
    for i, r in enumerate(summ["holdout_pcc_collapse_ratio"]):
        y = max(summ.loc[i, "real_holdout_pcc"], summ.loc[i, "null_median_holdout_pcc"]) + 0.035
        ax.text(i, y, f"null/real={r:.2f}", ha="center", va="bottom", fontsize=ANNOT_SIZE, color="0.25")
    for b in list(bars_real) + list(bars_null):
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h - 0.035 if h > 0.12 else h + 0.01,
            f"{h:.2f}",
            ha="center",
            va="top" if h > 0.12 else "bottom",
            fontsize=6.5,
            color="white" if h > 0.12 else "0.3",
            fontweight="medium",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=TICK_LABEL_SIZE)
    ax.set_ylabel("Holdout predictive PCC", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.55, 2.85)
    ax.tick_params(axis="both", length=3, width=0.8, labelsize=TICK_LABEL_SIZE)
    set_panel_title(ax, "Holdout PCC under matched temporal null", pad=8)
    ax.legend(fontsize=LEGEND_SIZE, frameon=False, loc="upper right")
    apply_ygrid(ax)

    hgs = summ[summ.dataset == "HGSOC"].iloc[0]
    y_real = float(hgs.real_holdout_pcc)
    y_null = float(hgs.null_median_holdout_pcc)
    x_span = 2.42
    callout = "black"
    ax.annotate(
        "",
        xy=(x_span, y_null),
        xytext=(x_span, y_real),
        arrowprops=dict(arrowstyle="<->", color=callout, lw=1.15, shrinkA=0, shrinkB=0),
        zorder=3,
    )
    ax.plot([2 + w / 2 + 0.02, x_span], [y_real, y_real], color=callout, lw=0.7, ls="-", alpha=0.85, zorder=3)
    ax.plot([2 + w / 2 + 0.02, x_span], [y_null, y_null], color=callout, lw=0.7, ls="-", alpha=0.85, zorder=3)
    ax.text(
        x_span + 0.06,
        0.5 * (y_real + y_null),
        f"{y_real:.3f} → {y_null:.3f}",
        ha="left",
        va="center",
        fontsize=8,
        color=callout,
        fontweight="semibold",
        zorder=3,
    )


# ---------------------------------------------------------------------------
# Assemble Figure 2
# ---------------------------------------------------------------------------
def assemble_figure2(out: Path | None = None) -> Path:
    """Row1: cohort. Row2: Supplementary Table 2 (L2 / MDC / VPA). Row3: temporal null."""
    _style()
    data = _load_counts()
    table2 = _load_table2()
    summ = load_null_summary()
    summ = summ.reset_index(drop=True)

    fig = plt.figure(figsize=(14.5, 12.8))
    outer = GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[1.08, 1.0, 1.0],
        hspace=0.38,
        left=0.065,
        right=0.985,
        top=0.97,
        bottom=0.055,
    )
    gs0 = outer[0].subgridspec(1, 3, width_ratios=[1.0, 1.1, 2.0], wspace=0.22)
    gs1 = outer[1].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.18)
    gs2 = outer[2].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.22)

    ax_p = fig.add_subplot(gs0[0, 0])
    _draw_pain(ax_p, data["pain_n"])
    set_panel_title(ax_p, DS_LABELS[0], pad=5)

    ax_l = fig.add_subplot(gs0[0, 1])
    _draw_lung(ax_l, data["lung_n"], data["lung_mice"])
    set_panel_title(ax_l, DS_LABELS[1], pad=5)

    ax_h = fig.add_subplot(gs0[0, 2])
    _draw_hgsoc_patients(ax_h, data["hg_patient"])
    set_panel_title(ax_h, DS_LABELS[2], pad=5)

    ax_l2 = fig.add_subplot(gs1[0, 0])
    ax_mdc = fig.add_subplot(gs1[0, 1])
    ax_vpa = fig.add_subplot(gs1[0, 2])
    _paint_table2_metric(
        ax_l2,
        table2,
        "centroid Euclidean distance",
        title="centroid Euclidean distance",
        ylabel="Distance (lower better)",
        ylim=(0, 14.8),
    )
    _paint_table2_metric(
        ax_mdc,
        table2,
        "Mean Displacement Cosine",
        title="Mean Displacement Cosine",
        ylabel="Cosine (higher better)",
        ylim=(-0.28, 1.12),
        zero_line=True,
    )
    _paint_table2_metric(
        ax_vpa,
        table2,
        "Velocity Projection Alignment",
        title="Velocity Projection Alignment",
        ylabel="Cosine (higher better)",
        ylim=(-0.22, 1.12),
        zero_line=True,
    )

    ax_ukde = fig.add_subplot(gs2[0, 0])
    _paint_fig1c(ax_ukde, summ)
    ax_hold = fig.add_subplot(gs2[0, 1])
    _paint_fig1d(ax_hold, summ)
    ax_hold.set_xlim(-0.55, 2.90)

    if out is None:
        out = OUT_DIR / "figure2.png"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Tight crop to artists so nothing past the shared legends remains
    with matplotlib.rc_context({"savefig.bbox": "tight"}):
        fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", out, flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out = Path(argv[0]) if argv else None
    assemble_figure2(out=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
