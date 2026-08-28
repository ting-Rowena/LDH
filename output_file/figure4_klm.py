#!/usr/bin/env python3
"""Figure 4: exploratory GSE141259 Club path-cost figure (3 panels).

Self-contained script consolidated from:
  - scripts/run_gse141259_three_dynamics_axes.py (plot_club_fate_bias_figure)
  - panel_style.py / plot_utils.py (minimal helpers inlined)

Default output:
  output_file/figure4_klm.png
  (= former GSE141259_club_fate_bias panel; manuscript Figure 4)

Usage:
  python output_file/figure4_klm.py
  python output_file/figure4_klm.py /path/to/out.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

DEFAULT_OUT = Path(__file__).resolve().parent / "figure4_klm.png"

# ---------------------------------------------------------------------------
# Style (inlined)
# ---------------------------------------------------------------------------
PANEL_TITLE_SIZE = 10
PANEL_TITLE_WEIGHT = "bold"
PANEL_TITLE_LOC = "center"
PANEL_TITLE_PAD = 4
INK = "#1f2933"


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


def _style_journal_ax(ax) -> None:
    ax.tick_params(labelsize=7.5, length=2.2, width=0.55, colors="#475467")
    for sp in ax.spines.values():
        sp.set_color("#CBD5E1")
        sp.set_linewidth(0.7)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.55, color="0.88", zorder=0)
    ax.set_axisbelow(True)


def plot_club_fate_bias_figure(
    cell_df: pd.DataFrame,
    summary: dict | pd.Series,
    *,
    out: Path | None = None,
) -> Path:
    """Compact 3-panel exploratory Club path figure (journal style).

    A · Club candidate path costs (ΔU + action)
    B · Descriptive stage-wise tilt (AT2 vs airway)
    C · MHC-II⁺ candidate route costs
    """
    if isinstance(summary, pd.DataFrame):
        summary = summary.iloc[0]
    summary = dict(summary)

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

    out_w, out_h, out_dpi = 3050, 860, 300
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
    _set_panel_title(ax_a0, "Club candidate path costs", pad=4)
    _style_journal_ax(ax_a0)
    ax_a0.tick_params(axis="x", labelbottom=False, length=0)

    # ---- A bottom: actions ----
    ax_a1.bar(xs, actions, color=fate_colors, edgecolor="none", width=0.62, zorder=3)
    ax_a1.set_ylim(0, max(actions) * 1.12)
    ax_a1.set_xticks(xs, fate_short, fontsize=7.2)
    ax_a1.set_xlabel("Candidate endpoint from Club cells", fontsize=7.6, labelpad=2)
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
    _set_panel_title(ax_b, "Club endpoint tilt by stage", pad=4)
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
    route_notes.append(rf"$\Delta U$={float(summary['Club_to_AT2_barrier']):+.2f}")

    xs_c = np.arange(len(route_vals))
    ax_c.bar(xs_c, route_vals, color=route_cols, edgecolor="none", width=0.64, zorder=3)
    ax_c.set_xticks(xs_c, route_labs, fontsize=6.4)
    ax_c.set_ylabel("Path action", fontsize=7.6)
    _set_panel_title(ax_c, r"MHC-II$^+$ candidate route costs", pad=4)
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

    if out is None:
        out = DEFAULT_OUT
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"savefig.bbox": None}):
        fig.savefig(out, dpi=out_dpi, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def _compute_club_fate() -> tuple[pd.DataFrame, pd.DataFrame]:
    c_path = cache_file("GSE141259_club_fate_tilt_cells.csv")
    s_path = cache_file("GSE141259_club_fate_bias_summary.csv")
    if c_path.is_file() and s_path.is_file():
        return pd.read_csv(c_path), pd.read_csv(s_path)
    from run_gse141259_three_dynamics_axes import run_club_fate_bias

    print("[figure4_klm] computing club fate bias from adopted GSE141259 checkpoint...", flush=True)
    cell_df, summary, _paths = run_club_fate_bias()
    if isinstance(summary, dict):
        summary = pd.DataFrame([summary])
    cell_df.to_csv(c_path, index=False)
    summary.to_csv(s_path, index=False)
    return cell_df, summary


def compose(out: Path | None = None) -> Path:
    _apply_style()
    cell_df, summary = _compute_club_fate()
    return plot_club_fate_bias_figure(cell_df, summary, out=out or DEFAULT_OUT)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out = Path(argv[0]) if argv else DEFAULT_OUT
    compose(out=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
