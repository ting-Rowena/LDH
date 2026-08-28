#!/usr/bin/env python3
"""Supplementary Figure 7: audit of curated macrophage 3D arrows (not a path discovery).

The interactive U_rel landscape draws three curated geodesics
  AM (Bleo) → M2 macrophages
  M2 macrophages → Resolution macrophages
  AM (PBS) → Resolution macrophages
Those arrows were specified a priori. This figure asks whether a complete
7-subtype transition graph is warranted on the same UMAP + U_rel geometry.
It does not certify lineage, does not replace the protocol coupling table
(PBS→Bleo / PBS→Resol), and does not treat the three drawn arrows as equally
data-supported (M2–Resolution is the strong local pair; AM(PBS)→Resolution
is a weak-mixing prior).

Layout:
  Row1 a–c  time programs / n vs mean U0 / UMAP kNN mixing
  Row2 d–f  geodesic action / shown-path scores / geodesic occupancy
  Row3 g–i  alternative rejection / pair scatter / macrophage U_rel top view

Default output:
  output_file/Supplementary_figure7.png
"""

from __future__ import annotations

import json
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "output_file"))
from _supp_compose import load_rgb, save_fig  # noqa: E402

TAB = ROOT / "output_file" / "mac_landscape_audit"
CK_LUNG = ROOT / (
    "GSE141259_checkpoints_5000_5000_512_0.01_recon0.1_valD28_timeX_lossnorm_qp_d0p01_z0p1_k0p2"
)
LUNG_PALETTE = CK_LUNG / "figures" / "GSE141259_umap_hierarchical_palette.csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "Supplementary_figure7.png"
TOPVIEW = (
    Path(__file__).resolve().parent
    / "GSE141259_mac_3d_urel_landscape_pastel_mintrose_blackpaths_arrowplus_planes_strongplanes_biggerarrows_prettyplanes_showstyle_boldUrel_blackaxes_majorgrid_topview_journal_v8_display_cbarAligned.png"
)

INK = "#1F2933"
MUTED = "#6B7280"
GRID = "#E9EEF3"
SHOWN_C = "#1D4E89"
REJ_C = "#B4533A"

_PAL = pd.read_csv(LUNG_PALETTE)
TYPE_COLORS = dict(zip(_PAL["cell.type"].astype(str), _PAL["subtype_color"].astype(str)))
FN1_C = TYPE_COLORS["Fn1+ macrophages"]
M2_C = TYPE_COLORS["M2 macrophages"]
RES_C = TYPE_COLORS["Resolution macrophages"]
AM_C = TYPE_COLORS["AM (Bleo)"]
AMPBS_C = TYPE_COLORS["AM (PBS)"]

# Display labels match Supplementary Figure 1 / obs cell.type exactly.
SHORT = {
    "Fn1+ macrophages": "Fn1+ macrophages",
    "M2 macrophages": "M2 macrophages",
    "Resolution macrophages": "Resolution macrophages",
    "AM (PBS)": "AM (PBS)",
    "AM (Bleo)": "AM (Bleo)",
    "Recruited macrophages": "Recruited macrophages",
    "Cd163-/Cd11c+ IMs": "Cd163-/Cd11c+ IMs",
    "Cd163+/Cd11c- IMs": "Cd163+/Cd11c- IMs",
}
# Compact names for panels c–h only, to keep heatmaps/ticks from overlapping.
COMPACT = {
    **SHORT,
    "Fn1+ macrophages": "Fn1+",
    "M2 macrophages": "M2",
    "Resolution macrophages": "Resolution",
}
ORDER = [
    "AM (PBS)",
    "AM (Bleo)",
    "M2 macrophages",
    "Resolution macrophages",
    "Fn1+ macrophages",
    "Cd163-/Cd11c+ IMs",
    "Cd163+/Cd11c- IMs",
]
SHOWN = [
    ("AM (Bleo)", "M2 macrophages"),
    ("M2 macrophages", "Resolution macrophages"),
    ("AM (PBS)", "Resolution macrophages"),
]
ALTS = [
    ("AM (PBS)", "M2 macrophages"),
    ("AM (Bleo)", "Resolution macrophages"),
    ("Fn1+ macrophages", "Resolution macrophages"),
    ("Cd163-/Cd11c+ IMs", "M2 macrophages"),
    ("AM (PBS)", "AM (Bleo)"),
]

# Tight label offsets (points): close to markers, staggered to limit overlap.
BASIN_LABEL_STYLE = {
    "AM (Bleo)": {"xytext": (0, 7), "ha": "center", "va": "bottom"},
    "M2 macrophages": {"xytext": (10, 2), "ha": "left", "va": "center"},
    "Resolution macrophages": {"xytext": (0, -8), "ha": "center", "va": "top"},
    "AM (PBS)": {"xytext": (5, -4), "ha": "left", "va": "top"},
    "Fn1+ macrophages": {"xytext": (10, 7), "ha": "center", "va": "bottom"},
    "Cd163-/Cd11c+ IMs": {"xytext": (-4, -4), "ha": "right", "va": "top"},
    "Cd163+/Cd11c- IMs": {"xytext": (4, 4), "ha": "left", "va": "bottom"},
}


def _path_xtick(src: str, dst: str, *, multiline: bool = True) -> str:
    if multiline:
        return f"{COMPACT[src]}\n$\\rightarrow$ {COMPACT[dst]}"
    return f"{COMPACT[src]} $\\rightarrow$ {COMPACT[dst]}"


def _set_rotated_xticklabels(
    ax,
    labels: list[str],
    *,
    rotation: float = 42,
    fontsize: float = 4.6,
    pad: float = 10,
    vertical: bool = False,
) -> None:
    if vertical or rotation >= 80:
        ax.set_xticklabels(labels, fontsize=fontsize, rotation=90, ha="center", va="top")
    else:
        ax.set_xticklabels(labels, fontsize=fontsize, rotation=rotation, ha="right", va="top")
    ax.tick_params(axis="x", pad=pad)


def _ensure_tables() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    needed = [
        TAB / "GSE141259_mac_fn1_m2_resolution_time_composition.csv",
        TAB / "GSE141259_mac_fn1_m2_resolution_basin_roles.csv",
        TAB / "GSE141259_mac_landscape_umap_edges.csv",
        TAB / "GSE141259_mac_landscape_knn_mixing.csv",
        TAB / "GSE141259_mac_landscape_path_endorsement.json",
    ]
    if not (TAB / "GSE141259_mac_fn1_m2_resolution_time_composition.csv").is_file() or not (
        TAB / "GSE141259_mac_fn1_m2_resolution_basin_roles.csv"
    ).is_file():
        print("[supp fig7] computing Fn1/M2/Resolution context tables...", flush=True)
        from analyze_mac_fn1_m2_resolution_triad import main as _triad

        _triad()
    if not all(p.is_file() for p in needed[-3:]):
        print("[supp fig7] computing UMAP path-endorsement tables...", flush=True)
        from analyze_mac_landscape_path_endorsement import main as _endorsement

        _endorsement()


def _bold_title(ax, text: str, *, fontsize: float = 8.4, pad: float = 3.2) -> None:
    ax.set_title(text, fontsize=fontsize, fontweight="bold", color=INK, pad=pad, loc="center")


Y_LABELPAD = 1.0
X_LABELPAD = 2.0


def _style_axis(ax, *, grid: str = "y") -> None:
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#D0D7DE")
    ax.spines["bottom"].set_color("#D0D7DE")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=6.8, length=2.4, width=0.6)
    ax.yaxis.labelpad = Y_LABELPAD
    ax.xaxis.labelpad = X_LABELPAD
    if grid == "y":
        ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    elif grid == "x":
        ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def _letter(ax, letter: str, *, x: float = -0.10, y: float = 1.10) -> None:
    ax.text(
        x,
        y,
        str(letter).upper(),
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=INK,
        clip_on=False,
    )


def _colorbar(
    cax,
    im,
    label: str,
    *,
    tick_side: str = "right",
    label_side: str | None = None,
) -> None:
    cbar = cax.figure.colorbar(im, cax=cax)
    cbar.set_label(label, fontsize=5.8, color=INK, labelpad=3)
    cbar.ax.tick_params(labelsize=5.3, colors=MUTED, pad=1.0)
    cbar.ax.yaxis.set_ticks_position(tick_side)
    cbar.ax.yaxis.set_label_position(label_side or tick_side)


def _mixing_colorbar(cax, im) -> None:
    """Panel c: ticks face the heatmap; title sits on the outer edge."""
    _colorbar(
        cax,
        im,
        "UMAP kNN mixing",
        tick_side="left",
        label_side="right",
    )


def _align_figure_panels(fig, axes: np.ndarray) -> None:
    return


def _edge_row(edges: pd.DataFrame, src: str, dst: str) -> pd.Series:
    sub = edges[(edges["src"] == src) & (edges["dst"] == dst)]
    if not len(sub):
        raise KeyError(f"missing edge {src} → {dst}")
    return sub.iloc[0]


def _load() -> dict:
    _ensure_tables()
    return {
        "time": pd.read_csv(TAB / "GSE141259_mac_fn1_m2_resolution_time_composition.csv"),
        "basin": pd.read_csv(TAB / "GSE141259_mac_fn1_m2_resolution_basin_roles.csv"),
        "edges": pd.read_csv(TAB / "GSE141259_mac_landscape_umap_edges.csv"),
        "mix": pd.read_csv(TAB / "GSE141259_mac_landscape_knn_mixing.csv"),
        "verdict": json.loads(
            (TAB / "GSE141259_mac_landscape_path_endorsement.json").read_text(encoding="utf-8")
        ),
    }


def _plot_time(ax, time_df: pd.DataFrame) -> None:
    stages = ["D0", "D3", "D7", "D10", "D14", "D21", "D28"]
    x = np.arange(len(stages))
    for ct, c, ls, lw in [
        ("M2 macrophages", M2_C, "-", 2.0),
        ("Resolution macrophages", RES_C, "-", 2.0),
        ("Fn1+ macrophages", FN1_C, "-", 1.6),
        ("AM (Bleo)", AM_C, "--", 1.4),
        ("AM (PBS)", AMPBS_C, "--", 1.4),
    ]:
        sub = time_df[(time_df["compartment"] == "macrophages") & (time_df["cell.type"] == ct)]
        sub = sub.set_index("stage").reindex(stages)
        ax.plot(x, sub["frac"].to_numpy(float), color=c, lw=lw, ls=ls, marker="o", ms=3.5, label=SHORT[ct], zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=6.5)
    ax.set_ylabel("Fraction within macrophages", fontsize=7.3, color=INK)
    _bold_title(ax, "Time programs: M2 window, Resolution sink")
    ax.legend(fontsize=5.0, frameon=False, loc="upper right", ncol=1, handlelength=1.6)
    _style_axis(ax)
    _letter(ax, "A")


def _plot_basin(ax, basin: pd.DataFrame) -> None:
    for _, r in basin.iterrows():
        ct = str(r["cell.type"])
        x = float(r["n"])
        y = float(r["mean_potential_stationary"])
        c = TYPE_COLORS.get(ct, "#9AA0A6")
        shown_node = ct in {a for pair in SHOWN for a in pair}
        s = 120 if shown_node else 42
        ax.scatter(
            [x],
            [y],
            s=s,
            c=c,
            edgecolors="white" if shown_node else "#777",
            linewidths=1.1 if shown_node else 0.4,
            zorder=3 if shown_node else 2,
            alpha=0.95 if shown_node else 0.65,
        )
        style = BASIN_LABEL_STYLE.get(ct, {"xytext": (5, 4), "ha": "left", "va": "bottom"})
        ax.annotate(
            SHORT.get(ct, ct),
            (x, y),
            textcoords="offset points",
            xytext=style["xytext"],
            fontsize=5.2 if shown_node else 4.8,
            fontweight="bold" if shown_node else "normal",
            color=INK,
            ha=style["ha"],
            va=style["va"],
        )
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(max(0, xmin - 0.03 * xmax), xmax * 1.02)
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin
    ax.set_ylim(ymin - 0.10 * yr, ymax + 0.12 * yr)
    ax.set_xlabel("Number of cells", fontsize=7.3, color=INK)
    ax.set_ylabel(r"Mean $U_0$", fontsize=7.3, color=INK)
    _bold_title(ax, r"Basin occupancy: $n$ vs mean $U_0$")
    _style_axis(ax)
    _letter(ax, "B")


def _matrix(df: pd.DataFrame, value: str) -> np.ndarray:
    mat = np.full((len(ORDER), len(ORDER)), np.nan)
    lookup = {(r.src, r.dst): float(getattr(r, value)) for r in df.itertuples()}
    for i, a in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            if a == b:
                mat[i, j] = 0.0
            elif (a, b) in lookup:
                mat[i, j] = lookup[(a, b)]
    return mat


def _plot_mixing(ax, mix: pd.DataFrame, *, cax=None) -> None:
    mat = _matrix(mix, "mixing")
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=max(0.28, np.nanmax(mat)), aspect="equal")
    labels = [COMPACT[t] for t in ORDER]
    ax.set_xticks(np.arange(len(ORDER)))
    ax.set_xticklabels(labels, fontsize=5.0, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ORDER)))
    ax.set_yticklabels(labels, fontsize=5.0)
    for i, a in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            if a == b:
                continue
            if (a, b) in SHOWN:
                ax.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=SHOWN_C, lw=1.4)
                )
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=5.4,
                color="white" if mat[i, j] > 0.16 else INK,
            )
    if cax is not None:
        _mixing_colorbar(cax, im)
    ax.tick_params(colors=MUTED, labelsize=6.8)
    try:
        ax.set_box_aspect(1)
    except AttributeError:
        pass
    _bold_title(ax, "Who touches whom on the 3D embedding")
    _letter(ax, "C")


def _plot_action(ax, edges: pd.DataFrame, *, cax=None) -> None:
    mat = _matrix(edges, "graph_action")
    masked = np.ma.masked_invalid(mat)
    vmax = float(np.nanpercentile(mat[mat > 0], 90))
    im = ax.imshow(masked, cmap="YlGnBu_r", vmin=0, vmax=vmax, aspect="auto")
    labels = [COMPACT[t] for t in ORDER]
    ax.set_xticks(np.arange(len(ORDER)))
    ax.set_xticklabels(labels, fontsize=5.0, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ORDER)))
    ax.set_yticklabels(labels, fontsize=5.0)
    ax.set_xlabel("Target", fontsize=7.0, color=INK)
    ax.set_ylabel("Source", fontsize=7.0, color=INK)
    for i, a in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            if a == b:
                ax.text(j, i, "0", ha="center", va="center", fontsize=5.3, color=MUTED)
                continue
            if (a, b) in SHOWN:
                ax.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=SHOWN_C, lw=1.4)
                )
            ax.text(
                j,
                i,
                f"{mat[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=5.3,
                color="white" if mat[i, j] < 2.2 else INK,
            )
    if cax is not None:
        _colorbar(cax, im, r"UMAP+$U_{\mathrm{rel}}$ action")
    ax.yaxis.labelpad = Y_LABELPAD
    ax.xaxis.labelpad = X_LABELPAD
    ax.tick_params(colors=MUTED, labelsize=6.8)
    _bold_title(ax, "Geodesic cost on the same landscape")
    _letter(ax, "D")


def _plot_shown_scores(ax, edges: pd.DataFrame) -> None:
    labs, mix_v, act_v = [], [], []
    for src, dst in SHOWN:
        r = _edge_row(edges, src, dst)
        labs.append(_path_xtick(src, dst))
        mix_v.append(float(r["mixing"]))
        act_v.append(float(r["graph_action"]))
    x = np.arange(len(labs))
    w = 0.36
    ax.bar(x - w / 2, mix_v, w, color=SHOWN_C, edgecolor="white", label="kNN mixing", zorder=2)
    ax2 = ax.twinx()
    ax2.bar(x + w / 2, act_v, w, color="#7A93A8", edgecolor="white", label="graph action", zorder=2)
    for i, (m, a) in enumerate(zip(mix_v, act_v)):
        ax.text(i - w / 2, m + 0.008, f"{m:.2f}", ha="center", fontsize=5.8, color=INK)
        ax2.text(i + w / 2, a + 0.12, f"{a:.2f}", ha="center", fontsize=5.8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=5.2, rotation=0, ha="center")
    ax.tick_params(axis="x", pad=4)
    ax.set_ylabel("UMAP kNN mixing", fontsize=7.1, color=INK, labelpad=Y_LABELPAD)
    ax2.set_ylabel("Geodesic action", fontsize=7.1, color=INK, labelpad=Y_LABELPAD)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(colors=MUTED, labelsize=6.4)
    _bold_title(ax, "Drawn paths: local pair vs remodel climb")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=5.4, frameon=False, loc="upper left")
    _style_axis(ax)
    ax2.grid(False)
    _letter(ax, "E")


def _plot_occupancy(ax, edges: pd.DataFrame) -> None:
    labels = []
    src_f, dst_f, oth_f, notes = [], [], [], []
    for src, dst in SHOWN:
        r = _edge_row(edges, src, dst)
        labels.append(_path_xtick(src, dst))
        src_f.append(float(r["path_frac_src"]))
        dst_f.append(float(r["path_frac_dst"]))
        oth_f.append(float(r["path_frac_other"]))
        other = COMPACT.get(str(r["top_other"]), str(r["top_other"]))
        notes.append(f"{other} {float(r['top_other_frac']):.0%}")
    x = np.arange(len(labels))
    src_cols = [TYPE_COLORS[s] for s, _ in SHOWN]
    dst_cols = [TYPE_COLORS[d] for _, d in SHOWN]
    ax.bar(x, src_f, color=src_cols, edgecolor="white", width=0.58, label="source", zorder=2)
    ax.bar(x, dst_f, bottom=src_f, color=dst_cols, edgecolor="white", width=0.58, label="target", zorder=2)
    ax.bar(
        x,
        oth_f,
        bottom=np.array(src_f) + np.array(dst_f),
        color="#D0D5DC",
        edgecolor="white",
        width=0.58,
        label="other subtypes",
        zorder=2,
    )
    for i, note in enumerate(notes):
        ax.text(i, 1.02, note, ha="center", va="bottom", fontsize=5.4, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.2, rotation=0, ha="center")
    ax.tick_params(axis="x", pad=4)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Geodesic occupancy", fontsize=7.3, color=INK)
    _bold_title(ax, "What the drawn geodesics actually traverse")
    ax.legend(fontsize=5.4, frameon=False, loc="upper right", ncol=3)
    _style_axis(ax)
    _letter(ax, "F")


def _plot_alternatives(ax, edges: pd.DataFrame) -> None:
    rows = []
    for src, dst in ALTS:
        r = _edge_row(edges, src, dst)
        rows.append(
            (
                _path_xtick(src, dst, multiline=False),
                float(r["graph_action"]),
                str(r["call"]),
            )
        )
    labs = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [REJ_C if r[2] != "same_identity" else "#8A94A0" for r in rows]
    x = np.arange(len(labs))
    ax.bar(x, vals, color=cols, edgecolor="white", width=0.62, zorder=2)
    shown_mean = float(
        np.mean([float(_edge_row(edges, a, b)["graph_action"]) for a, b in SHOWN[:2]])
    )
    ax.axhline(shown_mean, color=SHOWN_C, lw=1.15, ls="--", zorder=5)
    ax.text(
        0.99,
        shown_mean + 0.08,
        "mean local shown action",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=5.3,
        color=SHOWN_C,
    )
    for i, (lab, v, call) in enumerate(rows):
        ax.text(i, v + 0.12, f"{v:.2f}", ha="center", fontsize=5.8, color=INK)
    ax.set_xticks(x)
    _set_rotated_xticklabels(ax, labs, rotation=45, fontsize=5.0, pad=6)
    ax.set_ylabel("Geodesic action (higher = not independent)", fontsize=7.1, color=INK)
    _bold_title(ax, "Alternatives not required as extra 3D arrows")
    _style_axis(ax)
    _letter(ax, "G")


def _plot_pair_scatter(ax, edges: pd.DataFrame) -> None:
    seen = set()
    for _, r in edges.iterrows():
        key = frozenset((r["src"], r["dst"]))
        if key in seen:
            continue
        seen.add(key)
        sub = edges[
            ((edges["src"] == r["src"]) & (edges["dst"] == r["dst"]))
            | ((edges["src"] == r["dst"]) & (edges["dst"] == r["src"]))
        ]
        x = float(sub["mixing"].mean())
        y = float(sub["graph_action"].mean())
        pair = (str(r["src"]), str(r["dst"]))
        rev = (pair[1], pair[0])
        is_shown = pair in SHOWN or rev in SHOWN
        c = SHOWN_C if is_shown else "#C5CDD4"
        ax.scatter([x], [y], s=78 if is_shown else 36, c=c, edgecolors="white", linewidths=0.7, zorder=3 if is_shown else 2)
        if is_shown or str(r["call"]) in {"redundant_shortcut", "fn1_satellite", "same_identity"}:
            lab = f"{COMPACT[pair[0]]}–{COMPACT[pair[1]]}"
            if rev in SHOWN:
                lab = f"{COMPACT[rev[0]]}–{COMPACT[rev[1]]}"
            if pair in SHOWN:
                lab = f"{COMPACT[pair[0]]}–{COMPACT[pair[1]]}"
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=5.5, color=INK)

    ax.axvline(0.02, color="#CBD2D9", lw=0.8, ls="--", zorder=1)
    ax.set_xlabel("Mean UMAP kNN mixing", fontsize=7.3, color=INK)
    ax.set_ylabel("Mean geodesic action", fontsize=7.3, color=INK)
    _bold_title(ax, "Local pairs sit at high mix / low action")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=SHOWN_C, markersize=7, label="Drawn 3D pair"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#C5CDD4", markersize=6, label="Other undirected pair"),
        ],
        fontsize=5.4,
        frameon=False,
        loc="upper right",
    )
    _style_axis(ax)
    _letter(ax, "H")


def _plot_topview(ax) -> None:
    if not TOPVIEW.is_file():
        raise FileNotFoundError(f"Missing macrophage top-view panel: {TOPVIEW}")
    img = load_rgb(TOPVIEW, trim=True)
    ax.imshow(img)
    ax.set_axis_off()
    _bold_title(ax, r"Curated arrows: unequal support, not lineage (UMAP top view)")
    _letter(ax, "I", x=-0.02, y=1.08)


def compose(out: Path | None = None) -> Path:
    out = Path(out or DEFAULT_OUT)
    data = _load()

    fig = plt.figure(figsize=(13.2, 11.6), facecolor="white")
    gs = GridSpec(
        3,
        3,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.12],
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.40,
        wspace=0.32,
        left=0.08,
        right=0.955,
        top=0.96,
        bottom=0.11,
    )

    gs_c = gs[0, 2].subgridspec(1, 2, width_ratios=[1, 0.065], wspace=0.12)
    gs_d = gs[1, 0].subgridspec(1, 2, width_ratios=[1, 0.10], wspace=0.08)

    axes = np.empty((3, 3), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0])
    axes[0, 1] = fig.add_subplot(gs[0, 1])
    axes[0, 2] = fig.add_subplot(gs_c[0, 0])
    cax_c = fig.add_subplot(gs_c[0, 1])
    axes[1, 0] = fig.add_subplot(gs_d[0, 0])
    cax_d = fig.add_subplot(gs_d[0, 1])
    axes[1, 1] = fig.add_subplot(gs[1, 1])
    axes[1, 2] = fig.add_subplot(gs[1, 2])
    axes[2, 0] = fig.add_subplot(gs[2, 0])
    axes[2, 1] = fig.add_subplot(gs[2, 1])
    axes[2, 2] = fig.add_subplot(gs[2, 2])

    _plot_time(axes[0, 0], data["time"])
    _plot_basin(axes[0, 1], data["basin"])
    _plot_mixing(axes[0, 2], data["mix"], cax=cax_c)

    _plot_action(axes[1, 0], data["edges"], cax=cax_d)
    _plot_shown_scores(axes[1, 1], data["edges"])
    _plot_occupancy(axes[1, 2], data["edges"])

    _plot_alternatives(axes[2, 0], data["edges"])
    _plot_pair_scatter(axes[2, 1], data["edges"])
    _plot_topview(axes[2, 2])

    _align_figure_panels(fig, axes)

    return save_fig(fig, out)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    sys.path.insert(0, str(ROOT / "scripts"))
    from analyze_mac_fn1_m2_resolution_triad import main as _triad
    from analyze_mac_landscape_path_endorsement import main as _endorsement

    _triad()
    _endorsement()
    compose(out=Path(argv[0]) if argv else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
