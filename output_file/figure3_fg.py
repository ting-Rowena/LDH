#!/usr/bin/env python3
"""Figure 3: Nav-triad (Scn9a/10a/11a) injury-relU heatmap + 1×3 facets, side-by-side.

Self-contained script consolidated from:
  - scripts/compose_fig2b_nav_injury_relU.py
  - panel_style.py / plot_utils.py (minimal helpers inlined)

Default output:
  output_file/figure3_fg.png
  (= archived panel Fig2B_Nav_injury_relU_heatmap_facets.png; manuscript Figure 3)

Usage:
  python output_file/figure3_fg.py
  python output_file/figure3_fg.py /path/to/out.png
  python output_file/figure3_fg.py --rebuild   # regenerate heatmap/facets from h5ad
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg", force=True)

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from scipy import sparse
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _adopted import CACHE, CK_PAIN  # noqa: E402
from _supp_compose import compose_side_by_side  # noqa: E402
from dataset_pipeline import GSE155622, resolve_data_path  # noqa: E402

CK = CK_PAIN
DEFAULT_OUT = Path(__file__).resolve().parent / "figure3_fg.png"
HEAT_PATH = CACHE / "figure3_fg_heatmap.png"
FACETS_PATH = CACHE / "figure3_fg_facets.png"
# Backup figure3_fg.png: leftover SNIIC-heatmap-only left slot + gap=40.
PUBLISHED_SIZE = (4375, 887)
PUBLISHED_GAP = 40
REF_LEFT = (1786, 1114)
LEFT_WIDTH = int(round(REF_LEFT[0] * PUBLISHED_SIZE[1] / REF_LEFT[1]))  # 1421

GENES = ["Scn9a", "Scn10a", "Scn11a"]  # classical Nav triad only (no Trpv1)

# ---------------------------------------------------------------------------
# Style (inlined from panel_style / plot_utils)
# ---------------------------------------------------------------------------
PANEL_TITLE_SIZE = 10
PANEL_TITLE_WEIGHT = "bold"
PANEL_TITLE_LOC = "center"
PANEL_TITLE_PAD = 4
AXIS_LABEL_SIZE = 9
TICK_LABEL_SIZE = 8.5
ANNOT_SIZE = 7

INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e3e8ee"
PANEL_BG = "#ffffff"

HEAT_CMAP = LinearSegmentedColormap.from_list(
    "journal_zscore",
    [
        "#3E6F9F",
        "#7FA8C9",
        "#C9DCEB",
        "#F7F7F7",
        "#F2D0C6",
        "#D88B7C",
        "#B04A42",
    ],
)


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "font.size": TICK_LABEL_SIZE,
            "axes.titlesize": PANEL_TITLE_SIZE,
            "axes.titleweight": PANEL_TITLE_WEIGHT,
            "axes.titlelocation": PANEL_TITLE_LOC,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.labelcolor": INK,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.9,
            "text.color": INK,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": PANEL_BG,
            "axes.axisbelow": True,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
        }
    )


def _style_axis(ax, *, grid_axis: str = "y") -> None:
    ax.set_facecolor(PANEL_BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=INK, length=3.5, width=0.9)
    if grid_axis == "none":
        ax.grid(False)
    else:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, alpha=1.0)


def _set_panel_title(ax, title: str, **kwargs) -> None:
    kw = {
        "loc": PANEL_TITLE_LOC,
        "fontweight": PANEL_TITLE_WEIGHT,
        "fontsize": PANEL_TITLE_SIZE,
        "pad": PANEL_TITLE_PAD,
    }
    kw.update(kwargs)
    ax.set_title(title, **kw)


def _resolve_genes(var_names, aliases: list[str]) -> list[str]:
    name_set = {str(g): str(g) for g in var_names}
    lower = {str(g).lower(): str(g) for g in var_names}
    out = []
    for a in aliases:
        if a in name_set:
            out.append(name_set[a])
        elif a.lower() in lower:
            out.append(lower[a.lower()])
    return list(dict.fromkeys(out))


def _set_nav_heatmap_title(ax) -> None:
    """Title with bold-italic Naᵥ (mathtext \\boldsymbol is not visually bold here)."""
    fp_nav = FontProperties(family="DejaVu Sans", style="italic", weight="bold", size=PANEL_TITLE_SIZE)
    fp_rest = FontProperties(family="DejaVu Sans", style="normal", weight="bold", size=PANEL_TITLE_SIZE)
    pack = HPacker(
        children=[
            TextArea("Na\u1d65", textprops=dict(fontproperties=fp_nav, color="black")),
            TextArea(" genes along injury axis", textprops=dict(fontproperties=fp_rest, color="black")),
        ],
        align="baseline",
        pad=0,
        sep=0,
    )
    ax.set_title(" ", pad=PANEL_TITLE_PAD)
    box = AnchoredOffsetbox(
        loc="lower center",
        child=pack,
        pad=0.0,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.45,
    )
    ax.add_artist(box)


def _load() -> tuple[np.ndarray, np.ndarray]:
    obs = pd.read_csv(CK / "obs.csv", low_memory=False)
    if "barcode" in obs.columns:
        obs.index = obs["barcode"].astype(str)
    elif "Unnamed: 0" in obs.columns:
        obs.index = obs["Unnamed: 0"].astype(str)
    neu = obs[obs["annotation"].astype(str) == "Neuron"].copy()
    print("[figure3_fg] loading neuron channel genes...", flush=True)
    raw = ad.read_h5ad(resolve_data_path(GSE155622), backed="r")
    common = neu.index.intersection(raw.obs_names.astype(str))
    _resolve_genes(raw.var_names, GENES)
    lower = {str(g).lower(): str(g) for g in raw.var_names}
    use = []
    for g in GENES:
        if g in raw.var_names:
            use.append(g)
        elif g.lower() in lower:
            use.append(lower[g.lower()])
        else:
            raise KeyError(g)
    parts = [raw[common, g].to_memory() for g in use]
    raw.file.close()
    X = np.hstack(
        [
            (p.X.toarray() if sparse.issparse(p.X) else np.asarray(p.X)).reshape(-1, 1)
            for p in parts
        ]
    )
    X = np.log1p(X.astype(float))
    rel = neu.loc[common, "potential_relative_type"].astype(float).to_numpy()
    return X, rel


def regenerate_panels() -> tuple[Path, Path]:
    """Rebuild heatmap + facets PNGs under output_file/_cache/."""
    _apply_style()
    X, rel = _load()
    injury = -rel
    order_i = np.argsort(injury)
    x_i = np.linspace(0, 1, len(order_i))
    w = max(20, len(order_i) // 40)
    ker = np.ones(w) / w

    # --- Heatmap ---
    n_bins = 200
    Xz = np.column_stack(
        [(X[:, i] - np.nanmean(X[:, i])) / (np.nanstd(X[:, i]) + 1e-12) for i in range(len(GENES))]
    ).T
    Xz = Xz[:, order_i]
    edges = np.linspace(0, Xz.shape[1], n_bins + 1).astype(int)
    mat = np.zeros((len(GENES), n_bins))
    for b in range(n_bins):
        sl = slice(edges[b], edges[b + 1])
        mat[:, b] = np.nanmean(Xz[:, sl], axis=1) if edges[b + 1] > edges[b] else np.nan
    vmax = float(max(1.0, np.nanpercentile(np.abs(mat), 98)))
    vmax = float(np.ceil(vmax * 10.0) / 10.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig_h, ax_h = plt.subplots(figsize=(6.6, 3.35))
    im = ax_h.imshow(mat, aspect="auto", cmap=HEAT_CMAP, norm=norm, interpolation="nearest")
    ax_h.set_yticks(np.arange(len(GENES)))
    ax_h.set_yticklabels([rf"$\mathit{{{g}}}$" for g in GENES])
    ax_h.set_xticks([0, n_bins // 2, n_bins - 1])
    ax_h.set_xticklabels(
        [
            r"normal" + "\n" + r"(high $U_{\mathrm{rel}}$)",
            "→",
            r"injury" + "\n" + r"(low $U_{\mathrm{rel}}$)",
        ]
    )
    ax_h.set_xlabel(r"Neurons ordered by $-U_{\mathrm{rel}}$")
    _set_nav_heatmap_title(ax_h)
    cbar = fig_h.colorbar(im, ax=ax_h, fraction=0.035, pad=0.02)
    cbar.set_label("Row z-score")
    cbar.set_ticks([-vmax, 0.0, vmax])
    cbar.set_ticklabels([f"{-vmax:g}", "0", f"{vmax:g}"])
    _style_axis(ax_h, grid_axis="none")
    fig_h.tight_layout()
    HEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig_h.savefig(HEAT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_h)
    print(f"[ok] {HEAT_PATH.name}", flush=True)

    # --- Facets 1×3 ---
    curve_c = "#0B559F"
    fig_f, axes = plt.subplots(1, 3, figsize=(10.2, 3.15), sharex=True, sharey=True)
    for ax, g in zip(axes, GENES):
        y = X[:, GENES.index(g)]
        yz = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-12)
        yi = yz[order_i]
        step = max(1, len(yi) // 800)
        ax.scatter(
            x_i[::step],
            yi[::step],
            s=4,
            alpha=0.12,
            color=curve_c,
            rasterized=True,
            linewidths=0,
        )
        ax.plot(x_i, np.convolve(yi, ker, mode="same"), color=curve_c, lw=2.4)
        ax.axhline(0, color="0.6", ls="--", lw=0.8)
        rho = spearmanr(injury, yz, nan_policy="omit").correlation
        _set_panel_title(ax, g, fontstyle="italic")
        ax.text(
            0.97,
            0.95,
            rf"$\rho$(injury)$=${rho:+.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=ANNOT_SIZE,
            color="0.25",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.85", alpha=0.9),
        )
        ax.set_xlabel(r"Neurons ordered by $-U_{\mathrm{rel}}$")
        _style_axis(ax, grid_axis="y")
    axes[0].set_ylabel("Expression z-score (smoothed)")
    fig_f.tight_layout()
    fig_f.savefig(FACETS_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_f)
    print(f"[ok] {FACETS_PATH.name}", flush=True)
    return HEAT_PATH, FACETS_PATH


def compose(out: Path | None = None) -> Path:
    """Compute heatmap + facets from h5ad, then place them on the published canvas."""
    if out is None:
        out = DEFAULT_OUT
    out = Path(out)
    regenerate_panels()
    return compose_side_by_side(
        HEAT_PATH,
        FACETS_PATH,
        out,
        target_size=PUBLISHED_SIZE,
        gap=PUBLISHED_GAP,
        left_width=LEFT_WIDTH,
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = [a for a in argv if a != "--rebuild"]
    out = Path(argv[0]) if argv else DEFAULT_OUT
    compose(out=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
