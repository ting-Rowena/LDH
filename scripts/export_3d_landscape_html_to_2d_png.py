#!/usr/bin/env python3
"""Export 2D (top-down) PNG projections of the interactive 3D U_rel HTML landscapes.

Uses the same field / wells / LAP paths as
``export_mac_alv_3d_landscape_interactive.py`` so HTML and PNG stay aligned.

Outputs (<CK>/analysis_protocol_GSE141259/figures/)
------------------------------------
  GSE141259_alv_3d_urel_landscape_interactive_2d.png
  GSE141259_mac_3d_urel_landscape_interactive_2d.png
  GSE141259_club_3d_urel_landscape_interactive_2d.png
  GSE141259_club_3d_urel_landscape_interactive_plus_AT2_ADI_2d.png
  GSE141259_mac_alv_3d_urel_landscape_interactive_2d.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import export_mac_alv_3d_landscape_interactive as E  # noqa: E402
import plot_mac_alv_3d_potential_landscape as L  # noqa: E402
from panel_style import apply_panel_title_rc, set_panel_title  # noqa: E402
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
PROTO = L.PROTO
PATH_COLOR = "#B42318"
PATH_HALO = "#FFFFFF"
WELL_EDGE = "#FFFFFF"
HALO = [pe.withStroke(linewidth=3.0, foreground="white", alpha=0.92)]
# Soft journal terrain colormap (cooler wells → warm ridges)
TERRAIN_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "urel_2d_soft",
    [
        "#0B3D5C",
        "#1F6A8A",
        "#5FA8C0",
        "#D9EAF0",
        "#F7F1DE",
        "#F0C987",
        "#E08A3D",
        "#C23B22",
        "#7A1F14",
    ],
)


def _prepare_z(field: dict, *, w_cut: float = 0.18, feather: float = 0.12):
    from scipy import ndimage

    xx, yy = field["xx"], field["yy"]
    Z = np.asarray(field["Z"], float).copy()
    w = np.asarray(field.get("weight", np.ones_like(Z)), float).copy()
    # Light blur so the silhouette is not pixel-jagged
    w_s = ndimage.gaussian_filter(np.nan_to_num(w, nan=0.0), sigma=1.1)
    Z_s = ndimage.gaussian_filter(np.nan_to_num(Z, nan=np.nanmedian(Z[np.isfinite(Z)])), sigma=0.7)
    # Feathered alpha mask
    alpha = np.clip((w_s - (w_cut - feather)) / max(feather, 1e-6), 0.0, 1.0)
    Z_show = np.where(alpha > 0.05, Z_s, np.nan)
    finite = np.isfinite(Z_show) & (alpha > 0.05)
    if not np.any(finite):
        Z_show = Z
        alpha = np.where(np.isfinite(Z), 1.0, 0.0)
        finite = np.isfinite(Z_show)
    lo, hi = np.nanpercentile(Z_show[finite], [4, 96])
    return xx, yy, Z_show, alpha, float(lo), float(hi)


def _place_labels(ax, wells: list[dict], paths: list[dict]):
    """Compact label offsets: close to wells, avoid mutual / colorbar / path collisions."""
    # Explicit side overrides (override default left-bias for right-flank wells).
    force_right_types = {"M2 macrophages", "Resolution macrophages"}
    force_right_labels = {"M2 macrophages", "Resolution macrophages", "M2", "Resolution"}
    force_left_types = {"Club cells"}
    force_left_labels = {"Club cells", "Club"}

    pts = []
    for w in wells:
        pos = np.asarray(w["xy"], float)
        if np.all(np.isfinite(pos)):
            pts.append(pos)
    pts = np.asarray(pts, float) if pts else np.zeros((0, 2))

    # Densify LAP polylines for label–path clearance scoring.
    path_pts = []
    for p in paths or []:
        xy = np.asarray(p.get("xy", []), float)
        if xy.ndim != 2 or len(xy) < 2 or not np.all(np.isfinite(xy)):
            continue
        # subsample to keep scoring cheap
        step = max(1, len(xy) // 40)
        path_pts.append(xy[::step])
    path_cloud = np.vstack(path_pts) if path_pts else np.zeros((0, 2))

    # Compact radial directions (unit-ish); distance is applied separately.
    dirs = np.array(
        [
            [1.0, 0.05],
            [0.85, 0.45],
            [0.85, -0.45],
            [-1.0, 0.05],
            [-0.85, 0.45],
            [-0.85, -0.45],
            [0.15, 1.0],
            [0.15, -1.0],
            [-0.15, 1.0],
            [-0.15, -1.0],
            [0.65, 0.75],
            [0.65, -0.75],
            [-0.65, 0.75],
            [-0.65, -0.75],
            # Extra NW / SW options for path-crowded left labels (e.g. Club)
            [-0.95, 0.30],
            [-0.75, 0.65],
            [-0.55, 0.85],
            [-0.95, -0.30],
            [-0.75, -0.65],
        ],
        float,
    )
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xspan = abs(x1 - x0)
    yspan = abs(y1 - y0)
    # Prefer short offsets; allow a slightly longer fallback if crowded.
    radii = np.array([0.022, 0.032, 0.042]) * max(xspan, yspan)
    radii_left = np.array([0.038, 0.052, 0.068]) * max(xspan, yspan)
    right_safe = x1 - 0.11 * xspan
    placed = []  # (xy, approx half-width in data units)

    for well in wells:
        pos = np.asarray(well["xy"], float)
        if not np.all(np.isfinite(pos)):
            continue
        near_right = pos[0] > (x0 + 0.55 * xspan)
        label = str(well["label"])
        ct = str(well.get("cell_type", ""))
        force_right = ct in force_right_types or label in force_right_labels
        force_left = (not force_right) and (
            ct in force_left_types or label in force_left_labels
        )
        # Approximate label width in data coords (tight, for collision only).
        half_w = 0.0045 * len(label) * xspan
        half_h = 0.012 * yspan
        use_radii = radii_left if force_left else radii

        best = None
        best_score = -np.inf
        for rad in use_radii:
            for d in dirs:
                if force_right and d[0] < 0.2:
                    continue
                if force_left and d[0] > -0.25:
                    continue
                cand = pos + rad * d
                # Nudge left if the text box would hit the colorbar strip
                # (skipped when user forces right-of-node placement).
                if (not force_right) and (near_right or (cand[0] + half_w > right_safe)):
                    # Prefer left-side placement for right-flank wells
                    if d[0] > 0.2:
                        continue
                    cand = pos + rad * d
                    if cand[0] + (half_w if d[0] >= 0 else 0) > right_safe:
                        cand = cand.copy()
                        cand[0] = right_safe - half_w - 0.005 * xspan

                # Keep inside frame (force-right: allow closer to colorbar edge)
                x_hi = (x1 - 0.02 * xspan) if force_right else (right_safe - 0.01 * xspan)
                cand = np.array(
                    [
                        float(np.clip(cand[0], x0 + 0.02 * xspan, x_hi)),
                        float(np.clip(cand[1], y0 + 0.025 * yspan, y1 - 0.025 * yspan)),
                    ]
                )

                dist = float(np.linalg.norm(cand - pos))
                # Reward closeness (primary aesthetic ask)
                score = 3.2 / (0.35 + dist / (0.01 * xspan + 1e-9))

                # Prefer left for right-side wells, mild right preference otherwise
                if force_right:
                    score += 1.5 * max(0.0, d[0])
                    if cand[0] < pos[0]:
                        score -= 8.0
                elif force_left:
                    score += 1.5 * max(0.0, -d[0])
                    # Mild upward bias only (avoid high NW that hits Club↔Ciliated)
                    score += 0.6 * max(0.0, d[1])
                    if d[1] > 0.55:
                        score -= 2.5
                    if d[1] < -0.35:
                        score -= 1.5
                    if cand[0] > pos[0]:
                        score -= 8.0
                elif near_right:
                    score += 0.8 * max(0.0, -d[0])
                else:
                    score += 0.25 * max(0.0, d[0])

                # Separation from other wells / labels
                if len(pts):
                    score += 0.15 * float(np.min(np.linalg.norm(pts - cand, axis=1)))
                for q, qw, qh in placed:
                    dx = abs(cand[0] - q[0]) / (half_w + qw + 1e-9)
                    dy = abs(cand[1] - q[1]) / (half_h + qh + 1e-9)
                    overlap = max(0.0, 1.15 - max(dx, dy))
                    score -= 6.0 * overlap

                # Keep clear of LAP polylines (esp. Club label vs Club↔AT2)
                if len(path_cloud):
                    mind = float(np.min(np.linalg.norm(path_cloud - cand, axis=1)))
                    # Also consider label box center-left extent for left-aligned text
                    probe = cand.copy()
                    if cand[0] < pos[0]:
                        probe[0] -= 0.35 * half_w
                    mind = min(
                        mind,
                        float(np.min(np.linalg.norm(path_cloud - probe, axis=1))),
                    )
                    clear = 0.028 * max(xspan, yspan)
                    if mind < clear:
                        score -= 12.0 * (1.0 - mind / clear)

                if (not force_right) and cand[0] > right_safe - 0.01 * xspan:
                    score -= 10.0

                # Mild preference for shorter radius among equal options
                score -= 0.15 * (rad / use_radii[-1])

                if score > best_score:
                    best_score = score
                    best = (cand, d)

        assert best is not None
        cand, d = best
        # Hard guarantee: label anchor stays on the requested side of the node
        if force_right and cand[0] < pos[0] + 0.008 * xspan:
            cand = cand.copy()
            cand[0] = pos[0] + max(radii[0], 0.018 * xspan)
            cand[0] = float(np.clip(cand[0], x0 + 0.02 * xspan, x1 - 0.02 * xspan))
        if force_left:
            # Upper-left park: left of node, slightly above center to clear
            # Club↔AT2 inbound paths without climbing into Club↔Ciliated.
            if cand[0] > pos[0] - 0.012 * xspan or abs(cand[1] - pos[1]) > 0.055 * yspan:
                cand = pos + np.array([-0.050 * xspan, 0.018 * yspan])
                cand = np.array(
                    [
                        float(np.clip(cand[0], x0 + 0.02 * xspan, x_hi)),
                        float(np.clip(cand[1], y0 + 0.025 * yspan, y1 - 0.025 * yspan)),
                    ]
                )
        placed.append((cand, half_w, half_h))

        ha = "left" if cand[0] >= pos[0] - 1e-9 else "right"
        dy = cand[1] - pos[1]
        if dy > 0.25 * use_radii[0]:
            va = "bottom"
        elif dy < -0.25 * use_radii[0]:
            va = "top"
        else:
            va = "center"

        ax.annotate(
            label,
            xy=pos,
            xytext=cand,
            textcoords="data",
            fontsize=7.0,
            fontweight="bold",
            color="#0F172A",
            ha=ha,
            va=va,
            path_effects=HALO,
            arrowprops=None,
            zorder=10,
            clip_on=True,
        )


def _draw_2d_projection(
    ax,
    field: dict,
    paths: list[dict],
    wells: list[dict],
    *,
    title: str,
    show_legend: bool = False,
    show_saddles: bool = False,
):
    xx, yy, Z_show, alpha, lo, hi = _prepare_z(field)

    # Soft paper background inside axes
    ax.set_facecolor("#FBFBFC")

    # Terrain with feathered edges via RGBA
    norm = Normalize(vmin=lo, vmax=hi)
    rgba = TERRAIN_CMAP(norm(np.nan_to_num(Z_show, nan=lo)))
    rgba[..., 3] = np.clip(alpha, 0, 1) * 0.96
    pcm = ax.imshow(
        rgba,
        origin="lower",
        extent=(float(xx.min()), float(xx.max()), float(yy.min()), float(yy.max())),
        interpolation="bilinear",
        aspect="auto",
        zorder=0,
    )
    # Keep a ScalarMappable for the colorbar
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=TERRAIN_CMAP)
    mappable.set_array([])

    # Sparse elegant contours (ink, not white glare)
    try:
        cs = ax.contour(
            xx,
            yy,
            np.ma.masked_invalid(Z_show),
            levels=8,
            colors="#1E293B",
            linewidths=0.35,
            alpha=0.18,
            zorder=1,
        )
        lines = getattr(cs, "collections", None)
        if lines is None:
            lines = getattr(cs, "allsegs", [])
        else:
            for i, c in enumerate(lines):
                c.set_linewidth(0.55 if i % 2 == 0 else 0.28)
                c.set_alpha(0.22 if i % 2 == 0 else 0.12)
    except Exception:
        pass

    # Soft outer silhouette
    try:
        ax.contour(
            xx,
            yy,
            alpha,
            levels=[0.35],
            colors="#64748B",
            linewidths=0.7,
            alpha=0.35,
            zorder=2,
        )
    except Exception:
        pass

    for p in paths:
        xy = np.asarray(p["path_xy"], float)
        if len(xy) < 2:
            continue
        # Smooth path visually with denser polyline already from FMM
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=PATH_HALO,
            lw=4.0,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=4,
            alpha=0.95,
        )
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=PATH_COLOR,
            lw=1.65,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=5,
            alpha=0.95,
        )
        # One refined mid-path arrow
        mid = int(0.55 * (len(xy) - 1))
        j0, j1 = max(0, mid - 4), min(len(xy) - 1, mid + 4)
        d = xy[j1] - xy[j0]
        nrm = np.linalg.norm(d)
        if nrm > 1e-8:
            d = d / nrm
            tip = xy[mid] + 0.28 * d
            ax.annotate(
                "",
                xy=tip,
                xytext=xy[mid] - 0.05 * d,
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=PATH_COLOR,
                    lw=1.35,
                    mutation_scale=11,
                ),
                zorder=6,
            )
        if show_saddles and p.get("has_barrier") and p.get("is_strict_saddle", True):
            ts = int(np.clip(p.get("ts", len(xy) // 2), 0, len(xy) - 1))
            ax.scatter(
                [xy[ts, 0]],
                [xy[ts, 1]],
                s=55,
                c="#F5D76E",
                marker="*",
                edgecolors="#1F2937",
                linewidths=0.35,
                zorder=7,
            )

    # Wells: soft glow ring + filled marker
    for well in wells:
        pos = np.asarray(well["xy"], float)
        if not np.all(np.isfinite(pos)):
            continue
        ax.scatter(
            [pos[0]],
            [pos[1]],
            s=110,
            c="white",
            marker="o",
            linewidths=0,
            zorder=8,
            alpha=0.9,
        )
        ax.scatter(
            [pos[0]],
            [pos[1]],
            s=48,
            c=well.get("color", "#0F172A"),
            marker="o",
            edgecolors=WELL_EDGE,
            linewidths=0.85,
            zorder=9,
        )

    x0, x1 = float(np.nanmin(xx)), float(np.nanmax(xx))
    y0, y1 = float(np.nanmin(yy)), float(np.nanmax(yy))
    pad_x = 0.03 * (x1 - x0)
    pad_y = 0.03 * (y1 - y0)
    ax.set_xlim(x0 - pad_x, x1 + pad_x)
    ax.set_ylim(y0 - pad_y, y1 + pad_y)
    ax.set_box_aspect((y1 - y0) / max(x1 - x0, 1e-6))

    _place_labels(ax, wells, paths)

    ax.set_xlabel("UMAP 1", fontsize=9, color="#334155")
    ax.set_ylabel("UMAP 2", fontsize=9, color="#334155")
    set_panel_title(ax, title)
    ax.tick_params(labelsize=7.5, colors="#64748B", length=2.5, width=0.6)
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.7)
    ax.grid(False)
    if show_legend and paths:
        ax.legend(loc="upper left", fontsize=6.5, frameon=False)
    return mappable, lo, hi


def _add_cbar(fig, ax, pcm, *, label: str = r"$U_{\mathrm{rel}}$", tight: bool = False):
    """Short thin colorbar flush against the right of the map."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # Keep colorbar farther right so subtype labels stay clear of it.
    x_off = 1.045 if tight else 1.035
    cax = inset_axes(
        ax,
        width="2.6%",
        height="66%",
        loc="lower left",
        bbox_to_anchor=(x_off, 0.17, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.ax.tick_params(labelsize=6.8, length=1.5, width=0.45, colors="#64748B", pad=0.5)
    cbar.outline.set_linewidth(0.4)
    cbar.outline.set_edgecolor("#94A3B8")
    cbar.set_label(label, fontsize=8.0, color="#334155", labelpad=5)
    try:
        cbar.locator = mpl.ticker.MaxNLocator(nbins=5)
        cbar.update_ticks()
    except Exception:
        pass
    return cbar


def _save_single_2d(
    field, paths, wells, *, title: str, out: Path, footnote: str, show_saddles: bool = False
):
    fig, ax = plt.subplots(figsize=(7.6, 5.9), facecolor="white")
    pcm, _lo, _hi = _draw_2d_projection(
        ax, field, paths, wells, title=title, show_saddles=show_saddles
    )
    _add_cbar(fig, ax, pcm)
    fig.subplots_adjust(left=0.10, right=0.88, top=0.90, bottom=0.12)
    fig.text(0.01, 0.008, footnote, fontsize=6.2, style="italic", color=MUTED, ha="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.14)
    fig.savefig(
        PROTO / "figures" / out.name,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.14,
    )
    plt.close(fig)
    print(f"Wrote 2D projection: {out}", flush=True)


def _save_overview_2d(alv, mac, *, out: Path):
    field_a, paths_a, wells_a, _ = alv
    field_m, paths_m, wells_m, _ = mac
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5), facecolor="white")
    pcm_a, _, _ = _draw_2d_projection(
        axes[0],
        field_a,
        paths_a,
        wells_a,
        title="Alveolar epithelium · 2D $U_{\\mathrm{rel}}$ projection",
        show_saddles=False,
    )
    pcm_m, _, _ = _draw_2d_projection(
        axes[1],
        field_m,
        paths_m,
        wells_m,
        title="Macrophages · 2D $U_{\\mathrm{rel}}$ projection",
        show_saddles=False,
    )
    for ax, pcm in ((axes[0], pcm_a), (axes[1], pcm_m)):
        _add_cbar(fig, ax, pcm, tight=True)
    fig.suptitle(
        "GSE141259 · 3D→2D potential landscape projections",
        fontsize=12,
        fontweight="bold",
        color=INK,
        y=1.02,
    )
    fig.text(
        0.01,
        0.008,
        "Top-down UMAP projection of interactive 3D HTML landscapes · red = LAP",
        fontsize=6.2,
        style="italic",
        color=MUTED,
        ha="left",
    )
    fig.subplots_adjust(wspace=0.36, left=0.05, right=0.96, top=0.90, bottom=0.10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    fig.savefig(
        PROTO / "figures" / out.name,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.12,
    )
    plt.close(fig)
    print(f"Wrote 2D projection: {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Export 2D PNG projections of 3D HTML landscapes.")
    ap.add_argument(
        "--auto-edges",
        action="store_true",
        default=True,
        help="Match HTML auto-edges for alv/mac (default: True).",
    )
    ap.add_argument(
        "--curated",
        action="store_true",
        help="Force curated edges for all panels (Club default remains curated).",
    )
    args = ap.parse_args()
    auto = False if args.curated else True

    print(f"===== Alv 2D projection (auto_edges={auto}) =====", flush=True)
    alv = E.build_alv_bundle(auto_edges=auto)
    _save_single_2d(
        *alv[:3],
        title="Alveolar epithelium · 2D projection of 3D $U_{\\mathrm{rel}}$ landscape",
        out=PANELS / "GSE141259_alv_3d_urel_landscape_interactive_2d.png",
        footnote="Projected from interactive 3D HTML · XY=UMAP · color=$U_{rel}$ · red=LAP",
        show_saddles=False,
    )

    print(f"===== Mac 2D projection (auto_edges={auto}) =====", flush=True)
    mac = E.build_mac_bundle(auto_edges=auto)
    _save_single_2d(
        *mac[:3],
        title="Macrophages · 2D projection of 3D $U_{\\mathrm{rel}}$ landscape",
        out=PANELS / "GSE141259_mac_3d_urel_landscape_interactive_2d.png",
        footnote="Projected from interactive 3D HTML · XY=UMAP · color=$U_{rel}$ · red=LAP",
        show_saddles=False,
    )

    print("===== Club 2D projection (curated) =====", flush=True)
    club = E.build_club_bundle(auto_edges=False)
    _save_single_2d(
        *club[:3],
        title="Club lineage · 2D projection of 3D $U_{\\mathrm{rel}}$ landscape",
        out=PANELS / "GSE141259_club_3d_urel_landscape_interactive_2d.png",
        footnote="Projected from interactive 3D HTML · MHC-II⁺ Club / classical paths · red=LAP",
        show_saddles=False,
    )

    print("===== Club + AT2→ADI 2D projection (curated) =====", flush=True)
    club_plus = E.build_club_bundle(
        auto_edges=False,
        extra_edges=[("AT2 cells", "Krt8 ADI")],
    )
    _save_single_2d(
        *club_plus[:3],
        title=(
            "Club lineage + AT2→ADI · 2D projection of 3D $U_{\\mathrm{rel}}$ landscape"
        ),
        out=PANELS / "GSE141259_club_3d_urel_landscape_interactive_plus_AT2_ADI_2d.png",
        footnote=(
            "Projected from interactive 3D HTML · Club paths + AT2→ADI · red=LAP"
        ),
        show_saddles=False,
    )

    print("===== Mac+Alv overview 2D projection =====", flush=True)
    _save_overview_2d(
        alv,
        mac,
        out=PANELS / "GSE141259_mac_alv_3d_urel_landscape_interactive_2d.png",
    )
    print("\n[Done] 2D PNG projections written next to the HTML panels.", flush=True)


if __name__ == "__main__":
    main()
