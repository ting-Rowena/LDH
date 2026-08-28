#!/usr/bin/env python3
"""Interactive rotatable 3D U_rel landscapes (Plotly HTML) — journal style.

Fixes / design
--------------
1. Robust UMAP bounds via percentile cropping in ``_build_field`` (shared).
2. Spatial median wells (not low-U cores) so attractors stay on the manifold.
3. Unified solid LAP trajectories with 3D direction chevrons (Fig. 5F style).
4. Optional floor path shadows disabled for cleaner composition.

Edge modes
----------
- default (curated): literature-aligned candidate transitions
- ``--auto-edges``: data-driven edges via ``L._infer_data_driven_edges``

Outputs
-------
  <CK>/analysis_protocol_GSE141259/figures/GSE141259_alv_3d_urel_landscape_interactive.html
  <CK>/analysis_protocol_GSE141259/figures/GSE141259_mac_3d_urel_landscape_interactive.html
  <CK>/analysis_protocol_GSE141259/figures/GSE141259_mac_alv_3d_urel_landscape_interactive.html
  <CK>/analysis_protocol_GSE141259/figures/GSE141259_club_3d_urel_landscape_interactive.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import plot_mac_alv_3d_potential_landscape as L  # noqa: E402
from analyze_mac_alv_dynamics_first_paths import (  # noqa: E402
    ALV_TYPES,
    MAC_TYPES,
    SUB_COL,
    _wrap_label,
)

PANELS = L.PANELS
PROTO = L.PROTO
for path in (PANELS, PROTO / "figures"):
    path.mkdir(parents=True, exist_ok=True)

PATH_COLOR = "#C1121F"
HALO_COLOR = "rgba(255, 255, 255, 0.95)"
SADDLE_COLOR = "#F4D35E"
FONT_FAMILY = "Arial, Helvetica, sans-serif"

_EDGE_SHORT = {
    "AT2 cells": "AT2",
    "Activated AT2 cells": "Act.AT2",
    "Krt8 ADI": "ADI",
    "AT1 cells": "AT1",
    "AM (PBS)": "AM (PBS)",
    "AM (Bleo)": "AM (Bleo)",
    "M2 macrophages": "M2",
    "Resolution macrophages": "Resolution",
    "Fn1+ macrophages": "Fn1+",
    "Cd163-/Cd11c+ IMs": "IM−",
    "Cd163+/Cd11c- IMs": "IM+",
    "Club cells": "Club",
    "MHC-II+ Club cells": "MHC-II⁺ Club",
    "Ciliated cells": "Ciliated",
    "Goblet cells": "Goblet",
    "D0": "D0",
    "D28": "D28",
}

# Fallback colors for lineage labels not in hierarchical palette.
_CLUB_EXTRA_COL = {
    "MHC-II+ Club cells": "#C9A227",
    "Club cells": "#319B3F",
    "Ciliated cells": "#5E56BD",
    "Goblet cells": "#BA9E97",
    "AT2 cells": "#2A6A9A",
    "Krt8 ADI": "#1A4A78",
    "AT1 cells": "#7AA8C4",
}

SURF_SCALE = [
    [0.00, "#08306B"],
    [0.15, "#2171B5"],
    [0.30, "#6BAED6"],
    [0.45, "#C6DBEF"],
    [0.55, "#FFF7BC"],
    [0.70, "#FEE391"],
    [0.82, "#FE9929"],
    [0.92, "#EC7014"],
    [1.00, "#8C2D04"],
]


def _formal_display(name: str) -> str:
    return _wrap_label(str(name)).replace("\n", " ")


def _edge_legend(src: str, dst: str) -> str:
    return f"{_EDGE_SHORT.get(src, src)} → {_EDGE_SHORT.get(dst, dst)}"


def _spatial_medoid_centroid(xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Coordinate-wise spatial median of a subtype cloud (robust manifold center)."""
    ix = np.where(mask)[0]
    if ix.size == 0:
        return np.full(2, np.nan)
    return np.median(xy[ix], axis=0)


def _subtype_wells(wells_xy: dict, types: list[str]) -> list[dict]:
    wells = []
    for t in types:
        if t not in wells_xy or not np.all(np.isfinite(wells_xy[t])):
            continue
        wells.append(
            {
                "xy": wells_xy[t],
                "label": _formal_display(t),
                "color": SUB_COL.get(t, _CLUB_EXTRA_COL.get(t, "#2B3A42")),
                "cell_type": t,
            }
        )
    return wells


def _prepare_surface(field: dict):
    xx, yy = field["xx"], field["yy"]
    Z = np.asarray(field["Z"], float)
    w = np.asarray(field["weight"], float)
    Z_show = np.where(w >= 0.15, Z, np.nan)
    finite = np.isfinite(Z_show)
    if not np.any(finite):
        finite = np.isfinite(Z)
        Z_show = Z
    lo, hi = np.nanpercentile(Z_show[finite], [3, 97])
    z_floor = float(np.nanmin(Z_show[finite])) - 0.28 * (hi - lo)
    return xx, yy, Z_show, float(lo), float(hi), z_floor


def _midpath_chevron_xyz(
    path_xy: np.ndarray,
    U_path: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    lo: float,
    hi: float,
    *,
    frac: float = 0.55,
) -> np.ndarray | None:
    """3D chevron arrow points (left -> tip -> right) along the trajectory."""
    xy = np.asarray(path_xy, float)
    Uz = np.asarray(U_path, float)
    n = len(xy)
    if n < 8:
        return None
    xyz = np.c_[xy, Uz]
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    if cum[-1] <= 0:
        return None
    idx = int(np.clip(np.searchsorted(cum, frac * cum[-1]), 2, n - 3))
    mid = xyz[idx]

    xspan = max(float(np.nanmax(xx) - np.nanmin(xx)), 1e-6)
    yspan = max(float(np.nanmax(yy) - np.nanmin(yy)), 1e-6)
    zspan = max(float(hi - lo), 1e-6)
    scale = np.array([xspan, yspan, zspan], float)

    j0, j1 = max(0, idx - 3), min(n - 1, idx + 3)
    tang = (xyz[j1] - xyz[j0]) / scale
    if not np.any(np.isfinite(tang)) or np.linalg.norm(tang) < 1e-8:
        return None
    tang = tang / np.linalg.norm(tang)

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(tang, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    side = np.cross(tang, ref)
    side /= np.linalg.norm(side) + 1e-12

    # Compact, acute chevron that remains readable without dominating short paths.
    tip_len, wing = 0.006, 0.003
    mid_n = mid / scale
    tip = (mid_n + tang * tip_len) * scale
    left = (mid_n - tang * 0.45 * tip_len + side * wing) * scale
    right = (mid_n - tang * 0.45 * tip_len - side * wing) * scale
    return np.vstack([left, tip, right])


def _global_path_saddle(paths: list[dict]) -> dict | None:
    best = None
    best_U = -np.inf
    for p in paths:
        if not p.get("has_barrier", False):
            continue
        # Prefer strict 2D saddles when the metric is available
        if p.get("is_strict_saddle") is False:
            continue
        xy = np.asarray(p["path_xy"], float)
        Uz = np.asarray(p["U_path"], float)
        if len(Uz) < 3:
            continue
        ts = int(p["ts"]) if "ts" in p else int(1 + np.argmax(Uz[1:-1]))
        ts = int(np.clip(ts, 1, len(Uz) - 2))
        U_ts = float(Uz[ts])
        if U_ts > best_U:
            best_U = U_ts
            best = {
                "xy": xy[ts].copy(),
                "U": U_ts,
                "method": "path_ts_max_U",
                "path_label": p.get("label", ""),
                "ts_idx": ts,
                "barrier_height": p.get("barrier_height", 0.0),
            }
    return best


def _attach_nearest_subtype(saddle: dict | None, wells: list[dict]) -> dict | None:
    if saddle is None or not wells:
        return saddle
    pos = np.asarray(saddle["xy"], float)
    d = [np.linalg.norm(pos - np.asarray(w["xy"], float)) for w in wells]
    k = int(np.argmin(d))
    saddle = dict(saddle)
    saddle["nearest_subtype"] = wells[k]["label"]
    saddle["nearest_cell_type"] = wells[k].get("cell_type", wells[k]["label"])
    return saddle


def _add_landscape_traces(
    fig: go.Figure,
    field: dict,
    paths: list[dict],
    wells: list[dict],
    *,
    global_saddle: dict | None = None,
    row: int | None = None,
    col: int | None = None,
    showlegend: bool = True,
):
    xx, yy, Z_show, lo, hi, _z_floor = _prepare_surface(field)
    kw = {}
    if row is not None and col is not None:
        kw = {"row": row, "col": col}

    fig.add_trace(
        go.Surface(
            x=xx[0],
            y=yy[:, 0],
            z=Z_show,
            colorscale=SURF_SCALE,
            cmin=lo,
            cmax=hi,
            opacity=0.90,
            showscale=showlegend,
            colorbar=dict(
                title=dict(
                    text="Potential<br><i>U</i><sub>rel</sub>",
                    font=dict(family=FONT_FAMILY, size=11),
                ),
                len=0.45,
                thickness=12,
                tickfont=dict(family=FONT_FAMILY, size=9),
                outlinewidth=0.5,
                outlinecolor="#D0D5DD",
            ),
            name="Potential Surface",
            lighting=dict(
                ambient=0.55,
                diffuse=0.90,
                specular=0.25,
                roughness=0.50,
                fresnel=0.12,
            ),
            lightposition=dict(x=100, y=200, z=600),
            hovertemplate=(
                "<b>UMAP1</b>: %{x:.2f}<br><b>UMAP2</b>: %{y:.2f}"
                "<br><b>U_rel</b>: %{z:.3f}<extra></extra>"
            ),
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=False,
                    highlightcolor="rgba(255,255,255,0.35)",
                    project_z=False,
                    color="rgba(255,255,255,0.22)",
                    width=1,
                    start=lo,
                    end=hi,
                    size=(hi - lo) / 18.0 if hi > lo else 0.1,
                )
            ),
        ),
        **kw,
    )

    for i, p in enumerate(paths):
        xy = np.asarray(p["path_xy"], float)
        Uz = np.asarray(p["U_path"], float)
        label = p.get("label", f"Path {i + 1}")
        # Skip corrupted geodesics that would explode the scene camera.
        if (
            xy.ndim != 2
            or xy.shape[0] < 2
            or not np.all(np.isfinite(xy))
            or not np.all(np.isfinite(Uz))
            or float(np.nanmax(np.abs(xy))) > 1e3
        ):
            print(f"  skip corrupt LAP (out-of-range): {label}", flush=True)
            continue

        # Halo underlay
        fig.add_trace(
            go.Scatter3d(
                x=xy[:, 0],
                y=xy[:, 1],
                z=Uz,
                mode="lines",
                line=dict(color=HALO_COLOR, width=7),
                showlegend=False,
                hoverinfo="skip",
            ),
            **kw,
        )

        # Main solid trajectory
        fig.add_trace(
            go.Scatter3d(
                x=xy[:, 0],
                y=xy[:, 1],
                z=Uz,
                mode="lines",
                line=dict(color=PATH_COLOR, width=4),
                name=f"LAP: {label}",
                legendgroup="paths",
                showlegend=showlegend,
                hovertemplate=f"<b>{label}</b><br>U: %{{z:.3f}}<extra></extra>",
            ),
            **kw,
        )

        # One compact direction chevron at 55% arc length (Fig. 5F style).
        head = _midpath_chevron_xyz(xy, Uz, xx, yy, lo, hi, frac=0.55)
        if head is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=head[:, 0],
                    y=head[:, 1],
                    z=head[:, 2],
                    mode="lines",
                    line=dict(color="rgba(255, 255, 255, 0.95)", width=5.5),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                **kw,
            )
            fig.add_trace(
                go.Scatter3d(
                    x=head[:, 0],
                    y=head[:, 1],
                    z=head[:, 2],
                    mode="lines",
                    line=dict(color=PATH_COLOR, width=4.0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                **kw,
            )

    if global_saddle is not None:
        sx, sy = float(global_saddle["xy"][0]), float(global_saddle["xy"][1])
        su = float(global_saddle["U"])
        plab = global_saddle.get("path_label", "")
        near = global_saddle.get("nearest_subtype", "")
        hover = (
            f"<b>Transition State (Saddle)</b><br>"
            f"Potential <i>U</i>: {su:.3f}<br>"
            f"Barrier Δ<i>U</i>: {global_saddle.get('barrier_height', 0):.3f}<br>"
            f"Path: {plab}<br>Associated State: {near}"
        )
        fig.add_trace(
            go.Scatter3d(
                x=[sx],
                y=[sy],
                z=[su],
                mode="markers",
                marker=dict(
                    size=9,
                    color=SADDLE_COLOR,
                    symbol="diamond",
                    line=dict(width=1.8, color="#1E293B"),
                ),
                name="Saddle State",
                showlegend=showlegend,
                hovertemplate=hover + "<extra></extra>",
            ),
            **kw,
        )

    for well in wells:
        pos = np.asarray(well["xy"], float)
        u = float(field["U_func"](pos))
        fig.add_trace(
            go.Scatter3d(
                x=[pos[0]],
                y=[pos[1]],
                z=[u],
                mode="markers+text",
                marker=dict(
                    size=6.5,
                    color=well.get("color", "#08306B"),
                    line=dict(width=1.2, color="#FFFFFF"),
                ),
                text=[well["label"]],
                textposition="top center",
                textfont=dict(
                    family=FONT_FAMILY, size=9, color="#1E293B", weight="bold"
                ),
                name=well["label"],
                showlegend=False,
                hovertemplate=(
                    f"<b>{well['label']}</b><br>"
                    "Potential <i>U</i>: %{z:.3f}<extra></extra>"
                ),
            ),
            **kw,
        )

    return lo, hi


def _scene_layout(*, camera_eye=None):
    eye = camera_eye or dict(x=1.35, y=-1.45, z=0.85)
    return dict(
        xaxis=dict(
            title=dict(
                text="UMAP 1",
                font=dict(family=FONT_FAMILY, size=10, color="#475467"),
            ),
            tickfont=dict(family=FONT_FAMILY, size=8.5, color="#64748B"),
            gridcolor="#E2E8F0",
            showbackground=False,
            zerolinecolor="#CBD5E1",
        ),
        yaxis=dict(
            title=dict(
                text="UMAP 2",
                font=dict(family=FONT_FAMILY, size=10, color="#475467"),
            ),
            tickfont=dict(family=FONT_FAMILY, size=8.5, color="#64748B"),
            gridcolor="#E2E8F0",
            showbackground=False,
            zerolinecolor="#CBD5E1",
        ),
        zaxis=dict(
            title=dict(
                text="Potential U_rel",
                font=dict(family=FONT_FAMILY, size=10, color="#475467"),
            ),
            tickfont=dict(family=FONT_FAMILY, size=8.5, color="#64748B"),
            gridcolor="#E2E8F0",
            showbackground=False,
            zerolinecolor="#CBD5E1",
        ),
        aspectmode="manual",
        # Emphasize U_rel relief so the landscape is not visually flattened.
        aspectratio=dict(x=1.15, y=1.0, z=0.88),
        camera=dict(eye=eye, up=dict(x=0, y=0, z=1)),
        bgcolor="white",
    )


def _write_html(fig: go.Figure, out: Path):
    cfg = {
        "displayModeBar": True,
        "displaylogo": False,
        "scrollZoom": True,
        "responsive": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": out.stem,
            "height": 950,
            "width": 1300,
            "scale": 2.5,
        },
    }
    fig.write_html(out, include_plotlyjs="cdn", full_html=True, config=cfg)
    fig.write_html(
        PROTO / "figures" / out.name,
        include_plotlyjs="cdn",
        full_html=True,
        config=cfg,
    )
    print(f"Wrote publication-grade HTML: {out}", flush=True)


def build_alv_bundle(*, auto_edges: bool = False):
    adata = L._load_parent("alv_epithelium", ALV_TYPES)
    field = L._build_field(adata, n_grid=120, max_fit=None, smooth_sigma=3.8)
    print(f"  robust bbox={field.get('bbox')}", flush=True)
    xy = field["xy"]
    labels = adata.obs["cell.type"].astype(str).to_numpy()
    wells_xy = {
        t: _spatial_medoid_centroid(xy, labels == t)
        for t in ALV_TYPES
        if (labels == t).sum() >= 5
    }
    wells = _subtype_wells(wells_xy, ALV_TYPES)
    if auto_edges:
        print("  edge mode: auto (pseudotime + kNN mixing)", flush=True)
        edges = [
            (src, dst)
            for src, dst, _lab, _ts in L._infer_data_driven_edges(adata, field)
        ]
    else:
        print("  edge mode: curated prior", flush=True)
        edges = [
            ("AT2 cells", "AT1 cells"),
            ("Activated AT2 cells", "AT1 cells"),
            ("AT2 cells", "Krt8 ADI"),
            ("Krt8 ADI", "AT1 cells"),
        ]
    paths = []
    for src, dst in edges:
        if src not in wells_xy or dst not in wells_xy:
            continue
        if not np.all(np.isfinite(wells_xy[src])) or not np.all(np.isfinite(wells_xy[dst])):
            continue
        p = L._resolve_path(field, wells_xy[src], wells_xy[dst], try_flow=False)
        p["label"] = _edge_legend(src, dst)
        paths.append(p)
        print(
            f"  {p['label']}: has_barrier={p.get('has_barrier')} "
            f"ΔU={p.get('barrier_height', float('nan')):.3f}",
            flush=True,
        )
    saddle = _attach_nearest_subtype(_global_path_saddle(paths), wells)
    return field, paths, wells, saddle


def build_mac_bundle(*, auto_edges: bool = False):
    adata = L._load_parent("macrophages", MAC_TYPES)
    field = L._build_field(adata, n_grid=110, max_fit=None, smooth_sigma=4.0)
    print(f"  robust bbox={field.get('bbox')}", flush=True)
    xy = field["xy"]
    labels = adata.obs["cell.type"].astype(str).to_numpy()
    stages = adata.obs["stage"].astype(str).to_numpy()
    wells_xy = {
        t: _spatial_medoid_centroid(xy, labels == t)
        for t in MAC_TYPES
        if (labels == t).sum() >= 8
    }
    wells_xy["D0"] = _spatial_medoid_centroid(xy, stages == "D0")
    wells_xy["D28"] = _spatial_medoid_centroid(xy, stages == "D28")
    wells = _subtype_wells(wells_xy, MAC_TYPES)
    if auto_edges:
        print("  edge mode: auto (pseudotime + kNN mixing)", flush=True)
        edges = [
            (src, dst)
            for src, dst, _lab, _ts in L._infer_data_driven_edges(adata, field)
        ]
    else:
        print("  edge mode: curated prior", flush=True)
        edges = [
            ("AM (Bleo)", "M2 macrophages"),
            ("M2 macrophages", "Resolution macrophages"),
            ("AM (PBS)", "Resolution macrophages"),
        ]
    paths = []
    for src, dst in edges:
        if src not in wells_xy or dst not in wells_xy:
            continue
        if not np.all(np.isfinite(wells_xy[src])) or not np.all(np.isfinite(wells_xy[dst])):
            continue
        p = L._resolve_path(field, wells_xy[src], wells_xy[dst], try_flow=False)
        p["label"] = _edge_legend(src, dst)
        paths.append(p)
        print(
            f"  {p['label']}: has_barrier={p.get('has_barrier')} "
            f"ΔU={p.get('barrier_height', float('nan')):.3f}",
            flush=True,
        )
    saddle = _attach_nearest_subtype(_global_path_saddle(paths), wells)
    return field, paths, wells, saddle


def build_club_bundle(*, auto_edges: bool = False, extra_edges: list[tuple[str, str]] | None = None):
    """Club regenerative + classical paths on a shared U_rel surface.

    Parameters
    ----------
    extra_edges :
        Optional additional curated edges (e.g. alveolar AT2→ADI) drawn on the
        same Club-panel surface without changing the base Club HTML.
    """
    adata = L.load_club_lineage_adata()
    field = L._build_field(adata, n_grid=120, max_fit=None, smooth_sigma=3.8)
    print(f"  robust bbox={field.get('bbox')}", flush=True)
    xy = field["xy"]
    labels = adata.obs["cell.type"].astype(str).to_numpy()
    types = [t for t in L.CLUB_LINEAGE_TYPES if (labels == t).sum() >= 5]
    wells_xy = {
        t: _spatial_medoid_centroid(xy, labels == t)
        for t in types
    }
    wells = _subtype_wells(wells_xy, types)

    if auto_edges:
        print("  edge mode: auto (pseudotime + kNN mixing)", flush=True)
        edges = [
            (src, dst)
            for src, dst, _lab, _ts in L._infer_data_driven_edges(
                adata,
                field,
                terminal_sinks={"AT1 cells", "Ciliated cells", "Goblet cells"},
            )
        ]
    else:
        print("  edge mode: curated Club biology prior", flush=True)
        # Regenerative (injury): MHC-II+ Club → ADI → AT1; Club → AT2
        # Classical airway: Club → Ciliated / Goblet
        edges = [
            ("MHC-II+ Club cells", "Krt8 ADI"),
            ("Krt8 ADI", "AT1 cells"),
            ("Club cells", "AT2 cells"),
            ("Club cells", "Ciliated cells"),
            ("Club cells", "Goblet cells"),
            ("MHC-II+ Club cells", "AT2 cells"),
        ]

    if extra_edges:
        print(f"  + extra edges: {extra_edges}", flush=True)
        # Preserve order; skip duplicates
        seen = set(edges)
        for e in extra_edges:
            if e not in seen:
                edges.append(e)
                seen.add(e)

    paths = []
    for src, dst in edges:
        if src not in wells_xy or dst not in wells_xy:
            print(f"  skip missing well: {src}→{dst}", flush=True)
            continue
        if not np.all(np.isfinite(wells_xy[src])) or not np.all(np.isfinite(wells_xy[dst])):
            continue
        p = L._resolve_path(field, wells_xy[src], wells_xy[dst], try_flow=False)
        p["label"] = _edge_legend(src, dst)
        paths.append(p)
        print(
            f"  {p['label']}: has_barrier={p.get('has_barrier')} "
            f"ΔU={p.get('barrier_height', float('nan')):.3f}",
            flush=True,
        )
    saddle = _attach_nearest_subtype(_global_path_saddle(paths), wells)
    return field, paths, wells, saddle


def export_single(field, paths, wells, saddle, *, title: str, out: Path, eye=None):
    # Saddle markers disabled for all interactive HTML panels.
    _ = saddle
    fig = go.Figure()
    _add_landscape_traces(
        fig, field, paths, wells, global_saddle=None, showlegend=True
    )
    xx, yy, Z_show, lo, hi, _ = _prepare_surface(field)
    # Explicit axis ranges keep the camera framed on the manifold (not empty space).
    pad_x = 0.04 * float(xx.max() - xx.min())
    pad_y = 0.04 * float(yy.max() - yy.min())
    pad_z = 0.08 * max(float(hi - lo), 1e-3)
    scene = _scene_layout(camera_eye=eye)
    scene["xaxis"]["range"] = [float(xx.min()) - pad_x, float(xx.max()) + pad_x]
    scene["yaxis"]["range"] = [float(yy.min()) - pad_y, float(yy.max()) + pad_y]
    scene["zaxis"]["range"] = [float(lo) - pad_z, float(hi) + 1.5 * pad_z]
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(family=FONT_FAMILY, size=15, color="#0F172A"),
            x=0.03,
            y=0.97,
        ),
        scene=scene,
        height=880,
        width=1280,
        margin=dict(l=10, r=10, t=55, b=20),
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0.0,
            font=dict(family=FONT_FAMILY, size=9),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            itemsizing="constant",
        ),
    )
    _write_html(fig, out)


def export_overview(alv, mac, *, out: Path):
    field_a, paths_a, wells_a, _sad_a = alv
    field_m, paths_m, wells_m, _sad_m = mac
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "<b>Alveolar Epithelium Landscape</b>",
            "<b>Macrophage Remodeling Landscape</b>",
        ),
        horizontal_spacing=0.03,
    )
    # No saddle diamonds on either overview panel.
    _add_landscape_traces(
        fig, field_a, paths_a, wells_a, global_saddle=None, row=1, col=1, showlegend=True
    )
    n_before = len(fig.data)
    _add_landscape_traces(
        fig, field_m, paths_m, wells_m, global_saddle=None, row=1, col=2, showlegend=False
    )
    for tr in fig.data[n_before:]:
        if isinstance(tr, go.Surface):
            tr.showscale = False
            break

    fig.update_layout(
        title=dict(
            text=(
                "<b>GSE141259 · 3D Nonequilibrium Potential Landscapes "
                "& Least Action Paths</b>"
            ),
            font=dict(family=FONT_FAMILY, size=16, color="#0F172A"),
            x=0.03,
            y=0.97,
        ),
        height=760,
        margin=dict(l=10, r=10, t=65, b=25),
        paper_bgcolor="white",
        scene=_scene_layout(camera_eye=dict(x=1.35, y=-1.45, z=0.85)),
        scene2=_scene_layout(camera_eye=dict(x=1.45, y=-1.35, z=0.82)),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.98,
            x=0.25,
            font=dict(family=FONT_FAMILY, size=10),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#E2E8F0",
            borderwidth=1,
        ),
    )
    _write_html(fig, out)


def main():
    ap = argparse.ArgumentParser(
        description="Export interactive 3D U_rel landscapes (Plotly HTML)."
    )
    ap.add_argument(
        "--auto-edges",
        action="store_true",
        help=(
            "Infer paths from pseudotime/stage order and UMAP kNN mixing "
            "instead of curated biology priors"
        ),
    )
    ap.add_argument(
        "--panels",
        nargs="+",
        default=["alv", "mac", "overview", "club"],
        choices=["alv", "mac", "overview", "club", "club_plus_at2_adi"],
        help="Which HTML panels to export (default: all, including Club).",
    )
    args = ap.parse_args()
    mode = "auto-edges" if args.auto_edges else "curated"
    panels = set(args.panels)

    alv = mac = None
    if "alv" in panels or "overview" in panels:
        print(
            f"===== Building Alveolar Epithelium 3D Landscape ({mode}) =====",
            flush=True,
        )
        alv = build_alv_bundle(auto_edges=args.auto_edges)
        if "alv" in panels:
            export_single(
                *alv,
                title=(
                    "Alveolar Epithelium · Nonequilibrium Potential Landscape (3D LAP"
                    + (", auto edges)" if args.auto_edges else ")")
                ),
                out=PANELS / "GSE141259_alv_3d_urel_landscape_interactive.html",
                eye=dict(x=1.35, y=-1.45, z=0.85),
            )

    if "mac" in panels or "overview" in panels:
        print(f"===== Building Macrophages 3D Landscape ({mode}) =====", flush=True)
        mac = build_mac_bundle(auto_edges=args.auto_edges)
        if "mac" in panels:
            export_single(
                *mac,
                title=(
                    "Macrophages · Remodeling Potential Landscape (3D LAP"
                    + (", auto edges)" if args.auto_edges else ")")
                ),
                out=PANELS / "GSE141259_mac_3d_urel_landscape_interactive.html",
                eye=dict(x=1.45, y=-1.35, z=0.82),
            )

    if "overview" in panels:
        if alv is None or mac is None:
            raise RuntimeError("overview requires both alv and mac bundles")
        print("===== Building Dual Overview Landscape =====", flush=True)
        export_overview(
            alv,
            mac,
            out=PANELS / "GSE141259_mac_alv_3d_urel_landscape_interactive.html",
        )

    if "club" in panels:
        print(f"===== Building Club Lineage 3D Landscape ({mode}) =====", flush=True)
        club = build_club_bundle(auto_edges=args.auto_edges)
        export_single(
            *club,
            title=(
                "Club Lineage · Regenerative & Classical Potential Landscape (3D LAP"
                + (", auto edges)" if args.auto_edges else ")")
            ),
            out=PANELS / "GSE141259_club_3d_urel_landscape_interactive.html",
            eye=dict(x=1.40, y=-1.40, z=0.88),
        )

    if "club_plus_at2_adi" in panels:
        print(
            f"===== Building Club + AT2→ADI Landscape ({mode}) =====",
            flush=True,
        )
        club_plus = build_club_bundle(
            auto_edges=args.auto_edges,
            extra_edges=[("AT2 cells", "Krt8 ADI")],
        )
        export_single(
            *club_plus,
            title=(
                "Club Lineage · Base paths + AT2→ADI (same surface as Club HTML"
                + (", auto edges)" if args.auto_edges else ")")
            ),
            out=PANELS / "GSE141259_club_3d_urel_landscape_interactive_plus_AT2_ADI.html",
            eye=dict(x=1.40, y=-1.40, z=0.88),
        )

    print(
        f"\n[Done] edge_mode={mode}; panels={sorted(panels)}. "
        "Open the HTML files in a browser to rotate and export PNG."
    )


if __name__ == "__main__":
    main()
