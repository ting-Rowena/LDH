#!/usr/bin/env python3
"""3D potential landscapes for GSE141259 Macrophages & Alveolar epithelium.

Fig. 5F-style surfaces: XY = training UMAP, Z = U_rel, with least-cost LAP
curves, floor projection, and transition-state saddles.

Paths use the continuous-field geodesic on a smoothed U_rel field (same family
as the 2D Mac/Alv landscape panels). Flow-space LAP is attempted when
``--try-flow`` is set, but currently often degenerates on these subsets.

Outputs
-------
  <CK>/analysis_protocol_GSE141259/figures/GSE141259_alv_3d_urel_landscape_lap.png
  <CK>/analysis_protocol_GSE141259/figures/GSE141259_alv_3d_urel_landscape_lap_top.png
  <CK>/analysis_protocol_GSE141259/figures/GSE141259_mac_3d_urel_landscape_lap.png
  <CK>/analysis_protocol_GSE141259/figures/GSE141259_mac_3d_urel_landscape_lap_top.png
  <CK>/analysis_protocol_GSE141259/figures/GSE141259_mac_alv_3d_urel_landscape_overview.png
  <CK>/analysis_protocol_GSE141259/figures/GSE141259_mac_alv_3d_urel_landscape_overview_top.png
  output_file/mac_landscape_audit/GSE141259_mac_alv_3d_landscape_lap_summary.csv
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import interpolate, ndimage
from scipy.spatial import ConvexHull, cKDTree
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from analyze_mac_alv_dynamics_first_paths import ALV_TYPES, MAC_TYPES  # noqa: E402
from CellFateLandscape import NonEquilibriumCellFateLandscape  # noqa: E402
from dataset_pipeline import PROJECT_ROOT, recommended_checkpoint_dir  # noqa: E402
from landscape_core import build_safe_scalar_field  # noqa: E402
from panel_style import apply_panel_title_rc  # noqa: E402
from plot_utils import INK, configure_headless  # noqa: E402

configure_headless()
apply_panel_title_rc()

CK = Path(recommended_checkpoint_dir("GSE141259"))
PROTO = CK / "analysis_protocol_GSE141259"
TAB = PROJECT_ROOT / "output_file" / "mac_landscape_audit"
PANELS = PROTO / "figures"
for path in (TAB, PANELS, PROTO / "figures"):
    path.mkdir(parents=True, exist_ok=True)

POTENTIAL_KEY = "potential_relative_type"
SURF_CMAP = LinearSegmentedColormap.from_list(
    "urel_3d",
    [
        "#08306B",
        "#2171B5",
        "#6BAED6",
        "#C6DBEF",
        "#FFFFCC",
        "#FED976",
        "#FD8D3C",
        "#E31A1C",
        "#800026",
    ],
)
SHORT = {
    "AT2 cells": "AT2",
    "Activated AT2 cells": "Act.AT2",
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
    "D0": "D0",
    "D28": "D28",
}

# Cross-lineage Club regenerative / classical panel (cell.type labels).
CLUB_LINEAGE_TYPES = [
    "MHC-II+ Club cells",
    "Club cells",
    "AT2 cells",
    "Krt8 ADI",
    "AT1 cells",
    "Ciliated cells",
    "Goblet cells",
]
CLUB_LINEAGE_SOURCE_TYPES = [
    "Club cells",
    "AT2 cells",
    "Krt8 ADI",
    "AT1 cells",
    "Ciliated cells",
    "Ciliated cell subset",
    "Goblet cells",
]
MHC_CLUB_GENES = ("H2-Ab1", "H2-Aa", "H2-Eb1", "Cd74")
MHC_CLUB_SUPPORT = ("Cst3",)


# --------------------------------------------------------------------------- data
def _load_parent(parent: str, types: list[str]) -> ad.AnnData:
    obs = pd.read_csv(CK / "obs.csv", index_col=0, low_memory=False)
    obs.index = obs.index.astype(str)
    umap = np.load(CK / "training_umap.npz", allow_pickle=True)
    u_idx = pd.Index(np.asarray(umap["index"]).astype(str))
    X_umap = np.asarray(umap["X_umap"], float)
    mapper = {b: i for i, b in enumerate(u_idx)}

    m = obs["annotation"].astype(str).eq(parent) & obs["cell.type"].astype(str).isin(types)
    obs_p = obs.loc[m].copy()
    keep = [b for b in obs_p.index if b in mapper]
    obs_p = obs_p.loc[keep]
    xy = X_umap[[mapper[b] for b in keep]]
    U = pd.to_numeric(obs_p[POTENTIAL_KEY], errors="coerce").to_numpy(float)

    adata = ad.AnnData(X=np.zeros((len(obs_p), 1), dtype=np.float32))
    adata.obs_names = pd.Index(keep, dtype=str)
    adata.obs = obs_p.copy()
    adata.obs[POTENTIAL_KEY] = U
    adata.obsm["X_umap"] = xy
    # Parent-internal PCA of UMAP is unnecessary; analyzer uses X_umap directly.
    return adata


def _load_cell_types(types: list[str]) -> ad.AnnData:
    """Load cells by ``cell.type`` across metacelltype parents (shared UMAP / U_rel)."""
    obs = pd.read_csv(CK / "obs.csv", index_col=0, low_memory=False)
    obs.index = obs.index.astype(str)
    umap = np.load(CK / "training_umap.npz", allow_pickle=True)
    u_idx = pd.Index(np.asarray(umap["index"]).astype(str))
    X_umap = np.asarray(umap["X_umap"], float)
    mapper = {b: i for i, b in enumerate(u_idx)}

    m = obs["cell.type"].astype(str).isin(types)
    obs_p = obs.loc[m].copy()
    keep = [b for b in obs_p.index if b in mapper]
    obs_p = obs_p.loc[keep]
    xy = X_umap[[mapper[b] for b in keep]]
    U = pd.to_numeric(obs_p[POTENTIAL_KEY], errors="coerce").to_numpy(float)

    adata = ad.AnnData(X=np.zeros((len(obs_p), 1), dtype=np.float32))
    adata.obs_names = pd.Index(keep, dtype=str)
    adata.obs = obs_p.copy()
    adata.obs[POTENTIAL_KEY] = U
    adata.obsm["X_umap"] = xy
    return adata


def _club_gene_matrix(barcodes: list[str], genes: list[str]) -> tuple[np.ndarray, list[str]]:
    """Log-normalized expression for selected genes (Club MHC-II scoring)."""
    from dataset_pipeline import GSE141259, resolve_data_path
    import scanpy as sc

    h5ad = resolve_data_path(GSE141259)
    raw = ad.read_h5ad(h5ad, backed="r")
    name_to_i = {b: i for i, b in enumerate(raw.obs_names.astype(str))}
    idx = [name_to_i[b] for b in barcodes if b in name_to_i]
    present = [g for g in genes if g in set(map(str, raw.var_names))]
    if not idx or not present:
        raw.file.close()
        return np.zeros((len(barcodes), 0), dtype=float), []
    # h5py backed AnnData allows only one fancy index at a time.
    sub = raw[idx].to_memory()
    raw.file.close()
    present = [g for g in present if g in set(map(str, sub.var_names))]
    if not present:
        return np.zeros((len(barcodes), 0), dtype=float), []
    sub = sub[:, present].copy()
    if "log1p" not in sub.uns:
        sc.pp.normalize_total(sub, target_sum=1e4, inplace=True)
        sc.pp.log1p(sub)
    X = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X, float)
    # Align back to requested barcode order (missing → 0)
    out = np.zeros((len(barcodes), len(present)), dtype=float)
    fetched = {b: j for j, b in enumerate(np.asarray(sub.obs_names.astype(str)))}
    for i, b in enumerate(barcodes):
        j = fetched.get(b)
        if j is not None:
            out[i] = X[j]
    return out, present


def annotate_club_lineage_labels(adata: ad.AnnData, *, q_mhc: float = 0.60, q_cst: float = 0.60) -> ad.AnnData:
    """Merge ciliated subsets; split Club into MHC-II⁺ ∩ Cst3-high vs residual Club."""
    labels = adata.obs["cell.type"].astype(str).to_numpy().copy()
    labels[labels == "Ciliated cell subset"] = "Ciliated cells"

    club_mask = labels == "Club cells"
    if club_mask.any():
        genes = list(MHC_CLUB_GENES) + list(MHC_CLUB_SUPPORT)
        X, present = _club_gene_matrix(adata.obs_names.astype(str).tolist(), genes)
        if present:
            mhc_idx = [present.index(g) for g in MHC_CLUB_GENES if g in present]
            cst_idx = [present.index(g) for g in MHC_CLUB_SUPPORT if g in present]
            mhc = X[:, mhc_idx].mean(axis=1) if mhc_idx else np.zeros(adata.n_obs)
            cst = X[:, cst_idx[0]] if cst_idx else np.ones(adata.n_obs)
            mhc_c = mhc[club_mask]
            cst_c = cst[club_mask]
            thr_mhc = float(np.nanpercentile(mhc_c, 100.0 * q_mhc))
            thr_cst = float(np.nanpercentile(cst_c, 100.0 * q_cst))
            hi = club_mask.copy()
            hi[club_mask] = (mhc_c >= thr_mhc) & (cst_c >= thr_cst)
            # If intersection is too small, fall back to MHC-only top quantile.
            if int(hi.sum()) < 80:
                hi[club_mask] = mhc_c >= thr_mhc
            labels[hi] = "MHC-II+ Club cells"
            print(
                f"  MHC-II+ Club split: MHC≥{thr_mhc:.3f} (q={q_mhc:.2f}) "
                f"& Cst3≥{thr_cst:.3f} (q={q_cst:.2f}) → "
                f"n+={int(hi.sum())} / n_club={int(club_mask.sum())}",
                flush=True,
            )

    adata = adata.copy()
    adata.obs["cell.type"] = pd.Categorical(labels)
    keep = np.isin(labels, CLUB_LINEAGE_TYPES)
    return adata[keep].copy()


def load_club_lineage_adata() -> ad.AnnData:
    """Club regenerative + classical fate panel with MHC-II⁺ Club annotation."""
    adata = _load_cell_types(CLUB_LINEAGE_SOURCE_TYPES)
    return annotate_club_lineage_labels(adata)


def _core_idx(U: np.ndarray, mask: np.ndarray, *, frac: float = 0.25, min_n: int = 8) -> np.ndarray:
    ix = np.where(mask)[0]
    if ix.size == 0:
        return ix
    k = min(ix.size, max(min_n, int(np.ceil(frac * ix.size))))
    return ix[np.argsort(U[ix])[:k]]


def _core_centroid(xy: np.ndarray, U: np.ndarray, mask: np.ndarray) -> np.ndarray:
    core = _core_idx(U, mask)
    if core.size == 0:
        return np.full(2, np.nan)
    return xy[core].mean(axis=0)


def _spatial_median_centroid(xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Robust subtype center on the UMAP manifold (not low-U biased)."""
    ix = np.where(mask)[0]
    if ix.size == 0:
        return np.full(2, np.nan)
    return np.median(xy[ix], axis=0)


def _midpath_chevron_xyz(
    path_xy: np.ndarray,
    U_path: np.ndarray,
    xx,
    yy,
    lo: float,
    hi: float,
    *,
    frac: float = 0.55,
):
    """Tiny mid-path direction chevron for static 3D figures."""
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
    tip_len, wing = 0.016, 0.009
    mid_n = mid / scale
    tip = (mid_n + tang * tip_len) * scale
    left = (mid_n - tang * 0.45 * tip_len + side * wing) * scale
    right = (mid_n - tang * 0.45 * tip_len - side * wing) * scale
    return np.vstack([left, tip, right])


def _smooth_polyline(path_xy: np.ndarray, *, n_out: int = 100) -> np.ndarray:
    """Resample a polyline without cubic-spline overshoot.

    ``UnivariateSpline(..., s=0, k=3)`` can explode to 1e12+ when FMM/Dijkstra
    backtraces contain near-collinear or jittery vertices (seen on Club AT2→ADI),
    which then auto-zooms interactive 3D plots until only subtype labels remain.
    Use monotone PCHIP (no overshoot), with linear fallback + bbox guard.
    """
    path_xy = np.asarray(path_xy, float)
    if len(path_xy) < 4:
        return path_xy
    # Drop near-duplicate vertices that destabilize parametric interpolants.
    seg = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    keep = np.r_[True, seg > 1e-8]
    keep[-1] = True
    path_xy = path_xy[keep]
    if len(path_xy) < 4:
        return path_xy
    seg = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] <= 0:
        return path_xy
    s /= s[-1]
    ss = np.linspace(0.0, 1.0, n_out)

    def _linear():
        return np.c_[np.interp(ss, s, path_xy[:, 0]), np.interp(ss, s, path_xy[:, 1])]

    try:
        spl_x = interpolate.PchipInterpolator(s, path_xy[:, 0])
        spl_y = interpolate.PchipInterpolator(s, path_xy[:, 1])
        out = np.c_[spl_x(ss), spl_y(ss)]
    except Exception:
        out = _linear()

    out[0] = path_xy[0]
    out[-1] = path_xy[-1]
    if not np.all(np.isfinite(out)):
        return _linear()

    # Hard guard: reject any resampling that leaves a padded axis-aligned hull
    # of the raw vertices (catches residual interpolant blow-ups).
    lo = path_xy.min(axis=0)
    hi = path_xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pad = 0.75 * span
    if np.any(out.min(axis=0) < lo - pad) or np.any(out.max(axis=0) > hi + pad):
        return _linear()
    return out


def _detect_transition_state(
    U_path: np.ndarray,
    field_dynamic_range: float,
    *,
    barrier_fraction: float = 0.05,
) -> tuple[int, bool, float]:
    """Detect a significant internal path barrier instead of forcing a TS."""
    U_path = np.asarray(U_path, float)
    n_pts = len(U_path)
    if n_pts < 5 or not np.all(np.isfinite(U_path)):
        return 0, False, 0.0

    lo = max(1, int(0.08 * n_pts))
    hi = min(n_pts - 1, max(lo + 1, int(0.92 * n_pts)))
    ts_idx = int(lo + np.argmax(U_path[lo:hi]))
    barrier_height = float(U_path[ts_idx] - max(U_path[0], U_path[-1]))
    barrier_thresh = float(barrier_fraction * max(field_dynamic_range, 1e-4))

    # Require a strict local maximum away from both search-window boundaries.
    is_interior = lo < ts_idx < hi - 1
    has_negative_curvature = (
        is_interior
        and U_path[ts_idx] > U_path[ts_idx - 1]
        and U_path[ts_idx] > U_path[ts_idx + 1]
        and (U_path[ts_idx - 1] - 2.0 * U_path[ts_idx] + U_path[ts_idx + 1]) < 0
    )
    has_barrier = bool(barrier_height > barrier_thresh and has_negative_curvature)
    return ts_idx, has_barrier, barrier_height


def _field_dynamic_range(Z: np.ndarray) -> float:
    finite = np.asarray(Z, float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    return float(np.nanpercentile(finite, 95) - np.nanpercentile(finite, 5))


def _evaluate_saddle_point_2d(
    xx: np.ndarray,
    yy: np.ndarray,
    Z: np.ndarray,
    cand_xy: np.ndarray,
    *,
    grad_tol_ratio: float = 0.15,
) -> dict:
    """Strict 2D saddle test: ||∇U|| small and det(H) = Uxx·Uyy − Uxy² < 0."""
    Z = np.asarray(Z, float)
    xs, ys = xx[0], yy[:, 0]
    dx = float(abs(xs[1] - xs[0])) if len(xs) > 1 else 1.0
    dy = float(abs(ys[1] - ys[0])) if len(ys) > 1 else 1.0
    Z_fill = np.where(np.isfinite(Z), Z, np.nanmedian(Z[np.isfinite(Z)]))

    Zy, Zx = np.gradient(Z_fill, dy, dx)
    Zyy, Zyx = np.gradient(Zy, dy, dx)
    Zxy, Zxx = np.gradient(Zx, dy, dx)
    Uxy = 0.5 * (Zxy + Zyx)
    det_H_field = Zxx * Zyy - Uxy**2
    grad_norm_field = np.hypot(Zx, Zy)

    cand = np.asarray(cand_xy, float).reshape(-1)
    ix = float(np.interp(cand[0], xs, np.arange(len(xs))))
    iy = float(np.interp(cand[1], ys, np.arange(len(ys))))
    grad_norm = float(
        ndimage.map_coordinates(grad_norm_field, [[iy], [ix]], order=1, mode="nearest")[0]
    )
    det_H = float(
        ndimage.map_coordinates(det_H_field, [[iy], [ix]], order=1, mode="nearest")[0]
    )

    finite_grad = grad_norm_field[np.isfinite(Z) & np.isfinite(grad_norm_field)]
    if finite_grad.size:
        grad_thresh = float(np.nanpercentile(finite_grad, 25)) * (1.0 + grad_tol_ratio)
    else:
        grad_thresh = 1e-3
    is_saddle = bool((det_H < 0.0) and (grad_norm <= max(grad_thresh, 1e-3)))
    return {
        "grad_norm": grad_norm,
        "det_H": det_H,
        "grad_thresh": float(max(grad_thresh, 1e-3)),
        "is_strict_saddle": is_saddle,
    }


def _attach_path_saddle_metrics(
    path: dict, xx, yy, Z, *, U_func=None
) -> dict:
    """Annotate path dict with 1D barrier + strict 2D saddle metrics at TS."""
    path = dict(path)
    xy = np.asarray(path["path_xy"], float)
    Uz = np.asarray(path["U_path"], float)
    dynamic_range = _field_dynamic_range(Z)
    ts, has_barrier, barrier_height = _detect_transition_state(Uz, dynamic_range)
    ts = int(np.clip(ts, 0, len(xy) - 1))
    path["ts"] = ts
    path["has_barrier"] = bool(has_barrier)
    path["barrier_height"] = float(barrier_height)
    path["barrier_thresh"] = float(0.05 * max(dynamic_range, 1e-4))
    saddle = _evaluate_saddle_point_2d(xx, yy, Z, xy[ts])
    path.update(saddle)
    return path


def _support_weight(xx: np.ndarray, yy: np.ndarray, xy: np.ndarray) -> np.ndarray:
    pts = np.c_[xx.ravel(), yy.ravel()]
    if len(xy) < 3:
        return np.ones(xx.shape, float)
    try:
        hull = ConvexHull(xy)
        from matplotlib.path import Path as MplPath

        inside = MplPath(xy[hull.vertices]).contains_points(pts).reshape(xx.shape)
    except Exception:
        tree = cKDTree(xy)
        d, _ = tree.query(pts, k=1)
        d = d.reshape(xx.shape)
        d_cell, _ = tree.query(xy, k=min(8, len(xy)))
        scale = float(np.median(d_cell[:, 1:])) + 1e-8
        inside = d <= 5.0 * scale
    dx = float(abs(xx[0, 1] - xx[0, 0])) if xx.shape[1] > 1 else 1.0
    dy = float(abs(yy[1, 0] - yy[0, 0])) if xx.shape[0] > 1 else 1.0
    fade = 0.07 * float(max(xx.max() - xx.min(), yy.max() - yy.min()) + 1e-8)
    dist_out = ndimage.distance_transform_edt(~inside, sampling=(dy, dx))
    w = np.where(inside, 1.0, np.clip(1.0 - dist_out / fade, 0.0, 1.0))
    return np.clip(ndimage.gaussian_filter(w, sigma=2.2, mode="nearest"), 0.0, 1.0)


def _solve_eikonal_fmm_path(
    xx: np.ndarray,
    yy: np.ndarray,
    Z_path: np.ndarray,
    start,
    end,
    *,
    barrier_weight: float = 3.0,
    n_out: int = 100,
) -> np.ndarray | None:
    """Solve the Eikonal equation with FMM and backtrace a geodesic via RK2.

    Speed field: F = 1 / (1 + α · Z_norm). Falls back to ``None`` if scikit-fmm
    is unavailable or the march fails.
    """
    try:
        import skfmm
    except ImportError:
        return None

    xs, ys = xx[0], yy[:, 0]
    if len(xs) < 3 or len(ys) < 3:
        return None
    dx = float(abs(xs[1] - xs[0]))
    dy = float(abs(ys[1] - ys[0]))
    Zs = np.asarray(Z_path, float)
    finite = np.isfinite(Zs)
    if not np.any(finite):
        return None

    z_min = float(np.nanmin(Zs[finite]))
    z_scale = float(np.nanpercentile(Zs[finite] - z_min, 95)) + 1e-8
    z_norm = np.where(finite, np.clip((Zs - z_min) / z_scale, 0.0, None), 10.0)
    speed = 1.0 / (1.0 + barrier_weight * z_norm)
    speed = np.where(finite, np.clip(speed, 1e-4, None), 1e-4)

    def _nearest_idx(pt):
        j = int(np.argmin(np.abs(xs - pt[0])))
        i = int(np.argmin(np.abs(ys - pt[1])))
        if not finite[i, j]:
            ii, jj = np.where(finite)
            k = int(np.argmin((xs[jj] - pt[0]) ** 2 + (ys[ii] - pt[1]) ** 2))
            return int(ii[k]), int(jj[k])
        return i, j

    si, sj = _nearest_idx(start)
    phi = np.ones_like(Zs, dtype=float)
    phi[si, sj] = -1.0
    try:
        # skfmm expects dx as [dy, dx] for array indexed [i=y, j=x]
        t_field = skfmm.travel_time(phi, speed, dx=(dy, dx))
    except Exception:
        return None
    if not np.any(np.isfinite(t_field)):
        return None

    ty, tx = np.gradient(t_field, dy, dx)
    gnorm = np.hypot(tx, ty) + 1e-12
    tx = tx / gnorm
    ty = ty / gnorm

    cur_x, cur_y = float(end[0]), float(end[1])
    path = [[cur_x, cur_y]]
    step_size = 0.5 * min(dx, dy)
    max_steps = int(max(len(xs), len(ys)) * 6)
    for _ in range(max_steps):
        if np.hypot(cur_x - float(start[0]), cur_y - float(start[1])) <= step_size:
            break
        if not (xs[0] <= cur_x <= xs[-1] and ys[0] <= cur_y <= ys[-1]):
            break
        ix = float(np.interp(cur_x, xs, np.arange(len(xs))))
        iy = float(np.interp(cur_y, ys, np.arange(len(ys))))
        vx = float(ndimage.map_coordinates(tx, [[iy], [ix]], order=1, mode="nearest")[0])
        vy = float(ndimage.map_coordinates(ty, [[iy], [ix]], order=1, mode="nearest")[0])
        mid_x = cur_x - 0.5 * step_size * vx
        mid_y = cur_y - 0.5 * step_size * vy
        mid_ix = float(np.interp(mid_x, xs, np.arange(len(xs))))
        mid_iy = float(np.interp(mid_y, ys, np.arange(len(ys))))
        vx2 = float(
            ndimage.map_coordinates(tx, [[mid_iy], [mid_ix]], order=1, mode="nearest")[0]
        )
        vy2 = float(
            ndimage.map_coordinates(ty, [[mid_iy], [mid_ix]], order=1, mode="nearest")[0]
        )
        cur_x -= step_size * vx2
        cur_y -= step_size * vy2
        path.append([cur_x, cur_y])

    path.append([float(start[0]), float(start[1])])
    raw = np.asarray(path[::-1], float)
    if len(raw) < 4:
        return None
    return _smooth_polyline(raw, n_out=n_out)


def _dijkstra_field_path_xy(
    xx, yy, Z, start, end, *, barrier_weight: float = 3.0, n_out: int = 100
) -> tuple[np.ndarray, float]:
    """Existing weighted-grid Dijkstra geodesic (FMM fallback)."""
    ny, nx = Z.shape
    xs, ys = xx[0], yy[:, 0]
    Zs = np.asarray(Z, float)
    finite = np.isfinite(Zs)
    if not np.any(finite):
        raise RuntimeError("empty potential field")
    z_shift = np.where(finite, Zs - np.nanmin(Zs[finite]), np.nan)
    z_scale = np.nanpercentile(z_shift[finite], 95) + 1e-8
    z_norm = np.where(finite, np.clip(z_shift / z_scale, 0.0, None), np.inf)

    def _nearest_idx(pt):
        j = int(np.argmin(np.abs(xs - pt[0])))
        i = int(np.argmin(np.abs(ys - pt[1])))
        if not np.isfinite(z_norm[i, j]):
            ii, jj = np.where(finite)
            k = int(np.argmin((xs[jj] - pt[0]) ** 2 + (ys[ii] - pt[1]) ** 2))
            return int(ii[k]), int(jj[k])
        return i, j

    si, sj = _nearest_idx(start)
    ti, tj = _nearest_idx(end)
    start_id, end_id = si * nx + sj, ti * nx + tj
    neigh = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    dist = np.full(ny * nx, np.inf)
    prev = np.full(ny * nx, -1, dtype=int)
    dist[start_id] = 0.0
    heap = [(0.0, start_id)]
    while heap:
        du, uid = heapq.heappop(heap)
        if du > dist[uid]:
            continue
        if uid == end_id:
            break
        i, j = divmod(uid, nx)
        for di, dj in neigh:
            ni, nj = i + di, j + dj
            if not (0 <= ni < ny and 0 <= nj < nx) or not np.isfinite(z_norm[ni, nj]):
                continue
            step = float(np.hypot(xs[nj] - xs[j], ys[ni] - ys[i]))
            barrier = 0.5 * (z_norm[i, j] + z_norm[ni, nj])
            alt = du + step * (1.0 + barrier_weight * barrier)
            vid = ni * nx + nj
            if alt < dist[vid]:
                dist[vid] = alt
                prev[vid] = uid
                heapq.heappush(heap, (alt, vid))
    if not np.isfinite(dist[end_id]):
        raise RuntimeError("continuous field path failed")
    ids = []
    cur = end_id
    while cur >= 0:
        ids.append(cur)
        cur = int(prev[cur])
    ids = ids[::-1]
    raw = np.asarray([[xs[uid % nx], ys[uid // nx]] for uid in ids], float)
    raw[0] = np.asarray(start, float)
    raw[-1] = np.asarray(end, float)
    return _smooth_polyline(raw, n_out=n_out), float(dist[end_id])


def _continuous_field_path(
    xx, yy, Z, start, end, U_func, *, barrier_weight: float = 3.0, n_out: int = 100
):
    """Geodesic on the potential field: FMM Eikonal first, Dijkstra fallback."""
    method = "eikonal_fmm"
    action = np.nan
    path_xy = _solve_eikonal_fmm_path(
        xx, yy, Z, start, end, barrier_weight=barrier_weight, n_out=n_out
    )
    if path_xy is None:
        method = "continuous_field_geodesic"
        path_xy, action = _dijkstra_field_path_xy(
            xx, yy, Z, start, end, barrier_weight=barrier_weight, n_out=n_out
        )
    else:
        # Approximate action as integrated travel cost along the FMM path
        Zs = np.asarray(Z, float)
        finite = np.isfinite(Zs)
        z_min = float(np.nanmin(Zs[finite]))
        z_scale = float(np.nanpercentile(Zs[finite] - z_min, 95)) + 1e-8
        cost = 0.0
        for a, b in zip(path_xy[:-1], path_xy[1:]):
            mid = 0.5 * (a + b)
            # nearest-grid U for cost
            xs, ys = xx[0], yy[:, 0]
            j = int(np.clip(np.argmin(np.abs(xs - mid[0])), 0, len(xs) - 1))
            i = int(np.clip(np.argmin(np.abs(ys - mid[1])), 0, len(ys) - 1))
            zn = 0.0
            if np.isfinite(Zs[i, j]):
                zn = max((Zs[i, j] - z_min) / z_scale, 0.0)
            cost += float(np.linalg.norm(b - a) * (1.0 + barrier_weight * zn))
        action = cost

    U_path = np.asarray(
        [float(np.asarray(U_func(p)).reshape(-1)[0]) for p in path_xy], float
    )
    out = {
        "path_xy": path_xy,
        "U_path": U_path,
        "action": float(action),
        "method": method,
        "success": True,
        "degenerate": False,
    }
    return _attach_path_saddle_metrics(out, xx, yy, Z)


def _fit_tps_field(fit_xy: np.ndarray, fit_U: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Thin-plate spline approximation via scipy RBFInterpolator."""
    from scipy.interpolate import RBFInterpolator

    q = np.asarray(query, float)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    rbf = RBFInterpolator(
        np.asarray(fit_xy, float),
        np.asarray(fit_U, float),
        kernel="thin_plate_spline",
        smoothing=1e-3,
    )
    return np.asarray(rbf(q), float)


def _build_field(
    adata: ad.AnnData,
    *,
    n_grid: int = 140,
    max_fit: int | None = None,
    smooth_sigma: float = 3.5,
    method: str = "rbf",
):
    """Build a smoothed U_rel field on a robust UMAP grid.

    Parameters
    ----------
    max_fit :
        If set and smaller than n_cells, keep extreme-U anchors plus random
        fill (fast fitting for large parents).
    method :
        ``\"rbf\"`` (default, via ``build_safe_scalar_field``) or
        ``\"tps\"`` (thin-plate spline via ``RBFInterpolator``).
    """
    xy = np.asarray(adata.obsm["X_umap"], float)
    U = pd.to_numeric(adata.obs[POTENTIAL_KEY], errors="coerce").to_numpy(float)
    method = str(method).lower().strip()
    if method not in {"rbf", "tps"}:
        raise ValueError(f"Unsupported field method {method!r}; use 'rbf' or 'tps'.")

    # Extreme-U anchor sampling (kept for large parents / TPS cost control)
    if max_fit is not None and len(xy) > max_fit:
        rng = np.random.default_rng(0)
        order = np.argsort(U)
        n_anchor = min(100, max(40, len(xy) // 10))
        anchors = np.unique(np.r_[order[:n_anchor], order[-n_anchor:]])
        pool = np.setdiff1d(np.arange(len(xy)), anchors, assume_unique=False)
        n_rand = max(0, max_fit - len(anchors))
        chosen = np.unique(
            np.r_[anchors, rng.choice(pool, size=min(n_rand, len(pool)), replace=False)]
        )
        fit_xy, fit_U = xy[chosen], U[chosen]
        print(
            f"    field fit on {len(fit_xy)}/{len(xy)} cells "
            f"(anchors={len(anchors)}, method={method})",
            flush=True,
        )
    else:
        fit_xy, fit_U = xy, U
        print(f"    field fit on ALL {len(fit_xy)} cells (method={method})", flush=True)

    # Robust bounding box: drop extreme UMAP outliers that otherwise stretch the
    # landscape canvas and push the dense manifold into one visual corner.
    pad = 0.04
    x0, x1 = np.nanpercentile(xy[:, 0], [0.5, 99.5])
    y0, y1 = np.nanpercentile(xy[:, 1], [0.5, 99.5])
    dx, dy = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    xx, yy = np.meshgrid(
        np.linspace(x0 - pad * dx, x1 + pad * dx, n_grid),
        np.linspace(y0 - pad * dy, y1 + pad * dy, n_grid),
    )
    inlier = (
        (xy[:, 0] >= x0 - pad * dx)
        & (xy[:, 0] <= x1 + pad * dx)
        & (xy[:, 1] >= y0 - pad * dy)
        & (xy[:, 1] <= y1 + pad * dy)
    )
    xy_support = xy[inlier] if np.any(inlier) else xy
    pts = np.c_[xx.ravel(), yy.ravel()]

    if method == "tps":
        # Cache one interpolator for raw queries
        from scipy.interpolate import RBFInterpolator

        tps = RBFInterpolator(
            fit_xy, fit_U, kernel="thin_plate_spline", smoothing=1e-3
        )

        def U_func_raw(query):
            q = np.asarray(query, float)
            if q.ndim == 1:
                return float(tps(q.reshape(1, -1))[0])
            return np.asarray(tps(q), float)

        Z_raw = np.asarray(tps(pts), float).reshape(xx.shape)
    else:
        U_func_raw, _ = build_safe_scalar_field(fit_xy, fit_U, method="rbf")
        Z_raw = np.asarray(U_func_raw(pts), float).reshape(xx.shape)

    Z = ndimage.gaussian_filter(
        np.where(np.isfinite(Z_raw), Z_raw, np.nanmedian(Z_raw[np.isfinite(Z_raw)])),
        sigma=smooth_sigma,
        mode="nearest",
    )
    weight = _support_weight(xx, yy, xy_support)
    Z_path = np.where(weight >= 0.22, Z, np.nan)
    Z_disp = Z.copy()
    Z_disp[weight < 0.12] = np.nan

    def U_display(query):
        q = np.asarray(query, float).reshape(-1)
        xs, ys = xx[0], yy[:, 0]
        j = np.interp(q[0], xs, np.arange(len(xs)))
        i = np.interp(q[1], ys, np.arange(len(ys)))
        i0 = int(np.clip(np.floor(i), 0, Z.shape[0] - 2))
        j0 = int(np.clip(np.floor(j), 0, Z.shape[1] - 2))
        ti, tj = i - i0, j - j0
        return float(
            (1 - ti) * (1 - tj) * Z[i0, j0]
            + (1 - ti) * tj * Z[i0, j0 + 1]
            + ti * (1 - tj) * Z[i0 + 1, j0]
            + ti * tj * Z[i0 + 1, j0 + 1]
        )

    return {
        "xy": xy,
        "U": U,
        "xx": xx,
        "yy": yy,
        "Z": Z,
        "Z_disp": Z_disp,
        "Z_path": Z_path,
        "weight": weight,
        "U_func": U_display,
        "U_func_raw": U_func_raw,
        "fit_xy": fit_xy,
        "fit_U": fit_U,
        "fit_method": method,
        "bbox_percentile": (0.5, 99.5),
        "bbox": (float(x0), float(x1), float(y0), float(y1)),
    }


def _try_flow_lap(
    start,
    end,
    U_func,
    *,
    field_dynamic_range: float,
    n_points: int = 64,
) -> dict | None:
    try:
        from flow_space_lap import compute_flow_space_lap_path

        flow = compute_flow_space_lap_path(
            np.asarray(start, float),
            np.asarray(end, float),
            U_func,
            n_points=n_points,
            gamma=0.1,
            use_ensemble=False,
        )
        path = np.asarray(flow.get("path", []), float)
        if not flow.get("success") or flow.get("path_degenerate") or len(path) < 4:
            return None
        path = _smooth_polyline(path, n_out=max(n_points, 100))
        U_path = np.asarray([float(np.asarray(U_func(p)).reshape(-1)[0]) for p in path], float)
        out = {
            "path_xy": path,
            "U_path": U_path,
            "action": float(flow.get("total_action", np.nan)),
            "method": "flow_lap",
            "success": True,
            "degenerate": False,
        }
        # Flow-LAP has no grid Hessian context here; still compute 1D barrier.
        dyn = field_dynamic_range
        ts, has_barrier, barrier_height = _detect_transition_state(U_path, dyn)
        out["ts"] = ts
        out["has_barrier"] = bool(has_barrier)
        out["barrier_height"] = float(barrier_height)
        out["barrier_thresh"] = 0.05 * max(dyn, 1e-4)
        out["grad_norm"] = float("nan")
        out["det_H"] = float("nan")
        out["grad_thresh"] = float("nan")
        out["is_strict_saddle"] = False
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"    flow-LAP failed: {exc}", flush=True)
        return None


def _resolve_path(field, start, end, *, try_flow: bool = False) -> dict:
    if try_flow:
        flow = _try_flow_lap(
            start,
            end,
            field["U_func_raw"],
            field_dynamic_range=_field_dynamic_range(field["Z_path"]),
        )
        if flow is not None:
            # If we have a grid, upgrade with strict 2D saddle metrics
            flow = _attach_path_saddle_metrics(
                flow, field["xx"], field["yy"], field["Z_path"]
            )
            print(
                f"    flow-LAP OK action={flow['action']:.4g} "
                f"barrier={flow['has_barrier']} "
                f"strict_saddle={flow.get('is_strict_saddle')}",
                flush=True,
            )
            return flow
        print("    flow-LAP unused; falling back to field geodesic", flush=True)
    return _continuous_field_path(
        field["xx"], field["yy"], field["Z_path"], start, end, field["U_func"]
    )


def _infer_data_driven_edges(
    adata: ad.AnnData,
    field: dict,
    cell_type_col: str = "cell.type",
    stage_col: str = "stage",
    *,
    connectivity_threshold: float = 0.02,
    max_outgoing: int = 2,
    terminal_sinks: set[str] | None = None,
) -> list[tuple[str, str, str, None]]:
    """Infer forward subtype edges with stage/pseudotime and sink protection.

    Topology comes from cross-subtype neighbors in the observed UMAP manifold.
    Direction must not move backward in either experimental stage or intrinsic
    pseudotime. Terminal sinks have out-degree zero.
    """
    obs = adata.obs
    labels = obs[cell_type_col].astype(str).to_numpy()
    xy = np.asarray(field["xy"], float)
    types = [t for t in pd.unique(labels) if np.sum(labels == t) >= 5]
    if len(types) < 2:
        return []
    if terminal_sinks is None:
        terminal_sinks = {"AT1 cells", "Resolution macrophages"}
    terminal_sinks = set(terminal_sinks)

    # Intrinsic ordering (required).
    time_key = None
    time = None
    for key in ("pseudotime", "latent_time", "time"):
        if key in obs:
            candidate = pd.to_numeric(obs[key], errors="coerce").to_numpy(float)
            if np.isfinite(candidate).sum() >= 0.8 * len(candidate):
                time_key, time = key, candidate
                break
    if time is None:
        raise ValueError(
            "Automatic edge inference requires pseudotime, latent_time, or time."
        )

    # Experimental-stage ordering (used jointly when sufficiently complete).
    stage_days = np.full(len(obs), np.nan, dtype=float)
    stage_available = False
    if stage_col in obs:
        stage_num = (
            obs[stage_col]
            .astype(str)
            .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
        )
        stage_days = pd.to_numeric(stage_num, errors="coerce").to_numpy(float)
        stage_available = np.isfinite(stage_days).sum() >= 0.8 * len(stage_days)

    # Adaptive clocks: terminal sinks use late-phase (P75) so Day-0 healthy
    # cells do not pull their center of mass backward and block ADI→AT1.
    median_time: dict[str, float] = {}
    median_stage: dict[str, float] = {}
    for t in types:
        mask = labels == t
        t_pseudo = time[mask]
        if not np.isfinite(t_pseudo).any():
            continue
        if t in terminal_sinks:
            median_time[t] = float(np.nanpercentile(t_pseudo, 75))
            if stage_available:
                t_stage = stage_days[mask]
                if np.isfinite(t_stage).any():
                    median_stage[t] = float(np.nanpercentile(t_stage, 75))
        else:
            median_time[t] = float(np.nanmedian(t_pseudo))
            if stage_available:
                t_stage = stage_days[mask]
                if np.isfinite(t_stage).any():
                    median_stage[t] = float(np.nanmedian(t_stage))

    types = [
        t
        for t in types
        if t in median_time and (not stage_available or t in median_stage)
    ]
    if len(types) < 2:
        return []

    n_neighbors = min(16, len(xy) - 1)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(xy)
    neighbor_idx = nn.kneighbors(xy, return_distance=False)[:, 1:]
    counts = {t: int(np.sum(labels == t)) for t in types}
    cross = {}
    for src in types:
        src_idx = np.where(labels == src)[0]
        neighbor_labels = labels[neighbor_idx[src_idx]]
        for dst in types:
            if src == dst:
                continue
            cross[(src, dst)] = int(np.sum(neighbor_labels == dst))

    time_range = max(median_time.values()) - min(median_time.values())
    min_time_step = max(1e-8, 0.01 * time_range)
    stage_values = [median_stage[t] for t in types] if stage_available else [0.0]
    stage_range = max(stage_values) - min(stage_values)
    min_stage_step = max(1e-8, 0.01 * stage_range)
    candidates = []
    for i, src in enumerate(types):
        for dst in types[i + 1 :]:
            # Symmetric kNN mixing fraction, analogous to a coarse PAGA weight.
            denom = n_neighbors * (counts[src] + counts[dst])
            connectivity = (
                cross.get((src, dst), 0) + cross.get((dst, src), 0)
            ) / max(denom, 1)
            if connectivity < connectivity_threshold:
                continue

            dt_time = median_time[dst] - median_time[src]
            dt_stage = (
                median_stage[dst] - median_stage[src] if stage_available else 0.0
            )

            # Experimental stage defines orientation when separated; otherwise
            # intrinsic pseudotime resolves ties.
            if stage_available and abs(dt_stage) > min_stage_step:
                forward_src, forward_dst = (
                    (src, dst) if dt_stage > 0 else (dst, src)
                )
            elif abs(dt_time) > min_time_step:
                forward_src, forward_dst = (
                    (src, dst) if dt_time > 0 else (dst, src)
                )
            else:
                continue

            # Conjunctive dual gate: neither clock may run backward.
            pseudo_forward = (
                median_time[forward_dst] - median_time[forward_src]
            )
            stage_forward = (
                median_stage[forward_dst] - median_stage[forward_src]
                if stage_available
                else 0.0
            )
            if pseudo_forward < -min_time_step:
                continue
            if stage_available and stage_forward < -min_stage_step:
                continue
            if forward_src in terminal_sinks:
                continue

            candidates.append(
                (
                    forward_src,
                    forward_dst,
                    float(connectivity),
                    abs(float(pseudo_forward)),
                )
            )

    # Keep the strongest local forward branches rather than all N(N-1) pairs.
    selected = []
    sort_key = (
        (lambda t: (median_stage[t], median_time[t]))
        if stage_available
        else median_time.get
    )
    for src in sorted(types, key=sort_key):
        if src in terminal_sinks:
            continue
        outgoing = [c for c in candidates if c[0] == src]
        outgoing.sort(key=lambda x: (-x[2], x[3], median_time[x[1]]))
        selected.extend(outgoing[:max_outgoing])

    # Deduplicate and emit the same tuple shape as curated edges.
    edges = []
    seen = set()
    for src, dst, connectivity, _ in selected:
        if (src, dst) in seen:
            continue
        seen.add((src, dst))
        label = f"{SHORT.get(src, src)}→{SHORT.get(dst, dst)}"
        edges.append((src, dst, label, None))
        stage_text = (
            f", stage={median_stage[src]:.1f}→{median_stage[dst]:.1f}"
            if stage_available
            else ""
        )
        print(
            f"  auto edge {label}: W={connectivity:.3f}, "
            f"{time_key}={median_time[src]:.3f}→{median_time[dst]:.3f}"
            f"{stage_text}",
            flush=True,
        )
    return edges


def _maybe_analyzer(adata: ad.AnnData) -> NonEquilibriumCellFateLandscape | None:
    """Optional: validate AnnData + NonEquilibriumCellFateLandscape wiring."""
    try:
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            analyzer = NonEquilibriumCellFateLandscape(
                adata,
                potential_key=POTENTIAL_KEY,
                embedding_2d_key="X_umap",
                potential_transform="none",
                lap_force_mode="gradient",
                use_embedding_velocity=False,
            )
        print(f"  NonEquilibriumCellFateLandscape OK ({adata.n_obs} cells)", flush=True)
        return analyzer
    except Exception as exc:  # noqa: BLE001
        print(f"  analyzer init skipped: {exc}", flush=True)
        return None


# --------------------------------------------------------------------------- plot
def _style_3d_ax(ax, *, z_floor: float, z_top: float):
    ax.set_xlabel("UMAP 1", labelpad=8, fontsize=10, color=INK)
    ax.set_ylabel("UMAP 2", labelpad=8, fontsize=10, color=INK)
    ax.set_zlabel(r"$U_{\mathrm{rel}}$", labelpad=8, fontsize=10, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    ax.set_zlim(z_floor, z_top)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor("#D0D5DD")
        axis.pane.set_alpha(0.15)
    ax.grid(False)


def _draw_3d_landscape(
    ax,
    field: dict,
    paths: list[dict],
    wells: list[dict],
    *,
    title: str,
    elev: float = 28,
    azim: float = -58,
):
    xx, yy = field["xx"], field["yy"]
    Z = np.asarray(field["Z"], float)
    w = np.asarray(field["weight"], float)
    finite = np.isfinite(Z) & (w >= 0.20)
    if not np.any(finite):
        raise RuntimeError("empty display field")

    # Crop to support bbox so the mesh has no exterior walls ("curtains")
    ys, xs = np.where(finite)
    i0, i1 = max(0, ys.min() - 2), min(Z.shape[0], ys.max() + 3)
    j0, j1 = max(0, xs.min() - 2), min(Z.shape[1], xs.max() + 3)
    xx, yy, Z, w, finite = (
        xx[i0:i1, j0:j1],
        yy[i0:i1, j0:j1],
        Z[i0:i1, j0:j1],
        w[i0:i1, j0:j1],
        finite[i0:i1, j0:j1],
    )

    lo, hi = np.nanpercentile(Z[finite], [5, 95])
    fill = float(np.nanmedian(Z[finite]))
    Z_plot = np.where(np.isfinite(Z), Z, fill)
    # Soft rim: keep Z continuous but fade alpha — no NaN walls
    alpha = np.clip((w - 0.12) / 0.28, 0.0, 1.0)
    ls = LightSource(azdeg=315, altdeg=38)
    rgb = ls.shade(Z_plot, cmap=SURF_CMAP, vmin=lo, vmax=hi, blend_mode="soft")
    rgba = np.concatenate([rgb[..., :3], (alpha * 0.90)[..., None]], axis=-1)

    ax.plot_surface(
        xx,
        yy,
        Z_plot,
        facecolors=rgba,
        linewidth=0,
        antialiased=True,
        shade=False,
        rstride=max(1, xx.shape[0] // 90),
        cstride=max(1, xx.shape[1] // 90),
        zorder=1,
    )
    z_floor = float(np.nanmin(Z[finite])) - 0.45 * (hi - lo)
    z_top = float(np.nanmax(Z[finite])) + 0.10 * (hi - lo)

    Z_cont = np.ma.array(Z_plot, mask=alpha < 0.35)
    ax.contour(
        xx,
        yy,
        Z_cont,
        zdir="z",
        offset=z_floor,
        levels=np.linspace(lo, hi, 18),
        cmap=SURF_CMAP,
        linewidths=0.45,
        alpha=0.55,
        vmin=lo,
        vmax=hi,
        zorder=0,
    )

    path_color = "#C1121F"
    for i, p in enumerate(paths):
        xy = p["path_xy"]
        Uz = p["U_path"]
        ax.plot(xy[:, 0], xy[:, 1], Uz, color="white", lw=3.0, zorder=8, solid_capstyle="round")
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            Uz,
            color=path_color,
            lw=1.9,
            zorder=9,
            solid_capstyle="round",
            label=p.get("label", "LAP"),
        )
        # Mid-path direction chevron (same color as path)
        head = _midpath_chevron_xyz(xy, Uz, xx, yy, lo, hi, frac=0.55)
        if head is not None:
            ax.plot(
                head[:, 0],
                head[:, 1],
                head[:, 2],
                color=path_color,
                lw=1.8,
                zorder=10,
                solid_capstyle="round",
            )
        if p.get("has_barrier", False) and p.get("is_strict_saddle", True):
            ts = int(p["ts"])
            ax.scatter(
                [xy[ts, 0]],
                [xy[ts, 1]],
                [Uz[ts]],
                s=85,
                c="#F4D35E",
                marker="*",
                edgecolors="black",
                linewidths=0.45,
                zorder=12,
                depthshade=False,
            )
            if p.get("ts_label"):
                ax.text(
                    xy[ts, 0],
                    xy[ts, 1],
                    Uz[ts],
                    f"  {p['ts_label']} (ΔU={p['barrier_height']:.2f})",
                    fontsize=7.5,
                    color="#0F172A",
                    fontweight="bold",
                    zorder=13,
                )

    halo = [pe.withStroke(linewidth=2.2, foreground="white")]
    for well in wells:
        pos = np.asarray(well["xy"], float)
        u = float(field["U_func"](pos))
        ax.scatter(
            [pos[0]],
            [pos[1]],
            [u],
            s=46,
            c=well.get("color", "#08306B"),
            marker="o",
            edgecolors="white",
            linewidths=0.7,
            zorder=11,
            depthshade=False,
        )
        ax.text(
            pos[0],
            pos[1],
            u,
            f"  {well['label']}",
            fontsize=7.5,
            color="#0F172A",
            fontweight="bold",
            path_effects=halo,
            zorder=13,
        )

    _style_3d_ax(ax, z_floor=z_floor, z_top=z_top)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK, pad=8)
    return Normalize(vmin=lo, vmax=hi)


def _save_landscape_figure(
    *,
    field: dict,
    paths: list[dict],
    wells: list[dict],
    out: Path,
    title: str,
    elev: float,
    azim: float,
    footnote: str,
    legend: bool = True,
) -> Path:
    fig = plt.figure(figsize=(11.2, 8.4), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    norm = _draw_3d_landscape(
        ax,
        field,
        paths,
        wells,
        title=title,
        elev=elev,
        azim=azim,
    )
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=SURF_CMAP)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, aspect=16, pad=0.06)
    cbar.set_label(r"$U_{\mathrm{rel}}$", fontsize=10)
    if legend:
        ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    fig.text(0.5, 0.02, footnote, ha="center", fontsize=7.5, color="#667085", style="italic")
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(PROTO / "figures" / out.name, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}", flush=True)
    return out


def _path_summary(parent: str, src: str, dst: str, path: dict, kind: str) -> dict:
    return {
        "parent": parent,
        "src": src,
        "dst": dst,
        "kind": kind,
        "method": path["method"],
        "n_points": int(len(path["U_path"])),
        "ts_idx": int(path["ts"]),
        "relative_ts": float(path["ts"] / max(len(path["U_path"]) - 1, 1)),
        "U_start": float(path["U_path"][0]),
        "U_end": float(path["U_path"][-1]),
        "U_ts": float(path["U_path"][path["ts"]]),
        "barrier_from_start": float(path["U_path"][path["ts"]] - path["U_path"][0]),
        "has_barrier": bool(path.get("has_barrier", False)),
        "barrier_height": float(path.get("barrier_height", np.nan)),
        "barrier_thresh": float(path.get("barrier_thresh", np.nan)),
        "is_strict_saddle": bool(path.get("is_strict_saddle", False)),
        "det_H": float(path.get("det_H", np.nan)),
        "grad_norm": float(path.get("grad_norm", np.nan)),
        "path_action": float(path["action"]),
    }


# --------------------------------------------------------------------------- panels
def run_alv(
    *, try_flow: bool = False, auto_edges: bool = False, field_method: str = "rbf"
) -> tuple[pd.DataFrame, Path]:
    print("===== Alveolar epithelium 3D landscape =====", flush=True)
    adata = _load_parent("alv_epithelium", ALV_TYPES)
    _maybe_analyzer(adata)
    field = _build_field(
        adata, n_grid=150, max_fit=None, smooth_sigma=3.8, method=field_method
    )
    xy, U = field["xy"], field["U"]
    labels = adata.obs["cell.type"].astype(str).to_numpy()

    wells_xy = {
        t: _spatial_median_centroid(xy, labels == t)
        for t in ALV_TYPES
        if (labels == t).sum() >= 5
    }
    edges = (
        _infer_data_driven_edges(adata, field)
        if auto_edges
        else [
            ("AT2 cells", "AT1 cells", "AT2→AT1", "Krt8+ ADI saddle"),
            ("Activated AT2 cells", "AT1 cells", "Act.AT2→AT1", None),
            ("AT2 cells", "Krt8 ADI", "AT2→ADI", None),
            ("Krt8 ADI", "AT1 cells", "ADI→AT1", None),
        ]
    )
    paths = []
    rows = []
    for src, dst, label, ts_lab in edges:
        if src not in wells_xy or dst not in wells_xy:
            continue
        if not np.all(np.isfinite(wells_xy[src])) or not np.all(np.isfinite(wells_xy[dst])):
            continue
        print(f"  path {label}…", flush=True)
        p = _resolve_path(field, wells_xy[src], wells_xy[dst], try_flow=try_flow)
        p["label"] = label
        p["ts_label"] = ts_lab
        paths.append(p)
        rows.append(_path_summary("alv_epithelium", src, dst, p, label))
        print(
            f"    method={p['method']} has_barrier={p['has_barrier']} "
            f"strict_saddle={p.get('is_strict_saddle')} "
            f"detH={p.get('det_H', float('nan')):.3g} "
            f"ΔU={p['barrier_height']:.3f} "
            f"action={p['action']:.3f}",
            flush=True,
        )

    wells = [
        {"xy": wells_xy["AT2 cells"], "label": "AT2 well", "color": "#08306B"},
        {"xy": wells_xy["AT1 cells"], "label": "AT1 well", "color": "#238B45"},
        {"xy": wells_xy["Krt8 ADI"], "label": "Krt8+ ADI", "color": "#C1121F"},
    ]
    if "Activated AT2 cells" in wells_xy:
        wells.append(
            {"xy": wells_xy["Activated AT2 cells"], "label": "Act.AT2", "color": "#1D3557"}
        )

    footnote = (
        "XY: training UMAP · Z: potential_relative_type · path: FMM Eikonal "
        "(Dijkstra fallback) · saddle: ||∇U||≈0 & det(H)<0"
    )
    title = r"Alveolar epithelium · $U_{\mathrm{rel}}$ landscape with LAP"
    out_side = _save_landscape_figure(
        field=field,
        paths=paths,
        wells=wells,
        out=PANELS / "GSE141259_alv_3d_urel_landscape_lap.png",
        title=title,
        elev=30,
        azim=-52,
        footnote=footnote + " · side view",
    )
    _save_landscape_figure(
        field=field,
        paths=paths,
        wells=wells,
        out=PANELS / "GSE141259_alv_3d_urel_landscape_lap_top.png",
        title=title + r" (top view)",
        elev=78,
        azim=-52,
        footnote=footnote + " · top view",
    )
    return pd.DataFrame(rows), out_side


def run_mac(
    *, try_flow: bool = False, auto_edges: bool = False, field_method: str = "rbf"
) -> tuple[pd.DataFrame, Path]:
    print("===== Macrophages 3D landscape =====", flush=True)
    adata = _load_parent("macrophages", MAC_TYPES)
    _maybe_analyzer(adata)
    field = _build_field(
        adata, n_grid=140, max_fit=None, smooth_sigma=4.0, method=field_method
    )
    xy, U = field["xy"], field["U"]
    labels = adata.obs["cell.type"].astype(str).to_numpy()
    stages = adata.obs["stage"].astype(str).to_numpy()

    wells_xy = {}
    for t in MAC_TYPES:
        if (labels == t).sum() >= 8:
            wells_xy[t] = _spatial_median_centroid(xy, labels == t)
    wells_xy["D0"] = _spatial_median_centroid(xy, stages == "D0")
    wells_xy["D28"] = _spatial_median_centroid(xy, stages == "D28")

    edges = (
        _infer_data_driven_edges(adata, field)
        if auto_edges
        else [
            ("D0", "D28", "D0→D28 remodel", "peak remodel"),
            ("AM (PBS)", "Resolution macrophages", "AM(PBS)→Resol.", None),
            ("AM (Bleo)", "M2 macrophages", "AM(Bleo)→M2", "Arg1+ M2 ridge"),
            ("M2 macrophages", "Resolution macrophages", "M2→Resol.", None),
            ("Fn1+ macrophages", "Resolution macrophages", "Fn1+→Resol.", None),
        ]
    )
    paths = []
    rows = []
    for src, dst, label, ts_lab in edges:
        if src not in wells_xy or dst not in wells_xy:
            continue
        if not np.all(np.isfinite(wells_xy[src])) or not np.all(np.isfinite(wells_xy[dst])):
            continue
        print(f"  path {label}…", flush=True)
        p = _resolve_path(field, wells_xy[src], wells_xy[dst], try_flow=try_flow)
        p["label"] = label
        p["ts_label"] = ts_lab
        paths.append(p)
        rows.append(_path_summary("macrophages", src, dst, p, label))
        print(
            f"    method={p['method']} has_barrier={p['has_barrier']} "
            f"strict_saddle={p.get('is_strict_saddle')} "
            f"detH={p.get('det_H', float('nan')):.3g} "
            f"ΔU={p['barrier_height']:.3f} "
            f"action={p['action']:.3f}",
            flush=True,
        )

    wells = [
        {"xy": wells_xy["D0"], "label": "D0 well", "color": "#08306B"},
        {"xy": wells_xy["D28"], "label": "D28 well", "color": "#238B45"},
    ]
    if "M2 macrophages" in wells_xy:
        wells.append({"xy": wells_xy["M2 macrophages"], "label": "Arg1+ M2", "color": "#C1121F"})
    if "Resolution macrophages" in wells_xy:
        wells.append(
            {
                "xy": wells_xy["Resolution macrophages"],
                "label": "Mfge8+ Resol.",
                "color": "#2A9D8F",
            }
        )

    footnote = (
        "XY: training UMAP · Z: potential_relative_type · paths: curated priors with unequal support; "
        "not lineage"
    )
    title = r"Macrophages · curated $U_{\mathrm{rel}}$ arrows (unequal support)"
    out_side = _save_landscape_figure(
        field=field,
        paths=paths[:4],
        wells=wells,
        out=PANELS / "GSE141259_mac_3d_urel_landscape_lap.png",
        title=title,
        elev=28,
        azim=-62,
        footnote=footnote + " · side view",
    )
    _save_landscape_figure(
        field=field,
        paths=paths[:4],
        wells=wells,
        out=PANELS / "GSE141259_mac_3d_urel_landscape_lap_top.png",
        title=title + r" (top view)",
        elev=78,
        azim=-62,
        footnote=footnote + " · top view",
    )
    return pd.DataFrame(rows), out_side


def run_overview(
    *,
    elev_alv: float = 30,
    azim_alv: float = -52,
    elev_mac: float = 28,
    azim_mac: float = -62,
    out_name: str = "GSE141259_mac_alv_3d_urel_landscape_overview.png",
    view_tag: str = "side",
    auto_edges: bool = False,
) -> Path:
    """Side-by-side Alv | Mac overview at a fixed camera."""
    print(f"===== Overview dual 3D panel ({view_tag}) =====", flush=True)
    fig = plt.figure(figsize=(14.5, 6.6), facecolor="white")

    adata_a = _load_parent("alv_epithelium", ALV_TYPES)
    field_a = _build_field(adata_a, n_grid=120, max_fit=None, smooth_sigma=3.8)
    xy, U = field_a["xy"], field_a["U"]
    labels = adata_a.obs["cell.type"].astype(str).to_numpy()
    wells_a = {
        t: _spatial_median_centroid(xy, labels == t)
        for t in ALV_TYPES
        if (labels == t).sum() >= 5
    }
    paths_a = []
    edges_a = (
        _infer_data_driven_edges(adata_a, field_a)
        if auto_edges
        else [
            ("AT2 cells", "AT1 cells", "AT2→AT1", "ADI saddle"),
            ("Krt8 ADI", "AT1 cells", "ADI→AT1", None),
        ]
    )
    for src, dst, lab, ts_lab in edges_a:
        if src not in wells_a or dst not in wells_a:
            continue
        p = _resolve_path(field_a, wells_a[src], wells_a[dst], try_flow=False)
        p["label"], p["ts_label"] = lab, ts_lab
        paths_a.append(p)
    ax0 = fig.add_subplot(121, projection="3d")
    _draw_3d_landscape(
        ax0,
        field_a,
        paths_a,
        [
            {"xy": wells_a["AT2 cells"], "label": "AT2", "color": "#08306B"},
            {"xy": wells_a["AT1 cells"], "label": "AT1", "color": "#238B45"},
            {"xy": wells_a["Krt8 ADI"], "label": "ADI", "color": "#C1121F"},
        ],
        title="Alveolar epithelium",
        elev=elev_alv,
        azim=azim_alv,
    )

    adata_m = _load_parent("macrophages", MAC_TYPES)
    field_m = _build_field(adata_m, n_grid=110, max_fit=None, smooth_sigma=4.0)
    xy, U = field_m["xy"], field_m["U"]
    labels = adata_m.obs["cell.type"].astype(str).to_numpy()
    stages = adata_m.obs["stage"].astype(str).to_numpy()
    d0 = _spatial_median_centroid(xy, stages == "D0")
    d28 = _spatial_median_centroid(xy, stages == "D28")
    m2 = _spatial_median_centroid(xy, labels == "M2 macrophages")
    resol = _spatial_median_centroid(xy, labels == "Resolution macrophages")
    wells_m = {
        t: _spatial_median_centroid(xy, labels == t)
        for t in MAC_TYPES
        if (labels == t).sum() >= 8
    }
    paths_m = []
    edges_m = (
        _infer_data_driven_edges(adata_m, field_m)
        if auto_edges
        else [
            ("D0", "D28", "D0→D28", "remodel peak"),
            ("M2 macrophages", "Resolution macrophages", "M2→Resol.", None),
        ]
    )
    wells_m_with_stage = {**wells_m, "D0": d0, "D28": d28}
    for src, dst, lab, ts_lab in edges_m:
        if src not in wells_m_with_stage or dst not in wells_m_with_stage:
            continue
        src_xy, dst_xy = wells_m_with_stage[src], wells_m_with_stage[dst]
        p = _resolve_path(field_m, src_xy, dst_xy, try_flow=False)
        p["label"], p["ts_label"] = lab, ts_lab
        paths_m.append(p)
    ax1 = fig.add_subplot(122, projection="3d")
    _draw_3d_landscape(
        ax1,
        field_m,
        paths_m,
        [
            {"xy": d0, "label": "D0", "color": "#08306B"},
            {"xy": d28, "label": "D28", "color": "#238B45"},
            {"xy": m2, "label": "M2", "color": "#C1121F"},
            {"xy": resol, "label": "Resol.", "color": "#2A9D8F"},
        ],
        title="Macrophages",
        elev=elev_mac,
        azim=azim_mac,
    )

    fig.suptitle(
        rf"GSE141259 nonequilibrium $U_{{\mathrm{{rel}}}}$ landscapes (3D, {view_tag} view)",
        fontsize=13,
        fontweight="bold",
        color=INK,
        y=0.98,
    )
    out = PANELS / out_name
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(PROTO / "figures" / out.name, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--try-flow", action="store_true", help="Attempt flow-space LAP before geodesic")
    ap.add_argument(
        "--auto-edges",
        action="store_true",
        help="Infer forward subtype paths from pseudotime and kNN topology",
    )
    ap.add_argument(
        "--field-method",
        choices=("rbf", "tps"),
        default="rbf",
        help="Scalar-field interpolator: rbf (default) or thin-plate spline (tps)",
    )
    ap.add_argument("--skip-overview", action="store_true")
    args = ap.parse_args()

    alv_sum, alv_fig = run_alv(
        try_flow=args.try_flow,
        auto_edges=args.auto_edges,
        field_method=args.field_method,
    )
    mac_sum, mac_fig = run_mac(
        try_flow=args.try_flow,
        auto_edges=args.auto_edges,
        field_method=args.field_method,
    )
    summary = pd.concat([alv_sum, mac_sum], ignore_index=True)
    sum_path = TAB / "GSE141259_mac_alv_3d_landscape_lap_summary.csv"
    summary.to_csv(sum_path, index=False)

    overview = None
    overview_top = None
    if not args.skip_overview:
        overview = run_overview(
            elev_alv=30,
            azim_alv=-52,
            elev_mac=28,
            azim_mac=-62,
            out_name="GSE141259_mac_alv_3d_urel_landscape_overview.png",
            view_tag="side",
            auto_edges=args.auto_edges,
        )
        overview_top = run_overview(
            elev_alv=78,
            azim_alv=-52,
            elev_mac=78,
            azim_mac=-62,
            out_name="GSE141259_mac_alv_3d_urel_landscape_overview_top.png",
            view_tag="top",
            auto_edges=args.auto_edges,
        )

    audit = {
        "potential_key": POTENTIAL_KEY,
        "embedding": "training_umap.npz X_umap",
        "path_method_default": "eikonal_fmm_with_dijkstra_fallback",
        "field_method": args.field_method,
        "try_flow": bool(args.try_flow),
        "edge_mode": "auto_pseudotime_knn" if args.auto_edges else "curated",
        "barrier_rule": (
            "1D: interior path maximum with barrier_height > 5% of field P95-P5; "
            "2D strict saddle: ||∇U|| ≤ τ_grad and det(H)<0"
        ),
        "alv_figure": str(alv_fig),
        "alv_figure_top": str(PANELS / "GSE141259_alv_3d_urel_landscape_lap_top.png"),
        "mac_figure": str(mac_fig),
        "mac_figure_top": str(PANELS / "GSE141259_mac_3d_urel_landscape_lap_top.png"),
        "overview_figure": str(overview) if overview else "",
        "overview_figure_top": str(overview_top) if overview_top else "",
        "summary_table": str(sum_path),
        "n_paths": int(len(summary)),
        "n_paths_with_barrier": int(summary["has_barrier"].sum()),
        "n_paths_strict_saddle": int(summary["is_strict_saddle"].sum())
        if "is_strict_saddle" in summary.columns
        else 0,
        "paths": summary.to_dict(orient="records"),
        "notes": (
            "Path solver prefers scikit-fmm Eikonal travel_time + RK2 backtrace; "
            "falls back to weighted-grid Dijkstra if skfmm is unavailable. "
            "Field interpolator: rbf or tps (--field-method). "
            "Saddle marks require both 1D barrier and strict 2D Hessian signature."
        ),
    }
    (PROTO / "GSE141259_mac_alv_3d_landscape_audit.json").write_text(
        json.dumps(audit, indent=2, default=str)
    )
    (TAB / "GSE141259_mac_alv_3d_landscape_audit.json").write_text(
        json.dumps(audit, indent=2, default=str)
    )
    print(f"Wrote {sum_path}", flush=True)
    print("\n=== 3D LAP summary ===")
    cols = [
        c
        for c in [
            "parent",
            "kind",
            "method",
            "has_barrier",
            "is_strict_saddle",
            "det_H",
            "grad_norm",
            "barrier_height",
            "barrier_thresh",
            "ts_idx",
            "relative_ts",
            "barrier_from_start",
            "path_action",
            "U_ts",
        ]
        if c in summary.columns
    ]
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
