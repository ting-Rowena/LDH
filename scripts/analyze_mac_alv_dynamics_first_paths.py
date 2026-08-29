#!/usr/bin/env python3
"""Dynamics-first subtype paths for Macrophages and Alveolar epithelium.

Reverses the previous pipeline: landscape first, time composition last.

Potential fields (do not mix)
-----------------------------
potential
    U(z, t) = U0(z) + ε φ(z, t). Injury-clock mixed. Not used here.
potential_stationary
    U0(z). Time-invariant quasi-potential. Landscape geometry.
potential_relative_type
    z-score of U0 within the training major type (here: annotation parent).
    Ranking of wells vs slope states inside a parent, and the scaled field
    for graph-path weights (raw U0 span inside a parent is ~2e-4, too flat
    to modulate costs). This is still U0, in parent-internal units.
There is no stored relative of time-dependent `potential`.

Pipeline
--------
1. Attractors: low mean potential_relative_type (+ local density), not D28 abundance.
2. Candidate edges: U0-weighted (via U_rel) kNN shortest paths on parent-internal latent PCA; keep MST + locally cheap.
3. Panels C/D: within-parent expression HVG→PCA→neighbors→Leiden→UMAP for
   display; the same kept edges (from latent PCA2) are overlaid.
4. Direction: sign of ΔU_rel (climb vs relax) and forward/back action ratio.
5. Intermediates: high U_rel and/or occupancy on the well–well geodesic; inverted-U is corroboration.
6. Composition: whether the injury time course actually used that edge.

Outputs
  output_file/mac_landscape_audit/GSE141259_mac_alv_dynamics_first_*.csv
  <CK>/analysis_protocol_GSE141259/figures/GSE141259_mac_alv_dynamics_first_paths.png
  <CK>/analysis_protocol_GSE141259/figures/GSE141259_mac_alv_dynamics_first_action_heatmaps.png
  <CK>/analysis_protocol_GSE141259/GSE141259_mac_alv_dynamics_first_summary.json
"""

from __future__ import annotations

import heapq
import itertools
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_pipeline import PROJECT_ROOT, recommended_checkpoint_dir  # noqa: E402
from beautify_checkpoint_figures import _clean_umap_ax, _pt_size  # noqa: E402
from panel_style import (  # noqa: E402
    AXIS_LABEL_SIZE,
    TICK_LABEL_SIZE,
    apply_panel_title_rc,
    set_panel_title,
)
from analysis_protocol_utils import fast_wilcoxon_deg  # noqa: E402
from celltype_analysis import DATASET_REGISTRY, load_annotated_adata  # noqa: E402
from plot_gse141259_why_mac_alv_focus import (  # noqa: E402
    _load as _load_why_mac_alv,
    paint_landscape,
    paint_subtype_urel,
)
from plot_utils import ACCENT_DN, ACCENT_UP, GRID, INK, MUTED, configure_headless  # noqa: E402

configure_headless()
apply_panel_title_rc()

plt.rcParams.update(
    {
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

CK = Path(recommended_checkpoint_dir("GSE141259"))
PROTO = CK / "analysis_protocol_GSE141259"
TAB = PROJECT_ROOT / "output_file" / "mac_landscape_audit"
PANELS = PROTO / "figures"
for p in (TAB, PANELS, PROTO):
    p.mkdir(parents=True, exist_ok=True)

STAGES = ["D0", "D3", "D7", "D10", "D14", "D21", "D28"]
STAGE_DAYS = np.array([0.0, 3.0, 7.0, 10.0, 14.0, 21.0, 28.0])

ALV_TYPES = ["AT2 cells", "Activated AT2 cells", "Krt8 ADI", "AT1 cells"]
MAC_TYPES = [
    "AM (PBS)",
    "AM (Bleo)",
    "M2 macrophages",
    "Resolution macrophages",
    "Fn1+ macrophages",
    "Cd163-/Cd11c+ IMs",
    "Cd163+/Cd11c- IMs",
]
PARENTS = {
    "alv_epithelium": ALV_TYPES,
    "macrophages": MAC_TYPES,
}

_PAL = pd.read_csv(CK / "figures" / "GSE141259_umap_hierarchical_palette.csv")
_FMAP = pd.read_csv(CK / "figures" / "GSE141259_metacelltype_formal_label_mapping.csv")
FORMAL = dict(zip(_FMAP["metacelltype"].astype(str), _FMAP["formal_label"].astype(str)))
ALV_LAB = FORMAL["alv_epithelium"]
MAC_LAB = FORMAL["macrophages"]
SUB_COL = dict(zip(_PAL["cell.type"].astype(str), _PAL["subtype_color"].astype(str)))
MAC_COL = str(_PAL.loc[_PAL.metacelltype == "macrophages", "parent_color"].iloc[0])
ALV_COL = str(_PAL.loc[_PAL.metacelltype == "alv_epithelium", "parent_color"].iloc[0])

COL_CLIMB = ACCENT_UP
COL_RELAX = ACCENT_DN
COL_FLAT = "#8b949e"
COL_WELL_BG = "#eef3f6"
COL_SLOPE_BG = "#faf6ef"

DEG_TABLE = TAB / "GSE141259_15type_D0_vs_D28_deg_counts.csv"
DEG_PADJ = 0.05
DEG_LOGFC = 0.25
DEG_MIN_CELLS = 10

KNN = 12
CORE_FRAC = 0.25
MIN_CORE = 8


def _count_sig_degs(deg: pd.DataFrame) -> int:
    if deg is None or deg.empty:
        return 0
    padj = deg["pval_adj"] if "pval_adj" in deg.columns else deg["pval"]
    lfc = deg["logfoldchange"]
    m = np.isfinite(padj.to_numpy(float)) & np.isfinite(lfc.to_numpy(float))
    m &= (padj.to_numpy(float) < DEG_PADJ) & (np.abs(lfc.to_numpy(float)) > DEG_LOGFC)
    return int(m.sum())


def ensure_annotation_d0_d28_deg_counts(*, force: bool = False) -> pd.DataFrame:
    """Per major type (annotation): significant DEG count for D0 vs D28."""
    if DEG_TABLE.exists() and not force:
        return pd.read_csv(DEG_TABLE)

    print("Computing per-annotation D0 vs D28 DEG counts…", flush=True)
    import scanpy as sc

    profile = DATASET_REGISTRY["GSE141259"]
    adata = load_annotated_adata(profile, str(CK))
    # Expression scale for Wilcoxon (counts → log1p CPM).
    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    sc.pp.log1p(adata)

    rows = []
    for ann in sorted(adata.obs["annotation"].astype(str).unique()):
        sub = adata[adata.obs["annotation"].astype(str) == ann].copy()
        n_d0 = int((sub.obs["stage"].astype(str) == "D0").sum())
        n_d28 = int((sub.obs["stage"].astype(str) == "D28").sum())
        status = "ok"
        n_deg = 0
        if n_d0 < DEG_MIN_CELLS or n_d28 < DEG_MIN_CELLS:
            status = "underpowered"
            n_deg = 0
        else:
            try:
                deg = fast_wilcoxon_deg(
                    sub,
                    group_key="stage",
                    group_1="D28",
                    group_2="D0",
                    n_top_genes=3000,
                    min_cells=DEG_MIN_CELLS,
                )
                n_deg = _count_sig_degs(deg)
            except Exception as exc:  # noqa: BLE001
                status = f"failed:{type(exc).__name__}"
                n_deg = 0
        rows.append(
            {
                "annotation": ann,
                "n_D0": n_d0,
                "n_D28": n_d28,
                "n_deg_D0_vs_D28": int(n_deg),
                "padj_cutoff": DEG_PADJ,
                "abs_logfc_cutoff": DEG_LOGFC,
                "status": status,
            }
        )
        print(f"  {ann}: n_DEG={n_deg} (D0={n_d0}, D28={n_d28}, {status})", flush=True)

    out = pd.DataFrame(rows).sort_values("n_deg_D0_vs_D28", ascending=False)
    out.to_csv(DEG_TABLE, index=False)
    print(f"Wrote {DEG_TABLE}", flush=True)
    return out
ASYM_LO, ASYM_HI = 1.0 / 1.4, 1.4
_HALO = [pe.withStroke(linewidth=3.0, foreground="white")]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_obs_latent() -> tuple[pd.DataFrame, np.ndarray]:
    """Return obs and X_latent in the same row order."""
    usecols = [
        "annotation",
        "cell.type",
        "stage",
        "time",
        "potential",
        "potential_stationary",
        "potential_relative_type",
        "plasticity_score",
    ]
    raw = pd.read_csv(CK / "obs.csv", nrows=0)
    index_col = 0
    cols = [raw.columns[0]] + [c for c in usecols if c in raw.columns]
    obs = pd.read_csv(CK / "obs.csv", usecols=cols, index_col=index_col, low_memory=False)
    obs.index = obs.index.astype(str)
    obs["annotation"] = obs["annotation"].astype(str)
    obs["cell.type"] = obs["cell.type"].astype(str)
    obs["U0"] = pd.to_numeric(obs["potential_stationary"], errors="coerce")
    obs["U"] = pd.to_numeric(obs["potential"], errors="coerce")
    obs["U_rel"] = pd.to_numeric(obs["potential_relative_type"], errors="coerce")
    obs["plast"] = pd.to_numeric(obs["plasticity_score"], errors="coerce")
    lat = np.load(CK / "latent_embeddings.npz", allow_pickle=True)
    idx = pd.Index(np.asarray(lat["index"]).astype(str))
    Z = np.asarray(lat["X_latent"], float)
    mapper = {b: i for i, b in enumerate(idx)}

    keep, rows = [], []
    for i, b in enumerate(obs.index):
        j = mapper.get(b)
        if j is not None:
            keep.append(i)
            rows.append(Z[j])
    obs = obs.iloc[np.asarray(keep)].copy()
    return obs, np.vstack(rows)


def _expression_parent_umap(
    parent: str,
    barcodes: np.ndarray,
    *,
    adata_full=None,
    force: bool = False,
    n_top_genes: int = 3000,
    n_pcs: int = 40,
    n_neighbors: int = 15,
    leiden_resolution: float = 0.6,
    min_dist: float = 0.3,
    seed: int = 17,
) -> np.ndarray:
    """Conventional expression UMAP for display only (HVG → PCA → neighbors → Leiden → UMAP).

    Paths / edges are NOT computed here; those use latent PCA2.
    """
    import scanpy as sc

    barcodes = np.asarray(barcodes).astype(str)
    cache = PROTO / f"GSE141259_{parent}_expr_hvg_umap.npz"
    if cache.exists() and not force:
        z = np.load(cache, allow_pickle=True)
        cached_bc = np.asarray(z["barcodes"]).astype(str)
        xy = np.asarray(z["X_umap"], float)
        if xy.shape[0] == len(cached_bc):
            mapper = {b: i for i, b in enumerate(cached_bc)}
            if all(b in mapper for b in barcodes):
                return xy[np.fromiter((mapper[b] for b in barcodes), dtype=int, count=len(barcodes))]

    print(
        f"  Building expression UMAP for {parent} "
        f"(HVG={n_top_genes}, PCS={n_pcs}, neighbors={n_neighbors})…",
        flush=True,
    )
    if adata_full is None:
        profile = DATASET_REGISTRY["GSE141259"]
        adata_full = load_annotated_adata(profile, str(CK))

    key = "annotation" if "annotation" in adata_full.obs.columns else "metacelltype"
    parent_mask = adata_full.obs[key].astype(str) == parent
    adata = adata_full[parent_mask].copy()
    ad_names = pd.Index(adata.obs_names.astype(str))
    missing = [b for b in barcodes if b not in ad_names]
    if missing:
        raise RuntimeError(
            f"{parent}: {len(missing)}/{len(barcodes)} barcodes missing from expression adata "
            f"(e.g. {missing[:3]})"
        )
    adata = adata[barcodes].copy()  # same order as latent / obs subset

    # Conventional Scanpy workflow within this parent (all genes → HVG).
    # Prefer seurat_v3 on counts; fall back to seurat on log-normalized data.
    used_hvg = "seurat"
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3")
        used_hvg = "seurat_v3"
    except Exception:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")
        used_hvg = "seurat"
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()
    sc.pp.scale(adata, max_value=10)
    n_comps = int(min(n_pcs, adata.n_vars - 1, adata.n_obs - 1, 50))
    sc.tl.pca(adata, n_comps=max(2, n_comps), svd_solver="arpack")
    n_pcs_use = int(min(n_comps, adata.obsm["X_pca"].shape[1]))
    sc.pp.neighbors(
        adata,
        n_neighbors=int(min(n_neighbors, max(2, adata.n_obs - 1))),
        n_pcs=n_pcs_use,
    )
    try:
        sc.tl.leiden(
            adata,
            resolution=leiden_resolution,
            key_added="leiden",
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
    except TypeError:
        # Older scanpy without flavor=igraph kwargs.
        sc.tl.leiden(adata, resolution=leiden_resolution, key_added="leiden")
    sc.tl.umap(adata, min_dist=min_dist, spread=1.0, random_state=seed)
    xy = np.asarray(adata.obsm["X_umap"][:, :2], dtype=float)
    leiden = adata.obs["leiden"].astype(str).to_numpy()

    np.savez_compressed(
        cache,
        X_umap=np.asarray(xy, np.float32),
        barcodes=barcodes,
        leiden=leiden,
        parent=np.asarray(parent),
        n_top_genes=np.asarray(n_top_genes),
        n_pcs=np.asarray(n_pcs_use),
        n_neighbors=np.asarray(n_neighbors),
        leiden_resolution=np.asarray(leiden_resolution),
        min_dist=np.asarray(min_dist),
        seed=np.asarray(seed),
        hvg_flavor=np.asarray(used_hvg),
        pipeline=np.asarray("hvg_normalize_log1p_scale_pca_neighbors_leiden_umap"),
    )
    print(f"  Cached expression UMAP → {cache.name}  ({xy.shape[0]} cells)", flush=True)
    return xy


def _stage_frac(obs: pd.DataFrame, types: list[str]) -> pd.DataFrame:
    if "stage" in obs.columns:
        st = obs["stage"].astype(str)
    else:
        st = "D" + pd.to_numeric(obs["time"], errors="coerce").astype(int).astype(str)
    rows = []
    for s in STAGES:
        sub = obs.loc[st == s]
        n = max(len(sub), 1)
        row = {t: float((sub["cell.type"] == t).sum() / n) for t in types}
        rows.append(row)
    return pd.DataFrame(rows, index=STAGES)


def composition_score(frac_src: np.ndarray, frac_dst: np.ndarray) -> dict:
    fs = np.nan_to_num(np.asarray(frac_src, float), nan=0.0)
    fd = np.nan_to_num(np.asarray(frac_dst, float), nan=0.0)
    i_src, i_dst = int(np.argmax(fs)), int(np.argmax(fd))
    ptp_s, ptp_d = float(fs.max() - fs.min()), float(fd.max() - fd.min())
    transfer = float(np.sum(np.clip(-np.diff(fs), 0, None) * np.clip(np.diff(fd), 0, None)))
    lag_ok = i_dst >= i_src
    anti = (i_dst < i_src - 1) and (ptp_d > 0.05) and (fs[i_dst] > fd[i_dst])
    if anti and transfer < 1e-4:
        strength = "anti"
    elif lag_ok and ptp_s >= 0.10 and ptp_d >= 0.15 and transfer >= 0.01:
        strength = "strong"
    elif lag_ok and ptp_s >= 0.05 and ptp_d >= 0.05 and transfer >= 0.002:
        strength = "medium"
    elif ptp_d >= 0.03 and ptp_s >= 0.03:
        strength = "weak"
    else:
        strength = "none"
    return {
        "src_peak_day": float(STAGE_DAYS[i_src]),
        "dst_peak_day": float(STAGE_DAYS[i_dst]),
        "transfer_proxy": transfer,
        "composition": strength,
    }


def inverted_u(frac: np.ndarray) -> tuple[bool, float, str]:
    f = np.nan_to_num(np.asarray(frac, float), nan=0.0)
    if f.max() < 1e-8:
        return False, 0.0, STAGES[0]
    i = int(np.argmax(f))
    peak = STAGES[i]
    left, right = float(f[0]), float(f[-1])
    mid = float(f[i])
    score = float((mid + 1e-8) / (0.5 * (left + right) + 1e-8))
    is_u = (0 < i < len(f) - 1) and mid >= 1.3 * max(left, right, 1e-8)
    return bool(is_u), score, peak


# ---------------------------------------------------------------------------
# Graph shortest path on parent PCA2, weights from scaled U0 (= U_rel)
# ---------------------------------------------------------------------------

def _core_idx(U: np.ndarray, mask: np.ndarray, *, frac=CORE_FRAC, min_n=MIN_CORE) -> np.ndarray:
    ix = np.where(mask)[0]
    if ix.size == 0:
        return ix
    k = min(ix.size, max(min_n, int(np.ceil(frac * ix.size))))
    order = np.argsort(U[ix])  # low U0 / U_rel first
    return ix[order[:k]]


def _build_graph(xy: np.ndarray, U_rel: np.ndarray, *, k=KNN):
    n = len(xy)
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm="auto").fit(xy)
    dist, idx = nn.kneighbors(xy)
    graph = [[] for _ in range(n)]
    for i in range(n):
        for jj, j in enumerate(idx[i, 1:]):
            d = float(dist[i, jj + 1])
            j = int(j)
            dU = float(U_rel[j] - U_rel[i])
            w_fwd = d * np.exp(max(dU, 0.0))
            w_rev = d * np.exp(max(-dU, 0.0))
            graph[i].append((j, w_fwd))
            graph[j].append((i, w_rev))
    return graph


def _dijkstra(graph, start: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(graph)
    dist = np.full(n, np.inf)
    prev = np.full(n, -1, dtype=int)
    dist[start] = 0.0
    heap = [(0.0, start)]
    while heap:
        du, u = heapq.heappop(heap)
        if du > dist[u]:
            continue
        for v, w in graph[u]:
            alt = du + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt, v))
    return dist, prev


def _path_nodes(prev: np.ndarray, start: int, end: int) -> np.ndarray:
    if start == end:
        return np.array([start], dtype=int)
    nodes = []
    cur = int(end)
    seen = 0
    while cur >= 0 and seen < len(prev) + 2:
        nodes.append(cur)
        if cur == start:
            break
        cur = int(prev[cur])
        seen += 1
    if not nodes or nodes[-1] != start:
        return np.array([], dtype=int)
    return np.asarray(nodes[::-1], dtype=int)


def _mst_undirected(types: list[str], cost: dict[tuple[str, str], float]) -> set[tuple[str, str]]:
    """Kruskal MST on min(fwd, bwd) costs. Returns undirected frozen pairs as sorted tuples."""
    parent = {t: t for t in types}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = []
    for a, b in itertools.combinations(types, 2):
        cab = cost.get((a, b), np.inf)
        cba = cost.get((b, a), np.inf)
        c = min(cab, cba)
        if np.isfinite(c):
            edges.append((c, a, b))
    edges.sort()
    kept = set()
    for _, a, b in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        parent[ra] = rb
        kept.add(tuple(sorted((a, b))))
    return kept


# ---------------------------------------------------------------------------
# Per-parent analysis
# ---------------------------------------------------------------------------

def analyze_parent(
    obs: pd.DataFrame,
    Z: np.ndarray,
    parent: str,
    types: list[str],
    *,
    adata_full=None,
) -> dict:
    m = obs["annotation"].eq(parent) & obs["cell.type"].isin(types)
    sub = obs.loc[m].copy()
    Zp = Z[m.to_numpy()]
    # Display: conventional expression HVG→PCA→neighbors→Leiden→UMAP (parent-internal).
    # Computation: latent PCA2 kNN graph + U_rel-weighted paths (unchanged).
    umap = _expression_parent_umap(parent, sub.index.to_numpy(), adata_full=adata_full)
    xy_pca = PCA(n_components=2, random_state=0).fit_transform(Zp)
    labels = sub["cell.type"].to_numpy()
    U0 = sub["U0"].to_numpy(float)
    U_rel = sub["U_rel"].to_numpy(float)
    U_t = sub["U"].to_numpy(float)
    plast = sub["plast"].to_numpy(float)

    nn = NearestNeighbors(n_neighbors=min(16, max(3, len(xy_pca) - 1))).fit(xy_pca)
    dnn, _ = nn.kneighbors(xy_pca)
    rho = 1.0 / (dnn[:, 1:].mean(axis=1) + 1e-8)
    rho_med = float(np.median(rho))

    frac = _stage_frac(sub, types)

    att_rows = []
    cores = {}
    cents = {}
    for t in types:
        mi = labels == t
        n = int(mi.sum())
        if n < 5:
            continue
        core = _core_idx(U_rel, mi)
        cores[t] = core
        cents_pca = xy_pca[core].mean(axis=0) if core.size else xy_pca[mi].mean(axis=0)
        cents_umap = umap[core].mean(axis=0) if core.size else umap[mi].mean(axis=0)
        cents[t] = cents_pca
        is_u, tscore, peak = inverted_u(frac[t].to_numpy())
        mean_urel = float(np.nanmean(U_rel[mi]))
        mean_u0 = float(np.nanmean(U0[mi]))
        mean_rho = float(np.nanmean(rho[mi]))
        if n < 50 and mean_urel < 0:
            role = "underpowered_well"
        elif mean_urel < 0:
            role = "well"
        else:
            role = "high_U_slope"
        att_rows.append(
            {
                "parent": parent,
                "formal_parent": FORMAL.get(parent, parent),
                "cell.type": t,
                "n": n,
                "mean_potential_stationary": mean_u0,
                "median_potential_stationary": float(np.nanmedian(U0[mi])),
                "mean_potential": float(np.nanmean(U_t[mi])),
                "mean_potential_relative_type": mean_urel,
                "median_potential_relative_type": float(np.nanmedian(U_rel[mi])),
                "mean_local_density": mean_rho,
                "density_vs_parent_median": float(mean_rho / (rho_med + 1e-12)),
                "mean_plasticity": float(np.nanmean(plast[mi])),
                "pc1": float(cents_pca[0]),
                "pc2": float(cents_pca[1]),
                "umap1": float(cents_umap[0]),
                "umap2": float(cents_umap[1]),
                "role": role,
                "temporal_inverted_U": is_u,
                "transient_score": tscore,
                "peak_stage": peak,
                "field_ranking": "potential_relative_type",
                "field_geometry": "potential_stationary scaled as potential_relative_type",
            }
        )
    attractors = pd.DataFrame(att_rows).sort_values("mean_potential_relative_type")
    present = [t for t in types if t in cores]

    graph = _build_graph(xy_pca, U_rel)
    start_node = {}
    for t in present:
        core = cores[t]
        local = xy_pca[core]
        start_node[t] = int(core[int(np.argmin(np.linalg.norm(local - cents[t], axis=1)))])

    dist_from = {}
    prev_from = {}
    for t in present:
        dist_from[t], prev_from[t] = _dijkstra(graph, start_node[t])

    edge_rows = []
    action = {}
    for a, b in itertools.permutations(present, 2):
        d = float(dist_from[a][start_node[b]])
        action[(a, b)] = d
        nodes = _path_nodes(prev_from[a], start_node[a], start_node[b])
        if nodes.size:
            U_path = U_rel[nodes]
            barrier = float(np.nanmax(U_path) - U_path[0])
            occ = pd.Series(labels[nodes]).value_counts(normalize=True).to_dict()
        else:
            barrier = np.nan
            occ = {}
        du_rel = float(
            attractors.loc[attractors["cell.type"] == b, "mean_potential_relative_type"].iloc[0]
            - attractors.loc[attractors["cell.type"] == a, "mean_potential_relative_type"].iloc[0]
        )
        du0 = float(
            attractors.loc[attractors["cell.type"] == b, "mean_potential_stationary"].iloc[0]
            - attractors.loc[attractors["cell.type"] == a, "mean_potential_stationary"].iloc[0]
        )
        if du_rel > 0.05:
            slope = "climb"
        elif du_rel < -0.05:
            slope = "relax"
        else:
            slope = "flat"
        comp = composition_score(frac[a].to_numpy(), frac[b].to_numpy())
        edge_rows.append(
            {
                "parent": parent,
                "formal_parent": FORMAL.get(parent, parent),
                "src": a,
                "dst": b,
                "n_src": int((labels == a).sum()),
                "n_dst": int((labels == b).sum()),
                "graph_action": d,
                "delta_U_rel": du_rel,
                "delta_U0": du0,
                "barrier_U_rel": barrier,
                "direction": slope,
                "path_n_nodes": int(nodes.size),
                "path_frac_src": float(occ.get(a, 0.0)),
                "path_frac_dst": float(occ.get(b, 0.0)),
                "path_frac_other": float(1.0 - occ.get(a, 0.0) - occ.get(b, 0.0)),
                **comp,
            }
        )
    edges = pd.DataFrame(edge_rows)
    rev = {(r.dst, r.src): float(r.graph_action) for r in edges.itertuples()}
    edges["reverse_action"] = [rev.get((r.src, r.dst), np.nan) for r in edges.itertuples()]
    edges["action_ratio"] = edges["reverse_action"] / (edges["graph_action"] + 1e-12)
    edges["reversible"] = edges["action_ratio"].between(ASYM_LO, ASYM_HI)

    # Backbone on adequately sampled subtypes so n=38 AT1 cannot steal the MST.
    powered = [
        t
        for t in present
        if int(attractors.loc[attractors["cell.type"] == t, "n"].iloc[0]) >= 50
    ]
    mst = _mst_undirected(powered, action)
    k_out = 1 if len(powered) <= 3 else 2
    local = set()
    for a in powered:
        outs = sorted([(action[(a, b)], b) for b in powered if b != a])
        for _, b in outs[:k_out]:
            local.add((a, b))
    # Underpowered subtypes: single cheapest neighbor only.
    for t in present:
        if t in powered:
            continue
        outs = sorted([(action[(t, b)], b) for b in present if b != t and np.isfinite(action[(t, b)])])
        if outs:
            local.add((t, outs[0][1]))
            local.add((outs[0][1], t))

    finite = edges["graph_action"].replace([np.inf], np.nan).dropna()
    q40 = float(np.nanpercentile(finite, 40)) if len(finite) else np.inf

    keep_flags = []
    keep_why = []
    for r in edges.itertuples():
        pair = tuple(sorted((r.src, r.dst)))
        reasons = []
        if pair in mst:
            reasons.append("mst")
        if (r.src, r.dst) in local:
            reasons.append("local_cheap")
        keep_flags.append(bool(reasons))
        keep_why.append("+".join(reasons) if reasons else "not_kept")
    edges["keep"] = keep_flags
    edges["keep_reason"] = keep_why

    # well–well geodesic occupancy
    wells = attractors.loc[attractors.role.isin(["well", "underpowered_well"]), "cell.type"].tolist()
    wells_ok = [w for w in wells if attractors.loc[attractors["cell.type"] == w, "n"].iloc[0] >= 15]
    path_occ = {t: 0.0 for t in present}
    well_pair = None
    if len(wells_ok) >= 2:
        # two deepest
        wells_sorted = (
            attractors[attractors["cell.type"].isin(wells_ok)]
            .sort_values("mean_potential_relative_type")["cell.type"]
            .tolist()
        )
        a, b = wells_sorted[0], wells_sorted[1]
        well_pair = (a, b)
        nodes = _path_nodes(prev_from[a], start_node[a], start_node[b])
        if nodes.size:
            vc = pd.Series(labels[nodes]).value_counts(normalize=True)
            for t in present:
                path_occ[t] = float(vc.get(t, 0.0))
    attractors["well_well_path_occupancy"] = attractors["cell.type"].map(path_occ)
    attractors["intermediate_call"] = [
        bool(
            (r.role == "high_U_slope")
            and ((r.well_well_path_occupancy >= 0.08) or r.temporal_inverted_U)
        )
        for r in attractors.itertuples()
    ]

    # triples: is B on A→C geodesic?
    trip_rows = []
    for a, b, c in itertools.permutations(present, 3):
        sac, sab, sbc = action.get((a, c), np.inf), action.get((a, b), np.inf), action.get((b, c), np.inf)
        if not np.isfinite(sac):
            continue
        via = sab + sbc
        trip_rows.append(
            {
                "parent": parent,
                "a": a,
                "b": b,
                "c": c,
                "direct": sac,
                "via_b": via,
                "ratio_via_over_direct": float(via / (sac + 1e-12)),
                "b_on_geodesic": bool(via <= 1.15 * sac),
            }
        )
    triples = pd.DataFrame(trip_rows)

    return {
        "xy_umap": umap,
        "xy_pca": xy_pca,
        "labels": labels,
        "U_rel": U_rel,
        "attractors": attractors,
        "edges": edges,
        "triples": triples,
        "well_pair": well_pair,
        "q40": q40,
        "frac": frac,
    }


# ---------------------------------------------------------------------------
# Figures (journal-style panels)
# ---------------------------------------------------------------------------

def _wrap_label(name: str) -> str:
    """Line-break long formal subtype names for near-node placement (wording unchanged)."""
    wraps = {
        "Activated AT2 cells": "Activated AT2\ncells",
        "M2 macrophages": "M2\nmacrophages",
        "Resolution macrophages": "Resolution\nmacrophages",
        "Fn1+ macrophages": "Fn1+\nmacrophages",
        "Cd163-/Cd11c+ IMs": "Cd163-/Cd11c+\nIMs",
        "Cd163+/Cd11c- IMs": "Cd163+/Cd11c-\nIMs",
    }
    return wraps.get(name, name)


# Formal UMAP subtype legend style (plot_gse141259_formal_celltype_umap._legend_sub).
_SUBTYPE_LABEL_SIZE = 6.1
_SUBTYPE_LABEL_COLOR = "#3a3a3a"


_EDGE_SHORT = {
    "AT2 cells": "AT2",
    "Activated AT2 cells": "Act.AT2",
    "Krt8 ADI": "ADI",
    "AT1 cells": "AT1",
    "AM (PBS)": "PBS",
    "AM (Bleo)": "Bleo",
    "M2 macrophages": "M2",
    "Resolution macrophages": "Resol.",
    "Fn1+ macrophages": "Fn1+",
    "Cd163-/Cd11c+ IMs": "IM−",
    "Cd163+/Cd11c- IMs": "IM+",
}


_COMP_LW = {"strong": 2.85, "medium": 2.45, "weak": 2.05, "none": 1.75}
_COMP_ALPHA = {"strong": 1.0, "medium": 0.96, "weak": 0.90, "none": 0.82}
_COMP_HEAD = {"strong": 15.0, "medium": 14.0, "weak": 13.0, "none": 12.0}


def _edge_tag(src: str, dst: str) -> str:
    return f"{_EDGE_SHORT.get(src, src)}→{_EDGE_SHORT.get(dst, dst)}"


# Prefer short near-node placements for formal subtype names.
_NODE_LABEL_CFG: dict[str, dict[str, dict]] = {
    "alv_epithelium": {
        "AT2 cells": {"xytext": (-8, 10), "ha": "right", "va": "bottom"},
        "Activated AT2 cells": {"xytext": (0, 11), "ha": "center", "va": "bottom"},
        "AT1 cells": {"xytext": (10, 4), "ha": "left", "va": "bottom"},
        "Krt8 ADI": {"xytext": (8, -10), "ha": "left", "va": "top"},
    },
    "macrophages": {
        # Far outward placements into empty UMAP margins (avoid central edge bundle).
        "AM (PBS)": {"xytext": (2, 18), "ha": "center", "va": "bottom"},
        "AM (Bleo)": {"xytext": (22, 8), "ha": "left", "va": "center"},
        "M2 macrophages": {"xytext": (-22, 14), "ha": "right", "va": "bottom"},
        "Resolution macrophages": {"xytext": (22, -8), "ha": "left", "va": "top"},
        "Fn1+ macrophages": {"xytext": (-22, 2), "ha": "right", "va": "center"},
        "Cd163-/Cd11c+ IMs": {"xytext": (-22, -18), "ha": "right", "va": "top"},
        "Cd163+/Cd11c- IMs": {"xytext": (22, -20), "ha": "left", "va": "top"},
    },
}


def _main_direction_edges(kept: pd.DataFrame) -> pd.DataFrame:
    """One directed edge per undirected subtype pair.

    Prefer stronger composition corroboration, then cheaper graph_action,
    then climb > flat > relax.
    """
    if kept.empty:
        return kept.copy()
    out = kept.copy()
    comp_rank = {"strong": 0, "medium": 1, "weak": 2, "none": 3}
    dir_rank = {"climb": 0, "flat": 1, "relax": 2}
    out["_pair"] = [
        tuple(sorted((str(s), str(d)))) for s, d in zip(out["src"].astype(str), out["dst"].astype(str))
    ]
    out["_cr"] = out["composition"].map(comp_rank).fillna(9)
    out["_dr"] = out["direction"].map(dir_rank).fillna(9)
    out = out.sort_values(["_pair", "_cr", "graph_action", "_dr"])
    out = out.drop_duplicates(subset=["_pair"], keep="first")
    return out.drop(columns=["_pair", "_cr", "_dr"]).reset_index(drop=True)


def _edges_for_display(edges: pd.DataFrame, parent: str) -> pd.DataFrame:
    """Plot kept edges as one main direction per pair; Mac caps clutter at 9."""
    kept = _main_direction_edges(edges.loc[edges["keep"]].copy())
    if parent != "macrophages" or len(kept) <= 9:
        return kept
    comp_rank = {"strong": 0, "medium": 1, "weak": 2, "none": 3}
    kept = kept.copy()
    kept["_cr"] = kept["composition"].map(comp_rank).fillna(9)
    kept = kept.sort_values(["_cr", "graph_action"])
    must = kept.loc[kept["composition"].isin(["strong", "medium"])]
    rest = kept.loc[~kept.index.isin(must.index)]
    out = pd.concat([must, rest], ignore_index=True)
    return out.head(9).drop(columns=["_cr"], errors="ignore").copy()


def _node_display_name(name: str) -> str:
    """Use formal cell.type wording from GSE141259_metacelltype_formal_celltype_umap."""
    return _wrap_label(str(name))


def _node_radius(span: float, *, is_well: bool) -> float:
    return (0.024 if is_well else 0.019) * span


def _edge_arc_map(kept: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Small opposite curvatures for bidirectional pairs; 0 for one-way edges."""
    by_pair: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for r in kept.itertuples():
        key = tuple(sorted((r.src, r.dst)))
        by_pair.setdefault(key, []).append((r.src, r.dst))

    arcs: dict[tuple[str, str], float] = {}
    for _key, directed in sorted(by_pair.items()):
        directed = sorted(directed, key=lambda x: (x[0], x[1]))
        if len(directed) >= 2:
            arcs[directed[0]] = 0.18
            arcs[directed[1]] = -0.18
            for i, extra in enumerate(directed[2:], start=2):
                arcs[extra] = 0.18 * (1 + 0.35 * i) * (1 if i % 2 == 0 else -1)
        else:
            arcs[directed[0]] = 0.0
    return arcs


def _rim_point(
    center: tuple[float, float],
    toward: tuple[float, float],
    radius: float,
) -> tuple[float, float]:
    """Point on the circle around `center`, facing `toward` (data coordinates)."""
    a = np.asarray(center, float)
    b = np.asarray(toward, float)
    v = b - a
    n = float(np.linalg.norm(v)) + 1e-9
    return tuple(a + (v / n) * radius)


def _draw_oriented_edge(
    ax,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    r0: float,
    r1: float,
    color: str,
    lw: float,
    alpha: float,
    head: float,
    rad: float,
    zorder: int,
) -> None:
    """Arrow from node rim to node rim in data space (no display-point shrink / lateral offset)."""
    gap = 0.12 * max(r0, r1)
    start = _rim_point(p0, p1, r0 + gap)
    end = _rim_point(p1, p0, r1 + gap)
    # Too-close centroids: still aim rim-to-rim but soften the head.
    dist = float(np.linalg.norm(np.asarray(end) - np.asarray(start)))
    if dist < 1.5 * (r0 + r1):
        head = max(9.0, 0.75 * head)
        lw = max(1.35, 0.85 * lw)

    for halo, hw, ms, col, al, zo in (
        (True, lw + 2.8, 0.0, "white", min(1.0, alpha + 0.08), zorder),
        (False, lw, head, color, alpha, zorder + 1),
    ):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=ms,
                linewidth=hw,
                color=col,
                alpha=al,
                linestyle="-",
                connectionstyle=f"arc3,rad={rad:.3f}",
                shrinkA=0.0,
                shrinkB=0.0,
                zorder=zo,
                clip_on=False,
                capstyle="round",
                joinstyle="round",
            )
        )


def _draw_subtype_nodes(ax, att: pd.DataFrame, span: float) -> None:
    for _, r in att.iterrows():
        is_well = r.role in {"well", "underpowered_well"}
        rad = _node_radius(span, is_well=is_well)
        col = SUB_COL.get(r["cell.type"], "#888888")
        ax.add_patch(
            Circle(
                (float(r.umap1), float(r.umap2)),
                rad * 1.18,
                facecolor="none",
                edgecolor="white",
                linewidth=2.0,
                zorder=24,
                alpha=0.95,
            )
        )
        ax.add_patch(
            Circle(
                (float(r.umap1), float(r.umap2)),
                rad,
                facecolor=col,
                edgecolor="#1a1a1a",
                linewidth=0.70 if is_well else 0.55,
                zorder=26,
                alpha=0.96,
            )
        )


def _near_node_label_cfg(att: pd.DataFrame, parent: str) -> dict[str, dict]:
    """Place each subtype label just outside its node, pushed away from the cluster center."""
    cfg = dict(_NODE_LABEL_CFG.get(parent, {}))
    xs = att["umap1"].to_numpy(float)
    ys = att["umap2"].to_numpy(float)
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    # ~9–12 pt; keep labels hugging the marker, not floating with long leaders.
    dist_pt = 10.0 if parent == "alv_epithelium" else 11.0
    for _, r in att.iterrows():
        ct = str(r["cell.type"])
        if ct in cfg:
            continue
        dx = float(r.umap1) - cx
        dy = float(r.umap2) - cy
        n = float(np.hypot(dx, dy)) + 1e-9
        ux, uy = dx / n, dy / n
        ox, oy = dist_pt * ux, dist_pt * uy
        # Snap to cardinal-ish alignments for stable ha/va.
        if abs(ox) >= abs(oy):
            ha = "left" if ox >= 0 else "right"
            va = "center"
            oy = 0.35 * oy
        else:
            ha = "center"
            va = "bottom" if oy >= 0 else "top"
            ox = 0.35 * ox
        cfg[ct] = {"xytext": (ox, oy), "ha": ha, "va": va}
    return cfg


def _draw_node_labels(ax, att: pd.DataFrame, *, parent: str, parent_color: str) -> None:
    del parent_color  # kept for call-site stability; near-node labels need no leader color
    cfg = _near_node_label_cfg(att, parent)
    for _, r in att.iterrows():
        ct = str(r["cell.type"])
        xy = (float(r.umap1), float(r.umap2))
        lab_cfg = cfg.get(ct, {"xytext": (0, 10), "ha": "center", "va": "bottom"})
        ax.annotate(
            _node_display_name(ct),
            xy=xy,
            xytext=lab_cfg["xytext"],
            textcoords="offset points",
            ha=lab_cfg["ha"],
            va=lab_cfg["va"],
            fontsize=_SUBTYPE_LABEL_SIZE,
            fontweight="normal",
            color=_SUBTYPE_LABEL_COLOR,
            linespacing=1.05,
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.94),
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
            annotation_clip=False,
            zorder=60,
        )


def _draw_mac_subtype_legend(ax, att: pd.DataFrame) -> None:
    """Side legend for Mac: avoids label/edge collisions on the crowded UMAP."""
    order = att.sort_values("mean_potential_relative_type")["cell.type"].astype(str).tolist()
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SUB_COL.get(ct, "#888888"),
            markeredgecolor="#1a1a1a",
            markeredgewidth=0.55,
            markersize=6.2,
            label=_node_display_name(ct).replace("\n", " "),
        )
        for ct in order
    ]
    leg = ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=5.7,
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        edgecolor="#d0d5db",
        facecolor="white",
        borderpad=0.40,
        handletextpad=0.40,
        labelspacing=0.28,
        handlelength=1.0,
        markerscale=1.0,
    )
    leg.get_frame().set_linewidth(0.55)
    leg.set_zorder(80)


def _style_bar_axis(ax, *, parent_color: str) -> None:
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.tick_params(axis="x", colors=MUTED)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.55, color=GRID, zorder=0)
    ax.axvline(0.0, color="#b0b6bc", lw=0.9, zorder=1)
    ax.spines["left"].set_linewidth(2.2)
    ax.spines["left"].set_color(parent_color)
    ax.spines["left"].set_visible(True)


def _urel_bars(ax, att: pd.DataFrame, *, panel: str, parent_color: str, show_xlabel: bool) -> None:
    att = att.sort_values("mean_potential_relative_type")
    names = att["cell.type"].tolist()
    vals = att["mean_potential_relative_type"].to_numpy(float)
    cols = [SUB_COL.get(n, "#888888") for n in names]
    y = np.arange(len(att))[::-1]

    for yi, v in zip(y, vals):
        ax.axhspan(yi - 0.38, yi + 0.38, xmin=0, xmax=1, color=COL_WELL_BG if v < 0 else COL_SLOPE_BG, zorder=0)

    ax.barh(y, vals, color=cols, edgecolor="black", linewidth=0.55, height=0.62, zorder=2)
    for yi, v in zip(y, vals):
        dx = 0.03 if v >= 0 else -0.03
        ax.text(
            v + dx,
            yi,
            f"{v:+.2f}",
            ha="left" if v >= 0 else "right",
            va="center",
            fontsize=6.8,
            color=INK,
            path_effects=_HALO,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=_SUBTYPE_LABEL_SIZE, color=_SUBTYPE_LABEL_COLOR)
    if show_xlabel:
        ax.set_xlabel(
            r"Mean $U_{\mathrm{rel}}$  (within-parent z-score of $U_0$)",
            fontsize=AXIS_LABEL_SIZE,
            color=INK,
        )
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    _style_bar_axis(ax, parent_color=parent_color)
    set_panel_title(ax, panel)


def _style_umap_graph_ax(ax, *, parent_color: str) -> None:
    # Match row-1 frame (panel A): muted left/bottom spines; no parent-colored accent.
    del parent_color
    ax.set_facecolor("#f8f9fb")
    _clean_umap_ax(ax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("UMAP1", fontsize=8.5, color=MUTED, labelpad=1)
    ax.set_ylabel("UMAP2", fontsize=8.5, color=MUTED, labelpad=1)
    for side in ("left", "bottom"):
        sp = ax.spines[side]
        sp.set_visible(True)
        sp.set_color(MUTED)
        sp.set_linewidth(0.8)


def _draw_graph(
    ax,
    pack: dict,
    *,
    panel: str,
    parent_color: str,
    parent: str,
    cax=None,
) -> None:
    xy, labels = pack["xy_umap"], pack["labels"]
    att, edges = pack["attractors"], pack["edges"]
    rng = np.random.default_rng(0)
    take = rng.choice(len(xy), min(3200, len(xy)), replace=False)
    pt = max(1.6, 0.72 * _pt_size(len(xy)))
    # Background points: subtype palette (not U_rel).
    point_cols = np.asarray([SUB_COL.get(str(t), "#C8CDD1") for t in labels[take]], dtype=object)
    ax.scatter(
        xy[take, 0],
        xy[take, 1],
        c=point_cols,
        s=pt,
        alpha=0.32,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    yspan = float(xy[:, 1].max() - xy[:, 1].min()) + 1e-6
    xspan = float(xy[:, 0].max() - xy[:, 0].min()) + 1e-6
    span = max(xspan, yspan)

    kept = _edges_for_display(edges, parent)
    src_urel = att.set_index("cell.type")["mean_potential_relative_type"]
    kept["_src_u"] = kept["src"].map(src_urel)
    kept["_dst_u"] = kept["dst"].map(src_urel)
    comp_rank = {"strong": 0, "medium": 1, "weak": 2, "none": 3}
    kept["_comp_rank"] = kept["composition"].map(comp_rank).fillna(9)
    kept = kept.sort_values(["_comp_rank", "_src_u", "_dst_u", "graph_action"], ascending=[True, True, True, True])
    arc_map = _edge_arc_map(kept)
    rad_by_type = {
        str(r["cell.type"]): _node_radius(span, is_well=r.role in {"well", "underpowered_well"})
        for _, r in att.iterrows()
    }

    for draw_idx, (_, r) in enumerate(kept.iterrows()):
        a = att.loc[att["cell.type"] == r.src].iloc[0]
        b = att.loc[att["cell.type"] == r.dst].iloc[0]
        p0 = (float(a.umap1), float(a.umap2))
        p1 = (float(b.umap1), float(b.umap2))
        col = SUB_COL.get(r.src, "#333333")
        lw = _COMP_LW.get(str(r.composition), 1.75)
        alpha = _COMP_ALPHA.get(str(r.composition), 0.82)
        head = _COMP_HEAD.get(str(r.composition), 13.0)
        _draw_oriented_edge(
            ax,
            p0,
            p1,
            r0=rad_by_type[str(r.src)],
            r1=rad_by_type[str(r.dst)],
            color=col,
            lw=lw,
            alpha=alpha,
            head=head,
            rad=float(arc_map.get((r.src, r.dst), 0.0)),
            zorder=12 + draw_idx,
        )

    _draw_subtype_nodes(ax, att, span)
    if parent == "macrophages":
        _draw_mac_subtype_legend(ax, att)
    else:
        _draw_node_labels(ax, att, parent=parent, parent_color=parent_color)

    _style_umap_graph_ax(ax, parent_color=parent_color)
    set_panel_title(ax, panel, color="black")

    # Keep colorbar column as spacer so C/D align with A/B plot boxes.
    if cax is not None:
        cax.axis("off")

    pad = 0.06 * span
    cx = 0.5 * (float(xy[:, 0].min()) + float(xy[:, 0].max()))
    cy = 0.5 * (float(xy[:, 1].min()) + float(xy[:, 1].max()))
    half = 0.5 * span + pad
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)


def draw_figures(alv: dict, mac: dict) -> list[Path]:
    fig = plt.figure(figsize=(12.4, 10.6))
    # Shared 2×2 column geometry for both rows; reserved colorbar slots keep
    # A↔C and B↔D plot boxes left/right-aligned.
    outer = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.0, 1.0],
        height_ratios=[0.92, 1.28],
        wspace=0.30,
        hspace=0.34,
        left=0.075,
        right=0.985,
        top=0.935,
        bottom=0.10,
    )
    # Each column: main axes + thin colorbar column (empty spacer on row 1).
    gs_tl = outer[0, 0].subgridspec(1, 2, width_ratios=[1.0, 0.045], wspace=0.06)
    gs_tr = outer[0, 1].subgridspec(1, 2, width_ratios=[1.0, 0.045], wspace=0.06)
    gs_bl = outer[1, 0].subgridspec(1, 2, width_ratios=[1.0, 0.045], wspace=0.06)
    gs_br = outer[1, 1].subgridspec(1, 2, width_ratios=[1.0, 0.045], wspace=0.06)

    ranked, stats, subtypes = _load_why_mac_alv()
    ax_a = fig.add_subplot(gs_tl[0, 0])
    paint_landscape(ax_a, ranked, stats, panel_title="A  Landscape position versus abundance")
    # Spacer matching C's colorbar width so A/C plot boxes share the same right edge.
    fig.add_subplot(gs_tl[0, 1]).axis("off")

    gs_b = gs_tr[0, 0].subgridspec(2, 1, height_ratios=[4.0, 7.0], hspace=0.18)
    ax_b_alv = fig.add_subplot(gs_b[0])
    ax_b_mac = fig.add_subplot(gs_b[1], sharex=ax_b_alv)
    paint_subtype_urel(
        ax_b_alv,
        ax_b_mac,
        subtypes,
        panel_title="B  Subtype mean relative potential",
    )
    fig.add_subplot(gs_tr[0, 1]).axis("off")

    ax_c = fig.add_subplot(gs_bl[0, 0])
    cax_c = fig.add_subplot(gs_bl[0, 1])
    _draw_graph(
        ax_c,
        alv,
        panel=f"C  {ALV_LAB} · expression UMAP & kept paths",
        parent_color=ALV_COL,
        parent="alv_epithelium",
        cax=cax_c,
    )
    ax_d = fig.add_subplot(gs_br[0, 0])
    cax_d = fig.add_subplot(gs_br[0, 1])
    _draw_graph(
        ax_d,
        mac,
        panel=f"D  {MAC_LAB} · expression UMAP & kept paths",
        parent_color=MAC_COL,
        parent="macrophages",
        cax=cax_d,
    )

    fig.text(
        0.5,
        0.018,
        "A–B: why Mac / Alv focus (mean $U_0$ vs abundance; subtype $U_{\\mathrm{rel}}$). "
        "C–D paths: latent PCA2 $U_{\\mathrm{rel}}$-weighted graph (MST + locally cheap); "
        "display shows one main direction per subtype pair "
        "(prefer stronger composition, then cheaper graph-path). "
        "Panels C/D display: within-parent all-genes→HVG→PCA→neighbors→Leiden→UMAP; "
        "biological paths overlaid (computation coordinates ≠ display coordinates); "
        "points colored by subtype. "
        "Edge color = source subtype; line weight / opacity = composition corroboration.",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=MUTED,
        style="italic",
    )

    p1 = PANELS / "GSE141259_mac_alv_dynamics_first_paths.png"
    fig.savefig(p1, dpi=300, facecolor="white")
    plt.close(fig)

    # Secondary heatmap figure — lighter journal polish only
    fig = plt.figure(figsize=(11.6, 5.2))
    gs2 = fig.add_gridspec(1, 2, wspace=0.34, left=0.10, right=0.98, top=0.88, bottom=0.18)
    ax = fig.add_subplot(gs2[0, 0])
    _heatmap(ax, alv["edges"], ALV_TYPES, f"A  {ALV_LAB} · graph-path cost")
    ax = fig.add_subplot(gs2[0, 1])
    _heatmap(ax, mac["edges"], MAC_TYPES, f"B  {MAC_LAB} · graph-path cost")
    fig.text(
        0.5,
        0.03,
        r"Lower cost = cheaper on $U_0$ landscape (parent-internal $U_{\mathrm{rel}}$ weights). Boxed cells = kept edges.",
        ha="center",
        fontsize=6.8,
        color=MUTED,
        style="italic",
    )
    p2 = PANELS / "GSE141259_mac_alv_dynamics_first_action_heatmaps.png"
    fig.savefig(p2, dpi=300, facecolor="white")
    plt.close(fig)
    return [p1, p2]


def _heatmap(ax, edges: pd.DataFrame, types: list[str], title: str) -> None:
    mat = np.full((len(types), len(types)), np.nan)
    keep = np.zeros((len(types), len(types)), dtype=bool)
    idx = {t: i for i, t in enumerate(types)}
    for r in edges.itertuples():
        if r.src not in idx or r.dst not in idx:
            continue
        mat[idx[r.src], idx[r.dst]] = r.graph_action
        keep[idx[r.src], idx[r.dst]] = bool(r.keep)
    vmax = np.nanpercentile(mat[np.isfinite(mat)], 90) if np.isfinite(mat).any() else 1.0
    im = ax.imshow(mat, cmap="YlOrBr", vmin=0, vmax=vmax, aspect="equal")
    for i, j in np.ndindex(mat.shape):
        if i == j:
            continue
        if not np.isfinite(mat[i, j]):
            continue
        ax.text(
            j,
            i,
            f"{mat[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=5.6,
            color="white" if mat[i, j] > 0.65 * vmax else INK,
            fontweight="bold" if keep[i, j] else "normal",
        )
        if keep[i, j]:
            ax.add_patch(
                Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#1f4e6a", lw=1.1)
            )
    ax.set_xticks(range(len(types)))
    ax.set_yticks(range(len(types)))
    ax.set_xticklabels(types, rotation=40, ha="right", fontsize=6.4)
    ax.set_yticklabels(types, fontsize=6.4)
    ax.set_xlabel("destination", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("source", fontsize=AXIS_LABEL_SIZE)
    set_panel_title(ax, title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
        r"Graph-path cost ($U_{\mathrm{rel}}$-weighted)", fontsize=6.5
    )


def main() -> None:
    print("Loading obs + latent…", flush=True)
    obs, Z = load_obs_latent()
    # sanity: U_rel should track U0 within parent
    for parent in PARENTS:
        m = obs["annotation"] == parent
        if m.sum() < 20:
            continue
        r = float(np.corrcoef(obs.loc[m, "U0"], obs.loc[m, "U_rel"])[0, 1])
        print(f"  corr(U0, U_rel | {parent}) = {r:.4f}  (expect ~1)", flush=True)

    print("Loading expression adata for display UMAP…", flush=True)
    adata_full = load_annotated_adata(DATASET_REGISTRY["GSE141259"], str(CK))

    print("Alv…", flush=True)
    alv = analyze_parent(obs, Z, "alv_epithelium", ALV_TYPES, adata_full=adata_full)
    print("Mac…", flush=True)
    mac = analyze_parent(obs, Z, "macrophages", MAC_TYPES, adata_full=adata_full)

    att = pd.concat([alv["attractors"], mac["attractors"]], ignore_index=True)
    edges = pd.concat([alv["edges"], mac["edges"]], ignore_index=True)
    triples = pd.concat([alv["triples"], mac["triples"]], ignore_index=True)
    att.to_csv(TAB / "GSE141259_mac_alv_dynamics_first_attractors.csv", index=False)
    edges.to_csv(TAB / "GSE141259_mac_alv_dynamics_first_edges.csv", index=False)
    triples.to_csv(TAB / "GSE141259_mac_alv_dynamics_first_triples.csv", index=False)

    def _brief(pack):
        wells = pack["attractors"].query("role in ['well','underpowered_well']")["cell.type"].tolist()
        slopes = pack["attractors"].query("role == 'high_U_slope'")["cell.type"].tolist()
        kept = pack["edges"].query("keep == True")[["src", "dst", "direction", "graph_action", "composition", "keep_reason"]]
        return {
            "wells": wells,
            "high_U_slope": slopes,
            "well_well_pair": list(pack["well_pair"]) if pack["well_pair"] else None,
            "n_kept_directed_edges": int(pack["edges"].keep.sum()),
            "kept_edges": kept.to_dict(orient="records"),
        }

    summary = {
        "checkpoint": str(CK),
        "fields": {
            "potential": "U(z,t) time-dependent — not used for attractors or paths",
            "potential_stationary": "U0(z) quasi-potential — landscape geometry",
            "potential_relative_type": "z-score of U0 within major type — well ranking and graph weights",
            "no_relative_of_time_dependent_potential": True,
        },
        "display_embedding": {
            "scope": "independently recomputed within each parent",
            "input": "expression counts (all genes within parent)",
            "preprocessing": "HVG(3000) → scale → PCA40 → neighbors(15) → Leiden(0.6)",
            "method": "UMAP",
            "n_neighbors": 15,
            "min_dist": 0.3,
            "random_state": 17,
            "used_for_edge_selection": False,
            "path_computation": "latent PCA2 + U_rel-weighted Dijkstra / MST",
        },
        "alv_epithelium": _brief(alv),
        "macrophages": _brief(mac),
        "note": (
            "Kept edges = MST on subtypes with n≥50 + 2 cheapest outgoing per such node "
            "(underpowered subtypes attach to their single cheapest neighbor). "
            "Composition is corroboration only. Nodes are existing subtype labels."
        ),
    }
    (PROTO / "GSE141259_mac_alv_dynamics_first_summary.json").write_text(json.dumps(summary, indent=2))
    (TAB / "GSE141259_mac_alv_dynamics_first_summary.json").write_text(json.dumps(summary, indent=2))

    print("Drawing…", flush=True)
    paths = draw_figures(alv, mac)
    print("attractors:\n", att[["parent", "cell.type", "n", "mean_potential_relative_type", "role", "intermediate_call"]].to_string(index=False))
    print("\nkept edges:\n", edges.loc[edges.keep, ["parent", "src", "dst", "direction", "graph_action", "composition", "keep_reason"]].to_string(index=False))
    print("wrote", *[str(p) for p in paths], flush=True)


if __name__ == "__main__":
    main()
