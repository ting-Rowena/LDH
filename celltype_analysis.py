"""
Configurable cell-type LAP / pioneer / DE / GO workflow for all datasets.

Usage (per-dataset wrappers):
    python run_lap_analysis.py --dataset HGSOC --cell-type Immune --start IIIC --end IVB
    python GSE155622_LAP.py --cell-type Myelinated --multi-path
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import anndata as ad
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce

from CellFateLandscape import NonEquilibriumCellFateLandscape
from LandscapeVisualizer import analyze_multiple_cell_fate_transitions
from PioneerGene import PioneerGeneIdentifier, analyze_multiple_paths_with_pioneer_genes
from dataset_pipeline import (
    GSE155622,
    GSE155622_ANALYSIS_CELL_TYPES,
    GSE155622_MAIN_CELL_LAP_PATHS,
    GSE155622_MAIN_CELL_TYPES,
    GSE155622_NEURON_LAP_PATHS,
    GSE155622_STAGE_ORDER,
    GSE141259,
    GSE141259_ANALYSIS_CELL_TYPES,
    GSE141259_CELLTYPE_LAP_PATHS,
    GSE141259_STAGE_LABELS,
    GSE141259_STAGE_PALETTE,
    GSE225948_BRAIN,
    GSE225948_CELLTYPE_LAP_PATHS,
    GSE225948_ANALYSIS_CELL_TYPES,
    HGSOC_CELLTYPE_LAP_PATHS,
    GSE225948_TREATMENT_ORDER,
    HGSOC,
    annotate_gse141259,
    annotate_gse155622_from_checkpoint,
    annotate_gse225948,
    annotate_hgsoc_stage,
    get_save_dir,
    resolve_checkpoint_dir,
    harmony_integrate_adata,
    merge_checkpoint_obs,
    resolve_data_path,
    setup_analysis_dirs,
)
from lap_label_config import (
    cache_path_result,
    default_label_config_path,
    export_label_template,
    load_label_overrides,
    load_lap_path_cache,
    path_result_to_cache,
    save_lap_path_cache,
)
from latent_embeddings import ensure_latent_embeddings, get_lap_compute_coords, resolve_lap_compute_key
from landscape_core import stage_core_cell_indices
from lap_helpers import (
    LAP_UMAP_FIGSIZE,
    median_bootstrap_path,
    path_cell_indices_for_panels,
    path_nearest_cell_indices,
    path_result_for_display,
    plot_all_lap_paths_on_umap,
    plot_bootstrap_path_on_umap,
    plot_de_transition_heatmap,
    plot_one_path_on_umap,
    plot_path_cell_panels,
    plot_path_cell_panels_enhanced,
    plot_path_cell_panels_enhanced_canonical,
    plot_path_on_umap,
    plot_pioneer_gene_heatmap,
    plot_two_paths_on_umap,
    project_path_to_display_space,
    unique_ordered,
)
from plot_utils import configure_headless, setup_scanpy_figdir
from potential_derivative_plot import (
    derivative_curve_plot_path,
    plot_potential_derivative_with_canonical_path,
)

configure_headless()


def _slug(value: str) -> str:
    return re.sub(r"[^\w\-]+", "_", str(value).strip())


@dataclass
class DatasetProfile:
    key: str
    spec: object
    annotate_fn: Callable
    cell_type_column: str
    default_cell_type: str
    available_cell_types: Tuple[str, ...]
    default_start: str
    default_end: str
    stage_order: Tuple[str, ...]
    stage_palette: Dict[str, str]
    path_pairs: Tuple[Tuple[str, str], ...]
    go_organism: str
    harmony_key: Optional[str] = None
    subset_after_harmony: bool = False
    use_adaptive_min_path: bool = False
    min_path_n_floor: int = 0
    de_n_sample: int = 100
    de_log2fc_cutoff: float = 1.0
    de_pval_cutoff: float = 0.05
    heatmap_vmax: float = 2.5
    n_top_genes_hvg: int = 3000
    n_pcs: int = 50
    n_neighbors: int = 15
    lap_embedding_key: str = "X_latent_pca"
    lap_display_key: str = "X_umap"
    lap_n_pcs: int = 10
    max_path_points: int = 25
    standard_endpoint_modes: Tuple[str, ...] = ("medoid", "pseudotime_quantile")
    endpoint_selection_mode: str = "auto"
    endpoint_selection_strategy: str = "auto_two_stage"
    endpoint_selection_reliability_mode: str = "reliability_minimal"
    endpoint_candidate_modes: Tuple[str, ...] = (
        "legacy",
        "medoid",
        "pseudotime_quantile",
        "farthest_core",
        "density_core",
        "hybrid",
    )
    endpoint_separability_threshold: float = 2.0
    skip_lap_if_endpoint_not_separable: bool = False
    bootstrap_n: int = 50
    bootstrap_core_fraction: float = 0.5
    cell_type_lap_paths: Optional[Dict[str, Tuple[str, str]]] = None

    def lap_path_for_cell_type(self, cell_type: str) -> Tuple[str, str]:
        """Return (start_state, end_state) for a cell type, else profile defaults."""
        if self.cell_type_lap_paths and cell_type in self.cell_type_lap_paths:
            return self.cell_type_lap_paths[cell_type]
        return self.default_start, self.default_end


@dataclass
class CellTypeLAPConfig:
    profile: DatasetProfile
    cell_type: Optional[str] = None
    start_state: Optional[str] = None
    end_state: Optional[str] = None
    clustering_key: str = "stage"
    pioneer_path: str = "min"  # min | max | both
    run_multi_path: bool = False
    paths_only: bool = False
    run_bootstrap: bool = True
    run_go: bool = True
    run_de: bool = True
    top_n_pioneer: int = 20
    top_n_de: int = 20
    seed: int = 42
    save_dir: Optional[str] = None
    figure_dir: Optional[str] = None
    output_tag: Optional[str] = None
    label_config_path: Optional[str] = None
    regenerate_umap_only: bool = False
    regenerate_panels_only: bool = False
    export_label_template: bool = True
    umap_only_suffixes: Optional[Tuple[str, ...]] = None
    endpoint_selection_mode: str = "auto"
    endpoint_selection_strategy: str = "auto_two_stage"
    endpoint_selection_reliability_mode: str = "reliability_minimal"
    endpoint_separability_threshold: float = 2.0

    def resolved_label_config(self, figure_dir: str) -> str:
        return self.label_config_path or str(default_label_config_path(figure_dir))

    def resolved_cell_type(self) -> str:
        return self.cell_type or self.profile.default_cell_type

    def resolved_start(self) -> str:
        if self.start_state:
            return self.start_state
        return self.profile.lap_path_for_cell_type(self.resolved_cell_type())[0]

    def resolved_end(self) -> str:
        if self.end_state:
            return self.end_state
        return self.profile.lap_path_for_cell_type(self.resolved_cell_type())[1]

    def output_prefix(self) -> str:
        if self.output_tag:
            return _slug(self.output_tag)
        ct = _slug(self.resolved_cell_type())
        return f"{self.profile.key}_{ct}"


HGSOC_PROFILE = DatasetProfile(
    key="HGSOC",
    spec=HGSOC,
    annotate_fn=annotate_hgsoc_stage,
    cell_type_column="annotation",
    default_cell_type="EOC",
    available_cell_types=("EOC", "Immune", "Stromal"),
    default_start="IIIC",
    default_end="IVB",
    stage_order=("IIIC", "IVA", "IVB"),
    stage_palette={"IIIC": "#D2F1DC", "IVA": "#518463", "IVB": "#254750"},
    path_pairs=(("IIIC", "IVA"), ("IVA", "IVB"), ("IIIC", "IVB")),
    cell_type_lap_paths=dict(HGSOC_CELLTYPE_LAP_PATHS),
    go_organism="Human",
    de_log2fc_cutoff=1.0,
    heatmap_vmax=2.5,
)

GSE155622_PROFILE = DatasetProfile(
    key="GSE155622",
    spec=GSE155622,
    annotate_fn=annotate_gse155622_from_checkpoint,
    cell_type_column="neuron_subtype",
    default_cell_type="Myelinated",
    available_cell_types=GSE155622_MAIN_CELL_TYPES
    + GSE155622_ANALYSIS_CELL_TYPES,
    default_start="Control",
    default_end="SNI 14d",
    stage_order=tuple(GSE155622_STAGE_ORDER),
    stage_palette={
        "Control": "#7c9559",
        "SNI 6h": "#90ac7c",
        "SNI 24h": "#bdbb55",
        "SNI 2d": "#deb956",
        "SNI 7d": "#9dbdd2",
        "SNI 14d": "#779ebd",
    },
    path_pairs=(
        ("Control", "SNI 6h"),
        ("SNI 6h", "SNI 24h"),
        ("SNI 24h", "SNI 2d"),
        ("SNI 2d", "SNI 7d"),
        ("SNI 7d", "SNI 14d"),
        ("Control", "SNI 14d"),
        ("SNI 6h", "SNI 14d"),
    ),
    cell_type_lap_paths={**GSE155622_MAIN_CELL_LAP_PATHS, **GSE155622_NEURON_LAP_PATHS},
    go_organism="Mouse",
    use_adaptive_min_path=True,
    min_path_n_floor=26,
    de_n_sample=50,
    de_log2fc_cutoff=0.5,
    heatmap_vmax=3.5,
)

GSE225948_PROFILE = DatasetProfile(
    key="GSE225948_Brain",
    spec=GSE225948_BRAIN,
    annotate_fn=annotate_gse225948,
    cell_type_column="parent",
    default_cell_type="Mg",
    available_cell_types=GSE225948_ANALYSIS_CELL_TYPES,
    default_start="Sham",
    default_end="D14",
    stage_order=tuple(GSE225948_TREATMENT_ORDER),
    stage_palette={"Sham": "#D2F1DC", "D02": "#518463", "D14": "#254750"},
    path_pairs=(("Sham", "D02"), ("D02", "D14"), ("Sham", "D14")),
    cell_type_lap_paths=dict(GSE225948_CELLTYPE_LAP_PATHS),
    go_organism="Mouse",
    de_n_sample=100,
    heatmap_vmax=5.5,
)

GSE141259_PROFILE = DatasetProfile(
    key="GSE141259",
    spec=GSE141259,
    annotate_fn=annotate_gse141259,
    cell_type_column="metacelltype",
    default_cell_type="macrophages",
    available_cell_types=GSE141259_ANALYSIS_CELL_TYPES,
    default_start="D0",
    default_end="D28",
    stage_order=tuple(GSE141259_STAGE_LABELS),
    stage_palette=dict(GSE141259_STAGE_PALETTE),
    path_pairs=(
        ("D0", "D3"),
        ("D3", "D7"),
        ("D7", "D10"),
        ("D10", "D14"),
        ("D14", "D21"),
        ("D21", "D28"),
        ("D0", "D28"),
    ),
    cell_type_lap_paths=dict(GSE141259_CELLTYPE_LAP_PATHS),
    go_organism="Mouse",
    de_n_sample=100,
    heatmap_vmax=3.5,
)

DATASET_REGISTRY: Dict[str, DatasetProfile] = {
    "HGSOC": HGSOC_PROFILE,
    "GSE155622": GSE155622_PROFILE,
    "GSE155622_Brain": GSE155622_PROFILE,
    "GSE225948": GSE225948_PROFILE,
    "GSE225948_Brain": GSE225948_PROFILE,
    "GSE141259": GSE141259_PROFILE,
}


def list_cell_types(profile: DatasetProfile, save_dir: Optional[str] = None) -> pd.Series:
    save_dir = save_dir or get_save_dir(profile.spec)
    adata = sc.read_h5ad(resolve_data_path(profile.spec))
    adata = merge_checkpoint_obs(adata, save_dir)
    adata = profile.annotate_fn(adata)
    col = profile.cell_type_column
    if col not in adata.obs.columns:
        raise KeyError(f"Column {col!r} not in adata.obs. Available: {list(adata.obs.columns)}")
    counts = adata.obs[col].value_counts()
    print(f"\n=== {profile.key}: cell types in {col!r} ===")
    for ct, n in counts.items():
        print(f"  {ct}: {n}")
    return counts


def load_annotated_adata(profile: DatasetProfile, save_dir: str):
    adata = sc.read_h5ad(resolve_data_path(profile.spec))
    adata = merge_checkpoint_obs(adata, save_dir)
    adata = profile.annotate_fn(adata)
    if profile.cell_type_column in adata.obs.columns:
        adata.obs[profile.cell_type_column] = adata.obs[profile.cell_type_column].astype("category")
    if "stage" in adata.obs.columns:
        adata.obs["stage"] = adata.obs["stage"].astype("category")
    return adata


def subset_and_preprocess(adata, cfg: CellTypeLAPConfig):
    profile = cfg.profile
    cell_type = cfg.resolved_cell_type()
    col = profile.cell_type_column

    if col not in adata.obs.columns:
        raise KeyError(f"Missing cell type column {col!r} in adata.obs")

    available = adata.obs[col].astype(str).unique()
    if cell_type not in set(adata.obs[col].astype(str)):
        raise ValueError(
            f"Cell type {cell_type!r} not found in {col}. "
            f"Available: {sorted(available)[:30]}{'...' if len(available) > 30 else ''}"
        )

    if profile.harmony_key and not profile.subset_after_harmony:
        sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=profile.n_top_genes_hvg, flavor="seurat_v3")
        adata = adata[:, adata.var.highly_variable].copy()
        sc.tl.pca(adata, n_comps=profile.n_pcs)
        harmony_integrate_adata(
            adata, profile.harmony_key, adjusted_basis="X_harmony", max_iter_harmony=20
        )
        sc.pp.neighbors(adata, n_neighbors=profile.n_neighbors, n_pcs=profile.n_pcs, use_rep="X_harmony")

    adata = adata[adata.obs[col].astype(str) == cell_type].copy()
    print(f"Subset {col}={cell_type!r}: {adata.n_obs} cells")

    if not (profile.harmony_key and not profile.subset_after_harmony):
        sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=profile.n_top_genes_hvg, flavor="seurat_v3")
        adata = adata[:, adata.var.highly_variable].copy()
        sc.tl.pca(adata, n_comps=profile.n_pcs)

    if profile.harmony_key and profile.subset_after_harmony:
        harmony_integrate_adata(
            adata, profile.harmony_key, adjusted_basis="X_harmony", max_iter_harmony=20
        )
        sc.pp.neighbors(adata, n_neighbors=profile.n_neighbors, n_pcs=profile.n_pcs, use_rep="X_harmony")
    else:
        sc.pp.neighbors(adata, n_neighbors=profile.n_neighbors, n_pcs=profile.n_pcs)

    sc.tl.umap(adata, n_components=3)
    adata.obsm["X_umap_3d"] = adata.obsm["X_umap"].copy()
    sc.tl.umap(adata, n_components=2)

    if "X_pca" in adata.obsm:
        n_lap = min(profile.lap_n_pcs, adata.obsm["X_pca"].shape[1])
        adata.obsm["X_pca_lap"] = np.asarray(adata.obsm["X_pca"][:, :n_lap], dtype=float)

    ckpt = (
        cfg.save_dir
        or (get_save_dir(profile.spec) if hasattr(profile, "spec") else None)
    )
    compute_key, used_fallback = ensure_latent_embeddings(
        adata, checkpoint_dir=ckpt, warn=True
    )
    profile_lap_key = profile.lap_embedding_key
    if profile_lap_key in adata.obsm:
        adata.uns["lap_compute_space"] = profile_lap_key
        adata.uns["lap_display_space"] = profile.lap_display_key
        if used_fallback:
            adata.uns["lap_compute_fallback"] = compute_key
    else:
        adata.uns["lap_compute_space"] = compute_key
        adata.uns["lap_display_space"] = profile.lap_display_key
    return adata


def _make_analyzer(adata, profile: DatasetProfile) -> NonEquilibriumCellFateLandscape:
    embedding_key, _ = resolve_lap_compute_key(adata, preferred=profile.lap_embedding_key)
    if embedding_key == "X_latent_pca" and profile.lap_n_pcs:
        coords = get_lap_compute_coords(adata, embedding_key, n_pcs=profile.lap_n_pcs)
        adata.obsm["_lap_compute_slice"] = coords
        embedding_key = "_lap_compute_slice"
    return NonEquilibriumCellFateLandscape(
        adata,
        potential_key="potential",
        embedding_2d_key=embedding_key,
        potential_transform="none",
    )


def _path_n_points(adata, profile: DatasetProfile) -> int:
    raw = max(math.ceil(0.01 * adata.n_obs), profile.min_path_n_floor)
    return int(min(raw, profile.max_path_points))


def _resolve_endpoint_selection_strategy(
    cfg: CellTypeLAPConfig, profile: DatasetProfile, endpoint_mode: str
) -> str:
    """Return endpoint selection strategy for canonical LAP paths."""
    if endpoint_mode in ("min_potential", "max_potential"):
        return "legacy"
    strategy = getattr(
        cfg,
        "endpoint_selection_strategy",
        getattr(profile, "endpoint_selection_strategy", "auto_two_stage"),
    )
    mode = getattr(cfg, "endpoint_selection_mode", getattr(profile, "endpoint_selection_mode", "legacy"))
    if mode == "legacy":
        return "legacy"
    if strategy in ("legacy", "auto_pre", "auto_two_stage"):
        return strategy
    if mode == "auto":
        return "auto_pre"
    return "auto_pre"


def _resolve_endpoint_selection_mode(cfg: CellTypeLAPConfig, profile: DatasetProfile, endpoint_mode: str) -> Optional[str]:
    """Return None for legacy identify_cell_states; otherwise selection strategy key."""
    strategy = _resolve_endpoint_selection_strategy(cfg, profile, endpoint_mode)
    if strategy == "legacy":
        return None
    if strategy == "auto_pre":
        return "auto"
    return "auto_two_stage"


def _select_endpoint_positions(
    analyzer: NonEquilibriumCellFateLandscape,
    adata,
    profile: DatasetProfile,
    cfg: CellTypeLAPConfig,
    start_state: str,
    end_state: str,
    endpoint_mode: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[dict], Optional[list]]:
    """Select LAP start/end positions; optionally via multi-strategy endpoint search."""
    selection = _resolve_endpoint_selection_mode(cfg, profile, endpoint_mode)
    if selection is None:
        states = analyzer.identify_cell_states(
            clustering_key=cfg.clustering_key,
            endpoint_mode=endpoint_mode,
            start_state=start_state,
            end_state=end_state,
            core_fraction=profile.bootstrap_core_fraction,
            use_3d=False,
        )
        if start_state not in states or end_state not in states:
            missing = [s for s in (start_state, end_state) if s not in states]
            raise ValueError(f"States not found in subset: {missing}. Available: {list(states.keys())}")
        return (
            states[start_state]["position"],
            states[end_state]["position"],
            None,
            None,
        )

    from endpoint_selection import (
        build_public_endpoint_selection_meta,
        generate_endpoint_candidates,
        select_best_endpoint_candidate,
    )

    compute_key = getattr(analyzer, "embedding_2d_key", profile.lap_embedding_key)
    coords = np.asarray(adata.obsm[compute_key], dtype=float)
    modes = profile.endpoint_candidate_modes if selection == "auto" else (selection,)
    threshold = float(
        getattr(cfg, "endpoint_separability_threshold", profile.endpoint_separability_threshold)
    )
    candidates = generate_endpoint_candidates(
        adata,
        coords,
        start_state,
        end_state,
        stage_key=cfg.clustering_key,
        candidate_modes=modes,
        separability_threshold=threshold,
        core_fraction=profile.bootstrap_core_fraction,
    )
    best, all_inseparable = select_best_endpoint_candidate(candidates)
    endpoint_meta = build_public_endpoint_selection_meta(
        best,
        selection_strategy="auto_pre",
        reliability_mode=getattr(
            cfg,
            "endpoint_selection_reliability_mode",
            getattr(profile, "endpoint_selection_reliability_mode", "reliability_minimal"),
        ),
        all_candidates=candidates,
    )
    endpoint_meta["all_inseparable"] = all_inseparable
    return (
        coords[best.start_idx],
        coords[best.end_idx],
        endpoint_meta,
        candidates,
    )


def _compute_lap_path_from_positions(
    analyzer,
    adata,
    profile,
    start_pos,
    end_pos,
    endpoint_mode: str,
    start_state: str,
    end_state: str,
    *,
    n_points: Optional[int] = None,
    max_iter: int = 40,
    use_ensemble: bool = False,
) -> dict:
    n_points = n_points if n_points is not None else _path_n_points(adata, profile)
    lap_kw = dict(use_3d=False, max_iter=max_iter, use_ensemble=use_ensemble)
    if endpoint_mode == "min_potential" and profile.use_adaptive_min_path:
        path = analyzer.compute_least_action_path_adaptive(start_pos, end_pos, **lap_kw)
    else:
        path = analyzer.compute_least_action_path(
            start_pos, end_pos, n_points=n_points, **lap_kw
        )
    path["endpoint_mode"] = endpoint_mode
    path["start_state"] = start_state
    path["end_state"] = end_state
    return path


def _run_two_stage_endpoint_selection(
    analyzer,
    adata,
    profile: DatasetProfile,
    cfg: CellTypeLAPConfig,
    start_state: str,
    end_state: str,
    endpoint_mode: str,
    *,
    n_points: Optional[int] = None,
    max_iter: int = 40,
    use_ensemble: bool = False,
) -> Tuple[dict, List, dict]:
    from endpoint_selection import select_reliable_endpoints
    from lap_reliability_diagnostics import build_candidate_lap_reliability

    compute_key = getattr(analyzer, "embedding_2d_key", profile.lap_embedding_key)
    display_key = profile.lap_display_key
    coords = np.asarray(adata.obsm[compute_key], dtype=float)
    threshold = float(getattr(cfg, "endpoint_separability_threshold", profile.endpoint_separability_threshold))
    reliability_mode = getattr(
        cfg,
        "endpoint_selection_reliability_mode",
        getattr(profile, "endpoint_selection_reliability_mode", "reliability_minimal"),
    )
    mode_override = getattr(cfg, "endpoint_selection_mode", getattr(profile, "endpoint_selection_mode", "auto"))
    if mode_override in ("auto", "legacy"):
        mode_override = None

    print(
        f"  Reliability-aware endpoint selection ({len(profile.endpoint_candidate_modes)} generators, "
        f"reliability={reliability_mode})...",
        flush=True,
    )

    def compute_lap_path(cand, start_pos, end_pos):
        print(f"    LAP candidate ({cand.mode})...", flush=True)
        raw_path = _compute_lap_path_from_positions(
            analyzer, adata, profile, start_pos, end_pos, endpoint_mode,
            start_state, end_state, n_points=n_points, max_iter=max_iter,
            use_ensemble=use_ensemble,
        )
        return path_result_for_display(raw_path, adata, compute_key=compute_key, display_key=display_key)

    def compute_lap_reliability(cand, path_result):
        lap_mode = "full" if reliability_mode == "full" else "reliability_minimal"
        print(f"      Candidate reliability ({lap_mode})...", flush=True)
        return build_candidate_lap_reliability(
            path_result,
            analyzer,
            adata=adata,
            mode=lap_mode,
            start_state=start_state,
            end_state=end_state,
            clustering_key=cfg.clustering_key,
            pseudotime_key="pseudotime",
            core_fraction=profile.bootstrap_core_fraction,
            n_bootstrap=int(getattr(profile, "bootstrap_n", 50)),
            random_state=int(getattr(cfg, "seed", 42)),
        )

    out = select_reliable_endpoints(
        adata,
        coords,
        start_state,
        end_state,
        cfg.clustering_key,
        selection_strategy="auto_two_stage",
        reliability_mode=reliability_mode,
        candidate_modes=profile.endpoint_candidate_modes,
        candidate_mode_override=mode_override,
        separability_threshold=threshold,
        core_fraction=profile.bootstrap_core_fraction,
        compute_lap_path=compute_lap_path,
        compute_lap_reliability=compute_lap_reliability,
    )
    best = out["best_candidate"]
    best_path = out["best_path_result"]
    endpoint_meta = out["endpoint_selection"]
    updated = out["candidates"]
    if best_path is None:
        raise RuntimeError(f"Reliability-aware endpoint selection failed for mode={best.mode}")

    if best.lap_reliability is not None:
        best_path["precomputed_lap_reliability"] = best.lap_reliability
    best_path["endpoint_selection"] = endpoint_meta
    best_path["endpoint_candidates_two_stage"] = updated
    best_path["endpoint_candidates_df"] = out["endpoint_candidates_df"]
    return best_path, updated, endpoint_meta


def _compute_path_between_states(
    analyzer: NonEquilibriumCellFateLandscape,
    adata,
    profile: DatasetProfile,
    cfg: CellTypeLAPConfig,
    start_state: str,
    end_state: str,
    endpoint_mode: str,
    *,
    n_points: Optional[int] = None,
    max_iter: int = 40,
    use_ensemble: bool = False,
) -> dict:
    """Compute LAP path between density-based endpoint centroids."""
    from density_endpoints import select_density_endpoints

    compute_key = getattr(analyzer, "embedding_2d_key", profile.lap_embedding_key)
    coords = np.asarray(adata.obsm[compute_key], dtype=float)
    labels = adata.obs[cfg.clustering_key].values
    strategy = _resolve_endpoint_selection_strategy(cfg, profile, endpoint_mode)

    if strategy == "auto_two_stage" and endpoint_mode not in ("min_potential", "max_potential"):
        best_path, _, endpoint_meta = _run_two_stage_endpoint_selection(
            analyzer,
            adata,
            profile,
            cfg,
            start_state,
            end_state,
            endpoint_mode,
            n_points=n_points,
            max_iter=max_iter,
            use_ensemble=use_ensemble,
        )
        if (
            profile.skip_lap_if_endpoint_not_separable
            and not endpoint_meta.get("endpoint_is_separable", endpoint_meta.get("is_separable", True))
        ):
            return {
                "status": "endpoint_not_separable",
                "endpoint_mode": endpoint_mode,
                "start_state": start_state,
                "end_state": end_state,
                "endpoint_selection": endpoint_meta,
                "endpoint_candidates": best_path.get("endpoint_candidates_two_stage"),
                "endpoint_candidates_df": best_path.get("endpoint_candidates_df"),
            }
        print(
            f"  LAP done ({endpoint_mode or 'reliability_aware'}), "
            f"arc-length score={best_path.get('total_action', 0):.4g}",
            flush=True,
        )
        return best_path

    if endpoint_mode in ("min_potential", "max_potential"):
        start_pos, end_pos, endpoint_meta, candidates = _select_endpoint_positions(
            analyzer, adata, profile, cfg, start_state, end_state, endpoint_mode
        )
    else:
        start_pos, end_pos, endpoint_meta = select_density_endpoints(
            coords, labels, start_state, end_state
        )
        candidates = None

    if (
        endpoint_meta is not None
        and profile.skip_lap_if_endpoint_not_separable
        and not endpoint_meta.get("is_separable", True)
    ):
        return {
            "status": "endpoint_not_separable",
            "endpoint_mode": endpoint_mode,
            "start_state": start_state,
            "end_state": end_state,
            "endpoint_selection": endpoint_meta,
            "endpoint_candidates": candidates,
        }

    print(f"  LAP optimize ({endpoint_mode or 'density_centroid'})...", flush=True)
    path = _compute_lap_path_from_positions(
        analyzer, adata, profile, start_pos, end_pos, endpoint_mode,
        start_state, end_state, n_points=n_points, max_iter=max_iter,
        use_ensemble=use_ensemble,
    )
    print(f"  LAP done ({endpoint_mode or 'density_centroid'}), action={path.get('total_action', 0):.4g}", flush=True)

    if endpoint_meta is not None:
        path["endpoint_selection"] = endpoint_meta
    if candidates is not None:
        path["endpoint_candidates"] = candidates
    display_key = profile.lap_display_key
    result = path_result_for_display(path, adata, compute_key=compute_key, display_key=display_key)
    if endpoint_meta is not None:
        result["endpoint_selection"] = endpoint_meta
    return result


def bootstrap_canonical_paths(
    analyzer: NonEquilibriumCellFateLandscape,
    adata,
    profile: DatasetProfile,
    cfg: CellTypeLAPConfig,
    start_state: str,
    end_state: str,
    endpoint_mode: str = "medoid",
) -> dict:
    compute_key = getattr(analyzer, "embedding_2d_key", profile.lap_embedding_key)
    display_key = profile.lap_display_key
    positions = np.asarray(adata.obsm[compute_key], dtype=float)
    labels = adata.obs[cfg.clustering_key].values
    pseudotime = (
        np.asarray(adata.obs["pseudotime"].values, dtype=float)
        if "pseudotime" in adata.obs
        else None
    )
    core_by = "pseudotime" if endpoint_mode == "pseudotime_quantile" else "medoid"
    start_core = stage_core_cell_indices(
        positions,
        labels,
        start_state,
        pseudotime=pseudotime,
        core_fraction=profile.bootstrap_core_fraction,
        by=core_by,
    )
    end_core = stage_core_cell_indices(
        positions,
        labels,
        end_state,
        pseudotime=pseudotime,
        core_fraction=profile.bootstrap_core_fraction,
        by=core_by,
    )
    if len(start_core) < 2 or len(end_core) < 2:
        warnings.warn("Too few core cells for bootstrap; skipping.", UserWarning)
        return {}

    rng = np.random.default_rng(cfg.seed)
    n_points = min(25, _path_n_points(adata, profile))
    display_paths = []
    central = None

    for i in range(profile.bootstrap_n):
        if i == 0 or (i + 1) % 10 == 0:
            print(f"  bootstrap {endpoint_mode}: {i + 1}/{profile.bootstrap_n}", flush=True)
        s_idx = int(rng.choice(start_core))
        e_idx = int(rng.choice(end_core))
        start_pos = positions[s_idx]
        end_pos = positions[e_idx]
        path = analyzer.compute_least_action_path(
            start_pos, end_pos, n_points=n_points, use_3d=False, max_iter=60
        )
        if i == 0:
            central = path_result_for_display(path, adata, compute_key=compute_key, display_key=display_key)
        umap_path = project_path_to_display_space(path["path"], positions, adata.obsm[display_key])
        display_paths.append(umap_path)

    median, lower, upper = median_bootstrap_path(display_paths)
    return {
        "endpoint_mode": endpoint_mode,
        "median_path": median,
        "lower_path": lower,
        "upper_path": upper,
        "n_success": len(display_paths),
        "central_path": central,
        "display_paths": display_paths,
    }


def compute_all_lap_paths(adata, cfg: CellTypeLAPConfig) -> dict:
    profile = cfg.profile
    start_state = cfg.resolved_start()
    end_state = cfg.resolved_end()

    analyzer = _make_analyzer(adata, profile)
    results = {"analyzer": analyzer, "start_state": start_state, "end_state": end_state}

    results["min_path"] = _compute_path_between_states(
        analyzer, adata, profile, cfg, start_state, end_state, "min_potential"
    )
    results["max_path"] = _compute_path_between_states(
        analyzer, adata, profile, cfg, start_state, end_state, "max_potential"
    )

    for mode in profile.standard_endpoint_modes:
        if mode == "medoid":
            key = "canonical_medoid"
        elif mode == "pseudotime_quantile":
            key = "canonical_pseudotime"
        else:
            key = f"canonical_{mode}"
        print(f"Computing canonical path ({mode})...")
        results[key] = _compute_path_between_states(
            analyzer, adata, profile, cfg, start_state, end_state, mode
        )

    if cfg.run_bootstrap and profile.bootstrap_n > 0:
        print(f"Bootstrap canonical paths (n={profile.bootstrap_n})...")
        results["bootstrap_medoid"] = bootstrap_canonical_paths(
            analyzer, adata, profile, cfg, start_state, end_state, endpoint_mode="medoid"
        )
        if "pseudotime_quantile" in profile.standard_endpoint_modes and "pseudotime" in adata.obs:
            results["bootstrap_pseudotime"] = bootstrap_canonical_paths(
                analyzer,
                adata,
                profile,
                cfg,
                start_state,
                end_state,
                endpoint_mode="pseudotime_quantile",
            )

    return results


def compute_min_max_paths(adata, cfg: CellTypeLAPConfig):
    """Backward-compatible wrapper."""
    all_paths = compute_all_lap_paths(adata, cfg)
    return (
        all_paths["analyzer"],
        all_paths["analyzer"],
        all_paths["min_path"],
        all_paths["max_path"],
        (all_paths["start_state"], all_paths["end_state"]),
    )


def _lap_cache_path(figure_dir: str, prefix: str) -> str:
    return str(Path(figure_dir) / f"{prefix}_lap_cache.json")


def save_lap_figure_cache(
    figure_dir: str,
    prefix: str,
    lap_paths: dict,
    start_state: str,
    end_state: str,
    cell_type: str,
    dataset_key: str,
) -> str:
    cache_path = _lap_cache_path(figure_dir, prefix)
    existing_boot = None
    if Path(cache_path).is_file():
        try:
            existing_boot = load_lap_path_cache(cache_path).get("bootstrap_medoid")
        except Exception:
            existing_boot = None

    payload = {
        "dataset": dataset_key,
        "cell_type": cell_type,
        "start_state": start_state,
        "end_state": end_state,
        "paths": {},
        "bootstrap_medoid": None,
    }
    for key in ("min_path", "max_path", "canonical_medoid", "canonical_pseudotime"):
        if key in lap_paths and lap_paths[key] is not None:
            payload["paths"][key] = path_result_to_cache(lap_paths[key])
    boot = lap_paths.get("bootstrap_medoid") or {}
    if boot.get("median_path") is not None:
        payload["bootstrap_medoid"] = {
            "median_path": np.asarray(boot["median_path"], dtype=float).tolist(),
            "lower_path": np.asarray(boot["lower_path"], dtype=float).tolist(),
            "upper_path": np.asarray(boot["upper_path"], dtype=float).tolist(),
            "central_path": path_result_to_cache(boot["central_path"])
            if boot.get("central_path")
            else None,
            "n_success": int(boot.get("n_success", 0)),
        }
    elif existing_boot:
        payload["bootstrap_medoid"] = existing_boot
    save_lap_path_cache(cache_path, payload)
    return cache_path


def merge_bootstrap_medoid_into_lap_cache(
    figure_dir: str,
    prefix: str,
    bootstrap: dict,
    *,
    dataset_key: Optional[str] = None,
    cell_type: Optional[str] = None,
    start_state: Optional[str] = None,
    end_state: Optional[str] = None,
) -> str:
    """Merge bootstrap envelope into an existing LAP cache without dropping path entries."""
    cache_path = _lap_cache_path(figure_dir, prefix)
    if Path(cache_path).is_file():
        payload = load_lap_path_cache(cache_path)
    else:
        if not all(v is not None for v in (dataset_key, cell_type, start_state, end_state)):
            raise ValueError(
                "merge_bootstrap_medoid_into_lap_cache requires dataset/cell_type/start/end "
                "when no cache file exists"
            )
        payload = {
            "dataset": dataset_key,
            "cell_type": cell_type,
            "start_state": start_state,
            "end_state": end_state,
            "paths": {},
            "bootstrap_medoid": None,
        }

    boot = bootstrap or {}
    if boot.get("median_path") is None:
        return str(cache_path)

    payload["bootstrap_medoid"] = {
        "median_path": np.asarray(boot["median_path"], dtype=float).tolist(),
        "lower_path": np.asarray(boot["lower_path"], dtype=float).tolist(),
        "upper_path": np.asarray(boot["upper_path"], dtype=float).tolist(),
        "central_path": path_result_to_cache(boot["central_path"]) if boot.get("central_path") else None,
        "n_success": int(boot.get("n_success", 0)),
    }
    if dataset_key is not None:
        payload["dataset"] = dataset_key
    if cell_type is not None:
        payload["cell_type"] = cell_type
    if start_state is not None:
        payload["start_state"] = start_state
    if end_state is not None:
        payload["end_state"] = end_state
    save_lap_path_cache(cache_path, payload)
    return str(cache_path)


def persist_bootstrap_umap_from_validation_bootstrap(
    adata,
    cfg: CellTypeLAPConfig,
    bootstrap_metrics: dict,
    *,
    start_state: str,
    end_state: str,
) -> dict:
    """
    Persist validation bootstrap geometry to checkpoint LAP cache and draw
    {prefix}_umap_canonical_bootstrap.png under checkpoint figures/.
    """
    if bootstrap_metrics.get("status") != "ok" or bootstrap_metrics.get("median_path") is None:
        return {"status": "skipped", "reason": "no bootstrap median path"}

    profile = cfg.profile
    cell_type = cfg.resolved_cell_type()
    figure_dir = cfg.figure_dir or setup_analysis_dirs(cfg.save_dir or get_save_dir(profile.spec))
    prefix = cfg.output_prefix()

    boot_entry = {
        "median_path": bootstrap_metrics["median_path"],
        "lower_path": bootstrap_metrics["lower_path"],
        "upper_path": bootstrap_metrics["upper_path"],
        "n_success": bootstrap_metrics.get("n_success", 0),
    }
    path_results = bootstrap_metrics.get("path_results") or []
    if path_results:
        boot_entry["central_path"] = path_results[0]

    cache_path = merge_bootstrap_medoid_into_lap_cache(
        figure_dir,
        prefix,
        boot_entry,
        dataset_key=profile.key,
        cell_type=cell_type,
        start_state=start_state,
        end_state=end_state,
    )

    payload = load_lap_path_cache(cache_path)
    lap_paths = _load_lap_paths_from_cache_payload(payload)
    if "min_path" not in lap_paths or "max_path" not in lap_paths:
        return {
            "status": "cache_only",
            "cache_path": cache_path,
            "reason": "lap cache missing min/max paths; bootstrap saved but UMAP not drawn",
        }

    figures = plot_lap_umap_figures(
        adata,
        cfg,
        lap_paths,
        prefix=prefix,
        figure_dir=figure_dir,
        start_state=start_state,
        end_state=end_state,
        cell_type=cell_type,
        only_suffixes=("_umap_canonical_bootstrap",),
    )
    return {"status": "ok", "cache_path": cache_path, "figures": figures}


def plot_lap_umap_figures(
    adata,
    cfg: CellTypeLAPConfig,
    lap_paths: dict,
    *,
    prefix: str,
    figure_dir: str,
    start_state: str,
    end_state: str,
    cell_type: str,
    only_suffixes: Optional[Tuple[str, ...]] = None,
) -> dict:
    """Render standard LAP UMAP figures; honor lap_umap_label_overrides.json."""
    profile = cfg.profile
    label_path = cfg.resolved_label_config(figure_dir)
    label_overrides = load_label_overrides(label_path)
    umap_coords = np.asarray(adata.obsm[profile.lap_display_key], dtype=float)
    states_to_plot = [start_state, end_state]
    min_path = lap_paths["min_path"]
    max_path = lap_paths["max_path"]
    overlay = {"min": min_path, "max": max_path}
    if "canonical_medoid" in lap_paths:
        overlay["canonical_medoid"] = lap_paths["canonical_medoid"]

    figure_specs = [
        (
            f"{prefix}_umap_all_paths",
            "plot_all",
            lambda ax: plot_all_lap_paths_on_umap(
                umap_coords,
                overlay,
                adata,
                palette=profile.stage_palette,
                ax=ax,
                states_to_plot=states_to_plot,
                title=f"{cell_type}: {start_state} → {end_state}",
                figure_key=f"{prefix}_umap_all_paths",
                label_overrides=label_overrides,
            ),
            overlay.get("canonical_medoid", min_path)["path"],
        ),
        (
            f"{prefix}_umap_min_max_paths",
            "plot_two",
            lambda ax: plot_two_paths_on_umap(
                umap_coords,
                min_path,
                max_path,
                states_to_plot,
                adata,
                palette=profile.stage_palette,
                ax=ax,
                title=f"{cell_type}: min / max path",
                figure_key=f"{prefix}_umap_min_max_paths",
                label_overrides=label_overrides,
            ),
            min_path["path"],
        ),
        (
            f"{prefix}_umap_min_path",
            "plot_min",
            lambda ax: plot_path_on_umap(
                umap_coords,
                min_path,
                adata,
                palette=profile.stage_palette,
                ax=ax,
                line_color="k",
                states_to_plot=states_to_plot,
                title=f"{cell_type}: min path",
                figure_key=f"{prefix}_umap_min_path",
                label_overrides=label_overrides,
            ),
            min_path["path"],
        ),
        (
            f"{prefix}_umap_max_path",
            "plot_max",
            lambda ax: plot_path_on_umap(
                umap_coords,
                max_path,
                adata,
                palette=profile.stage_palette,
                ax=ax,
                line_color="k",
                line_style="--",
                states_to_plot=states_to_plot,
                title=f"{cell_type}: max path",
                figure_key=f"{prefix}_umap_max_path",
                label_overrides=label_overrides,
            ),
            max_path["path"],
        ),
    ]

    if "canonical_medoid" in lap_paths:
        canonical = lap_paths["canonical_medoid"]
        figure_specs.append(
            (
                f"{prefix}_umap_canonical_medoid",
                "plot_canonical",
                lambda ax: plot_path_on_umap(
                    umap_coords,
                    canonical,
                    adata,
                    palette=profile.stage_palette,
                    ax=ax,
                    line_color="red",
                    states_to_plot=states_to_plot,
                    title=f"{cell_type}: canonical (medoid)",
                    figure_key=f"{prefix}_umap_canonical_medoid",
                    label_overrides=label_overrides,
                ),
                canonical["path"],
            )
        )

    bootstrap = lap_paths.get("bootstrap_medoid") or {}
    if bootstrap.get("median_path") is not None:
        figure_specs.append(
            (
                f"{prefix}_umap_canonical_bootstrap",
                "plot_bootstrap",
                lambda ax: plot_bootstrap_path_on_umap(
                    umap_coords,
                    bootstrap["median_path"],
                    bootstrap["lower_path"],
                    bootstrap["upper_path"],
                    adata,
                    palette=profile.stage_palette,
                    ax=ax,
                    central_path_result=bootstrap.get("central_path"),
                    states_to_plot=states_to_plot,
                    title=f"{cell_type}: bootstrap canonical (n={bootstrap.get('n_success', 0)})",
                    figure_key=f"{prefix}_umap_canonical_bootstrap",
                    label_overrides=label_overrides,
                ),
                bootstrap["median_path"],
            )
        )

    saved = {}
    for figure_key, _, draw_fn, ref_path in figure_specs:
        if only_suffixes is not None and not any(figure_key.endswith(suffix) for suffix in only_suffixes):
            continue
        fig, ax = plt.subplots(figsize=LAP_UMAP_FIGSIZE)
        draw_fn(ax)
        plt.tight_layout()
        out_path = f"{figure_dir}/{figure_key}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        saved[figure_key] = out_path
        if cfg.export_label_template:
            export_label_template(
                label_path,
                figure_key,
                np.asarray(ref_path, dtype=float),
                start_state,
                end_state,
            )

    if cfg.export_label_template:
        print(f"Label template / overrides: {label_path}")
        print("  Edit dx/dy or absolute x/y, then: --regenerate-umap-only")

    return saved


def _load_lap_paths_from_cache_payload(payload: dict) -> dict:
    paths = payload.get("paths") or {}
    lap_paths: dict = {}
    for key in ("min_path", "max_path"):
        if key in paths:
            lap_paths[key] = cache_path_result(paths[key])
    for key in ("canonical_medoid", "canonical_pseudotime"):
        if key in paths:
            lap_paths[key] = cache_path_result(paths[key])
    if payload.get("bootstrap_medoid"):
        b = payload["bootstrap_medoid"]
        lap_paths["bootstrap_medoid"] = {
            "median_path": np.asarray(b["median_path"], dtype=float),
            "lower_path": np.asarray(b["lower_path"], dtype=float),
            "upper_path": np.asarray(b["upper_path"], dtype=float),
            "central_path": cache_path_result(b["central_path"]) if b.get("central_path") else None,
            "n_success": b.get("n_success", 0),
        }
    return lap_paths


def regenerate_lap_umap_from_cache(
    adata,
    cfg: CellTypeLAPConfig,
    *,
    only_suffixes: Optional[Tuple[str, ...]] = None,
) -> dict:
    """Re-draw LAP UMAP figures from saved cache + label overrides (no LAP recompute)."""
    profile = cfg.profile
    figure_dir = cfg.figure_dir or setup_analysis_dirs(cfg.save_dir or get_save_dir(profile.spec))
    prefix = cfg.output_prefix()
    cache_path = _lap_cache_path(figure_dir, prefix)
    if not Path(cache_path).is_file():
        raise FileNotFoundError(
            f"No LAP cache at {cache_path}. Run LAP once without --regenerate-umap-only first."
        )
    payload = load_lap_path_cache(cache_path)
    lap_paths = _load_lap_paths_from_cache_payload(payload)
    start_state = payload.get("start_state") or cfg.resolved_start()
    end_state = payload.get("end_state") or cfg.resolved_end()
    cell_type = payload.get("cell_type") or cfg.resolved_cell_type()
    return plot_lap_umap_figures(
        adata,
        cfg,
        lap_paths,
        prefix=prefix,
        figure_dir=figure_dir,
        start_state=start_state,
        end_state=end_state,
        cell_type=cell_type,
        only_suffixes=only_suffixes,
    )


def run_de_and_heatmap(adata, cfg: CellTypeLAPConfig, ts_pseudotime: float, prefix: str, figure_dir: str):
    profile = cfg.profile
    before_cells = adata.obs_names[adata.obs["pseudotime"] < ts_pseudotime]
    after_cells = adata.obs_names[adata.obs["pseudotime"] > ts_pseudotime]
    n_sample = min(profile.de_n_sample, len(before_cells), len(after_cells))
    if n_sample < 5:
        warnings.warn("Too few cells for DE; skipping.", UserWarning)
        return None, []

    rng = np.random.default_rng(cfg.seed)
    sampled_before = rng.choice(before_cells, size=n_sample, replace=False)
    sampled_after = rng.choice(after_cells, size=n_sample, replace=False)
    adata_before = adata[sampled_before].copy()
    adata_after = adata[sampled_after].copy()
    adata_before.obs["group"] = "before"
    adata_after.obs["group"] = "after"
    adata_de = ad.concat([adata_before, adata_after], merge="same")

    sc.tl.rank_genes_groups(
        adata_de,
        groupby="group",
        groups=["after"],
        reference="before",
        method="wilcoxon",
        n_genes=adata_de.n_vars,
    )
    fc = profile.de_log2fc_cutoff
    de_up = sc.get.rank_genes_groups_df(
        adata_de, group="after", pval_cutoff=profile.de_pval_cutoff, log2fc_min=fc
    ).sort_values("logfoldchanges", ascending=False)
    de_down = sc.get.rank_genes_groups_df(
        adata_de, group="after", pval_cutoff=profile.de_pval_cutoff, log2fc_max=-fc
    ).sort_values("logfoldchanges")
    de_combined = pd.concat([de_up, de_down], ignore_index=True)
    if de_combined.empty:
        warnings.warn("No DE genes passed cutoffs.", UserWarning)
        return adata_de, []
    de_combined["abs_log2FC"] = de_combined["logfoldchanges"].abs()
    de_combined = de_combined.sort_values("abs_log2FC", ascending=False).reset_index(drop=True)
    top_genes = list(de_combined.head(cfg.top_n_de)["names"])
    de_combined.to_csv(f"{figure_dir}/{prefix}_de_genes.csv", index=False)
    plot_de_transition_heatmap(
        adata_de,
        top_genes,
        ts_pseudotime,
        save_path=f"{figure_dir}/{prefix}_transition_state_heatmap.png",
        vmax=profile.heatmap_vmax,
    )
    return adata_de, top_genes


def run_go_enrichment(gene_list: Sequence[str], cfg: CellTypeLAPConfig, prefix: str, figure_dir: str):
    if not cfg.run_go or not gene_list:
        return None
    try:
        go_enrich = gp.enrichr(
            gene_list=list(gene_list),
            gene_sets=["GO_Biological_Process_2025"],
            organism=cfg.profile.go_organism,
            outdir=None,
        )
        go_enrich.results.to_csv(f"{figure_dir}/{prefix}_go_results.csv", index=False)
        gp.barplot(
            go_enrich.results,
            column="Adjusted P-value",
            group="Gene_set",
            size=10,
            ofname=f"{figure_dir}/{prefix}_go_barplot.png",
        )
        return go_enrich
    except Exception as exc:
        warnings.warn(f"GO enrichment failed: {exc}", UserWarning)
        return None


def _ensure_transition_on_path(flat_indices: np.ndarray, transition_state_indice: int) -> tuple[np.ndarray, int]:
    flat_indices = np.asarray(flat_indices, dtype=int)
    hits = np.where(flat_indices == int(transition_state_indice))[0]
    if len(hits):
        return flat_indices, int(hits[0])
    flat_indices = np.append(flat_indices, int(transition_state_indice))
    return flat_indices, len(flat_indices) - 1


def plot_path_umap_panels_for_path(
    adata,
    analyzer,
    path_result,
    cfg: CellTypeLAPConfig,
    path_label: str,
    start_state: str,
    end_state: str,
    prefix: str,
    figure_dir: str,
) -> dict:
    """Draw legacy, enhanced, and enhanced-canonical path-cell UMAP panel figures."""
    profile = cfg.profile
    indices, _, _, _ = path_nearest_cell_indices(
        analyzer, path_result, start_state, end_state, cfg.clustering_key
    )
    flat_path_cells = np.asarray(indices, dtype=int).flatten()
    transition_state_indice = int(flat_path_cells[int(path_result["transition_state_idx"])])

    legacy_indices = path_cell_indices_for_panels(analyzer, path_result, mode="legacy")
    enhanced_indices = path_cell_indices_for_panels(analyzer, path_result, mode="enhanced")
    legacy_indices, legacy_transition_idx = _ensure_transition_on_path(
        legacy_indices, transition_state_indice
    )
    enhanced_indices, enhanced_transition_idx = _ensure_transition_on_path(
        enhanced_indices, transition_state_indice
    )

    legacy_cells = adata[legacy_indices]
    enhanced_cells = adata[enhanced_indices]

    legacy_path = f"{figure_dir}/{prefix}_{path_label}_path_umap_panels.png"
    enhanced_path = f"{figure_dir}/{prefix}_{path_label}_path_umap_panels_enhanced.png"
    enhanced_canonical_path = f"{figure_dir}/{prefix}_{path_label}_path_umap_panels_enhanced_canonical.png"

    plot_path_cell_panels(
        adata,
        legacy_cells,
        legacy_transition_idx,
        profile.stage_palette,
        save_path=legacy_path,
    )
    plot_path_cell_panels_enhanced(
        adata,
        enhanced_cells,
        enhanced_transition_idx,
        profile.stage_palette,
        save_path=enhanced_path,
    )
    plot_path_cell_panels_enhanced_canonical(
        adata,
        enhanced_cells,
        enhanced_transition_idx,
        profile.stage_palette,
        path_result,
        save_path=enhanced_canonical_path,
    )
    return {
        "legacy_path": legacy_path,
        "enhanced_path": enhanced_path,
        "enhanced_canonical_path": enhanced_canonical_path,
        "legacy_indices": legacy_indices,
        "enhanced_indices": enhanced_indices,
        "n_legacy_path_cells": int(len(legacy_indices)),
        "n_enhanced_path_cells": int(len(enhanced_indices)),
        "transition_state_indice": transition_state_indice,
    }


def regenerate_path_umap_panels_from_cache(adata, cfg: CellTypeLAPConfig) -> dict:
    """Re-draw legacy/enhanced panel PNGs plus umap_canonical_medoid from saved LAP cache."""
    profile = cfg.profile
    figure_dir = cfg.figure_dir or setup_analysis_dirs(cfg.save_dir or get_save_dir(profile.spec))
    prefix = cfg.output_prefix()
    cache_path = _lap_cache_path(figure_dir, prefix)
    if not Path(cache_path).is_file():
        raise FileNotFoundError(
            f"No LAP cache at {cache_path}. Run LAP once without --regenerate-panels-only first."
        )
    payload = load_lap_path_cache(cache_path)
    lap_paths = _load_lap_paths_from_cache_payload(payload)
    path_label = cfg.pioneer_path if cfg.pioneer_path in ("min", "max", "canonical") else "canonical"
    cache_key = {"min": "min_path", "max": "max_path", "canonical": "canonical_medoid"}[path_label]
    if cache_key not in payload.get("paths", {}):
        raise KeyError(f"Path {cache_key!r} not found in cache {cache_path}")

    path_result = cache_path_result(payload["paths"][cache_key])
    start_state = payload.get("start_state") or cfg.resolved_start()
    end_state = payload.get("end_state") or cfg.resolved_end()
    cell_type = payload.get("cell_type") or cfg.resolved_cell_type()
    analyzer = _make_analyzer(adata, profile)

    umap_figures = {}
    if path_label == "canonical" and "canonical_medoid" in lap_paths:
        umap_figures = plot_lap_umap_figures(
            adata,
            cfg,
            lap_paths,
            prefix=prefix,
            figure_dir=figure_dir,
            start_state=start_state,
            end_state=end_state,
            cell_type=cell_type,
            only_suffixes=("_umap_canonical_medoid",),
        )

    panel_info = plot_path_umap_panels_for_path(
        adata,
        analyzer,
        path_result,
        cfg,
        path_label,
        start_state,
        end_state,
        prefix,
        figure_dir,
    )
    panel_info["umap_canonical_medoid"] = umap_figures.get(f"{prefix}_umap_canonical_medoid")
    print(
        "Saved canonical path figures:",
        flush=True,
    )
    if panel_info.get("umap_canonical_medoid"):
        print(f"  umap: {panel_info['umap_canonical_medoid']}", flush=True)
    print(
        f"  legacy panel: {panel_info['legacy_path']} ({panel_info['n_legacy_path_cells']} cells)",
        flush=True,
    )
    print(
        f"  enhanced panel (cell polyline): {panel_info['enhanced_path']} "
        f"({panel_info['n_enhanced_path_cells']} cells)",
        flush=True,
    )
    print(
        f"  enhanced canonical panel: {panel_info['enhanced_canonical_path']} "
        f"({panel_info['n_enhanced_path_cells']} cells)",
        flush=True,
    )
    return panel_info


def run_pioneer_for_path(
    adata,
    analyzer,
    path_result,
    cfg: CellTypeLAPConfig,
    path_label: str,
    start_state: str,
    end_state: str,
    prefix: str,
    figure_dir: str,
):
    profile = cfg.profile
    pioneer_id = PioneerGeneIdentifier(analyzer)
    pioneer_result = pioneer_id.identify_pioneer_genes_along_path(
        path_result, top_n_genes=cfg.top_n_pioneer, use_3d=False
    )
    panel_info = plot_path_umap_panels_for_path(
        adata,
        analyzer,
        path_result,
        cfg,
        path_label,
        start_state,
        end_state,
        prefix,
        figure_dir,
    )
    flat_indices = panel_info["legacy_indices"]
    transition_state_indice = panel_info["transition_state_indice"]
    transition_state_cell = adata[transition_state_indice]
    ts_pseudotime = float(transition_state_cell.obs["pseudotime"].iloc[0])
    transition_state_indice_cell = adata.obs_names[transition_state_indice]

    pioneer_genes = list(pioneer_result["pioneer_genes"].keys())
    plot_pioneer_gene_heatmap(
        adata=adata,
        indices=flat_indices,
        pioneer_gene=pioneer_genes,
        transition_state_indice_cell=[transition_state_indice_cell],
        save_path=f"{figure_dir}/{prefix}_{path_label}_pioneer_heatmap.png",
    )
    pioneer_result["transition_pseudotime"] = ts_pseudotime
    pioneer_result["path_indices"] = flat_indices
    pioneer_result["path_indices_enhanced"] = panel_info["enhanced_indices"]
    pioneer_result["pioneer_genes_list"] = pioneer_genes
    pioneer_result["path_umap_panels"] = panel_info
    return pioneer_result, ts_pseudotime


def run_celltype_lap(cfg: CellTypeLAPConfig) -> dict:
    profile = cfg.profile
    save_dir = cfg.save_dir or get_save_dir(profile.spec)
    figure_dir = cfg.figure_dir or setup_analysis_dirs(save_dir)
    if cfg.figure_dir:
        setup_scanpy_figdir(str(Path(save_dir).resolve()))
    np.random.seed(cfg.seed)

    prefix = cfg.output_prefix()
    start_state = cfg.resolved_start()
    end_state = cfg.resolved_end()
    cell_type = cfg.resolved_cell_type()

    print(f"\n{'=' * 60}")
    print(f"Dataset: {profile.key} | Cell type: {cell_type} | Path: {start_state} → {end_state}")
    print(f"Output prefix: {prefix}")
    print(f"{'=' * 60}\n")

    adata = load_annotated_adata(profile, save_dir)
    adata = subset_and_preprocess(adata, cfg)

    results = {
        "config": cfg,
        "adata": adata,
        "prefix": prefix,
        "figure_dir": figure_dir,
    }

    if cfg.regenerate_umap_only:
        print("Regenerating LAP UMAP figures from cache + label overrides...")
        results["umap_figures"] = regenerate_lap_umap_from_cache(
            adata, cfg, only_suffixes=cfg.umap_only_suffixes
        )
        print(f"\nDone. Figures saved under: {figure_dir}")
        return results

    if cfg.regenerate_panels_only:
        print(
            "Regenerating canonical path figures "
            "(umap + legacy panel + enhanced panel + enhanced canonical panel) from cache..."
        )
        results["path_umap_panels"] = regenerate_path_umap_panels_from_cache(adata, cfg)
        print(f"\nDone. Figures saved under: {figure_dir}")
        return results

    if cfg.run_multi_path:
        print("Running multi-path analysis...")
        multi = analyze_multiple_cell_fate_transitions(
            adata,
            list(profile.path_pairs),
            clustering_key=cfg.clustering_key,
            n_path_points=max(50, math.ceil(0.01 * adata.n_obs)),
            palette=profile.stage_palette,
        )
        multi_pioneer = analyze_multiple_paths_with_pioneer_genes(
            adata,
            list(profile.path_pairs),
            clustering_key=cfg.clustering_key,
            top_n_genes=cfg.top_n_pioneer,
        )
        results["multi_path"] = multi
        results["multi_pioneer"] = multi_pioneer

    lap_paths = compute_all_lap_paths(adata, cfg)
    analyzer = lap_paths["analyzer"]
    min_path = lap_paths["min_path"]
    max_path = lap_paths["max_path"]
    results["analyzer"] = analyzer
    results["lap_paths"] = lap_paths
    results["analyzer_min"] = analyzer
    results["analyzer_max"] = analyzer
    results["min_path"] = min_path
    results["max_path"] = max_path
    for key in ("canonical_medoid", "canonical_pseudotime"):
        if key in lap_paths:
            results[key] = lap_paths[key]

    cache_path = save_lap_figure_cache(
        figure_dir, prefix, lap_paths, start_state, end_state, cell_type, profile.key
    )
    results["lap_cache"] = cache_path
    results["umap_figures"] = plot_lap_umap_figures(
        adata,
        cfg,
        lap_paths,
        prefix=prefix,
        figure_dir=figure_dir,
        start_state=start_state,
        end_state=end_state,
        cell_type=cell_type,
    )

    if "canonical_medoid" in lap_paths:
        deriv_path = derivative_curve_plot_path(figure_dir, profile.key, cell_type)
        try:
            plot_potential_derivative_with_canonical_path(
                adata,
                analyzer,
                lap_paths["canonical_medoid"],
                cell_type_label=cell_type,
                clustering_key=cfg.clustering_key,
                save_path=deriv_path,
            )
            results["potential_derivative_plot"] = deriv_path
            print(f"Saved potential–pseudotime derivative plot: {deriv_path}")
        except Exception as exc:
            warnings.warn(
                f"Potential derivative plot failed: {exc}",
                UserWarning,
                stacklevel=2,
            )

    if cfg.paths_only:
        print(f"\nDone (paths only). Figures saved under: {figure_dir}")
        return results

    pioneer_paths = []
    if cfg.pioneer_path in ("min", "both"):
        pioneer_paths.append(("min", analyzer, min_path))
    if cfg.pioneer_path in ("max", "both"):
        pioneer_paths.append(("max", analyzer, max_path))
    if cfg.pioneer_path == "canonical" and "canonical_medoid" in lap_paths:
        pioneer_paths.append(("canonical", analyzer, lap_paths["canonical_medoid"]))

    all_top_genes = []
    for path_label, analyzer, path_result in pioneer_paths:
        print(f"Pioneer genes on {path_label} path...")
        pioneer_result, ts_pt = run_pioneer_for_path(
            adata, analyzer, path_result, cfg, path_label, start_state, end_state, prefix, figure_dir
        )
        results[f"pioneer_{path_label}"] = pioneer_result
        if cfg.run_de:
            adata_de, top_genes = run_de_and_heatmap(adata, cfg, ts_pt, f"{prefix}_{path_label}", figure_dir)
            results[f"de_{path_label}"] = adata_de
            if top_genes:
                all_top_genes.extend(top_genes)
                if cfg.run_go:
                    results[f"go_{path_label}"] = run_go_enrichment(
                        top_genes, cfg, f"{prefix}_{path_label}", figure_dir
                    )

    if all_top_genes and cfg.run_go and cfg.pioneer_path == "both":
        unique_genes = list(dict.fromkeys(all_top_genes))
        results["go_combined"] = run_go_enrichment(unique_genes, cfg, f"{prefix}_combined", figure_dir)

    print(f"\nDone. Figures saved under: {figure_dir}")
    return results


def build_arg_parser(profile: Optional[DatasetProfile] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cell-type LAP / pioneer / DE / GO workflow")
    if profile is None:
        parser.add_argument(
            "--dataset",
            required=True,
            choices=sorted(DATASET_REGISTRY.keys()),
            help="Dataset name",
        )
    parser.add_argument("--list-cell-types", action="store_true", help="List cell types and exit")
    parser.add_argument("--cell-type", type=str, default=None, help="Cell type to analyze")
    parser.add_argument("--cell-type-column", type=str, default=None, help="Override obs column for cell types")
    parser.add_argument("--start", type=str, default=None, help="Start temporal/state label")
    parser.add_argument("--end", type=str, default=None, help="End temporal/state label")
    parser.add_argument("--multi-path", action="store_true", help="Also run multi-path LAP analysis")
    parser.add_argument(
        "--all-cell-types",
        action="store_true",
        help="Run analysis for every cell type in the dataset profile (e.g. EOC, Immune, Stromal for HGSOC)",
    )
    parser.add_argument(
        "--paths-only",
        action="store_true",
        help="Only compute and plot min/max/canonical LAP paths on UMAP (skip pioneer/DE/GO)",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip bootstrap canonical path stability analysis",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=None,
        help="Bootstrap replicates for canonical path (default: profile value, usually 50)",
    )
    parser.add_argument(
        "--pioneer-path",
        choices=("min", "max", "canonical", "both"),
        default="canonical",
        help="Which LAP path to use for pioneer/DE/GO (both = min+max; use with --pioneer-path min for min only)",
    )
    parser.add_argument("--no-go", action="store_true", help="Skip GO enrichment")
    parser.add_argument("--no-de", action="store_true", help="Skip differential expression")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-tag", type=str, default=None, help="Custom output filename prefix")
    parser.add_argument(
        "--label-config",
        type=str,
        default=None,
        help="JSON file for custom start/end label positions (default: figures/lap_umap_label_overrides.json)",
    )
    parser.add_argument(
        "--regenerate-panels-only",
        action="store_true",
        help="Re-draw legacy + enhanced path-cell UMAP panels from saved LAP cache",
    )
    parser.add_argument(
        "--regenerate-umap-only",
        action="store_true",
        help="Re-draw LAP UMAP PNGs from saved cache + label overrides (skip LAP recompute)",
    )
    parser.add_argument(
        "--no-export-label-template",
        action="store_true",
        help="Do not update lap_umap_label_overrides.json after plotting",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint root (default: recommended checkpoint for this dataset)",
    )
    return parser


def config_from_args(profile: DatasetProfile, args: argparse.Namespace) -> CellTypeLAPConfig:
    if args.cell_type_column:
        profile = replace(profile, cell_type_column=args.cell_type_column)
    if args.bootstrap_n is not None:
        profile = replace(profile, bootstrap_n=args.bootstrap_n)
    return CellTypeLAPConfig(
        profile=profile,
        cell_type=args.cell_type,
        start_state=args.start,
        end_state=args.end,
        pioneer_path=args.pioneer_path,
        run_multi_path=args.multi_path,
        paths_only=args.paths_only,
        run_bootstrap=not args.no_bootstrap,
        run_go=not args.no_go,
        run_de=not args.no_de,
        seed=args.seed,
        save_dir=(
            resolve_checkpoint_dir(profile.spec, args.checkpoint_dir)
            if args.checkpoint_dir
            else None
        ),
        output_tag=args.output_tag,
        label_config_path=args.label_config,
        regenerate_umap_only=args.regenerate_umap_only,
        regenerate_panels_only=args.regenerate_panels_only,
        export_label_template=not args.no_export_label_template,
    )


def main_from_profile(profile: DatasetProfile):
    parser = build_arg_parser(profile)
    args = parser.parse_args()
    if args.list_cell_types:
        list_cell_types(profile)
        return
    if args.all_cell_types:
        for cell_type in profile.available_cell_types:
            cfg = config_from_args(profile, args)
            cfg = replace(cfg, cell_type=cell_type)
            run_celltype_lap(cfg)
        return
    cfg = config_from_args(profile, args)
    run_celltype_lap(cfg)


def main_unified():
    parser = build_arg_parser()
    args = parser.parse_args()
    profile = DATASET_REGISTRY[args.dataset]
    if args.list_cell_types:
        list_cell_types(profile)
        return
    cfg = config_from_args(profile, args)
    run_celltype_lap(cfg)


if __name__ == "__main__":
    main_unified()
