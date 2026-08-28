"""
Pre-LAP disease-remodeling cell-type screening.

This module ranks cell types by stage-associated expression, potential, pseudotime,
and embedding separation between a chosen start and end state. It is a screening step
only: it does not identify transition states, causal regulators, or validated
remodeling paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import pearsonr


SavePath = Union[str, Path]


def _as_dense(X) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray(X.toarray())
    return np.asarray(X)


def _infer_start_end_states(
    adata,
    stage_key: str,
    pseudotime_key: str,
    start_state: Optional[str],
    end_state: Optional[str],
) -> Tuple[str, str]:
    if start_state is not None and end_state is not None:
        return str(start_state), str(end_state)

    stages = adata.obs[stage_key].astype(str)
    uniq = list(dict.fromkeys(stages.tolist()))
    if len(uniq) < 2:
        raise ValueError(f"Need at least two stages in adata.obs[{stage_key!r}]")

    if pseudotime_key in adata.obs.columns:
        medians = {
            s: float(np.median(pd.to_numeric(adata.obs.loc[stages == s, pseudotime_key], errors="coerce")))
            for s in uniq
        }
        ordered = sorted(uniq, key=lambda s: medians.get(s, np.nan))
    else:
        ordered = uniq

    start = str(start_state if start_state is not None else ordered[0])
    end = str(end_state if end_state is not None else ordered[-1])
    if start == end:
        raise ValueError("start_state and end_state must differ")
    return start, end


def _resolve_embedding(adata) -> Tuple[str, np.ndarray]:
    for key in ("X_latent_pca", "X_pca", "X_umap"):
        if key in adata.obsm:
            return key, np.asarray(adata.obsm[key], dtype=float)
    raise KeyError(
        "No embedding found in adata.obsm; expected one of "
        "X_latent_pca, X_pca, X_umap"
    )


def _mean_expression_profile(adata) -> np.ndarray:
    X = _as_dense(adata.X)
    if X.size == 0:
        return np.array([], dtype=float)
    return np.asarray(X.mean(axis=0), dtype=float).reshape(-1)


def _expression_shift(raw_shift: float, global_median_shift: float) -> float:
    if not np.isfinite(raw_shift):
        return float("nan")
    denom = float(global_median_shift) if np.isfinite(global_median_shift) else 0.0
    if denom <= 1e-12:
        return float(raw_shift)
    return float(raw_shift / denom)


def _expression_pcc_shift(start_mean: np.ndarray, end_mean: np.ndarray) -> float:
    if start_mean.size == 0 or end_mean.size == 0:
        return float("nan")
    if np.allclose(start_mean, start_mean[0]) or np.allclose(end_mean, end_mean[0]):
        corr = 1.0 if np.allclose(start_mean, end_mean) else 0.0
    else:
        corr, _ = pearsonr(start_mean, end_mean)
        if not np.isfinite(corr):
            corr = 0.0
    return float(1.0 - corr)


def _within_celltype_stage_separation(
    coords: np.ndarray,
    start_mask: np.ndarray,
    end_mask: np.ndarray,
) -> float:
    start_pts = coords[start_mask]
    end_pts = coords[end_mask]
    if len(start_pts) == 0 or len(end_pts) == 0:
        return float("nan")

    c_start = start_pts.mean(axis=0)
    c_end = end_pts.mean(axis=0)
    between = float(np.linalg.norm(c_end - c_start))
    within_start = float(np.mean(np.linalg.norm(start_pts - c_start, axis=1)))
    within_end = float(np.mean(np.linalg.norm(end_pts - c_end, axis=1)))
    within = 0.5 * (within_start + within_end)
    return float(between / max(within, 1e-8))


def _lightweight_deg_count(
    adata,
    stage_key: str,
    start_state: str,
    end_state: str,
    *,
    min_cells: int,
    padj_cutoff: float = 0.05,
    logfc_cutoff: float = 0.25,
) -> Tuple[int, bool]:
    sub = adata[
        adata.obs[stage_key].astype(str).isin([start_state, end_state])
    ].copy()
    n_start = int((sub.obs[stage_key].astype(str) == start_state).sum())
    n_end = int((sub.obs[stage_key].astype(str) == end_state).sum())
    if n_start < min_cells or n_end < min_cells:
        return 0, False

    try:
        sc.tl.rank_genes_groups(
            sub,
            groupby=stage_key,
            groups=[end_state],
            reference=start_state,
            method="wilcoxon",
            use_raw=False,
        )
        result = sub.uns["rank_genes_groups"]
        names = result["names"][end_state]
        logfc = result["logfoldchanges"][end_state]
        pvals_adj = result["pvals_adj"][end_state]
        count = 0
        for name, lf, padj in zip(names, logfc, pvals_adj):
            if name is None:
                continue
            if padj is None or not np.isfinite(padj):
                continue
            if lf is None or not np.isfinite(lf):
                continue
            if float(padj) < padj_cutoff and abs(float(lf)) > logfc_cutoff:
                count += 1
        return int(count), True
    except Exception:
        return 0, False


def screen_disease_remodeling_celltypes(
    adata,
    cell_type_key: str,
    stage_key: str,
    start_state: Optional[str] = None,
    end_state: Optional[str] = None,
    potential_key: str = "potential",
    pseudotime_key: str = "pseudotime",
    min_cells: int = 50,
    top_k: Optional[int] = None,
    quantile: float = 0.70,
    save_path: Optional[SavePath] = None,
    candidate_cell_types: Optional[Sequence[str]] = None,
    cell_type_stage_paths: Optional[Dict[str, Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    Screen cell types for disease-associated remodeling prior to LAP analysis.

    This is a pre-LAP disease-remodeling screening step; it does not claim
    transition state or causality. Rankings combine stage-separated expression,
    learned dynamical potential, pseudotime, embedding separation, and optional
    lightweight DEG evidence.

    Parameters
    ----------
    adata
        Annotated data matrix with cell types, stages, expression, and optional
        potential / pseudotime / embedding coordinates.
    cell_type_key
        Column in ``adata.obs`` with cell-type labels.
    stage_key
        Column in ``adata.obs`` with stage or condition labels.
    start_state, end_state
        Reference and target stages. If omitted, inferred from stage order
        (median pseudotime when available, else observation order).
    potential_key
        Column in ``adata.obs`` with model-inferred dynamical potential ``U``.
        Shifts are reported as ``dynamical_potential_shift`` (not expression change).
    pseudotime_key
        Column in ``adata.obs`` with pseudotime values.
    min_cells
        Minimum total cells per cell type to include in screening. Lightweight DEG
        testing additionally requires both start and end groups to have at least
        ``min_cells`` cells.
    top_k
        If set, mark the top ``k`` eligible cell types by ``final_score`` as
        ``selected_for_lap`` regardless of quantile threshold.
    quantile
        Quantile threshold on ``final_score`` among eligible cell types when
        ``top_k`` is not provided.
    save_path
        Optional CSV output path.
    candidate_cell_types
        Optional subset of cell types to screen (default: all labels in ``cell_type_key``).
    cell_type_stage_paths
        Optional mapping ``cell_type -> (start_state, end_state)`` for cell-type-specific
        LAP path endpoints.

    Returns
    -------
    pandas.DataFrame
        One row per screened cell type with component scores and selection flag.
    """
    if cell_type_key not in adata.obs.columns:
        raise KeyError(f"Missing adata.obs[{cell_type_key!r}]")
    if stage_key not in adata.obs.columns:
        raise KeyError(f"Missing adata.obs[{stage_key!r}]")

    default_start, default_end = _infer_start_end_states(
        adata, stage_key, pseudotime_key, start_state, end_state
    )
    embedding_key, _ = _resolve_embedding(adata)

    if candidate_cell_types is not None:
        cell_types = [str(ct) for ct in candidate_cell_types]
    else:
        cell_types = adata.obs[cell_type_key].astype(str).unique().tolist()
    raw_rows: List[Dict[str, Any]] = []

    for cell_type in cell_types:
        if cell_type_stage_paths and cell_type in cell_type_stage_paths:
            start_state, end_state = cell_type_stage_paths[cell_type]
        else:
            start_state, end_state = default_start, default_end

        mask = adata.obs[cell_type_key].astype(str) == cell_type
        sub = adata[mask]
        n_cells = int(sub.n_obs)
        row: Dict[str, Any] = {
            "cell_type": cell_type,
            "n_cells": n_cells,
            "start_state": start_state,
            "end_state": end_state,
            "embedding_key": embedding_key,
            "eligible": n_cells >= min_cells,
        }

        if n_cells < min_cells:
            row.update(
                {
                    "expression_shift_score": np.nan,
                    "expression_pcc_shift": np.nan,
                    "dynamical_potential_shift": np.nan,
                    "potential_shift_score": np.nan,
                    "pseudotime_shift_score": np.nan,
                    "within_celltype_stage_separation": np.nan,
                    "deg_count": np.nan,
                    "deg_available": False,
                    "deg_score_normalized": np.nan,
                    "disease_remodeling_screen_score": np.nan,
                    "final_score": np.nan,
                    "selected_for_lap": False,
                }
            )
            raw_rows.append(row)
            continue

        start_mask = sub.obs[stage_key].astype(str) == start_state
        end_mask = sub.obs[stage_key].astype(str) == end_state
        if int(start_mask.sum()) == 0 or int(end_mask.sum()) == 0:
            row["eligible"] = False
            row.update(
                {
                    "expression_shift_score": np.nan,
                    "expression_pcc_shift": np.nan,
                    "dynamical_potential_shift": np.nan,
                    "potential_shift_score": np.nan,
                    "pseudotime_shift_score": np.nan,
                    "within_celltype_stage_separation": np.nan,
                    "deg_count": np.nan,
                    "deg_available": False,
                    "deg_score_normalized": np.nan,
                    "disease_remodeling_screen_score": np.nan,
                    "final_score": np.nan,
                    "selected_for_lap": False,
                }
            )
            raw_rows.append(row)
            continue

        start_mean = _mean_expression_profile(sub[start_mask])
        end_mean = _mean_expression_profile(sub[end_mask])
        raw_expr_shift = float(np.mean(np.abs(end_mean - start_mean)))

        dynamical_potential_shift = np.nan
        if potential_key in sub.obs.columns:
            pot = pd.to_numeric(sub.obs[potential_key], errors="coerce")
            dynamical_potential_shift = float(
                abs(pot[end_mask].median() - pot[start_mask].median())
            )

        pseudotime_shift_score = np.nan
        if pseudotime_key in sub.obs.columns:
            pt = pd.to_numeric(sub.obs[pseudotime_key], errors="coerce")
            pseudotime_shift_score = float(
                abs(pt[end_mask].median() - pt[start_mask].median())
            )

        _, coords = _resolve_embedding(sub)
        stage_sep = _within_celltype_stage_separation(
            coords, np.asarray(start_mask, dtype=bool), np.asarray(end_mask, dtype=bool)
        )

        deg_count, deg_available = _lightweight_deg_count(
            sub,
            stage_key,
            start_state,
            end_state,
            min_cells=min_cells,
        )

        raw_rows.append(
            {
                **row,
                "_raw_expression_shift": raw_expr_shift,
                "expression_pcc_shift": _expression_pcc_shift(start_mean, end_mean),
                "dynamical_potential_shift": dynamical_potential_shift,
                "potential_shift_score": dynamical_potential_shift,
                "pseudotime_shift_score": pseudotime_shift_score,
                "within_celltype_stage_separation": stage_sep,
                "deg_count": deg_count if deg_available else np.nan,
                "deg_available": deg_available,
            }
        )

    df = pd.DataFrame(raw_rows)
    eligible = df["eligible"].fillna(False).astype(bool)
    raw_shifts = df.loc[eligible, "_raw_expression_shift"]
    global_median_shift = float(np.nanmedian(raw_shifts)) if len(raw_shifts) else float("nan")

    expr_shift_scores = []
    screen_scores = []
    final_scores = []

    max_deg = 0
    if "deg_count" in df.columns:
        deg_vals = df.loc[eligible & df["deg_available"].fillna(False), "deg_count"]
        if len(deg_vals):
            max_deg = int(np.nanmax(deg_vals))

    for _, row in df.iterrows():
        if not row.get("eligible"):
            expr_shift_scores.append(np.nan)
            screen_scores.append(np.nan)
            final_scores.append(np.nan)
            continue

        expr_shift = _expression_shift(row["_raw_expression_shift"], global_median_shift)
        expr_shift_scores.append(expr_shift)

        components = [
            0.35 * (expr_shift if np.isfinite(expr_shift) else 0.0),
            0.20 * (row["expression_pcc_shift"] if np.isfinite(row["expression_pcc_shift"]) else 0.0),
            0.15 * (row["potential_shift_score"] if np.isfinite(row["potential_shift_score"]) else 0.0),
            0.15 * (row["pseudotime_shift_score"] if np.isfinite(row["pseudotime_shift_score"]) else 0.0),
            0.15
            * (
                row["within_celltype_stage_separation"]
                if np.isfinite(row["within_celltype_stage_separation"])
                else 0.0
            ),
        ]
        screen_score = float(np.sum(components))
        screen_scores.append(screen_score)

        if row.get("deg_available") and np.isfinite(row.get("deg_count", np.nan)):
            deg_norm = float(row["deg_count"]) / max(max_deg, 1)
            final_score = float(0.75 * screen_score + 0.25 * deg_norm)
        else:
            final_score = screen_score
        final_scores.append(final_score)

    df["expression_shift_score"] = expr_shift_scores
    df["disease_remodeling_screen_score"] = screen_scores
    df["final_score"] = final_scores

    deg_norm_col = []
    for _, row in df.iterrows():
        if row.get("deg_available") and np.isfinite(row.get("deg_count", np.nan)):
            deg_norm_col.append(float(row["deg_count"]) / max(max_deg, 1))
        else:
            deg_norm_col.append(np.nan)
    df["deg_score_normalized"] = deg_norm_col

    if "_raw_expression_shift" in df.columns:
        df = df.drop(columns=["_raw_expression_shift"])

    eligible_scores = df.loc[eligible, "final_score"].dropna()
    df["rank"] = np.nan
    if len(eligible_scores):
        ranked = eligible_scores.sort_values(ascending=False)
        rank_map = {idx: r for r, idx in enumerate(ranked.index, start=1)}
        df.loc[list(rank_map.keys()), "rank"] = [rank_map[i] for i in rank_map]

    selected = np.zeros(len(df), dtype=bool)
    if top_k is not None and len(eligible_scores):
        top_idx = eligible_scores.sort_values(ascending=False).head(int(top_k)).index
        selected[df.index.isin(top_idx)] = True
    elif len(eligible_scores):
        threshold = float(eligible_scores.quantile(quantile))
        selected = eligible & (df["final_score"] >= threshold).fillna(False).to_numpy()

    df["selected_for_lap"] = selected

    sort_cols = ["selected_for_lap", "final_score", "n_cells"]
    df = df.sort_values(sort_cols, ascending=[False, False, False]).reset_index(drop=True)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    return df
