"""
Reliability-aware endpoint selection for LAP candidate paths.

Heuristic candidate generators (legacy, hybrid, density_core, farthest_core, etc.)
propose start/end cell pairs. Reliability-aware scoring combines pre-LAP geometry,
post-LAP path diagnostics, and full path reliability to select endpoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from landscape_core import (
    estimate_potential_from_density,
    mean_neighbor_spacing,
    pick_cluster_representative_index,
    stage_core_cell_indices,
)


@dataclass
class CandidateFullReliabilityResult:
    mode: str = ""
    status: str = "ok"

    full_path_degenerate: Optional[bool] = None
    degeneracy_reason: str = ""
    endpoint_overlap_fraction: Optional[float] = None

    bootstrap_stability_class: str = "NA"
    bootstrap_relative_deviation: Optional[float] = None
    bootstrap_score: Optional[float] = None

    manifold_support_ratio: Optional[float] = None
    manifold_score: Optional[float] = None

    center_separation_score: Optional[float] = None
    density_calibration_score: Optional[float] = None

    reliability_score: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class EndpointCandidate:
    mode: str
    start_idx: int
    end_idx: int
    start_state: str
    end_state: str
    endpoint_distance: float
    endpoint_distance_ratio: float
    separation_score: float
    density_core_score: float
    pseudotime_order_score: float
    stage_purity_score: float
    endpoint_pre_score: float
    lightweight_path_degenerate: Optional[bool] = None
    path_degenerate: Optional[bool] = None
    full_path_degenerate: Optional[bool] = None
    degeneracy_source: str = ""
    full_degeneracy_reason: str = ""
    full_reliability_score: Optional[float] = None
    bootstrap_stability_class: str = "NA"
    bootstrap_relative_deviation: Optional[float] = None
    endpoint_overlap_fraction: Optional[float] = None
    path_length: Optional[float] = None
    path_length_ratio: Optional[float] = None
    manifold_support_ratio: Optional[float] = None
    remodeling_center_idx: Optional[int] = None
    remodeling_center_distance_to_start: Optional[float] = None
    remodeling_center_distance_to_end: Optional[float] = None
    center_separation_score: Optional[float] = None
    post_lap_score: Optional[float] = None
    endpoint_final_score: Optional[float] = None
    is_separable: bool = False
    selected: bool = False
    selected_reason: str = ""
    warnings: List[str] = field(default_factory=list)
    path_result: Optional[dict] = field(default=None, repr=False)
    lap_reliability: Optional[dict] = field(default=None, repr=False)
    full_reliability: Optional[CandidateFullReliabilityResult] = field(default=None, repr=False)
    endpoint_selection_reliability_mode: str = "lightweight"

    # backward-compat alias
    @property
    def endpoint_score(self) -> float:
        return float(self.endpoint_final_score if self.endpoint_final_score is not None else self.endpoint_pre_score)


def compute_endpoint_separability(
    coords: np.ndarray,
    start_indices: np.ndarray,
    end_indices: np.ndarray,
    neighbor_spacing: float,
    threshold: float = 2.0,
) -> dict:
    coords = np.asarray(coords, dtype=float)
    start_indices = np.asarray(start_indices, dtype=int)
    end_indices = np.asarray(end_indices, dtype=int)
    if len(start_indices) == 0 or len(end_indices) == 0:
        return {
            "start_centroid": np.full(coords.shape[1], np.nan),
            "end_centroid": np.full(coords.shape[1], np.nan),
            "endpoint_distance": np.nan,
            "endpoint_distance_ratio": np.nan,
            "is_separable": False,
        }
    start_centroid = coords[start_indices].mean(axis=0)
    end_centroid = coords[end_indices].mean(axis=0)
    endpoint_distance = float(np.linalg.norm(start_centroid - end_centroid))
    spacing = float(neighbor_spacing) if np.isfinite(neighbor_spacing) and neighbor_spacing > 0 else 1.0
    endpoint_distance_ratio = endpoint_distance / spacing
    return {
        "start_centroid": start_centroid,
        "end_centroid": end_centroid,
        "endpoint_distance": endpoint_distance,
        "endpoint_distance_ratio": endpoint_distance_ratio,
        "is_separable": endpoint_distance_ratio >= float(threshold),
    }


def _separation_score(endpoint_distance_ratio: float) -> float:
    if not np.isfinite(endpoint_distance_ratio):
        return 0.0
    return float(np.clip(endpoint_distance_ratio / 4.0, 0.0, 1.0))


def compute_endpoint_pre_score(candidate: EndpointCandidate) -> float:
    """
    Pre-LAP endpoint score from geometry and stage structure.

    pre_score = 0.30*separation + 0.15*density + 0.30*pseudotime_order + 0.25*stage_purity

    For disease-progression LAP, temporal/stage ordering and local stage purity should
    dominate over KDE density: learned U is not always calibrated to -log density.
    """
    return float(
        0.30 * candidate.separation_score
        + 0.15 * candidate.density_core_score
        + 0.30 * candidate.pseudotime_order_score
        + 0.25 * candidate.stage_purity_score
    )


def compute_post_lap_score(candidate: EndpointCandidate, eps: float = 1e-8) -> float:
    """
    Post-LAP path diagnostic score (lightweight path degeneracy / manifold / overlap).

    Continuous ranking contribution only — no hard valid/invalid endpoint gate.
    """
    degenerate = candidate.lightweight_path_degenerate
    if degenerate is None:
        degenerate = candidate.path_degenerate
    if degenerate is True:
        nondegenerate_score = 0.2
    elif degenerate is False:
        nondegenerate_score = 1.0
    else:
        nondegenerate_score = 0.5

    if candidate.manifold_support_ratio is not None and np.isfinite(candidate.manifold_support_ratio):
        manifold_score = float(1.0 / (1.0 + float(candidate.manifold_support_ratio)))
    else:
        manifold_score = 0.5

    center_sep = candidate.center_separation_score
    if center_sep is None or not np.isfinite(center_sep):
        center_sep = 0.5

    overlap = float(candidate.endpoint_overlap_fraction or 0.0)
    endpoint_overlap_score = 1.0 - float(np.clip(overlap, 0.0, 1.0))

    return float(
        0.40 * nondegenerate_score
        + 0.25 * manifold_score
        + 0.20 * float(center_sep)
        + 0.15 * endpoint_overlap_score
    )


def compute_endpoint_final_score(candidate: EndpointCandidate) -> float:
    """
    Reliability-aware endpoint final score.

    When full reliability is available:
      final_score = 0.20*pre + 0.20*post + 0.60*reliability_score
    Otherwise:
      final_score = 0.30*pre + 0.70*post

    reliability_score aggregates path degeneracy, bootstrap stability, manifold support,
    center separation, and density calibration from lap_reliability diagnostics.
    """
    pre = candidate.endpoint_pre_score if candidate.endpoint_pre_score is not None else compute_endpoint_pre_score(candidate)
    post = candidate.post_lap_score if candidate.post_lap_score is not None else 0.5
    reliability = candidate.full_reliability_score
    if reliability is not None and np.isfinite(reliability):
        return float(0.20 * pre + 0.20 * post + 0.60 * float(reliability))
    return float(0.30 * pre + 0.70 * post)


def compute_candidate_full_reliability_score(
    candidate: EndpointCandidate,
    lap_reliability: Optional[dict],
    path_result: Optional[dict] = None,
    density_calibration: Optional[dict] = None,
) -> CandidateFullReliabilityResult:
    """Score candidate using full lap_reliability diagnostics.

    Density calibration (U vs -log KDE) is included at low weight (0.05): it is
    diagnostic when Spearman is weak and should not dominate endpoint ranking.
    """
    mode = str(candidate.mode)
    status = "ok"
    warnings: List[str] = []
    if lap_reliability is None:
        return CandidateFullReliabilityResult(mode=mode, status="failed", reliability_score=0.0, warnings=["no_lap_reliability"])

    lap_rel = lap_reliability or {}
    status = str(lap_rel.get("status", "ok"))
    degeneracy = lap_rel.get("degeneracy") or {}
    bootstrap = lap_rel.get("bootstrap_comparison") or lap_rel.get("bootstrap") or {}
    interpretation = lap_rel.get("interpretation") or {}

    full_path_degenerate = degeneracy.get("is_degenerate")
    if full_path_degenerate is None and path_result is not None:
        full_path_degenerate = path_result.get("path_degenerate")
    if full_path_degenerate is not None:
        full_path_degenerate = bool(full_path_degenerate)

    degeneracy_reason = str(
        degeneracy.get("recommended_interpretation")
        or interpretation.get("recommended_sentence")
        or degeneracy.get("degeneracy_reason", "")
    )
    endpoint_overlap_fraction = degeneracy.get("endpoint_overlap_fraction")
    if endpoint_overlap_fraction is None:
        endpoint_overlap_fraction = max(
            float(degeneracy.get("start_transition_jaccard", 0) or 0),
            float(degeneracy.get("transition_end_jaccard", 0) or 0),
            float(degeneracy.get("start_end_jaccard", 0) or 0),
        )

    bootstrap_stability_class = str(
        bootstrap.get("path_stability_class")
        or bootstrap.get("bootstrap_stability_class")
        or "NA"
    )
    bootstrap_relative_deviation = bootstrap.get("relative_path_deviation")
    if bootstrap_relative_deviation is None:
        bootstrap_relative_deviation = bootstrap.get("canonical_bootstrap_deviation")

    if bootstrap_stability_class in {"stable", "moderately_stable"}:
        bootstrap_score = 1.0
    elif bootstrap_stability_class == "unstable":
        bootstrap_score = 0.2
    else:
        bootstrap_score = 0.5

    manifold_support_ratio = lap_rel.get("manifold_support_ratio")
    if manifold_support_ratio is None:
        manifold_support_ratio = degeneracy.get("manifold_support_ratio")
    if manifold_support_ratio is None and path_result is not None:
        manifold_support_ratio = path_result.get("manifold_support_ratio")
    if manifold_support_ratio is None or not np.isfinite(manifold_support_ratio):
        manifold_score = 0.5
    else:
        manifold_score = float(1.0 / (1.0 + float(manifold_support_ratio)))

    center_separation_score = candidate.center_separation_score
    if center_separation_score is None:
        center_separation_score = 0.5
    elif not np.isfinite(center_separation_score):
        center_separation_score = 0.5
    else:
        center_separation_score = float(center_separation_score)

    # Density calibration is diagnostic only: when U vs -log KDE Spearman is weak,
    # it should not dominate endpoint selection over path/manifold evidence.
    density_calibration_score = None
    if density_calibration:
        density_calibration_score = density_calibration.get("density_calibration_score")
        if density_calibration_score is None:
            spearman = density_calibration.get("spearman_U_vs_neglogp")
            if spearman is not None and np.isfinite(spearman):
                density_calibration_score = float(np.clip((float(spearman) + 0.1) / 0.4, 0.0, 1.0))
    if density_calibration_score is None:
        density_calibration_score = 0.5
    else:
        density_calibration_score = float(density_calibration_score)

    if full_path_degenerate is True:
        nondegenerate_score = 0.2
    elif full_path_degenerate is False:
        nondegenerate_score = 1.0
    else:
        nondegenerate_score = 0.5
    # Down-weight density calibration (0.05): keep as a tie-breaker/diagnostic, not a driver.
    reliability_score = float(
        0.30 * nondegenerate_score
        + 0.25 * bootstrap_score
        + 0.20 * manifold_score
        + 0.20 * center_separation_score
        + 0.05 * density_calibration_score
    )
    if full_path_degenerate is True:
        reliability_score = min(reliability_score, 0.55)
    if status != "ok":
        reliability_score = min(reliability_score, 0.35)

    return CandidateFullReliabilityResult(
        mode=mode,
        status=status,
        full_path_degenerate=full_path_degenerate,
        degeneracy_reason=degeneracy_reason,
        endpoint_overlap_fraction=float(endpoint_overlap_fraction) if endpoint_overlap_fraction is not None else None,
        bootstrap_stability_class=bootstrap_stability_class,
        bootstrap_relative_deviation=(
            float(bootstrap_relative_deviation) if bootstrap_relative_deviation is not None and np.isfinite(bootstrap_relative_deviation) else None
        ),
        bootstrap_score=bootstrap_score,
        manifold_support_ratio=(
            float(manifold_support_ratio) if manifold_support_ratio is not None and np.isfinite(manifold_support_ratio) else None
        ),
        manifold_score=manifold_score,
        center_separation_score=center_separation_score,
        density_calibration_score=density_calibration_score,
        reliability_score=reliability_score,
        warnings=warnings,
    )


def update_candidate_with_full_reliability(
    candidate: EndpointCandidate,
    lap_reliability: Optional[dict],
    *,
    density_calibration: Optional[dict] = None,
    reliability_mode: str = "reliability_minimal",
) -> EndpointCandidate:
    """Attach full/minimal lap_reliability and recompute reliability-aware final score."""
    full = compute_candidate_full_reliability_score(
        candidate,
        lap_reliability,
        path_result=candidate.path_result,
        density_calibration=density_calibration,
    )
    degeneracy_source = "full_lap_reliability" if lap_reliability else "lightweight"
    path_degenerate = (
        full.full_path_degenerate
        if full.full_path_degenerate is not None
        else candidate.lightweight_path_degenerate
    )
    updated = replace(
        candidate,
        lap_reliability=lap_reliability,
        full_reliability=full,
        full_path_degenerate=full.full_path_degenerate,
        full_degeneracy_reason=full.degeneracy_reason,
        full_reliability_score=full.reliability_score,
        bootstrap_stability_class=full.bootstrap_stability_class,
        bootstrap_relative_deviation=full.bootstrap_relative_deviation,
        degeneracy_source=degeneracy_source,
        path_degenerate=path_degenerate,
        endpoint_selection_reliability_mode=reliability_mode,
    )
    if full.manifold_support_ratio is not None and np.isfinite(full.manifold_support_ratio):
        updated = replace(updated, manifold_support_ratio=full.manifold_support_ratio)
    if full.endpoint_overlap_fraction is not None:
        updated = replace(updated, endpoint_overlap_fraction=full.endpoint_overlap_fraction)
    updated.endpoint_final_score = compute_endpoint_final_score(updated)
    return updated


def _state_mask(labels, state: str) -> np.ndarray:
    return np.asarray(labels) == state


def _stage_purity_score(coords, labels, start_idx, end_idx, start_state, end_state) -> float:
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=min(15, len(coords))).fit(coords)
    scores = []
    for idx, expected in ((start_idx, start_state), (end_idx, end_state)):
        _, nn = nbrs.kneighbors(coords[idx : idx + 1])
        scores.append(float(np.mean(labels[nn[0]] == expected)))
    return float(np.mean(scores))


def _pseudotime_order_score(pseudotime, start_idx, end_idx) -> float:
    if pseudotime is None:
        return 0.5
    pt = np.asarray(pseudotime, dtype=float)
    if not np.isfinite(pt[start_idx]) or not np.isfinite(pt[end_idx]):
        return 0.5
    if pt[start_idx] < pt[end_idx]:
        return float(np.clip((pt[end_idx] - pt[start_idx]) / max(np.ptp(pt), 1e-6), 0.0, 1.0))
    return float(np.clip(1.0 - (pt[start_idx] - pt[end_idx]) / max(np.ptp(pt), 1e-6), 0.0, 0.5))


def _density_core_score(neg_logp, start_idx, end_idx, start_core, end_core) -> float:
    if neg_logp is None:
        return 0.5
    nlp = np.asarray(neg_logp, dtype=float)
    start_rank = float(np.mean(nlp[start_core] >= nlp[start_idx])) if len(start_core) else 0.5
    end_rank = float(np.mean(nlp[end_core] >= nlp[end_idx])) if len(end_core) else 0.5
    return 0.5 * (start_rank + end_rank)


def _pick_indices_for_mode(
    mode: str,
    coords: np.ndarray,
    labels,
    potential: np.ndarray,
    pseudotime: Optional[np.ndarray],
    start_state: str,
    end_state: str,
    core_fraction: float,
    neg_logp: Optional[np.ndarray],
) -> Tuple[int, int, List[str]]:
    warnings: List[str] = []
    if mode in ("legacy", "medoid"):
        start_mask = _state_mask(labels, start_state)
        end_mask = _state_mask(labels, end_state)
        s_local = pick_cluster_representative_index(coords[start_mask], potential[start_mask], "medoid", role="start")
        e_local = pick_cluster_representative_index(coords[end_mask], potential[end_mask], "medoid", role="end")
        return int(np.where(start_mask)[0][s_local]), int(np.where(end_mask)[0][e_local]), warnings

    start_mask = _state_mask(labels, start_state)
    end_mask = _state_mask(labels, end_state)
    if not np.any(start_mask) or not np.any(end_mask):
        raise ValueError(f"Missing cells for states {start_state!r} or {end_state!r}")
    start_global = np.where(start_mask)[0]
    end_global = np.where(end_mask)[0]

    if mode == "pseudotime_quantile":
        if pseudotime is None:
            warnings.append("pseudotime missing; falling back to medoid")
            return _pick_indices_for_mode("medoid", coords, labels, potential, pseudotime, start_state, end_state, core_fraction, neg_logp)
        s_local = pick_cluster_representative_index(
            coords[start_mask], potential[start_mask], "pseudotime_quantile",
            pseudotime=pseudotime[start_mask], role="start",
        )
        e_local = pick_cluster_representative_index(
            coords[end_mask], potential[end_mask], "pseudotime_quantile",
            pseudotime=pseudotime[end_mask], role="end",
        )
        return int(start_global[s_local]), int(end_global[e_local]), warnings

    start_core = stage_core_cell_indices(coords, labels, start_state, pseudotime=pseudotime, core_fraction=core_fraction)
    end_core = stage_core_cell_indices(coords, labels, end_state, pseudotime=pseudotime, core_fraction=core_fraction)

    if mode == "farthest_core":
        if len(start_core) == 0 or len(end_core) == 0:
            warnings.append("empty core; falling back to medoid")
            return _pick_indices_for_mode("medoid", coords, labels, potential, pseudotime, start_state, end_state, core_fraction, neg_logp)
        best_d, best_pair = -1.0, (int(start_core[0]), int(end_core[0]))
        for s_idx in start_core:
            for e_idx in end_core:
                d = float(np.linalg.norm(coords[s_idx] - coords[e_idx]))
                if d > best_d:
                    best_d, best_pair = d, (int(s_idx), int(e_idx))
        return best_pair[0], best_pair[1], warnings

    if mode == "density_core":
        if neg_logp is None:
            warnings.append("density unavailable; falling back to medoid")
            return _pick_indices_for_mode("medoid", coords, labels, potential, pseudotime, start_state, end_state, core_fraction, neg_logp)
        s_idx = int(start_core[np.argmin(neg_logp[start_core])]) if len(start_core) else int(start_global[0])
        e_idx = int(end_core[np.argmin(neg_logp[end_core])]) if len(end_core) else int(end_global[0])
        return s_idx, e_idx, warnings

    if mode == "hybrid":
        if len(start_core) == 0 or len(end_core) == 0:
            warnings.append("empty core; falling back to medoid")
            return _pick_indices_for_mode("medoid", coords, labels, potential, pseudotime, start_state, end_state, core_fraction, neg_logp)
        best_score, best_pair = -1.0, (int(start_core[0]), int(end_core[0]))
        spacing = mean_neighbor_spacing(coords)
        for s_idx in start_core[: min(30, len(start_core))]:
            for e_idx in end_core[: min(30, len(end_core))]:
                dist = float(np.linalg.norm(coords[s_idx] - coords[e_idx]))
                ratio = dist / max(spacing, 1e-12)
                dens = _density_core_score(neg_logp, s_idx, e_idx, start_core, end_core)
                pt_ord = _pseudotime_order_score(pseudotime, s_idx, e_idx)
                purity = _stage_purity_score(coords, labels, s_idx, e_idx, start_state, end_state)
                sc = (
                    0.30 * _separation_score(ratio)
                    + 0.15 * dens + 0.30 * pt_ord + 0.25 * purity
                )
                if sc > best_score:
                    best_score, best_pair = sc, (int(s_idx), int(e_idx))
        return best_pair[0], best_pair[1], warnings

    raise ValueError(f"Unknown endpoint mode: {mode!r}")


def _build_candidate(
    mode: str,
    coords: np.ndarray,
    labels,
    potential: np.ndarray,
    pseudotime: Optional[np.ndarray],
    start_state: str,
    end_state: str,
    core_fraction: float,
    neg_logp: Optional[np.ndarray],
    neighbor_spacing: float,
    separability_threshold: float,
) -> EndpointCandidate:
    start_idx, end_idx, warnings = _pick_indices_for_mode(
        mode, coords, labels, potential, pseudotime, start_state, end_state, core_fraction, neg_logp,
    )
    start_core = stage_core_cell_indices(coords, labels, start_state, pseudotime=pseudotime, core_fraction=core_fraction)
    end_core = stage_core_cell_indices(coords, labels, end_state, pseudotime=pseudotime, core_fraction=core_fraction)
    sep = compute_endpoint_separability(coords, start_core, end_core, neighbor_spacing, separability_threshold)
    stage_purity = _stage_purity_score(coords, labels, start_idx, end_idx, start_state, end_state)
    pt_order = _pseudotime_order_score(pseudotime, start_idx, end_idx)
    dens_score = _density_core_score(neg_logp, start_idx, end_idx, start_core, end_core)
    separation = _separation_score(sep["endpoint_distance_ratio"])
    cand = EndpointCandidate(
        mode=mode,
        start_idx=start_idx,
        end_idx=end_idx,
        start_state=start_state,
        end_state=end_state,
        endpoint_distance=sep["endpoint_distance"],
        endpoint_distance_ratio=sep["endpoint_distance_ratio"],
        separation_score=separation,
        density_core_score=dens_score,
        pseudotime_order_score=pt_order,
        stage_purity_score=stage_purity,
        endpoint_pre_score=0.0,
        is_separable=sep["is_separable"],
        warnings=warnings,
    )
    cand.endpoint_pre_score = compute_endpoint_pre_score(cand)
    if not sep["is_separable"]:
        cand.warnings.append("endpoint_not_separable")
    return cand


def generate_endpoint_candidates(
    adata,
    coords: np.ndarray,
    start_state: str,
    end_state: str,
    stage_key: str,
    pseudotime_key: str = "pseudotime",
    potential_key: str = "potential",
    candidate_modes: Sequence[str] = (
        "legacy",
        "medoid",
        "pseudotime_quantile",
        "farthest_core",
        "density_core",
        "hybrid",
    ),
    neighbor_spacing: Optional[float] = None,
    core_fraction: float = 0.5,
    separability_threshold: float = 2.0,
) -> List[EndpointCandidate]:
    coords = np.asarray(coords, dtype=float)
    labels = adata.obs[stage_key].values
    potential = np.asarray(adata.obs[potential_key].values, dtype=float)
    pseudotime = np.asarray(adata.obs[pseudotime_key].values, dtype=float) if pseudotime_key in adata.obs else None
    spacing = float(neighbor_spacing) if neighbor_spacing is not None else mean_neighbor_spacing(coords)
    neg_logp, _ = estimate_potential_from_density(coords)

    modes = list(candidate_modes)
    if "legacy" not in modes:
        modes = ["legacy"] + [m for m in modes if m != "legacy"]

    candidates = []
    for mode in modes:
        try:
            candidates.append(
                _build_candidate(
                    mode, coords, labels, potential, pseudotime,
                    start_state, end_state, core_fraction, neg_logp, spacing, separability_threshold,
                )
            )
        except Exception as exc:
            candidates.append(
                EndpointCandidate(
                    mode=mode, start_idx=-1, end_idx=-1,
                    start_state=start_state, end_state=end_state,
                    endpoint_distance=np.nan, endpoint_distance_ratio=np.nan,
                    separation_score=0.0, density_core_score=0.0,
                    pseudotime_order_score=0.0, stage_purity_score=0.0,
                    endpoint_pre_score=0.0, is_separable=False, warnings=[str(exc)],
                )
            )
    return candidates


def update_candidate_with_lap_diagnostics(
    candidate: EndpointCandidate,
    path_result: dict,
    analyzer=None,
    adata=None,
) -> EndpointCandidate:
    """Attach post-LAP diagnostics from a lightweight path result."""
    from lap_reliability_diagnostics import diagnose_path_degeneracy

    path = np.asarray(path_result.get("path_compute", path_result.get("path")), dtype=float)
    degeneracy = {}
    try:
        if analyzer is not None:
            degeneracy = diagnose_path_degeneracy(path_result, analyzer, adata=adata)
    except Exception:
        degeneracy = {}

    lightweight_path_degenerate = bool(
        path_result.get("path_degenerate")
        if path_result.get("path_degenerate") is not None
        else degeneracy.get("is_degenerate", False)
    )
    overlap = max(
        float(degeneracy.get("start_transition_jaccard", 0) or 0),
        float(degeneracy.get("transition_end_jaccard", 0) or 0),
        float(degeneracy.get("start_end_jaccard", 0) or 0),
        float(path_result.get("endpoint_overlap_fraction", 0) or 0),
    )
    path_len = float(degeneracy.get("path_length", np.nan))
    if not np.isfinite(path_len) and len(path) >= 2:
        path_len = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))

    neighbor_spacing = float(degeneracy.get("neighbor_spacing", np.nan))
    path_length_ratio = (
        path_len / neighbor_spacing if np.isfinite(path_len) and np.isfinite(neighbor_spacing) and neighbor_spacing > 0 else np.nan
    )

    center_idx = path_result.get("remodeling_center_idx")
    if center_idx is None:
        center_idx = path_result.get("transition_state_idx_composite", path_result.get("transition_state_idx", 0))
    center_idx = int(center_idx) if center_idx is not None else 0
    center_idx = min(max(center_idx, 0), max(len(path) - 1, 0))

    if len(path) >= 1:
        center_pt = path[center_idx]
        d_start = float(np.linalg.norm(center_pt - path[0]))
        d_end = float(np.linalg.norm(center_pt - path[-1]))
    else:
        d_start = d_end = np.nan

    eps = 1e-8
    center_sep = (
        min(d_start, d_end) / max(path_len, eps) if np.isfinite(path_len) and path_len > 0 else 0.5
    )
    center_sep = float(np.clip(center_sep * 2.0, 0.0, 1.0))

    manifold_ratio = path_result.get("manifold_support_ratio", np.nan)
    if not np.isfinite(manifold_ratio):
        manifold_ratio = np.nan

    updated = replace(
        candidate,
        lightweight_path_degenerate=lightweight_path_degenerate,
        path_degenerate=lightweight_path_degenerate,
        endpoint_overlap_fraction=overlap,
        path_length=path_len,
        path_length_ratio=path_length_ratio,
        manifold_support_ratio=manifold_ratio,
        remodeling_center_idx=center_idx,
        remodeling_center_distance_to_start=d_start,
        remodeling_center_distance_to_end=d_end,
        center_separation_score=center_sep,
        path_result=path_result,
    )
    updated.post_lap_score = compute_post_lap_score(updated)
    updated.endpoint_final_score = compute_endpoint_final_score(updated)
    return updated


def build_public_endpoint_selection_meta(
    candidate: EndpointCandidate,
    *,
    selection_strategy: str = "reliability_aware",
    reliability_mode: str = "reliability_minimal",
    legacy_cand: Optional[EndpointCandidate] = None,
    auto_best_cand: Optional[EndpointCandidate] = None,
    all_candidates: Optional[Sequence[EndpointCandidate]] = None,
) -> dict:
    """Public-facing endpoint selection summary (reliability-aware semantics)."""
    warnings = list(candidate.warnings or [])
    fallback_used = candidate.selected_reason in {
        "fallback_to_legacy_non_degenerate",
        "fallback_to_legacy_full_reliability_non_degenerate",
    }
    if fallback_used and candidate.selected_reason:
        warnings.append(candidate.selected_reason)

    all_full_degenerate = None
    if all_candidates is not None and any(c.full_path_degenerate is not None for c in all_candidates):
        valid = [c for c in all_candidates if c.start_idx >= 0]
        all_full_degenerate = (
            all(c.full_path_degenerate is True for c in valid) if valid else None
        )

    path_deg = (
        candidate.full_path_degenerate
        if candidate.full_path_degenerate is not None
        else candidate.lightweight_path_degenerate
    )
    return {
        "selection_strategy": selection_strategy,
        "selected_endpoint_mode": candidate.mode,
        "selected_mode": candidate.mode,
        "mode": candidate.mode,
        "endpoint_distance": candidate.endpoint_distance,
        "endpoint_distance_ratio": candidate.endpoint_distance_ratio,
        "endpoint_is_separable": candidate.is_separable,
        "is_separable": candidate.is_separable,
        "endpoint_reliability_score": candidate.full_reliability_score,
        "endpoint_pre_score": candidate.endpoint_pre_score,
        "endpoint_post_lap_score": candidate.post_lap_score,
        "post_lap_score": candidate.post_lap_score,
        "endpoint_final_score": candidate.endpoint_final_score,
        "endpoint_score": candidate.endpoint_final_score,
        "endpoint_warning": "; ".join(w for w in warnings if w),
        "warnings": warnings,
        "selected_reason": candidate.selected_reason,
        "endpoint_selected_reason": candidate.selected_reason,
        "endpoint_selection_reliability_mode": reliability_mode,
        "endpoint_fallback_used": fallback_used,
        "endpoint_fallback_reason": candidate.selected_reason if fallback_used else "",
        "path_degenerate": path_deg,
        "lightweight_path_degenerate": candidate.lightweight_path_degenerate,
        "full_path_degenerate": candidate.full_path_degenerate,
        "endpoint_full_reliability_score": candidate.full_reliability_score,
        "endpoint_legacy_full_path_degenerate": (
            legacy_cand.full_path_degenerate if legacy_cand is not None else None
        ),
        "endpoint_auto_best_full_path_degenerate": (
            auto_best_cand.full_path_degenerate if auto_best_cand is not None else None
        ),
        "endpoint_all_candidates_full_degenerate": all_full_degenerate,
        "endpoint_legacy_path_degenerate": (
            legacy_cand.lightweight_path_degenerate if legacy_cand is not None else None
        ),
        "endpoint_auto_best_path_degenerate": (
            auto_best_cand.lightweight_path_degenerate if auto_best_cand is not None else None
        ),
        "endpoint_selected_path_degenerate": path_deg,
        "endpoint_selected_full_path_degenerate": candidate.full_path_degenerate,
        "start_idx": candidate.start_idx,
        "end_idx": candidate.end_idx,
        "precomputed_lap_reliability": candidate.lap_reliability,
    }


def select_reliable_endpoints(
    adata,
    coords: np.ndarray,
    start_state: str,
    end_state: str,
    stage_key: str,
    *,
    selection_strategy: str = "auto_two_stage",
    reliability_mode: str = "reliability_minimal",
    candidate_modes: Optional[Sequence[str]] = None,
    candidate_mode_override: Optional[str] = None,
    separability_threshold: float = 2.0,
    core_fraction: float = 0.5,
    pseudotime_key: str = "pseudotime",
    potential_key: str = "potential",
    compute_lap_path=None,
    compute_lap_reliability=None,
    density_calibration: Optional[dict] = None,
) -> dict:
    """
    Reliability-aware endpoint selection workflow.

    Generates heuristic candidate endpoint pairs, scores them with pre-LAP geometry,
    post-LAP path diagnostics, and optional full reliability, then selects the best pair.

    ``compute_lap_path(candidate, start_pos, end_pos)`` and
    ``compute_lap_reliability(candidate, path_result)`` are injected by the caller to
    avoid circular imports with LAP optimization code.
    """
    coords = np.asarray(coords, dtype=float)
    modes = list(candidate_modes or (
        "legacy", "medoid", "pseudotime_quantile", "farthest_core", "density_core", "hybrid",
    ))
    if candidate_mode_override and candidate_mode_override not in ("auto",):
        modes = [candidate_mode_override]

    candidates = generate_endpoint_candidates(
        adata,
        coords,
        start_state,
        end_state,
        stage_key=stage_key,
        pseudotime_key=pseudotime_key,
        potential_key=potential_key,
        candidate_modes=modes,
        separability_threshold=separability_threshold,
        core_fraction=core_fraction,
    )

    if selection_strategy == "auto_pre":
        best, all_inseparable = select_best_endpoint_candidate(candidates)
        if compute_lap_path is not None and best.start_idx >= 0:
            start_pos = coords[best.start_idx]
            end_pos = coords[best.end_idx]
            path_result = compute_lap_path(best, start_pos, end_pos)
            best = update_candidate_with_lap_diagnostics(best, path_result)
            best.path_result = path_result
        endpoint_meta = build_public_endpoint_selection_meta(
            best,
            selection_strategy="auto_pre",
            reliability_mode=reliability_mode,
            all_candidates=candidates,
        )
        endpoint_meta["all_inseparable"] = all_inseparable
        return {
            "best_candidate": best,
            "candidates": candidates,
            "best_path_result": best.path_result,
            "endpoint_selection": endpoint_meta,
            "endpoint_candidates_df": candidates_to_dataframe(candidates),
        }

    if compute_lap_path is None:
        raise ValueError("compute_lap_path is required for reliability-aware endpoint selection")

    updated: List[EndpointCandidate] = []
    for cand in candidates:
        if cand.start_idx < 0 or cand.end_idx < 0:
            updated.append(cand)
            continue
        start_pos = coords[cand.start_idx]
        end_pos = coords[cand.end_idx]
        path_result = compute_lap_path(cand, start_pos, end_pos)
        cand_up = update_candidate_with_lap_diagnostics(cand, path_result)
        cand_up.path_result = path_result

        if reliability_mode != "lightweight" and compute_lap_reliability is not None:
            lap_rel = compute_lap_reliability(cand_up, path_result)
            cand_up = update_candidate_with_full_reliability(
                cand_up,
                lap_rel,
                density_calibration=density_calibration,
                reliability_mode=reliability_mode,
            )
        else:
            cand_up.endpoint_selection_reliability_mode = "lightweight"
            cand_up.endpoint_final_score = cand_up.endpoint_final_score or 0.0
        updated.append(cand_up)

    legacy_cand = next((c for c in updated if c.mode == "legacy"), None)
    non_legacy = [c for c in updated if c.mode != "legacy" and c.start_idx >= 0]
    auto_best_cand = (
        max(non_legacy, key=lambda c: c.endpoint_final_score or 0.0) if non_legacy else None
    )
    best = select_best_endpoint_candidate_reliability_aware(updated)
    strategy_label = str(selection_strategy)
    endpoint_meta = build_public_endpoint_selection_meta(
        best,
        selection_strategy=strategy_label,
        reliability_mode=reliability_mode,
        legacy_cand=legacy_cand,
        auto_best_cand=auto_best_cand,
        all_candidates=updated,
    )

    return {
        "best_candidate": best,
        "candidates": updated,
        "best_path_result": best.path_result,
        "endpoint_selection": endpoint_meta,
        "endpoint_candidates_df": candidates_to_dataframe(updated),
    }


def select_best_endpoint_candidate_reliability_aware(
    candidates: Sequence[EndpointCandidate],
) -> EndpointCandidate:
    """Select best candidate with full-reliability legacy fallback when applicable."""
    valid = [c for c in candidates if c.start_idx >= 0 and c.end_idx >= 0 and c.post_lap_score is not None]
    if not valid:
        valid = [c for c in candidates if c.start_idx >= 0 and c.end_idx >= 0]
    if not valid:
        raise ValueError("No valid endpoint candidates")

    use_full = any(c.full_path_degenerate is not None for c in valid)
    legacy = next((c for c in valid if c.mode == "legacy"), None)
    non_legacy = [c for c in valid if c.mode != "legacy"]
    auto_best = max(non_legacy, key=lambda c: c.endpoint_final_score or 0.0) if non_legacy else None
    overall_best = max(valid, key=lambda c: c.endpoint_final_score or 0.0)

    if use_full:
        all_degenerate = all(c.full_path_degenerate is True for c in valid)
        if all_degenerate:
            best = max(valid, key=lambda c: c.full_reliability_score or 0.0)
            reason = "all_candidates_degenerate_selected_best_reliability_score"
        elif (
            auto_best is not None
            and auto_best.full_path_degenerate is True
            and legacy is not None
            and legacy.full_path_degenerate is False
        ):
            best = legacy
            reason = "fallback_to_legacy_full_reliability_non_degenerate"
        else:
            best = overall_best
            reason = "selected_by_reliability_aware_score"
    else:
        nondeg = [c for c in valid if (c.lightweight_path_degenerate or c.path_degenerate) is False]
        if nondeg:
            best = max(nondeg, key=lambda c: c.endpoint_final_score or 0.0)
            reason = "best_non_degenerate_final_score"
        else:
            best = overall_best
            reason = "best_degenerate_final_score"
        if (
            auto_best is not None
            and (auto_best.lightweight_path_degenerate or auto_best.path_degenerate) is True
            and legacy is not None
            and (legacy.lightweight_path_degenerate or legacy.path_degenerate) is False
            and legacy.post_lap_score is not None
        ):
            best = legacy
            reason = "fallback_to_legacy_non_degenerate"

    for c in candidates:
        c.selected = False
        c.selected_reason = ""
    best.selected = True
    best.selected_reason = reason
    return best


def select_best_endpoint_candidate_two_stage(
    candidates: Sequence[EndpointCandidate],
) -> EndpointCandidate:
    """Select best candidate with legacy fallback (delegates to reliability-aware when available)."""
    return select_best_endpoint_candidate_reliability_aware(candidates)


def select_best_endpoint_candidate(
    candidates: Sequence[EndpointCandidate],
    prefer_non_degenerate: bool = True,
) -> Tuple[EndpointCandidate, bool]:
    """Pre-LAP candidate generation — ranks all valid endpoints (no separability hard filter)."""
    valid = [c for c in candidates if c.start_idx >= 0 and c.end_idx >= 0]
    if not valid:
        return candidates[0], False

    def _rank_key(c: EndpointCandidate) -> float:
        sep_bonus = 0.05 if c.is_separable else 0.0
        nondeg_bonus = 0.0
        if prefer_non_degenerate and c.lightweight_path_degenerate is False:
            nondeg_bonus = 0.05
        return float(c.endpoint_pre_score) + sep_bonus + nondeg_bonus

    best = max(valid, key=_rank_key)
    all_inseparable = not any(c.is_separable for c in valid)
    return best, all_inseparable


def candidates_to_dataframe(candidates: Sequence[EndpointCandidate]) -> pd.DataFrame:
    return candidates_to_two_stage_dataframe(candidates)


def candidates_to_two_stage_dataframe(candidates: Sequence[EndpointCandidate]) -> pd.DataFrame:
    rows = []
    for c in candidates:
        path_deg = c.full_path_degenerate if c.full_path_degenerate is not None else c.lightweight_path_degenerate
        rows.append(
            {
                "mode": c.mode,
                "candidate_generator": c.mode,
                "start_idx": c.start_idx,
                "end_idx": c.end_idx,
                "endpoint_distance": c.endpoint_distance,
                "endpoint_distance_ratio": c.endpoint_distance_ratio,
                "separation_score": c.separation_score,
                "density_core_score": c.density_core_score,
                "pseudotime_order_score": c.pseudotime_order_score,
                "stage_purity_score": c.stage_purity_score,
                "endpoint_pre_score": c.endpoint_pre_score,
                "lightweight_path_degenerate": c.lightweight_path_degenerate,
                "full_path_degenerate": c.full_path_degenerate,
                "path_degenerate": path_deg,
                "degeneracy_source": c.degeneracy_source,
                "full_degeneracy_reason": c.full_degeneracy_reason,
                "bootstrap_stability_class": c.bootstrap_stability_class,
                "bootstrap_relative_deviation": c.bootstrap_relative_deviation,
                "full_reliability_score": c.full_reliability_score,
                "endpoint_reliability_score": c.full_reliability_score,
                "endpoint_overlap_fraction": c.endpoint_overlap_fraction,
                "path_length": c.path_length,
                "path_length_ratio": c.path_length_ratio,
                "manifold_support_ratio": c.manifold_support_ratio,
                "remodeling_center_idx": c.remodeling_center_idx,
                "center_separation_score": c.center_separation_score,
                "post_lap_score": c.post_lap_score,
                "endpoint_post_lap_score": c.post_lap_score,
                "endpoint_final_score": c.endpoint_final_score,
                "endpoint_selection_reliability_mode": c.endpoint_selection_reliability_mode,
                "selected": c.selected,
                "selected_reason": c.selected_reason,
                "endpoint_selected_reason": c.selected_reason,
                "is_separable": c.is_separable,
                "warnings": ";".join(c.warnings),
            }
        )
    return pd.DataFrame(rows)
