"""
LAP reliability diagnostics: path degeneracy, bootstrap stability, and conservative
transition interpretation.

Path-local potential maxima are treated as algorithmic candidates only until they pass
endpoint separability, non-degenerate geometry, bootstrap stability, and downstream
evidence checks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from lap_helpers import (
    clip_path_index,
    collect_knn_cells_for_path_windows,
    interpolate_path_arclength,
)
from landscape_core import mean_neighbor_spacing

INTERPRETATION_LABELS: Dict[str, str] = {
    "barrier_like_transition_region": "Model-supported remodeling window (legacy label)",
    "molecularly_supported_remodeling_region": "Molecularly supported remodeling window",
    "smooth_remodeling_trajectory": "Smooth remodeling trajectory",
    "algorithmic_candidate_only": "Algorithmic path-local maximum only",
    "unstable_path_no_supported_transition": "Unstable path",
    "dynamically_supported_candidate_region": "Dynamically supported remodeling window",
    "smooth_or_weak_remodeling": "Smooth or weak remodeling",
}


def _path_length(path: np.ndarray) -> float:
    path = np.asarray(path, dtype=float)
    if len(path) <= 1:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def _jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    set_a = set(int(x) for x in a)
    set_b = set(int(x) for x in b)
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return float(len(set_a & set_b) / len(union))


def _safe_ratio(distance: float, spacing: float) -> float:
    if not np.isfinite(distance) or not np.isfinite(spacing) or spacing <= 1e-12:
        return float("nan")
    return float(distance / spacing)


def _evidence_is_positive(result: Optional[Dict[str, Any]]) -> bool:
    """Return True when an evidence dict indicates strong or moderate support."""
    if not result:
        return False
    if result.get("passed") is True:
        return True
    support = str(result.get("support", "")).lower()
    if support in {"strong", "moderate"}:
        return True
    if result.get("path_stability_class") in {"stable", "moderately_stable"}:
        return True
    for key in (
        "barrier_support",
        "manifold_support",
        "pseudotime_order",
        "bootstrap_stability",
        "derivative_alignment",
        "molecular_evidence",
    ):
        if result.get(key) is True:
            return True
    return False


def _barrier_is_strong(barrier_result: Optional[Dict[str, Any]]) -> bool:
    if not barrier_result:
        return False
    if barrier_result.get("passed") is True:
        return True
    support = str(barrier_result.get("support", "")).lower()
    if support in {"strong", "moderate"}:
        return True
    if barrier_result.get("barrier_support") is True:
        return True
    barrier_height = barrier_result.get("barrier_height")
    barrier_threshold = barrier_result.get("barrier_threshold")
    if (
        barrier_height is not None
        and barrier_threshold is not None
        and np.isfinite(barrier_height)
        and np.isfinite(barrier_threshold)
        and float(barrier_height) > float(barrier_threshold)
    ):
        return True
    return False


def _molecular_is_strong(molecular_result: Optional[Dict[str, Any]]) -> bool:
    if not molecular_result:
        return False
    if molecular_result.get("passed") is True:
        return True
    support = str(molecular_result.get("support", "")).lower()
    if support in {"strong", "moderate"}:
        return True
    if molecular_result.get("molecular_evidence") is True:
        return True
    n_deg = molecular_result.get("n_deg")
    n_enrichment = molecular_result.get("n_enrichment_terms")
    if n_deg is not None and int(n_deg) > 0:
        return True
    if n_enrichment is not None and int(n_enrichment) > 0:
        return True
    return False


def diagnose_path_degeneracy(
    path_result: dict,
    analyzer,
    adata=None,
    *,
    window_size: int = 5,
    neighbors_per_path_point: int = 10,
    min_endpoint_distance_ratio: float = 2.0,
    min_ts_endpoint_distance_ratio: float = 1.0,
    max_window_jaccard: float = 0.3,
    endpoint_margin: float = 0.10,
) -> Dict[str, Any]:
    """
    Diagnose whether a LAP is degenerate or whether the path-local potential maximum
    is too close to the start/end endpoints to be interpreted as an independent
    remodeling window center.

    Works in LAP compute space, not UMAP display space.
    """
    warnings: List[str] = []
    path = np.asarray(path_result.get("path_compute", path_result.get("path")), dtype=float)
    n_path = len(path)
    ts_raw = path_result.get("transition_state_idx", 0)
    ts_idx = clip_path_index(int(ts_raw), n_path)

    relative_ts_position = float(ts_idx / max(n_path - 1, 1))
    path_len = _path_length(path)

    neighbor_spacing = float("nan")
    try:
        positions = np.asarray(analyzer.cell_positions_2d, dtype=float)
        if len(positions) >= 2:
            neighbor_spacing = mean_neighbor_spacing(positions)
    except Exception as exc:
        warnings.append(f"Could not compute neighbor spacing: {exc}")

    start_pt = path[0]
    end_pt = path[-1]
    ts_pt = path[ts_idx]

    start_end_distance = float(np.linalg.norm(end_pt - start_pt))
    start_ts_distance = float(np.linalg.norm(ts_pt - start_pt))
    ts_end_distance = float(np.linalg.norm(end_pt - ts_pt))

    if not np.isfinite(neighbor_spacing) or neighbor_spacing <= 1e-12:
        warnings.append("Invalid or zero neighbor spacing; distance ratios set to NaN.")
        start_end_distance_ratio = float("nan")
        start_ts_distance_ratio = float("nan")
        ts_end_distance_ratio = float("nan")
    else:
        start_end_distance_ratio = _safe_ratio(start_end_distance, neighbor_spacing)
        start_ts_distance_ratio = _safe_ratio(start_ts_distance, neighbor_spacing)
        ts_end_distance_ratio = _safe_ratio(ts_end_distance, neighbor_spacing)

    ts_near_endpoint = bool(
        relative_ts_position < endpoint_margin or relative_ts_position > 1.0 - endpoint_margin
    )
    endpoints_too_close = bool(
        np.isfinite(start_end_distance_ratio)
        and start_end_distance_ratio < min_endpoint_distance_ratio
    )
    ts_start_too_close = bool(
        np.isfinite(start_ts_distance_ratio)
        and start_ts_distance_ratio < min_ts_endpoint_distance_ratio
    )
    ts_end_too_close = bool(
        np.isfinite(ts_end_distance_ratio)
        and ts_end_distance_ratio < min_ts_endpoint_distance_ratio
    )

    start_transition_jaccard = float("nan")
    transition_end_jaccard = float("nan")
    start_end_jaccard = float("nan")
    high_window_overlap = False

    if analyzer is not None and n_path >= 1:
        try:
            region_cells = collect_knn_cells_for_path_windows(
                analyzer,
                path_result,
                window_size=window_size,
                neighbors_per_path_point=neighbors_per_path_point,
            )
            sw = region_cells.get("start_window", np.array([], dtype=int))
            tw = region_cells.get("transition_window", np.array([], dtype=int))
            ew = region_cells.get("end_window", np.array([], dtype=int))
            start_transition_jaccard = _jaccard(sw, tw)
            transition_end_jaccard = _jaccard(tw, ew)
            start_end_jaccard = _jaccard(sw, ew)
            high_window_overlap = any(
                j > max_window_jaccard
                for j in (start_transition_jaccard, transition_end_jaccard, start_end_jaccard)
                if np.isfinite(j)
            )
        except Exception as exc:
            warnings.append(f"Window overlap computation failed: {exc}")

    is_degenerate = bool(
        ts_near_endpoint
        or endpoints_too_close
        or ts_start_too_close
        or ts_end_too_close
        or high_window_overlap
    )

    if endpoints_too_close:
        recommended_interpretation = "degenerate_path_or_high_stage_overlap"
    elif ts_near_endpoint or ts_start_too_close or ts_end_too_close:
        recommended_interpretation = "algorithmic_endpoint_maximum_not_transition_state"
    elif high_window_overlap:
        recommended_interpretation = "overlapping_windows_smooth_or_weak_remodeling"
    else:
        recommended_interpretation = "non_degenerate_path_candidate_for_transition_testing"

    return {
        "n_path_points": int(n_path),
        "transition_state_idx": int(ts_idx),
        "relative_ts_position": relative_ts_position,
        "path_length": path_len,
        "neighbor_spacing": neighbor_spacing,
        "start_end_distance": start_end_distance,
        "start_ts_distance": start_ts_distance,
        "ts_end_distance": ts_end_distance,
        "start_end_distance_ratio": start_end_distance_ratio,
        "start_ts_distance_ratio": start_ts_distance_ratio,
        "ts_end_distance_ratio": ts_end_distance_ratio,
        "start_transition_jaccard": start_transition_jaccard,
        "transition_end_jaccard": transition_end_jaccard,
        "start_end_jaccard": start_end_jaccard,
        "ts_near_endpoint": ts_near_endpoint,
        "endpoints_too_close": endpoints_too_close,
        "ts_start_too_close": ts_start_too_close,
        "ts_end_too_close": ts_end_too_close,
        "high_window_overlap": high_window_overlap,
        "is_degenerate": is_degenerate,
        "warnings": warnings,
        "recommended_interpretation": recommended_interpretation,
    }


def compare_canonical_bootstrap_paths(
    canonical_path: np.ndarray,
    bootstrap_median_path: np.ndarray,
    bootstrap_lower_path: Optional[np.ndarray] = None,
    bootstrap_upper_path: Optional[np.ndarray] = None,
    *,
    n_interp: int = 100,
    stable_threshold: float = 0.10,
    moderate_threshold: float = 0.25,
) -> Dict[str, Any]:
    """
    Quantify deviation between canonical medoid path and bootstrap median path.
    Optionally quantify bootstrap uncertainty envelope width.
    """
    warnings: List[str] = []
    canonical_path = np.asarray(canonical_path, dtype=float)
    bootstrap_median_path = np.asarray(bootstrap_median_path, dtype=float)

    if len(canonical_path) == 0 or len(bootstrap_median_path) == 0:
        warnings.append("Empty canonical or bootstrap median path.")
        return {
            "n_interp": int(n_interp),
            "canonical_median_mean_distance": float("nan"),
            "canonical_median_max_distance": float("nan"),
            "canonical_path_length": 0.0,
            "relative_path_deviation": float("nan"),
            "mean_envelope_width": float("nan"),
            "max_envelope_width": float("nan"),
            "path_stability_class": "degenerate",
            "warnings": warnings,
        }

    canon_rs = interpolate_path_arclength(canonical_path, n_interp)
    median_rs = interpolate_path_arclength(bootstrap_median_path, n_interp)
    pointwise_dist = np.linalg.norm(canon_rs - median_rs, axis=1)
    mean_distance = float(np.mean(pointwise_dist))
    max_distance = float(np.max(pointwise_dist))
    canonical_path_length = _path_length(canonical_path)

    mean_envelope_width = float("nan")
    max_envelope_width = float("nan")
    if bootstrap_lower_path is not None and bootstrap_upper_path is not None:
        lower_rs = interpolate_path_arclength(np.asarray(bootstrap_lower_path, dtype=float), n_interp)
        upper_rs = interpolate_path_arclength(np.asarray(bootstrap_upper_path, dtype=float), n_interp)
        envelope_width = np.linalg.norm(upper_rs - lower_rs, axis=1)
        mean_envelope_width = float(np.mean(envelope_width))
        max_envelope_width = float(np.max(envelope_width))

    if canonical_path_length <= 1e-12:
        warnings.append("Canonical path length too small for relative deviation.")
        relative_path_deviation = float("nan")
        path_stability_class = "degenerate"
    else:
        relative_path_deviation = float(mean_distance / canonical_path_length)
        if relative_path_deviation < stable_threshold:
            path_stability_class = "stable"
        elif relative_path_deviation < moderate_threshold:
            path_stability_class = "moderately_stable"
        else:
            path_stability_class = "unstable"

    return {
        "n_interp": int(n_interp),
        "canonical_median_mean_distance": mean_distance,
        "canonical_median_max_distance": max_distance,
        "canonical_path_length": canonical_path_length,
        "relative_path_deviation": relative_path_deviation,
        "mean_envelope_width": mean_envelope_width,
        "max_envelope_width": max_envelope_width,
        "path_stability_class": path_stability_class,
        "warnings": warnings,
    }


def classify_transition_interpretation(
    *,
    degeneracy_result: Optional[Dict[str, Any]] = None,
    bootstrap_result: Optional[Dict[str, Any]] = None,
    barrier_result: Optional[Dict[str, Any]] = None,
    path_stability_result: Optional[Dict[str, Any]] = None,
    manifold_result: Optional[Dict[str, Any]] = None,
    pseudotime_result: Optional[Dict[str, Any]] = None,
    derivative_result: Optional[Dict[str, Any]] = None,
    molecular_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Combine multiple evidence layers into a final conservative interpretation.
    """
    warnings: List[str] = []
    evidence_summary: Dict[str, Any] = {}

    if degeneracy_result:
        evidence_summary["degeneracy"] = {
            "is_degenerate": degeneracy_result.get("is_degenerate"),
            "recommended_interpretation": degeneracy_result.get("recommended_interpretation"),
        }
        warnings.extend(degeneracy_result.get("warnings") or [])

    if bootstrap_result:
        evidence_summary["bootstrap"] = {
            "path_stability_class": bootstrap_result.get("path_stability_class"),
            "relative_path_deviation": bootstrap_result.get("relative_path_deviation"),
        }
        warnings.extend(bootstrap_result.get("warnings") or [])

    barrier_strong = _barrier_is_strong(barrier_result)
    molecular_strong = _molecular_is_strong(molecular_result)
    path_stable = (
        (bootstrap_result or {}).get("path_stability_class") in {"stable", "moderately_stable"}
        or (path_stability_result or {}).get("passed") is True
        or str((path_stability_result or {}).get("support", "")).lower() in {"strong", "moderate"}
    )
    manifold_positive = _evidence_is_positive(manifold_result)
    pseudotime_positive = _evidence_is_positive(pseudotime_result)
    derivative_positive = _evidence_is_positive(derivative_result)

    evidence_summary["barrier_strong"] = barrier_strong
    evidence_summary["molecular_strong"] = molecular_strong
    evidence_summary["path_stable"] = path_stable
    evidence_summary["manifold_positive"] = manifold_positive
    evidence_summary["pseudotime_positive"] = pseudotime_positive
    evidence_summary["derivative_positive"] = derivative_positive

    if degeneracy_result and degeneracy_result.get("is_degenerate"):
        rec = degeneracy_result.get("recommended_interpretation", "")
        if rec == "overlapping_windows_smooth_or_weak_remodeling":
            final_class = "smooth_or_weak_remodeling"
        else:
            final_class = "algorithmic_candidate_only"
        return {
            "final_class": final_class,
            "can_call_candidate_transition_state": False,
            "can_call_biologically_supported_transition_state": False,
            "can_call_biologically_supported_transition_region": False,
            "evidence_summary": evidence_summary,
            "warnings": warnings,
            "recommended_sentence": (
                "The path-local potential maximum was not upgraded to a high-confidence "
                "remodeling window because it overlapped with endpoint states or showed "
                "high path-window overlap."
            ),
            "interpretation_label": INTERPRETATION_LABELS.get(
                final_class, INTERPRETATION_LABELS["algorithmic_candidate_only"]
            ),
        }

    if bootstrap_result and bootstrap_result.get("path_stability_class") == "unstable":
        return {
            "final_class": "unstable_path_no_supported_transition",
            "can_call_candidate_transition_state": False,
            "can_call_biologically_supported_transition_state": False,
            "can_call_biologically_supported_transition_region": False,
            "evidence_summary": evidence_summary,
            "warnings": warnings,
            "recommended_sentence": (
                "The canonical medoid path was sensitive to bootstrap resampling and was "
                "therefore not interpreted as a reliable transition path."
            ),
            "interpretation_label": INTERPRETATION_LABELS["unstable_path_no_supported_transition"],
        }

    if (
        barrier_strong
        and path_stable
        and manifold_positive
        and pseudotime_positive
        and molecular_strong
    ):
        return {
            "final_class": "barrier_like_transition_region",
            "can_call_candidate_transition_state": True,
            "can_call_biologically_supported_transition_state": True,
            "can_call_biologically_supported_transition_region": True,
            "evidence_summary": evidence_summary,
            "warnings": warnings,
            "recommended_sentence": (
                "This cell type shows a model-supported remodeling window centered around "
                "the path-local potential maximum, supported by barrier height, path "
                "stability, manifold support, pseudotime ordering, and molecular evidence."
            ),
            "interpretation_label": INTERPRETATION_LABELS["barrier_like_transition_region"],
        }

    if molecular_strong and not barrier_strong:
        return {
            "final_class": "molecularly_supported_remodeling_region",
            "can_call_candidate_transition_state": False,
            "can_call_biologically_supported_transition_state": False,
            "can_call_biologically_supported_transition_region": True,
            "evidence_summary": evidence_summary,
            "warnings": warnings,
            "recommended_sentence": (
                "This cell type shows a molecularly supported remodeling window, but barrier "
                "evidence is weak or unavailable; the path-local maximum should not be called "
                "a high-confidence remodeling region without additional checks."
            ),
            "interpretation_label": INTERPRETATION_LABELS["molecularly_supported_remodeling_region"],
        }

    if (barrier_strong or path_stable or manifold_positive) and not molecular_strong:
        return {
            "final_class": "dynamically_supported_candidate_region",
            "can_call_candidate_transition_state": True,
            "can_call_biologically_supported_transition_state": False,
            "can_call_biologically_supported_transition_region": False,
            "evidence_summary": evidence_summary,
            "warnings": warnings,
            "recommended_sentence": (
                "This cell type shows dynamical support for a candidate remodeling window, "
                "but molecular evidence is insufficient for biological confirmation."
            ),
            "interpretation_label": INTERPRETATION_LABELS["dynamically_supported_candidate_region"],
        }

    if path_stable and not barrier_strong and not molecular_strong:
        return {
            "final_class": "smooth_remodeling_trajectory",
            "can_call_candidate_transition_state": False,
            "can_call_biologically_supported_transition_state": False,
            "can_call_biologically_supported_transition_region": False,
            "evidence_summary": evidence_summary,
            "warnings": warnings,
            "recommended_sentence": (
                "This cell type shows a stable path but no clear barrier or strong molecular "
                "evidence; the trajectory is interpreted as smooth remodeling."
            ),
            "interpretation_label": INTERPRETATION_LABELS["smooth_remodeling_trajectory"],
        }

    return {
        "final_class": "algorithmic_candidate_only",
        "can_call_candidate_transition_state": False,
        "can_call_biologically_supported_transition_state": False,
        "can_call_biologically_supported_transition_region": False,
        "evidence_summary": evidence_summary,
        "warnings": warnings,
        "recommended_sentence": (
            "A path-local potential maximum was detected algorithmically, but it lacks "
            "sufficient evidence for interpretation as a model-supported remodeling window."
        ),
        "interpretation_label": INTERPRETATION_LABELS["algorithmic_candidate_only"],
    }


def degeneracy_summary_columns(degeneracy_result: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten degeneracy diagnostics for scorecard / CSV export."""
    return {
        "path_degenerate": degeneracy_result.get("is_degenerate"),
        "degeneracy_reason": degeneracy_result.get("recommended_interpretation"),
        "relative_ts_position": degeneracy_result.get("relative_ts_position"),
        "start_end_distance_ratio": degeneracy_result.get("start_end_distance_ratio"),
        "start_ts_distance_ratio": degeneracy_result.get("start_ts_distance_ratio"),
        "ts_end_distance_ratio": degeneracy_result.get("ts_end_distance_ratio"),
        "start_transition_jaccard": degeneracy_result.get("start_transition_jaccard"),
        "transition_end_jaccard": degeneracy_result.get("transition_end_jaccard"),
        "start_end_jaccard": degeneracy_result.get("start_end_jaccard"),
    }


def bootstrap_comparison_summary_columns(bootstrap_result: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten bootstrap comparison for scorecard / CSV export."""
    return {
        "canonical_bootstrap_relative_deviation": bootstrap_result.get("relative_path_deviation"),
        "bootstrap_envelope_mean_width": bootstrap_result.get("mean_envelope_width"),
        "path_stability_class": bootstrap_result.get("path_stability_class"),
    }


def interpretation_summary_columns(interpretation_result: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten transition interpretation for scorecard / CSV export."""
    return {
        "final_transition_class": interpretation_result.get("final_class"),
        "can_call_candidate_transition_state": interpretation_result.get(
            "can_call_candidate_transition_state"
        ),
        "can_call_biologically_supported_transition_state": interpretation_result.get(
            "can_call_biologically_supported_transition_state"
        ),
        "can_call_biologically_supported_transition_region": interpretation_result.get(
            "can_call_biologically_supported_transition_region"
        ),
        "recommended_sentence": interpretation_result.get("recommended_sentence"),
        "interpretation_label": interpretation_result.get("interpretation_label"),
    }


LAP_RELIABILITY_REPORT_PARAGRAPH = (
    "Path-local potential maxima were treated as algorithmic candidates only. A maximum was "
    "interpreted as a model-supported remodeling window only when it passed additional checks "
    "including endpoint separability, non-degenerate path geometry, bootstrap path stability, "
    "barrier height, manifold support, pseudotime/stage ordering, derivative alignment, and "
    "molecular evidence. Cell types with endpoint-overlapping maxima, high window overlap, "
    "weak barriers, or unstable bootstrap localization were conservatively interpreted as "
    "smooth remodeling, weak remodeling windows, or algorithmic path-local maxima rather than "
    "high-confidence remodeling regions."
)


def median_bootstrap_compute_paths(
    bootstrap_path_results: Sequence[dict],
    *,
    n_interp: int = 100,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Median bootstrap path in LAP compute space."""
    if not bootstrap_path_results:
        return None, None, None
    paths = []
    for pr in bootstrap_path_results:
        if not isinstance(pr, dict):
            continue
        path = pr.get("path_compute", pr.get("path"))
        if path is None:
            continue
        paths.append(np.asarray(path, dtype=float))
    if not paths:
        return None, None, None
    from lap_helpers import median_bootstrap_path

    return median_bootstrap_path(paths, n_interp=n_interp)


def build_validation_evidence_for_interpretation(
    *,
    region_summary_row: Optional[Dict[str, Any]] = None,
    molecular_summary: Optional[Dict[str, Any]] = None,
    ts_alignment: Optional[Dict[str, Any]] = None,
    bootstrap_comparison: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Map existing validation outputs to classify_transition_interpretation inputs."""
    row = region_summary_row or {}
    barrier_result = {
        "barrier_height": row.get("barrier_height"),
        "barrier_threshold": row.get("barrier_threshold"),
        "barrier_support": row.get("barrier_support_pass"),
        "passed": row.get("barrier_support_pass"),
    }
    manifold_result = {
        "manifold_support": row.get("manifold_support_pass"),
        "passed": row.get("manifold_support_pass"),
        "manifold_support_ratio": row.get("manifold_support_ratio"),
    }
    pseudotime_result = {
        "pseudotime_order": row.get("pseudotime_order_pass"),
        "passed": row.get("pseudotime_order_pass"),
        "pseudotime_order_status": row.get("pseudotime_order_status"),
    }
    derivative_result = {
        "derivative_alignment": row.get("derivative_alignment_pass"),
        "passed": row.get("derivative_alignment_pass"),
    }
    if ts_alignment:
        derivative_result.setdefault(
            "inside_remodeling_window", ts_alignment.get("inside_remodeling_window")
        )
        if ts_alignment.get("inside_remodeling_window") is True:
            derivative_result["passed"] = True

    molecular_result = dict(molecular_summary or {})
    if row.get("molecular_evidence_pass") is True:
        molecular_result["passed"] = True
    if row.get("n_transition_deg") is not None:
        molecular_result.setdefault("n_deg", row.get("n_transition_deg"))
    if row.get("n_enrichment_terms") is not None:
        molecular_result.setdefault("n_enrichment_terms", row.get("n_enrichment_terms"))
    if row.get("molecular_support"):
        molecular_result.setdefault("support", row.get("molecular_support"))

    path_stability_result = dict(bootstrap_comparison or {})
    if row.get("bootstrap_stability_pass") is True:
        path_stability_result["passed"] = True

    return {
        "barrier_result": barrier_result,
        "manifold_result": manifold_result,
        "pseudotime_result": pseudotime_result,
        "derivative_result": derivative_result,
        "molecular_result": molecular_result or None,
        "path_stability_result": path_stability_result or None,
    }


def compute_path_manifold_support_ratio(analyzer, path_result: dict) -> float:
    """Nearest-neighbor distance at transition/remodeling center divided by median spacing."""
    from sklearn.neighbors import NearestNeighbors

    path = np.asarray(path_result.get("path_compute", path_result.get("path")), dtype=float)
    if len(path) == 0:
        return float("nan")
    positions = np.asarray(analyzer.cell_positions_2d, dtype=float)
    center_idx = path_result.get(
        "remodeling_center_idx",
        path_result.get("transition_state_idx_composite", path_result.get("transition_state_idx", 0)),
    )
    center_idx = clip_path_index(int(center_idx), len(path))
    ts_point = np.asarray(path[center_idx], dtype=float).reshape(1, -1)
    nbrs = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(positions)
    nearest_distance = float(nbrs.kneighbors(ts_point)[0][0, 0])
    median_neighbor = float(mean_neighbor_spacing(positions))
    return float(nearest_distance / (median_neighbor + 1e-8))


def bootstrap_lap_from_state_cores(
    analyzer,
    adata,
    start_state: str,
    end_state: str,
    *,
    clustering_key: str = "stage",
    pseudotime_key: str = "pseudotime",
    core_fraction: float = 0.5,
    n_bootstrap: int = 50,
    random_state: int = 42,
    n_points: int = 25,
) -> Dict[str, Any]:
    """Bootstrap LAP paths by resampling stage-core endpoints (validation-style)."""
    from landscape_core import stage_core_cell_indices

    positions = np.asarray(analyzer.cell_positions_2d, dtype=float)
    labels = adata.obs[clustering_key].values
    pseudotime = (
        np.asarray(adata.obs[pseudotime_key].values, dtype=float)
        if pseudotime_key in adata.obs
        else None
    )
    start_core = stage_core_cell_indices(
        positions, labels, start_state, pseudotime=pseudotime, core_fraction=core_fraction, by="medoid"
    )
    end_core = stage_core_cell_indices(
        positions, labels, end_state, pseudotime=pseudotime, core_fraction=core_fraction, by="medoid"
    )
    if len(start_core) < 2 or len(end_core) < 2:
        return {"status": "skipped", "warnings": ["Too few core cells for bootstrap"], "path_results": []}

    rng = np.random.default_rng(random_state)
    path_results: List[dict] = []
    for _ in range(n_bootstrap):
        s_idx = int(rng.choice(start_core))
        e_idx = int(rng.choice(end_core))
        try:
            raw = analyzer.compute_least_action_path(
                positions[s_idx], positions[e_idx], n_points=n_points, use_3d=False, max_iter=60
            )
            raw["path_compute"] = np.asarray(raw.get("path"), dtype=float)
            path_results.append(raw)
        except Exception:
            continue

    return {
        "status": "ok" if path_results else "failed",
        "n_bootstrap": n_bootstrap,
        "n_success": len(path_results),
        "path_results": path_results,
        "warnings": [] if path_results else ["No successful bootstrap paths"],
    }


def build_candidate_lap_reliability(
    path_result: dict,
    analyzer,
    adata=None,
    *,
    mode: str = "reliability_minimal",
    start_state: str = "",
    end_state: str = "",
    clustering_key: str = "stage",
    pseudotime_key: str = "pseudotime",
    core_fraction: float = 0.5,
    n_bootstrap: int = 50,
    random_state: int = 42,
    window_size: int = 5,
    neighbors_per_path_point: int = 10,
) -> Dict[str, Any]:
    """
    Build lap_reliability bundle for an endpoint candidate path.

    reliability_minimal: degeneracy + manifold support (no bootstrap).
    full: adds bootstrap + canonical-bootstrap comparison.
    """
    lap_rel: Dict[str, Any] = {"mode": mode, "status": "ok", "warnings": []}
    try:
        degeneracy = diagnose_path_degeneracy(
            path_result,
            analyzer,
            adata=adata,
            window_size=window_size,
            neighbors_per_path_point=neighbors_per_path_point,
        )
    except Exception as exc:
        degeneracy = {"is_degenerate": True, "recommended_interpretation": str(exc), "warnings": [str(exc)]}
        lap_rel["status"] = "failed"
        lap_rel["warnings"].append(str(exc))

    manifold_ratio = compute_path_manifold_support_ratio(analyzer, path_result)
    degeneracy["manifold_support_ratio"] = manifold_ratio
    overlap = max(
        float(degeneracy.get("start_transition_jaccard", 0) or 0),
        float(degeneracy.get("transition_end_jaccard", 0) or 0),
        float(degeneracy.get("start_end_jaccard", 0) or 0),
        float(path_result.get("endpoint_overlap_fraction", 0) or 0),
    )
    degeneracy["endpoint_overlap_fraction"] = overlap
    lap_rel["degeneracy"] = degeneracy
    lap_rel["manifold_support_ratio"] = manifold_ratio

    if mode == "full" and start_state and end_state and adata is not None:
        boot = bootstrap_lap_from_state_cores(
            analyzer,
            adata,
            start_state,
            end_state,
            clustering_key=clustering_key,
            pseudotime_key=pseudotime_key,
            core_fraction=core_fraction,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        lap_rel["bootstrap"] = boot
        if boot.get("status") == "ok" and boot.get("path_results"):
            path_compute = np.asarray(path_result.get("path_compute", path_result.get("path")), dtype=float)
            median_path, lower_path, upper_path = median_bootstrap_compute_paths(boot["path_results"])
            if median_path is not None:
                bootstrap_comparison = compare_canonical_bootstrap_paths(
                    path_compute,
                    median_path,
                    lower_path,
                    upper_path,
                    n_interp=100,
                )
                lap_rel["bootstrap_comparison"] = bootstrap_comparison
        else:
            lap_rel["warnings"].extend(boot.get("warnings", []))

    return lap_rel


def scorecard_columns_from_lap_reliability(lap_reliability: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten lap reliability bundle for validation scorecard."""
    if not lap_reliability:
        return {}
    out: Dict[str, Any] = {}
    degeneracy = lap_reliability.get("degeneracy") or {}
    bootstrap = lap_reliability.get("bootstrap_comparison") or {}
    interpretation = lap_reliability.get("interpretation") or {}
    if degeneracy:
        out.update(degeneracy_summary_columns(degeneracy))
    if bootstrap:
        out.update(bootstrap_comparison_summary_columns(bootstrap))
    if interpretation:
        out.update(interpretation_summary_columns(interpretation))
    return out
