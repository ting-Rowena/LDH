import numpy as np
from sklearn.neighbors import NearestNeighbors

DEFAULT_UNCERTAINTY_WEIGHTS = {
    "bootstrap": 0.35,
    "manifold": 0.25,
    "drift": 0.25,
    "endpoint": 0.15,
    "degeneracy": 0.30,
}

ALL_UNCERTAINTY_COMPONENTS = ("bootstrap", "manifold", "drift", "endpoint", "degeneracy")


def _interp_path(path, n_points=100):
    path = np.asarray(path, dtype=float)
    if len(path) <= 1:
        return np.repeat(path[:1], n_points, axis=0)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 1e-12:
        return np.repeat(path[:1], n_points, axis=0)
    grid = np.linspace(0, total, n_points)
    out = np.zeros((n_points, path.shape[1]))
    for d in range(path.shape[1]):
        out[:, d] = np.interp(grid, cum, path[:, d])
    return out


def bootstrap_path_uncertainty(canonical_path, bootstrap_paths, n_points=100):
    canonical = _interp_path(canonical_path, n_points=n_points)
    if not bootstrap_paths:
        return 1.0

    boots = []
    for p in bootstrap_paths:
        if isinstance(p, dict):
            p = p.get("path_compute", p.get("path"))
        if p is None:
            continue
        boots.append(_interp_path(p, n_points=n_points))

    if not boots:
        return 1.0

    arr = np.stack(boots, axis=0)
    mean_path = arr.mean(axis=0)
    var = np.mean(np.sum((arr - mean_path[None, :, :]) ** 2, axis=2))
    scale = np.mean(np.sum(np.diff(canonical, axis=0) ** 2, axis=1)) + 1e-8
    return float(min(1.0, var / scale))


def manifold_uncertainty(path, cell_coords, neighbor_spacing=None):
    path = np.asarray(path, dtype=float)
    cell_coords = np.asarray(cell_coords, dtype=float)

    if len(path) == 0 or len(cell_coords) == 0:
        return 1.0
    if path.ndim != 2 or cell_coords.ndim != 2 or path.shape[1] != cell_coords.shape[1]:
        return 1.0

    nbrs = NearestNeighbors(n_neighbors=1).fit(cell_coords)
    dist, _ = nbrs.kneighbors(path)

    if neighbor_spacing is None:
        k = min(2, len(cell_coords))
        nbrs2 = NearestNeighbors(n_neighbors=k).fit(cell_coords)
        d2, _ = nbrs2.kneighbors(cell_coords)
        neighbor_spacing = float(np.median(d2[:, -1])) + 1e-8

    return float(np.mean(dist[:, 0]) / (neighbor_spacing + 1e-8))


def drift_mismatch_uncertainty(path, drift_vectors=None, pseudotime=None, eps=1e-6):
    path = np.asarray(path, dtype=float)

    if len(path) < 2:
        return 1.0

    dz = np.diff(path, axis=0)

    if pseudotime is None:
        dt = np.ones(len(path) - 1)
    else:
        tau = np.asarray(pseudotime, dtype=float).ravel()
        # Must be path-aligned (one tau per path node). Cell-level τ is a common misuse.
        if tau.size != len(path):
            dt = np.ones(len(path) - 1)
        else:
            dt = np.clip(np.diff(tau), eps, None)

    velocity = dz / dt[:, None]

    if drift_vectors is None:
        step_norm = np.linalg.norm(dz, axis=1)
        vel_norm = np.linalg.norm(velocity, axis=1)
        return float(min(1.0, np.mean(vel_norm / (step_norm + 1e-8))))

    drift = np.asarray(drift_vectors, dtype=float)
    if len(drift) == len(path):
        drift = drift[:-1]
    if drift.shape != velocity.shape:
        drift = drift[: len(velocity), : velocity.shape[1]]

    mismatch = velocity - drift
    denom = np.mean(np.linalg.norm(velocity, axis=1)) + 1e-8
    return float(min(1.0, np.mean(np.linalg.norm(mismatch, axis=1)) / denom))


def endpoint_uncertainty(start_coords, end_coords, neighbor_spacing):
    start_coords = np.asarray(start_coords, dtype=float)
    end_coords = np.asarray(end_coords, dtype=float)

    if len(start_coords) == 0 or len(end_coords) == 0:
        return 1.0

    mu0 = start_coords.mean(axis=0)
    mu1 = end_coords.mean(axis=0)
    sep = np.linalg.norm(mu1 - mu0) / (neighbor_spacing + 1e-8)

    return float(np.exp(-sep))


def _valid_bootstrap_paths(bootstrap_paths):
    if not bootstrap_paths:
        return []
    boots = []
    for p in bootstrap_paths:
        if isinstance(p, dict):
            p = p.get("path_compute", p.get("path"))
        if p is None:
            continue
        arr = np.asarray(p, dtype=float)
        if arr.size == 0:
            continue
        boots.append(arr)
    return boots


def _uncertainty_status_from_evaluated(evaluated: list, missing_components: list) -> str:
    if not evaluated:
        return "insufficient"
    if not missing_components:
        return "complete"
    return "partial"


def compute_path_uncertainty(
    path,
    *,
    bootstrap_paths=None,
    cell_coords=None,
    drift_vectors=None,
    pseudotime=None,
    start_coords=None,
    end_coords=None,
    neighbor_spacing=None,
    weights=None,
    path_degenerate=False,
):
    weights = dict(DEFAULT_UNCERTAINTY_WEIGHTS if weights is None else weights)
    if "degeneracy" not in weights:
        weights["degeneracy"] = DEFAULT_UNCERTAINTY_WEIGHTS["degeneracy"]

    path = np.asarray(path, dtype=float)
    components = {
        "bootstrap": np.nan,
        "manifold": np.nan,
        "drift": np.nan,
        "endpoint": np.nan,
        "degeneracy": np.nan,
    }
    component_status = {
        "bootstrap": "not_evaluated",
        "manifold": "not_evaluated",
        "drift": "not_evaluated",
        "endpoint": "not_evaluated",
        "degeneracy": "not_evaluated",
    }

    boots = _valid_bootstrap_paths(bootstrap_paths)
    if boots:
        components["bootstrap"] = float(
            np.clip(bootstrap_path_uncertainty(path, boots), 0.0, 1.0)
        )
        component_status["bootstrap"] = "evaluated"

    if cell_coords is not None:
        cell_coords_arr = np.asarray(cell_coords, dtype=float)
        if (
            len(path) > 0
            and cell_coords_arr.size > 0
            and path.ndim == 2
            and cell_coords_arr.ndim == 2
            and path.shape[1] == cell_coords_arr.shape[1]
        ):
            components["manifold"] = float(
                np.clip(
                    manifold_uncertainty(path, cell_coords_arr, neighbor_spacing=neighbor_spacing),
                    0.0,
                    1.0,
                )
            )
            component_status["manifold"] = "evaluated"

    if len(path) >= 2:
        components["drift"] = float(
            np.clip(
                drift_mismatch_uncertainty(
                    path,
                    drift_vectors=drift_vectors,
                    pseudotime=pseudotime,
                ),
                0.0,
                1.0,
            )
        )
        component_status["drift"] = "evaluated"

    if start_coords is not None and end_coords is not None and neighbor_spacing is not None:
        start_arr = np.asarray(start_coords, dtype=float)
        end_arr = np.asarray(end_coords, dtype=float)
        if start_arr.size > 0 and end_arr.size > 0:
            components["endpoint"] = float(
                np.clip(
                    endpoint_uncertainty(start_arr, end_arr, neighbor_spacing),
                    0.0,
                    1.0,
                )
            )
            component_status["endpoint"] = "evaluated"

    if path_degenerate:
        components["degeneracy"] = 1.0
        component_status["degeneracy"] = "evaluated"

    evaluated = [
        key
        for key in component_status
        if component_status[key] == "evaluated"
    ]
    evaluated_components = list(evaluated)
    missing_components = [
        key
        for key in component_status
        if component_status[key] == "not_evaluated"
    ]
    n_evaluated_components = len(evaluated)
    uncertainty_status = _uncertainty_status_from_evaluated(evaluated, missing_components)

    if n_evaluated_components == 0:
        return {
            "path_uncertainty": float("nan"),
            "path_reliability": 1.0,
            "uncertainty_components": components,
            "component_status": component_status,
            "evaluated_components": evaluated_components,
            "missing_components": missing_components,
            "n_evaluated_components": 0,
            "uncertainty_status": "insufficient_evidence",
            "degeneracy_penalty": float(components.get("degeneracy", np.nan)) if path_degenerate else 0.0,
        }

    weight_sum = sum(float(weights.get(key, 0.0)) for key in evaluated)
    if weight_sum <= 0.0:
        return {
            "path_uncertainty": float("nan"),
            "path_reliability": 1.0,
            "uncertainty_components": components,
            "component_status": component_status,
            "evaluated_components": evaluated_components,
            "missing_components": missing_components,
            "n_evaluated_components": n_evaluated_components,
            "uncertainty_status": "insufficient_evidence",
        }

    total = sum(
        float(weights.get(key, 0.0)) * float(components[key])
        for key in evaluated
    ) / weight_sum
    total = float(np.clip(total, 0.0, 1.0))
    reliability = float(1.0 / (1.0 + total))

    return {
        "path_uncertainty": total,
        "path_reliability": reliability,
        "uncertainty_components": components,
        "component_status": component_status,
        "evaluated_components": evaluated_components,
        "missing_components": missing_components,
        "n_evaluated_components": n_evaluated_components,
        "uncertainty_status": uncertainty_status,
    }
