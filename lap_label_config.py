"""
Custom start/end state label positions for LAP UMAP figures.

Edit ``lap_umap_label_overrides.json`` after the first run, then regenerate figures:

    python scripts/regenerate_lap_umap_labels.py --dataset HGSOC --cell-type EOC

Each figure key (e.g. ``HGSOC_EOC_umap_canonical_medoid``) supports ``start`` / ``end``:

  - Offset mode (default): ``dx``, ``dy`` added to path endpoint in UMAP space
  - Absolute mode: set ``x`` and ``y`` directly (ignore path endpoint)
  - Optional: ``ha``, ``va``, ``fontsize``, ``color``, ``label`` (override text)"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

DEFAULT_LABEL_CONFIG_NAME = "lap_umap_label_overrides.json"


def default_label_config_path(figure_dir: str) -> Path:
    return Path(figure_dir) / DEFAULT_LABEL_CONFIG_NAME


def load_label_overrides(path: Optional[str | Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def save_label_overrides(path: str | Path, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _endpoint_style(
    side: str,
    overrides: Dict[str, Any],
    figure_key: str,
    default_label: str,
    path_point: np.ndarray,
) -> Dict[str, Any]:
    """Merge defaults with user override for start or end."""
    base = {
        "label": default_label,
        "dx": 0.0,
        "dy": 0.0,
        "ha": "center" if side == "start" else "center",
        "va": "bottom" if side == "start" else "top",
        "fontsize": 7,
        "color": "k",
        "ref_x": float(path_point[0]),
        "ref_y": float(path_point[1]),
    }
    entry = overrides.get(figure_key) or {}
    custom = entry.get(side) or {}
    if isinstance(custom, dict):
        base.update({k: v for k, v in custom.items() if v is not None})
    return base


def resolve_text_position(path_point: np.ndarray, style: Dict[str, Any]) -> Tuple[float, float]:
    if "x" in style and "y" in style:
        return float(style["x"]), float(style["y"])
    return (
        float(path_point[0]) + float(style.get("dx", 0.0)),
        float(path_point[1]) + float(style.get("dy", 0.0)),
    )


def endpoint_styles_for_figure(
    overrides: Dict[str, Any],
    figure_key: str,
    path: np.ndarray,
    start_label: str,
    end_label: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    path = np.asarray(path, dtype=float)
    start = _endpoint_style("start", overrides, figure_key, start_label, path[0])
    end = _endpoint_style("end", overrides, figure_key, end_label, path[-1])
    return start, end


def export_label_template(
    config_path: str | Path,
    figure_key: str,
    path: np.ndarray,
    start_label: str,
    end_label: str,
    *,
    merge: bool = True,
) -> None:
    """Write / update template entries without overwriting existing user offsets."""
    p = Path(config_path)
    data = load_label_overrides(p) if merge and p.is_file() else {}
    path = np.asarray(path, dtype=float)

    if figure_key not in data:
        data[figure_key] = {}

    for side, label, pt in (
        ("start", start_label, path[0]),
        ("end", end_label, path[-1]),
    ):
        if side in data[figure_key] and merge:
            entry = data[figure_key][side]
            entry.setdefault("label", label)
            entry.setdefault("ref_x", float(pt[0]))
            entry.setdefault("ref_y", float(pt[1]))
            entry.setdefault("dx", 0.0)
            entry.setdefault("dy", 0.0)
            entry.setdefault("ha", "center")
            entry.setdefault("va", "bottom" if side == "start" else "top")
        else:
            data[figure_key][side] = {
                "label": label,
                "ref_x": float(pt[0]),
                "ref_y": float(pt[1]),
                "dx": 0.0,
                "dy": 0.0,
                "ha": "center",
                "va": "bottom" if side == "start" else "top",
                "_comment": "Adjust dx/dy (UMAP units) or set absolute x/y; then regenerate figures.",
            }

    save_label_overrides(p, data)


def path_result_to_cache(path_result: dict) -> dict:
    """JSON-serializable subset of a LAP path result."""
    out = {
        "path": np.asarray(path_result["path"], dtype=float).tolist(),
        "transition_state_idx": int(path_result["transition_state_idx"]),
        "start_state": path_result.get("start_state"),
        "end_state": path_result.get("end_state"),
        "total_action": float(path_result.get("total_action", 0.0)),
    }
    if "path_compute" in path_result:
        out["path_compute"] = np.asarray(path_result["path_compute"], dtype=float).tolist()
    return out


def save_lap_path_cache(cache_path: str | Path, payload: dict) -> None:
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_lap_path_cache(cache_path: str | Path) -> dict:
    with Path(cache_path).open(encoding="utf-8") as f:
        return json.load(f)


def cache_path_result(entry: dict) -> dict:
    """Restore minimal path_result dict from cache entry."""
    out = {
        "path": np.asarray(entry["path"], dtype=float),
        "transition_state_idx": int(entry["transition_state_idx"]),
        "start_state": entry.get("start_state"),
        "end_state": entry.get("end_state"),
        "total_action": entry.get("total_action", 0.0),
    }
    if "path_compute" in entry:
        out["path_compute"] = np.asarray(entry["path_compute"], dtype=float)
    return out
