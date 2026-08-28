"""
Validation region label semantics.

Six labels are used across validation outputs; each has a distinct meaning:

- start_window / end_window: k-NN cell collections near the first/last path windows
  (groups may overlap; no mutual-exclusivity rule).
- transition_window: k-NN cells near the path-local remodeling window
  (legacy label `transition_window`; ±window_size around remodeling_center); not consensus evidence by itself.
- start_basin / end_basin: mutually exclusive cell attribution from assign_lap_regions
  (path k-NN + overlap priority + optional stage fallback).
- transition_region: consensus-supported remodeling_region cells from multi-path
  evidence (legacy label `transition_region`; combined_support_score threshold); used for DEG middle group and UMAP.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

# Path-index windows (k-NN collections; groups may overlap)
PATH_WINDOW_LABELS: Tuple[str, ...] = ("start_window", "transition_window", "end_window")

# Mutually exclusive basin attribution (assign_lap_regions)
BASIN_LABELS: Tuple[str, ...] = ("start_basin", "end_basin")

# All lap_region categories (path-local + consensus + other)
LAP_REGION_CATEGORIES: Tuple[str, ...] = (
    "start_basin",
    "transition_region",
    "transition_window",
    "end_basin",
    "other",
)

# DEG / heatmap display order when consensus labels are present
DEG_REGION_ORDER: Tuple[str, ...] = (
    "start_basin",
    "transition_region",
    "transition_window",
    "end_basin",
    "other",
    "high_potential",
    "low_potential",
)

# Violin plots mixing path windows (endpoints) with consensus middle group
DISTRIBUTION_VIOLIN_ORDER: Tuple[str, ...] = (
    "start_window",
    "transition_region",
    "end_window",
)

# UMAP region overlay order
REGION_UMAP_ORDER: Tuple[str, ...] = (
    "other",
    "start_basin",
    "end_basin",
    "transition_region",
)

REGION_PRIORITY: Dict[str, int] = {
    "transition_region": 4,
    "transition_window": 3,
    "start_basin": 2,
    "end_basin": 1,
    "other": 0,
}

# DEG when consensus transition_region cells are assigned
CONSENSUS_DEG_COMPARISON_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("transition_vs_start", "transition_region", "start_basin"),
    ("transition_vs_end", "transition_region", "end_basin"),
    ("end_vs_start", "end_basin", "start_basin"),
)

# DEG when only path-local assign_lap_regions labels exist
PATH_DEG_COMPARISON_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("transition_vs_start", "transition_window", "start_basin"),
    ("transition_vs_end", "transition_window", "end_basin"),
    ("end_vs_start", "end_basin", "start_basin"),
)

# Pairwise comparisons whose significant DEG union feeds the summary path-region heatmap
PATH_PAIRWISE_DEG_COMPARISONS: Tuple[str, ...] = (
    "transition_vs_start",
    "transition_vs_end",
    "end_vs_start",
)

PATH_REGION_HEATMAP_MAX_GENES = 30
PATH_REGION_HEATMAP_PER_COMPARISON_CAP = 15


def path_region_heatmap_categories(lap_region_values: Sequence[str], *, include_other: bool = True) -> Tuple[str, ...]:
    """Column order for summary path-region DEG heatmap (middle = consensus or path-local)."""
    present = set(str(v) for v in lap_region_values)
    middle = "transition_region" if "transition_region" in present else "transition_window"
    order: Tuple[str, ...] = ("start_basin", middle, "end_basin")
    if include_other and "other" in present:
        order = order + ("other",)
    return tuple(r for r in order if r in present)

REGION_LABEL_DESCRIPTIONS: Dict[str, str] = {
    "start_window": "k-NN cells near first path window (path-index window)",
    "end_window": "k-NN cells near last path window (path-index window)",
    "transition_window": "k-NN cells near path-local remodeling window (legacy label; see remodeling_center)",
    "start_basin": "mutually exclusive start attribution (path k-NN + stage fallback)",
    "end_basin": "mutually exclusive end attribution (path k-NN + stage fallback)",
    "transition_region": "consensus-supported remodeling_region cells (legacy label; multi-path evidence)",
    "other": "cells outside the above labels",
}


def deg_comparison_specs(lap_region_values: Sequence[str]) -> Tuple[Tuple[str, str, str], ...]:
    """Pick DEG comparisons based on whether consensus transition_region is present."""
    present = set(str(v) for v in lap_region_values)
    if "transition_region" in present:
        return CONSENSUS_DEG_COMPARISON_SPECS
    return PATH_DEG_COMPARISON_SPECS


def path_window_title_suffix() -> str:
    return " (start_window / transition_window / end_window)"


def distribution_violin_title_suffix() -> str:
    return " (start_window / transition_region / end_window)"


def basin_umap_title_suffix() -> str:
    return " (start_basin / end_basin / transition_region)"
