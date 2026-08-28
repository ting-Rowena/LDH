#!/usr/bin/env python
"""
\"""\"\"
Unified cell-type LAP analysis entry point.

Examples:
    python run_lap_analysis.py --dataset HGSOC --cell-type EOC --start IIIC --end IVB
    python run_lap_analysis.py --dataset GSE155622 --list-cell-types
    python run_lap_analysis.py --dataset GSE225948_Brain --all-cell-types --paths-only
"""

from __future__ import annotations

import argparse
import sys

from plot_utils import configure_headless

configure_headless()

from celltype_analysis import DATASET_REGISTRY, main_from_profile


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dataset", required=True, choices=sorted(DATASET_REGISTRY.keys()))
    pre_args, remaining = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    main_from_profile(DATASET_REGISTRY[pre_args.dataset])


if __name__ == "__main__":
    main()
