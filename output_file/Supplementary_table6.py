#!/usr/bin/env python3
"""Supplementary Table 6: deep-valley EOC DEGs (HGSOC).

Computes Wilcoxon DEGs (deep-valley vs other EOC) on the adopted HGSOC checkpoint.

Default output:
  output_file/Supplementary_table6.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _adopted import hgsoc_deep_valley_deg  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent / "Supplementary_table6.csv"


def compose(out: Path | None = None, *, top: int = 500) -> Path:
    out = Path(out or DEFAULT_OUT)
    df = hgsoc_deep_valley_deg()
    if "score" in df.columns:
        df = df.assign(_abs=df["score"].abs()).sort_values("_abs", ascending=False).drop(columns="_abs")
    elif "pval_adj" in df.columns:
        df = df.sort_values("pval_adj")
    if top and top > 0:
        df = df.head(int(top)).copy()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"rows={len(df)} cols={list(df.columns)[:8]}...", flush=True)
    print(f"wrote {out}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", nargs="?", default=str(DEFAULT_OUT))
    p.add_argument("--top", type=int, default=500, help="Keep top N genes (0 = all)")
    args = p.parse_args(argv)
    compose(out=Path(args.out), top=int(args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
