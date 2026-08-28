#!/usr/bin/env python3
"""Supplementary Table 2: LDH-scRNA vs scVelo vs CellRank (trajectory–time PCC).

Manuscript counterpart of main-text Table 1 / Fig. 2d.

Three named methods, each with its native order reconstruction on the same
``X_latent_pca`` (10 PCs):

  - LDH-scRNA — MomentumNetwork field → absorbing-Markov hitting time
    (same fate-order protocol as CellRank; **not** the supervised time head)
  - scVelo    — kNN time-ordered velocity proxy → graph-Laplacian order
  - CellRank  — same kNN velocity → absorbing-Markov hitting time

Default output:
  output_file/Supplementary_table2.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _adopted import SOTA_BIOLOGY, run_sota_pcc  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent / "Supplementary_table2.csv"


def compose(out: Path | None = None, *, device: str = "cpu", recompute: bool = False) -> Path:
    out = Path(out or DEFAULT_OUT)
    pcc = run_sota_pcc(device=device, recompute=recompute)
    rows = []
    for dataset in ("GSE155622", "GSE141259", "HGSOC"):
        r = pcc.loc[pcc["dataset"].astype(str) == dataset].iloc[0]
        ldh = r.get("MomentumNetwork_markov", r["MomentumNetwork"])
        rows.append(
            {
                "Dataset": dataset,
                "Biological Context & Model System": SOTA_BIOLOGY[dataset],
                "LDH-scRNA": round(float(ldh), 4),
                "scVelo": round(float(r["scVelo"]), 4),
                "CellRank": round(float(r["CellRank"]), 4),
            }
        )
    paper = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    paper.to_csv(out, index=False)
    print(paper.to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "out",
        nargs="?",
        default=str(DEFAULT_OUT),
        help="Output CSV path (default: output_file/Supplementary_table2.csv)",
    )
    p.add_argument("--device", default="cpu", help="Device for SOTA benchmark (cpu/cuda)")
    p.add_argument("--recompute", action="store_true", help="Ignore output_file/_cache and re-run")
    args = p.parse_args(argv)
    compose(out=Path(args.out), device=str(args.device), recompute=bool(args.recompute))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
