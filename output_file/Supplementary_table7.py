#!/usr/bin/env python3
"""Supplementary Table 7: CCC ligand–receptor ranks (HGSOC).

Computes targeted LR scores between deep-valley EOC and high-U stromal cells
on the adopted HGSOC checkpoint.

Default output:
  output_file/Supplementary_table7.xlsx
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _adopted import hgsoc_ccc  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent / "Supplementary_table7.xlsx"
P1 = ROOT / "output_file" / "robustness" / "p1_robustness"
P1_SHEETS = {
    "observed_focus_pairs": P1 / "ccc_observed_focus_pairs.csv",
    "patient_summary": P1 / "ccc_patient_summary.csv",
    "band_permutation_null": P1 / "ccc_band_permutation_null.csv",
}


def compose(out: Path | None = None) -> Path:
    out = Path(out or DEFAULT_OUT)
    tables = hgsoc_ccc()
    extra = {}
    missing_p1 = [name for name, src in P1_SHEETS.items() if not src.is_file()]
    if missing_p1:
        print("[table7] computing CCC permutation / patient nulls...", flush=True)
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_p1_robustness import main as _p1

        _p1()
    for name, src in P1_SHEETS.items():
        if src.is_file():
            extra[name] = pd.read_csv(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for name, df in {**tables, **extra}.items():
            if df is None or df.empty:
                print(f"[skip] empty {name}", flush=True)
                continue
            df.to_excel(writer, sheet_name=name[:31], index=False)
            print(f"  sheet {name}: {len(df)} rows", flush=True)
    print(f"wrote {out}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    compose(out=Path(argv[0]) if argv else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
