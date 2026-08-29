#!/usr/bin/env python3
"""Supplementary Table 8: patient-level PDVS OS analyses (TCGA-OV and GSE26712).

Sheets: summary, TCGA patients, AOCS patients, Cox TCGA, Cox AOCS.
Regenerates clinical tables if patient/Cox files are missing.

Default output:
  output_file/Supplementary_table8.xlsx
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _adopted import CK_HG  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent / "Supplementary_table8.xlsx"

REQUIRED = [
    ("summary", "pdvs_clinical_summary.csv"),
    ("TCGA_patients", "pdvs_TCGA_OV_patient_table.csv"),
    ("AOCS_patients", "pdvs_AOCS_GSE26712_patient_table.csv"),
    ("Cox_TCGA", "pdvs_cox_TCGA_OV.csv"),
    ("Cox_AOCS", "pdvs_cox_AOCS_GSE26712.csv"),
]


def compose(out: Path | None = None) -> Path:
    out = Path(out or DEFAULT_OUT)
    from run_clinical_pdvs_validation import methods_outdir, run_clinical_pdvs_validation

    me = methods_outdir(CK_HG)
    missing = [name for name, fn in REQUIRED if not (me / fn).is_file()]
    if missing:
        print(f"[table8] missing {missing}; running PDVS clinical validation...", flush=True)
        run_clinical_pdvs_validation(CK_HG)
    else:
        print("[table8] using patient-level PDVS tables", flush=True)

    extra = [
        ("TCGA_scores", me / "pdvs_TCGA_OV_scores.csv"),
        ("AOCS_scores", me / "pdvs_AOCS_GSE26712_scores.csv"),
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        n_sheets = 0
        for name, fn in REQUIRED:
            src = me / fn
            if not src.is_file():
                print(f"[skip] missing {src}", flush=True)
                continue
            df = pd.read_csv(src)
            df.to_excel(writer, sheet_name=name[:31], index=False)
            n_sheets += 1
            print(f"  sheet {name}: {len(df)} rows", flush=True)
        for name, src in extra:
            if not src.is_file():
                continue
            pd.read_csv(src).to_excel(writer, sheet_name=name[:31], index=False)
            print(f"  sheet {name}: extra scores", flush=True)
            n_sheets += 1
        if n_sheets == 0:
            pd.DataFrame({"status": ["no PDVS clinical tables written"]}).to_excel(
                writer, sheet_name="status", index=False
            )
    print(f"wrote {out}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    compose(out=Path(argv[0]) if argv else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
