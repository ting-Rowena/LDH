#!/usr/bin/env python3
r"""Supplementary Table 3: dual-metric matched null control audit.

Manuscript counterpart of former main-text Table 2
(“Dual-metric null control audit: Retrained shuffling versus true-time models”).

Canonical protocol (``scripts/run_matched_temporal_null.py``):
  - Preserve expression and cell type
  - GSE155622 / GSE141259: jointly permute temporal metadata (``temporal_matched``)
  - HGSOC: shuffle ``treatment_phase`` within patient (``pairing_matched``)
  - From-scratch retrain: 500 epochs × 4 replicates × 5000 cells × batch 128
  - Score on the clean unshuffled subset

Two metrics (real vs null median; collapse = null / real):
  1. U0–KDE Spearman (geometric consistency)
  2. Holdout forecasting PCC

Default mode **assembles** the checkpoint-recorded null experiment JSON
(``methods_enhancement/physical_retrain_controls_*_summary.json``) — the 500-epoch
retrain is a training-scale experiment, not a figure-time compute. Real metrics
are those stored with that experiment.

Optional ``--rebuild`` re-runs the full null protocol per dataset (very slow).

Default output:
  output_file/Supplementary_table3.csv

Usage:
  python output_file/Supplementary_table3.py
  python output_file/Supplementary_table3.py --rebuild --device cuda
  python output_file/Supplementary_table3.py /path/to/out.csv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _adopted import ADOPTED, NULL_MODE, load_null_summary  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent / "Supplementary_table3.csv"

PANELS = [(ds, ADOPTED[ds], NULL_MODE[ds]) for ds in ("GSE155622", "GSE141259", "HGSOC")]

METRIC_SPEARMAN = "U0–KDE Spearman (geometric consistency)"
METRIC_HOLDOUT = "Holdout forecasting PCC"


def _summary_json(ck: Path, dataset: str, mode: str) -> Path:
    tag = f"{dataset}_{mode}_e500_mc5000_bs128"
    return ck / "methods_enhancement" / f"physical_retrain_controls_{tag}_summary.json"


def _load_summary_frame() -> pd.DataFrame:
    return load_null_summary()


def rebuild_nulls(*, device: str | None = None) -> None:
    """Re-run canonical matched temporal null for each adopted checkpoint (slow)."""
    for dataset, ck, _mode in PANELS:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_matched_temporal_null.py"),
            "--dataset",
            dataset,
            "--checkpoint-dir",
            str(ck),
            "--n-replicates",
            "4",
            "--n-epochs",
            "500",
            "--max-cells",
            "5000",
            "--batch-size",
            "128",
            "--seed",
            "42",
        ]
        if device:
            cmd.extend(["--device", device])
        print("[rebuild]", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(ROOT), check=True)

    # Refresh manuscript summary from JSON shards when present
    rows = []
    for dataset, ck, mode in PANELS:
        jp = _summary_json(ck, dataset, mode)
        if not jp.is_file():
            raise FileNotFoundError(f"Expected summary JSON after rebuild: {jp}")
        rows.append(json.loads(jp.read_text(encoding="utf-8")))
    out = pd.DataFrame(rows)
    cache = Path(__file__).resolve().parent / "_cache" / "matched_temporal_null_summary.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    print(f"wrote {cache}", flush=True)


def compose(out: Path | None = None, *, rebuild: bool = False, device: str | None = None) -> Path:
    out = Path(out or DEFAULT_OUT)
    if rebuild:
        rebuild_nulls(device=device)

    summ = _load_summary_frame()
    summ = summ.set_index(summ["dataset"].astype(str))

    rows = []
    for dataset, _ck, mode in PANELS:
        if dataset not in summ.index:
            raise KeyError(f"{dataset} not in null summary")
        r = summ.loc[dataset]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        rows.append(
            {
                "Metric": METRIC_SPEARMAN,
                "Dataset": dataset,
                "Shuffle mode": mode,
                "Real Model": round(float(r["real_spearman"]), 3),
                "Null Model (Retrained)": round(float(r["null_median_spearman"]), 3),
                "Collapse ratio": round(float(r["collapse_ratio"]), 3),
            }
        )
    for dataset, _ck, mode in PANELS:
        r = summ.loc[dataset]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        rows.append(
            {
                "Metric": METRIC_HOLDOUT,
                "Dataset": dataset,
                "Shuffle mode": mode,
                "Real Model": round(float(r["real_holdout_pcc"]), 3),
                "Null Model (Retrained)": round(float(r["null_median_holdout_pcc"]), 3),
                "Collapse ratio": round(float(r["holdout_pcc_collapse_ratio"]), 3),
            }
        )

    paper = pd.DataFrame(rows)[
        ["Metric", "Dataset", "Real Model", "Null Model (Retrained)", "Collapse ratio"]
    ]
    for col in ("Real Model", "Null Model (Retrained)", "Collapse ratio"):
        paper[col] = paper[col].map(lambda x: f"{float(x):.3f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    paper.to_csv(out, index=False)
    print(paper.to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    for _, r in pd.DataFrame(rows).iterrows():
        print(
            f"  {r['Metric'][:24]:24s} {r['Dataset']:10s} "
            f"mode={r['Shuffle mode']:16s} "
            f"real={r['Real Model']:.3f} null={r['Null Model (Retrained)']:.3f} "
            f"collapse={r['Collapse ratio']:.3f}",
            flush=True,
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "out",
        nargs="?",
        default=str(DEFAULT_OUT),
        help="Output CSV path (default: output_file/Supplementary_table3.csv)",
    )
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-run matched temporal null (500ep×4rep×5000cells) then assemble (very slow)",
    )
    p.add_argument("--device", default=None, help="Device for --rebuild (cpu/cuda)")
    args = p.parse_args(argv)
    compose(out=Path(args.out), rebuild=bool(args.rebuild), device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
