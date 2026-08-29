#!/usr/bin/env python3
"""Reproduce manuscript figures and tables written under ``output_file/``.

Run from the repository root:

  python output_file/reproduce.py --check
  python output_file/reproduce.py --group fast
  python output_file/reproduce.py --group all

Adopted checkpoints must sit at the repo root (Zenodo
https://doi.org/10.5281/zenodo.22146979; see DATA_AND_CHECKPOINTS.md).
Raw ``.h5ad`` files are required only for jobs tagged ``h5ad``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "output_file"))

from _adopted import ADOPTED, CK_HG, CK_LUNG, CK_PAIN  # noqa: E402

PYTHON = sys.executable


@dataclass(frozen=True)
class Job:
    name: str
    argv: tuple[str, ...]
    groups: tuple[str, ...]
    needs_h5ad: bool = False
    note: str = ""


JOBS: tuple[Job, ...] = (
    Job("figure2", ("output_file/figure2.py",), ("fast", "main", "all")),
    Job("figure3_bc", ("output_file/figure3_bc.py",), ("fast", "main", "all")),
    Job("figure3_de", ("output_file/figure3_de.py",), ("fast", "main", "all")),
    Job("figure3_fg", ("output_file/figure3_fg.py",), ("fast", "main", "all")),
    Job("figure3_hijk", ("output_file/figure3_hijk.py",), ("fast", "main", "all")),
    Job("figure4_bc", ("output_file/figure4_bc.py",), ("fast", "main", "all")),
    Job("figure4_d", ("output_file/figure4_d.py",), ("slow", "main", "all"), note="rebuilds 3D field + LAP"),
    Job("figure4_efghi", ("output_file/figure4_efghi.py",), ("fast", "main", "all")),
    Job("figure4_j", ("output_file/figure4_j.py",), ("slow", "main", "all")),
    Job("figure4_klm", ("output_file/figure4_klm.py",), ("fast", "main", "all")),
    Job("figure5_b", ("output_file/figure5_b.py",), ("fast", "main", "all")),
    Job("figure5_cd", ("output_file/figure5_cd.py",), ("slow", "main", "all"), needs_h5ad=True, note="hybrid KO"),
    Job("figure5_ef", ("output_file/figure5_ef.py",), ("slow", "main", "all"), note="may download TCGA/GSE26712"),
    Job("supp_fig1", ("output_file/Supplementary_figure1.py",), ("fast", "supp", "all")),
    Job("supp_fig2", ("output_file/Supplementary_figure2.py",), ("fast", "supp", "all")),
    Job("supp_fig3", ("output_file/Supplementary_figure3.py",), ("fast", "supp", "all")),
    Job("supp_fig4", ("output_file/Supplementary_figure4.py",), ("slow", "supp", "all"), needs_h5ad=True),
    Job("supp_fig5", ("output_file/Supplementary_figure5.py",), ("slow", "supp", "all"), needs_h5ad=True),
    Job("supp_fig6", ("output_file/Supplementary_figure6.py",), ("fast", "supp", "all")),
    Job(
        "mac_triad",
        ("scripts/analyze_mac_fn1_m2_resolution_triad.py",),
        ("slow", "supp", "all"),
        note="feeds Supplementary Figure 7",
    ),
    Job(
        "mac_paths",
        ("scripts/analyze_mac_landscape_path_endorsement.py",),
        ("slow", "supp", "all"),
        note="feeds Supplementary Figure 7",
    ),
    Job("supp_fig7", ("output_file/Supplementary_figure7.py",), ("slow", "supp", "all")),
    Job("supp_fig8", ("output_file/Supplementary_figure8.py",), ("slow", "supp", "all"), needs_h5ad=True),
    Job("supp_tab1", ("output_file/Supplementary_table1.py",), ("fast", "tables", "all")),
    Job("supp_tab2", ("output_file/Supplementary_table2.py",), ("slow", "tables", "all"), note="SOTA PCC; optional --device cuda"),
    Job("supp_tab3", ("output_file/Supplementary_table3.py",), ("fast", "tables", "all"), note="uses recorded null JSON; not a 500-epoch retrain"),
    Job("supp_tab4", ("output_file/Supplementary_table4.py", "--no-rebuild"), ("fast", "tables", "all")),
    Job("supp_tab5", ("output_file/Supplementary_table5.py",), ("fast", "tables", "all")),
    Job("supp_tab6", ("output_file/Supplementary_table6.py",), ("fast", "tables", "all")),
    Job("supp_tab7", ("output_file/Supplementary_table7.py",), ("slow", "tables", "all")),
    Job("supp_tab8", ("output_file/Supplementary_table8.py",), ("slow", "tables", "all"), note="TCGA/GSE26712 PDVS OS"),
)

REQUIRED_CK_FILES = (
    "best_model.pth",
    "obs.csv",
    "Loss_epoch.csv",
    "training_summary.json",
    "training_var_names.json",
)


def check_adopted() -> list[str]:
    missing: list[str] = []
    for name, ck in ADOPTED.items():
        if not ck.is_dir():
            missing.append(f"missing checkpoint directory: {ck.name} ({name})")
            continue
        for fn in REQUIRED_CK_FILES:
            p = ck / fn
            if not p.is_file():
                missing.append(f"{ck.name}/{fn}")
        umap = ck / "training_umap.npz"
        if not umap.is_file():
            missing.append(f"{ck.name}/training_umap.npz (needed for Supplementary Figure 1)")
    table2 = ROOT / "deep_temporal_benchmark_compare" / "Supplementary_table2.csv"
    if not table2.is_file():
        missing.append("deep_temporal_benchmark_compare/Supplementary_table2.csv (needed for Figure 2)")
    return missing


def list_jobs(group: str) -> list[Job]:
    return [j for j in JOBS if group in j.groups]


def run_job(job: Job) -> int:
    cmd = [PYTHON, *job.argv]
    print(f"\n=== {job.name} ===\n{' '.join(cmd)}", flush=True)
    if job.note:
        print(f"  note: {job.note}", flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--group",
        choices=("fast", "slow", "main", "supp", "tables", "all"),
        default=None,
        help="Which job set to run",
    )
    p.add_argument("--check", action="store_true", help="Only verify adopted checkpoints")
    p.add_argument("--list", action="store_true", help="Print jobs and exit")
    p.add_argument("--only", nargs="+", default=None, help="Run named jobs only")
    args = p.parse_args(argv)

    problems = check_adopted()
    if problems:
        print("Checkpoint / asset check:", flush=True)
        for line in problems:
            print(f"  - {line}", flush=True)
    else:
        print("Adopted checkpoints OK:", flush=True)
        for name, ck in (("GSE155622", CK_PAIN), ("GSE141259", CK_LUNG), ("HGSOC", CK_HG)):
            print(f"  {name}: {ck.name}", flush=True)

    if args.check:
        return 1 if problems else 0

    if args.list:
        grp = args.group or "all"
        for job in list_jobs(grp):
            extra = " [h5ad]" if job.needs_h5ad else ""
            print(f"{job.name:16s}  {job.groups}{extra}")
        return 0

    if args.only:
        wanted = set(args.only)
        jobs = [j for j in JOBS if j.name in wanted]
        unknown = wanted - {j.name for j in jobs}
        if unknown:
            print(f"Unknown jobs: {sorted(unknown)}", flush=True)
            return 2
    elif args.group:
        jobs = list_jobs(args.group)
    else:
        p.print_help()
        return 0

    if problems and args.group in {"all", "main", "supp", "fast"}:
        print("Continue anyway? Missing files will cause some jobs to fail.", flush=True)

    failed: list[str] = []
    for job in jobs:
        rc = run_job(job)
        if rc != 0:
            failed.append(job.name)
            print(f"FAILED {job.name} (exit {rc})", flush=True)

    if failed:
        print("Failed:", ", ".join(failed), flush=True)
        return 1
    print("All requested jobs finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
