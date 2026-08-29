#!/usr/bin/env python3
"""Train LDH-scRNA transport variant (lat_disp + latent + recon; landscape OFF).

Checkpoints:
  deep_temporal_benchmark_compare/checkpoints/transportOT_v2/<DATASET>/

Example:
  python deep_temporal_benchmark_compare/train_transport_priority.py --datasets GSE155622 GSE141259 HGSOC
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from transport_priority_config import (
    COMMON_OVERRIDES,
    DATASET_RECIPES,
    HERE,
    REPO_ROOT,
    TAG,
    checkpoint_dir_for,
)

PYTHON = sys.executable


def _build_cmd(dataset: str, *, smoke_epochs: int | None) -> list[str]:
    recipe = DATASET_RECIPES[dataset]
    out = checkpoint_dir_for(dataset)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON,
        str(REPO_ROOT / "run_training.py"),
        "--dataset",
        dataset,
        "--checkpoint-dir",
        str(out),
        "--checkpoint-metric",
        str(COMMON_OVERRIDES["checkpoint_metric"]),
        "--early-stop-metric",
        str(COMMON_OVERRIDES["early_stop_metric"]),
        "--early-stop-patience",
        str(COMMON_OVERRIDES["early_stop_patience"]),
        "--lambda-recon",
        str(COMMON_OVERRIDES["lambda_recon"]),
        "--lambda-latent",
        str(COMMON_OVERRIDES["lambda_latent"]),
        "--lambda-lat-disp",
        str(COMMON_OVERRIDES["lambda_lat_disp"]),
        "--lambda-kinetic",
        str(COMMON_OVERRIDES["lambda_kinetic"]),
        "--lambda-energy",
        str(COMMON_OVERRIDES["lambda_energy"]),
        "--lambda-density",
        str(COMMON_OVERRIDES["lambda_density"]),
        "--lambda-residual-balance",
        str(COMMON_OVERRIDES["lambda_residual_balance"]),
        "--checkpoint-suffix",
        TAG,
        "--latent-disp-ot-coupling",
        "--no-latent-disp-fullpop-ot",
        "--no-density-regularization",
        "--loss-normalization",
    ]
    if recipe.get("profile"):
        cmd += ["--profile", str(recipe["profile"])]
    if recipe.get("val_mode") and recipe["val_mode"] != "patients":
        cmd += ["--val-mode", str(recipe["val_mode"])]
    if smoke_epochs is not None:
        cmd += ["--smoke-test", "--epochs", str(smoke_epochs)]
    elif recipe.get("epochs") is not None:
        cmd += ["--epochs", str(recipe["epochs"])]
    # recipe.extra_cli may duplicate flags already set; keep only profile-specific extras
    return cmd


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["GSE155622", "GSE141259", "HGSOC"],
        choices=sorted(DATASET_RECIPES.keys()),
    )
    p.add_argument("--smoke-epochs", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    print(f"[transport-train] tag={TAG}", flush=True)
    print(f"[transport-train] overrides={COMMON_OVERRIDES}", flush=True)
    for ds in args.datasets:
        cmd = _build_cmd(ds, smoke_epochs=args.smoke_epochs)
        print(f"\n[transport-train] {ds}", flush=True)
        print(" ", " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        ckpt = checkpoint_dir_for(ds) / "best_model.pth"
        if not ckpt.is_file():
            raise SystemExit(f"Missing {ckpt} after training {ds}")
        print(f"[transport-train] wrote {ckpt}", flush=True)

    manifest = HERE / "checkpoints" / TAG / "MANIFEST.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"tag={TAG}", f"overrides={COMMON_OVERRIDES}", ""]
    for ds in args.datasets:
        lines.append(f"{ds}={checkpoint_dir_for(ds)}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[transport-train] manifest {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
