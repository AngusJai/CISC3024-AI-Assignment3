#!/usr/bin/env python3
"""Reproduce Ass3 experiments: polluted-fit reconstruction + multi-seed recognition."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.common import DEFAULT_ETA1, DEFAULT_ETA2, DEFAULT_RSMPCA_ETA2
from experiments.recognition import run_recognition
from experiments.reconstruction import run_reconstruction


def main() -> None:
    p = argparse.ArgumentParser(description="CISC3024 AI Ass3 — RSSPCA experiments")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--max-components", type=int, default=40)
    p.add_argument("--eta1", type=float, default=DEFAULT_ETA1)
    p.add_argument("--eta2", type=float, default=DEFAULT_ETA2)
    p.add_argument("--rsmpca-eta2", type=float, default=DEFAULT_RSMPCA_ETA2)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--skip-recon", action="store_true")
    p.add_argument("--skip-recog", action="store_true")
    p.add_argument("--skip-extras", action="store_true", help="skip ablation/Yale extras")
    args = p.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary: dict = {
        "params": {
            "out": str(args.out),
            "max_components": args.max_components,
            "eta1": args.eta1,
            "eta2": args.eta2,
            "rsmpca_eta2": args.rsmpca_eta2,
            "seeds": args.seeds,
        }
    }

    if not args.skip_recon:
        summary["reconstruction"] = run_reconstruction(
            out,
            max_components=args.max_components,
            eta1=args.eta1,
            eta2=args.eta2,
            rsmpca_eta2=args.rsmpca_eta2,
        )
    if not args.skip_recog:
        summary["recognition"] = run_recognition(
            out,
            max_components=args.max_components,
            seeds=args.seeds,
            eta1=args.eta1,
            eta2=args.eta2,
            rsmpca_eta2=args.rsmpca_eta2,
        )
    if not args.skip_extras:
        from experiments.ablation import (
            run_eigenface_compare,
            run_eta_ablation,
            run_rel_error_curves,
        )
        from experiments.yale_eval import run_yale

        run_rel_error_curves(out)
        run_eigenface_compare(out)
        summary["eta_ablation"] = run_eta_ablation(out)
        try:
            summary["yale"] = run_yale(out, seeds=args.seeds)
        except Exception as e:
            summary["yale_error"] = str(e)
            print(f"[warn] Yale extras skipped: {e}")

    summary["elapsed_sec"] = round(time.time() - t0, 2)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Done in {summary['elapsed_sec']}s → {out / 'summary.json'}")


if __name__ == "__main__":
    main()
