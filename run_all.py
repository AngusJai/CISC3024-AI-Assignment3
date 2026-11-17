#!/usr/bin/env python3
"""Reproduce Ass3 experiments: reconstruction + recognition on Olivetti faces."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.recognition import run_recognition
from experiments.reconstruction import run_reconstruction


def main() -> None:
    p = argparse.ArgumentParser(description="CISC3024 AI Ass3 — RSSPCA experiments")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--max-components", type=int, default=40)
    p.add_argument("--eta1", type=float, default=0.001)
    p.add_argument("--eta2", type=float, default=0.1)
    p.add_argument("--rspca-lam", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-recon", action="store_true")
    p.add_argument("--skip-recog", action="store_true")
    args = p.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary: dict = {"params": vars(args)}
    # Path is not JSON-serializable
    summary["params"]["out"] = str(args.out)

    if not args.skip_recon:
        summary["reconstruction"] = run_reconstruction(
            out,
            max_components=args.max_components,
            random_state=args.seed,
            eta1=args.eta1,
            eta2=args.eta2,
            rspca_lam=args.rspca_lam,
        )
    if not args.skip_recog:
        summary["recognition"] = run_recognition(
            out,
            max_components=args.max_components,
            random_state=args.seed,
            eta1=args.eta1,
            eta2=args.eta2,
            rspca_lam=args.rspca_lam,
        )

    summary["elapsed_sec"] = round(time.time() - t0, 2)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Done in {summary['elapsed_sec']}s → {out / 'summary.json'}")


if __name__ == "__main__":
    main()
