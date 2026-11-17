"""Face recognition: 1-NN, z-score features, multi-seed like the paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from data.olivetti import load_orl, train_test_split_per_class, zscore_columns
from experiments.common import (
    DEFAULT_ETA1,
    DEFAULT_ETA2,
    DEFAULT_RSMPCA_ETA2,
    METHODS,
    fit_method,
    get_cached_laplacian,
    nearest_neighbor_accuracy,
)


def run_recognition(
    out_dir: Path,
    n_components_list: list[int] | None = None,
    max_components: int = 40,
    seeds: list[int] | None = None,
    eta1: float = DEFAULT_ETA1,
    eta2: float = DEFAULT_ETA2,
    rsmpca_eta2: float = DEFAULT_RSMPCA_ETA2,
) -> dict:
    if n_components_list is None:
        n_components_list = [5, 10, 15, 20, 30, 40]
    if seeds is None:
        seeds = [0, 1, 2]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_orl()
    lap = get_cached_laplacian(ds.height, ds.width)

    # per_seed[seed][method] = list of {r, acc}
    per_seed: dict[int, dict[str, list]] = {}

    for seed in seeds:
        print(f"[recog] seed={seed}, fitting r={max_components} ...")
        train_idx, test_idx = train_test_split_per_class(
            ds.y, n_train_per_class=7, random_state=seed
        )
        y_train, y_test = ds.y[train_idx], ds.y[test_idx]
        x_train_z, mean, std = zscore_columns(ds.X[:, train_idx])
        x_test_z = (ds.X[:, test_idx] - mean) / std

        ws = {}
        for m in tqdm(METHODS, desc=f"fit seed{seed}"):
            ws[m] = fit_method(
                m,
                x_train_z,
                max_components,
                ds.height,
                ds.width,
                eta1=eta1,
                eta2=eta2,
                rsmpca_eta2=rsmpca_eta2,
                laplacian=lap,
            )

        per_seed[seed] = {m: [] for m in METHODS}
        for r in n_components_list:
            for m in METHODS:
                acc = nearest_neighbor_accuracy(
                    ws[m][:, :r], x_train_z, y_train, x_test_z, y_test
                )
                per_seed[seed][m].append({"r": r, "acc": acc})
                print(f"  seed{seed} {m:8s} r={r:2d}  acc={acc:.4f}")

    # Aggregate mean ± std across seeds
    agg = {m: [] for m in METHODS}
    for i, r in enumerate(n_components_list):
        for m in METHODS:
            vals = [per_seed[s][m][i]["acc"] for s in seeds]
            agg[m].append(
                {
                    "r": r,
                    "acc_mean": float(np.mean(vals)),
                    "acc_std": float(np.std(vals, ddof=0)),
                    "per_seed": vals,
                }
            )

    results = {
        "dataset": "orl",
        "classifier": "1-NN Euclidean in z-scored projected space",
        "n_train_per_class": 7,
        "n_test_per_class": 3,
        "seeds": seeds,
        "params": {
            "eta1": eta1,
            "eta2": eta2,
            "rsmpca_eta2": rsmpca_eta2,
            "preprocess": "per-feature z-score (fit on train)",
            "image_size": [ds.height, ds.width],
        },
        "per_seed": {str(s): per_seed[s] for s in seeds},
        "accuracy": agg,
        "summary_at_max_r": {
            m: {
                "acc_mean": agg[m][-1]["acc_mean"],
                "acc_std": agg[m][-1]["acc_std"],
            }
            for m in METHODS
        },
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for m in METHODS:
        rs = [e["r"] for e in agg[m]]
        means = [e["acc_mean"] for e in agg[m]]
        stds = [e["acc_std"] for e in agg[m]]
        ax.errorbar(rs, means, yerr=stds, marker="o", capsize=3, label=m)
    ax.set_xlabel("# components")
    ax.set_ylabel("Recognition accuracy (mean ± std)")
    ax.set_title(f"ORL face recognition (1-NN, {len(seeds)} seeds, z-score)")
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "recognition_accuracy_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "recognition.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[recog] wrote {out_dir / 'recognition.json'}")
    return results
