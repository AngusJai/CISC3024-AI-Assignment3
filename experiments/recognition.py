"""Face recognition: 1-NN in projected subspace vs #components."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from data.olivetti import center_columns, load_olivetti, train_test_split_per_class
from experiments.common import (
    METHODS,
    fit_method,
    get_cached_laplacian,
    nearest_neighbor_accuracy,
)


def run_recognition(
    out_dir: Path,
    n_components_list: list[int] | None = None,
    max_components: int = 40,
    random_state: int = 0,
    eta1: float = 0.001,
    eta2: float = 0.1,
    rspca_lam: float = 0.05,
) -> dict:
    if n_components_list is None:
        n_components_list = [5, 10, 15, 20, 30, 40]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_olivetti(random_state=random_state)
    train_idx, test_idx = train_test_split_per_class(
        ds.y, n_train_per_class=7, random_state=random_state
    )
    y_train, y_test = ds.y[train_idx], ds.y[test_idx]
    x_train, mean = center_columns(ds.X[:, train_idx])
    x_test = ds.X[:, test_idx] - mean

    lap = get_cached_laplacian(ds.height, ds.width)

    print(f"[recog] fitting methods with r={max_components} ...")
    ws = {}
    for m in tqdm(METHODS, desc="fit"):
        ws[m] = fit_method(
            m,
            x_train,
            max_components,
            ds.height,
            ds.width,
            rspca_lam=rspca_lam,
            eta1=eta1,
            eta2=eta2,
            laplacian=lap,
        )

    results = {
        "dataset": "olivetti",
        "classifier": "1-NN Euclidean in projected space",
        "n_train_per_class": 7,
        "n_test_per_class": 3,
        "params": {"eta1": eta1, "eta2": eta2, "rspca_lam": rspca_lam},
        "accuracy": {m: [] for m in METHODS},
    }

    for r in n_components_list:
        for m in METHODS:
            acc = nearest_neighbor_accuracy(
                ws[m][:, :r], x_train, y_train, x_test, y_test
            )
            results["accuracy"][m].append({"r": r, "acc": acc})
            print(f"  {m:8s} r={r:2d}  acc={acc:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in METHODS:
        rs = [e["r"] for e in results["accuracy"][m]]
        accs = [e["acc"] for e in results["accuracy"][m]]
        ax.plot(rs, accs, marker="o", label=m)
    ax.set_xlabel("# components")
    ax.set_ylabel("Recognition accuracy")
    ax.set_title("Olivetti face recognition (1-NN)")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "recognition_accuracy_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # summary table at r=40
    summary = {
        m: results["accuracy"][m][-1]["acc"] for m in METHODS if results["accuracy"][m]
    }
    results["summary_at_max_r"] = summary

    with open(out_dir / "recognition.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[recog] wrote {out_dir / 'recognition.json'}")
    return results
