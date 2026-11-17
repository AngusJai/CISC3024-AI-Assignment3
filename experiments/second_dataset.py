"""Second-dataset evaluation: prefer Yale if present, else LFW subset."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from data.olivetti import train_test_split_per_class, zscore_columns
from experiments.common import (
    METHODS,
    add_salt_pepper,
    fit_method,
    get_cached_laplacian,
    nearest_neighbor_accuracy,
    psnr,
    reconstruct_z,
    relative_error,
)


def _load_second():
    try:
        from data.yale import load_yale

        ds = load_yale()
        return ds, 8
    except Exception as e:
        print(f"[second] Yale unavailable ({e}); falling back to LFW subset")
        from data.lfw_subset import load_lfw_subset

        ds = load_lfw_subset(min_faces=30, max_subjects=15, resize=(56, 46))
        # LFW subjects vary in count — use min(7, n_per_class-3) style: 70% train
        # For simplicity: 10 train if enough else half
        counts = [int(np.sum(ds.y == c)) for c in np.unique(ds.y)]
        n_train = max(5, min(counts) - 3)
        return ds, n_train


def run_second(out_dir: Path, max_components: int = 30, seeds: list[int] | None = None) -> dict:
    if seeds is None:
        seeds = [0, 1, 2]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds, n_train = _load_second()
    print(f"[second] {ds.name}: images={ds.images.shape}, subjects={len(np.unique(ds.y))}, train/subj={n_train}")
    lap = get_cached_laplacian(ds.height, ds.width)

    n_list = [5, 10, 15, 20, 30]
    per_seed = {}
    for seed in seeds:
        tr, te = train_test_split_per_class(ds.y, n_train_per_class=n_train, random_state=seed)
        xtr, mean, std = zscore_columns(ds.X[:, tr])
        xte = (ds.X[:, te] - mean) / std
        ytr, yte = ds.y[tr], ds.y[te]
        ws = {
            m: fit_method(m, xtr, max_components, ds.height, ds.width, laplacian=lap)
            for m in tqdm(METHODS, desc=f"{ds.name} fit seed{seed}")
        }
        per_seed[seed] = {m: [] for m in METHODS}
        for r in n_list:
            for m in METHODS:
                acc = nearest_neighbor_accuracy(ws[m][:, :r], xtr, ytr, xte, yte)
                per_seed[seed][m].append({"r": r, "acc": acc})
                print(f"  {ds.name} seed{seed} {m:8s} r={r:2d} acc={acc:.4f}")

    agg = {m: [] for m in METHODS}
    for i, r in enumerate(n_list):
        for m in METHODS:
            vals = [per_seed[s][m][i]["acc"] for s in seeds]
            agg[m].append(
                {"r": r, "acc_mean": float(np.mean(vals)), "acc_std": float(np.std(vals))}
            )

    rng = np.random.RandomState(0)
    n_pollute = max(20, len(ds.y) // 5)
    pollute_idx = rng.choice(len(ds.y), size=min(n_pollute, len(ds.y)), replace=False)
    sp = add_salt_pepper(ds.images, amount=0.1, random_state=0, indices=pollute_idx)
    x_sp_z, mean_sp, std_sp = zscore_columns(sp.reshape(len(ds.y), -1).T)
    recon = {}
    r_rec = 20
    for m in METHODS:
        w = fit_method(m, x_sp_z, r_rec, ds.height, ds.width, laplacian=lap)
        xhat = reconstruct_z(w, x_sp_z, mean_sp, std_sp)
        recon[m] = {"psnr": psnr(ds.X, xhat), "rel_error": relative_error(ds.X, xhat)}

    results = {
        "dataset": ds.name,
        "n_images": int(len(ds.y)),
        "n_subjects": int(len(np.unique(ds.y))),
        "n_train_per_class": n_train,
        "seeds": seeds,
        "accuracy": agg,
        "saltpepper_fit_r20": recon,
    }
    out_json = out_dir / "second_dataset.json"
    out_json.write_text(json.dumps(results, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for m in METHODS:
        axes[0].errorbar(
            [e["r"] for e in agg[m]],
            [e["acc_mean"] for e in agg[m]],
            yerr=[e["acc_std"] for e in agg[m]],
            marker="o",
            capsize=3,
            label=m,
        )
    axes[0].set_title(f"{ds.name} recognition (1-NN, 3 seeds)")
    axes[0].set_xlabel("# components")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[1].bar(list(recon.keys()), [recon[m]["psnr"] for m in METHODS])
    axes[1].set_title(f"{ds.name} salt-pepper polluted-fit PSNR (r=20)")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "second_dataset.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # keep yale_eval alias names for report compatibility
    (out_dir / "yale_eval.json").write_text(json.dumps(results, indent=2))
    fig.savefig  # noqa
    import shutil

    shutil.copy(out_dir / "second_dataset.png", out_dir / "yale_eval.png")
    print(f"[second] wrote {out_json}")
    return results


if __name__ == "__main__":
    run_second(Path("outputs"))
