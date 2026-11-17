"""η ablation + eigenface comparison + relative-error curves."""

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
    METHODS,
    add_salt_pepper,
    fit_method,
    get_cached_laplacian,
    nearest_neighbor_accuracy,
    psnr,
    reconstruct_z,
    relative_error,
)


def run_eta_ablation(out_dir: Path, r: int = 20, seed: int = 0) -> dict:
    """Sweep η1 (fix η2=1) and η2 (fix η1=0.001) for RSSPCA recognition + recon."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_orl()
    lap = get_cached_laplacian(ds.height, ds.width)

    # recognition split
    tr, te = train_test_split_per_class(ds.y, random_state=seed)
    xtr, mean, std = zscore_columns(ds.X[:, tr])
    xte = (ds.X[:, te] - mean) / std
    ytr, yte = ds.y[tr], ds.y[te]

    # salt-pepper polluted fit matrix for recon probe
    rng = np.random.RandomState(seed)
    pollute_idx = rng.choice(len(ds.y), size=80, replace=False)
    sp_imgs = add_salt_pepper(ds.images, amount=0.1, random_state=seed, indices=pollute_idx)
    x_sp_z, mean_sp, std_sp = zscore_columns(sp_imgs.reshape(len(ds.y), -1).T)

    eta1_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    eta2_grid = [1e-2, 1e-1, 1.0, 10.0, 100.0]

    sweep_eta1 = []
    print("[ablation] sweep η1 (η2=1) ...")
    for e1 in tqdm(eta1_grid):
        w = fit_method(
            "rsspca", xtr, r, ds.height, ds.width, eta1=e1, eta2=1.0, laplacian=lap
        )
        acc = nearest_neighbor_accuracy(w, xtr, ytr, xte, yte)
        w_rec = fit_method(
            "rsspca", x_sp_z, r, ds.height, ds.width, eta1=e1, eta2=1.0, laplacian=lap
        )
        xhat = reconstruct_z(w_rec, x_sp_z, mean_sp, std_sp)
        sweep_eta1.append(
            {
                "eta1": e1,
                "lg_eta1": float(np.log10(e1)),
                "acc": acc,
                "psnr": psnr(ds.X, xhat),
                "rel_error": relative_error(ds.X, xhat),
            }
        )

    sweep_eta2 = []
    print("[ablation] sweep η2 (η1=0.001) ...")
    for e2 in tqdm(eta2_grid):
        w = fit_method(
            "rsspca", xtr, r, ds.height, ds.width, eta1=0.001, eta2=e2, laplacian=lap
        )
        acc = nearest_neighbor_accuracy(w, xtr, ytr, xte, yte)
        w_rec = fit_method(
            "rsspca", x_sp_z, r, ds.height, ds.width, eta1=0.001, eta2=e2, laplacian=lap
        )
        xhat = reconstruct_z(w_rec, x_sp_z, mean_sp, std_sp)
        sweep_eta2.append(
            {
                "eta2": e2,
                "lg_eta2": float(np.log10(e2)),
                "acc": acc,
                "psnr": psnr(ds.X, xhat),
                "rel_error": relative_error(ds.X, xhat),
            }
        )

    results = {"r": r, "seed": seed, "sweep_eta1": sweep_eta1, "sweep_eta2": sweep_eta2}
    Path(out_dir / "eta_ablation.json").write_text(json.dumps(results, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7))
    x1 = [e["lg_eta1"] for e in sweep_eta1]
    axes[0, 0].plot(x1, [e["acc"] for e in sweep_eta1], "o-")
    axes[0, 0].axvline(np.log10(DEFAULT_ETA1), color="gray", ls="--", label="paper η1")
    axes[0, 0].set_xlabel(r"$\lg\,\eta_1$ (fixed $\eta_2=1$)")
    axes[0, 0].set_ylabel("1-NN accuracy")
    axes[0, 0].set_title("Recognition vs sparsity weight")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x1, [e["psnr"] for e in sweep_eta1], "o-", color="C1")
    axes[0, 1].axvline(np.log10(DEFAULT_ETA1), color="gray", ls="--")
    axes[0, 1].set_xlabel(r"$\lg\,\eta_1$ (fixed $\eta_2=1$)")
    axes[0, 1].set_ylabel("Salt-pepper recon PSNR")
    axes[0, 1].set_title("Reconstruction vs sparsity weight")
    axes[0, 1].grid(True, alpha=0.3)

    x2 = [e["lg_eta2"] for e in sweep_eta2]
    axes[1, 0].plot(x2, [e["acc"] for e in sweep_eta2], "o-", color="C2")
    axes[1, 0].axvline(np.log10(DEFAULT_ETA2), color="gray", ls="--", label="paper η2")
    axes[1, 0].set_xlabel(r"$\lg\,\eta_2$ (fixed $\eta_1=0.001$)")
    axes[1, 0].set_ylabel("1-NN accuracy")
    axes[1, 0].set_title("Recognition vs smoothness weight")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(x2, [e["psnr"] for e in sweep_eta2], "o-", color="C3")
    axes[1, 1].axvline(np.log10(DEFAULT_ETA2), color="gray", ls="--")
    axes[1, 1].set_xlabel(r"$\lg\,\eta_2$ (fixed $\eta_1=0.001$)")
    axes[1, 1].set_ylabel("Salt-pepper recon PSNR")
    axes[1, 1].set_title("Reconstruction vs smoothness weight")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"RSSPCA η ablation on ORL (r={r})")
    fig.tight_layout()
    fig.savefig(out_dir / "eta_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ablation] wrote {out_dir / 'eta_ablation.png'}")
    return results


def run_eigenface_compare(out_dir: Path, n_components: int = 12, seed: int = 0) -> None:
    """Side-by-side PCA / RSPCA / RSMPCA / RSSPCA first components."""
    out_dir = Path(out_dir)
    ds = load_orl()
    lap = get_cached_laplacian(ds.height, ds.width)
    rng = np.random.RandomState(seed)
    pollute_idx = rng.choice(len(ds.y), size=80, replace=False)
    sp_imgs = add_salt_pepper(ds.images, amount=0.1, random_state=seed, indices=pollute_idx)
    x_z, _, _ = zscore_columns(sp_imgs.reshape(len(ds.y), -1).T)

    methods = ["pca", "rspca", "rsmpca", "rsspca"]
    ws = {
        m: fit_method(m, x_z, n_components, ds.height, ds.width, laplacian=lap)
        for m in methods
    }

    n_show = 8
    fig, axes = plt.subplots(len(methods), n_show, figsize=(n_show * 1.35, len(methods) * 1.55))
    for mi, m in enumerate(methods):
        for j in range(n_show):
            ax = axes[mi, j]
            img = ws[m][:, j].reshape(ds.height, ds.width)
            vmax = np.max(np.abs(img)) + 1e-12
            ax.imshow(img, cmap="gray", vmin=-vmax, vmax=vmax)
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(m.upper(), fontsize=9)
            if mi == 0:
                ax.set_title(f"PC{j+1}", fontsize=8)
    fig.suptitle("Polluted-fit components: PCA / RSPCA / RSMPCA / RSSPCA")
    fig.tight_layout()
    fig.savefig(out_dir / "eigenfaces_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ablation] wrote {out_dir / 'eigenfaces_compare.png'}")


def run_rel_error_curves(out_dir: Path) -> None:
    """Paper-style relative error curves from existing reconstruction.json."""
    out_dir = Path(out_dir)
    path = out_dir / "reconstruction.json"
    if not path.exists():
        print("[ablation] skip rel-error curves: no reconstruction.json")
        return
    data = json.loads(path.read_text())
    block = data.get("polluted_fit", {})
    if not block:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in METHODS:
        if m not in block:
            continue
        rs = [e["r"] for e in block[m]]
        errs = [e["rel_error"] for e in block[m]]
        ax.plot(rs, errs, marker="o", label=m)
    ax.set_xlabel("# components")
    ax.set_ylabel(r"Relative error $\|X-\hat X\|_F/\|X\|_F$")
    ax.set_title("Block polluted-fit reconstruction error (paper-style)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "reconstruction_rel_error_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ablation] wrote {out_dir / 'reconstruction_rel_error_curves.png'}")


def main() -> None:
    out = Path("outputs")
    run_rel_error_curves(out)
    run_eigenface_compare(out)
    run_eta_ablation(out)


if __name__ == "__main__":
    main()
