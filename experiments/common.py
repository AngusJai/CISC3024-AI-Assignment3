"""Shared helpers: projection, reconstruction metrics, occlusion, plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rsspca import fit_pca, fit_pca_l1, fit_rspca, fit_rsspca
from rsspca.laplacian import spatial_laplacian


METHODS = ("pca", "pca_l1", "rspca", "rsspca")


def fit_method(
    name: str,
    x: np.ndarray,
    n_components: int,
    height: int,
    width: int,
    rspca_lam: float = 0.05,
    eta1: float = 0.001,
    eta2: float = 0.1,
    laplacian=None,
) -> np.ndarray:
    if name == "pca":
        return fit_pca(x, n_components)
    if name == "pca_l1":
        return fit_pca_l1(x, n_components)
    if name == "rspca":
        return fit_rspca(x, n_components, lam=rspca_lam)
    if name == "rsspca":
        return fit_rsspca(
            x,
            n_components,
            height=height,
            width=width,
            eta1=eta1,
            eta2=eta2,
            laplacian=laplacian,
        )
    raise ValueError(f"unknown method: {name}")


def project(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Y = W^T X  → (r, n)"""
    return w.T @ x


def reconstruct(w: np.ndarray, x: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """X_hat = W W^T X + mean  (mean shape d x 1)"""
    return w @ (w.T @ x) + mean


def relative_error(x_true: np.ndarray, x_hat: np.ndarray) -> float:
    num = np.linalg.norm(x_true - x_hat, ord="fro")
    den = np.linalg.norm(x_true, ord="fro") + 1e-12
    return float(num / den)


def psnr(x_true: np.ndarray, x_hat: np.ndarray, data_range: float = 1.0) -> float:
    mse = np.mean((x_true - x_hat) ** 2)
    if mse < 1e-12:
        return 99.0
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def add_block_occlusion(
    images: np.ndarray,
    block: int = 16,
    random_state: int = 0,
) -> np.ndarray:
    """Zero out a random block_size x block_size region per image. images: (n,h,w)."""
    rng = np.random.RandomState(random_state)
    out = images.copy()
    n, h, w = out.shape
    for i in range(n):
        r0 = rng.randint(0, max(1, h - block + 1))
        c0 = rng.randint(0, max(1, w - block + 1))
        out[i, r0 : r0 + block, c0 : c0 + block] = 0.0
    return out


def nearest_neighbor_accuracy(
    w: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """1-NN in projected space (cosine / Euclidean)."""
    z_tr = project(w, x_train).T  # (n_tr, r)
    z_te = project(w, x_test).T
    # Euclidean NN
    # (n_te, n_tr) squared distances
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    a2 = np.sum(z_te**2, axis=1, keepdims=True)
    b2 = np.sum(z_tr**2, axis=1, keepdims=True).T
    d2 = a2 + b2 - 2.0 * (z_te @ z_tr.T)
    nn = np.argmin(d2, axis=1)
    pred = y_train[nn]
    return float(np.mean(pred == y_test))


def save_eigenface_grid(
    w: np.ndarray,
    height: int,
    width: int,
    path: Path,
    title: str,
    n_show: int = 16,
) -> None:
    n_show = min(n_show, w.shape[1])
    cols = 4
    rows = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i < n_show:
            img = w[:, i].reshape(height, width)
            vmax = np.max(np.abs(img)) + 1e-12
            ax.imshow(img, cmap="gray", vmin=-vmax, vmax=vmax)
            ax.set_title(f"PC{i+1}", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_recon_gallery(
    originals: np.ndarray,
    recons: dict[str, np.ndarray],
    path: Path,
    n_show: int = 6,
) -> None:
    """originals/recons images as (n,h,w)."""
    methods = list(recons.keys())
    n_show = min(n_show, originals.shape[0])
    nrows = 1 + len(methods)
    fig, axes = plt.subplots(nrows, n_show, figsize=(n_show * 1.6, nrows * 1.6))
    for j in range(n_show):
        axes[0, j].imshow(originals[j], cmap="gray", vmin=0, vmax=1)
        axes[0, j].axis("off")
        if j == 0:
            axes[0, j].set_ylabel("orig", fontsize=9)
        for mi, m in enumerate(methods):
            axes[mi + 1, j].imshow(recons[m][j], cmap="gray", vmin=0, vmax=1)
            axes[mi + 1, j].axis("off")
            if j == 0:
                axes[mi + 1, j].set_ylabel(m, fontsize=8)
    fig.suptitle("Reconstruction gallery")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def get_cached_laplacian(height: int, width: int):
    return spatial_laplacian(height, width)
