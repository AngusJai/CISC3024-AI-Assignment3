"""Shared helpers: projection, reconstruction metrics, occlusion, plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rsspca import fit_pca, fit_pca_l1, fit_rspca, fit_rsspca, fit_rsmpca
from rsspca.laplacian import spatial_laplacian

# Paper family: PCA, PCA-L1, RSPCA (η2=0), RSMPCA (η1=0), RSSPCA
METHODS = ("pca", "pca_l1", "rspca", "rsmpca", "rsspca")

# Wang et al. 2025 Table 1 (reconstruction-optimal logs)
DEFAULT_ETA1 = 0.001  # lg η1 = -3
DEFAULT_ETA2 = 1.0  # lg η2 = 0
DEFAULT_RSMPCA_ETA2 = 10 ** (-0.2)  # lg η2 = -0.2


def fit_method(
    name: str,
    x: np.ndarray,
    n_components: int,
    height: int,
    width: int,
    rspca_lam: float = 0.05,
    eta1: float = DEFAULT_ETA1,
    eta2: float = DEFAULT_ETA2,
    rsmpca_eta2: float = DEFAULT_RSMPCA_ETA2,
    laplacian=None,
    use_mm_rspca: bool = True,
) -> np.ndarray:
    if name == "pca":
        return fit_pca(x, n_components)
    if name == "pca_l1":
        return fit_pca_l1(x, n_components)
    if name == "rspca":
        # Paper: RSPCA is RSSPCA with η2=0 (MM form). Soft-threshold kept as option.
        if use_mm_rspca:
            return fit_rsspca(
                x,
                n_components,
                height=height,
                width=width,
                eta1=eta1,
                eta2=0.0,
                laplacian=laplacian,
            )
        return fit_rspca(x, n_components, lam=rspca_lam)
    if name == "rsmpca":
        return fit_rsmpca(
            x,
            n_components,
            height=height,
            width=width,
            eta2=rsmpca_eta2,
            laplacian=laplacian,
        )
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
    return w.T @ x


def reconstruct_z(
    w: np.ndarray, x_z: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Reconstruct in pixel space from z-scored columns: unzscore(W W^T X_z)."""
    x_hat_z = w @ (w.T @ x_z)
    return x_hat_z * std + mean


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
    indices: np.ndarray | None = None,
) -> np.ndarray:
    """Zero out a random block per selected image. images: (n,h,w)."""
    rng = np.random.RandomState(random_state)
    out = images.copy()
    n, h, w = out.shape
    if indices is None:
        indices = np.arange(n)
    for i in indices:
        r0 = rng.randint(0, max(1, h - block + 1))
        c0 = rng.randint(0, max(1, w - block + 1))
        out[i, r0 : r0 + block, c0 : c0 + block] = 0.0
    return out


def add_salt_pepper(
    images: np.ndarray,
    amount: float = 0.1,
    random_state: int = 0,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    rng = np.random.RandomState(random_state)
    out = images.copy()
    n = out.shape[0]
    if indices is None:
        indices = np.arange(n)
    for i in indices:
        mask = rng.rand(*out[i].shape)
        out[i][mask < amount / 2] = 0.0
        out[i][mask > 1.0 - amount / 2] = 1.0
    return out


def nearest_neighbor_accuracy(
    w: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    z_tr = project(w, x_train).T
    z_te = project(w, x_test).T
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
    title: str = "Reconstruction gallery",
) -> None:
    methods = list(recons.keys())
    n_show = min(n_show, originals.shape[0])
    nrows = 1 + len(methods)
    fig, axes = plt.subplots(nrows, n_show, figsize=(n_show * 1.5, nrows * 1.45))
    for j in range(n_show):
        axes[0, j].imshow(originals[j], cmap="gray", vmin=0, vmax=1)
        axes[0, j].axis("off")
        if j == 0:
            axes[0, j].set_ylabel("input", fontsize=8)
        for mi, m in enumerate(methods):
            axes[mi + 1, j].imshow(
                np.clip(recons[m][j], 0, 1), cmap="gray", vmin=0, vmax=1
            )
            axes[mi + 1, j].axis("off")
            if j == 0:
                axes[mi + 1, j].set_ylabel(m, fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def get_cached_laplacian(height: int, width: int):
    return spatial_laplacian(height, width)
