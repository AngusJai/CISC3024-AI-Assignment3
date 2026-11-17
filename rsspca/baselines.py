"""PCA, PCA-L1, and RSPCA baselines (Wang et al. RSSPCA paper, Sec. 2)."""

from __future__ import annotations

import numpy as np


def _soft_threshold(v: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)


def _init_from_pca(x: np.ndarray, i: int) -> np.ndarray:
    """i-th principal direction of current (deflated) matrix X (d x n)."""
    u, _, _ = np.linalg.svd(x, full_matrices=False)
    w = u[:, min(i, u.shape[1] - 1)].copy()
    w /= np.linalg.norm(w) + 1e-12
    return w


def _deflate(x: np.ndarray, w_mat: np.ndarray) -> np.ndarray:
    """X <- (I - W W^T) X  (assumes W columns are orthonormal)."""
    return x - w_mat @ (w_mat.T @ x)


def _reorthogonalize(w: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    """Modified Gram-Schmidt against previous unit columns, then renormalize."""
    w = w.copy()
    for b in basis:
        w -= np.dot(b, w) * b
    nrm = np.linalg.norm(w)
    if nrm < 1e-12:
        return w
    return w / nrm


def fit_pca(x: np.ndarray, n_components: int) -> np.ndarray:
    """Classical PCA projection matrix W (d x r). X is d x n, mean-centered."""
    u, _, _ = np.linalg.svd(x, full_matrices=False)
    r = min(n_components, u.shape[1])
    return u[:, :r]


def fit_pca_l1(
    x: np.ndarray,
    n_components: int,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> np.ndarray:
    """PCA-L1: max ||X^T w||_1 s.t. ||w||_2 = 1 (Kwak)."""
    w_list: list[np.ndarray] = []
    x_work = x.copy()
    for _ in range(n_components):
        w = _init_from_pca(x_work, 0)
        if w_list:
            w = _reorthogonalize(w, w_list)
        f_prev = np.linalg.norm(x_work.T @ w, ord=1)
        for _ in range(max_iter):
            v = x_work @ np.sign(x_work.T @ w)
            nrm = np.linalg.norm(v)
            if nrm < 1e-12:
                break
            w = v / nrm
            if w_list:
                w = _reorthogonalize(w, w_list)
            f_new = np.linalg.norm(x_work.T @ w, ord=1)
            if abs(f_new - f_prev) / (f_prev + 1e-12) < tol:
                break
            f_prev = f_new
        w_list.append(w)
        w_mat = np.column_stack(w_list)
        x_work = _deflate(x, w_mat)
    return np.column_stack(w_list)


def fit_rspca(
    x: np.ndarray,
    n_components: int,
    lam: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-4,
    use_soft: bool = True,
) -> np.ndarray:
    """RSPCA: L1 objective + L1 sparsity (Meng / soft-thresholding form).

    Parameters
    ----------
    lam : soft-threshold level (larger → sparser). If use_soft=False, uses
          the elementwise reweighting form with the same scalar as λ.
    """
    w_list: list[np.ndarray] = []
    x_work = x.copy()
    for _ in range(n_components):
        w = _init_from_pca(x_work, 0)
        if w_list:
            w = _reorthogonalize(w, w_list)
        f_prev = np.linalg.norm(x_work.T @ w, ord=1)
        for _it in range(max_iter):
            v = x_work @ np.sign(x_work.T @ w)
            if use_soft:
                u = _soft_threshold(v, lam)
            else:
                abs_w = np.abs(w) + 1e-12
                u = v * (abs_w / (lam + abs_w))
            nrm = np.linalg.norm(u)
            if nrm < 1e-12:
                u = v
                nrm = np.linalg.norm(u) + 1e-12
            w = u / nrm
            if w_list:
                w = _reorthogonalize(w, w_list)
            f_new = np.linalg.norm(x_work.T @ w, ord=1)
            if abs(f_new - f_prev) / (f_prev + 1e-12) < tol:
                break
            f_prev = f_new
        w_list.append(w)
        w_mat = np.column_stack(w_list)
        x_work = _deflate(x, w_mat)
    return np.column_stack(w_list)
