"""Robust Sparse Smooth PCA (RSSPCA) — Wang et al., PLOS One 2025.

Update rule (paper Eq. 31–33 / Algorithm 1):
  U = diag(|w|)
  v = X sign(X^T w)
  u = (U + η1 I + η2 U L)^{-1} U v
  w <- u / ||u||_2

After each component we re-orthonormalize against previous columns so that
the deflation X <- (I - W W^T) X remains numerically stable.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .baselines import _deflate, _init_from_pca, _reorthogonalize
from .laplacian import spatial_laplacian


def fit_rsspca(
    x: np.ndarray,
    n_components: int,
    height: int,
    width: int,
    eta1: float = 0.001,
    eta2: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-4,
    laplacian: sparse.spmatrix | None = None,
) -> np.ndarray:
    """Fit RSSPCA projection matrix W (d x r).

    Parameters
    ----------
    x : array, shape (d, n)
        Mean-centered samples as columns.
    height, width : image spatial size so that d = height * width.
    eta1 : sparsity weight (η1 ≥ 0).
    eta2 : smoothness weight (η2 ≥ 0).
    """
    d, _ = x.shape
    assert d == height * width, f"d={d} != height*width={height * width}"
    if laplacian is None:
        laplacian = spatial_laplacian(height, width)
    else:
        laplacian = sparse.csr_matrix(laplacian)

    eye = sparse.eye(d, format="csr")
    w_list: list[np.ndarray] = []
    x_work = x.copy()

    for _comp in range(n_components):
        w = _init_from_pca(x_work, 0)
        if w_list:
            w = _reorthogonalize(w, w_list)
        f_prev = np.linalg.norm(x_work.T @ w, ord=1)

        for _it in range(max_iter):
            abs_w = np.maximum(np.abs(w), 1e-12)
            u_diag = sparse.diags(abs_w, format="csr")
            v = x_work @ np.sign(x_work.T @ w)

            # (U + η1 I + η2 U L) u = U v
            lhs = u_diag + eta1 * eye + eta2 * (u_diag @ laplacian)
            rhs = abs_w * v
            try:
                u_vec = spsolve(lhs.tocsc(), rhs)
            except Exception:
                u_vec = np.linalg.solve(lhs.toarray(), rhs)

            if not np.all(np.isfinite(u_vec)):
                break
            nrm = np.linalg.norm(u_vec)
            if nrm < 1e-12:
                break
            w = u_vec / nrm
            if w_list:
                w = _reorthogonalize(w, w_list)

            f_new = np.linalg.norm(x_work.T @ w, ord=1)
            if abs(f_new - f_prev) / (f_prev + 1e-12) < tol:
                break
            f_prev = f_new

        w_list.append(w.astype(np.float64))
        w_mat = np.column_stack(w_list)
        x_work = _deflate(x, w_mat)

    return np.column_stack(w_list)


def fit_rsmpca(
    x: np.ndarray,
    n_components: int,
    height: int,
    width: int,
    eta2: float = 10 ** (-0.2),
    max_iter: int = 100,
    tol: float = 1e-4,
    laplacian=None,
) -> np.ndarray:
    """Robust Smooth PCA = RSSPCA with η1=0 (paper Sec. 4.1 / Table 1)."""
    return fit_rsspca(
        x,
        n_components,
        height=height,
        width=width,
        eta1=0.0,
        eta2=eta2,
        max_iter=max_iter,
        tol=tol,
        laplacian=laplacian,
    )
