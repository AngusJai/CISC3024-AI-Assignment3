"""2D spatial graph Laplacian for image pixels (RSSPCA smooth constraint)."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def spatial_laplacian(height: int, width: int) -> sparse.csr_matrix:
    """4-neighbour grid Laplacian L = D - A for a height x width image.

    Pixel i (row-major) connects to up/down/left/right neighbours with A_ij=1.
    Then w^T L w = 0.5 * sum_{i,j} A_ij (w_i - w_j)^2 encourages smoothness.
    """
    d = height * width
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def idx(r: int, c: int) -> int:
        return r * width + c

    for r in range(height):
        for c in range(width):
            i = idx(r, c)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    j = idx(rr, cc)
                    rows.append(i)
                    cols.append(j)
                    data.append(1.0)

    a = sparse.coo_matrix((data, (rows, cols)), shape=(d, d)).tocsr()
    deg = np.asarray(a.sum(axis=1)).ravel()
    d_mat = sparse.diags(deg, format="csr")
    return (d_mat - a).tocsr()
