"""Second dataset: sklearn LFW subset (people with enough images).

Used when classic Yale mirrors are unavailable. Still a standard face-recognition
benchmark distinct from ORL.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_lfw_people

from .olivetti import FaceDataset


def load_lfw_subset(
    min_faces: int = 20,
    resize: tuple[int, int] = (56, 46),
    max_subjects: int = 20,
    random_state: int = 0,
) -> FaceDataset:
    """Load a compact LFW subset, resized to paper-like HxW."""
    bunch = fetch_lfw_people(
        min_faces_per_person=min_faces,
        resize=0.4,
        color=False,
        download_if_missing=True,
    )
    # Restrict to first max_subjects people (by frequency)
    y = bunch.target
    counts = {c: int(np.sum(y == c)) for c in np.unique(y)}
    top = sorted(counts, key=counts.get, reverse=True)[:max_subjects]
    mask = np.isin(y, top)
    images = bunch.images[mask]
    labels_raw = y[mask]
    # remap labels to 0..K-1
    remap = {old: i for i, old in enumerate(sorted(top))}
    labels = np.array([remap[v] for v in labels_raw], dtype=int)

    # resize each image to target
    out = []
    for im in images:
        pil = Image.fromarray((im * 255).astype(np.uint8))
        pil = pil.resize((resize[1], resize[0]), Image.BILINEAR)
        out.append(np.asarray(pil, dtype=np.float64) / 255.0)
    images_np = np.stack(out, axis=0)
    n, h, w = images_np.shape
    x = images_np.reshape(n, h * w).T
    return FaceDataset(
        images=images_np,
        X=x,
        y=labels,
        height=h,
        width=w,
        name="lfw_subset",
    )
