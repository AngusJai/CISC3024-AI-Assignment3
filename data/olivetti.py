"""ORL / AT&T face database loader (Olivetti Research Lab faces).

Primary source: Cambridge att_faces.zip cached under data/cache/.
Falls back to sklearn fetch_olivetti_faces if available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
ORL_DIR = CACHE / "orl"
ATT_ZIP = CACHE / "att_faces.zip"


@dataclass
class FaceDataset:
    images: np.ndarray  # (n, h, w) float in [0, 1]
    X: np.ndarray  # (d, n) column vectors, NOT centered
    y: np.ndarray  # (n,)
    height: int
    width: int
    name: str = "orl"
    mean: np.ndarray | None = None


def _find_orl_root() -> Path | None:
    if (ORL_DIR / "s1" / "1.pgm").exists():
        return ORL_DIR
    # zip may extract with nested folder
    for p in CACHE.rglob("s1"):
        if (p / "1.pgm").exists():
            return p.parent
    return None


def _load_orl_from_disk(resize: tuple[int, int] | None = (64, 64)) -> FaceDataset:
    root = _find_orl_root()
    if root is None:
        raise FileNotFoundError(
            f"ORL faces not found under {CACHE}. Run: unzip att_faces.zip into data/cache/orl"
        )
    images = []
    labels = []
    for subj in range(1, 41):
        for img_i in range(1, 11):
            path = root / f"s{subj}" / f"{img_i}.pgm"
            im = Image.open(path).convert("L")
            if resize is not None:
                im = im.resize((resize[1], resize[0]), Image.BILINEAR)  # (W,H)
            arr = np.asarray(im, dtype=np.float64) / 255.0
            images.append(arr)
            labels.append(subj - 1)
    images_np = np.stack(images, axis=0)
    n, h, w = images_np.shape
    x = images_np.reshape(n, h * w).T
    return FaceDataset(
        images=images_np,
        X=x,
        y=np.asarray(labels, dtype=int),
        height=h,
        width=w,
        name="orl",
    )


def load_olivetti(
    shuffle: bool = False,
    random_state: int = 0,
    resize: tuple[int, int] | None = (64, 64),
) -> FaceDataset:
    """Load ORL/Olivetti-style faces (400 images, 40 subjects).

    Prefer local AT&T ORL cache; optionally fall back to sklearn.
    `resize` defaults to 64x64 for tractable RSSPCA (original ORL is 112x92).
    """
    try:
        ds = _load_orl_from_disk(resize=resize)
    except FileNotFoundError:
        from sklearn.datasets import fetch_olivetti_faces

        bunch = fetch_olivetti_faces(shuffle=False, random_state=random_state)
        images = bunch.images.astype(np.float64)
        n, h, w = images.shape
        x = images.reshape(n, h * w).T
        ds = FaceDataset(
            images=images,
            X=x,
            y=bunch.target.astype(int),
            height=h,
            width=w,
            name="olivetti_sklearn",
        )

    if shuffle:
        rng = np.random.RandomState(random_state)
        perm = rng.permutation(len(ds.y))
        ds = FaceDataset(
            images=ds.images[perm],
            X=ds.X[:, perm],
            y=ds.y[perm],
            height=ds.height,
            width=ds.width,
            name=ds.name,
        )
    return ds


def train_test_split_per_class(
    y: np.ndarray,
    n_train_per_class: int = 7,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train_idx, test_idx with fixed shots per subject (10 images each)."""
    rng = np.random.RandomState(random_state)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        train_idx.extend(idx[:n_train_per_class].tolist())
        test_idx.extend(idx[n_train_per_class:].tolist())
    return np.array(train_idx), np.array(test_idx)


def center_columns(
    x: np.ndarray, mean: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Center feature rows using mean face (d x 1)."""
    if mean is None:
        mean = x.mean(axis=1, keepdims=True)
    return x - mean, mean
