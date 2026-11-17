"""AT&T / ORL face database loader (Olivetti Research Lab).

Primary source: Cambridge att_faces.zip under data/cache/.
Paper (Wang et al. 2025) resizes ORL to 56x46; we follow that by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
ORL_DIR = CACHE / "orl"


@dataclass
class FaceDataset:
    images: np.ndarray  # (n, h, w) float in [0, 1]
    X: np.ndarray  # (d, n) column vectors, NOT centered
    y: np.ndarray  # (n,)
    height: int
    width: int
    name: str = "orl"


def _find_orl_root() -> Path | None:
    if (ORL_DIR / "s1" / "1.pgm").exists():
        return ORL_DIR
    for p in CACHE.rglob("s1"):
        if (p / "1.pgm").exists():
            return p.parent
    return None


def _load_orl_from_disk(resize: tuple[int, int] | None = (56, 46)) -> FaceDataset:
    root = _find_orl_root()
    if root is None:
        raise FileNotFoundError(
            f"ORL faces not found under {CACHE}. Run: python scripts/ensure_orl.py"
        )
    images = []
    labels = []
    for subj in range(1, 41):
        for img_i in range(1, 11):
            path = root / f"s{subj}" / f"{img_i}.pgm"
            im = Image.open(path).convert("L")
            if resize is not None:
                # PIL resize is (W, H); our resize arg is (H, W)
                im = im.resize((resize[1], resize[0]), Image.BILINEAR)
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


def load_orl(
    shuffle: bool = False,
    random_state: int = 0,
    resize: tuple[int, int] | None = (56, 46),
) -> FaceDataset:
    """Load AT&T ORL faces (400 images, 40 subjects). Default size matches the paper."""
    ds = _load_orl_from_disk(resize=resize)
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


# Back-compat alias used by older scripts
def load_olivetti(*args, **kwargs) -> FaceDataset:
    return load_orl(*args, **kwargs)


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


def zscore_columns(
    x: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature z-score (paper protocol). X is d x n."""
    if mean is None:
        mean = x.mean(axis=1, keepdims=True)
    if std is None:
        std = x.std(axis=1, keepdims=True)
    std_safe = np.maximum(std, eps)
    return (x - mean) / std_safe, mean, std_safe


def unzscore(x_z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x_z * std + mean


def center_columns(
    x: np.ndarray, mean: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-center only (legacy helper)."""
    if mean is None:
        mean = x.mean(axis=1, keepdims=True)
    return x - mean, mean
