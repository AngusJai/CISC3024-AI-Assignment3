"""Classic Yale Face Database loader (15 subjects × 11 expressions)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .olivetti import FaceDataset

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
YALE_DIR = CACHE / "yale"

EXPRESSIONS = [
    "centerlight",
    "glasses",
    "happy",
    "leftlight",
    "noglasses",
    "normal",
    "rightlight",
    "sad",
    "sleepy",
    "surprised",
    "wink",
]


def _find_yale_root() -> Path | None:
    if any(YALE_DIR.glob("subject01*")):
        return YALE_DIR
    for p in CACHE.rglob("subject01.normal.gif"):
        return p.parent
    for p in CACHE.rglob("subject01.happy.gif"):
        return p.parent
    return None


def ensure_yale() -> Path:
    root = _find_yale_root()
    if root is not None:
        return root
    raise FileNotFoundError(
        "Classic Yale faces not found. Expected subjectXX.*.gif under data/cache/yale/. "
        "Populate from a public Yale mirror (e.g. cosmos-sajal/Face-Recognition)."
    )


def load_yale(resize: tuple[int, int] = (56, 46)) -> FaceDataset:
    """Load classic Yale faces → same interface as ORL."""
    root = ensure_yale()
    images = []
    labels = []
    for s in range(1, 16):
        for expr in EXPRESSIONS:
            candidates = [
                root / f"subject{s:02d}.{expr}.gif",
                root / f"subject{s:02d}.{expr}",
            ]
            # some mirrors rename centerlight → subjectXX.gif
            if expr == "centerlight":
                candidates.append(root / f"subject{s:02d}.gif")
            path = next((c for c in candidates if c.exists()), None)
            if path is None:
                matches = list(root.glob(f"subject{s:02d}.{expr}*"))
                if not matches and expr == "centerlight":
                    matches = list(root.glob(f"subject{s:02d}.gif"))
                if not matches:
                    continue
                path = matches[0]
            im = Image.open(path).convert("L")
            im = im.resize((resize[1], resize[0]), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float64) / 255.0
            images.append(arr)
            labels.append(s - 1)
    if len(images) < 100:
        raise RuntimeError(f"Only loaded {len(images)} Yale images from {root}")
    images_np = np.stack(images, axis=0)
    n, h, w = images_np.shape
    x = images_np.reshape(n, h * w).T
    return FaceDataset(
        images=images_np,
        X=x,
        y=np.asarray(labels, dtype=int),
        height=h,
        width=w,
        name="yale",
    )
