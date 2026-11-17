#!/usr/bin/env python3
"""Download AT&T / ORL faces into data/cache/ if missing."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
ZIP_PATH = CACHE / "att_faces.zip"
ORL_DIR = CACHE / "orl"
URL = "https://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip"


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    if (ORL_DIR / "s1" / "1.pgm").exists():
        print(f"ORL already present at {ORL_DIR}")
        return
    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size < 1000:
        print(f"Downloading {URL} ...")
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(ZIP_PATH, "wb") as f:
            f.write(resp.read())
    print(f"Extracting {ZIP_PATH} → {ORL_DIR}")
    ORL_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(ORL_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
