#!/usr/bin/env python3
"""Download classic Yale faces into data/cache/yale/ via a GitHub mirror."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
YALE = CACHE / "yale"
URL = "https://codeload.github.com/cosmos-sajal/Face-Recognition/zip/refs/heads/master"


def main() -> None:
    if len(list(YALE.glob("subject*.gif"))) >= 150:
        print(f"Yale already present at {YALE}")
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    YALE.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    req = Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=120) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            if "/yalefaces/yalefaces/subject" in name and name.endswith(".gif"):
                out = YALE / Path(name).name
                out.write_bytes(zf.read(name))
    n = len(list(YALE.glob("subject*.gif")))
    print(f"Extracted {n} Yale GIFs → {YALE}")


if __name__ == "__main__":
    main()
