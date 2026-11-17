"""Face reconstruction experiments: clean + occluded Olivetti."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from data.olivetti import center_columns, load_olivetti, train_test_split_per_class
from experiments.common import (
    METHODS,
    add_block_occlusion,
    fit_method,
    get_cached_laplacian,
    psnr,
    reconstruct,
    relative_error,
    save_eigenface_grid,
    save_recon_gallery,
)


def run_reconstruction(
    out_dir: Path,
    n_components_list: list[int] | None = None,
    max_components: int = 40,
    random_state: int = 0,
    eta1: float = 0.001,
    eta2: float = 0.1,
    rspca_lam: float = 0.05,
) -> dict:
    if n_components_list is None:
        n_components_list = [5, 10, 20, 30, 40]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_olivetti(random_state=random_state)
    train_idx, test_idx = train_test_split_per_class(
        ds.y, n_train_per_class=7, random_state=random_state
    )
    x_train_raw = ds.X[:, train_idx]
    x_test_raw = ds.X[:, test_idx]
    x_train, mean = center_columns(x_train_raw)
    x_test = x_test_raw - mean

    # Occluded test images (in image space) then flatten
    test_imgs = ds.images[test_idx]
    occ_imgs = add_block_occlusion(test_imgs, block=16, random_state=random_state)
    x_occ_raw = occ_imgs.reshape(len(test_idx), -1).T
    x_occ = x_occ_raw - mean

    lap = get_cached_laplacian(ds.height, ds.width)

    results = {
        "dataset": "olivetti",
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "height": ds.height,
        "width": ds.width,
        "params": {"eta1": eta1, "eta2": eta2, "rspca_lam": rspca_lam},
        "clean": {m: [] for m in METHODS},
        "occluded": {m: [] for m in METHODS},
    }

    # Fit once at max_components, slice for curves
    print(f"[recon] fitting methods with r={max_components} ...")
    ws = {}
    for m in tqdm(METHODS, desc="fit"):
        ws[m] = fit_method(
            m,
            x_train,
            max_components,
            ds.height,
            ds.width,
            rspca_lam=rspca_lam,
            eta1=eta1,
            eta2=eta2,
            laplacian=lap,
        )
        save_eigenface_grid(
            ws[m],
            ds.height,
            ds.width,
            out_dir / f"eigenfaces_{m}.png",
            title=f"{m.upper()} components",
            n_show=16,
        )

    gallery_recons_clean = {}
    gallery_recons_occ = {}
    r_gallery = 30

    for r in n_components_list:
        for m in METHODS:
            w = ws[m][:, :r]
            # clean reconstruction of test (centered space + mean)
            x_hat = reconstruct(w, x_test, mean)
            re_c = relative_error(x_test_raw, x_hat)
            ps_c = psnr(x_test_raw, x_hat)
            results["clean"][m].append({"r": r, "rel_error": re_c, "psnr": ps_c})

            x_hat_o = reconstruct(w, x_occ, mean)
            # Reconstruct from occluded observations; score against clean ground truth
            re_o_gt = relative_error(x_test_raw, x_hat_o)
            ps_o = psnr(x_test_raw, x_hat_o)
            results["occluded"][m].append(
                {
                    "r": r,
                    "rel_error": re_o_gt,
                    "psnr": ps_o,
                    "note": "recon from occluded input vs clean GT",
                }
            )

            if r == r_gallery:
                gallery_recons_clean[m] = np.clip(
                    x_hat.T.reshape(-1, ds.height, ds.width), 0, 1
                )
                gallery_recons_occ[m] = np.clip(
                    x_hat_o.T.reshape(-1, ds.height, ds.width), 0, 1
                )

    # Curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for m in METHODS:
        rs = [e["r"] for e in results["clean"][m]]
        axes[0].plot(rs, [e["psnr"] for e in results["clean"][m]], marker="o", label=m)
        axes[1].plot(rs, [e["psnr"] for e in results["occluded"][m]], marker="o", label=m)
    axes[0].set_title("Clean reconstruction PSNR")
    axes[1].set_title("Occluded→clean PSNR")
    for ax in axes:
        ax.set_xlabel("# components")
        ax.set_ylabel("PSNR (dB)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "reconstruction_psnr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    save_recon_gallery(
        test_imgs,
        gallery_recons_clean,
        out_dir / "recon_gallery_clean.png",
        n_show=6,
    )
    # show occluded originals in top row for second gallery
    save_recon_gallery(
        occ_imgs,
        gallery_recons_occ,
        out_dir / "recon_gallery_occluded.png",
        n_show=6,
    )

    with open(out_dir / "reconstruction.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[recon] wrote {out_dir / 'reconstruction.json'}")
    return results
