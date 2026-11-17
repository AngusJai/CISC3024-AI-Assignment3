"""Face reconstruction: paper-style polluted training + clean/occlusion probes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from data.olivetti import load_orl, zscore_columns
from experiments.common import (
    DEFAULT_ETA1,
    DEFAULT_ETA2,
    DEFAULT_RSMPCA_ETA2,
    METHODS,
    add_block_occlusion,
    add_salt_pepper,
    fit_method,
    get_cached_laplacian,
    psnr,
    reconstruct_z,
    relative_error,
    save_eigenface_grid,
    save_recon_gallery,
)


def run_reconstruction(
    out_dir: Path,
    n_components_list: list[int] | None = None,
    max_components: int = 40,
    random_state: int = 0,
    eta1: float = DEFAULT_ETA1,
    eta2: float = DEFAULT_ETA2,
    rsmpca_eta2: float = DEFAULT_RSMPCA_ETA2,
    n_pollute: int = 80,
) -> dict:
    """Paper-aligned reconstruction:

    Fit on a *polluted* ORL set (320 clean + 80 contaminated), then score
    reconstruction of *clean* ground-truth images (PSNR / relative error).
    """
    if n_components_list is None:
        n_components_list = [5, 10, 20, 30, 40]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_orl(random_state=random_state)
    rng = np.random.RandomState(random_state)
    pollute_idx = rng.choice(len(ds.y), size=n_pollute, replace=False)

    # Contaminate a subset (block occlusion), keep clean GT for scoring
    polluted_imgs = add_block_occlusion(
        ds.images, block=14, random_state=random_state, indices=pollute_idx
    )
    x_poll_raw = polluted_imgs.reshape(len(ds.y), -1).T
    x_clean_raw = ds.X

    x_poll_z, mean, std = zscore_columns(x_poll_raw)
    # For scoring we also need clean images in the same z-space for fitting comparison,
    # but reconstruction quality is measured in pixel space vs clean GT.
    lap = get_cached_laplacian(ds.height, ds.width)

    results = {
        "dataset": "orl",
        "protocol": "polluted_train_320clean_80block",
        "n_images": int(len(ds.y)),
        "n_pollute": int(n_pollute),
        "height": ds.height,
        "width": ds.width,
        "params": {
            "eta1": eta1,
            "eta2": eta2,
            "rsmpca_eta2": rsmpca_eta2,
            "note": "Wang et al. 2025 Table 1 reconstruction-optimal logs",
        },
        "polluted_fit": {m: [] for m in METHODS},
    }

    print(f"[recon] fitting on polluted ORL (r={max_components}) ...")
    ws = {}
    for m in tqdm(METHODS, desc="fit"):
        ws[m] = fit_method(
            m,
            x_poll_z,
            max_components,
            ds.height,
            ds.width,
            eta1=eta1,
            eta2=eta2,
            rsmpca_eta2=rsmpca_eta2,
            laplacian=lap,
        )
        save_eigenface_grid(
            ws[m],
            ds.height,
            ds.width,
            out_dir / f"eigenfaces_{m}.png",
            title=f"{m.upper()} (polluted-fit)",
            n_show=12,
        )

    # Score: project *polluted* observations, reconstruct, compare to *clean* GT
    x_poll_for_proj = x_poll_z
    gallery_recons = {}
    r_gallery = 30
    dirty_show = polluted_imgs[pollute_idx[:6]]

    for r in n_components_list:
        for m in METHODS:
            w = ws[m][:, :r]
            x_hat = reconstruct_z(w, x_poll_for_proj, mean, std)
            re = relative_error(x_clean_raw, x_hat)
            ps = psnr(x_clean_raw, x_hat)
            results["polluted_fit"][m].append(
                {"r": r, "rel_error": re, "psnr": ps}
            )
            if r == r_gallery:
                gallery_recons[m] = np.clip(
                    x_hat[:, pollute_idx[:6]].T.reshape(-1, ds.height, ds.width), 0, 1
                )

    # Curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for m in METHODS:
        rs = [e["r"] for e in results["polluted_fit"][m]]
        axes[0].plot(rs, [e["psnr"] for e in results["polluted_fit"][m]], marker="o", label=m)
        axes[1].plot(
            rs, [e["rel_error"] for e in results["polluted_fit"][m]], marker="o", label=m
        )
    axes[0].set_title("Polluted-fit recon PSNR vs clean GT")
    axes[1].set_title("Polluted-fit relative error vs clean GT")
    for ax in axes:
        ax.set_xlabel("# components")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("PSNR (dB)")
    axes[1].set_ylabel("||X-Xhat||_F / ||X||_F")
    fig.tight_layout()
    fig.savefig(out_dir / "reconstruction_psnr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    save_recon_gallery(
        dirty_show,
        gallery_recons,
        out_dir / "recon_gallery_polluted.png",
        n_show=6,
        title="Polluted inputs → recon (r=30) vs fit on polluted train",
    )

    # Extra salt-pepper pollution probe (same protocol, different contaminant)
    sp_imgs = add_salt_pepper(
        ds.images, amount=0.1, random_state=random_state, indices=pollute_idx
    )
    x_sp_raw = sp_imgs.reshape(len(ds.y), -1).T
    x_sp_z, mean_sp, std_sp = zscore_columns(x_sp_raw)
    sp_rows = {}
    r_sp = 30
    for m in METHODS:
        w = fit_method(
            m,
            x_sp_z,
            r_sp,
            ds.height,
            ds.width,
            eta1=eta1,
            eta2=eta2,
            rsmpca_eta2=rsmpca_eta2,
            laplacian=lap,
        )
        x_hat = reconstruct_z(w, x_sp_z, mean_sp, std_sp)
        sp_rows[m] = {
            "psnr": psnr(x_clean_raw, x_hat),
            "rel_error": relative_error(x_clean_raw, x_hat),
        }
        print(f"  saltpepper-fit {m}: PSNR={sp_rows[m]['psnr']:.3f}")
    results["saltpepper_fit_r30"] = sp_rows

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(sp_rows.keys()), [sp_rows[m]["psnr"] for m in METHODS])
    ax.set_ylabel("PSNR vs clean GT (dB)")
    ax.set_title("Salt-pepper polluted-fit reconstruction (r=30)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "saltpepper_psnr_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Keep a clean-gallery for qualitative (fit on polluted, show clean samples)
    clean_idx = [i for i in range(len(ds.y)) if i not in set(pollute_idx)][:6]
    clean_gallery = {}
    for m in METHODS:
        x_hat = reconstruct_z(ws[m][:, :r_gallery], x_poll_for_proj, mean, std)
        clean_gallery[m] = np.clip(
            x_hat[:, clean_idx].T.reshape(-1, ds.height, ds.width), 0, 1
        )
    save_recon_gallery(
        ds.images[clean_idx],
        clean_gallery,
        out_dir / "recon_gallery_clean.png",
        n_show=6,
        title="Clean faces reconstructed from polluted-fit subspace (r=30)",
    )

    with open(out_dir / "reconstruction.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[recon] wrote {out_dir / 'reconstruction.json'}")
    return results
