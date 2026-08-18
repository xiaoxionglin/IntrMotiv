import argparse
import math
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


XBOUND = (100.0, 2000.0)
YBOUND = (100.0, 2000.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    return parser.parse_args()


def plot_run_grid(rate_maps, occupancy, si, active_fraction, title, out_path):
    n_units = rate_maps.shape[-1]
    n_cols = min(8, max(4, int(math.ceil(math.sqrt(n_units)))))
    n_rows = int(math.ceil(n_units / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.7 * n_cols, 2.55 * n_rows), dpi=180)
    axes = np.asarray(axes).reshape(-1)
    occ_mask = occupancy.T <= 0
    finite = rate_maps[np.isfinite(rate_maps)]
    vmax = np.percentile(finite, 98) if finite.size else 1.0
    vmax = max(vmax, 1e-8)
    im = None
    for i, ax in enumerate(axes):
        ax.set_axis_off()
        if i >= n_units:
            continue
        data = rate_maps[:, :, i].T.copy()
        data[occ_mask] = np.nan
        im = ax.imshow(data, origin="lower", extent=[*XBOUND, *YBOUND], cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"DG {i}  SI={si[i]:.2f}  act={active_fraction[i]:.3f}", fontsize=11)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes[:n_units], fraction=0.025, pad=0.01)
        cbar.set_label("mean DG activation")
    fig.suptitle(title, fontsize=18)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=180)
    x = np.arange(len(summary))
    labels = [
        f"F{row.Hippo_n_feature} L{row.Hippo_L} theta={row.DG_BN_intercept:g}"
        for row in summary.itertuples(index=False)
    ]
    axes[0].bar(x, summary["mean_active_fraction"])
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_ylabel("mean active fraction")
    axes[0].set_title("DG density")
    axes[1].bar(x, summary["mean_si"])
    axes[1].set_xticks(x, labels, rotation=15, ha="right")
    axes[1].set_ylabel("mean spatial information")
    axes[1].set_title("Place-field selectivity")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    data_dir = pathlib.Path(args.data_dir)
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.size"] = 14
    summary_path = data_dir / "summary.csv"
    summary = pd.read_csv(summary_path)
    for run_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        npz_path = run_dir / "place_fields.npz"
        if not npz_path.exists():
            continue
        with np.load(npz_path, allow_pickle=True) as data:
            occupancy = data["occupancy"]
            rate_maps = data["rate_maps"]
            si = data["spatial_information"]
            active_fraction = data["active_fraction"]
            n_feature = int(data["Hippo_n_feature"])
            seq_len = int(data["Hippo_L"])
            theta = float(data["DG_BN_intercept"])
            checkpoint = pathlib.Path(str(data["checkpoint"])).name
        title = f"DG place fields | F={n_feature}, L={seq_len}, theta={theta:g}, {checkpoint}"
        plot_run_grid(rate_maps, occupancy, si, active_fraction, title, run_dir / "dg_place_fields.png")
    plot_summary(summary, data_dir / "summary.png")
    print(f"Rendered DG place-field plots in {data_dir}")


if __name__ == "__main__":
    main()
