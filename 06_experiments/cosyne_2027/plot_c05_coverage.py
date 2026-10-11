"""Render paired five-seed C05 factorial coverage from canonical scalars."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scalar_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = pd.concat([pd.read_csv(args.scalar_root / cohort / "per_run.csv")
                      for cohort in ("c05_pushpull_s3", "c05_pushpull_s2")])
    data["encoder_credit"] = data.encoder_credit.str.lower()
    data["worker_bonus"] = data.worker_bonus.str.lower()
    if data.duplicated(["encoder_credit", "worker_bonus", "seed"]).any() or len(data) != 20:
        raise ValueError("Expected complete five-seed C05 factorial")
    matrix = data.pivot(index="seed", columns=["encoder_credit", "worker_bonus"], values="coverage_auc")
    factor_table = pd.DataFrame({
        "seed": matrix.index,
        "constant_encoder_constant_worker": matrix["constant", "constant"].to_numpy(),
        "constant_encoder_temporal_worker": matrix["constant", "temporal"].to_numpy(),
        "temporal_encoder_constant_worker": matrix["temporal", "constant"].to_numpy(),
        "temporal_encoder_temporal_worker": matrix["temporal", "temporal"].to_numpy(),
    })
    factor_table["encoder_by_worker_interaction"] = (
        factor_table.temporal_encoder_temporal_worker
        - factor_table.constant_encoder_temporal_worker
        - factor_table.temporal_encoder_constant_worker
        + factor_table.constant_encoder_constant_worker
    )
    factor_table.to_csv(args.output.with_name("c05_coverage_factorial_seed.csv"), index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 17,
                         "axes.labelsize": 18, "axes.titlesize": 19,
                         "xtick.labelsize": 16, "ytick.labelsize": 16,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True, layout="constrained")
    for ax, encoder in zip(axes, ("temporal", "constant")):
        table = data[data.encoder_credit == encoder].pivot(
            index="seed", columns="worker_bonus", values="coverage_auc"
        )[["temporal", "constant"]].sort_index()
        values = table.to_numpy(dtype=float)
        for row in values:
            ax.plot([0, 1], row, color="#9b9b9b", lw=1, zorder=1)
            ax.scatter([0, 1], row, color="#555555", s=30, zorder=2)
        means = values.mean(axis=0)
        sem = values.std(axis=0, ddof=1) / np.sqrt(len(values))
        ax.errorbar([0, 1], means, yerr=sem, color="#0072B2" if encoder == "temporal" else "#D55E00",
                    fmt="o-", markersize=10, lw=2.6, capsize=5, zorder=3)
        ax.set_title(f"{encoder.capitalize()} encoder credit")
        ax.set_xticks([0, 1], ["Temporal\nworker bonus", "Constant\nworker bonus"])
        ax.set_xlim(-.3, 1.3)
        ax.set_ylim(0, 110)
        ax.grid(axis="y", color="#dddddd", lw=.6)
    axes[0].set_ylabel("Coverage AUC, final 90–100M frames")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
