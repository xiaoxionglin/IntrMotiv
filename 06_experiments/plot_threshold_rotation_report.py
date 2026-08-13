#!/usr/bin/env python3
"""Regenerate report figures for the threshold-rotation toy experiment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ("no_learning", "encourage", "punish")
COLORS = {"no_learning": "#666666", "encourage": "#2878b5", "punish": "#c43c39"}
LABELS = {"no_learning": "no learning", "encourage": "encourage", "punish": "punish"}


def configure_matplotlib(font_size: int) -> None:
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size + 2,
            "axes.labelsize": font_size + 1,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "figure.titlesize": font_size + 2,
            "lines.linewidth": 2.4,
            "lines.markersize": 7,
        }
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def beta_suffix(beta: str) -> str:
    return beta.replace(".", "p")


def load_aggregate(base: Path, prefix: str, betas: list[str]) -> dict[str, list[dict[str, str]]]:
    return {
        beta: read_rows(base / f"{prefix}_{beta_suffix(beta)}" / "aggregate.csv")
        for beta in betas
    }


def plot_metric_grid(
    *,
    base: Path,
    out: Path,
    prefix: str,
    betas: list[str],
    metric: str,
    ylabel: str,
    filename: str,
    include_no_learning: bool = True,
) -> None:
    rows_by_beta = load_aggregate(base, prefix, betas)
    fig, axes = plt.subplots(1, len(betas), figsize=(16, 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    conditions = CONDITIONS if include_no_learning else ("encourage", "punish")

    for ax, beta in zip(axes, betas):
        rows = rows_by_beta[beta]
        for condition in conditions:
            subset = sorted(
                [row for row in rows if row["condition"] == condition],
                key=lambda row: float(row["theta"]),
            )
            x = [float(row["theta"]) for row in subset]
            y = [float(row[f"{metric}_mean"]) for row in subset]
            yerr = [float(row[f"{metric}_std"]) for row in subset]
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                capsize=3.5,
                color=COLORS[condition],
                label=LABELS[condition],
            )
        ax.set_title(rf"$\beta={beta}$")
        ax.set_xlabel(r"$\theta$")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(ylabel)
    axes[-1].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out / filename, dpi=200)
    plt.close(fig)


def plot_metric_grid_wrapped(
    *,
    base: Path,
    out: Path,
    prefix: str,
    betas: list[str],
    metric: str,
    ylabel: str,
    filename: str,
    include_no_learning: bool = True,
    ncols: int = 3,
) -> None:
    rows_by_beta = load_aggregate(base, prefix, betas)
    nrows = int(np.ceil(len(betas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.3 * nrows), sharey=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    conditions = CONDITIONS if include_no_learning else ("encourage", "punish")

    for ax, beta in zip(axes.ravel(), betas):
        rows = rows_by_beta[beta]
        for condition in conditions:
            subset = sorted(
                [row for row in rows if row["condition"] == condition],
                key=lambda row: float(row["theta"]),
            )
            x = [float(row["theta"]) for row in subset]
            y = [float(row[f"{metric}_mean"]) for row in subset]
            yerr = [float(row[f"{metric}_std"]) for row in subset]
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                capsize=3.5,
                color=COLORS[condition],
                label=LABELS[condition],
            )
        ax.set_title(rf"$\beta={beta}$")
        ax.set_xlabel(r"$\theta$")
        ax.grid(alpha=0.25)

    for ax in axes.ravel()[len(betas) :]:
        ax.axis("off")

    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    axes[0, min(ncols - 1, len(betas) - 1)].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out / filename, dpi=200)
    plt.close(fig)


def plot_iid_all_beta_histograms(base: Path, out: Path, betas: list[str], filename: str) -> None:
    ncols = 3
    nrows = int(np.ceil(len(betas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.3 * nrows), sharey=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    prefix = "threshold_rotation_toy_iid_gg_beta"
    for ax, beta in zip(axes.ravel(), betas):
        arrays = np.load(base / f"{prefix}_{beta_suffix(beta)}" / "last_run_arrays.npz")
        values = arrays["latent_values"].reshape(-1)
        ax.hist(values, bins=80, density=True, color="#555555", alpha=0.85)
        ax.set_title(rf"$\beta={beta}$")
        ax.set_xlabel("latent value")
        ax.grid(alpha=0.2)
    for ax in axes.ravel()[len(betas) :]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("density")
    fig.tight_layout()
    fig.savefig(out / filename, dpi=200)
    plt.close(fig)


def plot_latent_histograms(
    *,
    base: Path,
    out: Path,
    prefix: str,
    betas: list[str],
    filename: str,
) -> None:
    fig, axes = plt.subplots(1, len(betas), figsize=(16, 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, beta in zip(axes, betas):
        arrays = np.load(base / f"{prefix}_{beta_suffix(beta)}" / "last_run_arrays.npz")
        values = arrays["latent_values"].reshape(-1)
        ax.hist(values, bins=80, density=True, color="#555555", alpha=0.85)
        ax.set_title(rf"$\beta={beta}$")
        ax.set_xlabel("latent value")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(out / filename, dpi=200)
    plt.close(fig)


def plot_timecourses(
    *,
    base: Path,
    out: Path,
    run_dir: str,
    filename: str,
    title: str,
    include_probs: bool,
) -> None:
    arrays = np.load(base / run_dir / "last_run_arrays.npz")
    latent_values = arrays["latent_values"]
    time = np.linspace(0.0, 1.0, latent_values.shape[0], endpoint=False)

    if include_probs:
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        axes[0].plot(time, arrays["latent_probs"], linewidth=1.5)
        axes[0].set_ylabel(r"$p_c(t)$ scale")
        axes[0].set_title(title)
        axes[0].grid(alpha=0.2)
        axes[1].plot(time, latent_values, linewidth=1.2)
        axes[1].set_xlabel("normalized time")
        axes[1].set_ylabel(r"$s_c(t)$")
        axes[1].grid(alpha=0.2)
    else:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(time, latent_values, linewidth=1.1)
        ax.set_title(title)
        ax.set_xlabel("normalized time")
        ax.set_ylabel(r"$s_c(t)$")
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out / filename, dpi=200)
    plt.close(fig)


def generate_all_figures(base: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    scheduled_betas = ["0.5", "1.0", "2.0"]
    dense_betas = ["4.0", "8.0", "16.0"]
    iid_all_betas = ["0.5", "1.0", "2.0", "4.0", "8.0", "16.0"]
    scheduled_prefix = "threshold_rotation_toy_gg_beta"
    iid_prefix = "threshold_rotation_toy_iid_gg_beta"

    plot_latent_histograms(
        base=base,
        out=out,
        prefix=scheduled_prefix,
        betas=scheduled_betas,
        filename="gg_latent_value_histograms.png",
    )
    plot_timecourses(
        base=base,
        out=out,
        run_dir="threshold_rotation_toy_gg_beta_0p5",
        filename="gg_beta_0p5_latent_timecourses.png",
        title=r"Latent schedules and sampled values, $\beta=0.5$",
        include_probs=True,
    )

    scheduled_specs = [
        ("mean_unit_temporal_density", "mean unit temporal density", "gg_density_vs_theta.png", True),
        ("mean_population_density", "mean active DG units per time", "gg_population_density_vs_theta.png", True),
        ("best_corr_to_latent_values", "best corr. to latent values", "gg_corr_values_vs_theta.png", False),
        ("zero_activity_time_fraction", "fraction of silent time bins", "gg_silent_bins_vs_theta.png", True),
        (
            "best_weight_alignment_to_latent_vectors",
            "best weight cosine to latent vector",
            "gg_weight_alignment_vs_theta.png",
            False,
        ),
        (
            "mean_weight_row_entropy_to_latents",
            "row entropy over latent directions",
            "gg_weight_row_entropy_vs_theta.png",
            False,
        ),
        (
            "mean_weight_col_entropy_over_rows",
            "column entropy over DG rows",
            "gg_weight_col_entropy_vs_theta.png",
            False,
        ),
        (
            "mean_active_unit_lifetime_hoyer_sparsity",
            "active-unit lifetime Hoyer sparsity",
            "gg_active_lifetime_hoyer_sparsity_vs_theta.png",
            False,
        ),
        (
            "mean_population_hoyer_sparsity",
            "population Hoyer sparsity",
            "gg_population_hoyer_sparsity_vs_theta.png",
            False,
        ),
        (
            "mean_active_unit_lifetime_kurtosis",
            "active-unit lifetime kurtosis",
            "gg_active_lifetime_kurtosis_vs_theta.png",
            False,
        ),
    ]

    for metric, ylabel, filename, include_no_learning in scheduled_specs:
        plot_metric_grid(
            base=base,
            out=out,
            prefix=scheduled_prefix,
            betas=scheduled_betas,
            metric=metric,
            ylabel=ylabel,
            filename=filename,
            include_no_learning=include_no_learning,
        )

    plot_latent_histograms(
        base=base,
        out=out,
        prefix=iid_prefix,
        betas=scheduled_betas,
        filename="iid_gg_latent_value_histograms.png",
    )
    plot_timecourses(
        base=base,
        out=out,
        run_dir="threshold_rotation_toy_iid_gg_beta_0p5",
        filename="iid_gg_beta_0p5_latent_timecourses.png",
        title=r"i.i.d. generalized-Gaussian latent values, $\beta=0.5$",
        include_probs=False,
    )

    iid_specs = [
        ("mean_unit_temporal_density", "mean unit temporal density", "iid_gg_density_vs_theta.png", True),
        ("mean_population_density", "mean active DG units per time", "iid_gg_population_density_vs_theta.png", True),
        ("best_corr_to_latent_values", "best corr. to latent values", "iid_gg_corr_values_vs_theta.png", False),
        ("zero_activity_time_fraction", "fraction of silent time bins", "iid_gg_silent_bins_vs_theta.png", True),
        (
            "best_weight_alignment_to_latent_vectors",
            "best weight cosine to latent vector",
            "iid_gg_weight_alignment_vs_theta.png",
            False,
        ),
        (
            "mean_weight_row_entropy_to_latents",
            "row entropy over latent directions",
            "iid_gg_weight_row_entropy_vs_theta.png",
            False,
        ),
        (
            "mean_weight_col_entropy_over_rows",
            "column entropy over DG rows",
            "iid_gg_weight_col_entropy_vs_theta.png",
            False,
        ),
        (
            "mean_active_unit_lifetime_hoyer_sparsity",
            "active-unit lifetime Hoyer sparsity",
            "iid_gg_active_lifetime_hoyer_sparsity_vs_theta.png",
            False,
        ),
        (
            "mean_population_hoyer_sparsity",
            "population Hoyer sparsity",
            "iid_gg_population_hoyer_sparsity_vs_theta.png",
            False,
        ),
        (
            "mean_active_unit_lifetime_kurtosis",
            "active-unit lifetime kurtosis",
            "iid_gg_active_lifetime_kurtosis_vs_theta.png",
            False,
        ),
    ]

    for metric, ylabel, filename, include_no_learning in iid_specs:
        plot_metric_grid(
            base=base,
            out=out,
            prefix=iid_prefix,
            betas=scheduled_betas,
            metric=metric,
            ylabel=ylabel,
            filename=filename,
            include_no_learning=include_no_learning,
        )

    plot_latent_histograms(
        base=base,
        out=out,
        prefix=iid_prefix,
        betas=dense_betas,
        filename="iid_gg_large_beta_latent_value_histograms.png",
    )

    dense_specs = [
        ("mean_unit_temporal_density", "mean unit temporal density", "iid_gg_large_beta_density_vs_theta.png", True),
        (
            "mean_population_density",
            "mean active DG units per time",
            "iid_gg_large_beta_population_density_vs_theta.png",
            True,
        ),
        (
            "best_corr_to_latent_values",
            "best corr. to latent values",
            "iid_gg_large_beta_corr_values_vs_theta.png",
            False,
        ),
        (
            "zero_activity_time_fraction",
            "fraction of silent time bins",
            "iid_gg_large_beta_silent_bins_vs_theta.png",
            True,
        ),
        (
            "best_weight_alignment_to_latent_vectors",
            "best weight cosine to latent vector",
            "iid_gg_large_beta_weight_alignment_vs_theta.png",
            False,
        ),
        (
            "mean_weight_row_entropy_to_latents",
            "row entropy over latent directions",
            "iid_gg_large_beta_weight_row_entropy_vs_theta.png",
            False,
        ),
        (
            "mean_weight_col_entropy_over_rows",
            "column entropy over DG rows",
            "iid_gg_large_beta_weight_col_entropy_vs_theta.png",
            False,
        ),
        (
            "mean_active_unit_lifetime_hoyer_sparsity",
            "active-unit lifetime Hoyer sparsity",
            "iid_gg_large_beta_active_lifetime_hoyer_sparsity_vs_theta.png",
            False,
        ),
        (
            "mean_population_hoyer_sparsity",
            "population Hoyer sparsity",
            "iid_gg_large_beta_population_hoyer_sparsity_vs_theta.png",
            False,
        ),
        (
            "mean_active_unit_lifetime_kurtosis",
            "active-unit lifetime kurtosis",
            "iid_gg_large_beta_active_lifetime_kurtosis_vs_theta.png",
            False,
        ),
    ]

    for metric, ylabel, filename, include_no_learning in dense_specs:
        plot_metric_grid(
            base=base,
            out=out,
            prefix=iid_prefix,
            betas=dense_betas,
            metric=metric,
            ylabel=ylabel,
            filename=filename,
            include_no_learning=include_no_learning,
        )

    plot_iid_all_beta_histograms(
        base=base,
        out=out,
        betas=iid_all_betas,
        filename="iid_gg_all_beta_latent_value_histograms.png",
    )

    all_beta_specs = [
        ("mean_unit_temporal_density", "mean unit temporal density", "iid_gg_all_beta_density_vs_theta.png", True),
        (
            "mean_population_density",
            "mean active DG units per time",
            "iid_gg_all_beta_population_density_vs_theta.png",
            True,
        ),
        (
            "zero_activity_time_fraction",
            "fraction of silent time bins",
            "iid_gg_all_beta_silent_bins_vs_theta.png",
            True,
        ),
        (
            "best_corr_to_latent_values",
            "best corr. to latent values",
            "iid_gg_all_beta_corr_values_vs_theta.png",
            False,
        ),
        (
            "best_weight_alignment_to_latent_vectors",
            "best weight cosine to latent vector",
            "iid_gg_all_beta_weight_alignment_vs_theta.png",
            False,
        ),
        (
            "mean_weight_row_entropy_to_latents",
            "row entropy over latent directions",
            "iid_gg_all_beta_weight_row_entropy_vs_theta.png",
            False,
        ),
        (
            "mean_weight_col_entropy_over_rows",
            "column entropy over DG rows",
            "iid_gg_all_beta_weight_col_entropy_vs_theta.png",
            False,
        ),
        (
            "mean_active_unit_lifetime_hoyer_sparsity",
            "active-unit lifetime Hoyer sparsity",
            "iid_gg_all_beta_active_lifetime_hoyer_sparsity_vs_theta.png",
            False,
        ),
        (
            "mean_active_unit_lifetime_kurtosis",
            "active-unit lifetime kurtosis",
            "iid_gg_all_beta_active_lifetime_kurtosis_vs_theta.png",
            False,
        ),
    ]

    for metric, ylabel, filename, include_no_learning in all_beta_specs:
        plot_metric_grid_wrapped(
            base=base,
            out=out,
            prefix=iid_prefix,
            betas=iid_all_betas,
            metric=metric,
            ylabel=ylabel,
            filename=filename,
            include_no_learning=include_no_learning,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("06_experiments/results"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("06_experiments/results/threshold_rotation_report_figures"),
    )
    parser.add_argument("--font-size", type=int, default=15)
    args = parser.parse_args()

    configure_matplotlib(args.font_size)
    generate_all_figures(args.base_dir, args.out_dir)
    print(f"Wrote report figures to {args.out_dir}")


if __name__ == "__main__":
    main()
