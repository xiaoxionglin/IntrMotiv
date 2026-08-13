#!/usr/bin/env python3
"""Toy experiment for batchnorm thresholding plus fixed-norm rotations.

This intentionally avoids the IntrMotiv/Sample Factory modules. It tests one
mechanism only:

    x -> fixed-norm linear projection -> batchnorm -> threshold -> local rotation

The two learning variants differ only by rotation direction for active units:

    encourage: rotate active rows toward active inputs
    punish:    rotate active rows away from active inputs

Synthetic inputs are generated from latent sparse causes with different
time-dependent activation probabilities. The latent schedules are used only for
analysis, not for training.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


CONDITIONS = ("no_learning", "encourage", "punish")


@dataclass(frozen=True)
class ToyData:
    x: np.ndarray
    latent_values: np.ndarray
    latent_active: np.ndarray
    latent_probs: np.ndarray
    cause_vectors: np.ndarray
    time: np.ndarray


@dataclass(frozen=True)
class RunResult:
    condition: str
    theta: float
    seed: int
    final_w: np.ndarray
    activity: np.ndarray
    h: np.ndarray
    metrics: dict[str, float]


def unit_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def make_latent_probabilities(n_steps: int, n_causes: int) -> np.ndarray:
    """Create diverse time-varying sparsity schedules.

    The first eight schedules are hand-shaped to make the analysis readable.
    Extra causes get smooth random mixtures of sinusoids and bumps.
    """

    t = np.linspace(0.0, 1.0, n_steps, endpoint=False)
    schedules: list[np.ndarray] = [
        0.04 + 0.70 * np.exp(-0.5 * ((t - 0.16) / 0.09) ** 2),  # early local
        0.04 + 0.70 * np.exp(-0.5 * ((t - 0.84) / 0.09) ** 2),  # late local
        0.03 + 0.48 * (np.sin(2 * np.pi * 3 * t) > 0.62),       # periodic sparse
        0.02 + 0.82 * np.exp(-0.5 * ((t - 0.50) / 0.035) ** 2), # narrow pulse
        0.22 + 0.18 * np.sin(2 * np.pi * t + 0.4),              # broad dense
        0.02 + 0.58 * np.exp(-0.5 * ((t - 0.36) / 0.16) ** 2),  # broad local
        0.015 + 0.12 * np.ones_like(t),                         # rare-ish
        0.40 + 0.20 * np.sin(2 * np.pi * 2 * t + 1.7),          # dense oscillatory
    ]

    rng = np.random.default_rng(12345)
    while len(schedules) < n_causes:
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.integers(1, 5)
        center = rng.uniform(0.05, 0.95)
        width = rng.uniform(0.04, 0.18)
        base = rng.uniform(0.015, 0.10)
        amp = rng.uniform(0.20, 0.65)
        wave = 0.5 + 0.5 * np.sin(2 * np.pi * freq * t + phase)
        bump = np.exp(-0.5 * ((t - center) / width) ** 2)
        schedules.append(base + amp * (0.35 * wave + 0.65 * bump))

    probs = np.stack(schedules[:n_causes], axis=1)
    return np.clip(probs, 0.0, 0.95)


def sample_unit_generalized_gaussian(rng: np.random.Generator, shape: tuple[int, ...], beta: float) -> np.ndarray:
    """Sample a unit-scale symmetric generalized Gaussian.

    Shape beta controls sparsity/heavy tails:
      beta < 1: sharper peak near zero and heavier tails
      beta = 1: Laplace-like
      beta = 2: Gaussian-like

    Sampling identity: if Y ~ Gamma(1 / beta, 1), then |X| = Y ** (1 / beta)
    has generalized-Gaussian tails exp(-|x| ** beta), up to scale.
    """

    if beta <= 0.0:
        raise ValueError("--gg-beta must be positive")

    magnitude = rng.gamma(shape=1.0 / beta, scale=1.0, size=shape) ** (1.0 / beta)
    sign = rng.choice(np.array([-1.0, 1.0]), size=shape)
    x = sign * magnitude

    # Normalize variance to roughly one so beta sweeps change sparsity/shape more
    # than raw scale. Var = Gamma(3/beta) / Gamma(1/beta) for scale=1.
    variance = math.gamma(3.0 / beta) / math.gamma(1.0 / beta)
    return x / math.sqrt(variance)


def make_toy_data(
    *,
    seed: int,
    n_steps: int,
    obs_dim: int,
    n_causes: int,
    noise_std: float,
    latent_mode: str,
    gg_beta: float,
    gg_activity_threshold: float,
) -> ToyData:
    rng = np.random.default_rng(seed)
    probs = make_latent_probabilities(n_steps, n_causes)

    if latent_mode == "bernoulli":
        latent_active = rng.binomial(1, probs).astype(np.float64)
        latent_values = latent_active * rng.lognormal(mean=0.0, sigma=0.20, size=latent_active.shape)
    elif latent_mode == "generalized_gaussian":
        gg = sample_unit_generalized_gaussian(rng, probs.shape, gg_beta)
        # The schedule controls time-varying channel strength. The generalized
        # Gaussian shape controls how often a channel is near-zero versus large.
        latent_values = probs * np.abs(gg)
        latent_active = (latent_values > gg_activity_threshold).astype(np.float64)
    elif latent_mode == "iid_generalized_gaussian":
        gg = sample_unit_generalized_gaussian(rng, probs.shape, gg_beta)
        # No designed temporal probability envelope. This keeps a continuous,
        # heavy-tailed latent distribution but removes direct temporal sparsity
        # structure that thresholding could trivially capture.
        latent_values = np.abs(gg)
        latent_active = (latent_values > gg_activity_threshold).astype(np.float64)
    else:
        raise ValueError(f"Unknown latent mode: {latent_mode}")

    cause_vectors = unit_normalize(rng.normal(size=(n_causes, obs_dim)), axis=1)
    x = latent_values @ cause_vectors
    x += noise_std * rng.normal(size=x.shape)

    empty = np.linalg.norm(x, axis=1) < 1e-8
    if empty.any():
        x[empty] = rng.normal(size=(empty.sum(), obs_dim))

    x = unit_normalize(x, axis=1)
    time = np.linspace(0.0, 1.0, n_steps, endpoint=False)
    return ToyData(
        x=x,
        latent_values=latent_values,
        latent_active=latent_active,
        latent_probs=probs,
        cause_vectors=cause_vectors,
        time=time,
    )


def batchnorm_channels(y: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = y.mean(axis=0, keepdims=True)
    std = y.std(axis=0, keepdims=True)
    return (y - mean) / np.maximum(std, eps)


def threshold_activity(x: np.ndarray, w: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = x @ w.T
    z = batchnorm_channels(y)
    h = np.maximum(z - theta, 0.0)
    active = h > 0.0
    return y, h, active


def train_one(
    *,
    data: ToyData,
    condition: str,
    theta: float,
    seed: int,
    n_units: int,
    batch_size: int,
    epochs: int,
    lr: float,
) -> RunResult:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")

    rng = np.random.default_rng(seed)
    obs_dim = data.x.shape[1]
    w = unit_normalize(rng.normal(size=(n_units, obs_dim)), axis=1)

    n_steps = data.x.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n_steps)
        for start in range(0, n_steps, batch_size):
            idx = order[start : start + batch_size]
            xb = data.x[idx]
            y, h, active = threshold_activity(xb, w, theta)

            if condition == "no_learning" or not active.any():
                continue

            signed_lr = lr if condition == "encourage" else -lr
            # Tangent-space rotation for fixed-norm rows:
            # x - (w_i dot x) w_i is the component of x orthogonal to w_i.
            # Vectorized form of mean_b h[b, i] * (x[b] - y[b, i] * w[i]).
            batch_n = float(xb.shape[0])
            toward_input = h.T @ xb / batch_n
            radial_component = (h * y).sum(axis=0)[:, None] * w / batch_n
            delta = toward_input - radial_component

            w = unit_normalize(w + signed_lr * delta, axis=1)

    _, h_full, active_full = threshold_activity(data.x, w, theta)
    metrics = compute_metrics(active_full, h_full, w, data)
    return RunResult(
        condition=condition,
        theta=theta,
        seed=seed,
        final_w=w,
        activity=active_full,
        h=h_full,
        metrics=metrics,
    )


def mean_field_width(active: np.ndarray) -> float:
    widths: list[int] = []
    for unit_active in active.T:
        current = 0
        for value in unit_active:
            if value:
                current += 1
            elif current:
                widths.append(current)
                current = 0
        if current:
            widths.append(current)
    return float(np.mean(widths)) if widths else 0.0


def best_profile_correlation(active: np.ndarray, latent_profiles: np.ndarray) -> float:
    """Mean best absolute correlation from learned units to latent schedules."""

    learned = active.astype(np.float64)
    learned = learned - learned.mean(axis=0, keepdims=True)
    latent = latent_profiles - latent_profiles.mean(axis=0, keepdims=True)

    learned_norm = np.linalg.norm(learned, axis=0)
    latent_norm = np.linalg.norm(latent, axis=0)
    valid_l = learned_norm > 1e-8
    valid_p = latent_norm > 1e-8
    if not valid_l.any() or not valid_p.any():
        return 0.0

    corr = learned[:, valid_l].T @ latent[:, valid_p]
    corr /= learned_norm[valid_l, None]
    corr /= latent_norm[valid_p][None, :]
    return float(np.mean(np.max(np.abs(corr), axis=1)))


def best_weight_alignment(w: np.ndarray, cause_vectors: np.ndarray) -> float:
    """Mean best absolute cosine similarity from each learned row to latent directions."""

    w_norm = unit_normalize(w, axis=1)
    cause_norm = unit_normalize(cause_vectors, axis=1)
    cosine = np.abs(w_norm @ cause_norm.T)
    return float(np.mean(np.max(cosine, axis=1)))


def entropy_from_scores(scores: np.ndarray, axis: int, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized entropy and effective support along axis."""

    total = scores.sum(axis=axis, keepdims=True)
    probs = scores / np.maximum(total, eps)
    entropy = -np.sum(np.where(probs > 0.0, probs * np.log(np.maximum(probs, eps)), 0.0), axis=axis)
    n = scores.shape[axis]
    if n > 1:
        normalized_entropy = entropy / math.log(n)
    else:
        normalized_entropy = np.zeros_like(entropy)
    effective_support = np.exp(entropy)
    return normalized_entropy, effective_support


def weight_alignment_distribution_metrics(w: np.ndarray, cause_vectors: np.ndarray) -> dict[str, float]:
    """Distributional alignment metrics from |W V^T|.

    Row-wise entropy asks whether each DG row is concentrated on one latent
    direction or spread across many. Column-wise entropy asks whether each latent
    direction is represented by one/few DG rows or spread across many rows.
    """

    w_norm = unit_normalize(w, axis=1)
    cause_norm = unit_normalize(cause_vectors, axis=1)
    scores = np.abs(w_norm @ cause_norm.T)

    row_entropy, row_effective = entropy_from_scores(scores, axis=1)
    col_entropy, col_effective = entropy_from_scores(scores, axis=0)
    row_participation = 1.0 / np.maximum(np.sum((scores / np.maximum(scores.sum(axis=1, keepdims=True), 1e-12)) ** 2, axis=1), 1e-12)
    col_participation = 1.0 / np.maximum(np.sum((scores / np.maximum(scores.sum(axis=0, keepdims=True), 1e-12)) ** 2, axis=0), 1e-12)

    return {
        "mean_weight_row_entropy_to_latents": float(row_entropy.mean()),
        "mean_weight_row_effective_latents": float(row_effective.mean()),
        "mean_weight_row_participation_latents": float(row_participation.mean()),
        "mean_weight_col_entropy_over_rows": float(col_entropy.mean()),
        "mean_weight_col_effective_rows": float(col_effective.mean()),
        "mean_weight_col_participation_rows": float(col_participation.mean()),
    }


def hoyer_sparsity(x: np.ndarray, axis: int, eps: float = 1e-12) -> np.ndarray:
    """Hoyer sparsity in [0, 1], where 1 is maximally sparse.

    This is scale invariant and works for nonnegative activations:

        (sqrt(n) - ||x||_1 / ||x||_2) / (sqrt(n) - 1)

    All-zero vectors are treated as maximally sparse.
    """

    n = x.shape[axis]
    if n <= 1:
        return np.ones(x.shape[:axis] + x.shape[axis + 1 :], dtype=np.float64)

    l1 = np.sum(np.abs(x), axis=axis)
    l2 = np.linalg.norm(x, axis=axis)
    raw = (math.sqrt(n) - l1 / np.maximum(l2, eps)) / (math.sqrt(n) - 1.0)
    raw = np.where(l2 <= eps, 1.0, raw)
    return np.clip(raw, 0.0, 1.0)


def pearson_kurtosis(x: np.ndarray, axis: int, eps: float = 1e-12) -> np.ndarray:
    """Pearson kurtosis of activation time series. Constant vectors return 0."""

    centered = x - np.mean(x, axis=axis, keepdims=True)
    var = np.mean(centered * centered, axis=axis)
    fourth = np.mean(centered**4, axis=axis)
    return np.where(var <= eps, 0.0, fourth / np.maximum(var * var, eps))


def compute_metrics(active: np.ndarray, h: np.ndarray, w: np.ndarray, data: ToyData) -> dict[str, float]:
    per_unit_density = active.mean(axis=0)
    per_time_population = active.sum(axis=1)
    total_activity = h.sum(axis=1)

    active_units = per_unit_density > 0.0
    density_values = per_unit_density[active_units] if active_units.any() else np.array([0.0])
    lifetime_hoyer = hoyer_sparsity(h, axis=0)
    active_lifetime_hoyer = lifetime_hoyer[active_units] if active_units.any() else np.array([1.0])
    lifetime_kurtosis = pearson_kurtosis(h, axis=0)
    active_lifetime_kurtosis = lifetime_kurtosis[active_units] if active_units.any() else np.array([0.0])
    population_hoyer = hoyer_sparsity(h, axis=1)

    metrics = {
        "mean_unit_temporal_density": float(per_unit_density.mean()),
        "median_active_unit_temporal_density": float(np.median(density_values)),
        "std_unit_temporal_density": float(per_unit_density.std()),
        "mean_population_density": float(per_time_population.mean()),
        "std_population_density": float(per_time_population.std()),
        "unit_coverage": float(active_units.mean()),
        "zero_activity_time_fraction": float((per_time_population == 0).mean()),
        "mean_field_width": mean_field_width(active),
        "mean_total_activation": float(total_activity.mean()),
        "mean_lifetime_hoyer_sparsity": float(lifetime_hoyer.mean()),
        "mean_active_unit_lifetime_hoyer_sparsity": float(active_lifetime_hoyer.mean()),
        "mean_population_hoyer_sparsity": float(population_hoyer.mean()),
        "mean_lifetime_kurtosis": float(lifetime_kurtosis.mean()),
        "mean_active_unit_lifetime_kurtosis": float(active_lifetime_kurtosis.mean()),
        "best_corr_to_latent_probs": best_profile_correlation(active, data.latent_probs),
        "best_corr_to_latent_values": best_profile_correlation(active, data.latent_values),
        "best_corr_to_latent_samples": best_profile_correlation(active, data.latent_active),
        "best_weight_alignment_to_latent_vectors": best_weight_alignment(w, data.cause_vectors),
    }
    metrics.update(weight_alignment_distribution_metrics(w, data.cause_vectors))
    return metrics


def write_summary_csv(results: Iterable[RunResult], path: Path) -> None:
    rows = []
    for result in results:
        row = {"condition": result.condition, "theta": result.theta, "seed": result.seed}
        row.update(result.metrics)
        rows.append(row)

    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate_csv(results: Iterable[RunResult], path: Path) -> None:
    grouped: dict[tuple[str, float], list[RunResult]] = {}
    for result in results:
        grouped.setdefault((result.condition, result.theta), []).append(result)

    rows = []
    for (condition, theta), group in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        metric_names = group[0].metrics.keys()
        row = {"condition": condition, "theta": theta, "n_seeds": len(group)}
        for name in metric_names:
            values = np.array([r.metrics[name] for r in group], dtype=np.float64)
            row[f"{name}_mean"] = float(values.mean())
            row[f"{name}_std"] = float(values.std(ddof=0))
        rows.append(row)

    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(results: list[RunResult], data: ToyData, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot latent schedules once.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.time, data.latent_probs)
    ax.set_title("Latent activation probabilities")
    ax.set_xlabel("normalized time")
    ax.set_ylabel("P(active)")
    fig.tight_layout()
    fig.savefig(out_dir / "latent_probabilities.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.time, data.latent_values)
    ax.set_title("Latent values")
    ax.set_xlabel("normalized time")
    ax.set_ylabel("value")
    fig.tight_layout()
    fig.savefig(out_dir / "latent_values.png", dpi=160)
    plt.close(fig)

    # Rasters for the first seed at each theta/condition.
    first_seed = min(result.seed for result in results)
    for theta in sorted({result.theta for result in results}):
        subset = [
            result
            for result in results
            if result.seed == first_seed and math.isclose(result.theta, theta)
        ]
        subset.sort(key=lambda r: CONDITIONS.index(r.condition))
        if not subset:
            continue

        fig, axes = plt.subplots(len(subset), 1, figsize=(10, 2.6 * len(subset)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, result in zip(axes, subset):
            ax.imshow(result.activity.T, aspect="auto", interpolation="nearest", cmap="Greys")
            ax.set_ylabel("unit")
            ax.set_title(
                f"{result.condition}, theta={theta:g}, "
                f"mean unit density={result.metrics['mean_unit_temporal_density']:.3f}"
            )
        axes[-1].set_xlabel("time step")
        fig.tight_layout()
        safe_theta = str(theta).replace(".", "p")
        fig.savefig(out_dir / f"activity_rasters_theta_{safe_theta}.png", dpi=160)
        plt.close(fig)

    # Aggregate density vs theta.
    fig, ax = plt.subplots(figsize=(8, 5))
    for condition in CONDITIONS:
        xs = []
        ys = []
        yerr = []
        for theta in sorted({result.theta for result in results}):
            values = [
                result.metrics["mean_unit_temporal_density"]
                for result in results
                if result.condition == condition and math.isclose(result.theta, theta)
            ]
            if values:
                xs.append(theta)
                ys.append(float(np.mean(values)))
                yerr.append(float(np.std(values)))
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=condition)
    ax.set_xlabel("theta")
    ax.set_ylabel("mean unit temporal density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "density_vs_theta.png", dpi=160)
    plt.close(fig)


def parse_theta(values: str) -> list[float]:
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("06_experiments/results/threshold_rotation_toy"))
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--obs-dim", type=int, default=24)
    parser.add_argument("--n-causes", type=int, default=8)
    parser.add_argument("--n-units", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument(
        "--latent-mode",
        choices=("bernoulli", "generalized_gaussian", "iid_generalized_gaussian"),
        default="bernoulli",
        help=(
            "Latent generator. generalized_gaussian uses time-varying scales and a beta shape parameter; "
            "iid_generalized_gaussian removes the designed temporal scale envelope."
        ),
    )
    parser.add_argument(
        "--gg-beta",
        type=float,
        default=0.5,
        help="Generalized Gaussian shape. Lower is more sparse/heavy-tailed; 1 is Laplace-like; 2 is Gaussian-like.",
    )
    parser.add_argument(
        "--gg-activity-threshold",
        type=float,
        default=0.15,
        help="Evaluation-only threshold for marking generalized-Gaussian latent causes active.",
    )
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=20, help="Number of weight initialization seeds.")
    parser.add_argument("--theta", type=parse_theta, default=parse_theta("0,0.5,1,1.5,2,2.43"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    data = make_toy_data(
        seed=args.data_seed,
        n_steps=args.n_steps,
        obs_dim=args.obs_dim,
        n_causes=args.n_causes,
        noise_std=args.noise_std,
        latent_mode=args.latent_mode,
        gg_beta=args.gg_beta,
        gg_activity_threshold=args.gg_activity_threshold,
    )

    results: list[RunResult] = []
    for theta in args.theta:
        for seed in range(args.seeds):
            for condition in CONDITIONS:
                result = train_one(
                    data=data,
                    condition=condition,
                    theta=theta,
                    seed=seed,
                    n_units=args.n_units,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr=args.lr,
                )
                results.append(result)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(results, args.out_dir / "summary.csv")
    write_aggregate_csv(results, args.out_dir / "aggregate.csv")

    np.savez_compressed(
        args.out_dir / "last_run_arrays.npz",
        x=data.x,
        latent_values=data.latent_values,
        latent_active=data.latent_active,
        latent_probs=data.latent_probs,
        cause_vectors=data.cause_vectors,
        latent_mode=args.latent_mode,
        gg_beta=args.gg_beta,
        gg_activity_threshold=args.gg_activity_threshold,
    )

    if not args.no_plots:
        maybe_plot(results, data, args.out_dir / "plots")

    print(f"Wrote results to {args.out_dir}")
    print(f"Summary CSV:   {args.out_dir / 'summary.csv'}")
    print(f"Aggregate CSV: {args.out_dir / 'aggregate.csv'}")


if __name__ == "__main__":
    main()
