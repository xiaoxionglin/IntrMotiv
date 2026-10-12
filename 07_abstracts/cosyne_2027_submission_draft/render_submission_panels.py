"""Render the final-size COSYNE architecture-variant strip and results figure.

All plotted comparisons come from archived seed-level tables. The selected
seed-8 example uses only the lightweight peak coordinates and occupancy mask
derived from two hash-verified NEMO2 evaluator arrays; raw NPZs are not stored
in this repository.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd


DRAFT = Path(__file__).resolve().parent
ROOT = DRAFT.parents[1]
RESULTS = ROOT / "06_experiments" / "results"
FIGURES = DRAFT / "figures"
DATA = DRAFT / "data"
SOURCES = {
    "online": RESULTS / "cosyne_2027_online_wandb" / "arm_summaries.csv",
    "behavior": DATA / "fresh_behavior_training_seeds.csv",
    "encoder": RESULTS / "cosyne_2027_checkpoint_fields" / "common_panel_analysis" / "paired_encoder_fields.csv",
    "peaks": RESULTS / "cosyne_2027_checkpoint_fields" / "common_panel_analysis" / "paired_peak_spread_by_seed.csv",
    "example": DATA / "c01_seed8_peak_example.json",
    "decoder": RESULTS / "cosyne_2027_temporal_credit_layers" / "paired_by_seed.csv",
    "worker": RESULTS / "cosyne_2027_online_wandb" / "paired_contrasts_by_seed.csv",
}
COLOR = {"C01": "#0072B2", "C05": "#D55E00", "C15": "#009E73"}
GRAY = "#5B646C"


def style() -> str:
    font = findfont(FontProperties(family="Arial"), fallback_to_default=False)
    if Path(font).suffix.lower() not in {".ttf", ".otf"}:
        raise RuntimeError(f"Arial did not resolve to a scalable font: {font}")
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.7,
        "svg.fonttype": "none",
        "svg.hashsalt": "COSYNE-2027-submission-panels",
        "pdf.fonttype": 42,
    })
    return font


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        target = FIGURES / f"{stem}.{suffix}"
        fig.savefig(target, dpi=300,
                    metadata={"Date": None} if suffix == "svg" else None)
        if suffix == "svg":
            target.write_text("\n".join(line.rstrip() for line in
                                        target.read_text().splitlines()) + "\n")
    plt.close(fig)


def variant_strip() -> None:
    """Three schematic policy variants, at the same width as the source art."""
    fig = plt.figure(figsize=(6.58689, 0.91))
    panels = [
        ("Baseline (C01)", "direct exploration", False, False, COLOR["C01"]),
        ("Goal-conditioned policy (C05)", "internal DG target", True, False, COLOR["C05"]),
        ("Goal-conditioned policy (C15)", "frontier goal choice", True, True, COLOR["C15"]),
    ]
    for i, (title, subtitle, has_target, frontier, color) in enumerate(panels):
        left = i * 2.195
        ax = fig.add_axes([left / 6.58689, 0.08, 2.10 / 6.58689, 0.72])
        ax.set_xlim(0, 2.10)
        ax.set_ylim(0, 0.72)
        ax.axis("off")
        ax.add_patch(Rectangle((0.02, 0.02), 2.06, 0.68, fill=False,
                               edgecolor="#BCC5CA", linewidth=0.8))
        ax.text(0.11, 0.55, title, fontsize=10, fontweight="bold", color="#252B30")
        ax.text(0.11, 0.39, subtitle, fontsize=9, color=GRAY)
        ax.add_patch(Circle((0.31, 0.16), 0.07, facecolor=color, edgecolor="white",
                            linewidth=0.5))
        ax.add_patch(FancyArrowPatch((0.40, 0.16), (1.64, 0.16),
                                     arrowstyle="-|>", mutation_scale=9,
                                     linewidth=1.2, color=color,
                                     connectionstyle="arc3,rad=-0.10"))
        if has_target:
            ax.add_patch(Circle((1.75, 0.17), 0.075, facecolor="white",
                                edgecolor=color, linewidth=1.2))
            ax.plot([1.75, 1.75], [0.10, 0.24], color=color, lw=0.7)
            ax.plot([1.68, 1.82], [0.17, 0.17], color=color, lw=0.7)
        if frontier:
            ax.plot([1.42, 1.42], [0.06, 0.31], color=GRAY, lw=0.8, ls="--")
    save(fig, "variant_schematic")


def axes_in(fig: plt.Figure, x: float, y: float, width: float, height: float):
    return fig.add_axes([x / 7.5, y / 5.9, width / 7.5, height / 5.9])


def section(fig: plt.Figure, x: float, y: float, label: str) -> None:
    fig.text(x / 7.5, y / 5.9, label, fontsize=10, fontweight="bold",
             color="#20272C", va="top")


def finish_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.2, width=0.6, pad=1.5)
    ax.grid(axis="x", color="#E6E8E9", lw=0.5, zorder=0)


def seed_rows(ax: plt.Axes, values: pd.DataFrame, groups: list[str],
              value_column: str, *, labels: bool, xlim: tuple[float, float]) -> None:
    ax.axvline(0, color="#70787F", lw=0.8, zorder=0)
    for row, group in enumerate(groups):
        data = values[values.group == group].sort_values("seed")
        if data.empty:
            continue
        y = np.linspace(row - 0.16, row + 0.16, len(data))
        color = COLOR["C01" if group == "Baseline" else
                      "C15" if group == "Frontier" else "C05"]
        ax.scatter(data[value_column], y, s=11, color=color, zorder=3,
                   linewidths=0)
        mean = data[value_column].mean()
        ax.plot([mean, mean], [row - 0.22, row + 0.22],
                color="#171D20", lw=1.5, zorder=4)
    ax.set_ylim(-0.45, len(groups) - 0.55)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_yticks(range(len(groups)), groups if labels else [""] * len(groups))
    finish_axes(ax)


def peak_grid(peaks: np.ndarray, mask: np.ndarray) -> np.ndarray:
    centers = np.arange(150.0, 2000.0, 100.0)
    gx, gy = np.meshgrid(centers, centers)
    squared = (gx[None] - peaks[:, 0, None, None]) ** 2
    squared += (gy[None] - peaks[:, 1, None, None]) ** 2
    grid = np.exp(-squared / (2 * 100.0**2)).sum(axis=0)
    grid[~mask] = 0
    return grid / grid.sum()


def results_figure() -> dict[str, float]:
    online = pd.read_csv(SOURCES["online"])
    behavior = pd.read_csv(SOURCES["behavior"])
    encoder = pd.read_csv(SOURCES["encoder"])
    peaks = pd.read_csv(SOURCES["peaks"])
    decoder = pd.read_csv(SOURCES["decoder"])
    worker = pd.read_csv(SOURCES["worker"])
    example = json.loads(SOURCES["example"].read_text())
    mask = np.asarray(example["visited_bin_mask"], dtype=bool)
    assert mask.shape == (19, 19) and mask.sum() == 242
    baseline_area = peaks[(peaks.contrast == "C01") &
                          (peaks.metric == "effective_area_fraction") &
                          (peaks.cutoff == 0.0025)]
    selected = baseline_area[baseline_area.seed == example["seed"]].iloc[0]
    assert len(baseline_area) == 5
    assert selected.difference == baseline_area.difference.max()
    assert np.isclose(selected.temporal, example["arms"]["temporal"]["effective_area_fraction"])
    assert np.isclose(selected.constant, example["arms"]["constant"]["effective_area_fraction"])

    fig = plt.figure(figsize=(7.5, 5.9), facecolor="white")
    # Sections 1 and 2: time course and behavior-conditioned expression.
    section(fig, 0.12, 5.84, "A  Spatial coding emerges")
    section(fig, 2.52, 5.84, "B  Policy shapes expression")
    ax = axes_in(fig, 0.45, 4.54, 1.64, 1.02)
    for family, worker_arm in (("C01", "inapplicable"), ("C05", "temporal"),
                               ("C15", "temporal")):
        rows = online[(online.family == family) &
                      (online.encoder_credit == "temporal") &
                      (online.worker_bonus == worker_arm) &
                      (online.metric == "spatial_information")].sort_values("env_steps")
        assert len(rows) > 50 and rows.n.eq(5).all()
        x = rows.env_steps.to_numpy() / 1e6
        mean = rows["mean"].to_numpy()
        sem = rows["sem"].to_numpy()
        ax.plot(x, mean, lw=1.2, color=COLOR[family], label=family)
        ax.fill_between(x, mean - sem, mean + sem, alpha=0.13, color=COLOR[family])
    ax.set(xlim=(0, 100), xlabel="Steps (millions)", ylabel="DG score")
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=3,
              handlelength=0.7, columnspacing=0.3, handletextpad=0.2,
              borderpad=0)
    finish_axes(ax)

    fresh = behavior[(behavior.cohort == "fresh") &
                     (behavior.layer == "decoder_1")].dropna(
                         subset=["information_difference", "common_split_half_difference"])
    assert len(fresh) == 38
    behavior_rows = fresh.rename(columns={"training_seed": "seed"}).copy()
    behavior_rows["group"] = behavior_rows.family.map({
        "C01": "Baseline", "C05": "Goal", "C15": "Frontier"})
    groups = ["Baseline", "Goal", "Frontier"]
    for x0, column, title, limits in [
        (3.00, "information_difference", "Decoder info", (-0.06, 1.13)),
        (5.26, "common_split_half_difference", "Map reliability", (-0.05, 0.61)),
    ]:
        ax = axes_in(fig, x0, 4.54, 1.70, 1.02)
        seed_rows(ax, behavior_rows, groups, column, labels=x0 < 4,
                  xlim=limits)
        ax.set_title(title, fontsize=10, pad=2)
        ax.set_xlabel("Own − random", fontsize=9, labelpad=1)

    # Section 3: the largest observed baseline peak-area effect, with all seeds beside it.
    fig.add_artist(plt.Line2D([0.02, 0.98], [4.13 / 5.9, 4.13 / 5.9],
                              transform=fig.transFigure, color="#CCD1D4", lw=0.6))
    section(fig, 0.12, 3.98, "C–D  Temporal push broadens and separates DG code")
    fig.text(0.37 / 7.5, 3.63 / 5.9, "Seed 8: largest of 5 pairs", fontsize=9,
             color=GRAY)
    grids = {}
    for arm in ("temporal", "constant"):
        grids[arm] = peak_grid(np.asarray(example["arms"][arm]["peaks_xy"]), mask)
    vmax = max(np.nanmax(value) for value in grids.values())
    for x0, arm, label in [(0.37, "temporal", "Temporal"), (1.58, "constant", "Constant")]:
        ax = axes_in(fig, x0, 2.44, 1.07, 1.04)
        grid = np.ma.masked_where(~mask, grids[arm])
        cmap = plt.get_cmap("Greys").copy()
        cmap.set_bad("#DDDDDD")
        ax.imshow(grid, origin="lower", extent=(100, 2000, 100, 2000),
                  cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        xy = np.asarray(example["arms"][arm]["peaks_xy"])
        ax.scatter(xy[:, 0], xy[:, 1], s=10, color="#D55E00",
                   edgecolor="white", linewidth=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
        area = example["arms"][arm]["effective_area_fraction"]
        ax.set_title(f"{label} ({area:.3f})", fontsize=9, pad=2)

    # Each DG statistic retains all available paired training seeds.
    contrasts = ["C01", "C05 · temporal hit", "C05 · constant hit", "C15"]
    row_names = ["Base", "Goal-T", "Goal-C", "Frontier"]
    metrics = [
        ("DG score", encoder, "mean_spatial_information_bits", (-0.06, 0.13)),
        ("Map cosine", encoder, "active_map_cosine_mean", (-0.17, 0.09)),
        ("Peak area", peaks, "effective_area_fraction", (-0.23, 0.20)),
    ]
    for index, (title, source, metric, limits) in enumerate(metrics):
        x0 = (3.72, 4.92, 6.12)[index]
        ax = axes_in(fig, x0, 2.46, 1.01, 1.04)
        subset = source[source.metric == metric].copy()
        if source is peaks:
            subset = subset[subset.cutoff == 0.0025]
        subset["group"] = subset.contrast.map(dict(zip(contrasts, row_names)))
        subset = subset.rename(columns={"difference": "value"})
        seed_rows(ax, subset, row_names, "value", labels=index == 0, xlim=limits)
        ax.set_title(title, fontsize=9, pad=2)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_xlabel("Δ", fontsize=9, labelpad=1)

    # Section 4: a narrow downstream effect and the still provisional pull test.
    fig.add_artist(plt.Line2D([0.02, 0.98], [2.18 / 5.9, 2.18 / 5.9],
                              transform=fig.transFigure, color="#CCD1D4", lw=0.6))
    section(fig, 0.12, 2.08, "E  Push reaches the readout")
    section(fig, 3.20, 2.08, "F  Temporal pull: work in progress")
    downstream = decoder[(decoder.family == "C01") &
                         (decoder.factor == "encoder") &
                         (decoder.layer == "decoder_1") &
                         (decoder.metric == "mean_bits_per_activation_eligible")]
    assert len(downstream) == 5 and downstream.difference.gt(0).all()
    ax = axes_in(fig, 0.48, 0.48, 2.05, 1.20)
    x = downstream.sort_values("seed").difference.to_numpy()
    ax.axvline(0, lw=0.8, color=GRAY)
    ax.scatter(x, np.linspace(-0.13, 0.13, len(x)), s=19,
               color=COLOR["C01"], zorder=3)
    ax.plot([x.mean(), x.mean()], [-0.22, 0.22], color="#171D20", lw=1.5)
    ax.set_ylim(-0.35, 0.35)
    ax.set_yticks([])
    ax.set_xlim(-0.05, 1.20)
    ax.set_xlabel("Δ decoder-1 info (bits/activation)", fontsize=9)
    ax.set_title(f"Baseline · 5/5 positive · mean +{x.mean():.2f}",
                 fontsize=9, pad=2)
    finish_axes(ax)

    pull = worker[(worker.contrast == "temporal_minus_constant_hit_bonus") &
                  (worker.family == "C05") &
                  (worker.metric == "visited_cell_fraction")]
    assert len(pull) == 10
    for x0, fixed, label in [(3.66, "temporal", "Temporal encoder"),
                             (5.48, "constant", "Constant encoder")]:
        subset = pull[pull.fixed_factor == fixed].sort_values("seed")
        ax = axes_in(fig, x0, 0.48, 1.48, 1.20)
        y = np.arange(5)
        for offset, row in enumerate(subset.itertuples(index=False)):
            ax.plot([row.second, row.first], [offset, offset],
                    color="#ADB5BA", lw=0.8, zorder=1)
        ax.scatter(subset["first"], y, s=19, color=COLOR["C05"], zorder=3)
        ax.scatter(subset["second"], y, s=19, marker="s", facecolor="white",
                   edgecolor=COLOR["C05"], linewidth=0.9, zorder=3)
        ax.set_xlim(0.52, 0.91)
        ax.set_ylim(-0.45, 4.45)
        ax.set_yticks([])
        ax.set_title(f"{label} · Δ{subset.difference.mean():+.3f}",
                     fontsize=9, pad=2)
        ax.set_xlabel("Visited fraction", fontsize=9)
        finish_axes(ax)
    save(fig, "results_sections")
    return {
        "selected_seed": 8,
        "selected_peak_area_temporal": example["arms"]["temporal"]["effective_area_fraction"],
        "selected_peak_area_constant": example["arms"]["constant"]["effective_area_fraction"],
        "behavior_models": len(fresh),
        "decoder_c01_mean_delta": float(x.mean()),
    }


def main() -> None:
    font = style()
    variant_strip()
    summary = results_figure()
    manifest = {
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in SOURCES.items()},
        "font_path": font,
        "results_size_inches": [7.5, 5.9],
        "variant_size_inches": [6.58689, 0.91],
        "figure_values": summary,
        "peak_example_choice": "largest C01 peak-area difference among five paired seeds",
    }
    (DRAFT / "PANEL_PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
