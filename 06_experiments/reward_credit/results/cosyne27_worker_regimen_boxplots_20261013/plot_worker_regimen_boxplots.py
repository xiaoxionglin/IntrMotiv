"""Plot seed-paired worker-bonus contrasts from a compact provenance table.

Each row is one matched training seed: temporal minus constant worker bonus.
The readouts are deliberately kept on separate axes because they have
different units and are measured by different telemetry protocols.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT = HERE / "paired_seed_differences.csv"
GROUPS = [("C05", "original"), ("C05", "high"),
          ("C15", "original"), ("C15", "high")]
SEEDS = (8, 17, 41, 99, 123)
SEED_MARKERS = ("o", "s", "D", "^", "v")
OFFSETS = np.linspace(-0.17, 0.17, len(SEEDS))
METRICS = (
    ("delta_spatial_information_bits", "A  DG spatial information", "bits; 100M own-policy probe"),
    ("delta_coverage_auc", "B  Coverage AUC", "AUC units; 90–100M frames"),
    ("delta_correct_hit_observations", "C  Correct-hit observations", "observations; 90–100M frames"),
    ("delta_reliable_edges", "D  Reliable graph edges", "edges; 100M own-policy probe"),
)
ORIGINAL = "#646a73"
HIGH = "#0072B2"


def load_data() -> pd.DataFrame:
    data = pd.read_csv(INPUT)
    required = {"family", "policy_credit", "seed", "source_bundle"} | {
        metric for metric, _, _ in METRICS
    }
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns: {required - set(data.columns)}")
    expected = {(family, credit, seed) for family, credit in GROUPS
                for seed in SEEDS}
    actual = set(zip(data.family, data.policy_credit, data.seed))
    if actual != expected or len(data) != len(expected):
        raise ValueError("Expected exactly five unique paired seeds in each group")
    if not np.isfinite(data[[metric for metric, _, _ in METRICS]].to_numpy()).all():
        raise ValueError("Non-finite paired difference")
    return data


def draw_panel(ax: plt.Axes, data: pd.DataFrame, groups: list[tuple[str, str]],
               metric: str, title: str, unit_label: str) -> None:
    values = []
    for x, (family, credit) in enumerate(groups):
        subset = data[(data.family == family) &
                      (data.policy_credit == credit)].set_index("seed")
        sample = subset.loc[list(SEEDS), metric].to_numpy(dtype=float)
        values.append(sample)
        color = ORIGINAL if credit == "original" else HIGH
        for seed, offset, marker, value in zip(SEEDS, OFFSETS, SEED_MARKERS, sample):
            ax.scatter(x + offset, value, marker=marker, s=39, facecolor="white",
                       edgecolor=color, linewidth=1.5, zorder=4)
    box = ax.boxplot(values, positions=np.arange(len(groups)), widths=0.58,
                     patch_artist=True, showmeans=True, showfliers=False,
                     whis=(0, 100),
                     medianprops={"color": "#1b1b1b", "linewidth": 1.5},
                     meanprops={"marker": "X", "markersize": 7,
                                "markerfacecolor": "#D55E00",
                                "markeredgecolor": "#1b1b1b", "zorder": 5},
                     whiskerprops={"linewidth": 1.1},
                     capprops={"linewidth": 1.1})
    for (_, credit), patch in zip(groups, box["boxes"]):
        color = ORIGINAL if credit == "original" else HIGH
        patch.set_facecolor(color)
        patch.set_alpha(0.14)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.3)
    ax.axhline(0, color="#858585", linewidth=1.1, zorder=0)
    if len(groups) == 4:
        ax.axvline(1.5, color="#d0d0d0", linewidth=0.9, zorder=0)
    ax.set_xlim(-0.53, len(groups) - 0.47)
    labels = ([f"{family}\n{credit}" for family, credit in groups]
              if len(groups) == 4 else [credit.title() for _, credit in groups])
    ax.set_xticks(range(len(groups)), labels)
    ax.set_title(title, loc="left", weight="bold", pad=9)
    ax.set_ylabel("Temporal − constant\n" + unit_label)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=7)
    ax.grid(axis="y", color="#ebebeb", linewidth=0.7, zorder=0)
    if metric == "delta_coverage_auc" and ("C15", "original") in groups:
        group_index = groups.index(("C15", "original"))
        ax.annotate("seed 123", xy=(group_index + OFFSETS[-1],
                                    values[group_index][-1]),
                    xytext=(group_index + 0.25, values[group_index][-1] + 8),
                    fontsize=12,
                    arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.8})


def render_figure(data: pd.DataFrame, groups: list[tuple[str, str]],
                  stem: str, heading: str, width: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(width, 9.0), constrained_layout=True)
    for ax, (metric, title, unit_label) in zip(axes.flat, METRICS):
        draw_panel(ax, data, groups, metric, title, unit_label)
    handles = [Line2D([], [], marker=marker, linestyle="none", color="#333333",
                      markerfacecolor="white", markersize=7, label=f"Seed {seed}")
               for seed, marker in zip(SEEDS, SEED_MARKERS)]
    handles.append(Line2D([], [], marker="X", linestyle="none", color="#D55E00",
                          markersize=8, label="Mean"))
    fig.legend(handles=handles, loc="outside lower center", ncol=6,
               frameon=False, columnspacing=1.0, handletextpad=0.35)
    fig.suptitle(heading + "\n"
                 "Original: γ=0.99, λ=0.95   ·   High: γ=0.999, λ=0.995",
                 fontsize=14, weight="bold")
    for suffix, options in (("pdf", {}), ("svg", {}),
                            ("png", {"dpi": 300})):
        output = HERE / f"{stem}.{suffix}"
        fig.savefig(output, **options)
        if suffix == "svg":
            # Matplotlib puts padding spaces at SVG path line endings. The
            # newlines already delimit path commands, so trim those spaces.
            output.write_text("\n".join(line.rstrip() for line in
                                        output.read_text().splitlines()) + "\n")
    plt.close(fig)


def main() -> None:
    data = load_data()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "intrmotiv-cosyne-worker-regimen-boxplots-v1",
    })
    render_figure(data, GROUPS, "worker_regimen_boxplots",
                  "Worker timing contrasts across policy-credit settings", 8.4)
    for family in ("C05", "C15"):
        groups = [(family, "original"), (family, "high")]
        render_figure(data, groups, f"{family.lower()}_worker_regimen_boxplots",
                      f"{family} worker timing contrasts", 7.2)


if __name__ == "__main__":
    main()
