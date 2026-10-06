"""Render the matched-age landmark cue comparison from canonical CSV outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
BASE_LABELS = {
    "SCR_ARR_DIRS": "SCR",
    "DGP_HIT_JOINT_LEG": "DGP",
    "WAYPOINT_F64_DDQN_HER": "Waypoint",
}
BASE_ORDER = list(BASE_LABELS)
RICH_SEEDS = (8, 99, 123)
RICH_OFFSETS = {8: -0.03, 99: 0.12, 123: 0.27}


def main() -> None:
    font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=False)
    if not font_path.lower().endswith((".ttf", ".otf")):
        raise RuntimeError(f"No scalable plotting font: {font_path}")
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 17})

    rich_online = pd.read_csv(ROOT / "rich_online_70_75m/per_run.csv")
    neutral_online = pd.read_csv(ROOT / "control_online_70_75m/per_run.csv")
    rich_spatial = pd.read_csv(ROOT / "rich_spatial/per_snapshot.csv")
    neutral_spatial = pd.read_csv(ROOT / "control_spatial/per_snapshot.csv")
    rich_spatial = rich_spatial.loc[rich_spatial.target_env_steps == 75_000_000]
    neutral_spatial = neutral_spatial.loc[neutral_spatial.target_env_steps == 75_000_000]

    panels = [
        (
            rich_online,
            neutral_online,
            "accessible_coverage_auc",
            "A  Training-window exploration",
            "Accessible coverage AUC (70–75M)",
        ),
        (
            rich_spatial,
            neutral_spatial,
            "active_only_map_cosine",
            "B  DG map separation",
            "Active-only map cosine (75M)",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), dpi=100, layout="constrained")
    for ax, (rich, neutral, metric, title, ylabel) in zip(axes, panels):
        for index, base in enumerate(BASE_ORDER):
            neutral_value = neutral.loc[neutral.base == base, metric].iloc[0]
            ax.scatter(
                index - 0.22,
                neutral_value,
                s=140,
                marker="D",
                color="#56545c",
                label="Neutral, seed 99" if index == 0 else None,
                zorder=3,
            )
            for seed in RICH_SEEDS:
                rich_value = rich.loc[(rich.base == base) & (rich.seed == seed), metric].iloc[0]
                ax.scatter(
                    index + RICH_OFFSETS[seed],
                    rich_value,
                    s=150,
                    marker="o",
                    color="#2274aa",
                    edgecolor="white",
                    linewidth=1,
                    label="Rich, seeds 8/99/123" if index == 0 and seed == 8 else None,
                    zorder=3,
                )
        ax.set_xticks(range(len(BASE_ORDER)), [BASE_LABELS[base] for base in BASE_ORDER])
        ax.set_xlim(-0.5, len(BASE_ORDER) - 0.5)
        ax.set_title(title, fontsize=20, loc="left", pad=14)
        ax.set_ylabel(ylabel, fontsize=17)
        ax.grid(axis="y", color="#d7dce1", linewidth=1)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=17)
    axes[0].legend(loc="upper left", frameon=False, fontsize=16)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "online_cue_comparison_75m.png", dpi=100)
    fig.savefig(FIGURES / "online_cue_comparison_75m.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
