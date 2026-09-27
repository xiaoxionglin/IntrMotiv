"""Render poster screening figures from canonical frozen-DG study tables."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/cued_reward5_frozen_dg_interim_20260926"
OUT = Path(__file__).resolve().parent / "poster_candidates"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 12, "svg.fonttype": "none"})
COLORS = {"W_SOURCE_DG": "#315c96", "W_RAND_DG": "#be6a37"}
LABELS = {"W_SOURCE_DG": "Source DG", "W_RAND_DG": "Random DG"}


def paired_panel(ax, frame, metric, title, multiplier=1):
    for site_index, site in enumerate(("dg50", "dg51")):
        subset = frame[frame.site == site]
        for seed_index, seed in enumerate((42, 1234, 9999)):
            pair = subset[subset.seed == seed].set_index("arm")
            if len(pair) != 2:
                continue
            x = [site_index * 2, site_index * 2 + 1]
            y = [pair.loc[arm, metric] * multiplier for arm in LABELS]
            ax.plot(x, y, color="0.65", linewidth=1.2, zorder=1)
            for xpos, value, arm in zip(x, y, LABELS):
                ax.scatter(xpos, value, s=42, color=COLORS[arm], zorder=2)
    ax.set_xticks([0, 1, 2, 3], ["source", "random", "source", "random"])
    ax.text(0.5, -0.19, "D50", transform=ax.get_xaxis_transform(), ha="center")
    ax.text(2.5, -0.19, "D51", transform=ax.get_xaxis_transform(), ha="center")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)


online = pd.read_csv(DATA / "online/per_run.csv")
spatial = pd.read_csv(DATA / "spatial/per_snapshot.csv")
spatial = spatial[spatial.target_env_steps == 50_000_000]
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
paired_panel(axes[0], online, "environment_reward_mean", "Reward mean, 40–50M", 1000)
axes[0].set_ylabel("Environment reward × 1,000")
paired_panel(axes[1], spatial, "active_only_map_cosine", "DG map overlap, 50M")
axes[1].set_ylabel("Active-only map cosine")
paired_panel(axes[2], spatial, "graph_reachable_pair_fraction", "Graph reachability, 50M", 100)
axes[2].set_ylabel("Reachable ordered pairs (%)")
fig.suptitle("Frozen source DG versus calibrated random DG: seed-paired observations", fontsize=15)
fig.savefig(OUT / "frozen_dg_transfer_50m.svg", bbox_inches="tight")
plt.close(fig)

# Existing CPU2048 adapter produced this canonical all-snapshot table. The
# comparison is restricted to the six cadence-2048 DDQN+HER rows at 25M.
cpu = pd.read_csv(ROOT / "data/cpu2048_analysis_20260917/all_snapshots.csv")
cpu = cpu[
    (cpu.target_env_steps == 25_000_000)
    & cpu.run_name.str.contains("DDQN_HER")
    & ~cpu.run_name.str.contains("D64")
]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
for ax, (metric, title, scale) in zip(
    axes,
    (
        ("active_only_map_cosine", "DG map overlap", 1),
        ("mono_field_unit_fraction", "Mono-field units", 100),
        ("graph_reachable_pair_fraction", "Graph reachability", 100),
    ),
):
    for seed in (8, 99, 123):
        pair = cpu[cpu.seed == seed].set_index("base")
        y = [pair.loc[base, metric] * scale for base in ("DIRECT_F16", "WAYPOINT_DECODER_F64")]
        ax.plot((0, 1), y, color="0.65", linewidth=1.2)
        ax.scatter((0, 1), y, color=("#315c96", "#be6a37"), s=44, zorder=2)
    ax.set_xticks((0, 1), ("Direct F16", "Waypoint F64"))
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
axes[0].set_ylabel("Active-only cosine")
axes[1].set_ylabel("Eligible units (%)")
axes[2].set_ylabel("Reachable ordered pairs (%)")
fig.suptitle("CPU2048 DDQN+HER, matched 25M: representation and graph separate", fontsize=15)
fig.savefig(OUT / "cpu2048_direct_waypoint_25m.svg", bbox_inches="tight")
plt.close(fig)
