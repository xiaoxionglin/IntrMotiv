"""Render seed-visible cue-transfer outcomes and selected maps from slim exports."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.analysis import linear_contrasts

plt.switch_backend("Agg")


def label(condition):
    return condition.removeprefix("CT_D51_").removeprefix("CE_").replace("_", " ").lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    out = args.data / "figures"
    out.mkdir(exist_ok=True)
    font = Path(font_manager.findfont("DejaVu Sans", fallback_to_default=False))
    assert font.suffix in {".ttf", ".otf"} and font.is_file()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    def save(fig, name):
        for ext in ["png", "pdf"]:
            fig.savefig(out / (name + "." + ext), dpi=120, bbox_inches="tight")
        plt.close(fig)

    held = pd.read_csv(args.data / "heldout_per_run.csv")
    correct = held[held.protocol == "correct"]
    details, summaries = [], []
    for study_id in correct.study.unique():
        study = load_study(Path(__file__).resolve().parent / "studies" / (study_id + ".study.json"))
        detail, summary = linear_contrasts(
            correct[correct.study == study_id].to_dict("records"),
            ["success_equal_cue"],
            [],
            ["seed"],
            study.analysis["contrasts"],
        )
        details.extend({**row, "study": study_id} for row in detail)
        summaries.extend({**row, "study": study_id} for row in summary)
    pd.DataFrame(details).to_csv(args.data / "declared_paired_contrasts.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.data / "declared_contrast_summary.csv", index=False)
    delta = held.pivot(index=["condition", "seed"], columns="protocol", values="success_equal_cue")
    for protocol in ["shuffled_cue", "zero_ca3", "zero_depth"]:
        delta["correct_minus_" + protocol] = delta.correct - delta[protocol]
    delta.to_csv(args.data / "paired_intervention_deltas.csv")
    order = correct.groupby("condition").success_equal_cue.mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(9, 14), layout="constrained")
    for y, condition in enumerate(order):
        x = correct[correct.condition == condition].sort_values("seed").success_equal_cue.to_numpy() * 100
        ax.plot(x, np.full(3, y) + np.array([-0.12, 0, 0.12]), "o", color="#0072B2", alpha=0.75)
        ax.plot(x.mean(), y, "|", color="black", markersize=18, markeredgewidth=3)
    ax.set(
        yticks=range(len(order)),
        yticklabels=[label(x) for x in order],
        xlabel="Equal-cue heldout success (%)",
        xlim=(0, 100),
        title="25M checkpoint: three seeds and their mean",
    )
    ax.grid(axis="x", alpha=0.2)
    save(fig, "heldout_success")
    protocols = ["correct", "shuffled_cue", "zero_ca3", "zero_depth"]
    matrix = held.groupby(["condition", "protocol"]).success_equal_cue.mean().unstack().loc[order, protocols] * 100
    fig, ax = plt.subplots(figsize=(9, 14), layout="constrained")
    im = ax.imshow(matrix, vmin=0, vmax=100, cmap="cividis", aspect="auto")
    ax.set(
        yticks=range(len(order)),
        yticklabels=[label(x) for x in order],
        xticks=range(4),
        xticklabels=["Correct", "Shuffled", "Zero CA3", "Zero depth"],
        title="Matched resets: success (%) averaged over three seeds",
    )
    ax.tick_params(axis="x", labelrotation=30)
    for y in range(len(order)):
        for x in range(4):
            v = matrix.iloc[y, x]
            ax.text(x, y, f"{v:.1f}", ha="center", va="center", color="white" if v < 45 else "black", fontsize=14)
    fig.colorbar(im, ax=ax, label="Success (%)", shrink=0.6)
    save(fig, "heldout_interventions")
    curves_path = args.data / "reward_curves.csv"
    if curves_path.exists():
        curves = pd.read_csv(curves_path)
        auc = curves.drop_duplicates(["condition", "seed"])
        columns = [x for x in auc.columns if x.startswith("reward_auc_mean_")]
        auc[["condition", "seed", *columns, "first_reward_log_step", "last_reward_log_step"]].to_csv(
            args.data / "reward_auc_per_run.csv", index=False
        )
        auc.groupby("condition")[columns].agg(["mean", "std"]).to_csv(args.data / "reward_auc_condition_summary.csv")
        comparisons = [
            (
                "reward_learning_curves",
                "Original four: seed curves and means",
                curves[curves.study == "cue_transfer_20261104"],
            ),
            (
                "adaptation_recipe_curves",
                "Adaptation: seed curves and means",
                curves[
                    curves.condition.isin(
                        ["CT_D51_SOURCE_FREE", "CT_D51_FRESH_FREE", "CE_RANDOM_PPO_FREE", "CE_FRESH_FREE_JOINT"]
                    )
                ],
            ),
        ]
        for figure_name, title, subset in comparisons:
            fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
            colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
            for color, (condition, group) in zip(colors, subset.groupby("condition")):
                for _, seed in group.groupby("seed"):
                    ax.plot(seed.frames / 1e6, seed.reward_mean, color=color, alpha=0.18, lw=1)
                mean = group.groupby("frames").reward_mean.mean()
                ax.plot(mean.index / 1e6, mean, color=color, lw=2, label=label(condition))
            ax.set(
                xlabel="Environment frames (millions)",
                ylabel="Logged mean external reward",
                title=title,
                xlim=(0, 25),
            )
            ax.legend(fontsize=13)
            save(fig, figure_name)
    maps = np.load(args.data / "representative_maps_and_paths.npz", allow_pickle=False)
    selected = [
        "CT_D51_FRESH_SIMPLEX",
        "CT_D51_FRESH_FREE",
        "CT_D51_SOURCE_SIMPLEX",
        "CT_D51_SOURCE_FREE",
        "CE_SOURCE_FREE_BN",
        "CE_SOURCE_FREE_PPO",
        "CE_SOURCE_FREE_JOINT",
        "CE_RANDOM_FROZEN_FREE",
    ]
    for pair in range(4):
        fig, axes = plt.subplots(2, 3, figsize=(10, 8), layout="constrained")
        for row, condition in enumerate(selected[2 * pair : 2 * pair + 2]):

            def get(key):
                return maps[condition + "__" + key]

            bounds = get("bounds")
            extent = [bounds[0], bounds[1], bounds[2], bounds[3]]
            pose = get("pose")
            # histogramdd stores x,y; image rows must be y,x for world-coordinate paths.
            axes[row, 0].imshow(get("occupancy").T, origin="lower", extent=extent, cmap="Greys")
            for segment in np.unique(pose[:, 2]):
                xy = pose[pose[:, 2] == segment, :2]
                axes[row, 0].plot(xy[:, 0], xy[:, 1], lw=0.5, alpha=0.55, color="#0072B2")
            active = get("active_fraction") > 0
            si = np.where(active, get("spatial_information"), -np.inf)
            unit = int(np.argmax(si))
            for col, key, title in [(1, "rate_maps", "Thresholded"), (2, "pre_threshold_rate_maps", "Pre-threshold")]:
                im = axes[row, col].imshow(get(key)[:, :, unit].T, origin="lower", extent=extent, cmap="viridis")
                axes[row, col].set_title(f"{title}: unit {unit}", fontsize=16)
                fig.colorbar(im, ax=axes[row, col], shrink=0.65)
            axes[row, 0].set_title(label(condition) + "\nOccupancy + paths", fontsize=16)
            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle("Seed 42, 25M: maximum-SI active unit per run", fontsize=18)
        save(fig, "maps_and_paths_" + str(pair + 1))


if __name__ == "__main__":
    main()
