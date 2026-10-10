"""Render readable terminal frozen-policy footprints for the reward ablation.

Input NPZ and pose CSV files come from the canonical 10k-decision place-field
manifest. The same arena, occupancy scale, and initial pose are used for every
seed-99 run. Each path is a separate stochastic policy rollout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


ARENA = (100, 2000)
LABELS = {"CONSTANT_STOP": "Constant encoder credit", "NONE_STOP": "No credit, PPO stopped", "NONE_JOINT": "No credit, PPO into DG"}


def setup_font() -> None:
    font = Path(font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans"), fallback_to_default=False
    ))
    if font.suffix.lower() not in {".ttf", ".otf"} or not font.is_file():
        raise RuntimeError("Verified scalable font required")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 24,
                         "axes.titlesize": 24, "axes.labelsize": 24,
                         "xtick.labelsize": 22, "ytick.labelsize": 22,
                         "pdf.fonttype": 42})


def load_run(root: Path, run_name: str) -> tuple[np.ndarray, pd.DataFrame]:
    folders = list(root.glob(f"{run_name}__*/place_fields.npz"))
    if len(folders) != 1:
        raise ValueError(f"Expected one frozen NPZ for {run_name}; got {len(folders)}")
    with np.load(folders[0], allow_pickle=False) as raw:
        occupancy = np.asarray(raw["occupancy"])
    pose = pd.read_csv(folders[0].with_name("pose.csv"))
    if len(pose) != 10_001 or occupancy.shape != (19, 19):
        raise ValueError(f"Unexpected 10k protocol in {run_name}")
    return occupancy, pose


def save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def footprint(occupancy: np.ndarray, family: str, bonus: str, credit: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 8.5), layout="constrained")
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad("#eeeeee")
    image = ax.imshow(np.ma.masked_equal(occupancy, 0), origin="lower",
                      extent=(*ARENA, *ARENA), cmap=cmap,
                      norm=LogNorm(vmin=1, vmax=400), interpolation="nearest")
    fig.colorbar(image, ax=ax, shrink=0.74, label="Visits per bin")
    ax.set(xlim=ARENA, ylim=ARENA, aspect="equal", xlabel="x (arena units)",
           ylabel="y (arena units)")
    ax.set_xticks((100, 1000, 2000))
    ax.set_yticks((100, 1000, 2000))
    ax.set_title(f"{family} {bonus} · seed 99\n{LABELS[credit]}\n"
                 f"{np.count_nonzero(occupancy)} / 361 visited bins")
    save(fig, output)


def first_episode(pose: pd.DataFrame, family: str, bonus: str, credit: str, output: Path) -> None:
    first = pose.loc[pose.num_traj == pose.num_traj.iloc[0]]
    fig, ax = plt.subplots(figsize=(8.5, 8.5), layout="constrained")
    ax.plot(first.x, first.y, color="#0072B2", linewidth=1.3)
    ax.scatter(first.x.iloc[0], first.y.iloc[0], s=160, color="#009E73",
               marker="o", zorder=4, label="Start")
    ax.scatter(first.x.iloc[-1], first.y.iloc[-1], s=170, color="#D55E00",
               marker="x", linewidths=3, zorder=4, label="End")
    ax.set(xlim=ARENA, ylim=ARENA, aspect="equal", xlabel="x (arena units)",
           ylabel="y (arena units)")
    ax.set_xticks((100, 1000, 2000))
    ax.set_yticks((100, 1000, 2000))
    ax.set_title(f"{family} {bonus} · seed 99\n{LABELS[credit]}\n"
                 f"First {len(first)} decisions")
    ax.legend(loc="upper right", fontsize=21)
    save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen_root", type=Path)
    parser.add_argument("online_terminal_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    setup_font()
    metadata = pd.read_csv(args.online_terminal_csv)
    seed99 = metadata.loc[metadata.seed == 99]
    if len(seed99) != 15:
        raise ValueError(f"Expected 15 seed-99 terminal runs, got {len(seed99)}")
    for run in seed99.itertuples():
        occupancy, pose = load_run(args.frozen_root, run.run_name)
        stem = f"{run.family.lower()}_{run.worker_bonus}_{run.credit.lower()}_s99"
        footprint(occupancy, run.family, run.worker_bonus, run.credit,
                  args.output_dir / f"{stem}_footprint")
        first_episode(pose, run.family, run.worker_bonus, run.credit,
                      args.output_dir / f"{stem}_first_episode")
    print(f"Rendered 15 footprints and first episodes in {args.output_dir}")


if __name__ == "__main__":
    main()
