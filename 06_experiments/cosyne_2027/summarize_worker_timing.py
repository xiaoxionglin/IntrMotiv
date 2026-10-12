"""Join existing C05 seed-paired evidence without conflating its outcome types.

The fresh behavior table is read from its immutable source commit so this PR
can consume that independent analysis without copying or editing its report.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "06_experiments/results"
OUT = RESULTS / "cosyne_2027_checkpoint_fields/worker_timing_summary"
BEHAVIOR_COMMIT = "6412596"
BEHAVIOR_PATH = "06_experiments/results/cosyne_2027_behavior_secondary/all_training_seeds.csv"
SEEDS = {8, 17, 41, 99, 123}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def add_pair(pairs: dict, source: str, metric: str, encoder: str, seed: int,
             temporal: float, constant: float) -> None:
    key = (source, metric, encoder, seed)
    if key in pairs:
        raise ValueError(f"Duplicate contrast: {key}")
    pairs[key] = (temporal, constant)


def main() -> None:
    sources = {
        "online_90_99m": RESULTS / "cosyne_2027_online_wandb/paired_contrasts_by_seed.csv",
        "correct_hit_90_100m": RESULTS / "cosyne_2027_checkpoint_fields/conditional_hits/conditional_hit_seed_metrics.csv",
        "same_observations_100m": RESULTS / "cosyne_2027_checkpoint_fields/common_panel_analysis/paired_worker_fields.csv",
        "trajectory_100m": RESULTS / "cosyne_2027_checkpoint_fields/graph_trajectory_summary/paired_worker_trajectory.csv",
        "stored_graph_100m": RESULTS / "cosyne_2027_checkpoint_fields/graph_trajectory_summary/paired_worker_directed_graph.csv",
    }
    training = rows(ROOT / "06_experiments/cosyne_2027/training_runs.csv")
    identities = {(r["encoder_credit"], r["worker_bonus"], int(r["seed"])): r
                  for r in training if r["family"] == "C05" and r["cohort"] == "cosyne"}
    assert len(identities) == 20, f"Expected 20 original C05 StudySpec identities, got {len(identities)}"
    for r in identities.values():
        assert r["schema"] == "intrmotiv/study/v1" and r["study_sha256"]

    pairs: dict[tuple[str, str, str, int], tuple[float, float]] = {}
    for r in rows(sources["online_90_99m"]):
        if r["family"] == "C05" and r["contrast"] == "temporal_minus_constant_hit_bonus" and r["metric"] == "visited_cell_fraction":
            add_pair(pairs, "online_90_99m", "visited_cell_fraction", r["fixed_factor"], int(r["seed"]), float(r["first"]), float(r["second"]))

    hit_by_cell = {(r["encoder_credit"], r["worker_bonus"], int(r["seed"])): r
                   for r in rows(sources["correct_hit_90_100m"]) if r["family"] == "C05"}
    assert set(hit_by_cell) == set(identities)
    for encoder in ("temporal", "constant"):
        for seed in sorted(SEEDS):
            t, c = (hit_by_cell[(encoder, bonus, seed)] for bonus in ("temporal", "constant"))
            assert (t["window_low"], t["window_high"]) == ("90000000", "100000000")
            for metric in ("correct_hit_observation_count", "censored_observation_count",
                           "correct_reward_magnitude_count_weighted", "correct_elapsed_count_weighted"):
                add_pair(pairs, "correct_hit_90_100m", metric, encoder, seed, float(t[metric]), float(c[metric]))

    for r in rows(sources["same_observations_100m"]):
        if r["metric"] in {"mean_spatial_information_bits", "active_map_cosine_mean"}:
            add_pair(pairs, "same_observations_100m", r["metric"], r["encoder_credit"], int(r["seed"]), float(r["temporal"]), float(r["constant"]))
    for source, metrics in (
        ("trajectory_100m", ("visited_cell_fraction", "path_efficiency")),
        ("stored_graph_100m", ("prospective_success_fraction", "graph_reachable_pair_fraction")),
    ):
        for r in rows(sources[source]):
            if int(r["target_env_steps"]) != 100_000_000:
                continue
            for metric in metrics:
                add_pair(pairs, source, metric, r["encoder_credit"], int(r["seed"]),
                         float(r[f"temporal_{metric}"]), float(r[f"constant_{metric}"]))

    behavior_text = subprocess.check_output(
        ["git", "show", f"{BEHAVIOR_COMMIT}:{BEHAVIOR_PATH}"], cwd=ROOT, text=True
    )
    behavior_rows = list(csv.DictReader(io.StringIO(behavior_text)))
    behavior = {(r["encoder_credit"], r["worker_bonus"], int(r["training_seed"])): r
                for r in behavior_rows if r["cohort"] == "fresh" and r["family"] == "C05" and r["layer"] == "decoder_1"}
    assert set(behavior) == set(identities)
    for cell, r in behavior.items():
        identity = identities[cell]
        assert (r["study_id"], r["study_sha256"], r["workflow_version"]) == (
            identity["study_id"], identity["study_sha256"], identity["workflow_version"]
        )
    for encoder in ("temporal", "constant"):
        for seed in sorted(SEEDS):
            t, c = (behavior[(encoder, bonus, seed)] for bonus in ("temporal", "constant"))
            assert t["completed_eval_repeats"] == c["completed_eval_repeats"] == "2"
            for metric in ("information_difference", "common_split_half_difference"):
                add_pair(pairs, "frozen_behavior_100m", metric, encoder, seed,
                         float(t[metric]), float(c[metric]))

    metrics = sorted({(source, metric) for source, metric, _, _ in pairs})
    for source, metric in metrics:
        for encoder in ("temporal", "constant"):
            assert {seed for s, m, e, seed in pairs if (s, m, e) == (source, metric, encoder)} == SEEDS, (source, metric, encoder)

    OUT.mkdir(parents=True, exist_ok=True)
    paired_rows = []
    for (source, metric, encoder, seed), (temporal, constant) in sorted(pairs.items()):
        t, c = (identities[(encoder, bonus, seed)] for bonus in ("temporal", "constant"))
        paired_rows.append({"source": source, "metric": metric, "encoder_credit": encoder, "seed": seed,
                            "temporal_worker": temporal, "constant_worker": constant, "difference": temporal - constant,
                            "temporal_study_id": t["study_id"], "temporal_study_sha256": t["study_sha256"],
                            "constant_study_id": c["study_id"], "constant_study_sha256": c["study_sha256"],
                            "schema": t["schema"], "workflow_version": t["workflow_version"]})
    write_csv(OUT / "paired_by_seed.csv", paired_rows)

    summary = []
    for source, metric in metrics:
        arm = {}
        for encoder in ("temporal", "constant"):
            contrast = np.array([pairs[(source, metric, encoder, seed)][0] - pairs[(source, metric, encoder, seed)][1]
                                 for seed in sorted(SEEDS)])
            arm[encoder] = contrast
            summary.append({"source": source, "metric": metric, "contrast": f"worker_temporal_minus_constant_given_{encoder}",
                            "n_training_seed_pairs": 5, "mean": contrast.mean(), "sd": contrast.std(ddof=1),
                            "positive": int((contrast > 0).sum()), "zero": int((contrast == 0).sum())})
        interaction = arm["temporal"] - arm["constant"]
        summary.append({"source": source, "metric": metric, "contrast": "encoder_by_worker_interaction",
                        "n_training_seed_pairs": 5, "mean": interaction.mean(), "sd": interaction.std(ddof=1),
                        "positive": int((interaction > 0).sum()), "zero": int((interaction == 0).sum())})
    write_csv(OUT / "factor_summary.csv", summary)
    render(pairs)
    metadata = {"design": "five matched training seeds per C05 2x2 factorial cell",
                "seeds": sorted(SEEDS), "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()},
                "behavior_source_commit": BEHAVIOR_COMMIT, "behavior_source_path": BEHAVIOR_PATH,
                "behavior_source_sha256": hashlib.sha256(behavior_text.encode()).hexdigest(),
                "graph_definition": "stored prospective detector outcomes, never executed commands or physical arrivals"}
    (OUT / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def write_csv(path: Path, data: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def render(pairs: dict) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 24,
                         "axes.titlesize": 25, "axes.labelsize": 24,
                         "xtick.labelsize": 22, "ytick.labelsize": 22,
                         "svg.fonttype": "none"})
    groups = [
        ("worker_reward_exposure", [("correct_hit_90_100m", "correct_hit_observation_count", "Correct-hit observations, 90–100M"),
                                     ("correct_hit_90_100m", "correct_elapsed_count_weighted", "Correct-hit elapsed decisions"),
                                     ("online_90_99m", "visited_cell_fraction", "Online visited fraction, 90–99M")]),
        ("worker_representation_graph", [("same_observations_100m", "mean_spatial_information_bits", "DG spatial information on shared panel (bits)"),
                                          ("stored_graph_100m", "prospective_success_fraction", "Stored prospective detector-hit fraction"),
                                          ("frozen_behavior_100m", "information_difference", "Own − random decoder-1 information\n(bits/activation)")]),
        ("worker_trajectory_reliability", [("trajectory_100m", "visited_cell_fraction", "Retained 100k visited fraction"),
                                           ("stored_graph_100m", "graph_reachable_pair_fraction", "Stored graph reachable-pair fraction"),
                                           ("frozen_behavior_100m", "common_split_half_difference", "Own − random common-support reliability")]),
    ]
    for name, panels in groups:
        fig, axes = plt.subplots(3, 1, figsize=(10, 14), constrained_layout=True)
        for ax, (source, metric, title) in zip(axes, panels):
            for x, encoder, color, marker in ((0, "temporal", "#2b5c8a", "o"), (1, "constant", "#b06036", "s")):
                values = np.array([pairs[(source, metric, encoder, seed)][0] - pairs[(source, metric, encoder, seed)][1]
                                   for seed in sorted(SEEDS)])
                ax.scatter(np.full(5, x) + np.linspace(-.12, .12, 5), values,
                           s=75, color=color, marker=marker, zorder=3)
                ax.plot([x - .23, x + .23], [values.mean()] * 2, color="black", lw=3)
            ax.axhline(0, color="#777777", lw=1)
            ax.set_xlim(-.5, 1.5)
            ax.set_xticks([0, 1], ["Temporal encoder", "Constant encoder"])
            ax.set_ylabel("T − C worker")
            ax.set_title(title, loc="left")
            ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
        fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
