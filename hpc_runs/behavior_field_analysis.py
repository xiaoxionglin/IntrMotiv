"""Summarize frozen-policy field expression from completed 50k probe traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from hpc_runs.behavior_field_metrics import (
    LAYERS, PREFIXES, comparison_details, layer_details, paired_information,
    split_half_reliability,
)
from hpc_runs.behavior_field_study import read_manifest


def single_summary(trace: dict[str, np.ndarray], layer: str, details: dict[str, np.ndarray]) -> dict:
    eligible = details["field_eligible"].astype(bool)
    info = details["spatial_information"]
    peaks = details["field_dominant_peak_bin"][eligible]
    peaks = peaks[(peaks >= 0).all(axis=1)]
    maps = details["rate_maps"][:, :, eligible]
    occupied = details["occupancy"] > 0
    flattened = maps[occupied].T.astype(np.float64)
    norms = np.linalg.norm(flattened, axis=1)
    flattened = flattened[norms > 0] / norms[norms > 0, None]
    cosine = None
    if len(flattened) >= 2:
        matrix = flattened @ flattened.T
        cosine = float(matrix[np.triu_indices(len(flattened), 1)].mean())
    return {
        "layer": layer,
        "units": int(trace[layer].shape[1]),
        "eligible_units": int(eligible.sum()),
        "silent_units": int((details["active_fraction"] == 0).sum()),
        "amplitude_weighted_spatial_score": float(info[eligible].mean()) if eligible.any() else None,
        "mono_fields": int(details["field_mono"].sum()),
        "mean_dominant_mass_fraction": float(details["field_dominant_mass_fraction"][:, eligible].mean()) if eligible.any() else None,
        "mean_component_count": float(details["field_component_count"][:, eligible].mean()) if eligible.any() else None,
        "active_only_map_cosine": cosine,
        "distinct_peak_bins": int(np.unique(peaks, axis=0).shape[0]),
        "occupied_bins": int((details["occupancy"] > 0).sum()),
        "split_half_map_correlation": split_half_reliability(
            trace["pose"], trace[layer], trace["done"]
        ),
    }


def behavior_summary(trace: dict[str, np.ndarray]) -> dict:
    pose = trace["pose"]
    done = trace["done"].astype(bool)
    motion = np.linalg.norm(np.diff(pose[:, :2], axis=0), axis=1)
    motion = motion[~done[:-1]]
    actions = trace["executed_action"].astype(int)
    return {
        "episodes_completed": int(done.sum()),
        "mean_motion_per_decision": float(motion.mean()) if len(motion) else None,
        "stationary_fraction": float((motion < 1e-3).mean()) if len(motion) else None,
        "action_counts": json.dumps(np.bincount(actions).tolist()),
        "heading_quadrant_counts": json.dumps(np.bincount(
            np.floor(np.mod(pose[:, 2] + 180, 360) / 90).astype(int), minlength=4
        ).tolist()),
    }


def plot_maps(path: Path, trace: dict[str, np.ndarray], layer: str,
              details: dict[str, np.ndarray], title: str) -> None:
    import matplotlib.pyplot as plt

    information = details["spatial_information"]
    selected = np.argsort(information)[-min(8, len(information)):][::-1]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=160)
    for ax, unit in zip(axes.flat, selected):
        data = details["rate_maps"][:, :, unit].copy()
        data[details["occupancy"] == 0] = np.nan
        ax.imshow(data, origin="lower", cmap="viridis")
        ax.set_title(f"unit {unit}; score {information[unit]:.3f}", fontsize=13)
        ax.tick_params(labelsize=12)
    for ax in list(axes.flat)[len(selected):]:
        ax.set_axis_off()
    fig.suptitle(f"{title}: {layer} (highest eight amplitude-weighted scores)", fontsize=15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_trace(folder: Path, expected_policy: str) -> dict[str, np.ndarray]:
    metadata = json.loads((folder / "metadata.json").read_text())
    if metadata["decisions"] != 50_000 or metadata["policy"] != expected_policy:
        raise ValueError(f"Probe metadata mismatch: {folder}")
    with np.load(folder / "trace.npz", allow_pickle=False) as payload:
        trace = {key: payload[key] for key in payload.files}
    if len(trace["pose"]) != 50_000:
        raise ValueError(f"Probe length mismatch: {folder}")
    return trace


def plot_occupancy(path: Path, trace: dict[str, np.ndarray], title: str) -> None:
    import matplotlib.pyplot as plt

    pose = trace["pose"]
    counts, _, _ = np.histogram2d(pose[:, 0], pose[:, 1],
                                  bins=19, range=((100, 2000), (100, 2000)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=160)
    image = axes[0].imshow(counts.T, origin="lower", cmap="magma")
    fig.colorbar(image, ax=axes[0], label="decisions per bin")
    axes[0].set_title("Occupancy", fontsize=15)
    axes[1].plot(pose[::50, 0], pose[::50, 1], linewidth=0.7)
    axes[1].set_xlim(100, 2000)
    axes[1].set_ylim(100, 2000)
    axes[1].set_title("Subsampled trajectory", fontsize=15)
    for ax in axes:
        ax.tick_params(labelsize=12)
        ax.set_xlabel("x", fontsize=13)
        ax.set_ylabel("y", fontsize=13)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_prethreshold(path: Path, trace: dict[str, np.ndarray], title: str) -> None:
    import matplotlib.pyplot as plt

    pose = trace["pose"]
    logits = trace["dg_pre_threshold"]
    occupancy, _, _ = np.histogram2d(pose[:, 0], pose[:, 1], bins=19,
                                     range=((100, 2000), (100, 2000)))
    fig, axes = plt.subplots(4, 4, figsize=(14, 14), dpi=160)
    for unit, ax in enumerate(axes.flat):
        if unit >= logits.shape[1]:
            ax.set_axis_off()
            continue
        summed, _, _ = np.histogram2d(pose[:, 0], pose[:, 1], bins=19,
                                      range=((100, 2000), (100, 2000)), weights=logits[:, unit])
        rate = np.divide(summed, occupancy, out=np.full_like(summed, np.nan), where=occupancy > 0)
        maximum = max(float(np.nanmax(np.abs(rate))), 1e-6)
        ax.imshow(rate.T, origin="lower", cmap="coolwarm", vmin=-maximum, vmax=maximum)
        ax.set_title(f"DG pre-threshold {unit}", fontsize=13)
        ax.tick_params(labelsize=12)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_paired_summary(path: Path, comparisons: list[dict]) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in comparisons if row["random_policy"] == "uniform"
                and row["decisions"] == 50_000 and row["difference"] is not None]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), dpi=160, sharey=True)
    colors = {"C01": "#4263a1", "C05": "#b06b23", "C15": "#33835d"}
    for ax, layer in zip(axes, LAYERS):
        for x, condition in enumerate(colors):
            values = []
            for seed in ("8", "99", "123"):
                candidates = [row["difference"] for row in selected if
                              row["layer"] == layer and row["condition"] == condition
                              and row["seed"] == seed]
                if candidates:
                    values.append(float(np.mean(candidates)))
                    ax.plot(x, values[-1], "o", color=colors[condition], markersize=7)
            if values:
                ax.plot([x - .14, x + .14], [np.mean(values)] * 2,
                        color="black", linewidth=2)
        ax.axhline(0, color="0.45", linewidth=1)
        ax.set_xticks(range(3), list(colors), fontsize=12)
        ax.set_title(layer.replace("_", " ").upper(), fontsize=14)
        ax.tick_params(labelsize=12)
    axes[0].set_ylabel("own − uniform random (bits/activation)", fontsize=13)
    fig.suptitle("Frozen-policy field expression; dots are training seeds", fontsize=15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def condition_summaries(comparisons: list[dict]) -> list[dict]:
    """Average technical repeats inside each training seed before condition means."""
    result = []
    for condition in ("C01", "C05", "C15"):
        for policy in ("uniform", "persistent8"):
            for layer in LAYERS:
                seed_differences = []
                seed_late = []
                seed_support = []
                for seed in ("8", "99", "123"):
                    values = {}
                    for prefix in (40_000, 50_000):
                        subset = [row for row in comparisons if
                                  row["condition"] == condition and row["seed"] == seed
                                  and row["random_policy"] == policy and row["layer"] == layer
                                  and row["decisions"] == prefix and row["supported"]]
                        if len(subset) == 2:
                            values[prefix] = float(np.mean([row["difference"] for row in subset]))
                    if 50_000 in values:
                        seed_differences.append(values[50_000])
                        seed_support.append(seed)
                    if 40_000 in values and 50_000 in values:
                        seed_late.append(values[40_000] > 0 and values[50_000] > 0)
                complete = len(seed_differences) == 3
                result.append({
                    "condition": condition, "random_policy": policy, "layer": layer,
                    "supported_training_seeds": len(seed_differences),
                    "seed_ids": ",".join(seed_support),
                    "mean_difference": float(np.mean(seed_differences)) if seed_differences else None,
                    "min_seed_difference": float(min(seed_differences)) if seed_differences else None,
                    "max_seed_difference": float(max(seed_differences)) if seed_differences else None,
                    "positive_all_three": bool(complete and all(value > 0 for value in seed_differences)),
                    "positive_and_late_stable": bool(complete and len(seed_late) == 3
                                                     and all(seed_late) and all(value > 0 for value in seed_differences)),
                })
    return result


def analyze(manifest: Path, raw_root: Path, output: Path, workspace_root: Path) -> dict:
    rows = read_manifest(manifest, workspace_root, require_inputs=False)
    output.mkdir(parents=True, exist_ok=True)
    completed: dict[tuple[str, str, str, str], Path] = {}
    run_rows = []
    for row in rows:
        folder = raw_root / row["label"]
        if not (folder / "trace.npz").is_file() or not (folder / "metadata.json").is_file():
            continue
        trace = read_trace(folder, row["policy"])
        completed[(row["condition"], row["seed"], row["eval_seed"], row["policy"])] = folder
        plot_occupancy(output / f"{row['label']}_occupancy.png", trace, row["label"])
        plot_prethreshold(output / f"{row['label']}_prethreshold.png", trace, row["label"])
        layer_maps = {}
        for layer in LAYERS:
            details = layer_details(trace["pose"], trace[layer])
            layer_maps[layer] = details
            run_rows.append({"condition": row["condition"], "seed": row["seed"],
                             "eval_seed": row["eval_seed"], "policy": row["policy"],
                             **single_summary(trace, layer, details), **behavior_summary(trace)})
            plot_maps(output / f"{row['label']}_{layer}.png", trace, layer, details, row["label"])
        np.savez_compressed(output / f"{row['label']}_maps.npz", **{
            f"{layer}_{key}": details[key]
            for layer, details in layer_maps.items()
            for key in ("occupancy", "rate_maps", "smoothed_rate_maps", "spatial_information",
                        "field_eligible", "field_mono", "field_component_count",
                        "field_dominant_mass_fraction", "field_dominant_peak_bin")
        })
        del layer_maps, trace
    comparisons = []
    for condition in ("C01", "C05", "C15"):
        for seed in ("8", "99", "123"):
            for eval_seed in ("51000", "52000"):
                own_folder = completed.get((condition, seed, eval_seed, "own"))
                if own_folder is None:
                    continue
                own = read_trace(own_folder, "own")
                for policy in ("uniform", "persistent8"):
                    random_folder = completed.get((condition, seed, eval_seed, policy))
                    if random_folder is None:
                        continue
                    random = read_trace(random_folder, policy)
                    for prefix in PREFIXES:
                        for layer in LAYERS:
                            own_details = comparison_details(own["pose"][:prefix], own[layer][:prefix])
                            random_details = comparison_details(random["pose"][:prefix], random[layer][:prefix])
                            comparisons.append({
                                "condition": condition, "seed": seed, "eval_seed": eval_seed,
                                "random_policy": policy, "decisions": prefix, "layer": layer,
                                **paired_information(own_details, random_details),
                            })
                    del random
                del own
    write_csv(output / "per_run.csv", run_rows)
    write_csv(output / "paired_prefixes.csv", comparisons)
    write_csv(output / "condition_summary.csv", condition_summaries(comparisons))
    if comparisons:
        plot_paired_summary(output / "paired_layer_summary.png", comparisons)
    status = {"expected_probes": 54, "completed_probes": len(completed),
              "completed_pairs": len(comparisons) // (len(PREFIXES) * len(LAYERS)),
              "result_complete": len(completed) == 54}
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.manifest, args.raw_root, args.output, args.workspace_root)))


if __name__ == "__main__":
    main()
