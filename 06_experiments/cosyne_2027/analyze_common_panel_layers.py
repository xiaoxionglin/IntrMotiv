"""Aggregate source-verified C05 common-panel CA3 and decoder replay summaries."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hpc_runs.intrmotiv_study.spec import load_study


DEFAULT_INPUT = ROOT / "06_experiments/results/cosyne_2027_common_panel_layers/raw"
DEFAULT_OUTPUT = ROOT / "06_experiments/results/cosyne_2027_common_panel_layers"
SEEDS = (8, 17, 41, 99, 123)
LAYERS = ("ca3", "decoder_1", "decoder_2")
METRICS = (
    "mean_bits_per_activation_eligible",
    "mean_amplitude_weighted_bits_all",
    "mean_positive_fraction",
    "mean_rectified_activation",
    "eligible_units",
    "silent_units",
)
PLOT_METRICS = METRICS[:2]
PANEL_SHA256 = "14e977ff526c71d04822dde8903e4d1eeda8ed44d16b8eb1a9b7d69b426fc54d"


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_sources(input_dir: Path) -> tuple[dict, dict]:
    training_path = ROOT / "06_experiments/cosyne_2027/training_runs.csv"
    training = [r for r in csv.DictReader(training_path.open())
                if r["family"] == "C05" and r["cohort"] == "cosyne"]
    if len(training) != 20:
        raise ValueError("Expected 20 fresh original C05 runs")
    identities = {}
    specs = {}
    for row in training:
        key = (row["encoder_credit"], row["worker_bonus"], int(row["seed"]))
        if key in identities or row["attempt_status"] != "verified_100m":
            raise ValueError(f"Duplicate or unverified training identity: {key}")
        identities[key] = row
        path = ROOT / row["study_path"]
        spec = load_study(path)
        if (spec.fingerprint != row["study_sha256"] or spec.raw["schema"] != row["schema"]
                or spec.raw["workflow_version"] != row["workflow_version"]):
            raise ValueError(f"StudySpec fingerprint or version changed: {path}")
        specs[row["study_id"]] = {"path": row["study_path"], "sha256": spec.fingerprint,
                                  "schema": spec.raw["schema"],
                                  "workflow_version": spec.raw["workflow_version"]}
    expected = {(encoder, bonus, seed) for encoder in ("temporal", "constant")
                for bonus in ("temporal", "constant") for seed in SEEDS}
    if set(identities) != expected:
        raise ValueError("Incomplete C05 factorial identities")
    files = sorted(input_dir.glob("*.json"))
    if len(files) != 20:
        raise ValueError(f"Expected 20 replay summaries, found {len(files)}")
    observations = {}
    input_sources = []
    for path in files:
        item = json.loads(path.read_text())
        condition = item["condition"]
        parts = condition.removeprefix("COS27_C05_").split("_")
        if len(parts) != 2 or item["schema"] != "intrmotiv/common-panel-layers/v1":
            raise ValueError(f"Invalid condition or schema: {path}")
        key = (parts[0].lower(), parts[1].lower(), int(item["seed"]))
        if key not in identities or key in observations:
            raise ValueError(f"Unexpected or repeated replay: {key}")
        identity = identities[key]
        if (condition != identity["condition"] or item["checkpoint_frames"] != 100_007_936
                or item["checkpoint"] != identity["run_dir"] + "/checkpoint_p0/milestones/checkpoint_000006104_100007936.pth"
                or item["panel_sha256"] != PANEL_SHA256
                or abs(item["dg_replay_mean_amplitude_weighted_bits"] -
                       item["dg_reference_mean_amplitude_weighted_bits"]) > 1e-4
                or set(item["layers"]) != set(LAYERS)):
            raise ValueError(f"Replay provenance or DG alignment failed: {path}")
        for layer in LAYERS:
            value = item["layers"][layer]
            if value["visited_bins"] != 242 or value["observations_in_bounds"] != 10_001:
                raise ValueError(f"Spatial support differs: {path} {layer}")
        observations[key] = item
        input_sources.append({"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path),
                              "checkpoint_sha256": item["checkpoint_sha256"],
                              "dg_reference_sha256": item["dg_reference_sha256"]})
    if set(observations) != expected:
        raise ValueError("Incomplete C05 replay factorial")
    sources = {"training_catalog": {"path": str(training_path.relative_to(ROOT)),
                                    "sha256": file_sha256(training_path)},
               "study_specs": specs, "replays": input_sources}
    return observations, sources


def aggregate(observations: dict) -> tuple[list[dict], list[dict], list[dict]]:
    absolute, paired, summary = [], [], []
    for (encoder, bonus, seed), item in sorted(observations.items()):
        for layer in LAYERS:
            absolute.append({"encoder_credit": encoder, "worker_bonus": bonus,
                             "seed": seed, "layer": layer, **item["layers"][layer]})
    for layer in LAYERS:
        for metric in METRICS:
            for contrast, fixed in (("worker_temporal_minus_constant", "encoder_credit"),
                                    ("encoder_temporal_minus_constant", "worker_bonus")):
                for fixed_value in ("temporal", "constant"):
                    differences = []
                    for seed in SEEDS:
                        if contrast.startswith("worker"):
                            first = observations[(fixed_value, "temporal", seed)]
                            second = observations[(fixed_value, "constant", seed)]
                        else:
                            first = observations[("temporal", fixed_value, seed)]
                            second = observations[("constant", fixed_value, seed)]
                        a = first["layers"][layer][metric]
                        b = second["layers"][layer][metric]
                        if a is None or b is None:
                            raise ValueError(f"Undefined metric: {layer} {metric} {seed}")
                        difference = float(a - b)
                        paired.append({"layer": layer, "metric": metric, "contrast": contrast,
                                       "fixed_factor": fixed, "fixed_value": fixed_value,
                                       "seed": seed, "temporal": a, "constant": b,
                                       "difference": difference})
                        differences.append(difference)
                    summary.append(summarize_differences(layer, metric, contrast,
                                                         fixed, fixed_value, differences))
            interaction = []
            for seed in SEEDS:
                cell = {
                    (encoder, worker): observations[(encoder, worker, seed)]["layers"][layer][metric]
                    for encoder in ("temporal", "constant")
                    for worker in ("temporal", "constant")
                }
                difference = float(
                    cell["temporal", "temporal"] - cell["temporal", "constant"]
                    - cell["constant", "temporal"] + cell["constant", "constant"]
                )
                paired.append({"layer": layer, "metric": metric,
                               "contrast": "worker_by_encoder_interaction",
                               "fixed_factor": "none", "fixed_value": "none", "seed": seed,
                               "temporal": "", "constant": "", "difference": difference})
                interaction.append(difference)
            summary.append(summarize_differences(layer, metric,
                                                  "worker_by_encoder_interaction",
                                                  "none", "none", interaction))
    return absolute, paired, summary


def summarize_differences(layer: str, metric: str, contrast: str, fixed: str,
                          value: str, differences: list[float]) -> dict:
    values = np.asarray(differences)
    return {"layer": layer, "metric": metric, "contrast": contrast,
            "fixed_factor": fixed, "fixed_value": value, "n": len(values),
            "mean_difference": float(values.mean()), "sd_difference": float(values.std(ddof=1)),
            "positive": int(np.count_nonzero(values > 0)),
            "negative": int(np.count_nonzero(values < 0)),
            "ties": int(np.count_nonzero(values == 0))}


def figures(absolute: list[dict], paired: list[dict], output: Path) -> str:
    font_path = Path(font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans"), fallback_to_default=False
    ))
    if not font_path.is_file() or font_path.suffix.lower() not in {".ttf", ".otf"}:
        raise RuntimeError("No verified scalable DejaVu Sans font")
    plt.rcParams["font.family"] = "DejaVu Sans"
    labels = {"ca3": "CA3", "decoder_1": "Decoder 1", "decoder_2": "Decoder 2"}
    colors = {"temporal": "#30363B", "constant": "#75838C"}
    for kind, data in (("absolute", absolute), ("worker_difference", paired)):
        # One layer per page-width image preserves readable text in Obsidian.
        with plt.rc_context({"font.size": 19, "axes.titlesize": 21,
                             "axes.labelsize": 19, "xtick.labelsize": 18,
                             "ytick.labelsize": 18}):
            for layer in LAYERS:
                detail, detail_axes = plt.subplots(2, 1, figsize=(6.5, 9))
                detail.subplots_adjust(left=0.19, right=0.98, top=0.92,
                                       bottom=0.08, hspace=0.48)
                for row_index, metric in enumerate(PLOT_METRICS):
                    ax = detail_axes[row_index]
                    for i, encoder in enumerate(("temporal", "constant")):
                        bonus_levels = ("constant", "temporal") if kind == "absolute" else ("temporal",)
                        for j, bonus in enumerate(bonus_levels):
                            if kind == "absolute":
                                group = sorted((r for r in data if r["layer"] == layer and
                                                r["encoder_credit"] == encoder and
                                                r["worker_bonus"] == bonus),
                                               key=lambda r: r["seed"])
                                values = np.asarray([r[metric] for r in group], dtype=float)
                                x = i * 3 + j
                                marker = "o" if bonus == "temporal" else "s"
                                fill = colors[encoder] if bonus == "temporal" else "white"
                            else:
                                group = sorted((r for r in data if r["layer"] == layer and
                                                r["metric"] == metric and
                                                r["contrast"] == "worker_temporal_minus_constant" and
                                                r["fixed_value"] == encoder),
                                               key=lambda r: r["seed"])
                                values = np.asarray([r["difference"] for r in group], dtype=float)
                                x = i
                                marker, fill = ("o", colors[encoder]) if i == 0 else ("s", "white")
                            jitter = np.linspace(-0.16, 0.16, 5)
                            ax.scatter(x + jitter, values, marker=marker, s=145,
                                       facecolor=fill, edgecolor=colors[encoder],
                                       linewidth=2.2, zorder=3)
                            ax.plot([x - 0.24, x + 0.24], [values.mean()] * 2,
                                    color="black", linewidth=3.5, zorder=4)
                    if kind == "absolute":
                        ax.set_xticks([0.5, 3.5], ["Encoder T", "Encoder C"])
                        ax.set_xlabel("Worker: □ constant  ● temporal")
                        maximum = max(float(r[metric]) for r in data if r["layer"] == layer)
                        ax.set_ylim(0, maximum * 1.12)
                    else:
                        ax.axhline(0, color="#BBC4CA", linewidth=1.5)
                        ax.set_xticks([0, 1], ["Encoder T", "Encoder C"])
                        ax.set_xlabel("Worker bonus: T − C")
                    ax.set_ylabel("Bits / activation" if row_index == 0 else
                                  "Weighted spatial score")
                    ax.grid(axis="y", alpha=0.16)
                detail.suptitle(f"C05 {labels[layer]} · fixed panel",
                                fontsize=22)
                for suffix in ("svg", "png"):
                    path = output / f"{kind}_{layer}.{suffix}"
                    detail.savefig(path, dpi=170)
                    if suffix == "svg":
                        # Matplotlib emits trailing spaces in glyph paths.
                        path.write_text("\n".join(line.rstrip() for line in
                                                  path.read_text().splitlines()) + "\n")
                plt.close(detail)
    return str(font_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    observations, sources = verify_sources(args.input_dir)
    absolute, paired, summary = aggregate(observations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "absolute_by_seed.csv", absolute)
    write_csv(args.output_dir / "paired_by_seed.csv", paired)
    write_csv(args.output_dir / "factor_summary.csv", summary)
    font_path = figures(absolute, paired, args.output_dir)
    manifest = {"schema": "cosyne27-common-panel-layers-analysis/v1",
                "protocol": "frozen C05 terminal checkpoints, common 10001-observation panel",
                "panel_sha256": PANEL_SHA256, "sources": sources,
                "source_code": [
                    {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
                    for path in (
                        ROOT / "hpc_runs/common_panel_layer_probe.py",
                        ROOT / "hpc_runs/run_common_panel_layer_single.sh",
                        ROOT / "hpc_runs/submit_common_panel_layers.py",
                        Path(__file__),
                    )
                ],
                "seed_count": 5, "factor_cells": 4, "layers": LAYERS,
                "figure_font_path": font_path,
                "figure_detail_size_inches": [6.5, 9],
                "figure_detail_dpi": 170,
                "metric_definitions": {
                    "mean_bits_per_activation_eligible": "occupancy-weighted spatial information divided by mean rectified activity, averaged over units active at least 20 observations in 3 bins",
                    "mean_amplitude_weighted_bits_all": "original Skaggs amplitude-weighted score averaged over all layer units; zeros retained",
                    "mean_positive_fraction": "fraction of rectified layer samples greater than zero",
                }}
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
