"""Factorial, seed-paired analysis of frozen common-panel layer information.

Reads compact compute-node summaries and the canonical training catalog. Missing
training or replay cells stay missing; no model or evaluation repeat is imputed.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from hpc_runs.intrmotiv_study.spec import load_study


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "06_experiments/results"
DEFAULT_INPUTS = (
    RESULTS / "cosyne_2027_common_panel_layers/raw",
    RESULTS / "cosyne_2027_temporal_credit_layers/raw",
)
DEFAULT_OUTPUT = RESULTS / "cosyne_2027_temporal_credit_layers"
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
PANEL_SHA256 = "14e977ff526c71d04822dde8903e4d1eeda8ed44d16b8eb1a9b7d69b426fc54d"


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def training_catalog() -> tuple[dict, dict]:
    path = ROOT / "06_experiments/cosyne_2027/training_runs.csv"
    selected = {}
    specs = {}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row["cohort"] not in ("cosyne", "cosyne_replacement", "cosyne_new"):
            continue
        if row["family"] not in ("C01", "C05", "C15") or row["encoder_credit"] not in ("temporal", "constant"):
            continue
        if row["attempt_status"] != "verified_100m":
            continue
        key = (row["family"], row["encoder_credit"], row["worker_bonus"], int(row["seed"]))
        if key in selected:
            raise ValueError(f"Two verified training attempts for {key}")
        spec_path = ROOT / row["study_path"]
        spec = load_study(spec_path)
        if (spec.fingerprint != row["study_sha256"] or spec.raw["schema"] != row["schema"]
                or spec.raw["workflow_version"] != row["workflow_version"]):
            raise ValueError(f"StudySpec identity changed: {spec_path}")
        specs[row["study_id"]] = {
            "path": row["study_path"], "schema": row["schema"],
            "workflow_version": row["workflow_version"], "sha256": row["study_sha256"],
        }
        selected[key] = row
    return selected, {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path),
                      "specs": specs}


def replay_summaries(input_dirs: tuple[Path, ...], training: dict) -> tuple[dict, list[dict]]:
    values = {}
    sources = []
    for folder in input_dirs:
        for path in sorted(folder.glob("*.json")):
            item = json.loads(path.read_text())
            if item.get("schema") != "intrmotiv/common-panel-layers/v1":
                raise ValueError(f"Unexpected replay schema: {path}")
            matching = [key for key, row in training.items()
                        if row["condition"] == item["condition"] and int(row["seed"]) == int(item["seed"])]
            if len(matching) != 1:
                raise ValueError(f"Replay lacks one verified catalog identity: {path}")
            key = matching[0]
            row = training[key]
            if key in values:
                raise ValueError(f"Duplicate replay for {key}")
            if (item["checkpoint"] != row["run_dir"] + "/checkpoint_p0/milestones/" +
                    Path(item["checkpoint"]).name or item["panel_sha256"] != PANEL_SHA256
                    or int(item["checkpoint_frames"]) < 100_000_000
                    or abs(item["dg_replay_mean_amplitude_weighted_bits"] -
                           item["dg_reference_mean_amplitude_weighted_bits"]) > 1e-4
                    or set(item["layers"]) != set(LAYERS)):
                raise ValueError(f"Replay panel, checkpoint, or DG parity failed: {path}")
            for layer in LAYERS:
                details = item["layers"][layer]
                if details["visited_bins"] != 242 or details["observations_in_bounds"] != 10_001:
                    raise ValueError(f"Spatial panel mismatch: {path} {layer}")
            values[key] = item
            sources.append({"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path),
                            "checkpoint_sha256": item["checkpoint_sha256"],
                            "dg_reference_sha256": item["dg_reference_sha256"]})
    return values, sources


def contrast_definitions() -> list[tuple[str, str, str, tuple[str, str], tuple[str, str]]]:
    """Return architecture, contrast, fixed arm, and temporal/constant cells."""
    result = [("C01", "encoder", "inapplicable", ("temporal", "inapplicable"),
               ("constant", "inapplicable"))]
    for family in ("C05", "C15"):
        for worker in ("temporal", "constant"):
            result.append((family, "encoder", worker, ("temporal", worker), ("constant", worker)))
        for encoder in ("temporal", "constant"):
            result.append((family, "worker", encoder, (encoder, "temporal"), (encoder, "constant")))
    return result


def aggregate(values: dict, training: dict) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    absolute, paired, summary, availability = [], [], [], []
    for key, item in sorted(values.items()):
        family, encoder, worker, seed = key
        source = training[key]
        for layer in LAYERS:
            absolute.append({"family": family, "encoder_credit": encoder,
                             "worker_bonus": worker, "seed": seed, "layer": layer,
                             "study_id": source["study_id"], "study_sha256": source["study_sha256"],
                             "training_job_id": source["job_id"],
                             "checkpoint_frames": item["checkpoint_frames"],
                             "checkpoint_sha256": item["checkpoint_sha256"],
                             **item["layers"][layer]})

    def append_contrast(family: str, factor: str, fixed: str, seed: int,
                        layer: str, metric: str, a_key: tuple, b_key: tuple,
                        label: str) -> None:
        if a_key not in values or b_key not in values:
            return
        a_item, b_item = values[a_key], values[b_key]
        if a_item["checkpoint_frames"] != b_item["checkpoint_frames"]:
            return  # Matched age is a condition of the absolute comparison.
        a = a_item["layers"][layer][metric]
        b = b_item["layers"][layer][metric]
        if a is None or b is None:
            return
        paired.append({"family": family, "factor": factor, "fixed_arm": fixed,
                       "contrast": label, "seed": seed, "layer": layer, "metric": metric,
                       "temporal": a, "constant": b, "difference": float(a - b),
                       "checkpoint_frames": a_item["checkpoint_frames"]})

    for family, factor, fixed, first, second in contrast_definitions():
        for seed in SEEDS:
            for layer in LAYERS:
                for metric in METRICS:
                    append_contrast(family, factor, fixed, seed, layer, metric,
                                    (family, *first, seed), (family, *second, seed),
                                    f"{factor}_temporal_minus_constant")
    for family in ("C05", "C15"):
        for seed in SEEDS:
            cells = [(family, e, w, seed) for e in ("temporal", "constant")
                     for w in ("temporal", "constant")]
            if any(key not in values for key in cells) or len({values[key]["checkpoint_frames"] for key in cells}) != 1:
                continue
            for layer in LAYERS:
                for metric in METRICS:
                    t_t, t_c, c_t, c_c = [values[key]["layers"][layer][metric] for key in cells]
                    if any(value is None for value in (t_t, t_c, c_t, c_c)):
                        continue
                    paired.append({"family": family, "factor": "interaction", "fixed_arm": "none",
                                   "contrast": "encoder_by_worker", "seed": seed, "layer": layer,
                                   "metric": metric, "temporal": "", "constant": "",
                                   "difference": float((t_t - t_c) - (c_t - c_c)),
                                   "checkpoint_frames": values[cells[0]]["checkpoint_frames"]})
    for family, factor, fixed, _first, _second in contrast_definitions() + [
        (family, "interaction", "none", ("", ""), ("", "")) for family in ("C05", "C15")]:
        for layer in LAYERS:
            for metric in METRICS:
                selected = [row for row in paired if row["family"] == family and row["factor"] == factor
                            and row["fixed_arm"] == fixed and row["layer"] == layer and row["metric"] == metric]
                differences = np.asarray([row["difference"] for row in selected], dtype=float)
                present = {row["seed"] for row in selected}
                summary.append({"family": family, "factor": factor, "fixed_arm": fixed,
                                "layer": layer, "metric": metric, "n": len(differences),
                                "missing_seeds": ",".join(map(str, sorted(set(SEEDS) - present))),
                                "mean_difference": float(differences.mean()) if len(differences) else "",
                                "sd_difference": float(differences.std(ddof=1)) if len(differences) > 1 else "",
                                "positive": int((differences > 0).sum()),
                                "negative": int((differences < 0).sum()),
                                "ties": int((differences == 0).sum())})
    for family in ("C01", "C05", "C15"):
        workers = ("inapplicable",) if family == "C01" else ("temporal", "constant")
        for encoder in ("temporal", "constant"):
            for worker in workers:
                present = sorted(seed for seed in SEEDS if (family, encoder, worker, seed) in values)
                trained = sorted(seed for seed in SEEDS if (family, encoder, worker, seed) in training)
                availability.append({"family": family, "encoder_credit": encoder,
                                     "worker_bonus": worker, "planned_seeds": 5,
                                     "verified_training_seeds": ",".join(map(str, trained)),
                                     "common_panel_seeds": ",".join(map(str, present)),
                                     "missing_panel_seeds": ",".join(map(str, sorted(set(SEEDS) - set(present))))})
    return absolute, paired, summary, availability


def plot_pairs(paired: list[dict], output: Path, factor: str) -> None:
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = Path(font_manager.findfont("DejaVu Sans", fallback_to_default=False))
    if font.suffix.lower() not in (".ttf", ".otf") or not font.is_file():
        raise RuntimeError("No verified scalable font")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16,
                         "axes.titlesize": 18, "axes.labelsize": 16,
                         "xtick.labelsize": 14, "ytick.labelsize": 14})
    groups = ([ ("C01", "inapplicable", "C01"), ("C05", "temporal", "C05\nworker T"),
                ("C05", "constant", "C05\nworker C"), ("C15", "temporal", "C15\nworker T"),
                ("C15", "constant", "C15\nworker C") ] if factor == "encoder" else
              [ ("C05", "temporal", "C05\nencoder T"), ("C05", "constant", "C05\nencoder C"),
                ("C15", "temporal", "C15\nencoder T"), ("C15", "constant", "C15\nencoder C") ])
    metrics = (("mean_bits_per_activation_eligible", "Bits per activation\n(eligible units)"),
               ("mean_amplitude_weighted_bits_all", "Amplitude-weighted bits per step\n(all units)"))
    colors = {"C01": "#4263A1", "C05": "#B06B23", "C15": "#33835D"}
    for layer in LAYERS:
        fig, axes = plt.subplots(2, 1, figsize=(9, 8))
        for row_index, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_index]
            ax.axhline(0, color="0.55", linewidth=1)
            tick_labels = []
            for x, (family, fixed, label) in enumerate(groups):
                selected = sorted((r for r in paired if r["factor"] == factor
                                   and r["family"] == family and r["fixed_arm"] == fixed
                                   and r["layer"] == layer and r["metric"] == metric),
                                  key=lambda r: r["seed"])
                if selected:
                    vals = np.asarray([r["difference"] for r in selected])
                    offsets = np.linspace(-0.16, 0.16, len(vals))
                    ax.scatter(x + offsets, vals, s=65, color=colors[family], zorder=3)
                    ax.plot([x - .24, x + .24], [vals.mean()] * 2,
                            color="black", linewidth=2.3, zorder=4)
                    tick_labels.append(f"{label}\n{len(vals)}/5 pairs")
                else:
                    ax.axvspan(x - .35, x + .35, color="0.96", zorder=0)
                    tick_labels.append(f"{label}\npending")
            ax.set_xlim(-.5, len(groups) - .5)
            ax.set_xticks(range(len(groups)), tick_labels)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=.15)
        action = "Encoder credit" if factor == "encoder" else "Worker hit bonus"
        layer_name = {"ca3": "CA3", "decoder_1": "Decoder 1", "decoder_2": "Decoder 2"}[layer]
        fig.suptitle(f"{layer_name}: {action} temporal − constant", fontsize=20, y=.98)
        fig.subplots_adjust(left=.20, right=.98, top=.90, bottom=.10, hspace=.55)
        for extension in ("svg", "png"):
            path = output / f"{factor}_paired_{layer}.{extension}"
            fig.savefig(path, dpi=120)
            if extension == "svg":
                # Matplotlib leaves trailing spaces on path-data lines. Keep
                # committed figures clean without changing rendered geometry.
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", help="Directory with per-run JSON files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    inputs = tuple(args.input) if args.input else DEFAULT_INPUTS
    training, catalog_source = training_catalog()
    values, sources = replay_summaries(inputs, training)
    absolute, paired, summary, availability = aggregate(values, training)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "absolute_by_seed.csv", absolute)
    write_csv(args.output / "paired_by_seed.csv", paired)
    write_csv(args.output / "factor_summary.csv", summary)
    write_csv(args.output / "availability.csv", availability)
    plot_pairs(paired, args.output, "encoder")
    plot_pairs(paired, args.output, "worker")
    import matplotlib
    from matplotlib import font_manager
    font_path = Path(font_manager.findfont("DejaVu Sans", fallback_to_default=False))
    figure_files = [args.output / f"{factor}_paired_{layer}.{suffix}"
                    for factor in ("encoder", "worker") for layer in LAYERS
                    for suffix in ("svg", "png")]
    manifest = {"schema": "cosyne27-temporal-credit-layers/v1",
                "protocol": "frozen terminal checkpoints replayed on one saved 10001-observation panel",
                "panel_sha256": PANEL_SHA256, "catalog": catalog_source, "replays": sources,
                "source_script_sha256": file_sha256(Path(__file__)),
                "summary_sha256": file_sha256(args.output / "factor_summary.csv"),
                "figure_specification": {"size_inches": [9, 8], "png_dpi": 120,
                                         "font_path": str(font_path),
                                         "matplotlib_version": matplotlib.__version__,
                                         "plots": [{"path": str(path.relative_to(ROOT)),
                                                    "sha256": file_sha256(path)}
                                                   for path in figure_files]},
                "metric_definitions": {
                    "mean_bits_per_activation_eligible": "Occupancy-weighted spatial score divided by mean rectified activity per eligible unit, then mean across eligible units",
                    "mean_amplitude_weighted_bits_all": "Original amplitude-weighted bits per step, mean over all units including silent units",
                    "mean_positive_fraction": "Fraction of positive rectified layer samples",
                    "mean_rectified_activation": "Mean positive part of the raw layer trace",
                    "eligibility": "At least 20 positive observations in three spatial bins",
                }, "matched_age_required": True,
                "planned_seed_ids": SEEDS, "source_count": len(sources)}
    (args.output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Analyzed {len(sources)} verified replays; {len(paired)} paired metric rows")


if __name__ == "__main__":
    main()
