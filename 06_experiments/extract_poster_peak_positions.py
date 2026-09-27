"""Extract DG peak coordinates from existing frozen place-field archives.

This reads the established evaluator NPZ contract. It does not run an
environment or checkpoint. One row is one DG unit in one frozen run.
"""

from __future__ import annotations

import argparse
import csv
import io
import tarfile
from pathlib import Path

import numpy as np


def archive_suite(condition: str) -> str:
    if condition.startswith(("SAT_", "CPU2048_") ) and condition.endswith("_DDQN_HER"):
        return "frozen"
    if condition.startswith(("SAT_", "DGP_")):
        return "frozen"
    if condition == "DGC_DIRECT_WORKER_F16":
        return "dgc_direct_latest_frozen"
    if condition == "DGC_WAYPOINT_DG_F64":
        return "dgc_waypoint_frozen"
    if condition.startswith("CR5C_D50_"):
        return "d50_75m_latest_frozen"
    if condition.startswith("CR5C_D51_"):
        return "d51_75m_frozen"
    return "mature_frozen"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--analysis-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--figure-tar", type=Path,
                        help="Copy existing trajectory, flow, and graph PNGs for selected runs")
    parser.add_argument("--field-tar", type=Path,
                        help="Save compact top-four active DG maps for each frozen run")
    parser.add_argument("--behavior-output", type=Path,
                        help="Collect existing frozen trajectory/flow summaries")
    args = parser.parse_args()
    with args.manifest.open(newline="") as stream:
        manifest = list(csv.DictReader(stream, delimiter="\t"))
    roles = {"endpoint", "endpoint_unmatched_age", "endpoint_exemplar"}
    rows = []
    figure_paths = []
    field_payloads = []
    behavior_rows = []
    behavior_by_suite = {}
    for run in manifest:
        if run["analysis_role"] not in roles:
            continue
        suite = archive_suite(run["condition"])
        if args.behavior_output is not None:
            if suite not in behavior_by_suite:
                summary = args.analysis_root / suite / "figures/behavior_graph_summary.csv"
                with summary.open(newline="") as stream:
                    behavior_by_suite[suite] = list(csv.DictReader(stream))
            matches = [row for row in behavior_by_suite[suite]
                       if row["label"].rsplit("__", 1)[-1] == run["label_suffix"]]
            if len(matches) != 1:
                raise ValueError(f"Expected one behavior row for {run['label_suffix']}: {matches}")
            behavior_rows.append({"condition": run["condition"], "seed": run["seed"],
                                  "checkpoint_frames": run["checkpoint_frames"],
                                  "label_suffix": run["label_suffix"], **matches[0]})
        archives = list((args.analysis_root / suite / "raw").glob(
            f"*{run['label_suffix']}/place_fields.npz"))
        if len(archives) != 1:
            raise ValueError(f"Expected one frozen archive for {run['label_suffix']}: {archives}")
        if args.figure_tar is not None:
            figures = list((args.analysis_root / suite / "figures").glob(
                f"*{run['label_suffix']}"))
            if len(figures) != 1:
                raise ValueError(f"Expected one frozen figure folder for {run['label_suffix']}: {figures}")
            for name in ("occupancy_trajectory.png", "occupancy_flow.png", "stored_graph.png"):
                path = figures[0] / name
                if not path.is_file():
                    raise FileNotFoundError(path)
                figure_paths.append((path, f"{run['label_suffix']}/{name}"))
        with np.load(archives[0], allow_pickle=False) as data:
            peaks = data["raw_dg_field_dominant_peak_bin"]
            activity = data["raw_dg_active_fraction"]
            eligible = data["raw_dg_field_eligible"]
            mono = data["raw_dg_field_mono"]
            information = data["raw_dg_spatial_information"]
            if args.field_tar is not None:
                active = np.flatnonzero((activity > 0) & np.isfinite(information))
                if len(active) < 4:
                    raise ValueError(f"Fewer than four active DG units: {archives[0]}")
                selected = active[np.argsort(information[active])[-4:][::-1]]
                buffer = io.BytesIO()
                np.savez_compressed(buffer, maps=data["rate_maps"][:, :, selected],
                                    occupancy=data["occupancy"], unit_ids=selected,
                                    spatial_information_bits=information[selected])
                field_payloads.append((f"{run['label_suffix']}.npz", buffer.getvalue()))
            if peaks.shape != (len(activity), 2):
                raise ValueError(f"Peak and activity dimensions disagree: {archives[0]}")
            for unit, (x_bin, y_bin) in enumerate(peaks):
                rows.append({
                    "condition": run["condition"], "seed": run["seed"],
                    "checkpoint_frames": run["checkpoint_frames"],
                    "analysis_role": run["analysis_role"],
                    "label_suffix": run["label_suffix"], "unit": unit,
                    "peak_x_bin": int(x_bin), "peak_y_bin": int(y_bin),
                    "active_fraction": float(activity[unit]),
                    "field_eligible": bool(eligible[unit]),
                    "mono_field": bool(mono[unit]),
                    "spatial_information_bits": float(information[unit]),
                    "frozen_archive": str(archives[0]),
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.figure_tar is not None:
        args.figure_tar.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.figure_tar, "w:gz") as archive:
            for source, label in figure_paths:
                archive.add(source, arcname=label)
    if args.field_tar is not None:
        args.field_tar.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.field_tar, "w:gz") as archive:
            for label, payload in field_payloads:
                metadata = tarfile.TarInfo(label)
                metadata.size = len(payload)
                archive.addfile(metadata, io.BytesIO(payload))
    if args.behavior_output is not None:
        args.behavior_output.parent.mkdir(parents=True, exist_ok=True)
        with args.behavior_output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(behavior_rows[0]))
            writer.writeheader()
            writer.writerows(behavior_rows)
    print(f"Extracted {len(rows)} DG units from {len({r['label_suffix'] for r in rows})} frozen runs")


if __name__ == "__main__":
    main()
