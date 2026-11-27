"""Summarize and render a certified saved field archive; never run a model.

Historical arrays use [x,y,unit]. Figures transpose only for image orientation,
mask unvisited bins, and use scales computed across all sixteen units. Stored
graph counters are frozen training estimates, not executed control evidence.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
import numpy as np

from hpc_runs.intrmotiv_study import spatial_contract
from hpc_runs.intrmotiv_study.spatial_contract import _spatial_information, calculate_graph_diagnostics
from .provenance import audit_source_snapshots, snapshot_sources


ROOT = Path(__file__).resolve().parents[2]
SELECTED_UNITS = (0, 4, 8, 12)
# Resolve the exact scalable file Matplotlib uses for this requested family;
# fail instead of permitting a fallback, and archive its hash in the receipt.
FONT = Path(font_manager.findfont("DejaVu Sans", fallback_to_default=False))
CANONICAL_METRICS = ROOT / "06_experiments/analyze_place_field_manifest.py"


def digest(path):
    """Hash a small saved input without importing the model/evaluator stack."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diversity_helpers():
    spec = importlib.util.spec_from_file_location("field_reuse_canonical_metrics", CANONICAL_METRICS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pairwise_map_cosine, module.peak_statistics


def unit_summary(data, bounds):
    """Keep every unit, saved mono-field thresholds, and explicit denominators."""
    occupancy = data["post_inhibition_occupancy"]
    maps = data["post_inhibition_rate_maps"]
    if maps.shape != (19, 19, 16) or occupancy.shape != (19, 19) or occupancy.sum() <= 0:
        raise ValueError("Expected the certified 19x19, F16 field archive")
    if not np.array_equal(occupancy, data["occupancy"]):
        raise ValueError("Post-inhibition and original occupancy differ")
    records = []
    for unit in range(16):
        score = _spatial_information(maps[:, :, unit], occupancy)
        saved_score = float(data["post_inhibition_spatial_information"][unit])
        if not np.isclose(score, saved_score, rtol=1e-5, atol=1e-7):
            raise ValueError("Saved spatial score disagrees with canonical amplitude-weighted formula")
        valid_map = np.where(occupancy > 0, maps[:, :, unit], -np.inf)
        peak = np.unravel_index(valid_map.argmax(), occupancy.shape)
        x = bounds[0] + (peak[0] + .5) * (bounds[1] - bounds[0]) / 19
        y = bounds[2] + (peak[1] + .5) * (bounds[3] - bounds[2]) / 19
        active_count = int(data["post_inhibition_field_active_observation_count"][unit])
        active_fraction = float(data["post_inhibition_active_fraction"][unit])
        if not np.isclose(active_count / occupancy.sum(), active_fraction, atol=1e-7):
            raise ValueError("Saved activity count/fraction denominator disagrees")
        records.append(dict(unit=unit, observation_count=int(occupancy.sum()), active_observation_count=active_count,
            active_fraction=active_fraction, silent=active_count == 0,
            mean_post_inhibition_activity=float(np.sum(maps[:, :, unit] * occupancy) / occupancy.sum()),
            amplitude_weighted_spatial_score=saved_score, unsmoothed_peak_activity=float(valid_map[peak]),
            unsmoothed_peak_x=x, unsmoothed_peak_y=y,
            pre_threshold_mean=float(data["pre_threshold_mean"][unit]),
            pre_threshold_std=float(data["pre_threshold_std"][unit]),
            field_eligible=bool(data["post_inhibition_field_eligible"][unit]),
            mono_field=bool(data["post_inhibition_field_mono"][unit]),
            mono_score=float(data["post_inhibition_field_mono_score"][unit]),
            active_bin_count=int(data["post_inhibition_field_active_bin_count"][unit]),
            components_30pct=int(data["post_inhibition_field_component_count"][0, unit]),
            components_50pct=int(data["post_inhibition_field_component_count"][1, unit]),
            components_70pct=int(data["post_inhibition_field_component_count"][2, unit]),
            dominant_component_peak_x=float(data["post_inhibition_field_dominant_peak_xy"][unit, 0]),
            dominant_component_peak_y=float(data["post_inhibition_field_dominant_peak_xy"][unit, 1])))
    cosine, peaks = diversity_helpers()
    active = np.flatnonzero(data["post_inhibition_active_fraction"] > 0)
    unique, entropy, distance = peaks(maps, active, require_positive=True)
    eligible, mono = sum(r["field_eligible"] for r in records), sum(r["mono_field"] for r in records)
    summary = dict(units=16, active_units=len(active), silent_units=16-len(active), field_eligible_units=eligible,
        mono_field_units=mono, mono_fraction_all_units=mono/16,
        mono_fraction_eligible_units=mono/eligible if eligible else None,
        observation_samples=int(occupancy.sum()), visited_bins=int((occupancy > 0).sum()), total_bins=occupancy.size,
        active_only_map_cosine=cosine(maps, occupancy, active), unique_active_peak_bins=unique,
        active_peak_bin_entropy=entropy, mean_pairwise_peak_distance_bins=distance,
        mean_amplitude_weighted_spatial_score=float(data["post_inhibition_spatial_information"].mean()),
        saved_field_criteria=dict(min_active_observations=int(data["post_inhibition_field_min_active_observations"]),
            min_active_bins=int(data["post_inhibition_field_min_active_bins"]),
            thresholds=data["post_inhibition_field_threshold_fractions"].tolist(),
            mono_mass_fraction=float(data["post_inhibition_field_mono_mass_fraction"])))
    return records, summary


def graph_summary(data):
    """Export all256 stored entries and canonical descriptive graph metrics."""
    attempts, confidence, tctrl = [np.asarray(data[key]) for key in
        ("control_attempts", "control_edge_confidence", "control_tctrl")]
    if any(value.shape != (16, 16) or not np.isfinite(value).all() or (value < 0).any()
           for value in (attempts, confidence, tctrl)):
        raise ValueError("Stored graph arrays must be finite nonnegative16x16")
    if (confidence > attempts + 1e-3).any():
        raise ValueError("Stored success mass exceeds attempted mass")
    diagnostics = calculate_graph_diagnostics(tctrl, confidence, attempts)
    rows = [dict(source=source, target=target, diagonal=source == target,
                 stored_attempt_mass=float(attempts[source, target]),
                 stored_success_mass=float(confidence[source, target]),
                 stored_success_elapsed_decisions=float(tctrl[source, target]),
                 stored_smoothed_success_ratio=float(diagnostics["graph_edge_reliability"][source, target]),
                 canonical_reliable_edge=bool(diagnostics["graph_reliable_adjacency"][source, target]))
            for source in range(16) for target in range(16)]
    # The helper defaults missing prospective arrays to zeros. Those fields are
    # explicitly excluded: missing prospective evidence is not zero success.
    structural_keys = ("graph_reliable_edge_count", "graph_reliable_edge_density", "graph_largest_weak_component_size",
        "graph_largest_strong_component_size", "graph_reachable_pair_fraction", "graph_mean_reachable_shortest_path_hops",
        "graph_reliable_global_efficiency", "graph_directed_reciprocity")
    summary = {key: np.asarray(diagnostics[key]).item() for key in structural_keys}
    summary.update(complete_entries=256, off_diagonal_entries=240,
        maximum_stored_success_mass=float(confidence.max()), maximum_stored_attempt_mass=float(attempts.max()),
        maximum_stored_elapsed_decisions=float(tctrl.max()),
        prospective_evidence_available=False,
        interpretation="Frozen pretrained policy-graph buffers, not new first-hit reachability or current executed success",
        canonical_analysis_thresholds=dict(success_mass=.5, smoothed_success_ratio=.5),
        semantics=dict(control_attempts="Exponentially decayed mass of completed directed option attempts",
            control_edge_confidence="Exponentially decayed successful-option mass; values greater than one are expected; not a probability",
            control_tctrl="Success-mass-weighted mean option completion time in decisions; zero means no stored successful timing",
            stored_smoothed_success_ratio="Canonical (success_mass+1)/(attempt_mass+2); descriptive stored estimate, not an independently validated control probability"))
    return rows, summary


def image_values(maps, occupancy, unit):
    """Transpose historical[x,y] to displayed[y,x], masking unknown bins."""
    return np.ma.array(maps[:, :, unit].T, mask=occupancy.T <= 0)


def render_unit(data, unit, output, bounds, scales):
    font_manager.fontManager.addfont(str(FONT))
    family = font_manager.FontProperties(fname=str(FONT)).get_name()
    if Path(font_manager.findfont(family, fallback_to_default=False)).resolve() != FONT.resolve():
        raise ValueError("Resolved font differs from the declared scalable TTF")
    with plt.rc_context({"font.family": family, "font.size": 30, "axes.labelsize": 30,
                         "xtick.labelsize": 30, "ytick.labelsize": 30, "svg.fonttype": "none"}):
        fig = plt.figure(figsize=(10, 19), dpi=100)
        fig.suptitle(f"DG {unit:02d} · frozen common panel", fontsize=34, y=.984)
        positions = [(.21, .59, .53, .53/1.9), (.21, .13, .53, .53/1.9)]
        specs = [("a  Post-inhibition mean", "post_inhibition_rate_maps", "cividis", "DG activity (a.u.)", "post"),
                 ("b  Pre-threshold mean", "pre_threshold_rate_maps", "RdBu_r", "DG logit (a.u.)", "pre")]
        for position, (title, key, cmap, label, scale) in zip(positions, specs):
            ax = fig.add_axes(position)
            palette = plt.get_cmap(cmap).copy()
            palette.set_bad("#c6c6c6")
            low, high = scales[scale]
            norm = colors.TwoSlopeNorm(0, low, high) if scale == "pre" else colors.Normalize(low, high)
            im = ax.imshow(image_values(data[key], data["occupancy"], unit), origin="lower", extent=bounds,
                           cmap=palette, norm=norm, interpolation="nearest")
            ax.set_title(title, fontsize=32, loc="left", pad=22)
            ax.set_xlabel("x (DMLab units)", labelpad=12)
            ax.set_ylabel("y (DMLab units)", labelpad=10)
            ax.set_xticks([100, 1000, 2000]); ax.set_yticks([100, 1000, 2000])
            cax = fig.add_axes((.78, position[1], .035, position[3]))
            colorbar = fig.colorbar(im, cax=cax)
            colorbar.set_label(label, labelpad=14, fontsize=30)
            colorbar.ax.tick_params(labelsize=30, pad=8)
            colorbar.set_ticks([0, 8, 16] if scale == "post" else [-1, 0, 9, 18])
        fig.text(.5, .025, "Gray: unvisited bins", ha="center", fontsize=30)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for text in fig.findobj(matplotlib.text.Text):
            if text.get_visible() and text.get_text():
                bbox = text.get_window_extent(renderer)
                if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > 1001 or bbox.y1 > 1901:
                    raise ValueError(f"Clipped figure text: {text.get_text()!r}")
        for suffix in ("png", "svg"):
            fig.savefig(output / f"dg_{unit:02d}.{suffix}", dpi=100, facecolor="white")
        plt.close(fig)


def run(npz, audit_path, graph_source, output):
    npz, audit_path, graph_source, output = map(lambda p: Path(p).resolve(), (npz, audit_path, graph_source, output))
    if output.exists():
        raise FileExistsError("Preserve a previous field rendering")
    audit = json.loads(audit_path.read_text())
    if not audit["all_recorded_checks_passed"] or digest(npz) != audit["artifacts"]["common_panel_fields"]["sha256"]:
        raise ValueError("Field NPZ does not match the exact donor reuse audit")
    graph_audit_path = audit_path.with_name("donor_graph_source_hash.json")
    graph_audit = json.loads(graph_audit_path.read_text())
    if digest(graph_source) != graph_audit["sha256"] or graph_source.stat().st_size != graph_audit["bytes"]:
        raise ValueError("Graph source differs from the independently verified donor file")
    if FONT.suffix.lower() not in (".ttf", ".otf") or not FONT.is_file():
        raise ValueError("A verified scalable font is required")
    with np.load(npz, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    bounds = tuple(audit["mapping_contract"]["world_bounds"][axis][end] for axis in ("x", "y") for end in (0, 1))
    units, fields = unit_summary(data, bounds)
    edges, graph = graph_summary(data)
    visited = data["occupancy"] > 0
    scales = dict(post=[0., float(data["post_inhibition_rate_maps"][visited].max())],
                  pre=[float(data["pre_threshold_rate_maps"][visited].min()), float(data["pre_threshold_rate_maps"][visited].max())])
    output.mkdir(parents=True)
    sources = [Path(__file__), CANONICAL_METRICS, Path(spatial_contract.__file__), graph_source,
               Path(__file__).with_name("provenance.py"), Path(__file__).parents[1] / "test_first_hit_field_reuse.py"]
    snapshot = snapshot_sources(sources, output / "source_snapshot")
    inputs = {str(path): digest(path) for path in (npz, audit_path, graph_audit_path, FONT)}
    protocol = dict(schema="intrmotiv/saved-field-reuse/v1", created_utc=datetime.now(timezone.utc).isoformat(),
        inputs=inputs, source_snapshot=snapshot, selected_units=list(SELECTED_UNITS),
        selection="Indices0,4,8,12 fixed by parent before main control outcomes", bounds=list(bounds),
        saved_map_axes=["x_bin", "y_bin", "unit"], displayed_map_axes=["y_bin", "x_bin"],
        plotted_arrays=["post_inhibition_rate_maps", "pre_threshold_rate_maps"],
        transform="Occupancy-corrected saved means; transpose for display only; no additional smoothing or normalization; unvisited bins gray",
        shared_limits_across_all16=scales, prethreshold_color_center=0,
        font_path=str(FONT), minimum_font_pt=30, figure_inches=[10, 19], dpi=100, png_pixels=[1000, 1900],
        graph_source_scope="Archived source blob from donor Git commit cdf9ffe9026dc483e9c31dae7a1365af67c28642; byte hash independently confirmed against actual remote donor file",
        graph_source_receipt=graph_audit,
        graph_missing_evidence="No prospective arrays in saved NPZ; omitted rather than reported as zero",
        matplotlib_version=matplotlib.__version__, numpy_version=np.__version__)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for name, rows in (("units.csv", units), ("stored_graph.csv", edges)):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for unit in SELECTED_UNITS:
        render_unit(data, unit, output, bounds, scales)
    source_audit = audit_source_snapshots(output / "source_snapshot")
    if not source_audit["archive_valid"] or not source_audit["current_sources_unchanged"] or any(digest(path) != sha for path, sha in inputs.items()):
        raise RuntimeError("Field inputs or source changed during rendering")
    result = dict(schema="intrmotiv/saved-field-reuse-results/v1", field_summary=fields, stored_graph_summary=graph,
        source_audit=source_audit, protocol_sha256=digest(output / "protocol.json"),
        outputs={p.name: digest(p) for p in output.iterdir() if p.suffix in ("png", "svg", "csv")},
        visual_qa="Exported text bounds checked automatically; manual inspection pending")
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("npz", "audit", "graph-source", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(args.npz, args.audit, args.graph_source, args.output)
