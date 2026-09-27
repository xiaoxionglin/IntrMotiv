"""Build compact poster-selection plots from completed frozen and replay analyses.

Only reads saved summary CSVs and the frozen DG peak table. Seed is the
replication unit; DGC Waypoint checkpoints are shown as candidates because
their ages differ from Direct and from one another.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIELD_FILES = (
    "frozen_derived_place_field_metrics.csv", "dgc_direct_latest_frozen_derived.csv",
    "dgc_waypoint_frozen_derived.csv", "d50_75m_latest_frozen_derived.csv",
    "d51_75m_frozen_derived.csv", "mature_frozen_derived.csv",
)
GRAPH_FILES = (
    "stored_graph_main.csv", "stored_graph_dgc_direct_latest.csv",
    "stored_graph_dgc_waypoint.csv", "stored_graph_d50_75m_latest.csv",
    "stored_graph_d51_75m_latest.csv", "stored_graph_mature.csv",
)
COMPARISONS = (
    ("Saturday", "SAT_C15_ARR_MON_FILM", "SAT_C15_SRC_MON_FILM", "ARR", "SRC"),
    ("CPU2048", "CPU2048_DIRECT_F16_DDQN_HER", "CPU2048_WAYPOINT_DECODER_F64_DDQN_HER", "Direct", "Waypoint"),
    ("DGP", "DGP_C15_HIT_JOINT_LEG", "DGP_C15_FIRST_JOINT_LEG", "HIT", "FIRST"),
    ("DGC candidates", "DGC_DIRECT_WORKER_F16", "DGC_WAYPOINT_DG_F64", "Direct", "Waypoint"),
    ("D50 transfer", "CR5C_D50_W_SOURCE_DG", "CR5C_D50_W_RAND_DG", "Source", "Random"),
    ("D51 transfer", "CR5C_D51_W_SOURCE_DG", "CR5C_D51_W_RAND_DG", "Source", "Random"),
)
SHORT = {condition: f"{family}: {label}" for family, first, second, a, b in COMPARISONS
         for condition, label in ((first, a), (second, b))}
SHORT.update({"CPU2048_DIRECT_F16_DDQN": "Mature Direct F16",
              "CPU2048_WAYPOINT_DECODER_F64_DDQN": "Mature Waypoint F64",
              "FSCS_WAYPOINT_DECODER_F64_PPO": "Mature full-system PPO"})
BLUE, ORANGE = "#0072B2", "#D55E00"


def style() -> None:
    from matplotlib import font_manager
    font = font_manager.findfont("Times New Roman", fallback_to_default=False)
    if not Path(font).is_file() or Path(font).suffix.lower() not in {".ttf", ".otf"}:
        raise RuntimeError("A scalable Times New Roman font is required")
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12,
                         "axes.titlesize": 13, "axes.labelsize": 12,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "legend.fontsize": 12, "lines.linewidth": 1.3,
                         "axes.linewidth": 1, "svg.fonttype": "none",
                         "pdf.fonttype": 42})


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def inventory(data: Path, output: Path) -> pd.DataFrame:
    with (data / "fullrange_plan/kernel_fullrange_manifest.tsv").open(newline="") as stream:
        manifest = pd.DataFrame(csv.DictReader(stream, delimiter="\t"))
    manifest = manifest[manifest.analysis_role.isin(
        ["endpoint", "endpoint_unmatched_age", "endpoint_exemplar"])].copy()
    if len(manifest) != 38:
        raise ValueError(f"Expected 38 saved-run candidates, found {len(manifest)}")
    fields = pd.concat([pd.read_csv(data / name) for name in FIELD_FILES], ignore_index=True)
    graphs = pd.concat([pd.read_csv(data / name) for name in GRAPH_FILES], ignore_index=True)
    fields = fields.drop_duplicates("label_suffix")
    graphs = graphs.drop_duplicates("label_suffix")
    kernels = pd.read_csv(data / "fullrange_summary/fullrange_bands_per_run.csv")
    kernels = kernels[(kernels.scope == "all_cues") &
                      (kernels.distance_band == "long_13_18")]
    wide = kernels.pivot(index="label_suffix", columns="layer",
                         values=["mean_correlation", "offsets_supported"])
    wide.columns = [f"{layer}_long_{'correlation' if value == 'mean_correlation' else 'supported_offsets'}"
                    for value, layer in wide.columns]
    wide = wide.reset_index()
    keep_fields = ["label_suffix", "active_units", "silent_units", "active_unit_mean_si_bits",
                   "active_map_cosine_mean", "active_unique_peak_bins", "mono_field_fraction",
                   "visited_cell_fraction"]
    keep_graphs = ["label_suffix", "stored_attempted_edges", "stored_confidence_per_attempt",
                   "graph_reliable_edge_count", "graph_reachable_pair_fraction",
                   "mono_field_nodes"]
    result = manifest[["condition", "seed", "checkpoint_frames", "label_suffix",
                       "analysis_role"]].merge(fields[keep_fields], on="label_suffix", validate="1:1")
    result = result.merge(graphs[keep_graphs], on="label_suffix", validate="1:1")
    result = result.merge(wide, on="label_suffix", validate="1:1")
    result["seed"] = result.seed.astype(int)
    result["checkpoint_frames"] = result.checkpoint_frames.astype(int)
    flow = pd.read_csv(data / "frozen_behavior_summary_all_candidates.csv")[[
        "label_suffix", "mean_flow_coherence", "flow_cells_at_least_5"]]
    flow = flow.drop_duplicates("label_suffix")
    result = result.merge(flow, on="label_suffix", how="left", validate="1:1")
    historical = pd.read_csv(data.parent.parent / "results/A0_poster_analysis_20260926/frozen_exemplar_metrics.csv")
    direct = historical[historical.run.str.contains("CPU2048_DIRECT_F16_DDQN_S99")].iloc[0]
    direct_label = "CPU2048_DIRECT_F16_DDQN_S99_300007424"
    direct_kernels = pd.read_csv(data / "fullrange_summary/historical_direct/fullrange_bands_per_run.csv")
    direct_kernels = direct_kernels[(direct_kernels.scope == "all_cues") &
                                    (direct_kernels.distance_band == "long_13_18")]
    direct_row = {"condition": "CPU2048_DIRECT_F16_DDQN", "seed": 99,
                  "checkpoint_frames": 300007424, "label_suffix": direct_label,
                  "analysis_role": "endpoint_exemplar",
                  "active_units": int(direct.active_units),
                  "silent_units": 16-int(direct.active_units),
                  "active_unit_mean_si_bits": direct.active_mean_spatial_information,
                  "active_map_cosine_mean": direct.active_only_map_cosine,
                  "active_unique_peak_bins": int(direct.distinct_peak_bins),
                  "mono_field_fraction": direct.mono_field_fraction,
                  "visited_cell_fraction": direct.coverage_fraction,
                  "mean_flow_coherence": direct.mean_flow_coherence}
    for layer in ("dg", "ca3", "decoder_1"):
        item = direct_kernels[direct_kernels.layer == layer].iloc[0]
        direct_row[f"{layer}_long_correlation"] = item.mean_correlation
        direct_row[f"{layer}_long_supported_offsets"] = item.offsets_supported
    result = pd.concat([result, pd.DataFrame([direct_row])], ignore_index=True)
    peaks = pd.read_csv(data / "frozen_peak_positions_with_mature_direct.csv")
    peaks = peaks[peaks.active_fraction > 0].copy()
    peaks["defined_peak"] = (peaks.field_eligible &
                             peaks.peak_x_bin.between(0, 18) &
                             peaks.peak_y_bin.between(0, 18))
    counts = peaks.groupby("label_suffix", as_index=False).agg(
        active_peak_units=("unit", "size"), defined_peak_units=("defined_peak", "sum"))
    counts["defined_peak_fraction"] = counts.defined_peak_units / counts.active_peak_units
    result = result.merge(counts, on="label_suffix", validate="1:1")
    result.to_csv(output / "exemplar_inventory.csv", index=False)
    return result


def peak_maps(peaks: pd.DataFrame, inventory: pd.DataFrame, output: Path) -> None:
    for family, first, second, label_a, label_b in COMPARISONS:
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.6), constrained_layout=True)
        matrices = []
        for condition in (first, second):
            units = peaks[(peaks.condition == condition) & (peaks.active_fraction > 0)]
            selected = units[(units.field_eligible) & (units.peak_x_bin.between(0, 18)) &
                             (units.peak_y_bin.between(0, 18))]
            matrix = np.zeros((19, 19), dtype=float)
            np.add.at(matrix, (selected.peak_x_bin.to_numpy(int),
                               selected.peak_y_bin.to_numpy(int)), 1)
            matrix /= max(len(selected), 1)
            matrices.append(matrix)
        vmax = max(matrix.max() for matrix in matrices)
        for ax, condition, label, matrix in zip(axes, (first, second),
                                                 (label_a, label_b), matrices):
            im = ax.imshow(matrix.T, origin="lower", vmin=0, vmax=vmax,
                           cmap="cividis", interpolation="nearest", extent=(0, 19, 0, 19))
            ax.set_xticks([0, 9, 18]); ax.set_yticks([0, 9, 18])
            ax.set(xlabel="x bin", ylabel="y bin", title=label)
            ax.set_aspect("equal")
            all_units = peaks[(peaks.condition == condition) & (peaks.active_fraction > 0)]
            defined = all_units[all_units.field_eligible &
                                all_units.peak_x_bin.between(0, 18) &
                                all_units.peak_y_bin.between(0, 18)]
            ax.text(.02, .02, f"{len(defined)} defined / {len(all_units)} active",
                    transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=12,
                    bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
        fig.colorbar(im, ax=axes, shrink=.75, label="Share of defined DG peaks")
        fig.suptitle(f"{family} · frozen-policy DG peak locations" +
                     (" · ages differ" if family == "DGC candidates" else ""), fontsize=13)
        save(fig, output / f"peak_map_{family.lower().replace(' ', '_')}.svg")


def bars(inventory: pd.DataFrame, output: Path) -> None:
    order = [condition for _, a, b, _, _ in COMPARISONS for condition in (a, b)]
    specs = (("active_map_cosine_mean", "Active-only DG map cosine"),
             ("active_unique_peak_bins", "Distinct DG peak bins"),
             ("visited_cell_fraction", "Visited coarse-cell fraction"),
             ("graph_reachable_pair_fraction", "Stored-graph reachable pairs"))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 10.5), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, specs):
        for index, condition in enumerate(order):
            values = inventory.loc[inventory.condition == condition, metric].dropna().to_numpy()
            if len(values) != 3:
                raise ValueError(f"Missing three seeds for {condition}/{metric}")
            color = BLUE if index % 2 == 0 else ORANGE
            ax.barh(index, values.mean(), height=.68, color=color, alpha=.55)
            ax.scatter(values, np.full(3, index)+np.array([-.14, 0, .14]),
                       s=25, color=color, edgecolor="black", linewidth=.45, zorder=3)
        ax.set_yticks(range(len(order)), [SHORT[name] for name in order])
        ax.invert_yaxis(); ax.set_title(title, loc="left")
        ax.grid(axis="x", alpha=.2, linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Mean bar; dots are trained seeds")
    fig.suptitle("Frozen-policy architecture overview · three seeds per arm\n"
                 "DGC Waypoint ages differ; graph reachability is checkpoint stored", fontsize=13)
    save(fig, output / "architecture_overview.svg")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 5.4), constrained_layout=True)
    for ax, metric, title in zip(axes,
        ("active_unit_mean_si_bits", "defined_peak_fraction"),
        ("Mean active-unit spatial information", "Active units with a defined field peak")):
        for index, condition in enumerate(order):
            values = inventory.loc[inventory.condition == condition, metric].dropna().to_numpy()
            if len(values) != 3:
                raise ValueError(f"Missing three seeds for {condition}/{metric}")
            color = BLUE if index % 2 == 0 else ORANGE
            ax.barh(index, values.mean(), height=.68, color=color, alpha=.55)
            ax.scatter(values, np.full(3, index)+[-.14, 0, .14],
                       s=25, color=color, edgecolor="black", linewidth=.45, zorder=3)
        ax.set_yticks(range(len(order)), [SHORT[name] for name in order])
        ax.invert_yaxis(); ax.set_title(title, loc="left")
        ax.grid(axis="x", alpha=.2, linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Bits" if metric.endswith("bits") else "Fraction of active DG units")
    fig.suptitle("Frozen-policy DG field quality · three seeds per arm", fontsize=13)
    save(fig, output / "field_quality_overview.svg")


def kernel_bands(inventory: pd.DataFrame, output: Path) -> None:
    for family, first, second, label_a, label_b in COMPARISONS:
        fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.7), constrained_layout=True)
        for ax, layer, layer_name in zip(axes, ("dg", "ca3", "decoder_1"),
                                          ("DG", "CA3", "Decoder-1")):
            key = f"{layer}_long_correlation"
            a = inventory[inventory.condition == first].set_index("seed")
            b = inventory[inventory.condition == second].set_index("seed")
            for seed in sorted(set(a.index) & set(b.index)):
                if family != "DGC candidates":
                    ax.plot([0, 1], [a.loc[seed, key], b.loc[seed, key]],
                            color="#888888", linewidth=1.3, zorder=1)
                ax.scatter([0, 1], [a.loc[seed, key], b.loc[seed, key]],
                           color=[BLUE, ORANGE], s=42, zorder=2)
            ax.set_xticks([0, 1], [label_a, label_b]); ax.set_xlim(-.35, 1.35)
            ax.set_ylim(-.08, 1.03); ax.set_title(layer_name)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.2, linewidth=.7)
            if ax is axes[0]: ax.set_ylabel("Mean correlation, 13–18 bins")
        fig.suptitle(f"{family} · common-history population kernels · 3 seeds" +
                     (" · ages differ" if family == "DGC candidates" else ""), fontsize=13)
        save(fig, output / f"kernel_long_{family.lower().replace(' ', '_')}.svg")


def field_panels(data: Path, inventory: pd.DataFrame, output: Path) -> None:
    records = inventory[["condition", "seed", "checkpoint_frames", "label_suffix"]].to_dict("records")
    color = plt.get_cmap("viridis").copy()
    color.set_bad("#d8d8d8")
    for run in records:
        label = run["label_suffix"]
        with np.load(data / "top_four_frozen_fields" / f"{label}.npz",
                     allow_pickle=False) as archive:
            maps = archive["maps"]
            occupancy = archive["occupancy"]
            units = archive["unit_ids"]
            information = archive["spatial_information_bits"]
        if maps.shape != (19, 19, 4) or occupancy.shape != (19, 19):
            raise ValueError(f"Unexpected top-field dimensions for {label}")
        fig, axes = plt.subplots(2, 2, figsize=(6.7, 6.6), constrained_layout=True)
        for index, ax in enumerate(axes.flat):
            peak = np.nanmax(maps[:, :, index])
            if peak <= 0:
                raise ValueError(f"Nonpositive selected field peak: {label}, unit {units[index]}")
            relative = maps[:, :, index] / peak
            image = ax.imshow(np.ma.array(relative.T, mask=(occupancy == 0).T),
                              origin="lower", cmap=color, vmin=0, vmax=1,
                              extent=(0, 19, 0, 19), interpolation="nearest")
            ax.set_xticks([0, 9, 18]); ax.set_yticks([0, 9, 18])
            ax.set(xlabel="x bin", ylabel="y bin",
                   title=f"DG {units[index]} · SI {information[index]:.2f} bits")
        fig.colorbar(image, ax=axes, shrink=.65, label="Activity / unit peak")
        title = SHORT.get(run["condition"], "Mature Direct F16")
        fig.suptitle(f"{title} · seed {run['seed']} · {int(run['checkpoint_frames'])/1e6:.1f}M\n"
                     "Frozen policy · four active units with highest spatial information",
                     fontsize=13)
        save(fig, output / "per_run" / label / "top_four_fields.svg")


def control_panels(data: Path, inventory: pd.DataFrame, output: Path) -> None:
    trials = pd.read_csv(data / "control_interventions_per_run.csv")
    order = [condition for _, a, b, _, _ in COMPARISONS[:3] for condition in (a, b)]
    order.append("DGC_DIRECT_WORKER_F16")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.8), constrained_layout=True)
    for index, condition in enumerate(order):
        run = trials[trials.condition == condition]
        if len(run) != 3:
            raise ValueError(f"Expected three command-probe seeds: {condition}")
        lift = run.executed_minus_shuffled.to_numpy()
        coverage = run.complete_pair_fraction.to_numpy()
        color = BLUE if index % 2 == 0 else ORANGE
        axes[0].barh(index, lift.mean(), height=.67, color=color, alpha=.55)
        axes[0].scatter(lift, np.full(3, index)+[-.13, 0, .13],
                        s=26, color=color, edgecolor="black", linewidth=.4, zorder=3)
        axes[1].barh(index, coverage.mean(), height=.67, color=color, alpha=.55)
        axes[1].scatter(coverage, np.full(3, index)+[-.13, 0, .13],
                        s=26, color=color, edgecolor="black", linewidth=.4, zorder=3)
    for ax, title, xlabel in zip(axes,
        ("Executed − shuffled target success", "Complete ordered-pair coverage"),
        ("Success probability difference", "Fraction of eligible source–target pairs")):
        ax.set_yticks(range(len(order)), [SHORT[name] for name in order])
        ax.invert_yaxis(); ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel); ax.grid(axis="x", alpha=.2, linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Bounded command probes · three seeds per arm\n"
                 "Protocols differ across families; complete-pair coverage limits generality",
                 fontsize=13)
    save(fig, output / "control_diagnostics.svg")


def representation_graph_scatter(inventory: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 7.4), constrained_layout=True,
                             sharex=True, sharey=True)
    for ax, (family, first, second, label_a, label_b) in zip(axes.flat, COMPARISONS):
        for condition, label, color, marker in ((first, label_a, BLUE, "o"),
                                                (second, label_b, ORANGE, "s")):
            group = inventory[inventory.condition == condition]
            ax.scatter(group.active_map_cosine_mean, group.graph_reachable_pair_fraction,
                       s=48, color=color, marker=marker, alpha=.65,
                       edgecolor="black", linewidth=.4, label=label)
            ax.scatter([group.active_map_cosine_mean.mean()],
                       [group.graph_reachable_pair_fraction.mean()],
                       s=115, color=color, marker=marker, edgecolor="black",
                       linewidth=1.1, zorder=4)
        ax.set_title(family + (" · ages differ" if family == "DGC candidates" else ""))
        ax.set_xlim(-.01, .28); ax.set_ylim(-.04, 1.04)
        ax.grid(alpha=.2, linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="best", handletextpad=.3)
    for ax in axes[:, 0]: ax.set_ylabel("Stored-graph reachable pairs")
    for ax in axes[-1, :]: ax.set_xlabel("Active-only DG map cosine")
    fig.suptitle("DG map overlap versus checkpoint-stored graph reachability\n"
                 "Small points are seeds; outlined large points are arm means", fontsize=13)
    save(fig, output / "representation_graph_scatter.svg")


def run_index(inventory: pd.DataFrame, output: Path) -> None:
    """Write a complete clickable index, retaining one row per saved run."""
    lines = ["# Complete poster exemplar index", "",
             "Each row is one frozen-policy checkpoint. DG and graph metrics use",
             "the saved frozen evaluation; the three kernel values are common-history",
             "mean correlations at 13–18 spatial bins. `Active/defined` gives active",
             "DG units and those with an eligible, localized peak. The mature Direct",
             "example is historical; its graph-reachability field is unavailable in",
             "the harmonized table. DGC Waypoint ages vary across seeds.", "",
             "| Run | Exact frames | DG active / defined | SI (bits) | Map cosine | Peak bins | Visit / flow | Graph reach | Long DG / CA3 / Dec-1 | Panels |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    order = [name for _, a, b, _, _ in COMPARISONS for name in (a, b)]
    extra = ["CPU2048_DIRECT_F16_DDQN", "CPU2048_WAYPOINT_DECODER_F64_DDQN",
             "FSCS_WAYPOINT_DECODER_F64_PPO"]
    for condition in order + extra:
        for row in inventory[inventory.condition == condition].sort_values("seed").itertuples():
            label = row.label_suffix
            prefix = f"per_run/{label}"
            links = " ".join(f"[{name}]({prefix}/{file})" for name, file in (
                ("fields", "top_four_fields.svg"), ("trajectory", "occupancy_trajectory.png"),
                ("flow", "occupancy_flow.png"), ("graph", "stored_graph.png"),
                ("kernel", "layerwise_kernels.svg"), ("radial", "layerwise_radial.svg")))
            short = SHORT[condition]
            reach = "—" if pd.isna(row.graph_reachable_pair_fraction) else f"{row.graph_reachable_pair_fraction:.2f}"
            flow = "—" if pd.isna(row.mean_flow_coherence) else f"{row.mean_flow_coherence:.2f}"
            lines.append(
                f"| {short} · seed {row.seed} | {row.checkpoint_frames:,} | "
                f"{row.active_units:.0f} / {row.defined_peak_units:.0f} | "
                f"{row.active_unit_mean_si_bits:.3f} | {row.active_map_cosine_mean:.3f} | "
                f"{row.active_unique_peak_bins:.0f} | {row.visited_cell_fraction:.2f} / {flow} | "
                f"{reach} | {row.dg_long_correlation:.2f} / {row.ca3_long_correlation:.2f} / "
                f"{row.decoder_1_long_correlation:.2f} | {links} |")
    lines += ["", "The source [CSV](exemplar_inventory.csv) retains exact labels,",
              "support counts, graph edge counts, and all numeric precision.", ""]
    (output / "run_index.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    style(); args.output.mkdir(parents=True, exist_ok=True)
    info = inventory(args.data, args.output)
    peaks = pd.read_csv(args.data / "frozen_peak_positions.csv")
    peak_maps(peaks, info, args.output)
    bars(info, args.output)
    kernel_bands(info, args.output)
    control_panels(args.data, info, args.output)
    representation_graph_scatter(info, args.output)
    field_panels(args.data, info, args.output)
    run_index(info, args.output)
    print(f"Built selection data for {len(info)} endpoint/candidate/exemplar runs")


if __name__ == "__main__":
    main()
