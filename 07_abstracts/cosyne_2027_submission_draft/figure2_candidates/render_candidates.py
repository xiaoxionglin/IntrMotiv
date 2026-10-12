"""Render exploratory COSYNE Figure 2 alternatives from archived seed-level tables.

These diagnostic figures are deliberately outside the compact Overleaf ZIP.
The existing submission renderer remains the source of shared colors, typography,
data paths, and the selected peak-map example.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DRAFT = HERE.parent
sys.path.insert(0, str(DRAFT))
import render_submission_panels as shared  # noqa: E402


EXTRA_SOURCES = {
    "worker_fields": shared.RESULTS / "cosyne_2027_checkpoint_fields" /
    "common_panel_analysis" / "paired_worker_fields.csv",
    "worker_outcomes": shared.RESULTS / "cosyne_2027_checkpoint_fields" /
    "worker_timing_summary" / "paired_by_seed.csv",
}
SOURCE_PATHS = {**shared.SOURCES, **EXTRA_SOURCES}
ENCODER_GROUPS = ["C01", "C05-wT", "C05-wC", "C15"]
WORKER_GROUPS = ["C05-eT", "C05-eC"]
CONTRAST_GROUPS = {
    "C01": "C01", "C05 · temporal hit": "C05-wT",
    "C05 · constant hit": "C05-wC", "C15": "C15",
}


def load_tables() -> dict[str, pd.DataFrame | dict]:
    tables = {key: (json.loads(path.read_text()) if path.suffix == ".json"
                    else pd.read_csv(path)) for key, path in SOURCE_PATHS.items()}
    for key in ("encoder", "peaks", "decoder", "worker_fields"):
        data = tables[key]
        assert isinstance(data, pd.DataFrame)
        if key == "decoder":
            data = data[data.factor != "interaction"]
        assert np.allclose(data.temporal - data.constant, data.difference,
                           atol=1e-6)
    data = tables["worker"]
    assert isinstance(data, pd.DataFrame)
    assert np.allclose(data["first"] - data.second, data.difference,
                       atol=1e-9)
    data = tables["worker_outcomes"]
    assert isinstance(data, pd.DataFrame)
    assert np.allclose(data.temporal_worker - data.constant_worker,
                       data.difference, atol=1e-9)
    return tables


def tagged(data: pd.DataFrame, *, kind: str) -> pd.DataFrame:
    """Attach short, unambiguous labels without parsing run names."""
    data = data.copy()
    if kind == "encoder_fields" or kind == "peaks":
        data["group"] = data.contrast.map(CONTRAST_GROUPS)
    elif kind == "encoder_layers":
        data["group"] = data.apply(
            lambda row: f"C05-w{row.fixed_arm[0].upper()}"
            if row.family == "C05" else row.family, axis=1)
    elif kind in ("worker_layers", "worker_online"):
        column = "fixed_arm" if kind == "worker_layers" else "fixed_factor"
        data["group"] = data[column].map(
            {"temporal": "C05-eT", "constant": "C05-eC"})
    elif kind == "worker_fields":
        data["group"] = data.encoder_credit.map(
            {"temporal": "C05-eT", "constant": "C05-eC"})
    elif kind == "worker_outcomes":
        data["group"] = data.encoder_credit.map(
            {"temporal": "C05-eT", "constant": "C05-eC"})
    else:
        raise ValueError(kind)
    assert data.group.notna().all()
    return data


def validate_selected_evidence(t: dict[str, pd.DataFrame | dict]) -> None:
    """Check the sample units and selected example described in the gallery."""
    behavior = t["behavior_absolute"]
    assert len(behavior) == 38
    assert behavior.groupby("family").size().to_dict() == {
        "C01": 10, "C05": 20, "C15": 8}
    encoder = tagged(t["encoder"], kind="encoder_fields")
    for _, rows in encoder.groupby("metric"):
        assert rows.groupby("group").size().reindex(ENCODER_GROUPS).tolist() == [5, 5, 5, 5]
    worker = tagged(t["worker_fields"], kind="worker_fields")
    for _, rows in worker.groupby("metric"):
        assert rows.groupby("group").size().reindex(WORKER_GROUPS).tolist() == [5, 5]
    layers = t["decoder"]
    for _, rows in layers.groupby(["layer", "metric"]):
        encoder_rows = tagged(rows[rows.factor == "encoder"], kind="encoder_layers")
        worker_rows = tagged(rows[rows.factor == "worker"], kind="worker_layers")
        assert encoder_rows.groupby("group").size().reindex(ENCODER_GROUPS).tolist() == [5, 5, 5, 5]
        assert worker_rows.groupby("group").size().reindex(WORKER_GROUPS).tolist() == [5, 5]
    long_rows = t["longer_credit_online"]
    assert long_rows.groupby("dg_family").size().to_dict() == {
        "Prescribed": 3, "C15": 3}
    example = t["example"]
    baseline_peaks = t["peaks"][(t["peaks"].contrast == "C01") &
                                (t["peaks"].metric == "effective_area_fraction") &
                                (t["peaks"].cutoff == 0.0025)]
    assert len(baseline_peaks) == 5
    assert example["seed"] == int(baseline_peaks.loc[
        baseline_peaks.difference.idxmax(), "seed"])


def color(group: str) -> str:
    return shared.COLOR[group.split("-")[0]]


def paired(ax: plt.Axes, data: pd.DataFrame, groups: list[str],
           control: str, condition: str, *, vertical: bool,
           limits: tuple[float, float], ylabel: str = "") -> None:
    """Plot original paired values; orientation changes no scientific quantity."""
    assert not data.empty
    values = data[[control, condition]].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert values.min() >= limits[0] - 1e-9 and values.max() <= limits[1] + 1e-9, (
        f"{ylabel}: plotted values {values.min():.4g}–{values.max():.4g} "
        f"exceed axis limits {limits}")
    for index, group in enumerate(groups):
        rows = data[data.group == group].sort_values("seed")
        if rows.empty:
            continue
        offsets = np.linspace(-0.16, 0.16, len(rows))
        for shift, (_, row) in zip(offsets, rows.iterrows()):
            position = index + shift
            first, second = row[control], row[condition]
            if vertical:
                ax.plot([position, position], [first, second], color="#AEB8BD",
                        lw=0.7, zorder=1)
                ax.scatter(position, first, marker="s", facecolors="white",
                           edgecolors=shared.GRAY, s=20, linewidths=0.9, zorder=3)
                ax.scatter(position, second, marker="o", color=color(group),
                           s=17, zorder=4)
            else:
                ax.plot([first, second], [position, position], color="#AEB8BD",
                        lw=0.7, zorder=1)
                ax.scatter(first, position, marker="s", facecolors="white",
                           edgecolors=shared.GRAY, s=20, linewidths=0.9, zorder=3)
                ax.scatter(second, position, marker="o", color=color(group),
                           s=17, zorder=4)
    if vertical:
        ax.set_xticks(range(len(groups)), groups)
        ax.set_xlim(-0.45, len(groups) - 0.55)
        ax.set_ylim(*limits)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", color="#E9ECEE", lw=0.55)
    else:
        ax.set_yticks(range(len(groups)), groups)
        ax.set_ylim(len(groups) - 0.55, -0.45)
        ax.set_xlim(*limits)
        ax.set_xlabel(ylabel, fontsize=9)
        ax.grid(axis="x", color="#E9ECEE", lw=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, pad=2)


def delta(ax: plt.Axes, data: pd.DataFrame, groups: list[str],
          *, title: str, unit: str) -> None:
    """Small multiples of signed seed effects; the zero line is shared in meaning."""
    assert not data.empty and set(data.group) <= set(groups)
    span = max(float(data.difference.abs().max()) * 1.14, 0.015)
    for index, group in enumerate(groups):
        rows = data[data.group == group].sort_values("seed")
        if rows.empty:
            continue
        offsets = np.linspace(-0.17, 0.17, len(rows))
        ax.scatter(rows.difference, index + offsets, s=18,
                   color=color(group), zorder=3)
        ax.plot([rows.difference.mean()], [index], marker="|", ms=11,
                mew=1.5, color="#20272C", zorder=4)
    ax.axvline(0, color="#747D83", lw=0.8, zorder=1)
    ax.set_xlim(-span, span)
    ax.set_ylim(len(groups) - 0.55, -0.45)
    ax.set_yticks(range(len(groups)), groups)
    ax.set_title(title, fontsize=10, loc="left", pad=2)
    ax.set_xlabel(f"Temporal − constant ({unit})", fontsize=8.5, labelpad=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, pad=1.5)
    ax.grid(axis="x", color="#E9ECEE", lw=0.5)


def label(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(f"{letter}  {title}", fontsize=10, fontweight="bold",
                 loc="left", pad=7)


def pair_legend(fig: plt.Figure) -> None:
    handles = [
        Line2D([], [], linestyle="none", marker="s", markersize=5,
               markerfacecolor="white", markeredgecolor=shared.GRAY,
               label="Control"),
        Line2D([], [], linestyle="none", marker="o", markersize=5,
               color="#30363B", label="Tested"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.993),
               frameon=False, ncol=2, fontsize=9, columnspacing=0.7,
               handletextpad=0.2)


def vertical_story(t: dict) -> plt.Figure:
    """Retain the submitted A–F evidence while turning values into y axes."""
    fig = plt.figure(figsize=(7.5, 9.4), facecolor="white")
    grid = fig.add_gridspec(5, 6, left=0.085, right=0.985, bottom=0.07,
                           top=0.925, hspace=0.77, wspace=0.85,
                           height_ratios=[1.0, 0.82, 1.0, 1.0, 1.0])
    ax = fig.add_subplot(grid[0, :2])
    online = t["online"]
    for family, worker_arm in (("C01", "inapplicable"), ("C05", "temporal"),
                               ("C15", "temporal")):
        rows = online[(online.family == family) &
                      (online.encoder_credit == "temporal") &
                      (online.worker_bonus == worker_arm) &
                      (online.metric == "spatial_information")].sort_values("env_steps")
        assert rows.n.eq(5).all()
        x = rows.env_steps / 1e6
        ax.plot(x, rows["mean"], color=color(family), lw=1.1, label=family)
        ax.fill_between(x, rows["mean"] - rows["sem"], rows["mean"] + rows["sem"],
                        color=color(family), alpha=0.13)
    ax.set(xlim=(0, 100), xlabel="Frames (millions)", ylabel="DG score")
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=8,
              handlelength=0.6, columnspacing=0.3, handletextpad=0.2)
    label(ax, "A", "Spatial emergence")

    behavior = t["behavior_absolute"].rename(columns={"training_seed": "seed"}).copy()
    behavior["group"] = behavior.family
    for location, control, condition, heading, limits, unit in [
        (grid[0, 2:4], "random_info", "own_info", "Readout MLP layer 1", (0, 1.9), "Bits/activation"),
        (grid[0, 4:6], "random_reliability", "own_reliability", "Map reliability", (0, 0.9), "Split-half r"),
    ]:
        ax = fig.add_subplot(location)
        paired(ax, behavior, ["C01", "C05", "C15"], control, condition,
               vertical=True, limits=limits, ylabel=unit)
        label(ax, "B", heading)

    example = t["example"]
    mask = np.asarray(example["visited_bin_mask"], dtype=bool)
    assert mask.shape == (19, 19) and mask.sum() == 242
    grids = {arm: shared.peak_grid(np.asarray(example["arms"][arm]["peaks_xy"]),
                                   mask) for arm in ("temporal", "constant")}
    vmax = max(float(np.max(grid)) for grid in grids.values())
    for location, arm, title in [(grid[1, :2], "temporal", "Temporal"),
                                 (grid[1, 2:4], "constant", "Constant")]:
        ax = fig.add_subplot(location)
        cmap = plt.get_cmap("Greys").copy()
        cmap.set_bad("#DDDDDD")
        ax.imshow(np.ma.masked_where(~mask, grids[arm]), cmap=cmap,
                  vmin=0, vmax=vmax, origin="lower", interpolation="nearest")
        xy = np.asarray(example["arms"][arm]["peaks_xy"])
        ax.scatter((xy[:, 0] - 150) / 100, (xy[:, 1] - 150) / 100,
                   s=12, color=shared.COLOR["C05"], edgecolor="white", linewidth=0.25)
        ax.set_xticks([])
        ax.set_yticks([])
        label(ax, "C", f"{title} ({example['arms'][arm]['effective_area_fraction']:.3f})")
    ax = fig.add_subplot(grid[1, 4:6])
    ax.axis("off")
    ax.text(0, 0.92, "C01 seed 8", fontweight="bold", fontsize=10, transform=ax.transAxes)
    ax.text(0, 0.67, "Largest peak-area gain\namong five pairs.\nShared 10,001 observations.",
            fontsize=9, va="top", transform=ax.transAxes)

    encoder = tagged(t["encoder"], kind="encoder_fields")
    peaks = tagged(t["peaks"], kind="peaks")
    for location, source, metric, title, limits, unit in [
        (grid[2, :2], encoder, "mean_spatial_information_bits", "DG spatial score", (0, 0.24), "Weighted bits/step"),
        (grid[2, 2:4], encoder, "active_map_cosine_mean", "Map overlap", (0, 0.18), "Active-map cosine"),
        (grid[2, 4:6], peaks[peaks.cutoff == 0.0025], "effective_area_fraction", "Peak spread", (0, 0.50), "Effective area fraction"),
    ]:
        ax = fig.add_subplot(location)
        paired(ax, source[source.metric == metric], ENCODER_GROUPS,
               "constant", "temporal", vertical=True, limits=limits,
               ylabel=unit)
        label(ax, "D", title)

    layers = tagged(t["decoder"][(t["decoder"].factor == "encoder") &
                                  (t["decoder"].layer == "decoder_1")],
                    kind="encoder_layers")
    ax = fig.add_subplot(grid[3, :2])
    paired(ax, layers[layers.metric == "mean_bits_per_activation_eligible"],
           ENCODER_GROUPS, "constant", "temporal", vertical=True,
           limits=(0, 3.0), ylabel="Bits/activation")
    label(ax, "E", "Readout MLP layer 1")
    worker = tagged(t["worker"][(t["worker"].family == "C05") &
                                (t["worker"].contrast ==
                                 "temporal_minus_constant_hit_bonus")],
                    kind="worker_online")
    for location, metric, title, limits, unit in [
        (grid[3, 2:4], "visited_cell_fraction", "Worker: coverage", (0, 1), "Visited fraction"),
        (grid[3, 4:6], "spatial_information", "Worker: DG score", (0, 0.25), "Weighted bits/step"),
    ]:
        ax = fig.add_subplot(location)
        paired(ax, worker[worker.metric == metric], WORKER_GROUPS,
               "second", "first", vertical=True, limits=limits, ylabel=unit)
        label(ax, "F", title)

    longer = t["longer_credit_online"].merge(t["longer_credit_fields"],
                                              on=["dg_family", "seed"],
                                              validate="one_to_one").copy()
    longer["group"] = longer.dg_family.map({"Prescribed": "P", "C15": "C15"})
    for location, before, after, title, limits, unit in [
        (grid[4, :3], "baseline_coverage_auc", "longer_coverage_auc",
         "Longer PPO credit: coverage", (0, 100), "Coverage AUC (points)"),
        (grid[4, 3:6], "learned_si_bits_before", "learned_si_bits_after",
         "Longer PPO credit: DG information", (0, 0.24), "Weighted bits/step"),
    ]:
        ax = fig.add_subplot(location)
        paired(ax, longer, ["P", "C15"], before, after,
               vertical=True, limits=limits, ylabel=unit)
        label(ax, "F", title)
    fig.text(0.085, 0.982, "Candidate 1 · vertical paired values",
             fontsize=12, fontweight="bold", va="top")
    pair_legend(fig)
    return fig


def factor_matrix(t: dict) -> plt.Figure:
    """Expose all matched DG/layer effects, including weaker and negative ones."""
    fig = plt.figure(figsize=(7.5, 11.5), facecolor="white")
    grid = fig.add_gridspec(8, 2, left=0.17, right=0.98, bottom=0.045,
                           top=0.895, hspace=0.90, wspace=0.75)
    encoder = tagged(t["encoder"], kind="encoder_fields")
    worker_dg = tagged(t["worker_fields"], kind="worker_fields")
    layers = t["decoder"]
    layer_names = [("ca3", "CA3"), ("decoder_1", "Readout MLP 1"),
                   ("decoder_2", "Readout MLP 2")]
    specs = [
        ("DG spatial score", "mean_spatial_information_bits", "bits/step"),
        ("DG active-map cosine", "active_map_cosine_mean", "cosine"),
    ]
    for row, (title, metric, unit) in enumerate(specs):
        ax = fig.add_subplot(grid[row, 0])
        delta(ax, encoder[encoder.metric == metric], ENCODER_GROUPS,
              title=title, unit=unit)
        ax = fig.add_subplot(grid[row, 1])
        delta(ax, worker_dg[worker_dg.metric == metric], WORKER_GROUPS,
              title=title, unit=unit)
    for block, (metric, unit) in enumerate([
        ("mean_bits_per_activation_eligible", "bits/activation"),
        ("mean_amplitude_weighted_bits_all", "bits/step"),
    ]):
        for index, (layer, title) in enumerate(layer_names):
            row = 2 + block * 3 + index
            subset = layers[(layers.layer == layer) &
                            (layers.metric == metric)]
            encoder_layer = tagged(subset[subset.factor == "encoder"],
                                   kind="encoder_layers")
            worker_layer = tagged(subset[subset.factor == "worker"],
                                  kind="worker_layers")
            ax = fig.add_subplot(grid[row, 0])
            delta(ax, encoder_layer, ENCODER_GROUPS, title=title, unit=unit)
            ax = fig.add_subplot(grid[row, 1])
            delta(ax, worker_layer, WORKER_GROUPS, title=title, unit=unit)
    fig.text(0.17, 0.982, "Candidate 2 · full matched representation matrix",
             fontsize=12, fontweight="bold", va="top")
    fig.text(0.17, 0.95, "DG credit: temporal − constant\nworker setting fixed",
             fontsize=9, fontweight="bold", va="top")
    fig.text(0.60, 0.95, "Worker hit credit: temporal − constant\nencoder setting fixed",
             fontsize=9, fontweight="bold", va="top")
    return fig


def control_outcomes(t: dict) -> plt.Figure:
    """Separate behavioral, online, stored-graph, and long-credit protocols."""
    fig = plt.figure(figsize=(7.5, 8.7), facecolor="white")
    grid = fig.add_gridspec(4, 3, left=0.12, right=0.985, bottom=0.075,
                           top=0.90, hspace=0.85, wspace=0.87)
    behavior = t["behavior_absolute"].rename(columns={"training_seed": "seed"}).copy()
    behavior["group"] = behavior.family
    for col, before, after, title, limits, unit in [
        (0, "random_info", "own_info", "Readout MLP 1", (0, 1.9), "Bits/activation"),
        (1, "random_reliability", "own_reliability", "Map reliability", (0, 0.9), "Split-half r"),
    ]:
        ax = fig.add_subplot(grid[0, col])
        paired(ax, behavior, ["C01", "C05", "C15"], before, after,
               vertical=True, limits=limits, ylabel=unit)
        label(ax, "A", title)
    ax = fig.add_subplot(grid[0, 2])
    ax.axis("off")
    ax.text(0, 0.88, "Frozen-model test", fontsize=9,
            fontweight="bold", transform=ax.transAxes)
    ax.text(0, 0.67, "Random □ → policy ●\n38 models; two repeats.\n"
            "Actions change;\nweights stay fixed.",
            fontsize=8.5, va="top", transform=ax.transAxes)

    worker = tagged(t["worker"][(t["worker"].family == "C05") &
                                (t["worker"].contrast ==
                                 "temporal_minus_constant_hit_bonus")],
                    kind="worker_online")
    worker_fields = tagged(t["worker_fields"], kind="worker_fields")
    for col, source, metric, title, before, after, limits, unit in [
        (0, worker, "visited_cell_fraction", "Online coverage", "second", "first", (0, 1), "Visited fraction"),
        (1, worker, "spatial_information", "Online DG coding", "second", "first", (0, 0.25), "Weighted bits/step"),
        (2, worker_fields, "mean_spatial_information_bits", "DG replay", "constant", "temporal", (0, 0.25), "Weighted bits/step"),
    ]:
        ax = fig.add_subplot(grid[1, col])
        paired(ax, source[source.metric == metric], WORKER_GROUPS,
               before, after, vertical=True, limits=limits, ylabel=unit)
        label(ax, "B", title)

    outcomes = tagged(t["worker_outcomes"], kind="worker_outcomes")
    for col, source, metric, title, limits, unit in [
        (0, "stored_graph_100m", "graph_reachable_pair_fraction", "Graph reachability", (0, 0.15), "Reachable-pair fraction"),
        (1, "stored_graph_100m", "prospective_success_fraction", "Graph prospects", (0, 1), "Prospective success fraction"),
        (2, "trajectory_100m", "path_efficiency", "Path efficiency", (0, 1), "Path efficiency"),
    ]:
        subset = outcomes[(outcomes.source == source) & (outcomes.metric == metric)]
        ax = fig.add_subplot(grid[2, col])
        paired(ax, subset, WORKER_GROUPS, "constant_worker", "temporal_worker",
               vertical=True, limits=limits, ylabel=unit)
        label(ax, "C", title)

    longer = t["longer_credit_online"].merge(t["longer_credit_fields"],
                                              on=["dg_family", "seed"],
                                              validate="one_to_one").copy()
    longer["group"] = longer.dg_family.map({"Prescribed": "P", "C15": "C15"})
    for col, before, after, title, limits, unit in [
        (0, "baseline_coverage_auc", "longer_coverage_auc", "Coverage AUC", (0, 100), "AUC (points)"),
        (1, "baseline_hits_per_1000", "longer_hits_per_1000", "Detector hits", (0.1, 30), "Hits / 1000 (log scale)"),
        (2, "learned_si_bits_before", "learned_si_bits_after", "Frozen DG", (0, 0.24), "Weighted bits/step"),
    ]:
        ax = fig.add_subplot(grid[3, col])
        paired(ax, longer, ["P", "C15"], before, after,
               vertical=True, limits=limits, ylabel=unit)
        if col == 1:
            ax.set_yscale("log")
        label(ax, "D", title)
    fig.text(0.12, 0.982, "Candidate 3 · behavior and control outcomes",
             fontsize=12, fontweight="bold", va="top")
    pair_legend(fig)
    return fig


def inventory_records(t: dict) -> pd.DataFrame:
    """Enumerate published-table contrasts; preserve signed means in a CSV.

    The heatmap displays positive-seed counts rather than incomparable effect
    magnitudes. Its rows cover metric tables, not raw telemetry channels.
    """
    pieces = []

    def add(block: str, data: pd.DataFrame, *, metric_column: str = "metric",
            difference_column: str = "difference", source: str) -> None:
        for (metric, group), rows in data.groupby([metric_column, "group"], sort=False):
            differences = rows[difference_column].to_numpy(dtype=float)
            assert np.isfinite(differences).all()
            pieces.append({
                "block": block, "metric": str(metric), "group": str(group),
                "raw_metric": str(rows.metric.iloc[0]) if "metric" in rows else str(metric),
                "n": len(differences), "positive": int((differences > 0).sum()),
                "negative": int((differences < 0).sum()),
                "ties": int((differences == 0).sum()),
                "mean_difference": float(differences.mean()),
                "source": source,
            })

    online = t["worker"]
    online_encoder = online[online.contrast == "temporal_minus_constant_encoder"].copy()
    online_encoder["group"] = online_encoder.apply(
        lambda row: f"C05-w{row.fixed_factor[0].upper()}"
        if row.family == "C05" else row.family, axis=1)
    online_worker = tagged(online[online.contrast ==
                                  "temporal_minus_constant_hit_bonus"],
                           kind="worker_online")
    add("Online", pd.concat([online_encoder, online_worker]), source="worker")
    add("DG replay", pd.concat([
        tagged(t["encoder"], kind="encoder_fields"),
        tagged(t["worker_fields"], kind="worker_fields")]),
        source="encoder + worker_fields")
    peaks = tagged(t["peaks"], kind="peaks").copy()
    peaks["metric_and_cutoff"] = peaks.apply(
        lambda row: f"{row.metric.replace('_', ' ')} @ {row.cutoff * 100:g}%",
        axis=1)
    add("Peak spread", peaks, metric_column="metric_and_cutoff", source="peaks")
    layers = t["decoder"]
    encoder_layer = tagged(layers[layers.factor == "encoder"],
                           kind="encoder_layers")
    worker_layer = tagged(layers[layers.factor == "worker"],
                          kind="worker_layers")
    interaction = layers[layers.factor == "interaction"].copy()
    interaction["group"] = "C05-int"
    all_layers = pd.concat([encoder_layer, worker_layer, interaction])
    layer_labels = {"ca3": "CA3", "decoder_1": "Readout MLP layer 1",
                    "decoder_2": "Readout MLP layer 2"}
    metric_labels = {
        "mean_bits_per_activation_eligible": "bits/activation",
        "mean_amplitude_weighted_bits_all": "weighted bits/step",
        "mean_positive_fraction": "positive fraction",
        "mean_rectified_activation": "mean response",
        "eligible_units": "eligible units", "silent_units": "silent units",
    }
    all_layers["layer_metric"] = (all_layers.layer.map(layer_labels) + " · " +
                                  all_layers.metric.map(metric_labels))
    add("CA3 / readout", all_layers, metric_column="layer_metric",
        source="decoder")
    outcomes = tagged(t["worker_outcomes"], kind="worker_outcomes")
    source_labels = {
        "correct_hit_90_100m": "Hit window", "frozen_behavior_100m": "Frozen behavior",
        "online_90_99m": "Online", "same_observations_100m": "Common replay",
        "stored_graph_100m": "Stored graph", "trajectory_100m": "Trajectory",
    }
    outcome_labels = {
        "censored_observation_count": "censored observations",
        "correct_elapsed_count_weighted": "elapsed hit count",
        "correct_hit_observation_count": "detector hit count",
        "correct_reward_magnitude_count_weighted": "weighted hit count",
        "common_split_half_difference": "policy reliability effect",
        "information_difference": "policy information effect",
        "visited_cell_fraction": "visited fraction",
        "active_map_cosine_mean": "active-map cosine",
        "mean_spatial_information_bits": "DG spatial score",
        "graph_reachable_pair_fraction": "reachable pairs",
        "prospective_success_fraction": "prospective success",
        "path_efficiency": "path efficiency",
    }
    outcomes["source_metric"] = (outcomes.source.map(source_labels) + " · " +
                                 outcomes.metric.map(outcome_labels))
    add("C05 worker outcomes", outcomes, metric_column="source_metric",
        source="worker_outcomes")
    behavior = t["behavior"][(t["behavior"].cohort == "fresh") &
                             (t["behavior"].layer == "decoder_1")].copy()
    behavior["group"] = behavior.family
    for metric in ("information_difference", "common_split_half_difference"):
        data = behavior[["group", metric]].dropna().copy()
        data["metric"] = metric
        data["difference"] = data[metric]
        add("Behavior", data, source="behavior")
    longer = t["longer_credit_online"].merge(t["longer_credit_fields"],
                                              on=["dg_family", "seed"],
                                              validate="one_to_one").copy()
    longer["group"] = longer.dg_family.map(
        {"Prescribed": "P-long", "C15": "C15-long"})
    long_pairs = [
        ("coverage_auc", "baseline_coverage_auc", "longer_coverage_auc"),
        ("hits_per_1000", "baseline_hits_per_1000", "longer_hits_per_1000"),
        ("goal_action_tv", "baseline_goal_action_tv", "longer_goal_action_tv"),
        ("visited_cells", "visited_cells_before", "visited_cells_after"),
        ("learned_si_bits", "learned_si_bits_before", "learned_si_bits_after"),
        ("active_map_cosine", "active_map_cosine_before", "active_map_cosine_after"),
        ("mono_field_count", "mono_field_count_before", "mono_field_count_after"),
        ("eligible_rows", "eligible_rows_before", "eligible_rows_after"),
    ]
    for metric, before, after in long_pairs:
        data = longer[["group", before, after]].copy()
        data["metric"] = metric
        data["difference"] = data[after] - data[before]
        add("Longer PPO credit", data, source="longer_credit_online + fields")
    records = pd.DataFrame(pieces)
    assert records.duplicated(["block", "metric", "group"]).sum() == 0
    return records


INVENTORY_COLUMNS = {
    "Online": ["C01", "C05-wT", "C05-wC", "C15", "C05-eT", "C05-eC"],
    "DG replay": ["C01", "C05-wT", "C05-wC", "C15", "C05-eT", "C05-eC"],
    "Peak spread": ["C01", "C05-wT", "C05-wC", "C15"],
    "CA3 / readout": ["C01", "C05-wT", "C05-wC", "C15", "C05-eT", "C05-eC", "C05-int"],
    "C05 worker outcomes": ["C05-eT", "C05-eC"],
    "Behavior": ["C01", "C05", "C15"],
    "Longer PPO credit": ["P-long", "C15-long"],
}


def inventory_atlas(records: pd.DataFrame) -> plt.Figure:
    """Tall audit view of every contrast metric in the selected source tables."""
    blocks = list(INVENTORY_COLUMNS)
    metric_lists = {block: sorted(records[records.block == block].metric.unique())
                    for block in blocks}
    heights = [len(metric_lists[block]) + 2 for block in blocks]
    fig = plt.figure(figsize=(7.5, 24.0), facecolor="white")
    grid = fig.add_gridspec(len(blocks), 1, left=0.43, right=0.98,
                           top=0.925, bottom=0.035, hspace=0.45,
                           height_ratios=heights)
    cmap = ListedColormap(["#C97155", "#EDD3C9", "#E7EAEB",
                           "#B9D9E8", "#4C9AC2"])
    norm = BoundaryNorm([-0.01, 0.2, 0.4, 0.6, 0.8, 1.01], cmap.N)
    for position, block in enumerate(blocks):
        metrics = metric_lists[block]
        groups = INVENTORY_COLUMNS[block]
        matrix = np.full((len(metrics), len(groups)), np.nan)
        annotations = np.full(matrix.shape, "—", dtype=object)
        subset = records[records.block == block]
        for _, row in subset.iterrows():
            r = metrics.index(row.metric)
            c = groups.index(row.group)
            matrix[r, c] = (row.positive - row.negative) / (2 * row.n) + 0.5
            annotations[r, c] = f"{row.positive}/{row.n}"
        ax = fig.add_subplot(grid[position])
        ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm,
                  aspect="auto", interpolation="nearest")
        ax.set_facecolor("#F2F3F3")
        ax.set_yticks(range(len(metrics)), metrics, fontsize=8.5)
        ax.set_xticks(range(len(groups)), groups, fontsize=8.5)
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=3)
        ax.set_title(block, loc="left", fontsize=10, fontweight="bold", pad=22)
        ax.set_xticks(np.arange(-0.5, len(groups), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(metrics), 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row in range(len(metrics)):
            for col in range(len(groups)):
                ax.text(col, row, annotations[row, col], ha="center", va="center",
                        fontsize=7.5, color="#20272C")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(0.03, 0.99, "Candidate 4 · archived metric direction inventory",
             fontsize=12, fontweight="bold", va="top")
    fig.text(0.03, 0.975,
             "Cell: positive / measured pairs. Color: signed count imbalance; ties neutral.\n"
             "Blank = unmeasured. Direction is not effect size or benefit.",
             fontsize=9, va="top")
    fig.text(0.03, 0.020,
             "Each block has its own intervention and sampling protocol. "
             "Exact signed means, ties, and source keys are in metric_inventory.csv.",
             fontsize=8.5)
    return fig


def export(fig: plt.Figure, stem: str) -> dict:
    files = {}
    hashes = {}
    for extension in ("pdf", "svg", "png"):
        path = HERE / f"{stem}.{extension}"
        metadata = ({"Date": None} if extension == "svg" else
                    {"CreationDate": None, "ModDate": None} if extension == "pdf" else
                    None)
        fig.savefig(path, dpi=300, metadata=metadata, facecolor="white")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in
                                      path.read_text().splitlines()) + "\n")
        files[extension] = path.name
        hashes[extension] = hashlib.sha256(path.read_bytes()).hexdigest()
    size = [float(item) for item in fig.get_size_inches()]
    plt.close(fig)
    return {"files": files, "sha256": hashes, "inches": size}


def main() -> None:
    font = shared.style()
    tables = load_tables()
    validate_selected_evidence(tables)
    figures = {
        "01_vertical_story": export(vertical_story(tables), "01_vertical_story"),
        "02_factor_matrix": export(factor_matrix(tables), "02_factor_matrix"),
        "03_control_outcomes": export(control_outcomes(tables), "03_control_outcomes"),
    }
    inventory = inventory_records(tables)
    inventory.to_csv(HERE / "metric_inventory.csv", index=False)
    figures["04_metric_inventory"] = export(
        inventory_atlas(inventory), "04_metric_inventory")
    for stem, mode in (("05_compact_vertical_pairs", "paired"),
                       ("06_compact_vertical_boxes", "box")):
        shared.results_figure(mode=mode, output_dir=HERE, stem=stem)
        files = {extension: f"{stem}.{extension}"
                 for extension in ("pdf", "svg", "png")}
        figures[stem] = {
            "files": files,
            "sha256": {extension: hashlib.sha256((HERE / filename).read_bytes()).hexdigest()
                       for extension, filename in files.items()},
            "inches": [7.5, shared.FIGURE_HEIGHT],
        }
    manifest = {
        "purpose": "Exploratory Figure 2 alternatives; excluded from the Overleaf ZIP",
        "font": font,
        "software": {"matplotlib": plt.matplotlib.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "source_paths": {key: str(path.relative_to(shared.ROOT))
                         for key, path in SOURCE_PATHS.items()},
        "source_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                          for key, path in SOURCE_PATHS.items()},
        "figures": figures,
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"font": font, "figures": figures}, indent=2))


if __name__ == "__main__":
    main()
