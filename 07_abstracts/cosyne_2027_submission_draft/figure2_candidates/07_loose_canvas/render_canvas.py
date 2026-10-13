"""Export Figure 2 as independent, uniformly sized evidence components.

This is an asset shelf for author-led layout, not another proposed Figure 2.
The ten compact box plots use the exact panel data of candidate 06. Additional
plots draw from archived gallery tables and pinned high-credit worker results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
GALLERY = HERE.parent
sys.path.insert(0, str(GALLERY))
import render_candidates as gallery  # noqa: E402

shared = gallery.shared
COMPONENTS = HERE / "components"
PLOT_WIDTH = 1.20  # Complete plot, including tick and axis labels; no title.
PLOT_HEIGHT = 1.00
DATA_AXES_LEFT = 0.93
DATA_AXES_BOTTOM = 0.47
DATA_AXES_WIDTH = 0.695
DATA_AXES_HEIGHT = 0.66
CARD_WIDTH = 2.25
CARD_HEIGHT = 1.95
EXAMPLE_WIDTH = 2.25
EXAMPLE_HEIGHT = 2.15
MAP_WIDTH = 2.25
MAP_HEIGHT = 2.15
PUSH_GROUPS = ("C01", "C05-wT", "C15")
LEGACY_WORKER_TIMING_SLUG = "F12_high_credit_worker_timing"
HIGH_CREDIT_DIRECTORY = (
    shared.ROOT / "06_experiments" / "reward_credit" / "results" /
    "cosyne27_worker_regimen_boxplots_20261013"
)
HIGH_CREDIT_SOURCES = {
    "paired": HIGH_CREDIT_DIRECTORY / "paired_seed_differences.csv",
    "hit_rates": HIGH_CREDIT_DIRECTORY / "high_credit_hit_rate_pairs.csv",
    "selected_figure": HIGH_CREDIT_DIRECTORY /
    "high_credit_worker_shared_hit_rate_dual_edges_compact.svg",
}
HIGH_CREDIT_SEEDS = (8, 17, 41, 99, 123)


@dataclass(frozen=True)
class BoxSpec:
    slug: str
    title: str
    section: str
    data: pd.DataFrame
    groups: tuple[str, ...]
    control: str
    tested: str
    unit: str
    contrast: str
    protocol: str
    source_keys: tuple[str, ...]
    original: bool = True
    raw_pair_example: bool = True


def rows(tables: dict) -> list[BoxSpec]:
    """Select measurements without parsing factor identities from run names."""
    specs: list[BoxSpec] = []
    behavior = tables["behavior_absolute"].rename(
        columns={"training_seed": "seed"}).copy()
    behavior["group"] = behavior.family
    encoder = gallery.tagged(tables["encoder"], kind="encoder_fields")
    peaks = gallery.tagged(tables["peaks"], kind="peaks")
    peaks = peaks[np.isclose(peaks.cutoff, 0.0025)]
    decoder = tables["decoder"]
    encoder_layers = gallery.tagged(decoder[decoder.factor == "encoder"],
                                    kind="encoder_layers")
    worker_layers = gallery.tagged(decoder[decoder.factor == "worker"],
                                   kind="worker_layers")
    worker = tables["worker"]
    worker = worker[(worker.family == "C05") &
                    (worker.contrast == "temporal_minus_constant_hit_bonus")]
    worker = gallery.tagged(worker, kind="worker_online")
    worker_fields = gallery.tagged(tables["worker_fields"],
                                   kind="worker_fields")
    worker_outcomes = gallery.tagged(tables["worker_outcomes"],
                                     kind="worker_outcomes")
    longer = tables["longer_credit_online"].merge(
        tables["longer_credit_fields"], on=["dg_family", "seed"],
        validate="one_to_one")
    longer["group"] = longer.dg_family.map({"Prescribed": "P", "C15": "C15"})

    def add(slug: str, title: str, section: str, data: pd.DataFrame,
            groups: tuple[str, ...] | list[str], control: str, tested: str, unit: str,
            contrast: str, protocol: str, source_keys: tuple[str, ...],
            *, original: bool = True, raw_pair_example: bool = True) -> None:
        # The source table retains C05-wC. This inventory displays the three
        # requested push-ablation arms, with C05-wT labeled simply C05.
        subset = data[data.group.isin(groups)].copy()
        assert subset.groupby("group").size().reindex(groups).notna().all(), slug
        identity = ["condition", "seed"] if section == "B" else ["group", "seed"]
        assert not subset.duplicated(identity).any(), slug
        assert np.isfinite(subset[[control, tested]].to_numpy(dtype=float)).all(), slug
        if "difference" in subset:
            assert np.allclose(subset[tested] - subset[control],
                               subset.difference, atol=1e-9), slug
        assert subset.groupby("group").size().reindex(groups).tolist() == (
            [10, 20, 8] if section == "B" else
            [3, 3] if contrast == "joint_longer_ppo_gae" else
            [5] * len(groups)), slug
        specs.append(BoxSpec(slug, title, section, subset, tuple(groups),
                             control, tested, unit, contrast, protocol,
                             source_keys, original, raw_pair_example))

    add("B01_readout_policy_information", "Readout coding under the learned policy",
        "B", behavior, ["C01", "C05", "C15"], "random_info", "own_info",
        "bits/activation", "own_policy_minus_random_actions", "frozen_policy",
        ("behavior_absolute", "behavior"), raw_pair_example=True)
    add("B02_policy_map_reliability", "Spatial-map reliability under policy",
        "B", behavior, ["C01", "C05", "C15"],
        "random_reliability", "own_reliability", "split-half r",
        "own_policy_minus_random_actions", "frozen_policy",
        ("behavior_absolute", "behavior"), raw_pair_example=True)

    for slug, title, key, unit in [
        ("D01_dg_information", "DG spatial coding", "mean_spatial_information_bits", "bits/step"),
        ("D02_map_overlap", "DG map overlap", "active_map_cosine_mean", "cosine"),
    ]:
        add(slug, title, "D", encoder[encoder.metric == key],
            PUSH_GROUPS, "constant", "temporal", unit,
            "temporal_minus_constant_encoder", "common_replay", ("encoder",),
            raw_pair_example=True)
    add("D03_peak_area", "Field-peak coverage", "D",
        peaks[peaks.metric == "effective_area_fraction"],
        PUSH_GROUPS, "constant", "temporal", "area fraction",
        "temporal_minus_constant_encoder", "common_replay", ("peaks", "example"),
        raw_pair_example=True)
    add("E01_readout1_information", "Readout layer 1 coding", "E",
        encoder_layers[(encoder_layers.layer == "decoder_1") &
                       (encoder_layers.metric == "mean_bits_per_activation_eligible")],
        PUSH_GROUPS, "constant", "temporal", "bits/activation",
        "temporal_minus_constant_encoder", "common_replay", ("decoder",),
        raw_pair_example=True)

    for slug, title, key, unit in [
        ("F01_worker_coverage", "Worker credit: visited coverage", "visited_cell_fraction", "fraction"),
        ("F02_worker_online_dg", "Worker credit: online DG coding", "spatial_information", "bits/step"),
    ]:
        add(slug, title, "F", worker[worker.metric == key],
            gallery.WORKER_GROUPS, "second", "first", unit,
            "temporal_minus_constant_worker", "online_90_99m", ("worker",),
            raw_pair_example=True)
    add("F03_longer_ppo_coverage", "Longer PPO credit: coverage", "F", longer,
        ["P", "C15"], "baseline_coverage_auc", "longer_coverage_auc",
        "AUC points", "joint_longer_ppo_gae", "online_100m",
        ("longer_credit_online", "longer_credit_fields"), raw_pair_example=True)
    add("F04_longer_ppo_dg", "Longer PPO credit: frozen DG coding", "F", longer,
        ["P", "C15"], "learned_si_bits_before", "learned_si_bits_after",
        "bits/step", "joint_longer_ppo_gae", "policy_rollout_100m",
        ("longer_credit_online", "longer_credit_fields"), raw_pair_example=True)

    # Candidate plots whose data existed but were omitted from compact 06.
    for slug, title, key, unit in [
        ("D04_dg_activity", "DG activity fraction", "mean_active_fraction", "fraction"),
        ("D05_prethreshold_overlap", "Prethreshold DG map overlap", "prethreshold_map_cosine_mean", "cosine"),
    ]:
        add(slug, title, "D", encoder[encoder.metric == key],
            PUSH_GROUPS, "constant", "temporal", unit,
            "temporal_minus_constant_encoder", "common_replay", ("encoder",),
            original=False)
    for slug, title, key, unit in [
        ("D06_peak_gini", "Unevenness of peak coverage", "peak_density_gini", "Gini"),
        ("D07_peak_separation", "Separation of field peaks", "mean_pairwise_peak_distance", "DMLab units"),
    ]:
        add(slug, title, "D", peaks[peaks.metric == key],
            PUSH_GROUPS, "constant", "temporal", unit,
            "temporal_minus_constant_encoder", "common_replay", ("peaks",),
            original=False)
    for slug, title, layer, metric, unit in [
        ("E02_ca3_information", "CA3 coding per activation", "ca3", "mean_bits_per_activation_eligible", "bits/activation"),
        ("E03_readout2_information", "Readout layer 2 coding", "decoder_2", "mean_bits_per_activation_eligible", "bits/activation"),
        ("E04_ca3_weighted", "CA3 amplitude-weighted coding", "ca3", "mean_amplitude_weighted_bits_all", "bits/step"),
        ("E05_readout1_weighted", "Readout layer 1 weighted coding", "decoder_1", "mean_amplitude_weighted_bits_all", "bits/step"),
        ("E06_readout2_weighted", "Readout layer 2 weighted coding", "decoder_2", "mean_amplitude_weighted_bits_all", "bits/step"),
    ]:
        subset = encoder_layers[(encoder_layers.layer == layer) &
                                (encoder_layers.metric == metric)]
        add(slug, title, "E", subset, PUSH_GROUPS,
            "constant", "temporal", unit, "temporal_minus_constant_encoder",
            "common_replay", ("decoder",), original=False)
    add("F05_worker_replay_dg", "Worker credit: replayed DG coding", "F",
        worker_fields[worker_fields.metric == "mean_spatial_information_bits"],
        gallery.WORKER_GROUPS, "constant", "temporal", "bits/step",
        "temporal_minus_constant_worker", "common_replay", ("worker_fields",),
        original=False)
    for slug, title, metric, protocol, unit in [
        ("F06_graph_reachability", "Stored-graph reachability", "graph_reachable_pair_fraction", "stored_graph_100m", "fraction"),
        ("F07_graph_prospects", "Stored-graph prospective success", "prospective_success_fraction", "stored_graph_100m", "fraction"),
        ("F08_path_efficiency", "Sampled-path efficiency", "path_efficiency", "trajectory_100m", "fraction"),
    ]:
        subset = worker_outcomes[(worker_outcomes.metric == metric) &
                                 (worker_outcomes.source == protocol)]
        add(slug, title, "F", subset, gallery.WORKER_GROUPS,
            "constant_worker", "temporal_worker", unit,
            "temporal_minus_constant_worker", protocol,
            ("worker_outcomes",), original=False)
    add("F09_longer_ppo_hits", "Longer PPO credit: detector hits", "F", longer,
        ["P", "C15"], "baseline_hits_per_1000", "longer_hits_per_1000",
        "hits/1,000", "joint_longer_ppo_gae", "online_100m",
        ("longer_credit_online",), original=False)
    for slug, title, layer in [
        ("F10_worker_readout1_weighted", "Worker credit: readout layer 1", "decoder_1"),
        ("F11_worker_readout2_weighted", "Worker credit: readout layer 2", "decoder_2"),
    ]:
        subset = worker_layers[(worker_layers.layer == layer) &
                               (worker_layers.metric == "mean_amplitude_weighted_bits_all")]
        add(slug, title, "F", subset, gallery.WORKER_GROUPS,
            "constant", "temporal", "bits/step",
            "temporal_minus_constant_worker", "common_replay", ("decoder",),
            original=False)
    return specs


def export(fig: plt.Figure, slug: str) -> dict[str, str]:
    COMPONENTS.mkdir(parents=True, exist_ok=True)
    files = {}
    for suffix in ("svg", "pdf", "png"):
        path = COMPONENTS / f"{slug}.{suffix}"
        metadata = ({"Date": None} if suffix == "svg" else
                    {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None)
        fig.savefig(path, dpi=300, metadata=metadata, facecolor="white")
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in
                                      path.read_text().splitlines()) + "\n")
        files[suffix] = hashlib.sha256(path.read_bytes()).hexdigest()
    plt.close(fig)
    return files


def card_title(fig: plt.Figure, title: str, *, original: bool) -> None:
    words = title.split()
    lines = []
    current = []
    for word in words:
        if len(" ".join(current + [word])) > 20 and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    lines.append(" ".join(current))
    fig.text(0.045, 0.97, "\n".join(lines), ha="left", va="top",
             fontsize=12, fontweight="bold", color="#20272C")
    if not original:
        fig.text(0.96, 0.97, "+", ha="right", va="top",
                 fontsize=13, fontweight="bold", color="#5B646C")


def box_plot(spec: BoxSpec) -> dict[str, str]:
    fig = plt.figure(figsize=(CARD_WIDTH, CARD_HEIGHT), facecolor="white")
    card_title(fig, spec.title, original=spec.original)
    ax = fig.add_axes([DATA_AXES_LEFT / CARD_WIDTH,
                       DATA_AXES_BOTTOM / CARD_HEIGHT,
                       DATA_AXES_WIDTH / CARD_WIDTH,
                       DATA_AXES_HEIGHT / CARD_HEIGHT])
    shared.paired_rows(ax, spec.data, list(spec.groups), spec.control,
                       spec.tested, labels=True, xlim=(0, 1),
                       marker_size=15, mode="box")
    for dots in ax.collections:
        dots.set_alpha(0.7)
        dots.set_zorder(2)
    for box in ax.patches:
        box.set_zorder(4)
    for line in ax.lines[:-1]:  # box edges, whiskers, caps, and medians
        line.set_zorder(5)
    ax.lines[-1].set_zorder(1)  # zero reference line
    tick_names = {"C05-wT": "C05", "C05-wC": "wC", "C05-eT": "eT",
                  "C05-eC": "eC"}
    ax.set_xticks(range(len(spec.groups)),
                  [tick_names.get(group, group) for group in spec.groups])
    ax.tick_params(axis="x", labelsize=8, rotation=55, length=2, pad=1)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.tick_params(axis="y", labelsize=8, length=2, pad=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.set_ylabel(r"$\Delta$ " + spec.unit, fontsize=10, labelpad=-1.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    plot_bounds = ax.get_tightbbox(renderer).transformed(
        fig.dpi_scale_trans.inverted())
    title_bounds = fig.texts[0].get_window_extent(renderer).transformed(
        fig.dpi_scale_trans.inverted())
    if (plot_bounds.width > PLOT_WIDTH + 0.001 or
            plot_bounds.height > PLOT_HEIGHT + 0.001 or
            plot_bounds.y1 >= title_bounds.y0):
        raise ValueError(
            f"{spec.slug}: box plot with labels measures "
            f"{plot_bounds.width:.3f} × {plot_bounds.height:.3f} in "
            "or overlaps its title")
    return export(fig, spec.slug)


def raw_pair_plot(spec: BoxSpec) -> dict[str, str]:
    """Show the un-subtracted observations behind one original box plot."""
    fig = plt.figure(figsize=(EXAMPLE_WIDTH, EXAMPLE_HEIGHT), facecolor="white")
    card_title(fig, spec.title, original=spec.original)
    ax = fig.add_axes([0.66 / EXAMPLE_WIDTH, 0.50 / EXAMPLE_HEIGHT,
                       1.13 / EXAMPLE_WIDTH, 1.13 / EXAMPLE_HEIGHT])
    values = spec.data[[spec.control, spec.tested]].to_numpy(dtype=float)
    low, high = float(values.min()), float(values.max())
    margin = max((high - low) * 0.07, 0.01)
    low -= margin
    high += margin
    ax.plot([low, high], [low, high], color="#7D858A", lw=0.8, zorder=1)
    for group in spec.groups:
        group_rows = spec.data[spec.data.group == group]
        marker = "D" if group.endswith(("wC", "eC")) else "o"
        ax.scatter(group_rows[spec.control], group_rows[spec.tested],
                   s=20, marker=marker,
                   color=shared.COLOR[group.split("-")[0]],
                   linewidth=0.25, edgecolor="white", zorder=2)
    ax.set(xlim=(low, high), ylim=(low, high))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Control", fontsize=12, labelpad=1)
    ax.set_ylabel("Tested", fontsize=12, labelpad=1)
    ax.tick_params(labelsize=12, length=2, pad=1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.spines[["top", "right"]].set_visible(False)
    return export(fig, spec.slug + "_raw_pairs")


def online_course(tables: dict) -> dict[str, str]:
    fig = plt.figure(figsize=(2.9, 2.15), facecolor="white")
    card_title(fig, "Spatial emergence", original=True)
    ax = fig.add_axes([0.63 / 2.9, 0.46 / 2.15, 1.76 / 2.9, 1.22 / 2.15])
    online = tables["online"]
    for family, worker_arm in (("C01", "inapplicable"), ("C05", "temporal"),
                               ("C15", "temporal")):
        group = online[(online.family == family) &
                       (online.encoder_credit == "temporal") &
                       (online.worker_bonus == worker_arm) &
                       (online.metric == "spatial_information")].sort_values("env_steps")
        assert len(group) > 50 and group.n.eq(5).all()
        x = group.env_steps.to_numpy() / 1e6
        mean = group["mean"].to_numpy()
        sem = group["sem"].to_numpy()
        ax.plot(x, mean, color=shared.COLOR[family], lw=1.2, label=family)
        ax.fill_between(x, mean - sem, mean + sem, color=shared.COLOR[family], alpha=.15)
    ax.set(xlim=(0, 100), xlabel="Frames (millions)", ylabel="DG bits/step")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left",
               bbox_to_anchor=(0.22, 0.86), frameon=False, ncol=3,
               fontsize=12, columnspacing=.3, handlelength=.6,
               handletextpad=.2, borderpad=0, labelspacing=.1)
    ax.tick_params(labelsize=12, length=2, pad=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E6E8E9", lw=.5)
    return export(fig, "A01_spatial_emergence")


def archived_tiles() -> list[dict]:
    """Keep source tiles and their separate-run provenance visible."""
    output = []
    for title, tile in shared.archived_example_tiles().items():
        fig = plt.figure(figsize=(MAP_WIDTH, MAP_HEIGHT), facecolor="white")
        card_title(fig, f"Archived C01 S8: {title.title()}", original=True)
        ax = fig.add_axes([0.46 / MAP_WIDTH, 0.30 / MAP_HEIGHT,
                           1.30 / MAP_WIDTH, 1.30 / MAP_HEIGHT])
        ax.imshow(tile, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        slug = {"trajectory": "C01_archived_trajectory", "DG 11": "C02_archived_dg11",
                "DG 0": "C03_archived_dg0"}[title]
        output.append({"slug": slug, "kind": "archived_tile", "hashes": export(fig, slug),
                       "sources": ["archived_trajectory" if title == "trajectory"
                                   else "archived_place_fields"]})
    return output


def peak_scatter(tables: dict, arm: str) -> dict[str, str]:
    """Show the actual unit-peak coordinates underlying the peak-area example."""
    example = tables["example"]
    mask = np.asarray(example["visited_bin_mask"], dtype=bool)
    locations = np.asarray(example["arms"][arm]["peaks_xy"], dtype=float)
    density = shared.peak_grid(locations, mask)
    common_vmax = max(float(shared.peak_grid(
        np.asarray(example["arms"][name]["peaks_xy"], dtype=float), mask).max())
        for name in ("temporal", "constant"))
    fig = plt.figure(figsize=(MAP_WIDTH, MAP_HEIGHT), facecolor="white")
    card_title(fig, f"C01 S8 peaks: {arm}", original=True)
    ax = fig.add_axes([0.46 / MAP_WIDTH, 0.30 / MAP_HEIGHT,
                       1.30 / MAP_WIDTH, 1.30 / MAP_HEIGHT])
    cmap = plt.get_cmap("Greys").copy()
    cmap.set_bad("#DDDDDD")
    ax.imshow(np.ma.masked_where(~mask, density), origin="lower",
              extent=(100, 2000, 100, 2000), cmap=cmap, interpolation="nearest",
              vmin=0, vmax=common_vmax)
    ax.scatter(locations[:, 0], locations[:, 1], s=17,
               color=shared.COLOR["C01"] if arm == "temporal" else shared.COLOR["C05"],
               edgecolor="white", linewidth=.35)
    ax.set(xlim=(100, 2000), ylim=(100, 2000))
    ax.set_xticks([])
    ax.set_yticks([])
    area = example["arms"][arm]["effective_area_fraction"]
    fig.text(.96, .025, f"{len(locations)} peaks · area {area:.3f}",
             ha="right", va="bottom", fontsize=12, color="#5B646C")
    return export(fig, f"C04_peaks_{arm}")


def text_elements() -> list[dict]:
    """Separate section headings and legend for drag-and-drop composition."""
    items = []
    headings = [
        ("A", "Spatial emergence"),
        ("B", "Behavior shapes coding"),
        ("C", "Spatial examples"),
        ("D", "Temporal DG credit"),
        ("E", "Downstream readout"),
        ("F", "Worker and PPO credit"),
    ]
    for letter, title in headings:
        fig = plt.figure(figsize=(3.25, .55), facecolor="white")
        fig.text(.04, .55, f"{letter}  {title}", ha="left", va="center",
                 fontsize=12, fontweight="bold", color="#20272C")
        slug = f"label_{letter.lower()}"
        items.append({"slug": slug, "section": letter, "kind": "section_label",
                      "original": True, "sources": [], "hashes": export(fig, slug)})
    fig = plt.figure(figsize=(3.0, .7), facecolor="white")
    for x, y, family, title in [
        (.08, .68, "C01", "C01"), (.47, .68, "C05", "C05"),
        (.08, .28, "C15", "C15"), (.47, .28, "P", "Prescribed")]:
        fig.text(x, y, "●", color=shared.COLOR[family], fontsize=14,
                 ha="left", va="center")
        fig.text(x + .09, y, title, color="#20272C", fontsize=12,
                 ha="left", va="center")
    slug = "label_color_key"
    items.append({"slug": slug, "section": "key", "kind": "legend",
                  "original": True, "sources": [], "hashes": export(fig, slug)})
    return items


def high_credit_rows() -> pd.DataFrame:
    """Load the pinned high-credit paired differences without inventing raw arms."""
    paired = pd.read_csv(HIGH_CREDIT_SOURCES["paired"])
    paired = paired[paired.policy_credit == "high"].copy()
    hits = pd.read_csv(HIGH_CREDIT_SOURCES["hit_rates"])
    expected = {(family, seed) for family in ("C05", "C15")
                for seed in HIGH_CREDIT_SEEDS}
    for name, frame in (("paired", paired), ("hit_rates", hits)):
        keys = list(zip(frame.family, frame.seed))
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError(f"High-credit {name} must contain five matched seeds per family")
    data = paired.merge(hits[["family", "seed", "delta_hits_per_1000_events"]],
                        on=["family", "seed"], validate="one_to_one")
    columns = ["delta_spatial_information_bits", "delta_coverage_auc",
               "delta_hits_per_1000_events", "delta_reliable_edges"]
    if not np.isfinite(data[columns].to_numpy(dtype=float)).all():
        raise ValueError("High-credit paired differences contain nonfinite values")
    return data


HIGH_CREDIT_METRICS = (
    ("F12_high_credit_dg", "High credit: DG coding", "delta_spatial_information_bits", "bits"),
    ("F13_high_credit_coverage", "High credit: coverage", "delta_coverage_auc", "AUC"),
    ("F14_high_credit_hit_rate", "High credit: hit rate", "delta_hits_per_1000_events", "hits/1,000"),
    ("F15_high_credit_graph_edges", "High credit: graph edges", "delta_reliable_edges", "edges"),
)


def high_credit_box(slug: str, title: str, values: pd.DataFrame,
                    metric: str, unit: str) -> dict[str, str]:
    """Draw one same-size component; each dot is one paired training seed."""
    fig = plt.figure(figsize=(CARD_WIDTH, CARD_HEIGHT), facecolor="white")
    card_title(fig, title, original=False)
    ax = fig.add_axes([DATA_AXES_LEFT / CARD_WIDTH,
                       DATA_AXES_BOTTOM / CARD_HEIGHT,
                       (0.59 if metric == "delta_reliable_edges" else DATA_AXES_WIDTH) / CARD_WIDTH,
                       DATA_AXES_HEIGHT / CARD_HEIGHT])
    twin = ax.twinx() if metric == "delta_reliable_edges" else None
    for index, family in enumerate(("C05", "C15")):
        target = twin if family == "C15" and twin is not None else ax
        sample = values.loc[values.family == family, metric].to_numpy(dtype=float)
        color = shared.COLOR[family]
        target.scatter(np.full(len(sample), index) + np.linspace(-.075, .075, len(sample)),
                       sample, s=15, color=color, alpha=.7, zorder=2,
                       edgecolors="none")
        target.boxplot([sample], positions=[index], widths=.28, patch_artist=True,
                       showfliers=False, whis=(0, 100), zorder=4,
                       boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 1},
                       whiskerprops={"color": color, "linewidth": 1},
                       capprops={"color": color, "linewidth": 1},
                       medianprops={"color": color, "linewidth": 1.3})
        target.scatter([index], [sample.mean()], marker="D", s=20,
                       facecolors="white", edgecolors=color, linewidths=1,
                       zorder=6)
    ax.axhline(0, color="#7D858A", lw=.7, zorder=1)
    ax.set_xlim(-.55, 1.55)
    ax.set_xticks([0, 1], ["C05", "C15"])
    ax.tick_params(axis="x", labelsize=8, length=2, pad=1)
    ax.tick_params(axis="y", labelsize=8, length=2, pad=0)
    if twin is None:
        limit = max(abs(values[metric].min()), abs(values[metric].max())) * 1.13
        if metric == "delta_hits_per_1000_events":
            limit = 15
        ax.set_ylim(-limit, limit)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    else:
        ax.set_ylim(-15, 15)
        twin.set_ylim(-60, 60)
        ax.set_yticks([-15, 0, 15])
        twin.set_yticks([-60, 0, 60])
        ax.tick_params(axis="y", colors=shared.COLOR["C05"])
        twin.tick_params(axis="y", colors=shared.COLOR["C15"],
                         labelsize=8, length=2, pad=0)
        twin.spines[["top", "bottom", "left"]].set_visible(False)
    ax.set_ylabel(r"$\Delta$ " + unit, fontsize=10, labelpad=-1.5)
    ax.spines["top"].set_visible(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = [axis.get_tightbbox(renderer) for axis in (ax, twin) if axis is not None]
    plot_bounds = Bbox.union(tight).transformed(fig.dpi_scale_trans.inverted())
    title_bounds = fig.texts[0].get_window_extent(renderer).transformed(
        fig.dpi_scale_trans.inverted())
    if (plot_bounds.width > PLOT_WIDTH + .001 or
            plot_bounds.height > PLOT_HEIGHT + .001 or
            plot_bounds.y1 >= title_bounds.y0):
        raise ValueError(f"{slug}: plot region {plot_bounds.width:.3f} × "
                         f"{plot_bounds.height:.3f} in exceeds standard or title")
    return export(fig, slug)


def high_credit_components() -> list[dict]:
    data = high_credit_rows()
    sources = [str(path.relative_to(shared.ROOT)) for path in HIGH_CREDIT_SOURCES.values()]
    source_hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest()
                     for key, path in HIGH_CREDIT_SOURCES.items()}
    items = []
    for slug, title, metric, unit in HIGH_CREDIT_METRICS:
        items.append({
            "slug": slug, "section": "F", "kind": "difference_box",
            "original": False, "title": title, "groups": ["C05", "C15"],
            "n": len(data), "unit": unit,
            "contrast": "temporal_minus_constant_worker",
            "protocol": "high_policy_credit_online_90_100m_and_own_policy_snapshot_100m",
            "difference_column": metric,
            "mean_delta_by_group": {family: float(data.loc[data.family == family, metric].mean())
                                    for family in ("C05", "C15")},
            "sources": sources, "source_sha256": source_hashes,
            "axis_scales": ({"C05": [-15, 15], "C15": [-60, 60]}
                            if metric == "delta_reliable_edges" else
                            {"shared": [-15, 15]}
                            if metric == "delta_hits_per_1000_events" else None),
            "hashes": high_credit_box(slug, title, data, metric, unit),
        })
    return items


def canvas_group(item: dict) -> str:
    """Group by the experimental intervention rather than panel letter."""
    if item["kind"] in {"section_label", "legend"}:
        return "Loose labels and key"
    if item["kind"] in {"archived_tile", "peak_example"}:
        return "Spatial examples"
    if item["kind"] == "course":
        return "Spatial emergence"
    if item["section"] == "B":
        return "Frozen-behavior comparison"
    return {
        "temporal_minus_constant_encoder": "Push ablation · DG encoder credit",
        "temporal_minus_constant_worker": "Pull ablation · worker credit",
        "joint_longer_ppo_gae": "Longer PPO and GAE credit",
    }[item["contrast"]]


MAIN_GROUPS = (
    "Spatial emergence", "Frozen-behavior comparison",
    "Push ablation · DG encoder credit", "Pull ablation · worker credit",
    "Longer PPO and GAE credit", "Spatial examples", "Loose labels and key",
)
SCATTER_GROUPS = (
    "Frozen-behavior comparison", "Push ablation · DG encoder credit",
    "Pull ablation · worker credit", "Longer PPO and GAE credit",
    "Spatial examples",
)
MAIN_CANVAS_NOTE = (
    "Grouped by intervention. + marks extra candidates; box plots are 1.2 × 1.0 in including labels, excluding titles."
)


def combined_canvas(items: list[dict], *, stem: str, groups: tuple[str, ...],
                    heading: str, note: str) -> dict[str, str]:
    """Place independent SVG components in labeled, movable asset groups."""
    columns = 5
    slot_width = 3.30 * 72
    slot_height = 2.50 * 72
    margin = 0.28 * 72
    header = 0.78 * 72
    group_label_height = 0.34 * 72
    group_gap = 0.15 * 72
    grouped = [(name, [item for item in items if canvas_group(item) == name])
               for name in groups]
    grouped = [(name, members) for name, members in grouped if members]
    assert sum(len(members) for _, members in grouped) == len(items)

    def positions(members: list[dict]) -> tuple[list[tuple[dict, int, int]], int]:
        return [(item, index // columns, index % columns)
                for index, item in enumerate(members)], (len(members) + columns - 1) // columns

    def row_height(name: str) -> float:
        return 0.85 * 72 if name == "Loose labels and key" else slot_height

    width = 2 * margin + columns * slot_width
    height = header + margin + sum(
        group_label_height + positions(members)[1] * row_height(name)
        + group_gap for name, members in grouped)
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    root = ET.Element(f"{{{ns}}}svg", {
        "width": f"{width / 72:.3f}in", "height": f"{height / 72:.3f}in",
        "viewBox": f"0 0 {width:.3f} {height:.3f}"})
    title = ET.SubElement(root, f"{{{ns}}}text", {
        "x": str(margin), "y": "25", "font-family": "Arial",
        "font-size": "16", "font-weight": "bold", "fill": "#20272C"})
    title.text = heading
    subtitle = ET.SubElement(root, f"{{{ns}}}text", {
        "x": str(margin), "y": "44", "font-family": "Arial",
        "font-size": "10", "fill": "#5B646C"})
    subtitle.text = note
    y = header
    for group_name, members in grouped:
        label = ET.SubElement(root, f"{{{ns}}}text", {
            "x": str(margin), "y": f"{y + 14:.2f}", "font-family": "Arial",
            "font-size": "14", "font-weight": "bold", "fill": "#20272C"})
        label.text = group_name
        y += group_label_height
        placed, row_count = positions(members)
        for item, row, col in placed:
            slug = item["slug"]
            source = (COMPONENTS / f"{slug}.svg").read_text()
            ids = re.findall(r'\bid="([^"]+)"', source)
            for old in sorted(ids, key=len, reverse=True):
                new = f"{slug}_{old}"
                source = source.replace(f'id="{old}"', f'id="{new}"')
                source = source.replace(f"url(#{old})", f"url(#{new})")
                source = source.replace(f'href="#{old}"', f'href="#{new}"')
            sub = ET.fromstring(source)
            view_box = [float(number) for number in sub.get("viewBox").split()]
            source_width = float(re.search(r"([0-9.]+)", sub.get("width", ""))[1])
            source_height = float(re.search(r"([0-9.]+)", sub.get("height", ""))[1])
            x = margin + col * slot_width
            item_y = y + row * row_height(group_name)
            scale_x = source_width / view_box[2]
            scale_y = source_height / view_box[3]
            slot = ET.SubElement(root, f"{{{ns}}}g", {
                "id": f"component_{slug}",
                "transform": (f"translate({x:.2f} {item_y:.2f}) "
                              f"scale({scale_x:.8g} {scale_y:.8g}) "
                              f"translate({-view_box[0]:.8g} {-view_box[1]:.8g})")})
            # Matplotlib exports a full-page white rectangle. It makes an editor
            # select the card bounds instead of the visible graph and masks nearby
            # components. Keep only actual plot objects as direct peer groups.
            for node in sub.iter():
                if node.get("id") == f"{slug}_figure_1":
                    for child in list(node):
                        if child.get("id") == f"{slug}_patch_1":
                            node.remove(child)
            slot.extend(child for child in sub if child.tag != f"{{{ns}}}metadata")
        y += row_count * row_height(group_name) + group_gap
    path = HERE / f"{stem}.svg"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    path.write_text("\n".join(line.rstrip() for line in
                              path.read_text().splitlines()) + "\n")
    files = {"svg": path}
    for suffix in ("pdf", "png"):
        target = HERE / f"{stem}.{suffix}"
        command = ["inkscape", str(path), f"--export-type={suffix}",
                   f"--export-filename={target}"]
        if suffix == "png":
            command.append("--export-width=3000")
        subprocess.run(command, check=True, capture_output=True, text=True)
        files[suffix] = target
    return {suffix: hashlib.sha256(file.read_bytes()).hexdigest()
            for suffix, file in files.items()}


def main_canvas(items: list[dict]) -> dict[str, str]:
    return combined_canvas(
        items, stem="figure2_loose_canvas", groups=MAIN_GROUPS,
        heading="Figure 2 · box plots and other loose components",
        note=MAIN_CANVAS_NOTE)


def box_plot_manifest_metadata() -> dict:
    return {
        "box_plot_region_inches": [PLOT_WIDTH, PLOT_HEIGHT],
        "box_data_axes_inches": [DATA_AXES_WIDTH, DATA_AXES_HEIGHT],
        "box_text_points": {"ticks": 8, "axis_labels": 10, "titles": 12},
        "box_dot_alpha": 0.7,
        "box_summary_layer": "above dots",
    }


def main(*, main_only: bool = False) -> None:
    font = shared.style()
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "svg.hashsalt": "COSYNE-2027-loose-canvas"})
    tables = gallery.load_tables()
    gallery.validate_selected_evidence(tables)
    specs = rows(tables)
    if main_only:
        manifest_path = HERE / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        current_sources = {
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in gallery.SOURCE_PATHS.items()}
        assert manifest["source_sha256"] == current_sources, (
            "Source tables changed; run the full renderer to refresh both canvases")
        items = manifest["items"]
        by_slug = {item["slug"]: item for item in items}
        for spec in specs:
            item = by_slug[spec.slug]
            assert item["kind"] == "box" and item["groups"] == list(spec.groups)
            assert item["n"] == len(spec.data)
            item["hashes"] = box_plot(spec)
        items[:] = [item for item in items
                    if item["slug"] != LEGACY_WORKER_TIMING_SLUG]
        by_slug.pop(LEGACY_WORKER_TIMING_SLUG, None)
        for worker_item in high_credit_components():
            worker_item["canvas_group"] = canvas_group(worker_item)
            if worker_item["slug"] in by_slug:
                by_slug[worker_item["slug"]].update(worker_item)
            else:
                items.append(worker_item)
                by_slug[worker_item["slug"]] = worker_item
        main_items = [by_slug[slug] for slug in
                      manifest["canvases"]["figure2_loose_canvas"]["items"]
                      if slug != LEGACY_WORKER_TIMING_SLUG]
        for slug, *_ in HIGH_CREDIT_METRICS:
            if by_slug[slug] not in main_items:
                main_items.append(by_slug[slug])
        main_items.sort(key=lambda item: (
            MAIN_GROUPS.index(item["canvas_group"]),
            0 if item["kind"] == "difference_box" else 1))
        manifest["canvases"]["figure2_loose_canvas"]["items"] = [
            item["slug"] for item in main_items]
        manifest["canvases"]["figure2_loose_canvas"]["sha256"] = main_canvas(main_items)
        manifest.pop("box_axes_inches", None)
        manifest.update(box_plot_manifest_metadata())
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Updated {len(specs)} standard boxes, four high-credit boxes, and main canvas")
        return
    items = [{"slug": "A01_spatial_emergence", "section": "A", "kind": "course",
              "original": True, "sources": ["online"],
              "hashes": online_course(tables)}]
    for spec in specs:
        item = {"slug": spec.slug, "section": spec.section, "kind": "box",
                "original": spec.original, "title": spec.title,
                "groups": list(spec.groups), "n": len(spec.data),
                "unit": spec.unit, "contrast": spec.contrast,
                "protocol": spec.protocol, "sources": list(spec.source_keys),
                "mean_delta_by_group": {
                    group: float((group_rows[spec.tested] -
                                  group_rows[spec.control]).mean())
                    for group, group_rows in spec.data.groupby("group")},
                "hashes": box_plot(spec)}
        items.append(item)
        if spec.raw_pair_example:
            items.append({"slug": spec.slug + "_raw_pairs", "section": spec.section,
                          "kind": "raw_pairs", "original": spec.original,
                          "box": spec.slug, "groups": list(spec.groups),
                          "contrast": spec.contrast, "protocol": spec.protocol,
                          "sources": list(spec.source_keys),
                          "hashes": raw_pair_plot(spec)})
    items.extend(archived_tiles())
    for arm in ("temporal", "constant"):
        slug = f"C04_peaks_{arm}"
        items.append({"slug": slug, "section": "C", "kind": "peak_example",
                      "original": False, "sources": ["example", "peaks"],
                      "hashes": peak_scatter(tables, arm)})
    items.extend(high_credit_components())
    items.extend(text_elements())
    for item in items:
        item["canvas_group"] = canvas_group(item)
    main_items = [item for item in items if item["kind"] not in
                  {"raw_pairs", "peak_example"}]
    scatter_items = [item for item in items if item["kind"] in
                     {"raw_pairs", "peak_example"}]
    main_items.sort(key=lambda item: (
        MAIN_GROUPS.index(item["canvas_group"]),
        0 if item["kind"] == "difference_box" else 1))
    scatter_items.sort(key=lambda item: SCATTER_GROUPS.index(item["canvas_group"]))
    main_board = main_canvas(main_items)
    scatter_board = combined_canvas(
        scatter_items, stem="figure2_scatter_canvas", groups=SCATTER_GROUPS,
        heading="Figure 2 · scatter components",
        note="Raw control-versus-tested pairs and field-peak locations; each component remains independently movable.")
    manifest = {
        "schema": "intrmotiv/cosyne-figure2-loose-canvas/v2",
        "basis": "06_compact_vertical_boxes",
        "purpose": "Movable assets for author-led Figure 2 layout; not submitted",
        **box_plot_manifest_metadata(),
        "font_path": font,
        "software": {"matplotlib": matplotlib.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "source_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                          for key, path in gallery.SOURCE_PATHS.items()},
        "source_paths": {key: str(path.relative_to(shared.ROOT))
                         for key, path in gallery.SOURCE_PATHS.items()},
        "items": items,
        "canvases": {
            "figure2_loose_canvas": {
                "kind": "main", "items": [item["slug"] for item in main_items],
                "sha256": main_board},
            "figure2_scatter_canvas": {
                "kind": "scatter", "items": [item["slug"] for item in scatter_items],
                "sha256": scatter_board}},
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(main_items)} main and {len(scatter_items)} scatter components")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-only", action="store_true",
                        help="Update box components and main canvas without touching scatter files")
    main(main_only=parser.parse_args().main_only)
