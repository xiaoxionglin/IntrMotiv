"""Build an editable Figure 2 asset canvas from pinned seed-level evidence."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DRAFT = HERE.parents[1]
sys.path.insert(0, str(DRAFT))
import render_submission_panels as shared  # noqa: E402

OLD = HERE.parent / "07_loose_canvas" / "components"
RESULTS = ROOT / "06_experiments" / "results" / "cosyne_2027_checkpoint_fields"
HIGH = ROOT / "06_experiments" / "reward_credit" / "results" / "cosyne27_worker_regimen_boxplots_20261013"
COMP = HERE / "components"
SEEDS = [8, 17, 41, 99, 123]
COLORS = shared.COLOR
NAMES = {"C01": "NG", "C05": "UG", "C15": "FG"}
METRICS = [
    ("dg_information", "DG spatial information", "bits/step"),
    ("map_overlap", "DG map overlap", "cosine"),
    ("peak_coverage", "Field-peak coverage", "area fraction"),
    ("peak_distance", "Peak distance", "DMLab units"),
    ("hit_rate", "Commanded hit rate", "hits/1,000"),
    ("graph_edges", "Reliable graph edges", "edges"),
]


def source(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def pair_table(frame: pd.DataFrame, metric: str, *, arm: str = "push") -> pd.DataFrame:
    frame = frame.copy()
    frame["metric"] = metric
    frame["arm"] = arm
    frame = frame[["family", "seed", "metric", "arm", "delta"]]
    if frame.duplicated(["family", "seed", "metric", "arm"]).any():
        raise ValueError(f"Duplicate seed pair: {arm}/{metric}")
    if not np.isfinite(frame.delta).all():
        raise ValueError(f"Nonfinite seed difference: {arm}/{metric}")
    return frame


def load_values() -> tuple[pd.DataFrame, dict[str, str]]:
    paths = {
        "push_dg": shared.SOURCES["encoder"],
        "push_peaks": shared.SOURCES["peaks"],
        "push_hits": RESULTS / "conditional_hits" / "conditional_hit_seed_metrics.csv",
        "push_graph": RESULTS / "graph_trajectory_summary" / "paired_directed_graph.csv",
        "pull_dg": HIGH / "high_credit_representation_pairs.csv",
        "pull_hits": HIGH / "high_credit_hit_rate_pairs.csv",
        "pull_graph": HIGH / "paired_seed_differences.csv",
        "emergence": shared.SOURCES["online"],
        "frozen_behavior": shared.SOURCES["behavior_absolute"],
        "additional_fg_hits": HERE / "additional_fg_hit_inputs.json",
    }
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in paths.values()}
    encoder = source(paths["push_dg"])
    encoder = encoder[(encoder.contrast == "C01") |
                      ((encoder.contrast == "C05 · temporal hit") |
                       (encoder.contrast == "C15"))]
    encoder = encoder.assign(family=encoder.family)
    peaks = source(paths["push_peaks"])
    peaks = peaks[(peaks.cutoff == .0025) &
                  peaks.contrast.isin(["C01", "C05 · temporal hit", "C15"])]
    values = []
    for metric, key in [("dg_information", "mean_spatial_information_bits"),
                        ("map_overlap", "active_map_cosine_mean")]:
        data = encoder[encoder.metric == key].rename(columns={"difference": "delta"})
        values.append(pair_table(data, metric))
    for metric, key in [("peak_coverage", "effective_area_fraction"),
                        ("peak_distance", "median_nearest_peak_distance")]:
        data = peaks[peaks.metric == key].rename(columns={"difference": "delta"})
        values.append(pair_table(data, metric))
    hits = source(paths["push_hits"])
    extra = json.loads(paths["additional_fg_hits"].read_text())
    assert extra["schema"] == "intrmotiv/focused-canvas-additional-fg-hits/v1"
    assert (extra["window_low"], extra["window_high"]) == (90_000_000, 100_000_000)
    assert {(row["condition"], row["seed"]) for row in extra["rows"]} == {
        (f"COS27_C15_{credit.upper()}", seed)
        for credit in ("temporal", "constant") for seed in (8, 99, 123)}
    extra_hits = pd.DataFrame(extra["rows"])
    extra_hits["family"] = "C15"
    extra_hits["encoder_credit"] = extra_hits.condition.str.removeprefix("COS27_C15_").str.lower()
    extra_hits["worker_bonus"] = "temporal"
    hits = pd.concat([hits, extra_hits[hits.columns.intersection(extra_hits.columns)]],
                     ignore_index=True)
    hits = hits[((hits.family == "C05") & (hits.worker_bonus == "temporal")) |
                ((hits.family == "C15") & (hits.worker_bonus == "temporal"))]
    hits = hits.assign(rate=1000 * hits.correct_hit_observation_count /
                      hits.target_hit_event_observation_count)
    hits = hits.pivot(index=["family", "seed"], columns="encoder_credit", values="rate")
    hits = (hits.temporal - hits.constant).rename("delta").reset_index()
    values.append(pair_table(hits, "hit_rate"))
    graph = source(paths["push_graph"])
    graph = graph[(graph.target_env_steps == 100_000_000) &
                  (((graph.family == "C05") & (graph.worker_bonus == "TEMPORAL")) |
                   ((graph.family == "C15") & (graph.worker_bonus == "temporal")))]
    graph = graph.rename(columns={"delta_graph_reliable_edge_count": "delta"})
    values.append(pair_table(graph, "graph_edges"))

    representation = source(paths["pull_dg"])
    for metric, key in [
        ("dg_information", "delta_active_unit_mean_spatial_information"),
        ("map_overlap", "delta_active_only_map_cosine"),
        ("peak_coverage", "delta_effective_peak_area_fraction"),
        ("peak_distance", "delta_cutoff_median_nearest_peak_distance"),
    ]:
        values.append(pair_table(representation.rename(columns={key: "delta"}),
                                 metric, arm="pull"))
    values.append(pair_table(source(paths["pull_hits"]).rename(
        columns={"delta_hits_per_1000_events": "delta"}), "hit_rate", arm="pull"))
    pull_graph = source(paths["pull_graph"])
    pull_graph = pull_graph[pull_graph.policy_credit == "high"]
    values.append(pair_table(pull_graph.rename(columns={"delta_reliable_edges": "delta"}),
                             "graph_edges", arm="pull"))
    result = pd.concat(values, ignore_index=True)
    expected = {("push", key): ({"C01": 5, "C05": 5, "C15": 5} if key not in
                {"hit_rate", "graph_edges"} else
                {"C05": 5, "C15": 5})
                for key, _, _ in METRICS}
    expected.update({("pull", key): {"C05": 5, "C15": 5} for key, _, _ in METRICS})
    for (arm, metric), sample in result.groupby(["arm", "metric"]):
        assert sample.groupby("family").size().to_dict() == expected[arm, metric], (arm, metric)
        assert set(sample.seed).issubset(SEEDS)
    return result, source_hashes


def card(arm: str, metric: str, title: str, unit: str, data: pd.DataFrame) -> str:
    slug = f"{arm}_{metric}"
    fig = plt.figure(figsize=(2.25, 1.95), facecolor="white")
    fig.text(.045, .965, title, ha="left", va="top", fontsize=12,
             fontweight="bold", color="#20272C")
    ax = fig.add_axes([.93 / 2.25, .47 / 1.95, .69 / 2.25, .66 / 1.95])
    groups = ["C01", "C05", "C15"] if metric not in {"hit_rate", "graph_edges"} and arm == "push" else ["C05", "C15"]
    samples = [data[data.family == family].sort_values("seed").delta.to_numpy()
               for family in groups]
    for index, (family, sample) in enumerate(zip(groups, samples)):
        color = COLORS[family]
        ax.scatter(index + np.linspace(-.07, .07, len(sample)), sample,
                   s=15, color=color, alpha=.7, edgecolors="none", zorder=2)
        ax.boxplot([sample], positions=[index], widths=.28, patch_artist=True,
                   showfliers=False, whis=(0, 100), zorder=4,
                   boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 1},
                   whiskerprops={"color": color, "linewidth": 1},
                   capprops={"color": color, "linewidth": 1},
                   medianprops={"color": color, "linewidth": 1.2})
    ax.axhline(0, color="#7D858A", lw=.7, zorder=1)
    ax.set_xlim(-.45, len(groups)-.55)
    ax.set_xticks(range(len(groups)), [NAMES[group] for group in groups])
    ax.tick_params(axis="both", labelsize=8, length=2, pad=1)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.set_ylabel(r"$\Delta$ " + unit, fontsize=10, labelpad=-1.5)
    ax.spines[["top", "right"]].set_visible(False)
    extent = max(abs(data.delta.min()), abs(data.delta.max()))
    ax.set_ylim(-max(extent * 1.13, 1e-9), max(extent * 1.13, 1e-9))
    fig.canvas.draw()
    bounds = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
        fig.dpi_scale_trans.inverted())
    if bounds.width > 1.201 or bounds.height > 1.001:
        raise ValueError(f"{slug}: plot including labels is "
                         f"{bounds.width:.3f} x {bounds.height:.3f} inches")
    COMP.mkdir(parents=True, exist_ok=True)
    path = COMP / f"{slug}.svg"
    fig.savefig(path, metadata={"Date": None})
    fig.savefig(COMP / f"{slug}.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return slug


def emergence() -> str:
    online = source(shared.SOURCES["online"])
    fig = plt.figure(figsize=(1.74, 2.15), facecolor="white")
    ax = fig.add_axes([.47 / 1.74, .44 / 2.15, 1.056 / 1.74, 1.12 / 2.15])
    for family, worker in [("C01", "inapplicable"), ("C05", "temporal"),
                           ("C15", "temporal")]:
        rows = online[(online.family == family) & (online.encoder_credit == "temporal") &
                      (online.worker_bonus == worker) &
                      (online.metric == "spatial_information")].sort_values("env_steps")
        assert len(rows) > 50 and rows.n.eq(5).all()
        x = rows.env_steps / 1e6
        ax.plot(x, rows["mean"], color=COLORS[family], lw=1.2, label=NAMES[family])
        ax.fill_between(x, rows["mean"]-rows["sem"], rows["mean"]+rows["sem"],
                        color=COLORS[family], alpha=.15)
    ax.set(xlim=(0, 100), xlabel="Frames (millions)", ylabel="DG bits/step")
    ax.tick_params(labelsize=8, length=2, pad=1)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E6E8E9", lw=.5)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left",
               bbox_to_anchor=(.25, .85), frameon=False, ncol=3,
               fontsize=8, columnspacing=.2, handlelength=.5, handletextpad=.2)
    path = COMP / "spatial_emergence.svg"
    fig.savefig(path, metadata={"Date": None})
    fig.savefig(COMP / "spatial_emergence.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return "spatial_emergence"


def compose(slugs: list[tuple[str, float, float]]) -> None:
    ns = "http://www.w3.org/2000/svg"
    ink = "http://www.inkscape.org/namespaces/inkscape"
    ET.register_namespace("", ns)
    ET.register_namespace("inkscape", ink)
    width, height = 6.15*72, 19.25*72
    root = ET.Element(f"{{{ns}}}svg", {"width": "6.15in", "height": "19.25in",
                                       "viewBox": f"0 0 {width} {height}"})
    layer = ET.SubElement(root, f"{{{ns}}}g", {"id": "layer_plots",
        f"{{{ink}}}groupmode": "layer", f"{{{ink}}}label": "Movable plots"})
    for label, y in [("A  Spatial emergence", 20), ("B  Frozen behavior", 205),
                     ("C  Push · original policy credit", 396),
                     ("D  Pull · high policy credit, temporal shaping", 887)]:
        node = ET.SubElement(layer, f"{{{ns}}}text", {"x": "20", "y": str(y),
            "font-family": "Arial", "font-size": "13", "font-weight": "bold"})
        node.text = label
    for slug, x, y in slugs:
        content = (COMP / f"{slug}.svg").read_text()
        for old in sorted(set(re.findall(r'\bid="([^"]+)"', content)), key=len, reverse=True):
            new = f"{slug}_{old}"
            content = content.replace(f'id="{old}"', f'id="{new}"')
            content = content.replace(f"url(#{old})", f"url(#{new})")
            content = content.replace(f'href="#{old}"', f'href="#{new}"')
        sub = ET.fromstring(content)
        box = [float(v) for v in sub.get("viewBox").split()]
        physical_width = float(re.match(r"[\d.]+", sub.get("width")).group())
        physical_height = float(re.match(r"[\d.]+", sub.get("height")).group())
        group = ET.SubElement(layer, f"{{{ns}}}g", {"id": f"component_{slug}",
            f"{{{ink}}}label": slug,
            "transform": f"translate({x} {y}) scale({physical_width/box[2]} {physical_height/box[3]})"})
        group.extend(node for node in sub if node.tag == f"{{{ns}}}defs")
        figure = next(node for node in sub if node.get("id") == f"{slug}_figure_1")
        group.extend(node for node in figure if node.get("id") != f"{slug}_patch_1")
    assert len([node for node in layer if node.tag == f"{{{ns}}}g"]) == len(slugs)
    assert not any(node.tag == f"{{{ns}}}svg" for node in root.iter() if node is not root)
    path = HERE / "focused_canvas_editable.svg"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    for suffix in ("pdf", "png"):
        target = HERE / f"focused_canvas.{suffix}"
        args = ["inkscape", str(path), f"--export-filename={target}"]
        if suffix == "png":
            args.append("--export-width=2200")
        subprocess.run(args, check=True, capture_output=True, text=True)


def main() -> None:
    font = shared.style()
    plt.rcParams["svg.hashsalt"] = "COSYNE-2027-focused-canvas"
    COMP.mkdir(parents=True, exist_ok=True)
    values, hashes = load_values()
    slugs = [(emergence(), 24, 33)]
    for column, name in enumerate(["B01_readout_policy_information", "B02_policy_map_reliability"]):
        slug = name
        (COMP / f"{slug}.svg").write_bytes((OLD / f"{slug}.svg").read_bytes())
        (COMP / f"{slug}.pdf").write_bytes((OLD / f"{slug}.pdf").read_bytes())
        slugs.append((slug, 24 + column * 210, 219))
    for arm, start_y in [("push", 411), ("pull", 902)]:
        for index, (key, title, unit) in enumerate(METRICS):
            sample = values[(values.arm == arm) & (values.metric == key)]
            slug = card(arm, key, title, unit, sample)
            slugs.append((slug, 24 + index % 2 * 210, start_y + index // 2 * 155))
    compose(slugs)
    values.to_csv(HERE / "plotted_seed_differences.csv", index=False)
    (HERE / "manifest.json").write_text(json.dumps({
        "schema": "intrmotiv/figure2-focused-canvas/v1",
        "source_sha256": hashes, "font": font,
        "components": [item[0] for item in slugs],
        "sample_counts": {f"{arm}/{metric}/{family}": len(group)
                          for (arm, metric, family), group in
                          values.groupby(["arm", "metric", "family"])},
    }, indent=2) + "\n")
    print(f"Rendered {len(slugs)} individually movable components")


if __name__ == "__main__":
    main()
