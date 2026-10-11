"""Render letter-page COSYNE panels from cached, provenance-pinned CSVs.

All contrasts keep the training seed as the independent observation. This
script changes presentation only; the upstream analysis tables stay intact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "06_experiments/results"
OUT = RESULTS / "cosyne_2027_compact_figures"
SOURCES = {
    "online": RESULTS / "cosyne_2027_online_wandb/arm_summaries.csv",
    "online_pairs": RESULTS / "cosyne_2027_online_wandb/paired_contrasts_by_seed.csv",
    "behavior": RESULTS / "cosyne_2027_behavior_secondary/paired_training_seeds.csv",
    "behavior_shared_raw": RESULTS / "behavior_field_expression_20261007/paired_prefixes.csv",
    "behavior_native_raw": RESULTS / "behavior_field_expression_20261007/per_run.csv",
    "encoder": RESULTS / "cosyne_2027_checkpoint_fields/common_panel_analysis/paired_encoder_fields.csv",
    "encoder_raw": RESULTS / "cosyne_2027_checkpoint_fields/common_panel_analysis/shared_observation_fields.csv",
    "worker_fields": RESULTS / "cosyne_2027_checkpoint_fields/common_panel_analysis/paired_worker_fields.csv",
    "peaks": RESULTS / "cosyne_2027_checkpoint_fields/common_panel_analysis/paired_peak_spread_by_seed.csv",
    "credit_presence": ROOT / "06_experiments/reward_credit/results/corrected_core_encoder_worker_20261010/common_panel_encoder_pairs.csv",
    "credit_presence_raw": ROOT / "06_experiments/reward_credit/results/corrected_core_encoder_worker_20261010/common_panel_place_fields.csv",
}
COLORS = {"C01": "#0072B2", "C05": "#D55E00", "C15": "#009E73"}
FACTOR_ORDER = ("C01", "C05 · temporal hit", "C05 · constant hit", "C15")
FACTOR_LABELS = ("C01", "C05 T", "C05 C", "C15")
BEHAVIOR_PANEL_SIZE_IN = (1.5, 1.8)


def style() -> str:
    path = findfont(FontProperties(family="Arial"), fallback_to_default=False)
    if Path(path).suffix.lower() not in {".ttf", ".otf"}:
        raise RuntimeError(f"Arial did not resolve to a scalable font: {path}")
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 10, "axes.labelsize": 10,
        "axes.titlesize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.7,
        "svg.fonttype": "none", "svg.hashsalt": "IntrMotiv-COSYNE-2027-compact",
        "pdf.fonttype": 42,
    })
    return path


def export(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / f"{name}.svg"
    fig.savefig(svg_path, dpi=300, metadata={"Date": None})
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def seed_differences(ax: plt.Axes, frame: pd.DataFrame, groups: tuple[str, ...],
                     *, group_column: str, labels: tuple[str, ...],
                     colors: tuple[str, ...],
                     markers: tuple[str, ...] | None = None) -> None:
    """Show paired training seeds, with optional open squares for a second factor."""
    markers = markers or ("o",) * len(groups)
    if not (len(groups) == len(labels) == len(colors) == len(markers)):
        raise ValueError("one label, color, and marker are required per group")
    ax.axhline(0, color="#888888", lw=0.8, zorder=0)
    for x, (group, color, marker) in enumerate(zip(groups, colors, markers)):
        values = frame[frame[group_column] == group].sort_values("seed")
        if len(values) < 2 or values.seed.nunique() != len(values):
            raise ValueError(f"unexpected paired-seed count for {group}: {len(values)}")
        offsets = np.linspace(-0.13, 0.13, len(values))
        y = values.difference.to_numpy(float)
        if marker == "s":
            ax.scatter(x + offsets, y, s=17, marker="s", facecolor="white",
                       edgecolor=color, linewidth=1.1, zorder=3)
        else:
            ax.scatter(x + offsets, y, s=17, marker=marker, color=color,
                       linewidth=0, zorder=3)
        ax.plot([x - 0.16, x + 0.16], [np.mean(y)] * 2,
                color="black", lw=1.2, zorder=4)
    ax.set_xticks(range(len(groups)), labels)
    ax.tick_params(axis="both", length=2.5, width=0.6, pad=2)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)


def paired_values_and_difference(
    data: pd.DataFrame, *, name: str, groups: tuple[str, ...],
    group_column: str, labels: tuple[str, ...], colors: tuple[str, ...],
    first_label: str, second_label: str, ylabel: str, delta_label: str,
    minimum_counts: tuple[int, ...],
) -> None:
    """Show both observed arms and their paired contrast at final-page size.

    Input rows must already be one observation per training seed. A line connects
    the two arms of the same seed; the difference panel uses those exact rows.
    """
    if not (len(groups) == len(labels) == len(colors) == len(minimum_counts)):
        raise ValueError("factor labels, colors, and minimum seed counts differ")
    if set(data[group_column]) != set(groups):
        raise ValueError(f"{name}: plotted factor cells differ from source rows")
    if data.duplicated([group_column, "seed"]).any():
        raise ValueError(f"duplicate seed in {name}")
    if data[["first", "second", "difference"]].isna().any().any():
        raise ValueError(f"missing paired value in {name}")
    if not np.allclose(data["first"] - data["second"], data["difference"],
                       rtol=0, atol=1e-8):
        raise ValueError(f"paired contrast does not match source values in {name}")

    fig, axes = plt.subplots(1, 2, figsize=(3.8, 1.9))
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.26, top=0.76,
                        wspace=0.45)
    absolute_ax, difference_ax = axes
    for x, (group, color, minimum) in enumerate(zip(groups, colors, minimum_counts)):
        rows = data[data[group_column] == group].sort_values("seed")
        if len(rows) < minimum or rows.seed.nunique() != len(rows):
            raise ValueError(f"{name}: expected at least {minimum} paired seeds for {group}")
        for offset, row in zip(np.linspace(-0.035, 0.035, len(rows)),
                               rows.itertuples(index=False)):
            absolute_ax.plot([x - 0.12 + offset, x + 0.12 + offset],
                             [row.second, row.first], color="#BBC4CA",
                             lw=0.7, zorder=1)
            absolute_ax.scatter(x - 0.12 + offset, row.second, s=17,
                                marker="s", facecolor="white", edgecolor=color,
                                linewidth=1.1, zorder=3)
            absolute_ax.scatter(x + 0.12 + offset, row.first, s=17,
                                marker="o", color=color, linewidth=0, zorder=3)
    absolute_ax.set_ylim(bottom=0)
    absolute_ax.set_xticks(range(len(groups)), labels)
    absolute_ax.set_ylabel(ylabel, labelpad=2)
    absolute_ax.set_title("a", loc="left", pad=6, fontweight="bold")
    absolute_ax.tick_params(axis="both", length=2.5, width=0.6, pad=2)
    absolute_ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)

    seed_differences(difference_ax, data, groups,
                     group_column=group_column, labels=labels, colors=colors)
    difference_ax.set_ylabel(delta_label, labelpad=2)
    difference_ax.set_title("b", loc="left", pad=6, fontweight="bold")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="none", color="#30363B",
                          markersize=4, label=first_label),
               plt.Line2D([0], [0], marker="s", linestyle="none", color="#75838C",
                          markerfacecolor="white", markeredgewidth=1.1,
                          markersize=4, label=second_label)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 1.02),
               ncol=2, frameon=False, handletextpad=0.2, columnspacing=0.6)
    export(fig, name)


def online_course(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(1.9, 1.9))
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.27, top=0.95)
    for family in ("C01", "C05", "C15"):
        rows = data[(data.family == family) & (data.encoder_credit == "temporal")
                    & (data.metric == "spatial_information")
                    & (data.worker_bonus == ("inapplicable" if family == "C01" else "temporal"))]
        if len(rows) != 99 or not rows.n.eq(5).all():
            raise ValueError(f"incomplete five-seed online course for {family}")
        rows = rows.sort_values("env_steps")
        x = rows.env_steps.to_numpy(float) / 1e6
        mean = rows["mean"].to_numpy(float)
        sem = rows["sem"].to_numpy(float)
        ax.plot(x, mean, lw=1.25, color=COLORS[family], label=family)
        ax.fill_between(x, mean - sem, mean + sem, color=COLORS[family], alpha=0.16,
                        linewidth=0)
        ax.text(102, mean[-1], family, color=COLORS[family], va="center", fontsize=10)
    ax.set(xlim=(0, 135), ylim=(0, 0.23), xlabel="Steps (M)", ylabel="DG score")
    ax.set_xticks((0, 50, 100))
    ax.tick_params(axis="both", length=2.5, width=0.6, pad=2)
    export(fig, "01_online_spatial_course")


def behavior_expression(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(2 * BEHAVIOR_PANEL_SIZE_IN[0],
                                            BEHAVIOR_PANEL_SIZE_IN[1]))
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.24, top=0.81, wspace=0.48)
    specs = (
        ("bits_per_activation", "a", "Δ info (bits/act.)"),
        ("split_half_map_correlation", "b", "Δ reliability"),
    )
    for ax, (metric, panel, ylabel) in zip(axes, specs):
        rows = data[(data.layer == "decoder_1") & (data.metric == metric)]
        seed_differences(ax, rows, ("C01", "C05", "C15"), group_column="condition",
                         labels=("C01", "C05", "C15"),
                         colors=tuple(COLORS.values()))
        ax.set_title(panel, loc="left", pad=8, fontweight="bold")
        ax.set_ylabel(ylabel, labelpad=2)
    export(fig, "02_behavior_expression")


def behavior_absolute_data(differences: pd.DataFrame) -> pd.DataFrame:
    """Average two evaluation repeats within each parent training seed."""
    shared = pd.read_csv(SOURCES["behavior_shared_raw"])
    shared = shared[(shared.random_policy == "uniform")
                    & (shared.decisions == 50_000)
                    & (shared.layer == "decoder_1")]
    if len(shared) != 18 or not shared.supported.all():
        raise ValueError("expected 18 supported decoder-1 common-support probes")
    info = (shared.groupby(["condition", "seed"], as_index=False)
            .agg(repeats=("eval_seed", "nunique"),
                 control=("random_bits_per_activation", "mean"),
                 experimental=("own_bits_per_activation", "mean")))
    info["metric"] = "bits_per_activation"
    info["support"] = "same units and equally weighted shared bins"

    native = pd.read_csv(SOURCES["behavior_native_raw"])
    native = native[(native.layer == "decoder_1")
                    & native.policy.isin(("own", "uniform"))]
    if len(native) != 36:
        raise ValueError("expected 36 decoder-1 native reliability rows")
    repeat_counts = native.groupby(["condition", "seed", "policy"]).eval_seed.nunique()
    if not repeat_counts.eq(2).all():
        raise ValueError("each native policy needs two evaluation repeats")
    reliability = (native.groupby(["condition", "seed", "policy"], as_index=False)
                   .split_half_map_correlation.mean()
                   .pivot(index=["condition", "seed"], columns="policy",
                          values="split_half_map_correlation")
                   .reset_index()
                   .rename(columns={"uniform": "control", "own": "experimental"}))
    reliability["repeats"] = 2
    reliability["metric"] = "split_half_map_correlation"
    reliability["support"] = "policy-native occupancy"
    result = pd.concat([info, reliability], ignore_index=True)
    if len(result) != 18 or not result.repeats.eq(2).all():
        raise ValueError("incomplete absolute-value seed table")
    result["difference"] = result.experimental - result.control
    check = differences[(differences.layer == "decoder_1")
                        & differences.metric.isin(result.metric.unique())]
    joined = result.merge(check, on=["condition", "seed", "metric", "support"],
                          suffixes=("", "_archived"), validate="one_to_one")
    if len(joined) != 18 or not np.allclose(joined.difference, joined.difference_archived,
                                             rtol=0, atol=1e-10):
        raise ValueError("absolute values disagree with archived paired differences")
    return result.sort_values(["metric", "condition", "seed"]).reset_index(drop=True)


def behavior_absolute(data: pd.DataFrame) -> None:
    """Candidate view: show control and experimental values for every seed."""
    fig, axes = plt.subplots(1, 2, figsize=(2 * BEHAVIOR_PANEL_SIZE_IN[0],
                                            BEHAVIOR_PANEL_SIZE_IN[1]))
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.24, top=0.76, wspace=0.48)
    control_color, experimental_color = "#75838C", "#30363B"
    specs = (("bits_per_activation", "a", "Info (bits/act.)"),
             ("split_half_map_correlation", "b", "Map reliability"))
    for ax, (metric, panel, ylabel) in zip(axes, specs):
        frame = data[data.metric == metric]
        for x, condition in enumerate(("C01", "C05", "C15")):
            seeds = frame[frame.condition == condition].sort_values("seed")
            if len(seeds) != 3:
                raise ValueError(f"expected three independent training seeds for {condition}")
            for offset, row in zip(np.linspace(-0.025, 0.025, len(seeds)),
                                   seeds.itertuples(index=False)):
                ax.plot([x - 0.11 + offset, x + 0.11 + offset],
                        [row.control, row.experimental], color="#BBC4CA",
                        lw=0.75, zorder=1)
                ax.scatter(x - 0.11 + offset, row.control, s=17, marker="s",
                           facecolor="white", edgecolor=control_color,
                           linewidth=1.1, zorder=3)
                ax.scatter(x + 0.11 + offset, row.experimental, s=17,
                           color=experimental_color, linewidth=0, zorder=3)
        ax.set_xticks(range(3), ("C01", "C05", "C15"))
        ax.set_ylabel(ylabel, labelpad=2)
        ax.set_title(panel, loc="left", pad=6, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="both", length=2.5, width=0.6, pad=2)
        ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
    handles = [plt.Line2D([0], [0], marker="s", linestyle="none", color=control_color,
                          markerfacecolor="white", markeredgewidth=1.1,
                          markersize=4, label="Random control"),
               plt.Line2D([0], [0], marker="o", linestyle="none",
                          color=experimental_color, markersize=4, label="Own policy")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, frameon=False, handletextpad=0.2, columnspacing=0.6)
    export(fig, "02_behavior_expression_absolute")


def credit_presence(data: pd.DataFrame) -> None:
    rows = data[data.contrast == "none_stop_minus_constant"].copy()
    rows["group"] = rows.family + rows.worker_bonus.map({
        "flat": "", "temporal": "T", "zero": "Z"})
    rows["difference"] = -rows.active_unit_mean_si_bits_delta
    groups = ("C01", "C05T", "C05Z", "C15T", "C15Z")
    fig, ax = plt.subplots(figsize=(1.9, 1.9))
    fig.subplots_adjust(left=0.26, right=0.98, bottom=0.25, top=0.79)
    seed_differences(ax, rows, groups, group_column="group",
                     labels=("01", "05T", "05Z", "15T", "15Z"),
                     colors=(COLORS["C01"], COLORS["C05"], COLORS["C05"],
                             COLORS["C15"], COLORS["C15"]),
                     markers=("o", "o", "s", "o", "s"))
    ax.set_title("Constant − no credit", loc="left", pad=8)
    ax.set_ylabel("Δ DG info (bits)")
    export(fig, "02_credit_presence")


def encoder_credit(fields: pd.DataFrame, peaks: pd.DataFrame) -> None:
    """Keep paired, same-source temporal-minus-constant effects separate."""
    fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.1))
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.25, top=0.87, wspace=0.48)
    specs = (
        (fields, "mean_spatial_information_bits", "a", "Δ DG spatial score"),
        (fields, "active_map_cosine_mean", "b", "Δ map cosine"),
        (peaks[peaks.cutoff == 0.0025], "effective_area_fraction",
         "c", "Δ peak area fraction"),
    )
    colors = (COLORS["C01"], COLORS["C05"], COLORS["C05"], COLORS["C15"])
    for ax, (table, metric, panel, ylabel) in zip(axes, specs):
        rows = table[table.metric == metric]
        seed_differences(ax, rows, FACTOR_ORDER, group_column="contrast",
                         labels=("C01", "C05\nT", "C05\nC", "C15"),
                         colors=colors, markers=("o", "o", "s", "o"))
        ax.set_title(panel, loc="left", pad=6, fontweight="bold")
        ax.set_ylabel(ylabel, labelpad=2)
    export(fig, "03_encoder_credit_deltas")


def encoder_original_values(fresh: pd.DataFrame, historical: pd.DataFrame,
                            pairs: pd.DataFrame) -> pd.DataFrame:
    """Assemble a descriptive three-level view with explicit cohort identity."""
    if len(fresh) < 38 or fresh.duplicated(["family", "worker_bonus",
                                             "encoder_credit", "seed"]).any():
        raise ValueError("fresh common-panel run table is incomplete or duplicated")
    cells = {("C01", "inapplicable"): "C01",
             ("C05", "temporal"): "C05 · temporal hit",
             ("C05", "constant"): "C05 · constant hit",
             ("C15", "temporal"): "C15"}
    metrics = ("mean_spatial_information_bits", "active_map_cosine_mean")
    records = []
    for row in fresh.itertuples(index=False):
        cell = cells.get((row.family, row.worker_bonus))
        if cell is None:
            raise ValueError("unexpected fresh encoder factor cell")
        for metric in metrics:
            records.append({"factor_cell": cell, "family": row.family,
                            "worker_bonus": row.worker_bonus,
                            "encoder_credit": row.encoder_credit,
                            "seed": row.seed, "metric": metric,
                            "value": getattr(row, metric), "cohort": "fresh"})
    older = historical[(historical.credit == "NONE_STOP") &
                       (((historical.family == "C01") &
                         (historical.worker_bonus == "flat")) |
                        ((historical.family.isin(("C05", "C15"))) &
                         (historical.worker_bonus == "temporal")))]
    if len(older) != 9 or older.groupby("family").seed.nunique().to_dict() != {
        "C01": 3, "C05": 3, "C15": 3}:
        raise ValueError("historical no-credit reference is incomplete")
    old_metric = {"mean_spatial_information_bits": "active_unit_mean_si_bits",
                  "active_map_cosine_mean": "active_map_cosine_mean"}
    for row in older.itertuples(index=False):
        cell = "C01" if row.family == "C01" else (
            "C05 · temporal hit" if row.family == "C05" else "C15")
        for metric, column in old_metric.items():
            records.append({"factor_cell": cell, "family": row.family,
                            "worker_bonus": row.worker_bonus,
                            "encoder_credit": "none", "seed": row.seed,
                            "metric": metric, "value": getattr(row, column),
                            "cohort": "historical"})
    result = pd.DataFrame.from_records(records)
    if len(result) != 2 * (len(fresh) + len(older)) or result.duplicated(
        ["factor_cell", "encoder_credit", "seed", "metric"]).any():
        raise ValueError("three-level values lost or duplicated runs")
    if not np.isfinite(result.value).all():
        raise ValueError("nonfinite three-level encoder values")
    paired_values = pairs[pairs.metric.isin(metrics)].melt(
        id_vars=["contrast", "seed", "metric"],
        value_vars=["constant", "temporal"],
        var_name="encoder_credit", value_name="paired_value")
    check = paired_values.merge(
        result[result.cohort == "fresh"],
        left_on=["contrast", "seed", "metric", "encoder_credit"],
        right_on=["factor_cell", "seed", "metric", "encoder_credit"],
        validate="one_to_one")
    if len(check) != len(paired_values) or not np.allclose(
        check.paired_value, check.value, rtol=0, atol=1e-8):
        raise ValueError("fresh original values disagree with paired delta inputs")
    return result.sort_values(["metric", "factor_cell", "encoder_credit", "seed"])


def plot_encoder_original_values(data: pd.DataFrame, *, metric: str,
                                 name: str, ylabel: str, upper: float) -> None:
    """Show three encoder settings while visibly separating source cohorts."""
    fig, axes = plt.subplots(2, 2, figsize=(4.2, 3.8), sharey=True)
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.93,
                        wspace=0.27, hspace=0.55)
    cells = ("C01", "C05 · temporal hit", "C05 · constant hit", "C15")
    labels = ("C01", "C05 · T hit", "C05 · C hit", "C15")
    minimums = {"C01": (3, 5, 5), "C05 · temporal hit": (3, 5, 5),
                "C05 · constant hit": (0, 5, 5), "C15": (3, 3, 5)}
    for ax, cell, label, panel in zip(axes.flat, cells, labels, "abcd"):
        rows = data[(data.metric == metric) & (data.factor_cell == cell)]
        color = COLORS[rows.family.iloc[0]]
        ax.axvspan(-0.35, 0.45, color="#F3F5F6", zorder=0)
        ax.axvline(0.5, color="#AAB3BA", ls=(0, (2, 2)), lw=0.75, zorder=1)
        for x, credit, marker, expected in zip(
            (0, 1, 2), ("none", "constant", "temporal"),
            ("D", "s", "o"), minimums[cell]):
            selected = rows[rows.encoder_credit == credit].sort_values("seed")
            if len(selected) < expected or selected.seed.nunique() != len(selected):
                raise ValueError(f"{name}: missing {cell} {credit} seed values")
            if selected.empty:
                ax.text(0, upper * 0.08, "n/a", ha="center", va="bottom",
                        color="#75838C", fontsize=10)
                continue
            offsets = np.linspace(-0.085, 0.085, len(selected))
            if credit == "temporal":
                ax.scatter(x + offsets, selected.value, s=19, color=color,
                           marker=marker, linewidth=0, zorder=3)
            else:
                ax.scatter(x + offsets, selected.value, s=19, marker=marker,
                           facecolor="white", edgecolor=color,
                           linewidth=1.1, zorder=3)
            ax.plot([x - 0.13, x + 0.13], [selected.value.mean()] * 2,
                    color="black", lw=1.05, zorder=4)
        ax.set(xlim=(-0.35, 2.35), ylim=(0, upper))
        ax.set_xticks((0, 1, 2), ("None\nold", "Const.\nfresh", "Temp.\nfresh"))
        ax.set_title(f"{panel}  {label}", loc="left", pad=5, fontweight="bold")
        ax.tick_params(axis="both", length=2.5, width=0.6, pad=2)
        ax.grid(axis="y", color="#E2E8EB", lw=0.5, zorder=0)
    axes[0, 0].set_ylabel(ylabel, labelpad=2)
    axes[1, 0].set_ylabel(ylabel, labelpad=2)
    export(fig, name)


def worker_bonus(data: pd.DataFrame) -> None:
    rows = data[data.contrast == "temporal_minus_constant_hit_bonus"]
    fig, axes = plt.subplots(1, 2, figsize=(3.8, 2.05))
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.25, top=0.81, wspace=0.35)
    specs = (("visited_cell_fraction", "a  Δ visited fraction"),
             ("spatial_information", "b  Δ online DG score"))
    for ax, (metric, label) in zip(axes, specs):
        selected = rows[rows.metric == metric]
        seed_differences(ax, selected, ("temporal", "constant"),
                         group_column="fixed_factor", labels=("T", "C"),
                         colors=("#30363B", "#75838C"),
                         markers=("o", "s"))
        ax.set_title(label, loc="left", pad=8)
        ax.set_xlabel("Encoder credit")
    export(fig, "04_worker_timing_factorized")


def main() -> None:
    font_path = style()
    tables = {name: pd.read_csv(path) for name, path in SOURCES.items()}
    online_course(tables["online"])
    behavior_expression(tables["behavior"])
    absolute = behavior_absolute_data(tables["behavior"])
    OUT.mkdir(parents=True, exist_ok=True)
    absolute.to_csv(OUT / "02_behavior_expression_absolute_by_seed.csv",
                    index=False, lineterminator="\n")
    behavior_absolute(absolute)
    for metric, name, ylabel, delta_label in (
        ("bits_per_activation", "02_behavior_info_values_and_delta",
         "Info (bits/act.)", "Δ info (bits/act.)"),
        ("split_half_map_correlation", "02_behavior_reliability_values_and_delta",
         "Map reliability", "Δ reliability"),
    ):
        paired_values_and_difference(
            absolute[absolute.metric == metric].rename(
                columns={"experimental": "first", "control": "second"}),
            name=name, groups=("C01", "C05", "C15"), group_column="condition",
            labels=("C01", "C05", "C15"), colors=tuple(COLORS.values()),
            first_label="Own policy", second_label="Random actions",
            ylabel=ylabel, delta_label=delta_label, minimum_counts=(3, 3, 3))
    credit_presence(tables["credit_presence"])
    encoder_values = encoder_original_values(tables["encoder_raw"],
                                             tables["credit_presence_raw"],
                                             tables["encoder"])
    encoder_values.to_csv(OUT / "03_encoder_credit_original_values.csv",
                          index=False, lineterminator="\n")
    plot_encoder_original_values(
        encoder_values, metric="mean_spatial_information_bits",
        name="03_encoder_credit_original_values", ylabel="DG spatial score",
        upper=0.23)
    plot_encoder_original_values(
        encoder_values, metric="active_map_cosine_mean",
        name="03_encoder_map_cosine_original_values", ylabel="Map cosine",
        upper=0.17)
    encoder_credit(tables["encoder"], tables["peaks"])
    worker_bonus(tables["online_pairs"])
    online_worker = tables["online_pairs"][
        tables["online_pairs"].contrast == "temporal_minus_constant_hit_bonus"]
    worker_specs = (
        (online_worker, "visited_cell_fraction",
         "04_worker_coverage_values_and_delta", "Visited fraction", "Δ visited fraction"),
        (online_worker, "spatial_information",
         "04_worker_online_info_values_and_delta", "Online DG score", "Δ online DG score"),
        (tables["worker_fields"].rename(
            columns={"temporal": "first", "constant": "second",
                     "encoder_credit": "fixed_factor"}),
         "mean_spatial_information_bits", "04_worker_frozen_info_values_and_delta",
         "DG info (bits)", "Δ DG info (bits)"),
    )
    for source, metric, name, ylabel, delta_label in worker_specs:
        paired_values_and_difference(
            source[source.metric == metric], name=name,
            groups=("temporal", "constant"), group_column="fixed_factor",
            labels=("T", "C"), colors=("#30363B", "#75838C"),
            first_label="Temporal hit", second_label="Constant hit",
            ylabel=ylabel, delta_label=delta_label, minimum_counts=(5, 5))
    metadata = {
        "schema": "cosyne27_compact_figures_v6", "font": "Arial",
        "verified_font_file": font_path, "font_pt": 10,
        "default_width_inches_per_panel": 1.9,
        "behavior_panel_size_inches": BEHAVIOR_PANEL_SIZE_IN,
        "unit_of_replication": "training seed",
        "matplotlib_version": matplotlib.__version__,
        "pandas_version": pd.__version__,
        "mean_marker": "black horizontal line; individual seeds are colored points",
        "online_uncertainty": "mean ± 1 SEM across five training seeds at each 1M-step grid point",
        "behavior_absolute_candidate": {
            "table": str((OUT / "02_behavior_expression_absolute_by_seed.csv").relative_to(ROOT)),
            "panels": ["decoder-1 bits per activation on equally weighted shared bins",
                       "decoder-1 split-half map correlation on policy-native occupancy"],
            "control": "uniform-random actions", "experimental": "own policy",
            "evaluation_repeats_averaged_per_training_seed": 2,
            "training_seeds_per_architecture": 3, "nonnegative_y_axes_start_at_zero": True,
        },
        "paired_absolute_and_difference_candidates": {
            "panel_inches": [1.9, 1.9],
            "point_unit": "one training seed, paired within each factor cell",
            "absolute_encoding": "open square second arm, filled circle first arm, line paired seed",
            "difference_encoding": "first minus second, points seeds, black segment mean",
            "fresh_worker_seed_counts": (
                online_worker[online_worker.metric == "visited_cell_fraction"]
                .groupby("fixed_factor").seed.nunique().to_dict()),
        },
        "encoder_three_level_originals": {
            "table": str((OUT / "03_encoder_credit_original_values.csv").relative_to(ROOT)),
            "table_sha256": hashlib.sha256(
                (OUT / "03_encoder_credit_original_values.csv").read_bytes()).hexdigest(),
            "historical_no_credit": "NONE_STOP, same saved observation panel",
            "fresh_constant_and_temporal": "100M COSYNE factorial checkpoints",
            "cross_cohort_differences_are_not_effect_estimates": True,
            "fresh_encoder_paired_seed_counts": (
                tables["encoder"][tables["encoder"].metric ==
                                  "mean_spatial_information_bits"]
                .groupby("contrast").seed.nunique().to_dict()),
            "fresh_C15_temporal_absolute_seeds": int(len(encoder_values[
                (encoder_values.metric == "mean_spatial_information_bits") &
                (encoder_values.factor_cell == "C15") &
                (encoder_values.encoder_credit == "temporal")])),
            "fresh_C15_paired_difference_seeds": int(len(tables["encoder"][
                (tables["encoder"].metric == "mean_spatial_information_bits") &
                (tables["encoder"].contrast == "C15")])),
            "C05_constant_hit_no_credit": "not run",
        },
        "sources": {name: {"path": str(path.relative_to(ROOT)),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for name, path in SOURCES.items()},
    }
    (OUT / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
