"""Render the compact, source-linked episode-long DG comparison figures.

Inputs are canonical StudySpec collectors and a compact summary of exact-start
trials. The raw training and evaluator artifacts remain in the NEMO2 workspace.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "episode_long_comparison_20261009"
FIGURES = RESULTS / "figures"
REWARD = HERE.parent / "reward_credit" / "results"
SEEDS = (8, 99, 123)
ARM_LABELS = {
    "oracle_film": "Prescribed: 4 fixed + 12 context",
    "learned4_film": "Learned-4: 4 goals + 12 context",
    "c15_film": "C15: 16 learned goals",
    "c05_film": "C05: 16 learned goals",
}
PRESCRIBED_CENTERS = ((550, 550), (1450, 550), (550, 1450), (1650, 1650))


def setup_style() -> None:
    findfont(FontProperties(family="DejaVu Sans"), fallback_to_default=False)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 19,
            "axes.titlesize": 21,
            "axes.labelsize": 19,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 16,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_peak_centers() -> None:
    units = pd.read_csv(RESULTS / "online_spatial_150m_per_unit.csv")
    units["arm"] = units.run_name.str.extract(r"^OEL(?:DG|C05)_(.*?)_S\d+$")[0].str.lower()
    observed = units.loc[units.dominant_peak_x.notna() & units.dominant_peak_y.notna()].copy()
    observed["outer_two_bin_ring"] = (
        observed.dominant_peak_x.le(250) | observed.dominant_peak_x.ge(1850) |
        observed.dominant_peak_y.le(250) | observed.dominant_peak_y.ge(1850)
    )
    summary = observed.groupby("run_name", as_index=False).agg(
        peak_rows=("unit_id", "count"),
        outer_two_bin_ring_peaks=("outer_two_bin_ring", "sum"),
    )
    summary["outer_two_bin_ring_fraction"] = (
        summary.outer_two_bin_ring_peaks / summary.peak_rows
    )
    summary.to_csv(RESULTS / "peak_boundary_150m_per_run.csv", index=False)
    arm_map = {
        "oracle_film": "oracle_film",
        "learned4_film": "learned4_film",
        "c15_film": "c15_film",
        "c05_film": "c05_film",
    }
    for arm, label in ARM_LABELS.items():
        selected = units.loc[units.arm.eq(arm_map[arm])].copy()
        assert len(selected) == 48, (arm, len(selected))
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 7.3), sharex=True, sharey=True)
        palette = plt.get_cmap("tab20")
        for ax, seed in zip(axes, SEEDS):
            rows = selected.loc[selected.run_name.str.endswith(f"_S{seed}")]
            assert len(rows) == 16
            for row in rows.itertuples(index=False):
                goal = arm in ("c15_film", "c05_film") or row.unit_id < 4
                color = palette(int(row.unit_id))
                marker = "s" if goal and arm in ("oracle_film", "learned4_film") else "o"
                if arm == "oracle_film" and row.unit_id < 4:
                    # The online mono-field classifier excludes replacement rows;
                    # their known Gaussian centers are authoritative here.
                    x, y = PRESCRIBED_CENTERS[row.unit_id]
                else:
                    x, y = row.dominant_peak_x, row.dominant_peak_y
                if not np.isfinite(x) or not np.isfinite(y):
                    continue
                ax.scatter(x, y, color=color,
                           s=125, marker=marker, edgecolors="black", linewidths=0.9, zorder=3)
            ax.set_title(f"Seed {seed}")
            ax.set_xlim(100, 2000)
            ax.set_ylim(100, 2000)
            ax.set_aspect("equal")
            ax.set_xticks([100, 550, 1000, 1450, 2000])
            ax.set_yticks([100, 550, 1000, 1450, 2000])
            ax.grid(color="#D8DEE4", linewidth=0.8)
            ax.set_xlabel("x (DMLab units)")
        axes[0].set_ylabel("y (DMLab units)")
        title = "150M DG centers/peaks" if arm == "oracle_film" else "150M observed DG peak bins"
        fig.suptitle(f"{title} | {label}", y=1.02, fontsize=23)
        handles = [Line2D([], [], marker=("s" if arm in ("oracle_film", "learned4_film") and unit < 4 else "o"),
                          color="none", markerfacecolor=palette(unit), markeredgecolor="black",
                          markersize=12, label=str(unit)) for unit in range(16)]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.035),
                   ncol=8, frameon=False, title="DG unit ID", fontsize=19, title_fontsize=19)
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        save(fig, f"peak_centers_150m_{arm}")


def plot_frozen_100m_peak_centers(npz_dir: Path) -> None:
    """Plot one location per active DG unit for every 100M run.

    Learned-unit peaks come from the post-inhibition frozen-policy field
    classifier. If activity is present but the classifier does not yield a
    qualified peak, use the maximum of its smoothed map over visited bins and
    mark that fallback hollow. Fixed IDs 0–3 use declared centers because the
    classifier leaves replacement rows blank. A peak is not a mono-field.
    """
    files = sorted(npz_dir.glob("*__s*.npz"))
    expected_conditions = set(pd.read_csv(RESULTS / "frozen_100m_reward_per_run.csv").condition)
    expected_conditions |= set(pd.read_csv(RESULTS / "frozen_100m_older_credit_per_run.csv").condition)
    assert len(files) == 42 and len(expected_conditions) == 14
    rows = []
    for path in files:
        condition, seed_text = path.stem.rsplit("__s", 1)
        seed = int(seed_text)
        assert condition in expected_conditions and seed in SEEDS
        with np.load(path, allow_pickle=False) as data:
            xy = data["post_inhibition_field_dominant_peak_xy"]
            bins = data["post_inhibition_field_dominant_peak_bin"]
            activity = data["post_inhibition_active_fraction"]
            smoothed = data["post_inhibition_smoothed_rate_maps"]
            occupancy = data["post_inhibition_occupancy"]
            assert xy.shape == (16, 2) and bins.shape == (16, 2)
            for unit in range(16):
                fixed = "ORACLE" in condition and unit < 4
                source = "prescribed_center" if fixed else "observed_peak"
                if fixed:
                    # The canonical field classifier leaves replacement rows
                    # blank; their declared Gaussian centers are the location.
                    x, y = PRESCRIBED_CENTERS[unit]
                    bin_x, bin_y = int((x - 100) // 100), int((y - 100) // 100)
                else:
                    x, y = map(float, xy[unit])
                    bin_x, bin_y = map(int, bins[unit])
                active = bool(activity[unit] > 0)
                if active and not fixed and not (np.isfinite(x) and np.isfinite(y)):
                    scores = np.where(occupancy > 0, smoothed[:, :, unit], -np.inf)
                    assert np.isfinite(scores).any()
                    bin_x, bin_y = map(int, np.unravel_index(np.argmax(scores), scores.shape))
                    x, y = 150 + 100 * bin_x, 150 + 100 * bin_y
                    source = "fallback_smoothed_max"
                assert not active or (np.isfinite(x) and np.isfinite(y))
                rows.append(dict(condition=condition, seed=seed, unit=unit,
                                 x=x if active else np.nan,
                                 y=y if active else np.nan,
                                 peak_bin_x=bin_x if active else -1,
                                 peak_bin_y=bin_y if active else -1,
                                 location_source=source,
                                 active=active,
                                 outer_two_bin_ring=(active and (x <= 250 or x >= 1850
                                                               or y <= 250 or y >= 1850))))
    peaks = pd.DataFrame(rows).sort_values(["condition", "seed", "unit"])
    assert len(peaks) == 42 * 16
    reported = pd.concat([
        pd.read_csv(RESULTS / "frozen_100m_reward_per_run.csv"),
        pd.read_csv(RESULTS / "frozen_100m_older_credit_per_run.csv"),
    ], ignore_index=True)
    counts = peaks.groupby(["condition", "seed"]).active.sum()
    for record in reported.itertuples(index=False):
        assert counts.loc[(record.condition, record.seed)] == record.active_units
    peaks.to_csv(RESULTS / "frozen_100m_peak_centers_per_unit.csv", index=False)
    summary = peaks.groupby(["condition", "seed"], as_index=False).agg(
        active_peaks=("active", "sum"), outer_two_bin_ring_peaks=("outer_two_bin_ring", "sum"),
    )
    summary["outer_two_bin_ring_fraction"] = (
        summary.outer_two_bin_ring_peaks / summary.active_peaks
    )
    summary.to_csv(RESULTS / "frozen_100m_peak_summary_per_run.csv", index=False)

    palette = plt.get_cmap("tab20")
    for condition in sorted(expected_conditions):
        selected = peaks.loc[peaks.condition.eq(condition)]
        assert len(selected) == 48
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 7.3), sharex=True, sharey=True)
        for ax, seed in zip(axes, SEEDS):
            unit_rows = selected.loc[selected.seed.eq(seed)]
            for row in unit_rows.itertuples(index=False):
                if not row.active:
                    continue
                fixed = "ORACLE" in condition and row.unit < 4
                fallback = row.location_source == "fallback_smoothed_max"
                ax.scatter(row.x, row.y, s=125, marker="s" if fixed else "o",
                           facecolor="white" if fallback else palette(row.unit),
                           edgecolor=palette(row.unit) if fallback else "black",
                           linewidth=2.0 if fallback else 0.9, zorder=3)
            ax.set_title(f"Seed {seed} | {int(unit_rows.active.sum())}/16 peaks")
            ax.set_xlim(100, 2000)
            ax.set_ylim(100, 2000)
            ax.set_aspect("equal")
            ax.set_xticks([100, 550, 1000, 1450, 2000])
            ax.set_yticks([100, 550, 1000, 1450, 2000])
            ax.grid(color="#D8DEE4", linewidth=0.8)
            ax.set_xlabel("x (DMLab units)")
        axes[0].set_ylabel("y (DMLab units)")
        title = ("fixed centers + learned peaks" if "ORACLE" in condition
                 else "DG dominant peaks")
        fig.suptitle(f"100M frozen: {title} | {condition}", y=1.02, fontsize=23)
        handles = [Line2D([], [], marker=("s" if "ORACLE" in condition and unit < 4 else "o"),
                          color="none", markerfacecolor=palette(unit), markeredgecolor="black",
                          markersize=12, label=str(unit)) for unit in range(16)]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.035),
                   ncol=8, frameon=False, title="DG unit ID", fontsize=19, title_fontsize=19)
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        save(fig, f"frozen_peak_centers_100m_{condition.lower()}")


def plot_command_lift() -> None:
    data = pd.read_csv(RESULTS / "matched_command_75m_per_horizon.csv")
    groups = [
        ("OFDG_LEARNED", "64-decision deadline", "#577590"),
        ("OELDG_LEARNED4_FILM", "Episode-long", "#C75028"),
        ("OFDG_ORACLE", "64-decision deadline", "#577590"),
        ("OELDG_ORACLE_FILM", "Episode-long", "#C75028"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.7), sharey=True)
    lower = float(data.paired_lift.min())
    upper = float(data.paired_lift.max())
    padding = max(0.025, 0.12 * (upper - lower))
    for ax, family in zip(axes, ("LEARNED", "ORACLE")):
        family_rows = data.loc[data.run_name.str.contains(family)]
        for run, label, color in groups:
            if family not in run:
                continue
            rows = data.loc[data.run_name.eq(run)]
            for seed, seed_rows in rows.groupby("seed"):
                ax.plot(seed_rows.horizon, seed_rows.paired_lift, marker="o",
                        markersize=7, linewidth=1.7, alpha=0.78, color=color,
                        label=label if seed == rows.seed.min() else None)
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 128, 256, 900])
        ax.set_xticklabels(["64", "128", "256", "900"])
        ax.set_xlim(56, 1050)
        ax.set_ylim(min(-0.02, lower - padding), max(0.02, upper + padding))
        ax.set_xlabel("Decision window")
        seed_counts = family_rows.groupby("run_name").seed.nunique().tolist()
        seed_label = (str(seed_counts[0]) if min(seed_counts) == max(seed_counts)
                      else f"{min(seed_counts)}–{max(seed_counts)}")
        ax.set_title(("Learned detector" if family == "LEARNED" else "Fixed field entry")
                     + f" ({seed_label} seeds per arm)")
        ax.grid(axis="y", color="#D8DEE4")
    axes[0].set_ylabel("Commanded − alternative hit probability")
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = {label: handle for handle, label in zip(handles, labels) if label in
            ("64-decision deadline", "Episode-long")}
    fig.legend(uniq.values(), uniq.keys(), loc="lower center",
               bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
    fig.tight_layout()
    save(fig, "matched_command_lift_75m")


def plot_deadline_paired_lift() -> None:
    """Show episode-minus-finite command lift by paired training seed.

    The executed command alternatives share exact starts within each arm;
    the finite and episode policies do not necessarily share physical starts.
    """
    data = pd.read_csv(RESULTS / "matched_command_75m_paired_horizon_effects.csv")
    assert len(data) == 18
    colors = {8: "#0072B2", 99: "#D55E00", 123: "#009E73"}
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharey=True)
    for ax, family in zip(axes, ("Prescribed", "Learned-4 detector")):
        rows = data.loc[data.family.eq(family)]
        assert len(rows) == 9
        for seed in SEEDS:
            selected = rows.loc[rows.seed.eq(seed)].sort_values("horizon")
            assert selected.horizon.tolist() == [64, 128, 256]
            ax.plot(selected.horizon, selected.episode_minus_finite_lift,
                    marker="o", linewidth=2.2, markersize=9,
                    color=colors[seed], label=f"Seed {seed}")
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 128, 256], ["64", "128", "256"])
        ax.set_xlim(56, 285)
        ax.set_xlabel("Decision window")
        ax.set_title(family)
        ax.grid(axis="y", color="#D8DEE4")
    axes[0].set_ylabel("Episode-long − finite command lift")
    fig.suptitle("75M goal-deadline contrast | paired training seeds", fontsize=23)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08),
               ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.055, 1, 0.92))
    save(fig, "deadline_paired_lift_75m")


def plot_c15_value_stop_field_tradeoff() -> None:
    """Show matched 100M C15 field specificity against frozen path support."""
    data = pd.read_csv(RESULTS / "frozen_100m_reward_per_run.csv")
    conditions = {
        ("Nearest DG", "Continue"): "LCDG_C15_FILM",
        ("Nearest DG", "Stop at hit"): "GVSD_C15_NEAREST",
        ("Command source", "Continue"): "SDHG_C15_SOURCE",
        ("Command source", "Stop at hit"): "GVSD_C15_SOURCE",
    }
    palette = {8: "#0072B2", 99: "#D55E00", 123: "#009E73"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.6))
    metrics = (
        ("active_unit_mean_si_bits", "Mean active-unit spatial info (bits)"),
        ("visited_cells", "Visited bins in frozen 10k path"),
    )
    for ax, (metric, ylabel) in zip(axes, metrics):
        for clock, linestyle in (("Nearest DG", "-"), ("Command source", "--")):
            for seed in SEEDS:
                values = []
                for stop in ("Continue", "Stop at hit"):
                    rows = data.loc[data.condition.eq(conditions[(clock, stop)]) & data.seed.eq(seed)]
                    assert len(rows) == 1
                    values.append(float(rows[metric].iloc[0]))
                ax.plot([0, 1], values, color=palette[seed], linestyle=linestyle,
                        marker="o", markersize=8, linewidth=2.2)
        ax.set_xticks([0, 1], ["Continue", "Stop at hit"])
        ax.set_xlim(-0.12, 1.12)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D8DEE4")
    fig.suptitle("C15 at 100M: goal-hit value boundary", fontsize=21)
    legend = [Line2D([], [], color=palette[seed], linewidth=2.5,
                     label=f"Seed {seed}") for seed in SEEDS]
    legend += [Line2D([], [], color="#333333", linestyle=style, linewidth=2.5,
                      label=clock) for clock, style in (("Nearest DG", "-"),
                                                        ("Command source", "--"))]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.06),
               ncol=5, frameon=False, fontsize=14)
    fig.tight_layout(rect=(0, 0.10, 1, 0.92))
    save(fig, "c15_value_stop_100m_field_tradeoff")


def reward_factor_table() -> pd.DataFrame:
    paths = [
        REWARD / "long_credit_dg_20261008/online_95_100m/per_run.csv",
        REWARD / "source_distance_hit_dg_20261009/online_95_100m/per_run.csv",
        REWARD / "goal_value_stop_dg_20261009/online_95_100m/per_run.csv",
    ]
    mapping = {
        "LCDG_ORACLE_FILM": ("Prescribed", "Nearest", "Continue"),
        "LCDG_C15_FILM": ("C15", "Nearest", "Continue"),
        "SDHG_ORACLE_SOURCE": ("Prescribed", "Source", "Continue"),
        "SDHG_C15_SOURCE": ("C15", "Source", "Continue"),
        "SDHG_C05_NEAREST": ("C05", "Nearest", "Continue"),
        "SDHG_C05_SOURCE": ("C05", "Source", "Continue"),
    }
    for family in ("ORACLE", "C15", "C05"):
        for clock in ("NEAREST", "SOURCE"):
            mapping[f"GVSD_{family}_{clock}"] = (
                {"ORACLE": "Prescribed", "C15": "C15", "C05": "C05"}[family],
                clock.title(), "Stop at hit",
            )
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame = frame.loc[frame.condition.isin(mapping)].copy()
    frame[["dg_family", "reward_clock", "value_target"]] = frame.condition.apply(
        lambda condition: pd.Series(mapping[condition])
    )
    assert len(frame) == 36 and not frame.duplicated(
        ["dg_family", "reward_clock", "value_target", "seed"]
    ).any()
    frame["hit_rate_per_1000"] = 1000 * frame.target_hit_numerator / frame.target_hit_event_count
    keep = ["condition", "seed", "dg_family", "reward_clock", "value_target", "coverage_auc",
            "hit_rate_per_1000", "active_target_fraction", "target_action_probability_tv",
            "value_loss", "advantage_std"]
    frame[keep].to_csv(RESULTS / "reward_factor_100m_per_run.csv", index=False)
    return frame


def plot_reward_factors(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0), sharey=True)
    colors = {8: "#0072B2", 99: "#D55E00", 123: "#009E73"}
    contrasts = [
        ("Source − nearest | continuing value", "Continue", "clock"),
        ("Stop − continue | source clock", "Source", "value"),
    ]
    for ax, (title, fixed, contrast) in zip(axes, contrasts):
        for x, family in enumerate(("Prescribed", "C15", "C05")):
            for seed in SEEDS:
                rows = frame.loc[frame.dg_family.eq(family) & frame.seed.eq(seed)]
                if contrast == "clock":
                    rows = rows.loc[rows.value_target.eq(fixed)].set_index("reward_clock")
                    delta = rows.loc["Source", "coverage_auc"] - rows.loc["Nearest", "coverage_auc"]
                else:
                    rows = rows.loc[rows.reward_clock.eq(fixed)].set_index("value_target")
                    delta = rows.loc["Stop at hit", "coverage_auc"] - rows.loc["Continue", "coverage_auc"]
                ax.scatter(x + (SEEDS.index(seed)-1)*0.10, delta, s=110,
                           c=colors[seed], edgecolors="#222222", linewidths=0.6,
                           label=f"Seed {seed}" if x == 0 else None)
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_xticks(range(3), ["Prescribed", "C15", "C05"])
        ax.set_title(title)
        ax.set_xlabel("DG family")
        ax.grid(axis="y", color="#D8DEE4")
    axes[0].set_ylabel("Paired coverage AUC change (points)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.07),
               ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, "reward_factors_100m_coverage")


def plot_longer_credit() -> None:
    """Compare the original and longer-credit runs at the same training age."""
    original = pd.read_csv(RESULTS / "online_95_100m_orthogonal_per_run.csv")
    longer = pd.read_csv(REWARD / "long_credit_dg_20261008/online_95_100m/per_run.csv")
    effects = []
    for family in ("ORACLE", "C15"):
        for seed in SEEDS:
            before = original.loc[
                original.condition.eq(f"OELDG_{family}_FILM") & original.seed.eq(seed)
            ].iloc[0]
            after = longer.loc[
                longer.condition.eq(f"LCDG_{family}_FILM") & longer.seed.eq(seed)
            ].iloc[0]
            before_hits = 1000 * before.target_hit_numerator / before.target_hit_event_count
            after_hits = 1000 * after.target_hit_numerator / after.target_hit_event_count
            effects.append(dict(
                dg_family="Prescribed" if family == "ORACLE" else "C15",
                seed=seed,
                baseline_coverage_auc=before.coverage_auc,
                longer_coverage_auc=after.coverage_auc,
                coverage_auc_delta=after.coverage_auc - before.coverage_auc,
                baseline_hits_per_1000=before_hits,
                longer_hits_per_1000=after_hits,
                hits_per_1000_delta=after_hits - before_hits,
                baseline_goal_action_tv=before.target_action_probability_tv,
                longer_goal_action_tv=after.target_action_probability_tv,
            ))
    frame = pd.DataFrame(effects)
    frame.to_csv(RESULTS / "longer_credit_100m_paired.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.7))
    colors = {8: "#0072B2", 99: "#D55E00", 123: "#009E73"}
    for ax, metric, ylabel in (
        (axes[0], "coverage_auc_delta", "Coverage AUC change (points)"),
        (axes[1], "hits_per_1000_delta", "Hit-rate change per 1,000 events"),
    ):
        for x, family in enumerate(("Prescribed", "C15")):
            subset = frame.loc[frame.dg_family.eq(family)]
            for seed in SEEDS:
                row = subset.loc[subset.seed.eq(seed)].iloc[0]
                ax.scatter(x + (SEEDS.index(seed)-1)*0.09, row[metric],
                           s=140, color=colors[seed], edgecolors="black", linewidths=0.6,
                           label=f"Seed {seed}" if x == 0 else None)
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_xticks((0, 1), ("Prescribed", "C15"))
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D8DEE4")
    axes[0].set_title("Longer credit − original | 100M")
    axes[1].set_title("Logged commanded hits | 100M")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08),
               ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, "longer_credit_100m_paired")


def plot_frozen_150m_fields() -> None:
    """Show replicated frozen visited support and qualifying DG field counts."""
    frame = pd.concat([
        pd.read_csv(RESULTS / "frozen_150m_eldg_per_run.csv"),
        pd.read_csv(RESULTS / "frozen_150m_c05_per_run.csv"),
    ], ignore_index=True)
    assert len(frame) == 12
    frame["mono_field_count"] = (
        frame.mono_field_fraction * frame.field_eligible_units
    ).round().astype(int)
    frame.to_csv(RESULTS / "frozen_150m_all_arms_per_run.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0))
    colors = {8: "#0072B2", 99: "#D55E00", 123: "#009E73"}
    pairs = [
        ("Prescribed − learned-4", "OELDG_ORACLE_FILM", "OELDG_LEARNED4_FILM"),
        ("C05 − C15", "OELC05_C05_FILM", "OELDG_C15_FILM"),
    ]
    for x, (label, treatment, reference) in enumerate(pairs):
        for seed in SEEDS:
            t = frame.loc[frame.condition.eq(treatment) & frame.seed.eq(seed)].iloc[0]
            r = frame.loc[frame.condition.eq(reference) & frame.seed.eq(seed)].iloc[0]
            axes[0].scatter(x + (SEEDS.index(seed)-1)*0.09,
                            t.visited_cells - r.visited_cells, s=140,
                            color=colors[seed], edgecolors="black", linewidths=0.6,
                            label=f"Seed {seed}" if x == 0 else None)
    axes[0].axhline(0, color="#333333", linewidth=1)
    axes[0].set_xticks((0, 1), ("Prescribed −\nlearned-4", "C05 − C15"))
    axes[0].set_xlim(-0.35, 1.35)
    axes[0].set_ylabel("Paired visited-bin change / 361")
    axes[0].set_title("Frozen 10k paths at 150M")
    axes[0].grid(axis="y", color="#D8DEE4")
    arm_order = [
        ("Prescribed\ncontext", "OELDG_ORACLE_FILM"),
        ("Learned-4", "OELDG_LEARNED4_FILM"),
        ("C15", "OELDG_C15_FILM"),
        ("C05", "OELC05_C05_FILM"),
    ]
    for x, (_, condition) in enumerate(arm_order):
        for seed in SEEDS:
            row = frame.loc[frame.condition.eq(condition) & frame.seed.eq(seed)].iloc[0]
            axes[1].scatter(x + (SEEDS.index(seed)-1)*0.09, row.mono_field_count,
                            s=140, color=colors[seed], edgecolors="black", linewidths=0.6)
    axes[1].set_xticks(range(4), [label for label, _ in arm_order])
    axes[1].set_xlim(-0.35, 3.35)
    axes[1].set_ylim(-0.5, 7)
    axes[1].set_ylabel("Qualifying mono-field DG rows")
    axes[1].set_title("Learned DG rows; fixed rows excluded")
    axes[1].grid(axis="y", color="#D8DEE4")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.10),
               ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, "frozen_150m_fields")


def plot_longer_credit_frozen_fields() -> None:
    """Pair 100M frozen field metrics by family and seed."""
    older = pd.read_csv(RESULTS / "frozen_100m_older_credit_per_run.csv")
    longer = pd.read_csv(RESULTS / "frozen_100m_reward_per_run.csv")
    older_units = pd.read_csv(RESULTS / "frozen_100m_older_credit_per_unit.csv")
    longer_units = pd.read_csv(
        REWARD / "long_credit_dg_20261008/frozen_100m/per_unit_place_field_metrics.csv"
    )
    rows = []
    for family in ("ORACLE", "C15"):
        for seed in SEEDS:
            before = older.loc[
                older.condition.eq(f"OELDG_{family}_FILM") & older.seed.eq(seed)
            ].iloc[0]
            after = longer.loc[
                longer.condition.eq(f"LCDG_{family}_FILM") & longer.seed.eq(seed)
            ].iloc[0]
            before_units = older_units.loc[
                older_units.condition.eq(f"OELDG_{family}_FILM") & older_units.seed.eq(seed)
            ]
            after_units = longer_units.loc[
                longer_units.condition.eq(f"LCDG_{family}_FILM") & longer_units.seed.eq(seed)
            ]
            if family == "ORACLE":
                before_units = before_units.loc[before_units.unit.ge(4)]
                after_units = after_units.loc[after_units.unit.ge(4)]
            before_si = before_units.spatial_information_bits.mean()
            after_si = after_units.spatial_information_bits.mean()
            rows.append(dict(
                dg_family="Prescribed" if family == "ORACLE" else "C15",
                seed=seed,
                visited_cells_before=before.visited_cells,
                visited_cells_after=after.visited_cells,
                visited_cells_delta=after.visited_cells - before.visited_cells,
                learned_si_bits_before=before_si,
                learned_si_bits_after=after_si,
                learned_si_bits_delta=after_si - before_si,
                active_map_cosine_before=before.active_map_cosine_mean,
                active_map_cosine_after=after.active_map_cosine_mean,
                active_map_cosine_delta=after.active_map_cosine_mean - before.active_map_cosine_mean,
                mono_field_count_before=round(before.field_eligible_units * before.mono_field_fraction),
                mono_field_count_after=round(after.field_eligible_units * after.mono_field_fraction),
                eligible_rows_before=before.field_eligible_units,
                eligible_rows_after=after.field_eligible_units,
            ))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "longer_credit_100m_frozen_fields_paired.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.7))
    colors = {8: "#0072B2", 99: "#D55E00", 123: "#009E73"}
    for ax, metric, ylabel in (
        (axes[0], "visited_cells_delta", "Visited-bin change / 361"),
        (axes[1], "learned_si_bits_delta", "Mean learned-unit SI change (bits)"),
    ):
        for x, family in enumerate(("Prescribed", "C15")):
            for seed in SEEDS:
                row = frame.loc[frame.dg_family.eq(family) & frame.seed.eq(seed)].iloc[0]
                ax.scatter(x + (SEEDS.index(seed)-1)*0.09, row[metric],
                           s=140, color=colors[seed], edgecolors="black", linewidths=0.6,
                           label=f"Seed {seed}" if x == 0 else None)
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_xticks((0, 1), ("Prescribed", "C15"))
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D8DEE4")
    axes[0].set_title("Frozen 10k paths at 100M")
    axes[1].set_title("Learned DG field information")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08),
               ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, "longer_credit_100m_frozen_fields")


def plot_frozen_atlas(npz_path: Path, arm: str, pre_threshold: bool) -> None:
    """Render canonical NPZ maps with report-size labels and the same map orientation."""
    with np.load(npz_path, allow_pickle=False) as data:
        occupancy = data["occupancy"].T
        key = "pre_threshold_rate_maps" if pre_threshold else "rate_maps"
        maps = data[key]
        assert maps.shape == (19, 19, 16)
        information = data["spatial_information"]
        activity = data["active_fraction"]
        finite = maps[np.isfinite(maps)]
        if pre_threshold:
            maximum = max(float(np.percentile(np.abs(finite), 98)), 1e-8)
            cmap = plt.get_cmap("coolwarm").copy()
            limits = dict(vmin=-maximum, vmax=maximum)
        else:
            maximum = max(float(np.percentile(finite, 98)), 1e-8)
            cmap = plt.get_cmap("viridis").copy()
            limits = dict(vmin=0, vmax=maximum)
        cmap.set_bad("#D9D9D9")
        fig, axes = plt.subplots(4, 4, figsize=(22, 23), constrained_layout=True)
        for unit, ax in enumerate(axes.flat):
            values = np.ma.masked_where(occupancy <= 0, maps[:, :, unit].T)
            im = ax.imshow(values, origin="upper", interpolation="nearest",
                           cmap=cmap, **limits)
            ax.set_xticks([])
            ax.set_yticks([])
            if pre_threshold:
                ax.set_title(f"DG {unit:02d} | pre-threshold", fontsize=24, pad=14)
            else:
                ax.set_title(
                    f"DG {unit:02d} | SI {information[unit]:.2f} bits | active {activity[unit]:.3f}",
                    fontsize=22, pad=14,
                )
        mode = "pre-threshold logits" if pre_threshold else "post-replacement DG activity"
        fig.suptitle(f"150M frozen fields | {ARM_LABELS[arm]} | seed 99\n{mode}; shared scale, gray = unvisited",
                     fontsize=32)
        colorbar = fig.colorbar(im, ax=axes, location="bottom", shrink=0.55,
                                pad=0.025, aspect=45)
        colorbar.ax.tick_params(labelsize=22)
        colorbar.set_label("Mean activation on occupied observations", fontsize=24)
        stem = f"frozen_{'prethreshold' if pre_threshold else 'fields'}_150m_{arm.removesuffix('_film')}_s99"
        save(fig, stem)


def plot_frozen_atlases(npz_dir: Path) -> None:
    for arm in ARM_LABELS:
        path = npz_dir / f"frozen_150m_{arm.removesuffix('_film')}_s99.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        for pre_threshold in (False, True):
            plot_frozen_atlas(path, arm, pre_threshold)


def plot_online_frozen_peak_boundary(npz_dir: Path) -> None:
    """Check whether boundary-heavy peak placement survives a frozen rollout."""
    rows = []
    for arm in ARM_LABELS:
        for seed in SEEDS:
            path = npz_dir / f"frozen_150m_{arm.removesuffix('_film')}_s{seed}.npz"
            with np.load(path, allow_pickle=False) as data:
                maps = data["rate_maps"]
                peaks = np.nanargmax(maps.reshape(-1, 16), axis=0)
                bins = np.column_stack(np.unravel_index(peaks, (19, 19)))
                if arm == "oracle_film":
                    bins = bins[4:]  # compare learned context; fixed centers are known
                outer = ((bins[:, 0] < 2) | (bins[:, 0] >= 17) |
                         (bins[:, 1] < 2) | (bins[:, 1] >= 17))
                rows.append(dict(arm=arm, seed=seed, frozen_peak_rows=len(bins),
                                 frozen_outer_peaks=int(outer.sum()),
                                 frozen_outer_fraction=float(outer.mean()),
                                 frozen_unique_peak_bins=len({tuple(row) for row in bins}),
                                 frozen_visited_cells=int((data["occupancy"] > 0).sum())))
    frozen = pd.DataFrame(rows)
    online = pd.read_csv(RESULTS / "peak_boundary_150m_per_run.csv")
    online["arm"] = online.run_name.str.extract(r"^OEL(?:DG|C05)_(.*?)_S\d+$")[0].str.lower()
    online["seed"] = online.run_name.str.extract(r"_S(\d+)$")[0].astype(int)
    merged = frozen.merge(online, on=["arm", "seed"], validate="one_to_one")
    merged.to_csv(RESULTS / "peak_boundary_online_frozen_150m.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharey=True)
    colors = {8: "#0072B2", 99: "#D55E00", 123: "#009E73"}
    for ax, arm in zip(axes, ("c15_film", "c05_film")):
        for seed in SEEDS:
            row = merged.loc[merged.arm.eq(arm) & merged.seed.eq(seed)].iloc[0]
            ax.plot((0, 1), (row.outer_two_bin_ring_fraction,
                             row.frozen_outer_fraction), marker="o", markersize=10,
                    linewidth=2.2, color=colors[seed], label=f"Seed {seed}")
        ax.set_xticks((0, 1), ("Online 100k", "Frozen 10k"))
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(0, 1)
        ax.set_title("C15" if arm == "c15_film" else "C05")
        ax.grid(axis="y", color="#D8DEE4")
    axes[0].set_ylabel("DG peak maxima in outer two-bin ring")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08),
               ncol=3, frameon=False)
    fig.suptitle("150M boundary peak concentration | same checkpoints, distinct rollouts")
    fig.tight_layout()
    save(fig, "peak_boundary_online_frozen_150m")


def plot_frozen_150m_paths(npz_dir: Path) -> None:
    """Show first four complete reset segments over each 10k frozen occupancy map."""
    arms = ("oracle_film", "learned4_film", "c15_film", "c05_film")
    short_names = {"oracle_film": "Prescribed", "learned4_film": "Learned-4",
                   "c15_film": "C15", "c05_film": "C05"}
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 13.7), sharex=True, sharey=True)
    for ax, arm in zip(axes.flat, arms):
        stem = f"frozen_150m_{arm.removesuffix('_film')}_s99"
        poses = pd.read_csv(npz_dir / f"{stem}_pose.csv")
        with np.load(npz_dir / f"{stem}.npz", allow_pickle=False) as data:
            occupancy = data["occupancy"]
            bounds = data["bounds"]
        assert len(poses) == 10001 and poses.num_traj.nunique() == 12
        ax.imshow(np.log1p(occupancy.T), extent=bounds, origin="lower",
                  cmap="Greys", vmin=0, vmax=7, alpha=0.43,
                  interpolation="nearest")
        for segment_id in range(4):
            segment = poses.loc[poses.num_traj.eq(segment_id)]
            assert len(segment) == 900
            ax.plot(segment.x, segment.y, color=colors[segment_id],
                    linewidth=1.5, alpha=0.9, label=f"Episode {segment_id + 1}")
            ax.scatter(segment.x.iloc[0], segment.y.iloc[0],
                       color=colors[segment_id], s=45, edgecolors="black",
                       linewidths=0.5, zorder=4)
        for x, y in PRESCRIBED_CENTERS:
            ax.scatter(x, y, marker="*", s=135, facecolors="white",
                       edgecolors="black", linewidths=0.9, zorder=5)
        ax.set_xlim(100, 2000)
        ax.set_ylim(100, 2000)
        ax.set_aspect("equal")
        ax.set_xticks([100, 550, 1000, 1450, 2000])
        ax.set_yticks([100, 550, 1000, 1450, 2000])
        ax.set_xlabel("x (DMLab units)")
        ax.set_ylabel("y (DMLab units)")
        ax.set_title(f"{short_names[arm]} | {int((occupancy > 0).sum())}/361 bins")
    fig.suptitle("150M frozen paths | seed 99\nFirst four 900-decision episodes over all 10k occupied observations",
                 fontsize=23)
    handles = [Line2D([], [], color=color, linewidth=3, label=f"Episode {index + 1}")
               for index, color in enumerate(colors)]
    handles.append(Line2D([], [], marker="*", color="none", markerfacecolor="white",
                          markeredgecolor="black", markersize=13,
                          label="Four fixed-site locations"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.015),
               ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    save(fig, "frozen_paths_150m_seed99")


def plot_frozen_100m_path_contrasts(npz_dir: Path) -> None:
    """Compare reset-segment paths for selected 100M frozen-policy contrasts.

    Each panel is an independent policy-driven rollout. Equal evaluation length
    makes occupancy comparable, but the plotted paths are not matched starts.
    """
    panels = (
        (
            "longer_credit_100m_paths_seed99",
            "100M frozen paths | older versus longer PPO credit | seed 99",
            (
                ("OELDG_ORACLE_FILM", "Prescribed | older credit"),
                ("LCDG_ORACLE_FILM", "Prescribed | longer credit"),
                ("OELDG_C15_FILM", "C15 | older credit"),
                ("LCDG_C15_FILM", "C15 | longer credit"),
            ),
        ),
        (
            "c15_clock_value_100m_paths_seed99",
            "100M frozen paths | C15 reward clock and value boundary | seed 99",
            (
                ("LCDG_C15_FILM", "Nearest clock | continue"),
                ("SDHG_C15_SOURCE", "Source clock | continue"),
                ("GVSD_C15_NEAREST", "Nearest clock | stop"),
                ("GVSD_C15_SOURCE", "Source clock | stop"),
            ),
        ),
    )
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    for stem, title, conditions in panels:
        fig, axes = plt.subplots(2, 2, figsize=(14.5, 13.7), sharex=True, sharey=True)
        for ax, (condition, label) in zip(axes.flat, conditions):
            poses = pd.read_csv(npz_dir / f"{condition}__s99_pose.csv")
            with np.load(npz_dir / f"{condition}__s99.npz", allow_pickle=False) as data:
                occupancy = data["occupancy"]
                bounds = data["bounds"]
            assert len(poses) == 10001 and poses.num_traj.nunique() == 12
            ax.imshow(np.log1p(occupancy.T), extent=bounds, origin="lower",
                      cmap="Greys", vmin=0, vmax=7, alpha=0.43,
                      interpolation="nearest")
            for segment_id, color in enumerate(colors):
                segment = poses.loc[poses.num_traj.eq(segment_id)]
                assert len(segment) == 900
                ax.plot(segment.x, segment.y, color=color, linewidth=1.5, alpha=0.9)
                ax.scatter(segment.x.iloc[0], segment.y.iloc[0], color=color,
                           s=45, edgecolors="black", linewidths=0.5, zorder=4)
            for x, y in PRESCRIBED_CENTERS:
                ax.scatter(x, y, marker="*", s=135, facecolors="white",
                           edgecolors="black", linewidths=0.9, zorder=5)
            ax.set_xlim(100, 2000)
            ax.set_ylim(100, 2000)
            ax.set_aspect("equal")
            ax.set_xticks([100, 550, 1000, 1450, 2000])
            ax.set_yticks([100, 550, 1000, 1450, 2000])
            ax.set_xlabel("x (DMLab units)")
            ax.set_ylabel("y (DMLab units)")
            ax.set_title(f"{label} | {int((occupancy > 0).sum())}/361 bins", pad=18)
        fig.suptitle(f"{title}\nFirst four episodes shown over all 10k occupied observations",
                     fontsize=23)
        handles = [Line2D([], [], color=color, linewidth=3,
                          label=f"Episode {index + 1}")
                   for index, color in enumerate(colors)]
        handles.append(Line2D([], [], marker="*", color="none",
                              markerfacecolor="white", markeredgecolor="black",
                              markersize=13, label="Four fixed-site locations"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.015),
                   ncol=5, frameon=False)
        fig.tight_layout(rect=(0, 0.06, 1, 0.94))
        save(fig, stem)


def plot_c05_100m_graph_attempts(npz_dir: Path) -> None:
    """Show which directed C05 pairs ever entered the graph at 100M.

    Attempted support is a graph-availability diagnostic. The stored confidence
    omits episode-censored commands and is not used as navigation evidence.
    """
    conditions = (
        ("SDHG_C05_NEAREST", "Nearest clock | continue"),
        ("SDHG_C05_SOURCE", "Source clock | continue"),
        ("GVSD_C05_NEAREST", "Nearest clock | stop"),
        ("GVSD_C05_SOURCE", "Source clock | stop"),
    )
    cmap = ListedColormap(["#E0E0E0", "#185A84"])
    cmap.set_bad("white")
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 13.2), sharex=True, sharey=True)
    for ax, (condition, label) in zip(axes.flat, conditions):
        with np.load(npz_dir / f"{condition}__s99.npz", allow_pickle=False) as data:
            attempts = data["control_attempts"]
        assert attempts.shape == (16, 16)
        attempted = attempts > 0
        assert not np.diag(attempted).any()
        status = attempted.astype(float)
        np.fill_diagonal(status, np.nan)
        ax.imshow(status, interpolation="nearest", cmap=cmap, norm=norm,
                  origin="upper")
        ax.set_xticks([0, 4, 8, 12, 15])
        ax.set_yticks([0, 4, 8, 12, 15])
        ax.set_xlabel("Target DG ID")
        ax.set_ylabel("Source DG ID")
        ax.set_title(f"{label} | {int(attempted.sum())}/240", pad=18)
    fig.suptitle("100M C05 directed graph support | seed 99\nAttempted pairs, not executed-command control",
                 fontsize=23)
    handles = [Patch(facecolor="#E0E0E0", label="Unattempted"),
               Patch(facecolor="#185A84", label="Attempted")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.015),
               ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    save(fig, "c05_clock_value_100m_graph_attempts_seed99")


def plot_150m_graph_seed99() -> None:
    """Show attempted support and stored reliability on one common 16×16 scale."""
    edges = pd.read_csv(RESULTS / "graph_edges_150m_seed99.csv")
    assert len(edges) == 4 * 16 * 15
    order = [
        ("OELDG_ORACLE_FILM_S99", "Prescribed"),
        ("OELDG_LEARNED4_FILM_S99", "Learned-4"),
        ("OELDG_C15_FILM_S99", "C15"),
        ("OELC05_C05_FILM_S99", "C05"),
    ]
    cmap = ListedColormap(["#E0E0E0", "#79B9D9", "#185A84"])
    cmap.set_bad("white")
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 13.2), sharex=True, sharey=True)
    for ax, (name, title) in zip(axes.flat, order):
        rows = edges.loc[edges.run_name.eq(name)]
        assert len(rows) == 240
        status = np.full((16, 16), np.nan)
        for row in rows.itertuples(index=False):
            status[int(row.source_unit), int(row.target_unit)] = (
                2 if bool(row.reliable) else 1 if row.attempts > 0 else 0
            )
        ax.imshow(status, interpolation="nearest", cmap=cmap, norm=norm,
                  origin="upper")
        ax.set_xticks([0, 4, 8, 12, 15])
        ax.set_yticks([0, 4, 8, 12, 15])
        ax.set_xlabel("Target DG ID")
        ax.set_ylabel("Source DG ID")
        attempted = int((rows.attempts > 0).sum())
        reliable = int(rows.reliable.sum())
        ax.set_title(f"{title} | attempted {attempted}/240; reliable {reliable}")
    fig.suptitle("150M directed graph records | seed 99\nStored reliability omits episode-end censored commands",
                 fontsize=23)
    handles = [
        Patch(facecolor="#E0E0E0", label="Unattempted"),
        Patch(facecolor="#79B9D9", label="Attempted, not reliable"),
        Patch(facecolor="#185A84", label="Stored as reliable"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.015),
               ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    save(fig, "graph_150m_seed99_status")


def plot_age_matched_coverage() -> None:
    old = HERE / "results" / "orthogonal_film_dg_20261008"
    age75 = pd.concat(
        [pd.read_csv(old / "episode_online_70_75m_per_run.csv"),
         pd.read_csv(old / "c05_online_70_75m_per_run.csv")],
        ignore_index=True,
    )
    age150 = pd.read_csv(RESULTS / "online_145_150m_per_run.csv")
    age75["age_m"] = 75
    age150["age_m"] = 150
    data = pd.concat([age75, age150], ignore_index=True)
    pairs = [
        ("Prescribed − learned-4", "OELDG_ORACLE_FILM", "OELDG_LEARNED4_FILM"),
        ("C05 − C15 package", "OELC05_C05_FILM", "OELDG_C15_FILM"),
    ]
    rows = []
    for name, treatment, reference in pairs:
        for age in (75, 150):
            for seed in SEEDS:
                selected = data.loc[data.age_m.eq(age) & data.seed.eq(seed)].set_index("condition")
                rows.append(dict(contrast=name,age_m=age,seed=seed,
                                 coverage_auc_delta=float(selected.loc[treatment, "coverage_auc"] -
                                                          selected.loc[reference, "coverage_auc"])))
    effects = pd.DataFrame(rows)
    effects.to_csv(RESULTS / "coverage_75_150m_paired.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), sharey=True)
    colors = {8: "#0072B2", 99: "#D55E00", 123: "#009E73"}
    for ax, (name, _, _) in zip(axes, pairs):
        block = effects.loc[effects.contrast.eq(name)]
        for seed in SEEDS:
            sub = block.loc[block.seed.eq(seed)].sort_values("age_m")
            ax.plot(sub.age_m, sub.coverage_auc_delta, marker="o", markersize=9,
                    linewidth=2.3, color=colors[seed], label=f"Seed {seed}")
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_xticks([75, 150], ["75M", "150M"])
        ax.set_xlim(65, 160)
        ax.set_xlabel("Matched training age")
        ax.set_title(name)
        ax.grid(axis="y", color="#D8DEE4")
    axes[0].set_ylabel("Paired coverage AUC difference (points)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08),
               ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, "coverage_75_150m_paired")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-npz-dir", type=Path,
                        help="Optional directory of canonical 150M seed-99 NPZs for legible atlases")
    parser.add_argument("--frozen-100m-npz-dir", type=Path,
                        help="Optional directory of 42 canonical 100M NPZs for run-wise peak maps")
    parser.add_argument("--frozen-100m-path-dir", type=Path,
                        help="Optional directory of seed-99 100M NPZs and pose CSVs for path contrasts")
    args = parser.parse_args()
    setup_style()
    plot_peak_centers()
    plot_command_lift()
    if (RESULTS / "matched_command_75m_paired_horizon_effects.csv").is_file():
        plot_deadline_paired_lift()
    plot_reward_factors(reward_factor_table())
    plot_c15_value_stop_field_tradeoff()
    plot_longer_credit()
    plot_frozen_150m_fields()
    plot_longer_credit_frozen_fields()
    plot_150m_graph_seed99()
    if args.frozen_npz_dir is not None:
        plot_frozen_atlases(args.frozen_npz_dir)
        plot_online_frozen_peak_boundary(args.frozen_npz_dir)
        plot_frozen_150m_paths(args.frozen_npz_dir)
    if args.frozen_100m_npz_dir is not None:
        plot_frozen_100m_peak_centers(args.frozen_100m_npz_dir)
        plot_c05_100m_graph_attempts(args.frozen_100m_npz_dir)
    if args.frozen_100m_path_dir is not None:
        plot_frozen_100m_path_contrasts(args.frozen_100m_path_dir)
    plot_age_matched_coverage()


if __name__ == "__main__":
    main()
