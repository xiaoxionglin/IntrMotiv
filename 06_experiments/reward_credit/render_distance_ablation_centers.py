"""Plot one post-inhibition DG peak per active unit on each common-panel map.

All 44 terminal checkpoints see the same 10,001 observations and 242 visited
arena bins. A peak is the maximum visited-bin response, not proof of a compact
field. Shared peak bins list every unit occupying that bin.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARENA = (100, 2000)


def setup_font() -> None:
    path = Path(font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans"), fallback_to_default=False
    ))
    if path.suffix.lower() not in {".ttf", ".otf"} or not path.is_file():
        raise RuntimeError("Verified scalable font required")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 24,
                         "axes.titlesize": 26, "axes.labelsize": 24,
                         "xtick.labelsize": 22, "ytick.labelsize": 22,
                         "pdf.fonttype": 42})


def load_maps(root: Path, run_name: str) -> dict[str, np.ndarray]:
    matches = list(root.glob(f"{run_name}__*/place_fields.npz"))
    if len(matches) != 1:
        raise ValueError(f"Expected one common-panel NPZ for {run_name}; got {len(matches)}")
    with np.load(matches[0], allow_pickle=False) as raw:
        return {key: np.asarray(raw[key]) for key in (
            "occupancy", "post_inhibition_rate_maps", "post_inhibition_active_fraction",
            "post_inhibition_field_eligible", "post_inhibition_field_mono",
        )}


def centers_for_run(meta: pd.Series, arrays: dict[str, np.ndarray]) -> list[dict]:
    visited = arrays["occupancy"] > 0
    maps = arrays["post_inhibition_rate_maps"]
    if maps.shape != (19, 19, 16) or int(visited.sum()) != 242:
        raise ValueError(f"Unexpected common panel in {meta.run_name}")
    rows = []
    for unit in range(16):
        observed = np.where(visited, maps[:, :, unit], -np.inf)
        peak = float(observed.max())
        active = peak > 0
        row, col = np.unravel_index(int(observed.argmax()), visited.shape) if active else (-1, -1)
        rows.append({
            "run_name": meta.run_name, "family": meta.family,
            "worker_bonus": meta.worker_bonus, "credit": meta.credit,
            "seed": int(meta.seed), "unit": unit, "active": active,
            "peak_row": row if active else np.nan,
            "peak_col": col if active else np.nan,
            "x_center": 150 + col * 100 if active else np.nan,
            "y_center": 150 + row * 100 if active else np.nan,
            "peak_rate": peak if active else 0,
            "active_fraction": float(arrays["post_inhibition_active_fraction"][unit]),
            "outer_border": bool(row in (0, 18) or col in (0, 18)) if active else False,
            "field_eligible": bool(arrays["post_inhibition_field_eligible"][unit]),
            "single_field": bool(arrays["post_inhibition_field_mono"][unit]),
        })
    return rows


def plot_run(meta: pd.Series, arrays: dict[str, np.ndarray], centers: pd.DataFrame, output_dir: Path) -> None:
    visited = arrays["occupancy"] > 0
    cmap = plt.get_cmap("Greys").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(9.2, 9.2), layout="constrained")
    ax.imshow(np.ma.masked_where(~visited, np.ones(visited.shape)), origin="lower",
              extent=(*ARENA, *ARENA), cmap=cmap, vmin=0, vmax=3,
              interpolation="nearest")
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for item in centers.itertuples():
        if item.active:
            groups[(int(item.peak_row), int(item.peak_col))].append(int(item.unit))
    for (row, col), units in groups.items():
        x, y = 150 + col * 100, 150 + row * 100
        shared = len(units) > 1
        ax.scatter(x, y, s=1700 if shared else 1050,
                   c="#B44F10" if shared else "#006BA4",
                   edgecolors="white", linewidths=1.8, zorder=3)
        # A shared-bin marker shows its count; the companion per-unit CSV gives
        # the exact IDs without squeezing a multi-line list into one grid cell.
        ax.text(x, y, str(len(units)) if shared else str(units[0]),
                color="white", fontsize=25,
                ha="center", va="center", weight="bold", zorder=4)
    active = int(centers.active.sum())
    border = int(centers.outer_border.sum())
    ax.set(xlim=(50, 2050), ylim=(50, 2050), aspect="equal",
           xlabel="x (arena units)", ylabel="y (arena units)")
    ax.set_xticks((100, 1000, 2000))
    ax.set_yticks((100, 1000, 2000))
    ax.set_title(f"{meta.family} {meta.worker_bonus} · seed {meta.seed}\n"
                 f"{meta.credit.replace('_', ' ').title()}\n"
                 f"{active} active · {len(groups)} peak bins · {border} outer peaks")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{meta.family.lower()}_{meta.worker_bonus}_{meta.credit.lower()}_s{meta.seed}_peak_centers"
    fig.savefig(output_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_unit_atlas(meta: pd.Series, arrays: dict[str, np.ndarray], output_dir: Path) -> None:
    """Four units per page keep field-shape inspection readable in Obsidian."""
    maps = arrays["post_inhibition_rate_maps"]
    visited = arrays["occupancy"] > 0
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    base = f"{meta.family.lower()}_{meta.worker_bonus}_{meta.credit.lower()}_s{meta.seed}"
    for page in range(4):
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.5), layout="constrained")
        for ax, unit in zip(axes.flat, range(4 * page, 4 * page + 4)):
            this_map = maps[:, :, unit]
            peak = float(np.max(this_map[visited]))
            normalized = this_map / peak if peak > 0 else this_map
            ax.imshow(np.ma.array(normalized, mask=~visited), origin="lower",
                      cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            note = "silent" if peak <= 0 else (
                "single" if arrays["post_inhibition_field_mono"][unit] else ""
            )
            ax.set_title(f"DG {unit}" + (f" · {note}" if note else ""), fontsize=26)
            ax.set_xticks(())
            ax.set_yticks(())
        fig.suptitle(f"{meta.family} {meta.worker_bonus} · {meta.credit.replace('_', ' ').title()}\n"
                     f"Seed {meta.seed} · DG {4*page}–{4*page+3} · each unit / own peak",
                     fontsize=26)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_dir / f"{base}_fields_units_{4*page:02d}_{4*page+3:02d}"
        fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel_root", type=Path)
    parser.add_argument("online_terminal_csv", type=Path)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("figures_dir", type=Path)
    args = parser.parse_args()
    setup_font()
    metadata = pd.read_csv(args.online_terminal_csv)
    if len(metadata) != 44 or not metadata.run_name.is_unique:
        raise ValueError("Expected 44 completed terminal runs")
    records = []
    summaries = []
    reference_occupancy = None
    for meta in metadata.itertuples(index=False):
        arrays = load_maps(args.panel_root, meta.run_name)
        if reference_occupancy is None:
            reference_occupancy = arrays["occupancy"]
        elif not np.array_equal(reference_occupancy, arrays["occupancy"]):
            raise ValueError(f"Observation panel changed for {meta.run_name}")
        run_rows = centers_for_run(meta, arrays)
        records.extend(run_rows)
        current = pd.DataFrame(run_rows)
        points = current.loc[current.active, ["peak_row", "peak_col"]].to_numpy(float)
        distances = np.sqrt(((points[:, None] - points[None, :]) ** 2).sum(-1))
        pairwise = distances[np.triu_indices(len(points), k=1)]
        summaries.append({
            "run_name": meta.run_name, "family": meta.family,
            "worker_bonus": meta.worker_bonus, "credit": meta.credit,
            "seed": int(meta.seed), "active_units": len(points),
            "unique_peak_bins": len({tuple(p) for p in points}),
            "outer_border_peaks": int(current.outer_border.sum()),
            "mean_pairwise_peak_distance_bins": float(pairwise.mean()) if len(pairwise) else np.nan,
        })
        plot_run(meta, arrays, current, args.figures_dir)
        if (meta.seed == 99 and meta.family in {"C05", "C15"}
                and meta.worker_bonus == "temporal"
                and meta.credit in {"CONSTANT_STOP", "NONE_STOP"}):
            plot_unit_atlas(meta, arrays, args.figures_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.results_dir / "common_panel_peak_centers_per_unit.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.results_dir / "common_panel_peak_center_summary.csv", index=False)
    print(f"Rendered {len(summaries)} one-map-per-run peak-center figures")


if __name__ == "__main__":
    main()
