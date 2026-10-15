# Reusable IntrMotiv visual assets

This is the entry point for making later paper figures and presentation slides from the COSYNE work. It points to **standalone parts and source data**, not just assembled submission figures. The [submission README](README.md) indexes the selected Figure 1/2 and scientifically distinct whole-figure variants.

Use the SVG for editable vector artwork, PDF for print placement, and PNG for quick review. Most small statistical cards were sized for a 7.5-inch abstract figure; check their text at the intended slide or page size before reuse. The source README and manifest beside each asset give its run identities, metric definitions, and source hashes. NG, UG, and FG mean no-goal C01, uniform-goal C05, and frontier-goal C15.

## Fast route for a short project introduction

| What to show | Ready source | What it supports |
| --- | --- | --- |
| Architecture and feedback paths | [Author architecture SVG](figures/architecture_uploaded.svg) · [selected Figure 1 SVG](Figure1.svg) | Editable overview of visual input, DG, CA3, readout, and push/pull paths. [Figure 1 notes](figure1_candidates/06_architecture_v3_goal_gate_header_README.md) explain which modules are fixed and learned. |
| One policy/goal schematic | [Side-gate icon SVG](figure1_candidates/goal_policy_icon_options/simple/05_side_gate.svg) · [other gate options](figure1_candidates/goal_policy_icon_options/simple/README.md) | Small vector insert for a bridge slide. The icon is explanatory artwork, not measured data. |
| Three exploration paths | [NG](figure2_candidates/09_trajectory_thumbnails/NG_65_fragments_tile.svg) · [UG](figure2_candidates/09_trajectory_thumbnails/UG_65_fragments_tile.svg) · [FG](figure2_candidates/09_trajectory_thumbnails/FG_65_fragments_tile.svg) | Equal-size illustrative tiles from selected 100k-observation own-policy windows. Keep the [trajectory source notes](figure2_candidates/09_trajectory_thumbnails/README.md) with them; selected seeds are examples, not group estimates. |
| One DG field example | [C15 temporal seed-123 atlas](figure1_candidates/full_seed_atlases/c15_temporal_s123.png) · [compact field options](figure1_candidates/place_field_options/README.md) | Shows response maps, with occupancy visible. Do not treat a selected unit as field prevalence or a fixed-trajectory treatment effect. |
| One result plot | [Online spatial course](figure2_candidates/08_focused_canvas/components/spatial_emergence.svg) · [DG behavior information](figure2_candidates/08_focused_canvas/components/behavior_dg_information.svg) | The first is a five-seed online training course; the second is frozen-model learned-policy versus random-action expression. They answer different questions. |

For a brief Berlin talk bridge from *Lin et al. 2026* to newer projects, these IntrMotiv pieces can be used independently. The separate [SimGaY repository](https://github.com/xiaoxionglin/SimGaY) has a [task-state diagram](https://github.com/xiaoxionglin/SimGaY/blob/main/03_task_design/Full%20task%20state%20transitions.png) and [representative training illustration](https://github.com/xiaoxionglin/SimGaY/blob/main/03_task_design/Representative%20animal%20training%20over%20time.png). Check that project's current interpretation before placing either in a talk. This index does not equate the three projects' results.

## Diagrams and metric icons

| Topic | Individual assets | Source notes |
| --- | --- | --- |
| Goal selection and policy gate | [Two lanes](figure1_candidates/goal_policy_icon_options/01_two_lanes.svg), [event timeline](figure1_candidates/goal_policy_icon_options/02_event_timeline.svg), [goal gate](figure1_candidates/goal_policy_icon_options/03_goal_gate.svg), and [six simple gate variants](figure1_candidates/goal_policy_icon_options/simple/README.md) | [Icon option notes](figure1_candidates/goal_policy_icon_options/README.md) |
| Path efficiency | [SVG](Figure2_metric_icon_path_efficiency.svg) · [PNG](Figure2_metric_icon_path_efficiency.png) | [Metric icon definitions](Figure2_metric_icons_README.md) |
| DG map overlap | [SVG](Figure2_metric_icon_overlap.svg) · [PNG](Figure2_metric_icon_overlap.png) | Same notes; cosine compares maps from distinct DG units. |
| Peak coverage | [SVG](Figure2_metric_icon_coverage.svg) · [PNG](Figure2_metric_icon_coverage.png) | Same notes; the drawing is illustrative, not a measured map. |
| Reliable directed edges | [SVG](Figure2_metric_icon_edges.svg) · [PNG](Figure2_metric_icon_edges.png) | Same notes; stored edges do not establish executed goal arrival. |
| Split-half field reliability | [SVG](Figure2_metric_icon_reliability.svg) · [PNG](Figure2_metric_icon_reliability.png) | Same notes; the two examples depict the same unit across time. |
| Spatial information | [Selected SVG](figure2_icon_candidates/spatial_information/perlin_strength_variants/07_very_high.svg) · [other concepts](figure2_icon_candidates/spatial_information/README.md) | [Strength-variant notes](figure2_icon_candidates/spatial_information/perlin_strength_variants/README.md); the icon is schematic. |

The [metric-icon preview sheet](Figure2_metric_icons_preview.png) and [goal-gate overview](figure1_candidates/goal_policy_icon_options/overview.png) help browse without opening each vector file.

## Standalone quantitative plot cards

The [focused canvas](figure2_candidates/08_focused_canvas/README.md) is the compact source for **17 individual SVG/PDF cards**. Its [plotted-differences table](figure2_candidates/08_focused_canvas/plotted_seed_differences.csv) and [manifest](figure2_candidates/08_focused_canvas/manifest.json) record the exact comparisons. Each card below has a PDF with the same stem in the same `components/` directory.

| Family | Individual SVG cards | Comparison to carry into a caption |
| --- | --- | --- |
| Online training | [Spatial emergence](figure2_candidates/08_focused_canvas/components/spatial_emergence.svg) | Five-seed online DG spatial-information course; network and policy change during training. |
| Frozen behavior | [DG information](figure2_candidates/08_focused_canvas/components/behavior_dg_information.svg), [map overlap](figure2_candidates/08_focused_canvas/components/behavior_map_overlap.svg), [peak coverage](figure2_candidates/08_focused_canvas/components/behavior_peak_coverage.svg), [readout reliability](figure2_candidates/08_focused_canvas/components/behavior_readout_reliability.svg) | Learned actions minus random actions in 38 frozen models. The last card measures readout MLP layer 1, while the other three measure DG. Policies visit different paths. |
| Encoder push | [DG information](figure2_candidates/08_focused_canvas/components/push_dg_information.svg), [overlap](figure2_candidates/08_focused_canvas/components/push_map_overlap.svg), [coverage](figure2_candidates/08_focused_canvas/components/push_peak_coverage.svg), [peak distance](figure2_candidates/08_focused_canvas/components/push_peak_distance.svg), [hit rate](figure2_candidates/08_focused_canvas/components/push_hit_rate.svg), [graph edges](figure2_candidates/08_focused_canvas/components/push_graph_edges.svg) | Temporal minus constant DG encoder credit. The first four use common-observation terminal replay; hits and edges use policy-dependent measurements. |
| Worker pull | [DG information](figure2_candidates/08_focused_canvas/components/pull_dg_information.svg), [overlap](figure2_candidates/08_focused_canvas/components/pull_map_overlap.svg), [coverage](figure2_candidates/08_focused_canvas/components/pull_peak_coverage.svg), [peak distance](figure2_candidates/08_focused_canvas/components/pull_peak_distance.svg), [hit rate](figure2_candidates/08_focused_canvas/components/pull_hit_rate.svg), [graph edges](figure2_candidates/08_focused_canvas/components/pull_graph_edges.svg) | Temporal minus constant worker-hit credit in the **separate high-policy-credit regimen**. The spatial cards use own-policy snapshots; do not compare their absolute peak-area denominators with push. |

The [loose-canvas component index](figure2_candidates/07_loose_canvas/README.md) covers **69 SVG cards and labels**, each also available as PDF and PNG. It adds raw control-versus-tested pair plots, CA3 and readout-layer plots, trajectory and field crops, graph/coverage cards, and high-credit worker comparisons. Its [component manifest](figure2_candidates/07_loose_canvas/manifest.json) maps every card to the source table and protocol. Use that index when the focused cards omit a metric or when a paper needs original paired values instead of a difference box.

## Spatial examples, trajectories, and graphs

| Family | Where to browse | Source and use |
| --- | --- | --- |
| Complete DG atlases | [Atlas index](figure1_candidates/full_place_field_atlases.md) · [shared-scale PNGs](figure1_candidates/full_seed_atlases/) · [per-unit min–max PNGs](figure1_candidates/full_seed_atlases_minmax/) | Forty condition/seed atlases with all 16 DG units. [Seed](figure1_candidates/full_seed_atlases/terminal_seed_metrics.csv) and [unit](figure1_candidates/full_seed_atlases/terminal_unit_metrics.csv) tables help select a field without relying on appearance alone. State which scale is used. |
| Smaller screened field sets | [Five four-field options](figure1_candidates/place_field_options/README.md) · [eight per architecture](figure1_candidates/place_field_options/eight_per_architecture/README.md) | Use the accompanying selected-unit CSV and occupancy notes when making an example panel. |
| Trajectory tiles | [9/33/65-fragment variants](figure2_candidates/09_trajectory_thumbnails/README.md) · [selected fragment arrays](figure2_candidates/09_trajectory_thumbnails/selected_fragments.npz) | Standalone SVG/PDF/PNG paths and a small array for regeneration. They are illustrations from selected runs, not matched treatment estimates. |
| Full selected trajectories and graphs | [Selected-figure source table](../../06_experiments/results/cosyne_2027_checkpoint_fields/selected_figures/selected_figure_sources.csv) · [trajectory PNGs](../../06_experiments/results/cosyne_2027_checkpoint_fields/selected_figures/trajectories/) · [graph PNGs](../../06_experiments/results/cosyne_2027_checkpoint_fields/selected_figures/graphs/) | Selected policy-generated examples. Stored graph reachability is distinct from executed command success. |

## Data for rebuilding or changing a figure

| Question | Lightweight source |
| --- | --- |
| Replot the 17 focused cards | [Plotted values](figure2_candidates/08_focused_canvas/plotted_seed_differences.csv), [frozen behavior geometry by model](figure2_candidates/08_focused_canvas/behavior_geometry_by_model.csv), [repeat values](figure2_candidates/08_focused_canvas/behavior_geometry_by_repeat.csv), and [source manifest](figure2_candidates/08_focused_canvas/manifest.json). |
| Change Figure 2's selected behavior metric | [DG reliability](data/fresh_behavior_dg_reliability.csv), [fresh behavior training seeds](data/fresh_behavior_training_seeds.csv), [DG bits per step](figure2_candidates/08_focused_canvas/fresh_behavior_dg_bits_per_step.csv), and [story manifest](Figure2_story_manifest.json). Match the layer and unit to the label. |
| Choose a different DG field | [Terminal unit metrics](figure1_candidates/full_seed_atlases/terminal_unit_metrics.csv), [seed metrics](figure1_candidates/full_seed_atlases/terminal_seed_metrics.csv), [atlas manifest](figure1_candidates/full_seed_atlases/manifest.json), and [selected-unit table](figure1_candidates/place_field_options/selected_units.csv). |
| Replot training, field, trajectory, or graph statistics | The canonical [COSYNE results folders](../../06_experiments/results/) contain source CSVs beside their analytical plots. Start with [online curves](../../06_experiments/results/cosyne_2027_online_wandb/), [checkpoint fields](../../06_experiments/results/cosyne_2027_checkpoint_fields/), and [snapshot peak spread](../../06_experiments/results/cosyne_2027_snapshot_peak_spread/). Use the corresponding analysis note under [COSYNE studies](../../06_experiments/cosyne_2027/) for metric support and caveats. |

The editable source canvases and renderers live next to these assets: [focused canvas](figure2_candidates/08_focused_canvas/focused_canvas_editable.svg), [loose canvas](figure2_candidates/07_loose_canvas/figure2_loose_canvas_editable.svg), and [scatter canvas](figure2_candidates/07_loose_canvas/figure2_scatter_canvas_editable.svg). The [Overleaf package builder](build_overleaf_package.py) selects upload assets; it does not replace the source tables or the full local component library.
