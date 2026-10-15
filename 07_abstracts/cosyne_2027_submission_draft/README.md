# COSYNE 2027 submission assets

For later papers and talks, start with the [reusable visual asset index](REUSABLE_ASSETS.md). It links individual plot cards, icons, fields, trajectories, graphs, and the lightweight tables needed to revise them.

## Selected submission

- [Figure 1 PDF](Figure1.pdf) · [editable SVG](Figure1.svg): architecture, learning course, and example DG fields. Its author-edited source is documented in [Figure 1 source notes](figure1_candidates/06_architecture_v3_goal_gate_header_README.md).
- [Figure 2 PDF](Figure2.pdf) · [editable SVG](Figure2.svg) · [preview](Figure2.png): trajectory and policy controls, push and pull contrasts, and metric icons. See the [layout notes](Figure2_layout_README.md) and [manifest](Figure2_layout_manifest.json).
- [main.tex](main.tex) uses the selected PDFs through `figures/Figure1.pdf` and `figures/Figure2.pdf`. The former checked-in `main.pdf` predated this pairing and was removed; compile the current source in Overleaf.
- [Overleaf ZIP](cosyne_2027_overleaf_sources.zip) packages the selected figures, source assets, data, and retained variants. Regenerate it with `build_overleaf_package.py`; `--full` adds individual analysis plots.

The manuscript remains a draft. [Open questions](OPEN_QUESTIONS.md) records claim and length checks, and [panel provenance](PANEL_PROVENANCE.json) records source hashes. The selected palette is orange push, violet pull, blue/red/green/cyan DG identities, and slate/olive/rose NG/UG/FG; its [manifest](figure_palette_candidates/manifest.json) and retained [Figure 2 palette source](figure_palette_candidates/priority_rgb_orange_violet_figure2.svg) document the transformation.

## Distinct figure variants retained

| Asset | Reason to retain |
| --- | --- |
| [Figure 2 with DG response maps](Figure2_story.pdf) · [SVG](Figure2_story.svg) · [notes](Figure2_story_README.md) | Shows four C15 fields beside the A–D statistics. |
| [Figure 2 with paired peak maps](Figure2_story_peaks.pdf) · [SVG](Figure2_story_peaks.svg) · [notes](Figure2_story_peaks_README.md) | Compares temporal and constant-credit peak locations for NG, UG, and FG. |
| [Figure 1 with absolute reward](figure1_candidates/04_architecture_v3_dg_reward_fields.pdf) and [relative reward](figure1_candidates/05_architecture_v3_relative_reward.pdf) | Displays different reward interpretations alongside DG examples. |
| [Figure 2 matched representation matrix](figure2_candidates/02_factor_matrix.pdf), [behavior and control outcomes](figure2_candidates/03_control_outcomes.pdf), [metric inventory](figure2_candidates/04_metric_inventory.pdf), and [original paired values](figure2_candidates/05_compact_vertical_pairs.pdf) | Preserves distinct result selections or absolute-value views for scientific review. See the [candidate notes](figure2_candidates/README.md). |
| [Linear](../../06_experiments/results/cosyne_2027_intrinsic_reward_course/01b_online_intrinsic_reward_course.pdf) and [log-y](../../06_experiments/results/cosyne_2027_intrinsic_reward_course/01c_online_intrinsic_reward_course_log.pdf) intrinsic-reward courses | Same data with complementary absolute and low-magnitude views. |

## Reusable source assets

- [Full DG atlases and unit tables](figure1_candidates/full_place_field_atlases.md), [screened field options](figure1_candidates/place_field_options/README.md), and [goal-policy icons](figure1_candidates/goal_policy_icon_options/README.md).
- [Individual metric icons](Figure2_metric_icons_README.md), [spatial-information icon options](figure2_icon_candidates/spatial_information/README.md), [behavior-stat plots](Figure2_story_README.md), and [trajectory tiles](figure2_candidates/09_trajectory_thumbnails/README.md).
- [Focused Figure 2 source tables](figure2_candidates/08_focused_canvas/README.md), [loose editable canvas](figure2_candidates/07_loose_canvas/README.md), and `data/` for lightweight figure inputs. The original author canvas (`Figure2_fiddle.svg`), derived construction SVGs (`Figure2_tetris.svg`, `Figure2_story_icons.svg`), and build scripts remain because the selected artwork depends on them.
- `06_experiments/results/cosyne_2027_*` contains the canonical analysis tables and source plots. Raw NEMO2 evaluator arrays remain in allocated workspace storage.

Early whole-figure layout and palette exports, duplicate manuscript composites, and dated ZIP backups were removed. Git history retains their provenance if an earlier design must be examined; regeneration scripts may still create exploratory outputs during future design work.
