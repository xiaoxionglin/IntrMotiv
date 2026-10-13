# Figure 2 story layout

[Editable figure](Figure2_story.svg) · [print PDF](Figure2_story.pdf) · [preview](Figure2_story.png) · [behavior-stat alternatives](Figure2_behavior_candidate_stats.pdf) · [source manifest](Figure2_story_manifest.json)

This **7.5 × 3.19-inch** option puts exploration (A) and the behavior ablation (B) on one line, then shows push (C) and pull (D) ablations below, with four C15 DG maps stacked at the right (E). A's narrow path-efficiency chart follows the Bernstein poster's colored seed dots with short black mean bars. B's black schematic reads Env → DG → Pol.; its crossed loop marks the policy intervention. The ten original quantitative result plots retain their size and data. The push and pull mini-schematics retain both directed arrows; their crosses mark the ablated connection. The author's [original SVG](Figure2_fiddle.svg) and prior [compact options](Figure2_tetris_README.md) remain available.

## A: behavior examples and swappable statistics

The trajectory thumbnails show 10,001-observation frozen own-policy probes at 100 million training frames. They keep the source segment colors and a common spatial extent. NG is seed 99, UG is seed 123, and FG is seed 17. The UG example was changed after comparing all five terminal own-policy probes: seed 123 has far less repeated circling than seed 99 and visits 312 versus 190 of 361 bins. Its probe path efficiency is 0.062 versus 0.012 for seed 99; seed 41 is slightly higher at 0.079 but visits only 205 bins. These examples are illustrative, not matched-seed or median examples. The [UG seed-123 full plot](figure2_candidates/UG_s123_100m_trajectory.png) was rendered with the canonical spatial plotter from its stored pose trace; its source path and hash are in the manifest.

The path-efficiency chart uses **five terminal training seeds per design** (8, 17, 41, 99, 123) from the canonical [trajectory summary](../../06_experiments/results/cosyne_2027_checkpoint_fields/graph_trajectory_summary/trajectory_by_snapshot.csv). Each design uses temporal DG credit; UG also uses temporal worker-hit credit. Path efficiency is the sum of each independent segment's start-to-end displacement divided by total traveled distance across all segments. The colored dots are seeds and the black bars are arithmetic means, following the Bernstein poster's compact dot-plot style. They are descriptive summaries, not confidence intervals. **The dots come from the latest 100,000 online own-policy observations near 100 million training frames; the adjacent trajectory examples are separate 10,001-observation frozen-policy probes at those checkpoints.**

Three other measures from those **same 15 runs** are rendered as individual SVG/PDF/PNG cards for swapping into A:

| Measure | Selectable PDF | Meaning |
| --- | --- | --- |
| Path efficiency | [path efficiency](Figure2_behavior_path_efficiency.pdf) | Segment end-to-end displacement divided by traveled path length. |
| Visited bins | [visited-bin fraction](Figure2_behavior_visited_cell_fraction.pdf) | Fraction of 361 spatial bins visited in the snapshot. |
| Stationary steps | [stationary-step fraction](Figure2_behavior_stationary_step_fraction.pdf) | Fraction of valid transitions below the workflow's stationary-distance threshold. |
| Step distance | [mean physical step distance](Figure2_behavior_mean_physical_step_distance.pdf) | Mean distance per valid transition in DMLab units. |

The [candidate sheet](Figure2_behavior_candidate_stats.pdf) shows the four alternatives together at print size. The visited-bin plot intentionally shows 0.8–0.9 on its vertical axis to resolve this cohort's narrow range. None of these alternatives is a separate experiment or a causal ablation.

## B–E: retained comparisons and examples

**B, behavior ablation:** The cross denotes replacement of learned actions with uniform random actions while the network remains frozen. DG information and readout-layer split-half reliability are learned-policy minus random-action differences. Dots are 10 NG, 20 UG, and eight FG terminal models, each averaged over two paired 50,000-decision probes. The policies sample different trajectories.

**C, push ablation:** Temporal minus mean-matched constant DG credit across five matched seeds per design. The first three representation measures compare trained networks on the same 10,001 observations within each pair; reliable graph edges come from separate own-policy snapshots. The worker setting is held fixed within each pair, and UG keeps temporal worker-hit credit.

**D, pull ablation:** Temporal minus constant worker-hit credit while DG credit remains temporal, across five matched UG and FG seeds at PPO discount $\gamma=0.999$ and GAE $\lambda=0.995$. The representation and graph measures come from separate policy-driven 100,000-decision snapshots and therefore include trajectory differences. The push and pull cohorts have different policy-credit settings, so equal-looking vertical positions do not imply equal effect magnitudes.

**E, DG fields:** Units 4, 7, 12, and 15 come from a separate C15 temporal DG-credit, seed-123 frozen-policy probe near 100 million frames. Its 10,001 decisions visited 307 of 361 bins. Gray denotes unvisited bins; all four maps use the source atlas's common within-run color scale. These examples do not enter A–D. Unit 4 meets the archived strict single-component criterion; the other three show different spatial responses.

The main manuscript currently includes a different Figure 2 composite. This PDF is a selectable layout option; its caption and panel references must be used together if promoted into `main.tex`.

To regenerate after changing the author's source canvas, run `build_figure2_tetris.py` first, then `build_figure2_story.py`, and finally `build_overleaf_package.py` (plus `--full` for the expanded ZIP). The story builder reads the checked-in compact SVG and does not rebuild unrelated PDFs.

## Concise caption draft

**Policy behavior and temporal credit shape spatial representations.** **A,** Example NG, UG, and FG terminal 10k-probe trajectories (seeds 99, 123, and 17) and path efficiency from separate 100k online windows across five training seeds per design. Path efficiency is summed segment displacement divided by total path length; colored dots are seeds and black bars are means. **B,** Replacing learned actions with random actions changes DG information and readout reliability in frozen networks. **C,** Temporal versus constant DG credit (push), with matched observations for representation measures. **D,** Temporal versus constant worker-hit credit (pull), with temporal DG credit fixed; these measures use each policy's own trajectories. **E,** Four example DG maps (units 4, 7, 12, and 15) from a separate FG temporal-credit model (seed 123); gray bins were unvisited. In B–D, dots represent models or matched seed pairs, boxes show quartiles and medians, and whiskers show observed extrema. Positive differences in B favor learned actions; in C–D they favor temporal credit.
