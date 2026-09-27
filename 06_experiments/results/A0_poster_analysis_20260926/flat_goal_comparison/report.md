# Flat versus goal-conditioned architectures: poster choice

**Analysis date:** 28 September 2026. This comparison extends the [A0 analysis plan](../../../../05_plans/A0_poster_analysis_plan_20260926.md) with a matched corrected-core flat anchor. The [editable all-seed SVG](architecture_summary.svg) and [per-run table](matched_terminal_per_run.csv) are the entry points. All five cells used the same fixed 900-decision no-reward open field, F16 DG, seeds 8/99/123, and the exact **100,040,704-frame** terminal checkpoint. The training curves are terminal-window means over the last 10M frames; spatial and looping measures come from separate 10,000-decision stochastic frozen-policy probes.

## What the matched comparison shows

| Cell | Architecture | Coverage AUC / episode | Unique cells / episode | 20-decision short returns among mobile windows | 40-decision short returns among mobile windows | Target-hit lift |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| C01 | Flat temporal-distance decoder | 38.0 | 47.1 | 69.9% | 25.7% | n/a |
| C02 | Goal-conditioned, delayed target | 45.1 | 63.6 | 46.4% | 29.4% | 0.959 |
| C03 | Goal-conditioned, immediate target | 23.2 | 27.9 | 88.5% | 14.9% | 1.003 |
| C05 | Goal-conditioned + DG regularization | 42.0 | 53.3 | 13.1% | 35.9% | 1.091 |
| C15 | Goal-conditioned + UCB frontier | 78.6 | 128.1 | 5.9% | 14.6% | 0.898 |

Each value is the mean of the **three seed values**, not a pooled estimate from overlapping trajectory windows. Seed dots remain visible in the [figure](architecture_summary.svg). C15 exceeds C01 in coverage AUC and unique cells and has fewer 20-decision short returns in **all three paired seeds**. The coverage-AUC increases are +37.1, +51.4, and +33.3; the short-return changes are −28.4, −76.3, and −87.2 percentage points for seeds 8, 99, and 123. C05 also has lower 20-decision returns and slightly higher coverage in all seeds. Its 40-decision return rate, however, is **higher** than C01 on average. C02's direction is mixed across seeds, and C03 has worse coverage and more 20-decision returns than flat. Goal conditioning alone is therefore **not** a replicated fix in this set of cells.

The short-return test asks whether a 20- or 40-decision window traverses more than 500 DMLab position units yet ends within 100 units of its start. Windows crossing an episode reset are excluded. The reported fraction is conditional on such *mobile* windows, to distinguish looping from standing still. C01 is mobile in 99.5% of valid 20-decision windows; C05 in only 19.8%; C15 in 58.4%. C05's apparent short-cycle reduction accompanies much slower motion and cannot be read as an unqualified exploration gain. C15's three seed-99 frozen trajectories visually show the stronger spread, but the formal coverage figures above come from training episodes, not those probes. [Probe table and denominator counts](matched_terminal_per_run.csv).

The DG representation does not improve with the behavior result. Across the frozen probes, mean active-unit spatial information is 0.182 bits for C01, 0.110 for C05, and 0.135 for C15; mean active-map cosine is 0.062, 0.127, and 0.116, respectively. Higher cosine means more redundant maps. C15 reaches 16 distinct active-unit peak bins on average, compared with 15 for C01, but that count alone does not establish compact place fields. These maps are policy-trajectory conditioned, so even the observed representation differences are descriptive.

The mechanism claim is weaker than the behavior claim. The flat decoder receives a dense reward for time to *some* DG event, which can reward repeatable cycles; the goal-conditioned worker receives a target-gated reward. This is the design distinction explained in [Control Representation Principle](../../../../05_plans/control_representation_principle.md). Yet C15 also adds a UCB frontier manager and passive graph, and C05 adds DG punishment and row repulsion. C15's target-hit lift is **below shuffled-target reference in every seed** (mean 0.898), and its mean action sensitivity is only 0.0019. Its 99.2% known-edge fraction is stored evidence, not proof of directed navigation. C05's lift is 1.059, 1.063, and 1.150 across seeds, with mean action sensitivity 0.026, so it is the more relevant decoder candidate; its modest coverage gain and time-scale-dependent looping result keep it exploratory. We cannot attribute the reduced loops or C15 coverage to goal conditioning or improved decoder learning in isolation.

**Cross-batch check:** The older fixed-flat `encourage` versus global-HRL 5k/sim seed-99 terminal probes do **not** repeat the corrected-core loop ordering: their 20-decision mobile-window return fractions are 1.3% flat versus 2.0% HRL, and 40-decision fractions are 2.5% versus 6.7%. This is a different algorithm generation and one seed, so it is a boundary on generalization rather than a pooled counterestimate. The [calculated cross-batch table](older_batch_seed99_check.csv) records both original NEMO2 pose paths; the [older batch report](../../../recent_batch_statistics_report.md) provides the experiment context.

## Detailed exemplars for poster selection

The A0 plan limits the final poster to two detailed examples. **C01 seed 99 versus C15 seed 99** gives the clearest visual contrast in short cycling and arena coverage, with a prominent caption that C15 is a goal-conditioned *frontier package*. C01's repeated circular routes appear in its [trajectory and occupancy](c01_seed99/trajectory_occupancy.svg); C15's larger spread appears in its [trajectory and occupancy](c15_seed99/trajectory_occupancy.svg). The seed-99 20-decision short-return fractions are 79.2% versus 2.9%, and frozen-probe visited bins are 264 versus 310 of 361. These are illustrations of the predeclared three-seed comparison, not selected as typical seed values.

| Diagnostic | Flat C01 seed 99 | Goal + frontier C15 seed 99 | Goal + DG regularization C05 seed 99 (alternate) |
| --- | --- | --- | --- |
| Top-four DG place fields | [SVG](c01_seed99/place_fields.svg) | [SVG](c15_seed99/place_fields.svg) | [SVG](c05_seed99/place_fields.svg) |
| Full trajectory and occupancy | [SVG](c01_seed99/trajectory_occupancy.svg) | [SVG](c15_seed99/trajectory_occupancy.svg) | [SVG](c05_seed99/trajectory_occupancy.svg) |
| Local occupancy flow | [SVG](c01_seed99/flow.svg) | [SVG](c15_seed99/flow.svg) | [SVG](c05_seed99/flow.svg) |
| DG spatial kernel | [SVG](c01_seed99/dg_kernel.svg) | [SVG](c15_seed99/dg_kernel.svg) | [SVG](c05_seed99/dg_kernel.svg) |
| Stored graph | Flat policy has no target graph | [C15 and C05 graph matrices](stored_graphs_seed99.svg) | [C15 and C05 graph matrices](stored_graphs_seed99.svg) |

### Option events on goal-conditioned trajectories

The earlier trajectory and occupancy panels contain only position records; their
source probes did not save option resets or target-hit flags. To mark the actual
events, we ran fresh stochastic frozen-policy replays from the **same terminal
checkpoints** for the two goal-conditioned exemplars. The event figures show
blue hollow circles at `option_reset` and gold stars at `target_hit`, read from
the policy core at the **same observation** as each plotted position. A target
hit can coincide with the start of the next option. These replays are separate
sampled trajectories, so their event markers must not be transferred to the
older paths above.

| Goal-conditioned exemplar | Full-replay starts / hits | Complete marked trajectory | First episode, with less overplotting | Event records |
| --- | ---: | --- | --- | --- |
| C05 seed 99, goal + DG regularization | 1,505 / 483 | [Editable SVG](c05_seed99/trajectory_option_events_full.svg) | [Editable SVG](c05_seed99/trajectory_option_events_first_episode.svg) | [CSV](c05_seed99/option_events.csv) and [summary](c05_seed99/event_summary.json) |
| C15 seed 99, goal + UCB frontier | 123 / 5 | [Editable SVG](c15_seed99/trajectory_option_events_full.svg) | [Editable SVG](c15_seed99/trajectory_option_events_first_episode.svg) | [CSV](c15_seed99/option_events.csv) and [summary](c15_seed99/event_summary.json) |

Each replay contains 10,001 observation-time decisions (the canonical evaluator
includes decision zero), split into 12 reset segments. The first 900-decision
episode has 134 starts and 40 hits for C05, and 11 starts and **zero** hits for
C15. All recorded hits coincide with option starts. C05's hits visibly cluster
along the lower and right boundaries, whereas C15 records very few hits despite
its broad trajectory. For the poster, the full C15 panel makes the gap between
arena exploration and target-hit control clear; the first C05 episode makes
the repeated boundary events easier to inspect. These markers do not strengthen
the claim that C15 learned a better target-conditioned decoder.

The complete pose-plus-event streams remain in the active NEMO2 analysis
workspace at `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/flat_goal_option_events_20260928/output/raw/`.
The [two-row replay manifest](goal_option_manifest.tsv) identifies the staged
checkpoint inputs; [runtime provenance](goal_option_provenance.json) records
the evaluator commit, source hashes, Python/PyTorch versions, and job IDs.
The original checkpoint paths are in the
[per-run table](matched_terminal_per_run.csv); their SHA-256 values and exact
event counts are in the linked summaries. We reused the established
[place-field evaluator](../../../capture_goal_option_trajectory.py) for the
rollouts, adding only a read-only event capture; the
[SVG renderer](../../../render_goal_option_trajectory.py) and
[compute-node runner](../../../run_goal_option_sweep_single.sh) reproduce these
panels. As with the rest of this comparison, a star records the model's own
DG-target criterion, not independently verified spatial arrival.
In the event CSV, `new_goal_id` is the goal selected after the core update;
on a hit row it can therefore identify the next option rather than the completed
target. The full streams also preserve timeout flags, although the plots mark
only starts and hits.

The [graph plot](stored_graphs_seed99.svg) shows checkpoint-stored edge confidence divided by attempts for each directed DG pair; white means no attempt or the diagonal. C15's nearly full matrix must be paired with its below-reference target-hit lift. The spatial kernels use the existing [population-vector Pearson kernel implementation](../../../collect_poster_population_kernels.py) on each policy's **own** frozen DG rate map, with at least five observations per cell and ten eligible cell pairs per offset. They are descriptive kernels, **not** a common-history causal representation comparison, and they do not include CA3 or decoder-1. The complete [C01](c01_seed99/dg_kernel.npz), [C05](c05_seed99/dg_kernel.npz), and [C15](c15_seed99/dg_kernel.npz) kernel arrays retain eligible pair counts. The local flow panels reuse the [existing flow implementation](../../../render_poster_frozen.py), and each exemplar has its [flow cells](c01_seed99/flow_cells.csv), [C05 cells](c05_seed99/flow_cells.csv), or [C15 cells](c15_seed99/flow_cells.csv).

## Original files and interpretation limits

- **Training scalars:** [corrected-core per-run terminal table](../../corrected_core_reevaluation_20260902/per_run_terminal_10m.csv), interpreted by the [original study report](../../../corrected_core_reevaluation_20260901.md). Source events remain in the NEMO2 run directories named in that table.
- **Frozen probes:** original files remain under `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/corrected_core_candidates_20260902_place_fields/raw/`. The [per-run table](matched_terminal_per_run.csv) gives the exact original `pose.csv`, `place_fields.npz`, and checkpoint paths for each condition and seed. These bulk raw probes stay in the NEMO2 workspace.
- **Stored graphs:** [compact C05/C15 terminal graph arrays](../../../data/flat_goal_comparison_20260927/stored_graph_seed99.json), extracted read-only from the checkpoint paths embedded in that file.
- **Reproduction:** [analysis and SVG renderer](../../../render_flat_goal_comparison.py) and [checkpoint graph extractor](../../../export_flat_goal_graphs.py). The renderer reuses the canonical [flow](../../../render_poster_frozen.py) and [kernel](../../../collect_poster_population_kernels.py) functions. To regenerate it locally, stage the listed original probe files under the renderer input root; the archived [derived field summary](../../../data/flat_goal_comparison_20260927/corrected_core_candidates_20260902_place_fields/summary/derived_place_field_metrics.csv) and [graph arrays](../../../data/flat_goal_comparison_20260927/stored_graph_seed99.json) are lightweight inputs in the vault. All new plots are editable SVG.

For a poster claim specifically about the *decoder learning mechanism*, the next experiment should hold the DG encoder, manager, frontier rule, and reward scale fixed while changing only whether the decoder is given the requested DG identity; evaluate target-conditioned arrival above shuffled targets and loop returns across multiple lags on matched reset seeds. Current data support **“a goal-conditioned frontier architecture reduced short cycles and improved coverage”**, with the qualification that the local worker did not show convincing commanded-target control. They do not support **“goal conditioning trained a better decoder and thereby removed loops.”**
