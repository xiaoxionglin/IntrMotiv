# Odor and CA3 goal quality: place fields, trajectories, and graphs

**Status, 8 October 2026:** the complete declared place-field protocol is analyzed: 70 independent 10k-decision rollouts. All 200 retained online spatial/graph snapshots are collected. The 40 ordinary target-hit and 40 first-distinct evaluations completed; their [command analysis](odor_goal_quality_v2_command_interventions_20261007.md) identifies stopping-rule and variable-deadline limitations. The first full-panel exact-start replay timed out without a summary; one bounded four-source replacement qualification is running. This spatial report does not infer command efficacy from graph or field plots.

## Where the outputs are

- [Checkpoint field metrics](results/odor_v2_spatial_20261007/derived_place_field_metrics.csv): seed 99 at 5M/10M/25M/50M/75M for all ten conditions, plus terminal seeds 8 and 123. The [terminal three-seed aggregate](results/odor_v2_spatial_20261007/terminal_three_seed_aggregate.csv) and [map-stability tables](results/odor_v2_spatial_20261007/stability/) are beside it.
- [Full online spatial/graph snapshot table](results/odor_v2_spatial_20261007/per_snapshot.csv): 40 runs at each of five declared milestones. The [graph-edge exposure summary](results/odor_v2_spatial_20261007/graph_terminal_exposure.csv) adds attempted-edge count and effective edge diversity at 75M. The [collector provenance](results/odor_v2_spatial_20261007/analysis_manifest.json) records the original StudySpec SHA-256.
- [Place-field checkpoint trajectories](results/odor_v2_spatial_20261007/place_field_checkpoint_trajectory.png) and [graph reachability curves](results/odor_v2_spatial_20261007/graph_reachability.png) show the key comparisons. Representative seed-99 75M [ON/HEBB4 offline field maps](results/odor_v2_spatial_20261007/offline_maps/ODOR_ON_HEBB4_place_fields.png), [pre-threshold maps](results/odor_v2_spatial_20261007/offline_maps/ODOR_ON_HEBB4_pre_threshold_logits.png), [online occupancy/trajectory](results/odor_v2_spatial_20261007/atlases/ODOR_ON_HEBB4_S99_V2/target_000075000000_policy_00_trajectory.png), and [directed graph outcome map](results/odor_v2_spatial_20261007/graphs/ODOR_ON_HEBB4_S99_V2/target_000075000000_policy_00_graph_outcomes.png) are provided. Matched [ON/RANDOM4 fields](results/odor_v2_spatial_20261007/offline_maps/ODOR_ON_RANDOM4_place_fields.png) and [graph outcomes](results/odor_v2_spatial_20261007/graphs/ODOR_ON_RANDOM4_S99_V2/target_000075000000_policy_00_graph_outcomes.png), plus OFF/HEBB4 and OFF/RANDOM4 atlases, are in the same result directory.

Bulk artifacts remain in the allocated NEMO2 workspace. The canonical full manifest is `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/odor_ca3_goal_quality_v2_full_20261007/`. Its `complete_10k/raw/` links the 30 original and 40 later NPZs by exact manifest label; `complete_10k/source_mapping.json` records each original. Standard summaries and all 70 field/pre-threshold plot sets are under `complete_10k/summary/`. The 200 snapshots and complete graph-edge table are summarized under `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/odor_ca3_goal_quality_v2_spatial_20261007/`. These paths contain data and plots, never training outputs in the home filesystem.

## Place fields

The manifest-driven evaluator found no silent DG unit in any of the 70 rollouts. At 75M, the 30 rollouts for seeds 8, 99, and 123 had a mean of 15.4 distinct active peak bins across 16 units. That peak count alone overstates clean place-field structure: only 5.2% of eligible unit maps met the established mono-field criterion, and the mean unit map had 4.54 connected components at the 50% peak threshold. The selected offline and online contact sheets show multiple separated hotspots in many units. Their gray bins were unvisited; the plotted thresholded maps are normalized per unit for shape comparison, so their color does not compare response amplitude between units.

| Condition | Terminal visited bins | Active-only map cosine | Distinct active peak bins | Mono-field fraction |
| --- | ---: | ---: | ---: | ---: |
| OFF/ALL16 | 256 | 0.110 | 15.7 | 0.021 |
| OFF/RANDOM4 | 236 | 0.080 | 15.7 | 0.042 |
| OFF/HEBB4 | 164 | 0.057 | 15.7 | 0.104 |
| ON/ALL16 | 214 | 0.098 | 14.7 | 0.042 |
| ON/RANDOM4 | 240 | 0.077 | 15.3 | 0.042 |
| ON/HEBB4 | 180 | 0.064 | 15.7 | 0.021 |

Values are means of three terminal rollout seeds. The CSV includes the remaining RANDOM8 and HEBB8 conditions, spatial information, silent units, and pre-threshold measures. HEBB4's lower active-map cosine accompanies fewer visited bins; it is not enough to conclude that its DG representation is better. The canonical spatial-information score is amplitude-weighted, and policy occupancy differs across conditions.

## Trajectories and map comparisons

The [five-checkpoint plot](results/odor_v2_spatial_20261007/place_field_checkpoint_trajectory.png) uses one 10k-decision seed-99 policy rollout per condition and checkpoint. HEBB4 visited fewer spatial bins than RANDOM4 by 75M in both odor settings; the three-seed terminal means above show the same direction. The selected online atlas shows occupancy and 100k retained behavior observations as independent trajectory segments, with starts, ends, and heading arrows. It is a separate observation window from the offline 10k rollout.

The standard [map-stability summaries](results/odor_v2_spatial_20261007/stability/) compare each seed-99 checkpoint map with its 75M map over jointly visited cells. Mean correlation across conditions rose from 0.117 at 5M to 0.571 at 50M. Each checkpoint follows its own learned policy and visits different cells, so this is a descriptive cross-checkpoint comparison, **not** a fixed-trajectory representational-drift test.

## Policy graphs

The canonical collector validated graph buffers and recomputed graph diagnostics for all 200 snapshots. At 75M, HEBB4 had less reliable directed-graph connectivity than RANDOM4 in all four paired seeds under both odor settings:

| Odor | Selector | Reliable directed edges | Reachable ordered-pair fraction | Prospective attempts | Effective attempted edges |
| --- | --- | ---: | ---: | ---: | ---: |
| OFF | RANDOM4 | 79.5 | 1.000 | 48,541 | 148.2 |
| OFF | HEBB4 | 42.5 | 0.531 | 78,400 | 60.2 |
| ON | RANDOM4 | 83.5 | 0.984 | 46,660 | 133.6 |
| ON | HEBB4 | 43.0 | 0.540 | 82,794 | 65.0 |

Means are across four paired seeds. Effective attempted edges are $\exp[-\sum_e p_e\log p_e]$, with $p_e$ the fraction of prospective attempts on directed edge $e$; the maximum possible attempted-edge count is 240. The paired HEBB4 minus RANDOM4 difference in reachable fraction was -0.469 OFF and -0.445 ON, negative in all eight paired comparisons. The corresponding differences in reliable-edge count were -37.0 and -40.5. HEBB4 made **more** total prospective attempts but concentrated them on fewer directed pairs. Its conditional prospective hit fraction was higher on those attempted edges; that does not establish better control of arbitrary requested goals.

The [reachability curves](results/odor_v2_spatial_20261007/graph_reachability.png) show HEBB4's separation by 25M and partial recovery by 75M; HEBB8 has an intermediate pattern. In the selected outcome heatmaps, rows are source DG units, columns are target units, color is prospective hits divided by attempts on a fixed 0–1 scale, and gray means unattempted. Gray is missing edge evidence, not a failed edge. All 16 DG identities remained available and supported; limiting candidates at manager choices nevertheless appears to concentrate the experienced transition graph. This is a mechanism consistent with HEBB4's lower visited-bin count and coverage, not proof of why any individual commanded trial succeeds or fails.

## What remains

The 40 target-hit and 40 first-distinct evaluations are complete; see the [command analysis](odor_goal_quality_v2_command_interventions_20261007.md). Eventual requested-goal hit within the deadline is the primary command outcome, and HEBB4 has higher ordinary target-hit rates than RANDOM4 in both odor settings under longer mean deadlines. Commanded first identity occurred in 6.75% of first-distinct trials, versus 6.57% for a context-matched shuffled label on the same path; this secondary diagnostic does not rule out later goal hits. The shuffled label is retrospective, so neither protocol measures the effect of executing an alternate command. The [coverage analysis](odor_goal_quality_40_run_v2_20261007.md#interim-analysis-complete-training-histories) reports the 40-run external learning curves and seed-paired contrasts; those results show no consistent coverage benefit. Exact-start replay is being requalified in bounded four-source jobs after the full-panel timeout. Online graph hit fractions are not intervention success rates.

## Reproducibility note

The canonical `collect-spatial --require-complete --include-details` command consumed the immutable study and the 200 snapshots. The established `analyze_place_field_manifest.py`, `summarize_place_fields.py`, and per-condition `map_stability.py` consumed the 70 manifest-linked offline NPZs. [Graph rendering](render_odor_v2_graphs.py), [readable selected offline maps](render_odor_v2_offline_maps.py), and [comparative figures](analyze_odor_v2_spatial.py) are thin adapters around those standard artifacts; the last computes effective edge exposure from the canonical graph-edge CSV. Rendering all 70 old-style contact sheets was quick but produced labels too small at report width, so the selected offline maps were rerendered with larger scalable text. For future studies, collect canonical tables once, keep bulk graph-edge and NPZ data in the workspace, and copy only the summary tables and selected inspected figures into the vault.
