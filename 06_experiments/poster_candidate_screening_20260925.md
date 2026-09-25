# Broader poster-candidate screening: evidence refresh, 25 September 2026

This report refreshes the evidence pool from existing checkpoint inventories,
canonical spatial outputs, and the experiment reports linked below. It records
what can be compared at matched ages and what remains unavailable. It does not
select figures, rank architectures, or update the poster synthesis.

## Matched checkpoint inventory

| Study family | Primary matched checkpoint and coverage | Later artifacts | Comparison status and source |
|---|---|---|---|
| DG capacity / goal conditioning | 25M; 27/27 runs (three seeds × nine cells) | 75M: 18/18 direct-family runs; no WAYPOINT_DG F64 75M snapshots | Full 25M matrix; 75M is a restricted direct-only subset; selected F16 S99 is a longitudinal run comparison. [Detailed report and canonical tables](dg_capacity_goal_conditioning_interim_20260911.md#poster-exemplar-evidence-update-25-september-2026) |
| Historical Navigation8 SCR/SAT/DGP/CPD/reference families | 75M; 18/18 runs across six configurations and three seeds | No 150M/300M snapshots in the analyzed inventory | Full matched 75M screen. [Report](navigation8_algorithm_screen_interim_20260916.md), [snapshot inventory](data/navigation8_algorithm_screen_interim_20260916/online_spatial/snapshot_inventory.csv), [all-seed atlas](data/navigation8_algorithm_screen_interim_20260916/canonical_panels/all_seed_75m/) |
| Selected historical DGP/SCR/SAT terminal analyses | Declared 65–75M TensorBoard window for the selected DGP comparisons | Separate 75M, 100k-sample spatial snapshots; these are not the same measurement as the TensorBoard window | Retain each protocol as its own result; do not combine option success, snapshot graph values, or terminal records. [High-option report](06_high_option_success_goal_sets_and_controls_20260914.md), [late-outlier report](late_training_outliers_20260908.md) |
| CPU2048 architecture/cadence study | 25M; 24/24 runs (eight conditions × three seeds) | 75M: 12/12 cadence-2048 runs; cadence-64 only 3/12. 150M: 5/24 | Full 25M comparison; 75M only a complete cadence-2048 subset, not a cadence factorial. [Report and atlas](cpu2048_analysis_20260917.md), [run/snapshot inventory](data/cpu2048_analysis_20260917/run_status.json) |
| CA3 predictive active goals | 75M; 21/21 runs (seven cells × three seeds) | 150M: 9/21 | Full matched 75M matrix; 150M excluded from the complete architecture comparison. [Report](ca3_predictive_active_goals_interim_analysis_20260924.md), [75M canonical atlas](results/recent_architecture_batches_20260924/predictive/) |
| CA3 state-goal follow-up | 25M; 12/12 runs (four cells × three seeds) | 75M: 10/12; both missing rows are fixed-anchor seed 8 | Full matched 25M factorial; later data do not complete the full factorial. [Report](ca3_state_goal_followup_interim_analysis_20260924.md), [25M atlas](results/recent_architecture_batches_20260924/followup/) |
| Easy-landmark maze qualification | 2M; six matched rows (three architectures × rich/neutral, seed 99) | Production: 12 runs launched, but no mature production checkpoint is represented in the analyzed bundle | Qualification only, not replicated production. Frozen coverage used two matched reset seeds and two policy episodes per row. [Qualification analysis](easy_landmark_maze_qualification_analysis_20260924.md), [production status](easy_landmark_maze_implementation_20260923.md) |
| Five-cue reward transfer | No mature matched comparison in this bundle | 48-run, three-seed production planned to 75M; qualification evidence is 262,144 frames | Do not treat launch or short qualification as a transfer result. [Campaign record](cued_reward5_transfer_20260925.md) |

The workflow-generated `per_snapshot.csv`, `snapshot_inventory.csv`, figure
atlases, and linked study reports are the analysis sources. No run was moved to
its own latest checkpoint for a cross-condition comparison. When checkpoint
availability supports only a subset, it is labeled as such above.

## Factual side-by-side observations

### Representation and graph can separate

| Example and checkpoint | Representation evidence | Graph evidence | Reading boundary |
|---|---|---|---|
| WAYPOINT_DG F64 S8, 25M | Active-only cosine 0.083; 45.3% mono-field; 32 peak bins | 21 reliable edges; 0.62% reachable pairs | Low overlap and many mono-field-qualified units coexist with a sparse graph; repeated dominant peaks also remain. |
| DGC_DIRECT_WORKER F16 S99, 25M | Cosine 0.150; 66.7% mono-field; only 9 peak bins for 16 active units, with exact peak clustering | 77 reliable edges; all allocated-node pairs reachable; 61.3% prospective success | This is a checkpoint-specific transient local-control candidate; its three screened edges and their raw counts are in the DGC report. By 75M, mono-field fraction is 6.25%, reachable pairs 81.25%, and grounded proxy 0. |
| DGP HIT–JOINT–LEG S123, 75M | Five of 16 eligible units classified mono-field; cosine 0.107; the five fields cluster in one corner | Option success 55.5% in the 65–75M window; target/shuffle hit lift 0.9973 | Specialized local fields and high eventual HIT completion do not show command-specific destinations. This is an individual seed observation within a three-seed terminal screen. |
| CA3 ZGOAL_FIXED_H16, 75M | Three-seed mean cosine 0.227, mono-field 12.0%, 46.7 distinct peak bins | 43.3 reliable edges; 3.1% reachable pairs; grounded control 0 | Better populated graph than the matched contextual H16 cell, but no grounded command control. |
| CA3 CTX_FULL_H16, 75M | Three-seed mean cosine 0.259, mono-field 1.7%, 46 distinct peak bins | 2.0 reliable edges; 0.1% reachable pairs; grounded control 0 | Similar peak-bin count does not prevent a collapse in reliable graph evidence; the selected matched scalar scan was incomplete. |
| CA3 state-goal FIXED/UNIQUE, 25M | Mono-field rises from 4.7% to 10.2% in three-seed means | Reliable edges fall from 23.0 to 0.3; reachable pairs are 0.7% versus 0 | This is a consistent representation/graph trade-off association for the declared candidate rule, not a control benefit. |
| Easy landmark rich Waypoint, 2M qualification | Cosine 0.270; mono-field 26.6%; 31 peak bins | 30 reliable edges; 0.8% reachable pairs | One-seed early qualification evidence only; no command-control result. |

The preceding table combines different tasks only to summarize observations;
its rows are not estimates of one pooled architecture effect.

### Exploration evidence depends on protocol

- In the matched Navigation8 75M online snapshots, all three seeds are present
  for all six configurations. SAT has mean map cosine 0.209; DGP has full
  ordered-pair reachability and graph efficiency near 0.87. These are online
  policy visitation and passive graph summaries, not frozen command outcomes.
- In the easy-landmark qualification, rich Waypoint has frozen accessible-
  coverage AUC 0.577 versus 0.349 for its neutral control and 0.386 for the
  uniform-random baseline. This is two evaluation episodes per policy for one
  seed; the rich-minus-neutral effect differs in direction for DGP.
- At 25M, all 24 CPU2048 runs visited 318/361 pooled grid cells, but that
  saturated pooled fraction does not compare per-run or per-episode exploration.
  In the complete cadence-2048 subset at 75M, all four architecture/learner
  cells have three seeds; cadence-64 lacks matched coverage for the same
  factorial comparison.
- No trajectory-coverage AUC is present for the DGC and CA3 online snapshot
  tables. Their saved segmented trajectory atlases remain useful descriptive
  artifacts; the online visit pattern is not a frozen-policy evaluation.

## Node control and prospective evidence

No exact-start alternative-command result was found in the analyzed local
artifacts for the requested DGC F16/F64 or DGP candidates, CPU2048 Direct F16
DDQN/DDQN+HER, Navigation8, or the new CA3 matrices. Consequently there is no
measured action-probability TV, first-reached-internal-node confusion matrix,
timeout rate, or wrong-node rate for these targets. The strongest available
passive DGC edge counts are not causal evidence. The high DGP HIT completion
and near-unit target/shuffle lift specifically leave command dependence
unresolved. The DGP report's activation lift and FIRST control are not a
same-checkpoint, same-start counterfactual for that HIT checkpoint.

The current records therefore contain no demonstrated strongest node-control
example. Grounded-controllability summaries are zero for all 24 CPU2048 runs
at 25M, every row of the matched CA3 predictive 75M comparison, and all four
CA3 state-goal cells at 25M. The selected DGC F64 waypoint seeds have small
nonzero proxy values (0.065 and 0.027), while their internal graphs remain
sparse. DGC F16 S99 has a nonzero passive proxy at 25M (0.318), which falls to
zero at 75M; this remains a screening signal, not matched-start intervention
evidence.

For CPU2048 at the complete matched 75M cadence-2048 subset, Direct F16 / DDQN
and Direct F16 / DDQN+HER have 91.7% and 93.8% reachable-pair fractions,
respectively, but both have zero mono-field fraction and no grounded-control
score. The report gives the full seed-level measurements and atlas; no exact
start command intervention is present.

## Representation stability and CA3 diagnostics

- DGC F16 S99 already has a longitudinal policy-window summary at 25M and 75M;
  it shows a mono-field and grounded-proxy drop alongside higher cosine.
  Cross-checkpoint field-map correlation, per-field peak displacement, and
  edge-set persistence were not calculated. The samples are policy-driven, so
  this is not a fixed-trajectory stability test.
- CA3 predictive has complete 5M/25M/75M spatial inventories; state-goal has
  complete 5M/25M and only 10/12 at 75M. Neither report computes unitwise
  cross-checkpoint map correlation or peak displacement.
- The requested CA3 prediction-loss/shuffle, latent effective-rank, positive /
  background similarity-quantile, and production contextual accepted/zero/
  multi-match / HER error metrics are not available as a validated matched
  comparison. The CA3 reports document incomplete/stale TensorBoard history
  discovery; qualification and offline alias diagnostics exist separately but
  are not a substitute for the production matched-age metrics.
- The state-goal factorial specifically lacks 75M fixed-anchor seed-8 rows, so
  no full-factorial 75M main effect is reported.

## Transfer and missing analyses

Five-cue transfer remains in the learning phase. Existing records establish
that the source and random DG controls passed short runtime qualifications, but
the paired 75M per-cue success, reward AUC, time-to-reward, and frozen-DG
place-field/exploration comparison are not yet available. No mature transfer
conclusion is made.

Further missing or inconclusive evidence is: same-start goal interventions for
the selected exemplars; checkpoint-to-checkpoint map and edge persistence;
complete matched 150M/300M panels where current inventories are partial;
validated full production CA3 predictive/context scalar comparisons; mature
easy-landmark production; and five-cue transfer outcomes. Historical 65–75M
TensorBoard outcomes remain distinct from the 75M spatial snapshots.

## Detailed reports and artifacts

- [DG capacity / goal conditioning and selected exemplar edge counts](dg_capacity_goal_conditioning_interim_20260911.md)
- [Poster exemplar batch detail from the prior pass](poster_exemplar_analysis_20260925.md)
- [Historical high-option DGP controls](06_high_option_success_goal_sets_and_controls_20260914.md)
- [Navigation8 75M matched atlas](navigation8_algorithm_screen_interim_20260916.md)
- [CPU2048 matched spatial, movement, graph, and inventory](cpu2048_analysis_20260917.md)
- [CA3 predictive active-goal 75M comparison](ca3_predictive_active_goals_interim_analysis_20260924.md)
- [CA3 state-goal matched 25M comparison](ca3_state_goal_followup_interim_analysis_20260924.md)
- [Easy-landmark 2M qualification and frozen coverage evaluation](easy_landmark_maze_qualification_analysis_20260924.md)
- [Five-cue transfer launch and qualification status](cued_reward5_transfer_20260925.md)

## Reusable workflow note

The reliable shortcut for this pass was to use each study's existing canonical
`per_snapshot.csv`, `snapshot_inventory.csv`, `analysis_manifest.json`, and
`segmented-atlas/v1` figures, then apply the common-checkpoint rule before
summarizing. Two limits remain recurring: snapshot success does not ensure a
complete scalar history, and passive graph metrics do not substitute for
matched-start interventions. No new analysis code or duplicate run inventory
was introduced.
