# IntrMotiv Metrics Guidebook

**Updated:** 2026-09-06  
**Current study:** Saturday batch, workflow `1.4.1`  
**Companion reference:** [[IntrMotiv_metric_reference|complete tag dictionary]]

This guide is for deciding what a run means. It deliberately separates
scientific outcomes from implementation checks and intermediate mechanism
diagnostics. No single metric establishes navigation or controllability.

## The evidence hierarchy

Read results from the top down:

1. **Frozen intervention:** does commanding another target causally change
   success and action selection from matched source/context states?
2. **Spatial landmark quality:** does each DG identity describe a localized,
   mostly unimodal physical field rather than a distributed visual class?
3. **Reliable graph quality:** are useful directed edges distributed across
   nodes, empirically successful, and capable of supporting routes?
4. **Online worker behavior:** are target-conditioned options succeeding,
   without looping or merely exploiting common sink nodes?
5. **Learning mechanism:** are encoder credit, retirement, replay, and goal
   conditioning exercising the intended branches?
6. **Numerical health:** are losses finite and training advancing?

A lower layer can explain why a higher layer failed. It cannot replace the
higher-layer evidence. In particular, coverage, a dense graph, or different
action logits do not by themselves prove control.

## Five-minute run triage

| Question | Read first | Healthy evidence | Red flag |
|---|---|---|---|
| Is it running correctly? | `train/env_steps`, losses, replay/publication mismatch | Frames advance; finite losses; mismatches exactly zero | Stalled frames, NaN/Inf, any persistent mismatch |
| Is DG alive and balanced? | density, silent fraction, usage entropy, duty-cycle max | Nonzero sparse activity; nearly all rows participate; entropy near one | Many silent rows or a few dominant rows |
| Are landmarks spatially meaningful? | spatial information, mono-field fraction, component counts, map cosine | Higher information and mono-field mass; fewer components; lower cross-unit cosine | “High mono-field” accompanied by low information and many silent units |
| Is the graph useful? | outgoing-node coverage, SCC, reachable pairs, top-three incoming share | Broad outgoing support, large SCC, distributed incoming confidence | Many edges funnel into a few destinations |
| Does the target matter? | frozen commanded-vs-shuffled advantage, action-probability TV | Commanded success beats matched shuffle and actions change by target | Lift only, logit change only, or commanded and shuffled success are equal |
| Is retirement active and safe? | eligibility-to-conversion funnel, totals, repeats, generation drops/defer | Eligible victims convert; replacement is followed by a deferred stale batch and recovery | No opportunity, repeated churn, mismatched FiLM reset, post-replacement NaN |

## Metric-name grammar

The suffix usually identifies the denominator:

| Suffix | Meaning |
|---|---|
| `_rate` | Events divided by valid learner transitions. Usually **not** a per-option probability. |
| `_fraction` | A named subset divided by its stated population. Check the population. |
| `_mean` | Mean over the stated events or entries; it may be zero when no such event exists. |
| `_per_rollout` | Count in the latest accepted actor rollout. Not cumulative. |
| `_total` | Cumulative counter. For StudySpec analysis, take the last value rather than a window mean. |
| `_count` | Context-dependent. It may describe the current minibatch, current graph, or a cumulative buffer. |

W&B smoothing changes only the display. Most `intrmotiv/` online values are
learner-minibatch summaries, not physical episode statistics.

## 1. Hard implementation invariants

These are pass/fail checks, not scientific outcomes.

| Metric | Required interpretation |
|---|---|
| `intrmotiv/hrl/behavior_replay_mismatch` | Must be zero. The learner must replay the behavior target and manager condition exactly. |
| `intrmotiv/dg/update_contract/publication_generation_mismatch` | Must be zero. Published DG weights and normalization state must describe one representation generation. |
| `intrmotiv/dg/update_contract/forward_count` | One DG forward per active minibatch under the corrected contract. |
| `intrmotiv/dg/update_contract/running_stats_update_count` | One in encoder-active/simultaneous legacy-BN updates; zero in decoder-only phases and actor inference. |
| `intrmotiv/replay/stale_generation_rejected_fraction` | Can spike after replacement. It is safe only when whole old rollouts are removed before learning. |
| `intrmotiv/replay/deferred_updates_total` | Should increment when replacement leaves less than one normal fresh minibatch. A defer is protective, not a failure. |
| `intrmotiv/dg/recruitment/goal_adapter_reset_total` | In FiLM retirement cells, must match the replacement total. It remains zero for legacy conditioning. |
| `intrmotiv/reward/environment_nonzero_fraction` | Must be zero in the no-reward open field. |

The old retirement crash had a characteristic signature: one replacement,
roughly 80--90% stale samples, no deferral, then NaN advantage normalization.
The corrected path drops complete old-generation rollouts and defers that
learner update.

## 2. Representation health

### Online DG activity

| Metric | What it answers | Read with |
|---|---|---|
| `intrmotiv/dg/density` | How often post-threshold DG entries are active | active count, silent fraction |
| `intrmotiv/dg/silent_unit_fraction` | How many rows were unused in this minibatch | place-field active-unit fraction |
| `intrmotiv/dg/usage_entropy` | How evenly activation mass is distributed across rows | duty-cycle maximum, top-three activity share |
| `intrmotiv/dg/unit_duty_cycle_max` | Whether one row dominates the batch | usage entropy |
| `intrmotiv/dg/multi_activation_fraction` | How often an observation activates multiple identities | behavior multi-onset fraction |
| `intrmotiv/dg/pre_threshold_above_fraction` | Smooth activity around the hard threshold | density and BN semantics |

There is no universal ideal density. The useful regime has sparse events but
enough events to populate transitions, with nearly all 16 rows participating.
For the restored 5M C15 baseline, density around `0.036`, zero silent rows,
and usage entropy around `0.994` formed a healthy reference. These are
diagnostic anchors, not acceptance thresholds for all domains.

### Place-field quality

The authoritative spatial claim comes from the fixed-checkpoint 10k-decision
manifest evaluation. Online fields are early warnings.

| Metric | Preferred direction | Important qualification |
|---|---|---|
| `active_unit_mean_spatial_information` | Higher | Average only over active units; report active count. |
| `mono_field_unit_fraction` | Higher | A dead or tiny field can look unimodal; require coverage and information. |
| `mean_components_{30,50,70}pct` | Lower, toward one | Report all thresholds; a single threshold is fragile. |
| `mean_dominant_component_mass` | Higher | Shows whether one field component dominates scattered fragments. |
| `active_only_map_cosine` | Lower | High cosine means different DG rows fire in similar places. |
| `unique_active_peak_bins` | Higher, up to active rows | Low diversity exposes co-located landmark identities. |
| `median_dominant_peak_nearest_neighbor_distance` | Neither extreme by itself | Very small means duplicates; very large may reflect missing fields. |
| `incoming_confidence_field_spread_correlation` | Near zero or negative is preferable | Positive values mean graph sinks tend to be spatially scattered. |

Never call a representation improved from mono-field fraction alone. In the
failed running-stat batch, mono-field fraction sometimes rose while spatial
information fell roughly tenfold and the graph collapsed.

## 3. Graph structure

A directed edge `j -> k` is reliable exactly when:

```text
Tctrl[j,k] > 0
and edge_confidence[j,k] >= 0.5
and (edge_confidence[j,k] + 1) / (control_attempts[j,k] + 2) >= 0.5
```

The last term is a Beta(1,1)-smoothed empirical success estimate. Direction is
never inferred from the reverse edge.

| Metric | What it measures | Desired pattern |
|---|---|---|
| `intrmotiv/hrl/reliable/outgoing_node_fraction` | Fraction of DG nodes with at least one reliable outgoing edge | Broad source coverage |
| `intrmotiv/hrl/reliable/largest_scc` | Largest mutually reachable directed component | Large enough to support alternate routes |
| `intrmotiv/hrl/reliable/reachable_pair_fraction` | Ordered node pairs connected by some reliable path | High, with grounded endpoints |
| `intrmotiv/hrl/reliable/reciprocal_fraction` | Reliable edges whose reverse is also reliable | Useful context, not a universal objective |
| `intrmotiv/hrl/reliable/top3_incoming_confidence_share` | Incoming confidence captured by the three strongest destinations | Low; high values reveal funnel/sink domination |
| `intrmotiv/hrl/summary/reliable_global_efficiency` | Mean reciprocal directed hop distance; unreachable pairs contribute zero | High only after checking field quality |

Saturday's graph gate requires at least 12/16 nodes with reliable outgoing
connectivity, an SCC of at least 8/16 in two seeds, top-three incoming share at
most 60%, and at least 50% empirical success on routed edges. These criteria
belong to this experiment and should not silently become universal constants.

### Grounded controllability

```text
grounded_controllability
  = prospective_success_fraction
  * spatial_endpoint_valid_fraction
```

`prospective_success_fraction` uses later completed intentional attempts on
edges that were already reliable before the learner batch. A hit is success
and a target timeout is failure. `spatial_endpoint_valid_fraction` is the
fraction of current reliable edges whose source and destination are both
eligible mono-field units.

This avoids using the same hit both to create an edge and validate it, but it
is still a product of two aggregate fractions. It is not a per-edge causal
intervention score and cannot replace commanded-versus-shuffled evaluation.

## 4. Target-conditioned control

Use this order of evidence:

1. **Frozen intervention advantage.** From matched source position,
   orientation, and context, compare the commanded target with shuffled
   targets. Aggregate hit numerators and event counts before forming rates.
2. **Counterfactual action-probability sensitivity.** In the frozen evaluator,
   hold source/context fixed, change only target ID, and compare action
   distributions across all alternate targets.
3. **Online completed-option success.** Useful for learning curves, but it can
   be inflated by common sinks and unequal target sampling.
4. **Logit sensitivity.** A debugging signal only; logit differences need not
   produce meaningful probability or trajectory changes.

| Metric | Definition or role |
|---|---|
| `intrmotiv/hrl/target_hit_numerator` / `target_hit_event_count` | Components for commanded hit rate. Sum first, then divide. |
| `intrmotiv/hrl/shuffled_hit_numerator` / `shuffled_hit_event_count` | Matched-shuffle components. Sum first, then divide. |
| `intrmotiv/hrl/goal_condition/action_probability_tv` | Online proxy: mean total variation between the behavior-target action distribution and a one-position rolled target in the same minibatch. |
| `intrmotiv/hrl/goal_condition/action_sensitivity` | Online proxy: mean absolute raw-logit change under that rolled target. Diagnostic only. |
| `intrmotiv/hrl/option_success_fraction` | Hits divided by hits plus target timeouts. This is the per-completed-option statistic. |
| `intrmotiv/hrl/target_hit_rate` | Hits divided by valid transitions. This is an event frequency, not option success. |
| `intrmotiv/hrl/target_hit_lift` | Commanded activation rate divided by a shifted-target rate. Unstable when the denominator is near zero. |

The online rolled-target metrics are cheap trend diagnostics, not the full
15-alternative counterfactual evaluation. The minimum credible statement is
“the commanded target beats the matched shuffle.” High option success without
that advantage may mean the policy reaches common landmarks regardless of the
command.

## 5. Retirement

Retirement has three separate questions:

1. **Was there an opportunity?** A valid `L=64` endpoint occurred.
2. **Was there an eligible victim?** DIR or PRED identified a row, and the
   residual initializer passed.
3. **Was replacement committed and recovered from safely?** Graph and policy
   state were invalidated atomically and learning resumed on fresh rollouts.

Read the funnel rather than only `recruitment/total`:

```text
candidate endpoint
  -> eligible victim endpoint
  -> residual pass
  -> replacement conversion
  -> cumulative replacement
```

| Metric | Interpretation |
|---|---|
| `candidate_count` | Valid L-endpoint opportunities in the accepted rollout |
| `active_endpoint_count` / `silent_endpoint_count` | Whether another DG unit covered the endpoint |
| `activity_blocked_count` | Eligible open-class opportunity rejected only by the legacy silent gate |
| `eligible_victim_endpoint_count` | Endpoint where a victim rule actually selected a row |
| `residual_pass_count` | Selected victim had a usable orthogonal residual initializer |
| `replacement_conversion` | Fraction/count reaching assignment, subject to the one-per-rollout cap |
| `recruitment/total` | Cumulative committed replacements |
| `repeat_total` | Replacements of rows that had already been replaced; representation churn |

### DIR diagnostics

DIR asks whether a landmark is a usable **source** in the controllability
graph. A node is a bad source only after all 15 alternative targets have at
least `0.5` decayed attempt evidence and it still has zero reliable outgoing
edges. Incoming edges do not protect it. DIR also handles mutual reliable
duplicates with `Tctrl <= 4` in both directions.

Read `attempt_coverage_fraction`, `fully_tested_count`,
`untested_zero_outdegree_count`, `bad_source_count`, and
`reliable_out_degree_mean` together. A zero-outdegree but untested node is
protected; that is missing evidence, not proof of a bad landmark.

### PRED diagnostics

PRED asks whether one source identity hides incompatible predecessor
contexts. It requires the same `(source, goal)` under at least two distinct
predecessor contexts, at least `4.0` decayed attempts in each, and a reliability
split across `0.5`.

Read context coverage and attempt mass before the reliability gap. A large gap
with little support is not eligible evidence.

## 6. Encoder credit

For both ARR and SRC, behavior-time matching first constructs the same event:
a dominant destination onset, its nearest distinct non-simultaneous CA3
predecessor, and distance `d`. Events crossing a rollout boundary or failing
behavior-label alignment are dropped from both treatments.

| Metric group | Question |
|---|---|
| `total_events -> matchable_events -> credited_events` | Where does source alignment lose events? |
| `boundary_dropped_events`, `alignment_failures`, `invalid_intervals` | Why were events rejected? |
| `scheduled_count/mass` | How much behavior-time credit was requested? |
| `applied_count/mass` | How much remained active when DG was recomputed by the learner? |
| `replay_match` | Applied divided by scheduled credit under the current activity intersection |
| `arrival_loss`, `source_loss` | Which recipient branch received the encoder loss? |

ARR and SRC must have identical matched event counts and reward mass in a
matched construction test. During stochastic training their trajectories can
diverge, so do not expect their later online event totals to remain identical.

`replay_match < 1` is not automatically a bug: the behavior-credited row can
recompute below threshold and is intentionally assigned zero gradient. A low
or strongly condition-dependent value does reduce effective encoder
supervision and must be reported.

## 7. Exploration and trajectory behavior

| Metric | What it can establish |
|---|---|
| coverage unique cells / coverage AUC | Physical extent and how early it was reached |
| coverage entropy | Evenness of occupancy, only meaningful with extent |
| pose AUC / pose entropy | Joint position-orientation coverage |
| path efficiency | Endpoint displacement divided by traveled length |
| straightness | Net command displacement divided by command-path length |
| stationary-step fraction | Time spent moving no more than one integrated unit |
| mean absolute circular yaw change | Turning intensity with wraparound handled correctly |

Coverage is not control. A circular or random policy can cover the arena while
ignoring targets. Compare trajectory metrics by manager mode and pair them
with intervention success.

## 8. Optimization metrics

Use Sample Factory losses to detect instability, not to rank algorithms.

- `train/policy_loss`, `train/value_loss`, and `train/entropy` describe the
  optimizer, critic, and policy distribution.
- `intrmotiv/encoder/loss` and `intrmotiv/decoder/loss` verify loss routing.
- `intrmotiv/update/phase` is `0` simultaneous, `1` decoder-only, and `2`
  DG-only under iterative schedules.
- A lower loss is not necessarily a better representation or controller.
- Finite but flat losses do not prove that the relevant event branch has any
  samples; always read event counts and reward nonzero fraction.

## Saturday batch scorecard

Analyze the current batch in this order:

### A. Common baseline

Use the four MON cells first. Reject scientific interpretation if any has
non-finite learning, replay mismatch, representation collapse, or unsupported
target comparisons.

### B. Source versus arrival credit

Compare `SRC - ARR` within MON, separately for LEG and FiLM. Prefer SRC only
if spatial identity improves in at least two seeds and in the three-seed mean,
while retaining at least 90% of ARR coverage and causal target advantage.

### C. Retirement

Compare DIRO and PREDO with their exactly matched MON cells. First require the
retirement funnel to show that replacement actually occurred. Then require
better spatial identity without excessive repeats, loss of more than 20% of
MON reachability, or weaker intervention advantage.

### D. Goal conditioning

Compare FILM with LEG within every credit-retirement cell. Use action-
probability TV and frozen commanded-versus-shuffled success. Do not select
FiLM from modulation norm or raw-logit sensitivity alone.

### E. Final claim language

| Evidence obtained | Maximum defensible claim |
|---|---|
| Healthy DG only | Sparse landmark identities are maintained |
| Localized fields plus healthy DG | Spatially grounded landmark identities emerge |
| Useful reliable graph plus localized fields | The identities support a navigational graph |
| Commanded target beats matched shuffle | Target-dependent event control is present |
| Above plus efficient physical trajectories | Spatial navigation is demonstrated in this environment |

Transfer to web or abstract-graph domains requires replacing physical
place-field grounding with an appropriate domain-level state-consistency test;
the causal intervention and directed reachability logic still transfer.

## Source-of-truth map

- Exact custom tag definitions: [[IntrMotiv_metric_reference]]
- Online and fixed-checkpoint spatial protocol:
  [[reusable_place_field_telemetry]]
- Standard collection, contrasts, and StudySpec rules:
  [[standardized_study_workflow]]
- Current batch design and acceptance gates: [[../05_plans/saturday_batch]]
- Current declarative metric set:
  `hpc_runs/studies/saturday_batch.study.json`

When this guide and code disagree, code is authoritative. Update both the
complete reference and this guide whenever a metric's numerator, denominator,
window, or reset behavior changes.
