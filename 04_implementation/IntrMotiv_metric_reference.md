# IntrMotiv Metric Reference

**Scope:** current custom intrmotiv/ learner and environment metrics written to TensorBoard and W&B. This supplements the standard Sample Factory train/ metrics.

**Authoritative implementation:** NEMO2 SF_hipposlam, in dmlab/custom_learner.py (calculation), dmlab/reward_summaries.py (tag mapping), and evaluation/schema.py (current and legacy analysis tags). The NEMO2 IntrMotiv LOGGING.md remains a concise overview; this note is the complete interpretation reference.

## Scope And Denominators

Except environment telemetry, every custom value is computed on valid transitions in the learner's current PPO minibatch. It is not a whole-run total or a physical DMLab episode statistic. W&B plots it at the learner environment-frame count; visual smoothing does not change the definition.

Event rates have a transition denominator:

    target_hit_rate         = hits / valid transitions
    option_timeout_rate     = timeouts / valid transitions
    option_success_fraction = hits / (hits + timeouts)

Hits and timeouts are mutually exclusive but most transitions are neither. A hit rate of 0.005 and timeout rate of 0.100 mean roughly 5 hits and 100 timeouts per 1,000 valid transitions, hence an option success fraction near 0.005 / 0.105 = 4.76%. The remaining 895 transitions are normally in-progress options. option_reset_rate should be near the sum of these rates, plus any reset caused by an empty target.

## Sample Factory Metrics

These standard train/ metrics are framework diagnostics, rather than IntrMotiv outcome metrics. Exact availability varies with Sample Factory version and enabled features.

| Tag | Meaning | Use |
| --- | --- | --- |
| train/env_steps | Frames seen by this policy. | Primary horizontal axis. Do not add independent no-PBT jobs. |
| train/fps, train/avg_fps | Recent/average throughput. | Diagnose CPU, DMLab, and learner bottlenecks. |
| train/policy_loss | PPO clipped policy-surrogate loss. | Optimizer diagnostic, not exploration performance. |
| train/value_loss | Critic regression loss. | Persistent explosions suggest critic or reward-scale instability. |
| train/entropy | Action-distribution entropy. | Falling entropy alone is neither success nor failure. |
| train/kl_divergence, train/kl_loss | Rollout-policy divergence, where logged. | Detect overly large updates. |
| train/grad_norm | Gradient norm, where logged. | Watch spikes or clipping saturation. |
| train/learning_rate | Current optimizer learning rate. | Verify configuration and schedule. |
| train/num_invalids | Padded/invalid rollout entries, where logged. | High values reduce useful batch size. |

reward/reward and ordinary episode-return metrics are external DMLab reward. They should be zero in the no-reward open field and are not IntrMotiv worker reward.

## Distance And DG Activity

| Tag | Exact quantity | Interpretation |
| --- | --- | --- |
| intrmotiv/distance/mean | Mean pairwise CA3 progression distance. | Internal temporal-separation scale; observational, not exploration performance. |
| intrmotiv/distance/min, max, std | Extrema/spread of unmasked progression distances. | Saturation, clipping, and distance spread. |
| intrmotiv/distance/masked_mean | Mean distance on the valid progression-mask subset. | Distance population selected by the progression mask. |
| intrmotiv/distance/masked_min, masked_max, masked_std | Extrema/spread of masked distances. | Compare with unmasked values to diagnose the mask. |
| intrmotiv/dg/active_count | Mean post-threshold active DG units per transition. | Approximately F times density. |
| intrmotiv/dg/active_count_min, active_count_max, active_count_std | Distribution of active-unit count per transition. | Silent, sparse, and collision behavior. |
| intrmotiv/dg/density | Active post-threshold DG entries / all DG entries. | Landmark-event density. |
| intrmotiv/dg/multi_activation_fraction | Transitions with more than one active DG. | Collision/ambiguity rate. |
| intrmotiv/dg/silent_unit_fraction | DG rows with no activation in minibatch. | Local collapse diagnostic, not proof of global death. |
| intrmotiv/dg/unit_duty_cycle_min | Lowest unit-wise active fraction in minibatch. | Whether every unit participates locally. |
| intrmotiv/dg/unit_duty_cycle_mean | Mean unit-wise active fraction. | Unit-balanced counterpart to density. |
| intrmotiv/dg/unit_duty_cycle_max | Highest unit-wise active fraction. | Dominant-unit behavior. |
| intrmotiv/dg/usage_entropy | Entropy of duty-cycle mass, normalized by log(F). | One means even usage; zero means one or no used units. |
| intrmotiv/dg/learner_active_transition_fraction | Valid learner transitions with at least one post-threshold DG activation / valid transitions. | Learner-side activity frequency. It is not directly comparable to behavior onset frequency because it includes continued activations. |
| intrmotiv/dg/behavior_dominant_event_fraction | Valid transitions containing a dominant DG onset / valid transitions. | Frequency of behavior-side landmark events that can receive encoder feedback. |
| intrmotiv/dg/behavior_multi_onset_event_fraction | Dominant-onset events with at least one simultaneous non-dominant onset / dominant-onset events. | Conditional collision rate for corrected simultaneous-onset handling. |
| intrmotiv/dg/behavior_non_dominant_onsets_per_event | Simultaneous non-dominant onset count / dominant-onset event count. | Average number of DG competitors penalized at each feedback event. |
| intrmotiv/dg/valid_minibatch_unused_unit_count | DG units with no positive pre-threshold logit among valid minibatch transitions. | Exact population receiving corrected batch recruitment. |
| intrmotiv/dg/valid_minibatch_unused_unit_fraction | Unused-unit count / F. | Batch recruitment prevalence independent of F. |
| intrmotiv/dg/pre_threshold_mean | Mean BatchNorm DG logit before hard threshold/ReLU. | Population shift relative to threshold. |
| intrmotiv/dg/pre_threshold_above_fraction | Pre-threshold logits above threshold / all logits. | Smooth counterpart to density. |
| intrmotiv/dg/ca3_conflict_fraction | Unit-transition entries masked because another DG unit was active in the preceding R decisions / all valid unit-transition entries. | Diagnostic potential-conflict coverage. One recent unit can mask `(F-1)/F` entries, so this is not a violation rate; since 2026-09-03 this mask no longer gates the event-level temporal-margin loss. |
| intrmotiv/dg/ca3_conflicting_activation_fraction | Current post-threshold active DG entries that are CA3-conflict-masked / all current post-threshold active DG entries. | Primary temporal-exclusion violation rate; zero when there are no current activations. |
| intrmotiv/dg/ca3_conflict_activity | Mean current post-threshold DG amplitude over CA3-conflict-masked entries. | Violation magnitude, including zero activity on masked entries. |
| intrmotiv/dg/recruitment/connected_fraction, isolated_fraction | Fraction of DG vertices with, or without, any incoming or outgoing edge satisfying `confidence > threshold` and positive elapsed time. | In graph mode, a connected nonredundant mature field is protected from reassignment. |
| intrmotiv/dg/recruitment/redundant_pair_count | Bidirectionally supported pairs whose two elapsed-time estimates are at most the configured redundancy threshold. | Counts pairwise redundancy opportunities, not victims; a vertex can occur in several pairs. |
| intrmotiv/dg/recruitment/eligible_vertex_count | Birth-mature isolated vertices plus birth-mature redundant-pair losers. | Zero means a silent endpoint cannot cause reassignment. |
| intrmotiv/dg/recruitment/birth_protected_count | Vertices whose birth support is above the recruitment connectivity threshold. | Newly assigned fields remain protected until graph-clock decay expires. |
| intrmotiv/dg/recruitment/repeat_total | Cumulative assignments to rows that had already been assigned at least once. | Direct representation-churn measure; should plateau late in training. |
| intrmotiv/dg/recruitment/isolated_assignments_per_rollout, redundant_assignments_per_rollout | Assignments in the latest accepted rollout, split by eligibility reason. | The two values sum to graph-mode assignments for that rollout. |
| intrmotiv/dg/recruitment/passive_graph_density | Above-threshold directed passive edges / `F(F-1)`. | Passive fallback connectivity, reported even when policy-buffer HRL selects the controllability graph for eligibility. |
| intrmotiv/dg/recruitment/passive_updates_per_rollout | Accepted passive transitions in the latest rollout. | Confirms that behavior-time exclusive DG history is producing graph evidence. |
| intrmotiv/dg/recruitment/passive_stale_per_rollout | Candidate passive transitions rejected for a representation-generation mismatch. | Expected immediately around reassignment; persistent growth suggests delayed stale evidence. |
| intrmotiv/dg/recruitment/passive_over_gap_per_rollout | Candidate passive transitions rejected because the landmark gap exceeded `L`. | Confirms that remote event pairs are not treated as local graph edges. |
| intrmotiv/encoder/credit/total_events, matchable_events, credited_events | Dominant arrival onsets, arrivals with an in-rollout predecessor candidate, and events that pass full behavior-label/validity alignment. | `credited_events / total_events` is the source-credit preflight manipulation check. |
| intrmotiv/encoder/credit/boundary_dropped_events, alignment_failures, invalid_intervals, collisions | Reasons matched credit was dropped, plus multiple accepted credits accumulated on one recipient row/time. | ARR and SRC must retain the same events and total mass; only the replayed recipient differs. |
| intrmotiv/encoder/credit/reward_mass, source_lag_mean, source_lag_max | Total matched `0.1 d` credit and predecessor distance. | Verifies exact temporal credit without adding success shaping. |
| intrmotiv/dg/recruitment/active_endpoint_count, activity_blocked_count, eligible_victim_endpoint_count, residual_pass_count, replacement_conversion | L-endpoint trigger funnel from active endpoint through victim/residual eligibility to assignment. | `silent` blocks active endpoints; `open` permits them up to the one-per-rollout cap. |
| intrmotiv/dg/recruitment/predictive_supported_context_count, predictive_context_coverage_fraction, predictive_decayed_attempt_mass, predictive_invalidation_mass | Persistent PRED support, coverage, global decay, and evidence cleared by reassignment. | Evidence is checkpointed and indexed by source, goal, and predecessor context. |
| intrmotiv/hrl/target_hit_numerator, target_hit_event_count, shuffled_hit_numerator, shuffled_hit_event_count | Raw target-hit and matched-shuffle components. | Aggregate these components before computing rates or lift; do not average the unstable lift ratio. |

These are minibatch diagnostics. Use trends and the 10k-decision place-field evaluation before claiming irreversible DG collapse.

The reusable offline rollout, manifest, active-only diversity metrics, and
interpretation rules are documented in
[[reusable_place_field_telemetry|Reusable DG Place-Field Telemetry]].

## Rewards

| Tag | Exact quantity | Interpretation |
| --- | --- | --- |
| intrmotiv/reward/advantage_mean | Mean reward supplied to PPO/GAE. | Actual worker learning signal. |
| intrmotiv/reward/advantage_sum | PPO/GAE reward sum in minibatch. | Batch-size dependent. |
| intrmotiv/reward/advantage_abs_mean | Mean absolute PPO/GAE reward. | Reward scale without sign cancellation. |
| intrmotiv/reward/advantage_nonzero_fraction | Nonzero PPO/GAE rewards / valid transitions. | Supervision sparsity. |
| intrmotiv/reward/advantage_min, advantage_max | PPO/GAE reward extrema. | Outlier and clipping check. |
| intrmotiv/reward/intrinsic_mean, intrinsic_sum | Mean/sum internal reward before PPO-specific use. | Reward construction; sum is batch-size dependent. |
| intrmotiv/reward/intrinsic_nonzero_fraction | Nonzero internal rewards / valid transitions. | In hit_distance HRL, closely tracks target-hit rate. |
| intrmotiv/reward/intrinsic_negative_fraction | Negative internal rewards / valid transitions. | Should be zero for current nonnegative hit_distance HRL. |
| intrmotiv/reward/environment_mean, environment_sum | Original DMLab reward. | Expected zero in no-reward open field. |
| intrmotiv/reward/environment_nonzero_fraction | Nonzero DMLab rewards / valid transitions. | Detect unintended external rewards. |

The flat decoder uses dense temporal-distance reward. HRL hit_distance gives zero worker reward on a non-hit, then hit reward plus optional bounded distance bonus on a hit. Rare hits therefore imply sparse HRL PPO supervision.

## Optimization And Encoder Objectives

| Tag                                      | Exact quantity                                                            | Interpretation                                                                    |
| ---------------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| intrmotiv/update/phase                   | 0 simultaneous; 1 decoder-only; 2 DG-projection encoder-only.             | Confirms iterative schedule, not update quality.                                  |
| intrmotiv/encoder/loss                   | Total encoder objective after configured coefficients and gradient scale. | Objective differentiated during encoder phases.                                   |
| intrmotiv/decoder/loss                   | Actor, critic, and enabled decoder-side auxiliary objective.              | Objective differentiated during decoder phases.                                   |
| intrmotiv/decoder/auxiliary_loss         | Optional clipped decoder auxiliary term.                                  | Zero when disabled.                                                               |
| intrmotiv/encoder/feedback_mean          | Mean encoder feedback/reward.                                             | Sign follows encourage/punish/mean; not comparable as performance across methods. |
| intrmotiv/encoder/dominant_event_count   | Number of valid dominant-onset feedback events in the current minibatch.   | Effective event sample count for the encoder reward.                               |
| intrmotiv/encoder/feedback_on_dominant_event_mean | Signed encoder feedback averaged only over dominant-onset events. | Event-conditional learning signal without dilution by non-event transitions.       |
| intrmotiv/encoder/feedback_abs_on_dominant_event_mean | Absolute encoder feedback averaged only over dominant-onset events. | Event-conditional signal magnitude.                                                |
| intrmotiv/encoder/multi_activation_loss  | DG coactivation penalty.                                                  | Zero when disabled.                                                               |
| intrmotiv/encoder/unused_sequence_loss   | Sequence-level inactive-unit feedback.                                    | Zero when disabled.                                                               |
| intrmotiv/encoder/batch_usage_loss       | Learner-minibatch DG usage loss.                                          | Can be zero when the batch satisfies its activity criterion.                      |
| intrmotiv/encoder/population_loss        | Batch-wise population-activity regularizer.                               | Zero when disabled.                                                               |
| intrmotiv/encoder/usage_loss             | Batch-wise usage-balancing regularizer.                                   | Zero when disabled.                                                               |
| intrmotiv/encoder/density_loss           | Batch-wise density-target regularizer.                                    | Zero when disabled.                                                               |
| intrmotiv/encoder/collision_loss         | Batch-wise collision regularizer.                                         | Zero when disabled.                                                               |
| intrmotiv/encoder/global_punishment_loss | Weighted all-unit pre-threshold penalty.                                  | Suppresses frequent high logits; row normalization can rotate DG directions.      |
| intrmotiv/encoder/row_repulsion_loss     | Weighted squared off-diagonal cosine similarity of normalized DG rows.    | Separates duplicate directions without an activation target.                      |

The ImageNet-pretrained ResNet-18 layer-2 trunk stays fixed. Encoder metrics concern the trainable DG projection and its BatchNorm running statistics.

## Optional CA3 Predictor

| Tag | Exact quantity | Interpretation |
| --- | --- | --- |
| intrmotiv/predictor/loss | Configured auxiliary target-prediction loss. | Optimization diagnostic. |
| intrmotiv/predictor/hit_accuracy | Target-hit classification accuracy. | Compare with positive fraction; high accuracy can be trivial for rare hits. |
| intrmotiv/predictor/hit_time_mae | Mean absolute predicted hit-time error. | Meaningful only on defined positive/valid targets. |
| intrmotiv/predictor/positive_fraction | Positive targets / predictor targets. | Class-imbalance baseline. |

The predictor does not affect target choice or worker actions unless an architecture explicitly wires it into decisions.

## Optional PBT Routing

These tags are absent in the current no-PBT batches. They describe the old PBT
selection path, not model performance by themselves.

| Tag | Exact quantity | Interpretation |
| --- | --- | --- |
| intrmotiv/pbt/distance_metric | Historical only; no longer emitted. | `distance_metric` is not a valid PBT objective. Use the guarded exploration objective or an external coverage metric. |
| intrmotiv/pbt/hrl_validity | One when HRL activity/silence/reward checks pass. | Guard against promoting structurally invalid HRL policies. |
| intrmotiv/pbt/objective | Coverage objective when valid, otherwise zero. | The value PBT used for policy selection. |

## HRL Option Metrics

These appear only with hrl_controllable_graph=true. Target, source, hit, expiry, age, and deadline are stored in RNN option state; a normal terminal reset never creates a physical graph transition.

| Tag | Exact quantity | Interpretation |
| --- | --- | --- |
| intrmotiv/hrl/active_target_fraction | Stored nonempty target IDs / valid transitions. | Normally near one after startup. |
| intrmotiv/hrl/source_fraction | Stored nonempty source IDs / valid transitions. | Landmark source-assignment health. |
| intrmotiv/hrl/target_hit_rate | Hit pulses / valid transitions. | Per-step event rate, not option success probability. |
| intrmotiv/hrl/option_timeout_rate | Expiry pulses / valid transitions. | Per-step event rate, not fraction of options that fail. |
| intrmotiv/hrl/option_success_fraction | Hits / (hits + timeouts). | Correct per-completed-option success fraction. |
| intrmotiv/hrl/option_reset_rate | Option-reset pulses / valid transitions. | Near hit plus timeout; extra resets can be empty-target resets. |
| intrmotiv/hrl/tctrl_update_rate | Intended successful graph updates / valid transitions. | Event frequency, not graph quality; can be below hit rate. |
| intrmotiv/hrl/learned_deadline_fraction | Resets with feasible graph-derived deadline / resets. | Whether T_ctrl changes timing rather than fallback horizon. |
| intrmotiv/hrl/selected_deadline_mean | Selected deadline sum on reset transitions / reset count. | Expected option duration. Compare to timeout elapsed time. |
| intrmotiv/hrl/elapsed_on_hit_mean | Option elapsed actions averaged only over hits. | Successful-arrival time. |
| intrmotiv/hrl/elapsed_on_timeout_mean | Option elapsed actions averaged only over timeouts. | Effective deadline; exposes unexpectedly short options. |

Hit and timeout cannot co-occur: expiry is masked when a hit occurs. In policy-buffer mode, each sampled target is retained in compact RNN state, so learner graph changes never alter replayed action target conditioning.

### Manager Exploration

These metrics appear when the opt-in manager exploration action is available.
Exploration is stored as reserved target ID `F` and receives the flat temporal-
distance worker reward; ordinary DG targets retain `hit_distance` reward.

| Tag | Exact quantity | Interpretation |
| --- | --- | --- |
| intrmotiv/hrl/active_option_fraction | Transitions with a normal DG target or exploration action / valid transitions. | Overall manager-action availability. |
| intrmotiv/hrl/exploration/mode_fraction | Exploration-conditioned transitions / valid transitions. | Time allocation, not exploration quality. |
| intrmotiv/hrl/exploration/selection_fraction | Exploration selections / option resets. | Effective manager exploration frequency, including forced selections. |
| intrmotiv/hrl/exploration/forced_selection_fraction | Target-timeout-forced selections / exploration selections. | Separates recovery from probabilistic choice. |
| intrmotiv/hrl/exploration/completion_rate | Completed exploration horizons / valid transitions. | Exploration option event rate. |
| intrmotiv/hrl/exploration/elapsed_mean | Exploration decisions / completed exploration option. | Should match the configured horizon. |
| intrmotiv/hrl/exploration/reward_mean | Mean worker reward on exploration transitions. | Flat temporal-distance signal magnitude. |
| intrmotiv/hrl/exploration/reward_nonzero_fraction | Nonzero worker rewards on exploration transitions / exploration transitions. | Exploration supervision density. |
| intrmotiv/hrl/exploration/selected_deadline_mean | Selected exploration deadline / exploration selections. | Configuration check. |
| intrmotiv/hrl/target_selected_deadline_mean | Selected DG-target deadline / DG-target selections. | Target timing without exploration or zero cases. |

`option_timeout_rate`, `elapsed_on_timeout_mean`, and
`option_success_fraction` remain target-only. Exploration completions do not
count as target failures.

## HRL Graph Metrics

F=16 gives 16 DG nodes. Fractions exclude diagonal self-edges. The policy-buffer graph is one non-gradient graph per policy and updates once per accepted rollout. The long per-stream graph lives in stream RNN state. Do not compare raw graph weights across scopes without accounting for event population and half-life.

| Tag                                           | Exact quantity                                                            | Interpretation                                                    |
| --------------------------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| intrmotiv/hrl/node_coverage_fraction          | Positive node visit weights / all represented node weights.               | Breadth of observed DG nodes.                                     |
| intrmotiv/hrl/node_visit_weight_mean          | Mean decayed node-visit weight.                                           | Memory scale only; depends on half-life and graph scope.          |
| intrmotiv/hrl/selected_target_visit_mean      | Selected target visit weight, averaged at option resets.                  | Novelty-selection check. Lower means less-visited targets.        |
| intrmotiv/hrl/known_edge_fraction             | Off-diagonal T_ctrl edges with confidence at/above threshold.             | Usable graph density.                                             |
| intrmotiv/hrl/forgotten_edge_fraction         | Previously observed off-diagonal T_ctrl edges below confidence threshold. | Decay/forgetting effect.                                          |
| intrmotiv/hrl/known_controllability_time_mean | Mean T_ctrl over confidence-qualified edges.                              | Empirical arrival-time estimate, not target-ranking cost.         |
| intrmotiv/hrl/edge_confidence_mean            | Mean decayed off-diagonal edge confidence.                                | Strength/recency; compare only matched half-life and graph scope. |

On completion or timeout, persistent fast weights decay by gamma = 0.5 ** (1 / half_life_options). An intended i -> j success in tau steps increments confidence and updates T_ctrl as a confidence-weighted arrival-time mean. Edges below hrl_edge_confidence_threshold are infeasible for graph closure and learned deadlines. Target ranking remains novelty-first: T_ctrl gates feasibility and supplies timing, not destination cheapness.

## Topological Frontier, Planning, And Path Metrics

These metrics appear only for `hrl_manager_mode=frontier_direct` or
`frontier_waypoint`. Event rates use valid learner transitions. Values ending
in `per_rollout` describe the most recently accepted rollout and are not
cumulative totals.

| Tag | Exact quantity | Interpretation |
| --- | --- | --- |
| intrmotiv/hrl/target_hit_lift | Current target's DG activation rate divided by a one-position shifted-target activation rate in the same minibatch. | Above one indicates target-specific activation beyond marginal DG/target frequency. Treat unstable values near a zero shuffled baseline cautiously. |
| intrmotiv/hrl/passive/updates_per_rollout | Accepted exclusive local DG transitions in the latest rollout. | Must become nonzero before edge validation can operate. |
| intrmotiv/hrl/passive/known_edge_fraction | Passive confidence-qualified directed pairs / all off-diagonal pairs. | Observed local topology, not controllable topology. |
| intrmotiv/hrl/passive/candidate_edge_fraction | Confidence-qualified passive pairs whose deliberate confidence is below threshold / all off-diagonal pairs. | Validation backlog. |
| intrmotiv/hrl/passive/traversal_time_mean | Confidence-weighted passive elapsed time over observed edges. | Local temporal scale. |
| intrmotiv/hrl/passive/path_length_mean | Confidence-weighted integrated path length over observed edges. | Action-enabled local motion scale; elapsed-time proxy in no-action ablations. |
| intrmotiv/hrl/passive/reject_nonexclusive_rate | Multi-DG candidate observations rejected / valid transitions. | Ambiguous landmark gating. |
| intrmotiv/hrl/passive/reject_time_rate | Distinct exclusive transitions beyond L / valid transitions. | Temporal locality gate. |
| intrmotiv/hrl/passive/reject_path_rate | Distinct exclusive transitions with integrated path above L / valid transitions. | Motion-path locality gate. |
| intrmotiv/hrl/passive/reject_motion_rate | Distinct exclusive transitions below minimum net displacement / valid transitions. | Filters visually changing but locally stationary activations. |
| intrmotiv/hrl/frontier/score_mean | Mean selected UCB frontier score. | Manager curiosity score; travel time is not included. |
| intrmotiv/hrl/frontier/selection_rate | Frontier-selection pulses / valid transitions. | Selection event frequency. |
| intrmotiv/hrl/frontier/attempts_per_rollout | Completed or discovery-terminated frontier explorations in latest rollout. | Denominator population for frontier yield. |
| intrmotiv/hrl/frontier/discoveries_per_rollout | Explorations ending in a new stable transition in latest rollout. | Productive exploration events. |
| intrmotiv/hrl/frontier/yield | Decayed discoveries / decayed attempts. | Per-node aggregated frontier productivity. |
| intrmotiv/hrl/frontier/reached_fraction | Final-frontier reach pulses / frontier-selection pulses in minibatch. | Route completion diagnostic; sparse at per-step resolution. |
| intrmotiv/hrl/planning/route_available_rate | Route-available pulses / valid transitions. | Whether validated topology supports requested routes. |
| intrmotiv/hrl/planning/hop_count_mean | Validated route hop count averaged on route selections. | Explicit multi-hop use. |
| intrmotiv/hrl/planning/target_hit_rate | Deliberate target-hit pulses / valid transitions. | Generic commanded-target hit rate; includes direct targets, returns, validations, and waypoints. |
| intrmotiv/hrl/planning/waypoint_navigation_fraction | Navigation transitions with more than one remaining graph hop / valid transitions. | Whether the manager is executing a genuine multi-hop route. |
| intrmotiv/hrl/planning/waypoint_step_hit_rate | Target-hit pulses during genuine multi-hop navigation / those navigation transitions. | Per-step local success signal for waypoint execution, not per-option route success. |
| intrmotiv/hrl/planning/replan_rate | Option-reset pulses / valid transitions. | Replanning frequency. |
| intrmotiv/hrl/planning/final_frontier_reach_rate | Final-frontier reach pulses / valid transitions. | Per-step final-route event rate. |
| intrmotiv/hrl/validation/queued_edges | Current passive candidate count. | Global validation backlog. |
| intrmotiv/hrl/validation/return_success_rate | Successful RETURN pulses / valid transitions. | Ability to get back to candidate sources. |
| intrmotiv/hrl/validation/success_rate | Successful deliberate VALIDATE pulses / valid transitions. | Direct confirmation rate for passively proposed edges. Any deliberate target hit updates `T_ctrl`; passive edges are never promoted by observation alone. |
| intrmotiv/hrl/validation/timeout_rate | RETURN or VALIDATE timeout pulses / valid transitions. | Validation failure frequency. |
| intrmotiv/path/path_length_mean | Mean current stable-landmark trace traveled command distance. | Motion-history extent; repeated turning actions contribute their full repeated path, not chord length. |
| intrmotiv/path/displacement_mean | Mean current stable-landmark trace net displacement. | Compare with path length for tortuosity. |
| intrmotiv/path/straightness_mean | Displacement / path length. | Near one for straight command traces, near zero for loop returns. |
| intrmotiv/path/scatter_conflict_fraction | Active DG entries that are far from their same-unit anchor on a straight trace / all active entries. | Primary same-landmark scattering violation rate. |
| intrmotiv/path/scatter_loss | Weighted pre-threshold softplus loss on those conflicts. | Optimizer pressure, not a behavioral score. |
| intrmotiv/path/telemetry_error | Command-integrated versus DMLab debug-position trajectory RMSE after translation, rotation, and scale alignment, normalized by actual trajectory RMS extent. | Zero is exact trajectory shape; larger values expose collision and fixed-egomotion mismatch without treating DMLab coordinates as DG inputs. Emitted once per physical episode. |
| intrmotiv/geometry/se2_stress | Confidence-weighted passive pose-constraint residual after the latest fit. | Metric-control fit health. Compare only SE(2) runs. |
| intrmotiv/geometry/valid_landmark_fraction | DG nodes with initialized SE(2) poses / F. | Pose-graph coverage. |
| intrmotiv/geometry/proposed_edge_fraction | Current SE(2)-nearest unvalidated pairs / all off-diagonal pairs. | Geometry-only proposal density; zero when the SE(2) control is disabled. |

Diagnosis order is representation health, passive updates, deliberate
validation, validated routes, target-hit lift, then matched external coverage.
Coverage alone does not establish that the graph algorithm is functioning.

## Environment Exploration Telemetry

These are environment statistics, not learner minibatch statistics. Fixed-length runs emit physical episode data, typically under policy_stats/avg_z_.... Long-episode runs additionally emit intrmotiv/exploration/window/ every 900 policy decisions. A window does not reset DMLab, CA3, option state, or graph memory.

When `exploration_coverage_telemetry` is enabled, DMLab supplies privileged
horizontal pose `(x, y, yaw)` to the environment wrapper. The wrapper removes
the private value from `info` after updating the statistics below. Pose is not
declared in the training observation space and therefore cannot enter
observation normalization, rollout tensors, the encoder, the policy, rewards,
or checkpoints. `exploration_heading_bin_degrees` controls yaw discretization
and defaults to 15 degrees.

| Tag suffix / window tag | Exact quantity | Interpretation |
| --- | --- | --- |
| ...coverage_unique_cells | Distinct discretized cells in interval. | Spatial extent; higher is better at fixed length. |
| ...coverage_auc | (1/N) times the sum over t of U_t, with U_t unique cells observed through t. | Rewards early coverage and final extent; compare equal-length intervals only. |
| ...coverage_entropy | Shannon entropy of discretized-cell occupancy, normalized as configured by telemetry. | Evenness can be high despite low spatial extent. |
| ...pose_unique_bins | Distinct joint `(x-cell, y-cell, yaw-bin)` states in the interval. | Viewpoint coverage; distinguishes revisiting a location from different headings. |
| ...pose_auc | `(1/N)` times the sum over `t` of joint pose bins observed through `t`. | Rewards early viewpoint coverage; compare only identical spatial and yaw bins. |
| ...pose_entropy | Shannon entropy of joint `(x-cell, y-cell, yaw-bin)` occupancy. | Viewpoint evenness; interpret alongside spatial coverage. |
| ...window_return / episode return | DMLab external reward sum. | Expected zero in no-reward open field. |
| ...window_length / episode length | Decisions in telemetry window or physical episode. | Validate interval before comparing totals. |

## Compact Online Spatial Telemetry

`online_spatial_telemetry=True` is default-on and independently disableable.
It retains the latest 100,000 valid behavior samples per policy in a fixed ring
buffer. It uses the latest 10,000 samples for the scalar series below every 1M
environment frames and the full 100,000 for milestone maps and artifacts. The
privileged `telemetry_pose` channel is removed before tensor conversion,
normalization, and every model call. Training-time image logging is prohibited.

All place-field quantities use a 19×19 occupancy-corrected grid over
`x,y=100..2000`. “Active” means that a unit has positive thresholded DG
activity on at least one in-bounds sample. Trajectory deltas exclude policy-lag
invalid samples, rollout gaps, transitions after terminal samples, and
unmarked physical relocations larger than the configurable 250-unit default.

| Tag | Exact quantity |
| --- | --- |
| `intrmotiv/online/place_field/valid_sample_count` | Valid behavior samples in the scalar-analysis tail, at most 10,000 by default. |
| `.../in_bounds_fraction` | Retained finite poses inside both configured bounds / retained poses. |
| `.../visited_cell_fraction` | Occupied grid cells / 361. |
| `.../active_unit_fraction` | Units active at least once in bounds / DG units. |
| `.../silent_unit_fraction` | One minus active-unit fraction. |
| `.../active_unit_mean_spatial_information` | Mean occupancy-weighted Skaggs spatial information over active units. |
| `.../active_only_map_cosine` | Mean pairwise cosine of active occupancy-corrected maps over visited cells. |
| `.../unique_active_peak_bins` | Distinct peak grid bins across active units. |
| `.../mono_field_unit_fraction` | Eligible units whose dominant 8-connected component contains at least 80% of superlevel mass at 30%, 50%, and 70% of peak / eligible units. Eligibility requires 20 in-bounds active observations across three bins. |
| `.../mean_primary_secondary_peak_distance` | Mean physical distance between the two highest-mass 50%-of-peak components, over units that have both. |
| `.../median_dominant_peak_nearest_neighbor_distance` | Median per-unit distance to the nearest other eligible unit's dominant peak. |
| `intrmotiv/online/trajectory/mean_physical_step_distance` | Mean Euclidean `(x,y)` displacement over valid within-segment transitions. |
| `.../stationary_step_fraction` | Fraction of those displacements no larger than the configured 1-unit default. |
| `.../path_efficiency` | Sum of segment endpoint displacements / sum of within-segment path lengths. |
| `.../mean_absolute_circular_yaw_change` | Mean absolute yaw delta wrapped to `[-180,180)` degrees. |
| `intrmotiv/hrl/summary/reliable_global_efficiency` | Mean reciprocal directed shortest-path hop count over all ordered unit pairs; unreachable pairs contribute zero. |
| `intrmotiv/hrl/summary/grounded_controllability` | Pre-update reliable-edge prospective success fraction times the fraction of reliable edges joining two eligible mono-field units. |

Compressed latest-100k snapshots are written at 5M, 25M, 50M, 75M, and 100M
frames under the NEMO workspace analysis root. They cache evaluator-compatible
maps, field components, graph buffers, prospective edge outcomes, and detailed
graph diagnostics. Use `collect-spatial --include-details` for StudySpec-aware
per-unit, per-field, and graph-edge CSVs, or explicit selected-run maps. These
are monitoring diagnostics; fixed-checkpoint manifest telemetry remains the
authoritative scientific analysis.

The fixed-checkpoint manifest analyzer uses the same 8-connected 30/50/70%
component definition and adds `mono_field_fraction`,
`mean_components_{30,50,70}pct`, `mean_dominant_component_mass`, and
`incoming_confidence_field_spread_correlation` to its checkpoint summary. The
last quantity is the finite Pearson correlation, over spatially eligible DG
units, between total incoming edge confidence and `1 - mono_score`; it is
undefined when either side has zero variance.

## Recommended Panels

1. Representation health: density, silent fraction, duty-cycle min/max, usage entropy, and pre-threshold-above fraction.
2. Intrinsic supervision: advantage nonzero fraction, intrinsic nonzero fraction, target-hit rate, timeout rate, and option success fraction.
3. HRL mechanism: target/source fractions, selected deadline, timeout elapsed time, learned-deadline fraction, usable edges, and edge confidence.
4. External behavior: coverage AUC, unique cells, occupancy entropy, joint-pose AUC/entropy, and matched episode/window length.
5. Optimization: PPO loss/entropy/throughput with update phase, encoder loss, decoder loss, and enabled regularizers.

No single internal metric is a sufficient run objective. A result is credible only when representation health, worker supervision, graph behavior, and matched external coverage agree.
