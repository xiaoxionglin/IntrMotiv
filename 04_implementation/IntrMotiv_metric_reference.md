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
| intrmotiv/dg/pre_threshold_mean | Mean BatchNorm DG logit before hard threshold/ReLU. | Population shift relative to threshold. |
| intrmotiv/dg/pre_threshold_above_fraction | Pre-threshold logits above threshold / all logits. | Smooth counterpart to density. |

These are minibatch diagnostics. Use trends and the 10k-decision place-field evaluation before claiming irreversible DG collapse.

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

| Tag | Exact quantity | Interpretation |
| --- | --- | --- |
| intrmotiv/update/phase | 0 simultaneous; 1 decoder-only; 2 DG-projection encoder-only. | Confirms iterative schedule, not update quality. |
| intrmotiv/encoder/loss | Total encoder objective after configured coefficients and gradient scale. | Objective differentiated during encoder phases. |
| intrmotiv/decoder/loss | Actor, critic, and enabled decoder-side auxiliary objective. | Objective differentiated during decoder phases. |
| intrmotiv/decoder/auxiliary_loss | Optional clipped decoder auxiliary term. | Zero when disabled. |
| intrmotiv/encoder/feedback_mean | Mean encoder feedback/reward. | Sign follows encourage/punish/mean; not comparable as performance across methods. |
| intrmotiv/encoder/multi_activation_loss | DG coactivation penalty. | Zero when disabled. |
| intrmotiv/encoder/unused_sequence_loss | Sequence-level inactive-unit feedback. | Zero when disabled. |
| intrmotiv/encoder/batch_usage_loss | Learner-minibatch DG usage loss. | Can be zero when the batch satisfies its activity criterion. |
| intrmotiv/encoder/population_loss | Batch-wise population-activity regularizer. | Zero when disabled. |
| intrmotiv/encoder/usage_loss | Batch-wise usage-balancing regularizer. | Zero when disabled. |
| intrmotiv/encoder/density_loss | Batch-wise density-target regularizer. | Zero when disabled. |
| intrmotiv/encoder/collision_loss | Batch-wise collision regularizer. | Zero when disabled. |
| intrmotiv/encoder/global_punishment_loss | Weighted all-unit pre-threshold penalty. | Suppresses frequent high logits; row normalization can rotate DG directions. |
| intrmotiv/encoder/row_repulsion_loss | Weighted squared off-diagonal cosine similarity of normalized DG rows. | Separates duplicate directions without an activation target. |

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
| intrmotiv/pbt/distance_metric | Current distance metric mirrored for PBT. | A routing statistic; it is not a coverage objective. |
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

## HRL Graph Metrics

F=16 gives 16 DG nodes. Fractions exclude diagonal self-edges. The policy-buffer graph is one non-gradient graph per policy and updates once per accepted rollout. The long per-stream graph lives in stream RNN state. Do not compare raw graph weights across scopes without accounting for event population and half-life.

| Tag | Exact quantity | Interpretation |
| --- | --- | --- |
| intrmotiv/hrl/node_coverage_fraction | Positive node visit weights / all represented node weights. | Breadth of observed DG nodes. |
| intrmotiv/hrl/node_visit_weight_mean | Mean decayed node-visit weight. | Memory scale only; depends on half-life and graph scope. |
| intrmotiv/hrl/selected_target_visit_mean | Selected target visit weight, averaged at option resets. | Novelty-selection check. Lower means less-visited targets. |
| intrmotiv/hrl/known_edge_fraction | Off-diagonal T_ctrl edges with confidence at/above threshold. | Usable graph density. |
| intrmotiv/hrl/forgotten_edge_fraction | Previously observed off-diagonal T_ctrl edges below confidence threshold. | Decay/forgetting effect. |
| intrmotiv/hrl/known_controllability_time_mean | Mean T_ctrl over confidence-qualified edges. | Empirical arrival-time estimate, not target-ranking cost. |
| intrmotiv/hrl/edge_confidence_mean | Mean decayed off-diagonal edge confidence. | Strength/recency; compare only matched half-life and graph scope. |

On completion or timeout, persistent fast weights decay by gamma = 0.5 ** (1 / half_life_options). An intended i -> j success in tau steps increments confidence and updates T_ctrl as a confidence-weighted arrival-time mean. Edges below hrl_edge_confidence_threshold are infeasible for graph closure and learned deadlines. Target ranking remains novelty-first: T_ctrl gates feasibility and supplies timing, not destination cheapness.

## Environment Exploration Telemetry

These are environment statistics, not learner minibatch statistics. Fixed-length runs emit physical episode data, typically under policy_stats/avg_z_.... Long-episode runs additionally emit intrmotiv/exploration/window/ every 900 policy decisions. A window does not reset DMLab, CA3, option state, or graph memory.

| Tag suffix / window tag | Exact quantity | Interpretation |
| --- | --- | --- |
| ...coverage_unique_cells | Distinct discretized cells in interval. | Spatial extent; higher is better at fixed length. |
| ...coverage_auc | (1/N) times the sum over t of U_t, with U_t unique cells observed through t. | Rewards early coverage and final extent; compare equal-length intervals only. |
| ...coverage_entropy | Shannon entropy of discretized-cell occupancy, normalized as configured by telemetry. | Evenness can be high despite low spatial extent. |
| ...window_return / episode return | DMLab external reward sum. | Expected zero in no-reward open field. |
| ...window_length / episode length | Decisions in telemetry window or physical episode. | Validate interval before comparing totals. |

## Recommended Panels

1. Representation health: density, silent fraction, duty-cycle min/max, usage entropy, and pre-threshold-above fraction.
2. Intrinsic supervision: advantage nonzero fraction, intrinsic nonzero fraction, target-hit rate, timeout rate, and option success fraction.
3. HRL mechanism: target/source fractions, selected deadline, timeout elapsed time, learned-deadline fraction, usable edges, and edge confidence.
4. External behavior: coverage AUC, unique cells, occupancy entropy, and matched episode/window length.
5. Optimization: PPO loss/entropy/throughput with update phase, encoder loss, decoder loss, and enabled regularizers.

No single internal metric is a sufficient run objective. A result is credible only when representation health, worker supervision, graph behavior, and matched external coverage agree.
