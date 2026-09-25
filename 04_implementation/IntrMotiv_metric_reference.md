# IntrMotiv Metric Reference

This is the lookup-oriented metric dictionary for the IntrMotiv project. For interpretation, failure patterns, and decision rules, use [[IntrMotiv_metrics_guidebook|IntrMotiv Metrics Guidebook]]. This page is organized by mechanism and namespace so an unfamiliar W&B/TensorBoard tag can be located quickly.

**Audit basis, 2026-09-25:** the current transfer/controller line in SF_hipposlam at branch <code>codex/fixed-reward-transfer-corrected-20260925</code> (commit <code>e389a7ab3d6908873431bba1bea0dc33eadbf054</code>), plus <code>codex/ca3-state-goal-followup-20260922</code> for the CA3 anti-collapse/readout diagnostics that are branch-specific. The destination document is <code>xiaoxionglin/IntrMotiv:main</code>.

**Primary code sources:** <code>sf_working_directories/IntrMotiv/dmlab/custom_learner.py</code>, <code>reward_summaries.py</code>, <code>online_spatial_telemetry.py</code>, <code>controller_learner.py</code>, <code>ca3_state_readout.py</code>, <code>evaluation/schema.py</code>, and <code>hpc_runs/intrmotiv_offpolicy/*</code>.

Not every run emits every family. A missing architecture-specific tag means “not produced by this run” unless the experiment contract says otherwise; do not silently interpret absence as zero.

## Quick namespace map

| Namespace | What it describes | Typical question |
| --- | --- | --- |
| <code>train/*</code> | Sample Factory/framework optimizer statistics | Is optimization numerically healthy? |
| <code>intrmotiv/reward/*</code> | Environment, intrinsic, and PPO/GAE reward streams | What signal is actually training the worker? |
| <code>intrmotiv/dg/*</code> | DG sparsity, usage, normalization, gradients, contextualization, prediction, recruitment | Is the landmark representation healthy and stable? |
| <code>intrmotiv/encoder/*</code> | DG projection objectives and temporal credit | What is shaping DG and with how much effective supervision? |
| <code>intrmotiv/memory/*</code> | CA3 event novelty and goal-gate diagnostics | Are events novel, familiar, ambiguous, or replay-mismatched? |
| <code>intrmotiv/ca3_readout/*</code> | Learned CA3-state goal/readout embedding | Does the readout encode predictive state/action information without collapse? |
| <code>intrmotiv/hrl/*</code> | Options, controllability graph, interventions, planning, manager modes | Is target-dependent control and useful graph structure present? |
| <code>intrmotiv/her/*</code> | PPO-side empirical HER | Is hindsight relabeling producing usable counterfactual supervision? |
| <code>intrmotiv/controller/*</code> | Event-driven controller DDQN, contextual anchors, replay, calibration | Is the separate high-level controller learning and recognizing contextual goals? |
| <code>intrmotiv/online/*</code> | Fixed-window place-field and trajectory telemetry | Are landmarks spatially grounded and is behavior physically sensible? |
| <code>intrmotiv/exploration/window/*</code>, <code>policy_stats/*</code> | Environment-side privileged-pose coverage | How much of physical state space was explored? |
| <code>ddqn/*</code> | Native off-policy recurrent DDQN/HER path | Is replay/TD learning and per-goal supervision healthy? |

## Denominators and aggregation rules

Unless a section says otherwise, learner-side fractions are computed over valid transitions in the current minibatch. They are not whole-run totals and they are not necessarily physical-episode statistics.

$\mathrm{target\_hit\_rate}=\mathrm{hits}/\mathrm{valid\ transitions}$

$\mathrm{option\_timeout\_rate}=\mathrm{timeouts}/\mathrm{valid\ transitions}$

$\mathrm{option\_success\_fraction}=\mathrm{hits}/(\mathrm{hits}+\mathrm{timeouts})$

Hits and timeouts are sparse option-completion events; most transitions can be neither. For causal-control metrics, sum numerators and event counts across runs/windows before dividing. Do not average unstable ratios such as hit lift across windows.

Suffix conventions:

- <code>*_count</code> is a count in the stated batch/rollout/window unless explicitly described as cumulative.
- <code>*_total</code> is cumulative unless the producing code says otherwise; use the final value rather than an average.
- <code>*_per_rollout</code> describes the latest accepted rollout.
- <code>*_fraction</code>/<code>*_rate</code> requires the family-specific denominator below.
- Environment coverage and compact spatial telemetry use physical behavior samples, not PPO minibatch transitions.

## 1. Sample Factory and framework diagnostics

| Tag | Meaning | Use |
| --- | --- | --- |
| <code>train/env_steps</code> | Frames seen by the policy. | Primary horizontal axis. |
| <code>train/fps</code>, <code>train/avg_fps</code> | Recent/average throughput. | CPU, DMLab, learner bottlenecks. |
| <code>train/policy_loss</code> | PPO clipped policy-surrogate loss. | Optimizer diagnostic, not behavior quality. |
| <code>train/value_loss</code> | Critic regression loss. | Persistent explosions indicate value/reward-scale problems. |
| <code>train/entropy</code> | Action-distribution entropy. | Exploration/optimization diagnostic only. |
| <code>train/kl_divergence</code>, <code>train/kl_loss</code> | Rollout-policy divergence where available. | Detect oversized updates. |
| <code>train/grad_norm</code> | Gradient norm where available. | Detect spikes or clipping saturation. |
| <code>train/learning_rate</code> | Current optimizer learning rate. | Verify schedules. |
| <code>train/num_invalids</code> | Padded/invalid rollout entries where available. | Effective minibatch size. |
| <code>train/dg_goal_modulation_norm</code> | Norm of the raw DG goal-modulation parameter when emitted through the generic SF route. | Parameter-magnitude diagnostic. |
| <code>train/dg_goal_modulation_gradient_norm</code> | Gradient norm of that modulation parameter. | Checks whether the modulation branch receives gradient. |

Ordinary <code>reward/reward</code> and episode-return metrics are external DMLab reward. They are not IntrMotiv worker reward.

## 2. Reward streams

| Tag family | Exact definition |
| --- | --- |
| <code>intrmotiv/reward/advantage_{mean,sum,abs_mean,min,max}</code> | Statistics of the reward supplied to PPO/GAE. This is the worker learning signal. Sums are minibatch-size dependent. |
| <code>intrmotiv/reward/advantage_nonzero_fraction</code> | Valid transitions with nonzero PPO/GAE reward divided by valid transitions. |
| <code>intrmotiv/reward/intrinsic_{mean,sum}</code> | Internal IntrMotiv reward before PPO-specific use. |
| <code>intrmotiv/reward/intrinsic_nonzero_fraction</code> | Valid transitions with nonzero internal reward divided by valid transitions. |
| <code>intrmotiv/reward/intrinsic_negative_fraction</code> | Valid transitions with negative internal reward divided by valid transitions. |
| <code>intrmotiv/reward/environment_{mean,sum}</code> | Original DMLab reward. Expected zero in no-reward open-field studies. |
| <code>intrmotiv/reward/environment_nonzero_fraction</code> | Valid transitions with nonzero DMLab reward divided by valid transitions. |

For hit-distance HRL, non-hit transitions normally receive zero worker reward; therefore reward nonzero fraction should be interpreted together with target-hit events.

## 3. Representation and memory

### 3.1 CA3 progression distance

<code>intrmotiv/distance/{mean,min,max,std}</code> summarize the unmasked CA3 progression-distance population. <code>intrmotiv/distance/masked_{mean,min,max,std}</code> summarize the subset selected by the progression mask. They are internal temporal-separation diagnostics, not direct exploration scores.

### 3.2 DG activation and usage

| Tag | Exact quantity / interpretation |
| --- | --- |
| <code>intrmotiv/dg/active_count</code>, <code>active_count_min</code>, <code>active_count_max</code>, <code>active_count_std</code> | Distribution of post-threshold active DG units per valid transition. |
| <code>intrmotiv/dg/density</code> | Active post-threshold DG entries divided by all DG entries. |
| <code>intrmotiv/dg/multi_activation_fraction</code> | Valid transitions with more than one active DG unit divided by valid transitions. |
| <code>intrmotiv/dg/silent_unit_fraction</code> | DG rows with no activation in the minibatch divided by DG rows. Local diagnostic, not proof of global death. |
| <code>intrmotiv/dg/unit_duty_cycle_{min,mean,max}</code> | Min/mean/max unit-wise active fraction in the minibatch. |
| <code>intrmotiv/dg/usage_entropy</code> | Entropy of duty-cycle mass normalized by <code>log(F)</code>; one is even use, zero is one/no used unit. |
| <code>intrmotiv/dg/learner_active_transition_fraction</code> | Valid learner transitions with at least one active DG unit divided by valid transitions. |
| <code>intrmotiv/dg/behavior_dominant_event_fraction</code> | Valid behavior transitions containing a dominant DG onset divided by valid transitions. |
| <code>intrmotiv/dg/behavior_multi_onset_event_fraction</code> | Dominant-onset events with at least one simultaneous non-dominant onset divided by dominant-onset events. |
| <code>intrmotiv/dg/behavior_non_dominant_onsets_per_event</code> | Simultaneous non-dominant onsets divided by dominant-onset event count. |
| <code>intrmotiv/dg/valid_minibatch_unused_unit_count</code> | DG units with no positive pre-threshold logit on valid minibatch transitions. |
| <code>intrmotiv/dg/valid_minibatch_unused_unit_fraction</code> | Unused count divided by <code>F</code>. |

### 3.3 Pre-threshold, normalization, and gradient diagnostics

| Tag | Exact quantity |
| --- | --- |
| <code>intrmotiv/dg/pre_threshold_mean</code> | Mean BatchNorm-normalized DG logit before hard threshold/ReLU. |
| <code>intrmotiv/dg/pre_threshold_above_fraction</code> | Fraction of pre-threshold logits above the activation threshold. |
| <code>intrmotiv/dg/normalization/raw_logit_mean_abs</code> | Across valid transitions, compute each row's mean raw logit; then average the absolute row means. |
| <code>intrmotiv/dg/normalization/raw_logit_variance_mean</code> | Mean per-row raw-logit variance over valid transitions. |
| <code>intrmotiv/dg/normalization/normalized_logit_mean_abs</code> | Mean absolute per-row mean after normalization. |
| <code>intrmotiv/dg/normalization/normalized_logit_variance_error</code> | Mean absolute deviation of normalized per-row variance from 1. |
| <code>intrmotiv/dg/gradient/ppo_norm</code> | DG-projection gradient norm attributable to the PPO/worker path when the diagnostic is active. |
| <code>intrmotiv/dg/gradient/encoder_norm</code> | DG-projection gradient norm attributable to the encoder objective. |
| <code>intrmotiv/dg/gradient/ppo_encoder_ratio</code> | PPO DG-gradient norm divided by encoder DG-gradient norm under the diagnostic convention. |
| <code>intrmotiv/dg/gradient/cosine</code> | Cosine similarity between PPO and encoder DG-gradient vectors. |
| <code>intrmotiv/dg/gradient/row_conflict_fraction</code> | DG rows whose PPO and encoder gradient contributions conflict under the row-wise criterion. |

On the controller-DDQN path these gradient diagnostics can intentionally be zeroed when PPO losses are not computed; read them with the active learning mode.

### 3.4 CA3 temporal exclusion and path scatter

<code>intrmotiv/dg/ca3_conflict_fraction</code> is the fraction of valid unit-transition entries masked by recent competing DG activity. It is potential-conflict coverage, not a violation rate. <code>intrmotiv/dg/ca3_conflicting_activation_fraction</code> is the primary violation rate: currently active DG entries that are conflict-masked divided by currently active DG entries. <code>intrmotiv/dg/ca3_conflict_activity</code> measures activity magnitude on conflict-masked entries. The related encoder losses are <code>intrmotiv/encoder/ca3_temporal_exclusion_loss</code> and <code>intrmotiv/path/scatter_loss</code>; <code>intrmotiv/path/scatter_conflict_fraction</code> reports the scatter conflict fraction.

### 3.5 Contextual DG

| Tag | Meaning |
| --- | --- |
| <code>intrmotiv/dg/context/created_fraction</code> | Fraction of contextual-feedback cases that create a contextual modulation/update. |
| <code>intrmotiv/dg/context/suppressed_fraction</code> | Fraction suppressed by contextual logic. |
| <code>intrmotiv/dg/context/unchanged_fraction</code> | Fraction left unchanged. |
| <code>intrmotiv/dg/context/modulation_abs_mean</code> | Mean absolute contextual DG modulation. |
| <code>intrmotiv/dg/context/modulation_saturation_fraction</code> | Fraction at the modulation saturation criterion. |
| <code>intrmotiv/dg/context/adapter_gradient_norm</code> | Gradient norm of the contextual adapter. |

These are mechanism diagnostics; they do not by themselves establish that contextual identity is correct.

### 3.6 DG transition prediction

| Tag | Meaning |
| --- | --- |
| <code>intrmotiv/dg/prediction/loss</code> | Total transition-prediction objective. |
| <code>intrmotiv/dg/prediction/main_loss</code> | Main prediction component. |
| <code>intrmotiv/dg/prediction/control_loss</code> | Control-conditioned component. |
| <code>intrmotiv/dg/prediction/validation_main_ce</code> | Held-out/validation main cross-entropy. |
| <code>intrmotiv/dg/prediction/validation_control_ce</code> | Held-out/validation control-conditioned cross-entropy. |
| <code>intrmotiv/dg/prediction/validation_state_gain</code> | Predictive gain attributable to state information under the validation contrast. |
| <code>intrmotiv/dg/prediction/validation_accuracy</code> | Validation prediction accuracy. |
| <code>intrmotiv/dg/prediction/validation_count</code> | Validation sample count. |
| <code>intrmotiv/dg/prediction/scheduled_count</code> | Prediction targets scheduled from behavior-time events. |
| <code>intrmotiv/dg/prediction/applied_count</code> | Scheduled targets actually applied after learner alignment. |
| <code>intrmotiv/dg/prediction/boundary_drop_count</code> | Scheduled targets dropped for crossing rollout boundaries. |
| <code>intrmotiv/dg/prediction/replay_match</code> | Applied count divided by <code>max(1, scheduled_count - boundary_drop_count)</code>. |

### 3.7 CA3 memory/event gate

| Tag | Meaning |
| --- | --- |
| <code>intrmotiv/memory/onset_fraction</code> | Fraction of steps containing a dominant DG onset. |
| <code>intrmotiv/memory/novel_onset_fraction</code> | Fraction satisfying the CA3 absent-event/novelty gate. |
| <code>intrmotiv/memory/familiar_onset_fraction</code> | Dominant onset that is not novel. |
| <code>intrmotiv/memory/gate_violation_fraction</code> | Nonzero gated decoder reward on a non-novel event; should remain zero when the absent-event gate is active. |
| <code>intrmotiv/memory/goal_ambiguous_fraction</code> | Stored goal event marked ambiguous. |
| <code>intrmotiv/memory/goal_active_fraction</code> | Fraction of aligned transitions carrying an active goal command. |
| <code>intrmotiv/memory/goal_hit_fraction</code> | Fraction with a positive aligned goal-hit pulse. |
| <code>intrmotiv/memory/replay_mismatch</code> | Memory/event replay mismatch diagnostic. |

### 3.8 CA3 state readout and goal representation

The readout learns a latent <code>z = W S</code> from CA3 state and predicts future DG innovation from <code>z</code>, an action prefix, and the prediction horizon.

| Tag | Exact quantity / interpretation |
| --- | --- |
| <code>intrmotiv/ca3_readout/enabled</code> | One when the CA3 state readout/predictor is active, otherwise zero. |
| <code>intrmotiv/ca3_readout/prediction_loss</code> | Active/zero-stratified Smooth-L1 prediction loss for future DG activity. |
| <code>intrmotiv/ca3_readout/active_loss</code> | Smooth-L1 averaged only over positive future-DG target entries. |
| <code>intrmotiv/ca3_readout/zero_loss</code> | Smooth-L1 averaged over zero target entries. |
| <code>intrmotiv/ca3_readout/valid_targets</code> | Number of causal prediction-window rows used. |
| <code>intrmotiv/ca3_readout/active_fraction</code> | Positive entries in future-DG targets divided by all target entries. |
| <code>intrmotiv/ca3_readout/state_shuffle_delta</code> | Within-horizon shuffled-state prediction loss minus unshuffled base loss. Positive values mean state identity carries predictive information. |
| <code>intrmotiv/ca3_readout/action_shuffle_delta</code> | Within-horizon shuffled-action-prefix prediction loss minus base loss. Positive values mean action history carries predictive information. |
| <code>intrmotiv/ca3_readout/var_loss</code> | CA3-follow-up anti-collapse term: mean <code>ReLU(1 - latent_std)</code> across latent dimensions. |
| <code>intrmotiv/ca3_readout/cov_loss</code> | CA3-follow-up anti-collapse term: squared off-diagonal latent covariance, normalized by latent dimension. |
| <code>intrmotiv/ca3_readout/latent_std_mean</code> | Mean latent standard deviation across dimensions over valid current states. |
| <code>intrmotiv/ca3_readout/latent_std_min</code> | Minimum latent standard deviation across dimensions; near zero identifies collapsed dimensions. |
| <code>intrmotiv/ca3_readout/total_loss</code> | CA3-follow-up only: <code>prediction_loss + var_coeff * var_loss + cov_coeff * cov_loss</code>. |

The five anti-collapse tags <code>total_loss</code>, <code>var_loss</code>, <code>cov_loss</code>, <code>latent_std_mean</code>, and <code>latent_std_min</code> are present on <code>codex/ca3-state-goal-followup-20260922</code> but not on the later corrected transfer branch. Keep that branch distinction when comparing W&B runs.

## 4. Encoder objectives, temporal credit, and update contract

### 4.1 Core objectives

<code>intrmotiv/update/phase</code> is 0 for simultaneous updates, 1 for decoder-only, and 2 for DG/encoder-only. <code>intrmotiv/encoder/loss</code> is the total differentiated encoder objective and <code>intrmotiv/decoder/loss</code> the decoder-side actor/critic/auxiliary objective.

The regularizer family comprises <code>intrmotiv/encoder/multi_activation_loss</code>, <code>unused_sequence_loss</code>, <code>batch_usage_loss</code>, <code>population_loss</code>, <code>usage_loss</code>, <code>density_loss</code>, <code>collision_loss</code>, <code>global_punishment_loss</code>, <code>row_repulsion_loss</code>, and <code>ca3_temporal_exclusion_loss</code>. Disabled objectives should be zero.

<code>intrmotiv/encoder/feedback_mean</code> is the signed feedback averaged over the minibatch. <code>dominant_event_count</code>, <code>feedback_on_dominant_event_mean</code>, and <code>feedback_abs_on_dominant_event_mean</code> expose the event-conditional sample size and signal magnitude, avoiding dilution by non-event transitions.

### 4.2 Encoder temporal credit

| Tag group | Definition |
| --- | --- |
| <code>total_events -> matchable_events -> credited_events</code> | Dominant arrival onsets, those with an in-rollout predecessor candidate, then events passing behavior-label/validity alignment. |
| <code>boundary_dropped_events</code>, <code>alignment_failures</code>, <code>invalid_intervals</code> | Reasons matched credit was rejected. |
| <code>collisions</code> | Multiple accepted credits accumulated on one recipient row/time. |
| <code>reward_mass</code> | Total matched credit mass. |
| <code>source_lag_mean</code>, <code>source_lag_max</code> | Behavior-time predecessor lag/distance statistics. |
| <code>scheduled_count</code>, <code>scheduled_reward_mass</code> | Requested behavior-labeled supervision before learner activity intersection. |
| <code>applied_count</code>, <code>applied_reward_mass</code> | Supervision whose chosen row is active in learner recomputation. |
| <code>replay_match</code> | <code>applied_count / max(1, scheduled_count)</code>. |
| <code>arrival_loss</code>, <code>source_loss</code> | Credit loss routed to the configured recipient branch; only the selected branch should be active in ARR/SRC treatments. |
| <code>credited_row_count</code> | Distinct/accepted credited learner rows under the current implementation. |

### 4.3 DG publication and stale-replay barrier

<code>intrmotiv/dg/update_contract/forward_count</code> and <code>running_stats_update_count</code> check the single-forward normalization contract. <code>weight_generation</code> and <code>statistics_generation</code> are the published representation/statistics generations; <code>publication_generation_mismatch</code> is a hard invariant and should remain zero.

<code>intrmotiv/replay/stale_generation_rejected_count</code> and <code>stale_generation_rejected_fraction</code> quantify samples rejected because their representation generation is stale. <code>dropped_rollouts_total</code>, <code>dropped_decisions_total</code>, and <code>deferred_updates_total</code> are cumulative replacement costs.

## 5. DG recruitment and retirement

Read recruitment as a funnel rather than only the final cumulative total.

### 5.1 Core funnel

<code>candidate_count</code> counts valid endpoint opportunities; <code>silent_endpoint_count</code>/<code>active_endpoint_count</code> split endpoint activity. <code>activity_blocked_count</code> counts opportunities rejected only by the legacy silent gate. <code>eligible_victim_endpoint_count</code> means a retirement rule selected a row. <code>residual_pass_count</code>/<code>residual_reject_count</code> report whether the orthogonal residual initializer is usable. <code>replacement_conversion</code> reports conversion to an assignment under the one-per-rollout cap. <code>total</code> is cumulative committed replacement; <code>repeat_total</code> is cumulative reassignment of rows that had already been assigned.

<code>residual_norm</code>, <code>tiny_residual_total</code>, <code>rollout_count</code>, <code>committed_fraction</code>, <code>endpoint_active_unit_count</code>/<code>endpoint_active_unit_mean</code>, and <code>victim_active_count</code>/<code>victim_active_fraction</code> diagnose where the funnel fails.

### 5.2 Graph protection, isolation, and redundancy

<code>connected_fraction</code>/<code>isolated_fraction</code> classify DG vertices by above-threshold graph connectivity and positive elapsed time. <code>redundant_pair_count</code> counts bidirectionally supported pairs below the configured redundancy-time threshold. <code>eligible_vertex_count</code> is the currently retireable isolated/redundant pool. <code>birth_protected_count</code> counts recently assigned vertices protected by birth support. Per-rollout assignment reasons are <code>repeat_assignments_per_rollout</code>, <code>isolated_assignments_per_rollout</code>, and <code>redundant_assignments_per_rollout</code>.

### 5.3 DIR source-quality diagnostics

<code>attempt_coverage_fraction</code>, <code>fully_tested_count</code>, <code>zero_outdegree_count</code>, <code>untested_zero_outdegree_count</code>, <code>bad_source_count</code>, <code>reliable_out_degree_mean</code>, and <code>reliable_outgoing_confidence_mean</code> distinguish genuinely bad controllability sources from untested nodes. <code>bad_source_assignments_per_rollout</code> records retirements attributed to that rule.

### 5.4 PRED contextual diagnostics

<code>predictive_event_count</code>, <code>predictive_context_group_count</code>, <code>predictive_eligible_count</code>, <code>predictive_reliability_gap</code>, <code>predictive_decayed_attempt_mass</code>, <code>predictive_supported_context_count</code>, <code>predictive_invalidation_mass</code>, and <code>predictive_context_coverage_fraction</code> describe support for context-dependent reliability splits. <code>predictive_assignments_per_rollout</code> is the resulting retirement count.

### 5.5 Eligibility and active-victim breakdown

<code>eligible_bad_source_endpoint_count</code>, <code>eligible_redundant_endpoint_count</code>, and <code>eligible_predictive_endpoint_count</code> split victim eligibility by rule. The matching <code>victim_active_*_count</code> and <code>victim_active_*_fraction</code> tags report whether those selected victims are active at the endpoint.

<code>goal_adapter_resets_per_rollout</code> and <code>goal_adapter_reset_total</code> count FiLM/goal-adapter rows reset by replacement. <code>forced_preflight_assignments_per_rollout</code> exists only for qualification forcing and should not be treated as natural retirement.

### 5.6 Passive recruitment evidence

<code>passive_graph_density</code> is the above-threshold directed passive-edge density. <code>passive_updates_per_rollout</code> counts accepted passive transitions. <code>passive_stale_per_rollout</code> and <code>passive_over_gap_per_rollout</code> count passive candidates rejected for stale generation or landmark gap greater than <code>L</code>.

## 6. HRL options, control, and graph

### 6.1 Option occupancy, deadlines, and outcomes

| Tag | Meaning |
| --- | --- |
| <code>intrmotiv/hrl/active_target_fraction</code> | Valid transitions carrying a normal target. |
| <code>intrmotiv/hrl/active_option_fraction</code> | Valid transitions carrying either a normal target or exploration option. |
| <code>intrmotiv/hrl/source_fraction</code> | Fraction with a valid source landmark. |
| <code>intrmotiv/hrl/target_hit_rate</code> | Hit events divided by valid transitions. |
| <code>intrmotiv/hrl/option_timeout_rate</code> | Target-timeout events divided by valid transitions. |
| <code>intrmotiv/hrl/option_reset_rate</code> | Option resets divided by valid transitions. |
| <code>intrmotiv/hrl/option_success_fraction</code> | Hits divided by hits plus target timeouts. |
| <code>intrmotiv/hrl/tctrl_update_rate</code> | Controllability-time updates divided by the relevant valid transition/event population. |
| <code>intrmotiv/hrl/selected_deadline_mean</code> | Mean selected deadline including zero/no-deadline cases according to producer. |
| <code>intrmotiv/hrl/selected_deadline_positive_mean</code> | Mean over positive selected deadlines only. |
| <code>intrmotiv/hrl/deadline_selection_fraction</code> | Fraction for which a positive/learned deadline is selected. |
| <code>intrmotiv/hrl/learned_deadline_fraction</code> | Fraction using learned rather than fallback deadline. |
| <code>intrmotiv/hrl/elapsed_on_hit_mean</code>, <code>elapsed_on_timeout_mean</code> | Mean elapsed decisions conditional on hit or timeout. |

### 6.2 First-outcome commanded control

The first-outcome ledger separates correct arrival, wrong-landmark arrival, timeout, censoring, and completion.

<code>intrmotiv/hrl/control/correct_count</code>, <code>wrong_count</code>, <code>timeout_count</code>, <code>censored_count</code>, and <code>completed_count</code> are raw counts in the learner batch. <code>correct_elapsed_mean</code>/<code>wrong_elapsed_mean</code> and the matching reward-magnitude means are conditional means.

For the matched online control contrast, <code>commanded_numerator = correct_count</code> and <code>commanded_event_count = correct_count + wrong_count</code>. The shuffled numerator uses the observed first outcome against a deterministic alternate target formed by rolling the command while avoiding the source; <code>shuffled_event_count</code> has the same outcome-event denominator. Aggregate the numerators and event counts before forming commanded/shuffled success rates.

<code>observed_pair_coverage</code>, <code>normalized_command_entropy</code>, <code>local_candidate_pair_count</code>, <code>local_candidate_source_fraction</code>, <code>local_candidate_count_mean</code>, and <code>behavior_candidate_count_mean</code> diagnose whether the control comparison covers enough source-target pairs.

The older <code>intrmotiv/hrl/target_hit_numerator</code>, <code>target_hit_event_count</code>, <code>shuffled_hit_numerator</code>, and <code>shuffled_hit_event_count</code> are also raw aggregation components. <code>target_hit_lift</code> is a ratio and is unstable when shuffled success is near zero.

### 6.3 Goal conditioning

| Tag | Meaning |
| --- | --- |
| <code>intrmotiv/hrl/goal_condition/target_valid_fraction</code> | Fraction of rows with a valid behavior target for the online counterfactual. |
| <code>intrmotiv/hrl/goal_condition/action_sensitivity</code> | Mean absolute raw-logit change when the target is rolled while state/context is held fixed. |
| <code>intrmotiv/hrl/goal_condition/action_probability_tv</code> | Mean total-variation distance between action distributions under behavior target and rolled target. Prefer this to raw-logit sensitivity. |
| <code>intrmotiv/hrl/goal_condition/value_span</code> | Target-conditioned value variation under the diagnostic counterfactual. |
| <code>intrmotiv/hrl/goal_condition/decoder_parameter_count</code> | Parameter count of the goal-conditioned decoder path. |
| <code>intrmotiv/hrl/goal_condition/film_modulation_norm_mean</code>, <code>film_modulation_norm_max</code> | FiLM modulation magnitude diagnostics; not evidence of successful target control by themselves. |
| <code>intrmotiv/hrl/behavior_replay_mismatch</code> | Behavior-label/recomputed target mismatch diagnostic. |

Branch losses <code>intrmotiv/hrl/branch/{goal,free}_{policy,value,entropy}_loss</code> isolate goal-conditioned and free-policy optimization contributions.

### 6.4 Graph occupancy, reliability, and reachability

<code>node_coverage_fraction</code>, <code>node_visit_weight_mean</code>, and <code>selected_target_visit_mean</code> measure landmark use. <code>known_edge_fraction</code>, <code>forgotten_edge_fraction</code>, <code>known_controllability_time_mean</code>, <code>edge_confidence_mean</code>, and <code>edge_reliability_mean</code> summarize graph evidence.

<code>intrmotiv/hrl/edge/reliability_brier</code> is the calibration error of edge reliability. <code>promotions_per_rollout</code>/<code>demotions_per_rollout</code> and <code>promotion_rate</code>/<code>demotion_rate</code> track threshold crossings.

Reliable-graph structure is summarized by <code>intrmotiv/hrl/reliable/largest_scc</code>, <code>reachable_pair_fraction</code>, <code>outgoing_node_fraction</code>, <code>outgoing_node_count</code>, <code>reciprocal_fraction</code>, and <code>top3_incoming_confidence_share</code>. Online spatial summaries add <code>intrmotiv/hrl/summary/reliable_global_efficiency</code> and <code>grounded_controllability</code>.

### 6.5 Manager modes and by-mode behavior

<code>intrmotiv/hrl/mode/{free,navigate,goal,probe}_fraction</code> gives manager occupancy. For goal/free/probe, <code>intrmotiv/hrl/by_mode/<mode>_{target_hit_rate,time_to_hit,path_length,straightness,loop_fraction}</code> separates behavioral consequences by mode rather than averaging incompatible behaviors.

Manager-exploration tags under <code>intrmotiv/hrl/exploration/*</code> include mode/selection/forced-selection fractions, completion rate, elapsed mean, selected-deadline mean, reward mean, and reward nonzero fraction.

### 6.6 Edge probing, passive graph, frontier, planning, and validation

- <code>intrmotiv/hrl/edge/candidate_fraction</code>, <code>probe_fraction</code>, probe successes/timeouts per rollout, and probe success/timeout rates describe active edge testing.
- <code>intrmotiv/hrl/passive/*</code> reports passive graph updates, known/candidate edge fractions, traversal time/path length, and rejection rates for nonexclusive, time, path, or motion failures.
- <code>intrmotiv/hrl/frontier/*</code> reports frontier score, selection rate, attempts, discoveries, yield, and reached fraction.
- <code>intrmotiv/hrl/planning/*</code> reports route availability, hop count, waypoint success, replanning, and final-frontier reach.
- <code>intrmotiv/hrl/validation/*</code> reports queued edges, return success, validation success, and timeout.

### 6.7 Empirical HER on the PPO path

<code>intrmotiv/her/loss</code>, <code>policy_loss</code>, and <code>value_loss</code> are HER optimization terms. <code>behavior_logprob_ratio</code> and <code>clip_fraction</code> diagnose off-policy mismatch. <code>valid_fraction</code>, <code>accepted_segments_per_rollout</code>, <code>skipped_no_endpoint_per_rollout</code>, <code>skipped_same_source_per_rollout</code>, <code>segment_length</code>, <code>positive_fraction</code>, and <code>terminal_reward</code> describe the relabeled sample population.

## 7. Controller-DDQN and contextual CA3 goals

The event-driven controller writes <code>controller/*</code> into learner summaries; <code>write_intrmotiv_summaries</code> routes them to <code>intrmotiv/controller/*</code>. These are separate from native off-policy <code>ddqn/*</code>.

### 7.1 Optimization and replay

<code>main_loss</code>/<code>auxiliary_loss</code>, <code>q_mean</code>/<code>q_abs_max</code>, and auxiliary Q statistics describe controller TD learning. <code>main_updates</code>, <code>main_td_positions</code>, <code>auxiliary_td_positions</code>, <code>target_age</code>, and <code>update_debt</code> check the exact update budget. <code>stored_state_replay</code> is one when replay stores controller worker states rather than reconstructing them. Compute-cost metrics are <code>main_compute_seconds</code>, <code>her_compute_seconds</code>, <code>her_overhead_ratio</code>, and <code>transaction_seconds</code>.

<code>physical_interactions</code>, <code>physical_frames</code>, and <code>accepted_replay_decisions</code> separate environment exposure from usable replay. <code>fresh_dg_steps</code> and <code>fresh_graph_batches</code> monitor representation/graph freshness. <code>dg_optimizer_steps_{min,max}</code> and <code>main_optimizer_steps_{min,max}</code> catch optimizer-ownership drift.

Actor-memory restart diagnostics are <code>actor_memory_rebuilds</code>, <code>actor_memory_rebuild_seconds</code>, and <code>actor_memory_version_failures</code>. Exact rejection reasons appear as <code>intrmotiv/controller/rejected/&lt;reason&gt;</code>.

### 7.2 Contextual anchor calibration and recognition

<code>active_goal_count</code>, <code>anchor_registrations</code>, <code>confirmation_attempts</code>/<code>confirmation_successes</code>, <code>anchor_replacements</code>, and <code>anchor_deactivations</code> describe the contextual goal set. <code>calibration_ready</code>, <code>recognition_threshold</code>, <code>prediction_absolute_threshold</code>, <code>prediction_excess_threshold</code>, <code>calibration_pair_count</code>, and <code>calibration_seconds</code> describe recognition calibration. <code>activation_latency_mean</code> measures time to recognized activation; <code>empty_set_exploration</code> counts decisions made with no selectable contextual goal.

On the CA3 state-goal follow-up line, anchor refinement adds <code>anchor_refinement_attempts</code>, <code>anchor_refinements</code>, <code>anchor_centrality_gain_mean</code>, and <code>anchor_age_mean</code>.

### 7.3 Context identity quality

<code>context_raw_multi_activation</code> is the raw multi-match pressure before contextual acceptance. <code>context_accepted_events</code>, <code>context_unique_rescues</code>, <code>context_zero_match</code>, and <code>context_multi_match</code> show how often the contextual discriminator accepts, uniquely rescues, misses, or remains ambiguous.

Similarity calibration is summarized by <code>positive_similarity_q10/q50/q90</code> for matched positive contexts and <code>background_similarity_q50/q90/q99</code> for background/counterfactual comparisons. <code>background_above_threshold_fraction</code> measures false-positive pressure at the recognition threshold. <code>active_anchor_collision_fraction</code> measures simultaneous recognition/collision among active anchors.

These quantiles are more informative than one mean because a useful threshold requires separation between the lower tail of positives and upper tail of background.

### 7.4 Contextual HER checks

<code>her_contextual_candidates</code> is the number of contextual HER candidates and <code>her_contextual_positive_hits</code> the accepted contextual positives. <code>her_contextual_positive_rate</code> is positives divided by candidates. <code>her_contextual_wrong_context</code>, <code>her_contextual_start_achieved</code>, <code>her_contextual_missing_calibration</code>, and <code>her_terminal_successor_ca3_missing</code> diagnose why DG-ID matches do not become valid contextual HER positives.

## 8. Native off-policy DDQN/HER

The native recurrent path publishes every runtime metric as <code>ddqn/&lt;key&gt;</code> at <code>ddqn/frames</code> and also writes <code>train/env_steps</code>.

### 8.1 Exposure and update budget

<code>frames</code>, <code>decisions</code>, <code>accepted</code>, <code>updates</code>, <code>invalid_final</code>, <code>attempts</code>, and <code>arrivals</code> are cumulative runtime counters. <code>requested_her_fraction</code> is configuration; <code>realized_her_fraction</code> is sampled HER fraction. <code>decisions_per_update</code>, <code>td_positions_per_update</code>, <code>effective_loss_positions_per_decision</code>, <code>valid_loss_positions_total</code>, <code>her_loss_positions</code>, <code>original_loss_positions</code>, and <code>update_debt</code> verify the exact gradient budget.

<code>inference_decisions</code> is the shared actor decision clock; <code>epsilon</code> is labeled against that clock, not learner ingestion time.

### 8.2 TD learning

<code>td_loss</code> is Smooth-L1 loss between chosen-action online Q and the Double-DQN target over valid masked positions. <code>grad_norm</code> is the clipped pre-step norm returned by gradient clipping. <code>target_copies</code> is completed hard target-network copies. <code>td_target_{min,max,mean}</code>, <code>q_{min,max,mean}</code>, <code>q_out_of_range_fraction</code>, and <code>action_gap_mean</code> diagnose scale and action separation.

### 8.3 Replay, transport, and timing

<code>replay_size</code>, <code>transport_received</code>, <code>transport_pending</code>, <code>ingestion_seconds</code>, <code>learning_seconds</code>, <code>prefix_and_batch_seconds</code>, <code>learner_update_seconds</code>, and <code>throughput_fps</code> separate sampling/transport/reconstruction/learning costs.

### 8.4 Per-goal supervision

The dynamic <code>ddqn/goal_&lt;g&gt;/*</code> family records commanded attempts/arrivals, incidental achieved events/episodes, requested and realized HER, original-versus-HER segment/reward/episode support, remaining-budget histograms, selected HER offsets, per-goal Q prediction/TD-target ranges, and action gaps. These counters establish support and learning coverage; they are not substitutes for independent commanded-versus-shuffled intervention evaluation.

## 9. Compact online spatial telemetry

<code>online_spatial_telemetry=True</code> retains the latest 100,000 valid behavior samples per policy in a fixed ring buffer. Scalar summaries use the latest 10,000 samples at the configured frame cadence; milestone artifacts use the full retained buffer. Privileged pose is removed before model input and is telemetry-only.

All place-field quantities use the configured occupancy-corrected grid. “Active” means positive thresholded DG activity on at least one in-bounds sample.

| Tag | Exact quantity |
| --- | --- |
| <code>intrmotiv/online/place_field/valid_sample_count</code> | Valid behavior samples in the scalar-analysis tail. |
| <code>in_bounds_fraction</code> | Finite retained poses inside configured bounds divided by retained poses. |
| <code>visited_cell_fraction</code> | Occupied grid cells divided by all grid cells. |
| <code>active_unit_fraction</code> | DG units active at least once in bounds divided by DG units. |
| <code>silent_unit_fraction</code> | One minus active-unit fraction. |
| <code>active_unit_mean_spatial_information</code> | Mean occupancy-weighted Skaggs spatial information over active units. |
| <code>active_only_map_cosine</code> | Mean pairwise cosine of active occupancy-corrected maps over visited cells. |
| <code>unique_active_peak_bins</code> | Number of distinct peak grid bins across active units. |
| <code>mono_field_unit_fraction</code> | Eligible units whose dominant 8-connected component contains at least 80% of superlevel mass at 30%, 50%, and 70% of peak, divided by eligible units. |
| <code>mean_primary_secondary_peak_distance</code> | Mean physical distance between the two highest-mass 50%-of-peak components for units with both. |
| <code>median_dominant_peak_nearest_neighbor_distance</code> | Median nearest-neighbor distance between dominant peaks of eligible units. |
| <code>intrmotiv/online/trajectory/mean_physical_step_distance</code> | Mean Euclidean displacement over valid within-segment transitions. |
| <code>stationary_step_fraction</code> | Fraction of those displacements at or below the configured stationary threshold. |
| <code>path_efficiency</code> | Sum of segment endpoint displacement divided by sum of within-segment path length. |
| <code>mean_absolute_circular_yaw_change</code> | Mean absolute yaw delta wrapped to <code>[-180,180)</code> degrees. |
| <code>intrmotiv/online/window/target_env_steps</code> | Frame target associated with the scalar telemetry window. |

<code>intrmotiv/hrl/summary/reliable_global_efficiency</code> is mean reciprocal directed shortest-path hop count over ordered unit pairs, with unreachable pairs contributing zero. <code>grounded_controllability</code> multiplies pre-update reliable-edge prospective success by the fraction of reliable edges connecting spatially eligible mono-field units.

## 10. Environment exploration telemetry

These are environment statistics, not learner minibatch statistics. Fixed-length runs typically expose physical-episode summaries under <code>policy_stats/avg_z_...</code>. Long-episode runs can also emit <code>intrmotiv/exploration/window/*</code> on a fixed decision window without resetting DMLab, CA3, option state, or graph memory.

| Suffix | Exact quantity |
| --- | --- |
| <code>coverage_unique_cells</code> | Distinct discretized cells in the interval. |
| <code>coverage_auc</code> | Mean over time of cumulative unique-cell count; compare equal-length intervals only. |
| <code>coverage_entropy</code> | Entropy of discretized-cell occupancy under the configured normalization. |
| <code>pose_unique_bins</code> | Distinct joint position/yaw bins. |
| <code>pose_auc</code> | Mean over time of cumulative joint pose-bin count. |
| <code>pose_entropy</code> | Entropy of joint position/yaw occupancy. |
| <code>window_return</code> / episode return | External DMLab reward sum. |
| <code>window_length</code> / episode length | Decisions in the interval; validate before comparing totals. |

Coverage is not control. A random or circular policy can cover the arena while ignoring targets.

## 11. Path, geometry, predictor, and PBT utilities

<code>intrmotiv/path/path_length_mean</code>, <code>displacement_mean</code>, and <code>straightness_mean</code> summarize path geometry; <code>scatter_loss</code> and <code>scatter_conflict_fraction</code> belong to the DG path-separation objective.

<code>intrmotiv/geometry/se2_stress</code>, <code>valid_landmark_fraction</code>, and <code>proposed_edge_fraction</code> describe geometry-fitting/proposal diagnostics.

The optional CA3 outcome predictor logs <code>intrmotiv/predictor/loss</code>, <code>hit_accuracy</code>, <code>hit_time_mae</code>, and <code>positive_fraction</code>.

When PBT routing is enabled, <code>intrmotiv/pbt/hrl_validity</code> is the binary validity gate used for the PBT objective and <code>intrmotiv/pbt/objective</code> is coverage when valid, otherwise zero. PBT metrics are routing diagnostics, not a scientific endpoint.

## 12. Recommended dashboard panels

1. **Representation health:** DG density, silent fraction, duty-cycle min/max, usage entropy, pre-threshold-above fraction, online mono-field fraction, active map cosine.
2. **Goal representation:** CA3 readout prediction/active/zero losses, state/action shuffle deltas, latent std mean/min when available, contextual positive/background similarity quantiles.
3. **Intrinsic supervision:** advantage and intrinsic nonzero fractions, target-hit rate, option timeout rate, option success fraction, encoder dominant-event count, encoder-credit replay match.
4. **Causal control:** commanded/shuffled numerators and event counts, action-probability TV, first-outcome correct/wrong counts, pair coverage and command entropy.
5. **Controller health:** main/auxiliary TD loss, Q scale, update debt, main/auxiliary TD positions, contextual HER positive rate, recognition thresholds, anchor activity.
6. **External behavior:** coverage AUC/unique cells, pose AUC, path efficiency, stationary fraction, and by-mode target hit/time-to-hit.
7. **Transfer/off-policy:** per-goal DDQN attempts/arrivals/support plus TD loss and action gap; keep these separate from independent transfer evaluation.

No single internal metric is a sufficient run objective. Representation health, goal identity, learning support, causal target dependence, and external behavior must agree.

## 13. Complete exact tag inventory

The tables below are generated from the current summary-router dictionaries during this audit. “Current” means present on <code>codex/fixed-reward-transfer-corrected-20260925</code>. “CA3 follow-up only” marks the five readout anti-collapse tags present on <code>codex/ca3-state-goal-followup-20260922</code> but not on the later transfer line. The producer key is the learner-stat name mapped into the final TensorBoard/W&B tag.

### CA3 memory / event gate

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/memory/familiar_onset_fraction` | `memory_familiar_onset` | current |
| `intrmotiv/memory/gate_violation_fraction` | `memory_gate_violation` | current |
| `intrmotiv/memory/goal_active_fraction` | `memory_goal_active` | current |
| `intrmotiv/memory/goal_ambiguous_fraction` | `memory_goal_ambiguous` | current |
| `intrmotiv/memory/goal_hit_fraction` | `memory_goal_hit` | current |
| `intrmotiv/memory/novel_onset_fraction` | `memory_novel_onset` | current |
| `intrmotiv/memory/onset_fraction` | `memory_onset` | current |
| `intrmotiv/memory/replay_mismatch` | `memory_replay_mismatch` | current |

### CA3 progression distance

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/distance/masked_max` | `distance_metric_masked_max` | current |
| `intrmotiv/distance/masked_mean` | `distance_metric_masked` | current |
| `intrmotiv/distance/masked_min` | `distance_metric_masked_min` | current |
| `intrmotiv/distance/masked_std` | `distance_metric_masked_std` | current |
| `intrmotiv/distance/max` | `distance_metric_max` | current |
| `intrmotiv/distance/mean` | `distance_metric` | current |
| `intrmotiv/distance/min` | `distance_metric_min` | current |
| `intrmotiv/distance/std` | `distance_metric_std` | current |

### CA3 state readout

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/ca3_readout/action_shuffle_delta` | `ca3_readout_action_shuffle_delta` | current |
| `intrmotiv/ca3_readout/active_fraction` | `ca3_readout_active_fraction` | current |
| `intrmotiv/ca3_readout/active_loss` | `ca3_readout_active_loss` | current |
| `intrmotiv/ca3_readout/cov_loss` | `ca3_readout_cov_loss` | CA3 follow-up only |
| `intrmotiv/ca3_readout/enabled` | `ca3_readout_enabled` | current |
| `intrmotiv/ca3_readout/latent_std_mean` | `ca3_readout_latent_std_mean` | CA3 follow-up only |
| `intrmotiv/ca3_readout/latent_std_min` | `ca3_readout_latent_std_min` | CA3 follow-up only |
| `intrmotiv/ca3_readout/prediction_loss` | `ca3_readout_prediction_loss` | current |
| `intrmotiv/ca3_readout/state_shuffle_delta` | `ca3_readout_state_shuffle_delta` | current |
| `intrmotiv/ca3_readout/total_loss` | `ca3_readout_total_loss` | CA3 follow-up only |
| `intrmotiv/ca3_readout/valid_targets` | `ca3_readout_valid_targets` | current |
| `intrmotiv/ca3_readout/var_loss` | `ca3_readout_var_loss` | CA3 follow-up only |
| `intrmotiv/ca3_readout/zero_loss` | `ca3_readout_zero_loss` | current |

### Contextual DG

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/dg/context/adapter_gradient_norm` | `dg_context_adapter_gradient_norm` | current |
| `intrmotiv/dg/context/created_fraction` | `dg_context_created_fraction` | current |
| `intrmotiv/dg/context/modulation_abs_mean` | `dg_context_modulation_abs_mean` | current |
| `intrmotiv/dg/context/modulation_saturation_fraction` | `dg_context_modulation_saturation_fraction` | current |
| `intrmotiv/dg/context/suppressed_fraction` | `dg_context_suppressed_fraction` | current |
| `intrmotiv/dg/context/unchanged_fraction` | `dg_context_unchanged_fraction` | current |

### DG activity and usage

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/dg/active_count` | `activated_sequences` | current |
| `intrmotiv/dg/active_count_max` | `activated_sequences_max` | current |
| `intrmotiv/dg/active_count_min` | `activated_sequences_min` | current |
| `intrmotiv/dg/active_count_std` | `activated_sequences_std` | current |
| `intrmotiv/dg/behavior_dominant_event_fraction` | `dg_behavior_dominant_event_fraction` | current |
| `intrmotiv/dg/behavior_multi_onset_event_fraction` | `dg_behavior_multi_onset_event_fraction` | current |
| `intrmotiv/dg/behavior_non_dominant_onsets_per_event` | `dg_behavior_non_dominant_onsets_per_event` | current |
| `intrmotiv/dg/ca3_conflict_activity` | `dg_ca3_conflict_activity` | current |
| `intrmotiv/dg/ca3_conflict_fraction` | `dg_ca3_conflict_fraction` | current |
| `intrmotiv/dg/ca3_conflicting_activation_fraction` | `dg_ca3_conflicting_activation_fraction` | current |
| `intrmotiv/dg/density` | `dg_density` | current |
| `intrmotiv/dg/learner_active_transition_fraction` | `dg_learner_active_transition_fraction` | current |
| `intrmotiv/dg/multi_activation_fraction` | `dg_multi_activation_rate` | current |
| `intrmotiv/dg/pre_threshold_above_fraction` | `dg_pre_threshold_above_fraction` | current |
| `intrmotiv/dg/pre_threshold_mean` | `dg_pre_threshold_mean` | current |
| `intrmotiv/dg/silent_unit_fraction` | `dg_silent_unit_frac` | current |
| `intrmotiv/dg/unit_duty_cycle_max` | `dg_unit_duty_cycle_max` | current |
| `intrmotiv/dg/unit_duty_cycle_mean` | `dg_unit_duty_cycle_mean` | current |
| `intrmotiv/dg/unit_duty_cycle_min` | `dg_unit_duty_cycle_min` | current |
| `intrmotiv/dg/usage_entropy` | `dg_usage_entropy` | current |
| `intrmotiv/dg/valid_minibatch_unused_unit_count` | `dg_valid_minibatch_unused_unit_count` | current |
| `intrmotiv/dg/valid_minibatch_unused_unit_fraction` | `dg_valid_minibatch_unused_unit_fraction` | current |

### DG gradient routing

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/dg/gradient/cosine` | `dg_gradient_cosine` | current |
| `intrmotiv/dg/gradient/encoder_norm` | `dg_encoder_gradient_norm` | current |
| `intrmotiv/dg/gradient/ppo_encoder_ratio` | `dg_gradient_norm_ratio` | current |
| `intrmotiv/dg/gradient/ppo_norm` | `dg_ppo_gradient_norm` | current |
| `intrmotiv/dg/gradient/row_conflict_fraction` | `dg_gradient_row_conflict_fraction` | current |

### DG normalization

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/dg/normalization/normalized_logit_mean_abs` | `dg_normalized_logit_mean_abs` | current |
| `intrmotiv/dg/normalization/normalized_logit_variance_error` | `dg_normalized_logit_variance_error` | current |
| `intrmotiv/dg/normalization/raw_logit_mean_abs` | `dg_raw_logit_mean_abs` | current |
| `intrmotiv/dg/normalization/raw_logit_variance_mean` | `dg_raw_logit_variance_mean` | current |

### DG recruitment and retirement

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/dg/recruitment/active_endpoint_count` | `dg_recruitment_active_endpoint_count` | current |
| `intrmotiv/dg/recruitment/activity_blocked_count` | `dg_recruitment_activity_blocked_count` | current |
| `intrmotiv/dg/recruitment/attempt_coverage_fraction` | `dg_recruitment_attempt_coverage_fraction` | current |
| `intrmotiv/dg/recruitment/bad_source_assignments_per_rollout` | `dg_recruitment_bad_source_assignment_count` | current |
| `intrmotiv/dg/recruitment/bad_source_count` | `dg_recruitment_bad_source_count` | current |
| `intrmotiv/dg/recruitment/birth_protected_count` | `dg_recruitment_birth_protected_count` | current |
| `intrmotiv/dg/recruitment/candidate_count` | `dg_recruitment_candidate_count` | current |
| `intrmotiv/dg/recruitment/committed_fraction` | `dg_recruitment_committed_fraction` | current |
| `intrmotiv/dg/recruitment/connected_fraction` | `dg_recruitment_connected_fraction` | current |
| `intrmotiv/dg/recruitment/eligible_bad_source_endpoint_count` | `dg_recruitment_eligible_bad_source_endpoint_count` | current |
| `intrmotiv/dg/recruitment/eligible_predictive_endpoint_count` | `dg_recruitment_eligible_predictive_endpoint_count` | current |
| `intrmotiv/dg/recruitment/eligible_redundant_endpoint_count` | `dg_recruitment_eligible_redundant_endpoint_count` | current |
| `intrmotiv/dg/recruitment/eligible_vertex_count` | `dg_recruitment_eligible_vertex_count` | current |
| `intrmotiv/dg/recruitment/eligible_victim_endpoint_count` | `dg_recruitment_eligible_victim_endpoint_count` | current |
| `intrmotiv/dg/recruitment/endpoint_active_unit_count` | `dg_recruitment_endpoint_active_unit_count` | current |
| `intrmotiv/dg/recruitment/endpoint_active_unit_mean` | `dg_recruitment_endpoint_active_unit_mean` | current |
| `intrmotiv/dg/recruitment/forced_preflight_assignments_per_rollout` | `dg_recruitment_forced_preflight_assignment_count` | current |
| `intrmotiv/dg/recruitment/fully_tested_count` | `dg_recruitment_fully_tested_count` | current |
| `intrmotiv/dg/recruitment/goal_adapter_reset_total` | `dg_recruitment_goal_adapter_reset_total` | current |
| `intrmotiv/dg/recruitment/goal_adapter_resets_per_rollout` | `dg_recruitment_goal_adapter_reset_count` | current |
| `intrmotiv/dg/recruitment/isolated_assignments_per_rollout` | `dg_recruitment_isolated_assignment_count` | current |
| `intrmotiv/dg/recruitment/isolated_fraction` | `dg_recruitment_isolated_fraction` | current |
| `intrmotiv/dg/recruitment/passive_graph_density` | `dg_recruitment_passive_graph_density` | current |
| `intrmotiv/dg/recruitment/passive_over_gap_per_rollout` | `dg_recruitment_passive_over_gap_count` | current |
| `intrmotiv/dg/recruitment/passive_stale_per_rollout` | `dg_recruitment_passive_stale_count` | current |
| `intrmotiv/dg/recruitment/passive_updates_per_rollout` | `dg_recruitment_passive_update_count` | current |
| `intrmotiv/dg/recruitment/predictive_assignments_per_rollout` | `dg_recruitment_predictive_assignment_count` | current |
| `intrmotiv/dg/recruitment/predictive_context_coverage_fraction` | `dg_recruitment_predictive_context_coverage_fraction` | current |
| `intrmotiv/dg/recruitment/predictive_context_group_count` | `dg_recruitment_predictive_context_group_count` | current |
| `intrmotiv/dg/recruitment/predictive_decayed_attempt_mass` | `dg_recruitment_predictive_decayed_attempt_mass` | current |
| `intrmotiv/dg/recruitment/predictive_eligible_count` | `dg_recruitment_predictive_eligible_count` | current |
| `intrmotiv/dg/recruitment/predictive_event_count` | `dg_recruitment_predictive_event_count` | current |
| `intrmotiv/dg/recruitment/predictive_invalidation_mass` | `dg_recruitment_predictive_invalidation_mass` | current |
| `intrmotiv/dg/recruitment/predictive_reliability_gap` | `dg_recruitment_predictive_reliability_gap` | current |
| `intrmotiv/dg/recruitment/predictive_supported_context_count` | `dg_recruitment_predictive_supported_context_count` | current |
| `intrmotiv/dg/recruitment/redundant_assignments_per_rollout` | `dg_recruitment_redundant_assignment_count` | current |
| `intrmotiv/dg/recruitment/redundant_pair_count` | `dg_recruitment_redundant_pair_count` | current |
| `intrmotiv/dg/recruitment/reliable_out_degree_mean` | `dg_recruitment_reliable_out_degree_mean` | current |
| `intrmotiv/dg/recruitment/reliable_outgoing_confidence_mean` | `dg_recruitment_reliable_outgoing_confidence_mean` | current |
| `intrmotiv/dg/recruitment/repeat_assignments_per_rollout` | `dg_recruitment_repeat_assignment_count` | current |
| `intrmotiv/dg/recruitment/repeat_total` | `dg_recruitment_repeat_total` | current |
| `intrmotiv/dg/recruitment/replacement_conversion` | `dg_recruitment_replacement_conversion` | current |
| `intrmotiv/dg/recruitment/residual_norm` | `dg_recruitment_residual_norm` | current |
| `intrmotiv/dg/recruitment/residual_pass_count` | `dg_recruitment_residual_pass_count` | current |
| `intrmotiv/dg/recruitment/residual_reject_count` | `dg_recruitment_residual_reject_count` | current |
| `intrmotiv/dg/recruitment/rollout_count` | `dg_recruitment_rollout_count` | current |
| `intrmotiv/dg/recruitment/silent_endpoint_count` | `dg_recruitment_silent_endpoint_count` | current |
| `intrmotiv/dg/recruitment/tiny_residual_total` | `dg_recruitment_tiny_residual_total` | current |
| `intrmotiv/dg/recruitment/total` | `dg_recruitment_total` | current |
| `intrmotiv/dg/recruitment/untested_zero_outdegree_count` | `dg_recruitment_untested_zero_outdegree_count` | current |
| `intrmotiv/dg/recruitment/victim_active_bad_source_count` | `dg_recruitment_victim_active_bad_source_count` | current |
| `intrmotiv/dg/recruitment/victim_active_bad_source_fraction` | `dg_recruitment_victim_active_bad_source_fraction` | current |
| `intrmotiv/dg/recruitment/victim_active_count` | `dg_recruitment_victim_active_count` | current |
| `intrmotiv/dg/recruitment/victim_active_fraction` | `dg_recruitment_victim_active_fraction` | current |
| `intrmotiv/dg/recruitment/victim_active_predictive_count` | `dg_recruitment_victim_active_predictive_count` | current |
| `intrmotiv/dg/recruitment/victim_active_predictive_fraction` | `dg_recruitment_victim_active_predictive_fraction` | current |
| `intrmotiv/dg/recruitment/victim_active_redundant_count` | `dg_recruitment_victim_active_redundant_count` | current |
| `intrmotiv/dg/recruitment/victim_active_redundant_fraction` | `dg_recruitment_victim_active_redundant_fraction` | current |
| `intrmotiv/dg/recruitment/zero_outdegree_count` | `dg_recruitment_zero_outdegree_count` | current |

### DG transition prediction

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/dg/prediction/applied_count` | `dg_transition_prediction_applied_count` | current |
| `intrmotiv/dg/prediction/boundary_drop_count` | `dg_transition_prediction_boundary_drop_count` | current |
| `intrmotiv/dg/prediction/control_loss` | `dg_transition_prediction_control_loss` | current |
| `intrmotiv/dg/prediction/loss` | `dg_transition_prediction_loss` | current |
| `intrmotiv/dg/prediction/main_loss` | `dg_transition_prediction_main_loss` | current |
| `intrmotiv/dg/prediction/replay_match` | `dg_transition_prediction_replay_match` | current |
| `intrmotiv/dg/prediction/scheduled_count` | `dg_transition_prediction_scheduled_count` | current |
| `intrmotiv/dg/prediction/validation_accuracy` | `dg_transition_prediction_validation_accuracy` | current |
| `intrmotiv/dg/prediction/validation_control_ce` | `dg_transition_prediction_validation_control_ce` | current |
| `intrmotiv/dg/prediction/validation_count` | `dg_transition_prediction_validation_count` | current |
| `intrmotiv/dg/prediction/validation_main_ce` | `dg_transition_prediction_validation_main_ce` | current |
| `intrmotiv/dg/prediction/validation_state_gain` | `dg_transition_prediction_validation_state_gain` | current |

### DG update contract and replay barrier

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/dg/update_contract/forward_count` | `dg_forward_count` | current |
| `intrmotiv/dg/update_contract/publication_generation_mismatch` | `dg_publication_generation_mismatch` | current |
| `intrmotiv/dg/update_contract/running_stats_update_count` | `dg_running_stats_update_count` | current |
| `intrmotiv/dg/update_contract/statistics_generation` | `dg_statistics_generation` | current |
| `intrmotiv/dg/update_contract/weight_generation` | `dg_weight_generation` | current |

### Decoder objectives

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/decoder/auxiliary_loss` | `extra_decoder_loss` | current |
| `intrmotiv/decoder/loss` | `decoder_loss` | current |

### Empirical HER

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/her/accepted_segments_per_rollout` | `empirical_her_accepted_segments` | current |
| `intrmotiv/her/behavior_logprob_ratio` | `empirical_her_behavior_ratio` | current |
| `intrmotiv/her/clip_fraction` | `empirical_her_clip_fraction` | current |
| `intrmotiv/her/loss` | `empirical_her_loss` | current |
| `intrmotiv/her/policy_loss` | `empirical_her_policy_loss` | current |
| `intrmotiv/her/positive_fraction` | `empirical_her_positive_fraction` | current |
| `intrmotiv/her/segment_length` | `empirical_her_segment_length` | current |
| `intrmotiv/her/skipped_no_endpoint_per_rollout` | `empirical_her_skipped_no_endpoint` | current |
| `intrmotiv/her/skipped_same_source_per_rollout` | `empirical_her_skipped_same_source` | current |
| `intrmotiv/her/terminal_reward` | `empirical_her_terminal_reward` | current |
| `intrmotiv/her/valid_fraction` | `empirical_her_valid_fraction` | current |
| `intrmotiv/her/value_loss` | `empirical_her_value_loss` | current |

### Encoder objectives

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/encoder/batch_usage_loss` | `batch_reward_loss` | current |
| `intrmotiv/encoder/ca3_temporal_exclusion_loss` | `encoder_ca3_temporal_exclusion_loss` | current |
| `intrmotiv/encoder/collision_loss` | `encoder_collision_loss` | current |
| `intrmotiv/encoder/density_loss` | `encoder_density_loss` | current |
| `intrmotiv/encoder/dominant_event_count` | `encoder_dominant_event_count` | current |
| `intrmotiv/encoder/feedback_abs_on_dominant_event_mean` | `encoder_feedback_abs_on_dominant_event_mean` | current |
| `intrmotiv/encoder/feedback_mean` | `encoder_punishment` | current |
| `intrmotiv/encoder/feedback_on_dominant_event_mean` | `encoder_feedback_on_dominant_event_mean` | current |
| `intrmotiv/encoder/global_punishment_loss` | `encoder_global_punishment_loss` | current |
| `intrmotiv/encoder/loss` | `encoder_loss` | current |
| `intrmotiv/encoder/multi_activation_loss` | `encoder_penalty_loss` | current |
| `intrmotiv/encoder/population_loss` | `encoder_population_loss` | current |
| `intrmotiv/encoder/row_repulsion_loss` | `encoder_row_repulsion_loss` | current |
| `intrmotiv/encoder/unused_sequence_loss` | `encoder_reward_loss` | current |
| `intrmotiv/encoder/usage_loss` | `encoder_usage_loss` | current |

### Encoder temporal credit

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/encoder/credit/alignment_failures` | `encoder_credit_alignment_failure` | current |
| `intrmotiv/encoder/credit/applied_count` | `encoder_credit_applied_count` | current |
| `intrmotiv/encoder/credit/applied_reward_mass` | `encoder_credit_applied_mass` | current |
| `intrmotiv/encoder/credit/arrival_loss` | `encoder_arrival_credit_loss` | current |
| `intrmotiv/encoder/credit/boundary_dropped_events` | `encoder_credit_boundary_dropped` | current |
| `intrmotiv/encoder/credit/collisions` | `encoder_credit_collisions` | current |
| `intrmotiv/encoder/credit/credited_events` | `encoder_credit_credited` | current |
| `intrmotiv/encoder/credit/credited_row_count` | `encoder_credited_row_count` | current |
| `intrmotiv/encoder/credit/invalid_intervals` | `encoder_credit_invalid_interval` | current |
| `intrmotiv/encoder/credit/matchable_events` | `encoder_credit_matchable` | current |
| `intrmotiv/encoder/credit/replay_match` | `encoder_credit_replay_match` | current |
| `intrmotiv/encoder/credit/reward_mass` | `encoder_credit_reward_mass` | current |
| `intrmotiv/encoder/credit/scheduled_count` | `encoder_credit_scheduled_count` | current |
| `intrmotiv/encoder/credit/scheduled_reward_mass` | `encoder_credit_scheduled_mass` | current |
| `intrmotiv/encoder/credit/source_lag_max` | `encoder_credit_source_lag_max` | current |
| `intrmotiv/encoder/credit/source_lag_mean` | `encoder_credit_source_lag_mean` | current |
| `intrmotiv/encoder/credit/source_loss` | `encoder_source_credit_loss` | current |
| `intrmotiv/encoder/credit/total_events` | `encoder_credit_total` | current |

### Geometry

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/geometry/proposed_edge_fraction` | `geometry_proposed_edge_fraction` | current |
| `intrmotiv/geometry/se2_stress` | `geometry_se2_stress` | current |
| `intrmotiv/geometry/valid_landmark_fraction` | `geometry_valid_landmark_fraction` | current |

### HRL behavior by mode

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/hrl/by_mode/free_loop_fraction` | `hrl_free_loop_fraction` | current |
| `intrmotiv/hrl/by_mode/free_path_length` | `hrl_free_path_length` | current |
| `intrmotiv/hrl/by_mode/free_straightness` | `hrl_free_straightness` | current |
| `intrmotiv/hrl/by_mode/free_target_hit_rate` | `hrl_free_target_hit_rate` | current |
| `intrmotiv/hrl/by_mode/free_time_to_hit` | `hrl_free_time_to_hit` | current |
| `intrmotiv/hrl/by_mode/goal_loop_fraction` | `hrl_goal_loop_fraction` | current |
| `intrmotiv/hrl/by_mode/goal_path_length` | `hrl_goal_path_length` | current |
| `intrmotiv/hrl/by_mode/goal_straightness` | `hrl_goal_straightness` | current |
| `intrmotiv/hrl/by_mode/goal_target_hit_rate` | `hrl_goal_target_hit_rate` | current |
| `intrmotiv/hrl/by_mode/goal_time_to_hit` | `hrl_goal_time_to_hit` | current |
| `intrmotiv/hrl/by_mode/probe_loop_fraction` | `hrl_probe_loop_fraction` | current |
| `intrmotiv/hrl/by_mode/probe_path_length` | `hrl_probe_path_length` | current |
| `intrmotiv/hrl/by_mode/probe_straightness` | `hrl_probe_straightness` | current |
| `intrmotiv/hrl/by_mode/probe_target_hit_rate` | `hrl_probe_target_hit_rate` | current |
| `intrmotiv/hrl/by_mode/probe_time_to_hit` | `hrl_probe_time_to_hit` | current |

### HRL branch losses

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/hrl/branch/free_entropy_loss` | `hrl_free_entropy_loss` | current |
| `intrmotiv/hrl/branch/free_policy_loss` | `hrl_free_policy_loss` | current |
| `intrmotiv/hrl/branch/free_value_loss` | `hrl_free_value_loss` | current |
| `intrmotiv/hrl/branch/goal_entropy_loss` | `hrl_goal_entropy_loss` | current |
| `intrmotiv/hrl/branch/goal_policy_loss` | `hrl_goal_policy_loss` | current |
| `intrmotiv/hrl/branch/goal_value_loss` | `hrl_goal_value_loss` | current |

### HRL edge reliability / probing

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/hrl/edge/candidate_fraction` | `hrl_edge_candidate_rate` | current |
| `intrmotiv/hrl/edge/demotion_rate` | `hrl_edge_demotion_rate` | current |
| `intrmotiv/hrl/edge/demotions_per_rollout` | `hrl_edge_demotions` | current |
| `intrmotiv/hrl/edge/probe_fraction` | `hrl_edge_probe_rate` | current |
| `intrmotiv/hrl/edge/probe_success_rate` | `hrl_edge_probe_success_rate` | current |
| `intrmotiv/hrl/edge/probe_successes_per_rollout` | `hrl_edge_probe_successes` | current |
| `intrmotiv/hrl/edge/probe_timeout_rate` | `hrl_edge_probe_timeout_rate` | current |
| `intrmotiv/hrl/edge/probe_timeouts_per_rollout` | `hrl_edge_probe_timeouts` | current |
| `intrmotiv/hrl/edge/promotion_rate` | `hrl_edge_promotion_rate` | current |
| `intrmotiv/hrl/edge/promotions_per_rollout` | `hrl_edge_promotions` | current |
| `intrmotiv/hrl/edge/reliability_brier` | `hrl_edge_reliability_brier` | current |

### HRL first-outcome control

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/hrl/control/behavior_candidate_count_mean` | `hrl_control_behavior_candidate_count_mean` | current |
| `intrmotiv/hrl/control/censored_count` | `hrl_control_censored_count` | current |
| `intrmotiv/hrl/control/commanded_event_count` | `hrl_first_outcome_commanded_event_count` | current |
| `intrmotiv/hrl/control/commanded_numerator` | `hrl_first_outcome_commanded_numerator` | current |
| `intrmotiv/hrl/control/completed_count` | `hrl_control_completed_count` | current |
| `intrmotiv/hrl/control/correct_count` | `hrl_control_correct_count` | current |
| `intrmotiv/hrl/control/correct_elapsed_mean` | `hrl_control_correct_elapsed_mean` | current |
| `intrmotiv/hrl/control/correct_reward_magnitude_mean` | `hrl_control_reward_magnitude_correct_mean` | current |
| `intrmotiv/hrl/control/local_candidate_count_mean` | `hrl_control_local_candidate_count_mean` | current |
| `intrmotiv/hrl/control/local_candidate_pair_count` | `hrl_control_local_candidate_pair_count` | current |
| `intrmotiv/hrl/control/local_candidate_source_fraction` | `hrl_control_local_candidate_source_fraction` | current |
| `intrmotiv/hrl/control/normalized_command_entropy` | `hrl_control_command_entropy` | current |
| `intrmotiv/hrl/control/observed_pair_coverage` | `hrl_control_observed_pair_coverage` | current |
| `intrmotiv/hrl/control/shuffled_event_count` | `hrl_first_outcome_shuffled_event_count` | current |
| `intrmotiv/hrl/control/shuffled_numerator` | `hrl_first_outcome_shuffled_numerator` | current |
| `intrmotiv/hrl/control/timeout_count` | `hrl_control_timeout_count` | current |
| `intrmotiv/hrl/control/wrong_count` | `hrl_control_wrong_count` | current |
| `intrmotiv/hrl/control/wrong_elapsed_mean` | `hrl_control_wrong_elapsed_mean` | current |
| `intrmotiv/hrl/control/wrong_reward_magnitude_mean` | `hrl_control_reward_magnitude_wrong_mean` | current |

### HRL frontier

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/hrl/frontier/attempts_per_rollout` | `hrl_frontier_attempts` | current |
| `intrmotiv/hrl/frontier/discoveries_per_rollout` | `hrl_frontier_discoveries` | current |
| `intrmotiv/hrl/frontier/reached_fraction` | `hrl_frontier_reached_fraction` | current |
| `intrmotiv/hrl/frontier/score_mean` | `hrl_frontier_score_mean` | current |
| `intrmotiv/hrl/frontier/selection_rate` | `hrl_frontier_selection_rate` | current |
| `intrmotiv/hrl/frontier/yield` | `hrl_frontier_yield` | current |

### HRL goal conditioning

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/hrl/goal_condition/action_probability_tv` | `hrl_goal_condition_action_probability_tv` | current |
| `intrmotiv/hrl/goal_condition/action_sensitivity` | `hrl_goal_condition_action_sensitivity` | current |
| `intrmotiv/hrl/goal_condition/decoder_parameter_count` | `hrl_goal_decoder_parameter_count` | current |
| `intrmotiv/hrl/goal_condition/film_modulation_norm_max` | `hrl_goal_condition_film_modulation_norm_max` | current |
| `intrmotiv/hrl/goal_condition/film_modulation_norm_mean` | `hrl_goal_condition_film_modulation_norm_mean` | current |
| `intrmotiv/hrl/goal_condition/target_valid_fraction` | `hrl_goal_condition_target_valid_fraction` | current |
| `intrmotiv/hrl/goal_condition/value_span` | `hrl_goal_condition_value_span` | current |

### HRL manager mode occupancy

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/hrl/mode/free_fraction` | `hrl_mode_free_fraction` | current |
| `intrmotiv/hrl/mode/goal_fraction` | `hrl_mode_goal_fraction` | current |
| `intrmotiv/hrl/mode/navigate_fraction` | `hrl_mode_navigate_fraction` | current |
| `intrmotiv/hrl/mode/probe_fraction` | `hrl_mode_probe_fraction` | current |

### HRL options and graph

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/hrl/active_option_fraction` | `hrl_active_option_frac` | current |
| `intrmotiv/hrl/active_target_fraction` | `hrl_active_target_frac` | current |
| `intrmotiv/hrl/behavior_replay_mismatch` | `hrl_behavior_replay_mismatch` | current |
| `intrmotiv/hrl/deadline_selection_fraction` | `hrl_deadline_selection_fraction` | current |
| `intrmotiv/hrl/edge_confidence_mean` | `hrl_edge_confidence_mean` | current |
| `intrmotiv/hrl/edge_reliability_mean` | `hrl_edge_reliability_mean` | current |
| `intrmotiv/hrl/elapsed_on_hit_mean` | `hrl_elapsed_on_hit_mean` | current |
| `intrmotiv/hrl/elapsed_on_timeout_mean` | `hrl_elapsed_on_timeout_mean` | current |
| `intrmotiv/hrl/exploration/completion_rate` | `hrl_exploration_completion_rate` | current |
| `intrmotiv/hrl/exploration/elapsed_mean` | `hrl_exploration_elapsed_mean` | current |
| `intrmotiv/hrl/exploration/forced_selection_fraction` | `hrl_forced_exploration_fraction` | current |
| `intrmotiv/hrl/exploration/mode_fraction` | `hrl_exploration_mode_fraction` | current |
| `intrmotiv/hrl/exploration/reward_mean` | `hrl_exploration_reward_mean` | current |
| `intrmotiv/hrl/exploration/reward_nonzero_fraction` | `hrl_exploration_reward_nonzero_fraction` | current |
| `intrmotiv/hrl/exploration/selected_deadline_mean` | `hrl_exploration_selected_deadline_mean` | current |
| `intrmotiv/hrl/exploration/selection_fraction` | `hrl_exploration_selection_fraction` | current |
| `intrmotiv/hrl/forgotten_edge_fraction` | `hrl_forgotten_edge_fraction` | current |
| `intrmotiv/hrl/known_controllability_time_mean` | `hrl_known_controllability_time_mean` | current |
| `intrmotiv/hrl/known_edge_fraction` | `hrl_known_edge_fraction` | current |
| `intrmotiv/hrl/learned_deadline_fraction` | `hrl_learned_deadline_fraction` | current |
| `intrmotiv/hrl/node_coverage_fraction` | `hrl_node_coverage_fraction` | current |
| `intrmotiv/hrl/node_visit_weight_mean` | `hrl_node_visit_weight_mean` | current |
| `intrmotiv/hrl/option_reset_rate` | `hrl_option_reset_rate` | current |
| `intrmotiv/hrl/option_success_fraction` | `hrl_option_success_fraction` | current |
| `intrmotiv/hrl/option_timeout_rate` | `hrl_option_timeout_rate` | current |
| `intrmotiv/hrl/selected_deadline_mean` | `hrl_selected_deadline_mean` | current |
| `intrmotiv/hrl/selected_deadline_positive_mean` | `hrl_selected_deadline_positive_mean` | current |
| `intrmotiv/hrl/selected_target_visit_mean` | `hrl_selected_target_visit_mean` | current |
| `intrmotiv/hrl/shuffled_hit_event_count` | `hrl_shuffled_hit_event_count` | current |
| `intrmotiv/hrl/shuffled_hit_numerator` | `hrl_shuffled_hit_numerator` | current |
| `intrmotiv/hrl/source_fraction` | `hrl_source_frac` | current |
| `intrmotiv/hrl/target_hit_event_count` | `hrl_target_hit_event_count` | current |
| `intrmotiv/hrl/target_hit_lift` | `hrl_target_hit_lift` | current |
| `intrmotiv/hrl/target_hit_numerator` | `hrl_target_hit_numerator` | current |
| `intrmotiv/hrl/target_hit_rate` | `hrl_target_hit_rate` | current |
| `intrmotiv/hrl/target_selected_deadline_mean` | `hrl_target_selected_deadline_mean` | current |
| `intrmotiv/hrl/tctrl_update_rate` | `hrl_tctrl_update_rate` | current |

### HRL passive graph

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/hrl/passive/candidate_edge_fraction` | `hrl_passive_candidate_edge_fraction` | current |
| `intrmotiv/hrl/passive/known_edge_fraction` | `hrl_passive_known_edge_fraction` | current |
| `intrmotiv/hrl/passive/path_length_mean` | `hrl_passive_path_length_mean` | current |
| `intrmotiv/hrl/passive/reject_motion_rate` | `hrl_passive_reject_motion_rate` | current |
| `intrmotiv/hrl/passive/reject_nonexclusive_rate` | `hrl_passive_reject_nonexclusive_rate` | current |
| `intrmotiv/hrl/passive/reject_path_rate` | `hrl_passive_reject_path_rate` | current |
| `intrmotiv/hrl/passive/reject_time_rate` | `hrl_passive_reject_time_rate` | current |
| `intrmotiv/hrl/passive/traversal_time_mean` | `hrl_passive_time_mean` | current |
| `intrmotiv/hrl/passive/updates_per_rollout` | `hrl_passive_updates` | current |

### HRL planning

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/hrl/planning/final_frontier_reach_rate` | `hrl_planning_final_frontier_reach_rate` | current |
| `intrmotiv/hrl/planning/hop_count_mean` | `hrl_planning_hop_count_mean` | current |
| `intrmotiv/hrl/planning/replan_rate` | `hrl_planning_replan_rate` | current |
| `intrmotiv/hrl/planning/route_available_rate` | `hrl_planning_route_available_rate` | current |
| `intrmotiv/hrl/planning/waypoint_success_rate` | `hrl_planning_waypoint_success_rate` | current |

### HRL reliable graph

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/hrl/reliable/largest_scc` | `hrl_reliable_largest_scc` | current |
| `intrmotiv/hrl/reliable/outgoing_node_count` | `hrl_reliable_outgoing_node_count` | current |
| `intrmotiv/hrl/reliable/outgoing_node_fraction` | `hrl_reliable_outgoing_node_fraction` | current |
| `intrmotiv/hrl/reliable/reachable_pair_fraction` | `hrl_reliable_reachable_pair_fraction` | current |
| `intrmotiv/hrl/reliable/reciprocal_fraction` | `hrl_reliable_reciprocal_fraction` | current |
| `intrmotiv/hrl/reliable/top3_incoming_confidence_share` | `hrl_reliable_top3_incoming_confidence_share` | current |

### HRL spatial graph summaries

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/hrl/summary/grounded_controllability` | `online_spatial_graph_grounded_controllability` | current |
| `intrmotiv/hrl/summary/reliable_global_efficiency` | `online_spatial_graph_reliable_global_efficiency` | current |

### HRL validation

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/hrl/validation/queued_edges` | `hrl_validation_queued_edges` | current |
| `intrmotiv/hrl/validation/return_success_rate` | `hrl_validation_return_success_rate` | current |
| `intrmotiv/hrl/validation/success_rate` | `hrl_validation_success_rate` | current |
| `intrmotiv/hrl/validation/timeout_rate` | `hrl_validation_timeout_rate` | current |

### Online place-field telemetry

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/online/place_field/active_only_map_cosine` | `online_spatial_place_active_only_map_cosine` | current |
| `intrmotiv/online/place_field/active_unit_fraction` | `online_spatial_place_active_unit_fraction` | current |
| `intrmotiv/online/place_field/active_unit_mean_spatial_information` | `online_spatial_place_active_unit_mean_spatial_information` | current |
| `intrmotiv/online/place_field/in_bounds_fraction` | `online_spatial_place_in_bounds_fraction` | current |
| `intrmotiv/online/place_field/mean_primary_secondary_peak_distance` | `online_spatial_place_mean_primary_secondary_peak_distance` | current |
| `intrmotiv/online/place_field/median_dominant_peak_nearest_neighbor_distance` | `online_spatial_place_median_dominant_peak_nearest_neighbor_distance` | current |
| `intrmotiv/online/place_field/mono_field_unit_fraction` | `online_spatial_place_mono_field_unit_fraction` | current |
| `intrmotiv/online/place_field/silent_unit_fraction` | `online_spatial_place_silent_unit_fraction` | current |
| `intrmotiv/online/place_field/unique_active_peak_bins` | `online_spatial_place_unique_active_peak_bins` | current |
| `intrmotiv/online/place_field/valid_sample_count` | `online_spatial_place_valid_sample_count` | current |
| `intrmotiv/online/place_field/visited_cell_fraction` | `online_spatial_place_visited_cell_fraction` | current |

### Online telemetry window

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/online/window/target_env_steps` | `online_spatial_scalar_target_env_steps` | current |

### Online trajectory telemetry

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/online/trajectory/mean_absolute_circular_yaw_change` | `online_spatial_trajectory_mean_absolute_circular_yaw_change` | current |
| `intrmotiv/online/trajectory/mean_physical_step_distance` | `online_spatial_trajectory_mean_physical_step_distance` | current |
| `intrmotiv/online/trajectory/path_efficiency` | `online_spatial_trajectory_path_efficiency` | current |
| `intrmotiv/online/trajectory/stationary_step_fraction` | `online_spatial_trajectory_stationary_step_fraction` | current |

### Optional CA3 predictor

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/predictor/hit_accuracy` | `ca3_predictor_hit_accuracy` | current |
| `intrmotiv/predictor/hit_time_mae` | `ca3_predictor_time_mae` | current |
| `intrmotiv/predictor/loss` | `ca3_predictor_loss` | current |
| `intrmotiv/predictor/positive_fraction` | `ca3_predictor_positive_fraction` | current |

### Path / scatter

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/path/displacement_mean` | `path_displacement_mean` | current |
| `intrmotiv/path/path_length_mean` | `path_length_mean` | current |
| `intrmotiv/path/scatter_conflict_fraction` | `dg_path_scatter_conflict_fraction` | current |
| `intrmotiv/path/scatter_loss` | `encoder_path_scatter_loss` | current |
| `intrmotiv/path/straightness_mean` | `path_straightness_mean` | current |

### Replay / generation barrier

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/replay/deferred_updates_total` | `stale_generation_deferred_updates_total` | current |
| `intrmotiv/replay/dropped_decisions_total` | `stale_generation_dropped_decisions_total` | current |
| `intrmotiv/replay/dropped_rollouts_total` | `stale_generation_dropped_rollouts_total` | current |
| `intrmotiv/replay/stale_generation_rejected_count` | `stale_generation_rejected_count` | current |
| `intrmotiv/replay/stale_generation_rejected_fraction` | `stale_generation_rejected_fraction` | current |

### Reward streams

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/reward/advantage_abs_mean` | `reward_for_advantage_abs_mean` | current |
| `intrmotiv/reward/advantage_max` | `reward_for_advantage_max` | current |
| `intrmotiv/reward/advantage_mean` | `reward_for_advantage_mean` | current |
| `intrmotiv/reward/advantage_min` | `reward_for_advantage_min` | current |
| `intrmotiv/reward/advantage_nonzero_fraction` | `reward_for_advantage_nonzero_frac` | current |
| `intrmotiv/reward/advantage_sum` | `reward_for_advantage_sum` | current |
| `intrmotiv/reward/environment_mean` | `env_reward_mean` | current |
| `intrmotiv/reward/environment_nonzero_fraction` | `env_reward_nonzero_frac` | current |
| `intrmotiv/reward/environment_sum` | `env_reward_sum` | current |
| `intrmotiv/reward/intrinsic_mean` | `intrinsic_reward_mean` | current |
| `intrmotiv/reward/intrinsic_negative_fraction` | `intrinsic_reward_negative_frac` | current |
| `intrmotiv/reward/intrinsic_nonzero_fraction` | `intrinsic_reward_nonzero_frac` | current |
| `intrmotiv/reward/intrinsic_sum` | `intrinsic_reward_sum` | current |

### Update schedule

| Exact tag | Producer key | Availability |
| --- | --- | --- |
| `intrmotiv/update/phase` | `iterative_phase` | current |

### Controller-DDQN dynamic namespace

| Exact tag or pattern | Meaning |
| --- | --- |
| `intrmotiv/controller/main_loss` | See controller-DDQN section above. |
| `intrmotiv/controller/auxiliary_loss` | See controller-DDQN section above. |
| `intrmotiv/controller/q_mean` | See controller-DDQN section above. |
| `intrmotiv/controller/q_abs_max` | See controller-DDQN section above. |
| `intrmotiv/controller/auxiliary_q_mean` | See controller-DDQN section above. |
| `intrmotiv/controller/auxiliary_q_abs_max` | See controller-DDQN section above. |
| `intrmotiv/controller/stored_state_replay` | See controller-DDQN section above. |
| `intrmotiv/controller/main_compute_seconds` | See controller-DDQN section above. |
| `intrmotiv/controller/her_compute_seconds` | See controller-DDQN section above. |
| `intrmotiv/controller/her_overhead_ratio` | See controller-DDQN section above. |
| `intrmotiv/controller/actor_memory_rebuilds` | See controller-DDQN section above. |
| `intrmotiv/controller/actor_memory_rebuild_seconds` | See controller-DDQN section above. |
| `intrmotiv/controller/actor_memory_version_failures` | See controller-DDQN section above. |
| `intrmotiv/controller/physical_interactions` | See controller-DDQN section above. |
| `intrmotiv/controller/physical_frames` | See controller-DDQN section above. |
| `intrmotiv/controller/accepted_replay_decisions` | See controller-DDQN section above. |
| `intrmotiv/controller/main_updates` | See controller-DDQN section above. |
| `intrmotiv/controller/main_td_positions` | See controller-DDQN section above. |
| `intrmotiv/controller/auxiliary_td_positions` | See controller-DDQN section above. |
| `intrmotiv/controller/target_age` | See controller-DDQN section above. |
| `intrmotiv/controller/update_debt` | See controller-DDQN section above. |
| `intrmotiv/controller/fresh_dg_steps` | See controller-DDQN section above. |
| `intrmotiv/controller/fresh_graph_batches` | See controller-DDQN section above. |
| `intrmotiv/controller/transaction_seconds` | See controller-DDQN section above. |
| `intrmotiv/controller/calibration_seconds` | See controller-DDQN section above. |
| `intrmotiv/controller/active_goal_count` | See controller-DDQN section above. |
| `intrmotiv/controller/anchor_registrations` | See controller-DDQN section above. |
| `intrmotiv/controller/confirmation_attempts` | See controller-DDQN section above. |
| `intrmotiv/controller/confirmation_successes` | See controller-DDQN section above. |
| `intrmotiv/controller/anchor_replacements` | See controller-DDQN section above. |
| `intrmotiv/controller/anchor_deactivations` | See controller-DDQN section above. |
| `intrmotiv/controller/calibration_ready` | See controller-DDQN section above. |
| `intrmotiv/controller/recognition_threshold` | See controller-DDQN section above. |
| `intrmotiv/controller/prediction_absolute_threshold` | See controller-DDQN section above. |
| `intrmotiv/controller/prediction_excess_threshold` | See controller-DDQN section above. |
| `intrmotiv/controller/calibration_pair_count` | See controller-DDQN section above. |
| `intrmotiv/controller/activation_latency_mean` | See controller-DDQN section above. |
| `intrmotiv/controller/empty_set_exploration` | See controller-DDQN section above. |
| `intrmotiv/controller/dg_optimizer_steps_min` | See controller-DDQN section above. |
| `intrmotiv/controller/dg_optimizer_steps_max` | See controller-DDQN section above. |
| `intrmotiv/controller/main_optimizer_steps_min` | See controller-DDQN section above. |
| `intrmotiv/controller/main_optimizer_steps_max` | See controller-DDQN section above. |
| `intrmotiv/controller/anchor_refinement_attempts` | See controller-DDQN section above. |
| `intrmotiv/controller/anchor_refinements` | See controller-DDQN section above. |
| `intrmotiv/controller/anchor_centrality_gain_mean` | See controller-DDQN section above. |
| `intrmotiv/controller/anchor_age_mean` | See controller-DDQN section above. |
| `intrmotiv/controller/context_raw_multi_activation` | See controller-DDQN section above. |
| `intrmotiv/controller/context_accepted_events` | See controller-DDQN section above. |
| `intrmotiv/controller/context_unique_rescues` | See controller-DDQN section above. |
| `intrmotiv/controller/context_zero_match` | See controller-DDQN section above. |
| `intrmotiv/controller/context_multi_match` | See controller-DDQN section above. |
| `intrmotiv/controller/positive_similarity_q10` | See controller-DDQN section above. |
| `intrmotiv/controller/positive_similarity_q50` | See controller-DDQN section above. |
| `intrmotiv/controller/positive_similarity_q90` | See controller-DDQN section above. |
| `intrmotiv/controller/background_similarity_q50` | See controller-DDQN section above. |
| `intrmotiv/controller/background_similarity_q90` | See controller-DDQN section above. |
| `intrmotiv/controller/background_similarity_q99` | See controller-DDQN section above. |
| `intrmotiv/controller/background_above_threshold_fraction` | See controller-DDQN section above. |
| `intrmotiv/controller/active_anchor_collision_fraction` | See controller-DDQN section above. |
| `intrmotiv/controller/her_contextual_candidates` | See controller-DDQN section above. |
| `intrmotiv/controller/her_contextual_positive_hits` | See controller-DDQN section above. |
| `intrmotiv/controller/her_contextual_wrong_context` | See controller-DDQN section above. |
| `intrmotiv/controller/her_contextual_start_achieved` | See controller-DDQN section above. |
| `intrmotiv/controller/her_contextual_missing_calibration` | See controller-DDQN section above. |
| `intrmotiv/controller/her_terminal_successor_ca3_missing` | See controller-DDQN section above. |
| `intrmotiv/controller/her_contextual_positive_rate` | See controller-DDQN section above. |
| `intrmotiv/controller/command_slot_XX` | Cumulative command count for contextual graph slot XX. |
| `intrmotiv/controller/rejected/<reason>` | Replay/controller rejection count keyed by the exact rejection reason. |

### Native off-policy DDQN/HER namespace

| Exact tag or pattern | Meaning |
| --- | --- |
| `ddqn/frames` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/decisions` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/accepted` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/updates` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/invalid_final` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/attempts` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/arrivals` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/throughput_fps` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/requested_her_fraction` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/target_period_updates` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/decisions_per_update` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/td_positions_per_update` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/effective_loss_positions_per_decision` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/inference_decisions` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/epsilon` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/realized_her_fraction` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/valid_loss_positions_total` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/her_loss_positions` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/original_loss_positions` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/ingestion_seconds` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/learning_seconds` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/transport_received` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/transport_pending` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/update_debt` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/replay_size` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/td_loss` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/grad_norm` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/valid_loss_positions` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/target_copies` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/td_target_min` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/td_target_max` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/td_target_mean` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/action_gap_mean` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/q_min` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/q_max` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/q_mean` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/q_out_of_range_fraction` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/prefix_and_batch_seconds` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/learner_update_seconds` | Native recurrent DDQN/HER learner/runtime diagnostic. |
| `ddqn/goal_<g>/attempts` | Completed commanded attempts for goal g. |
| `ddqn/goal_<g>/achieved_events` | Observed recognition events for goal g, regardless of command. |
| `ddqn/goal_<g>/achieved_episodes` | Distinct stream/episode pairs containing an event for goal g. |
| `ddqn/goal_<g>/commanded_arrivals` | Attempts whose commanded goal g was reached. |
| `ddqn/goal_<g>/requested_her_segments` | Segments for which HER was requested. |
| `ddqn/goal_<g>/realized_her_segments` | Segments actually relabeled to goal g. |
| `ddqn/goal_<g>/{original,her}/segments` | Sampled original or relabeled replay segments. |
| `ddqn/goal_<g>/{original,her}/reward_segments` | Sampled segments containing positive reward. |
| `ddqn/goal_<g>/{original,her}/contributing_episodes` | Distinct source episodes contributing replay. |
| `ddqn/goal_<g>/her/selected_offset_<k>` | HER selected-goal offset histogram. |
| `ddqn/goal_<g>/{original,her}/budget_<lo>_<hi>` | Valid TD positions grouped by remaining-budget bin. |
| `ddqn/goal_<g>/prediction_{min,max,mean}` | Chosen-action online Q prediction statistics for goal g. |
| `ddqn/goal_<g>/td_target_{min,max,mean}` | Double-DQN TD-target statistics for goal g. |
| `ddqn/goal_<g>/action_gap_mean` | Mean top-two online-Q action gap for goal g. |

## Maintenance rule

When code changes a metric's numerator, denominator, window, reset behavior, namespace, or availability, update this reference in the same change. For static <code>intrmotiv/*</code> tags, compare against <code>INTRMOTIV_SUMMARY_TAGS</code>. For contextual controller metrics, inspect <code>controller_learner.py</code> and remember that <code>controller/*</code> is routed to <code>intrmotiv/controller/*</code>. For native off-policy metrics, inspect <code>NativeLearner.metrics()</code>, <code>DoubleDQNLearner.update()</code>, and <code>Coverage</code>; <code>ddqn_summary</code> prefixes every key with <code>ddqn/</code>.
