# IntrMotiv experiment reports: factorized synthesis and status map

This is the canonical entry point for experiment reports. Dates are provenance;
the scientific organization is by **which factor changed**. Start with the
status map below, then read a study report for its matched comparison.
Historical launch notes retain exact manifests and implementation details.
The [open-analysis register](open_analyses.md) lists the specific evidence
still needed to close major scientific claims.
The [organization strategy](ORGANIZATION.md) explains the study-line grouping
and the criteria used to merge overlapping notes. Study notes are physically
grouped in `syntheses/`, `hrl_graph/`, `dg_representation/`, `controllers/`,
`ca3_goals/`, and `environments_transfer/`. Only this index, the strategy,
the open-analysis register, and two provenance-pinned reports remain as
Markdown files at this folder's top level.

For a report, use the owner table below. Study inputs and collected tables are
under `data/<study>/`; figures and analysis bundles are under
`results/<study>/`. Analysis and rendering scripts remain at this folder's top
level, generally named `analyze_*`, `plot_*`, or `render_*`. The validated run
definition and reusable tooling are in [hpc_runs](../hpc_runs/README.md).

## Report status and ownership

| Line | Current result report | Evidence boundary | Earlier material |
| --- | --- | --- | --- |
| CA3 state-goal follow-up | [[ca3_goals/ca3_followup_analysis_20260926|Matched CA3 follow-up]] | Complete 75M CPU and G500 factorials; later endpoints have unequal ages | The 25M interim table, restricted 75M snapshot, and atlas are integrated in the same report |
| Five-cue frozen-DG controls | [[cued_reward5_transfer_20260925|Five-cue campaign and controls]]; [exact 75M endpoint](results/A0_poster_analysis_20260926/batch_summary.md#main-quantitative-findings) | Paired 50M online evidence and exact 75M frozen/heldout control evidence; the wider eight-arm transfer study has no matched outcome report here | The September 26 interim map, manager, and input-probe analysis is integrated in the campaign report |
| CA3 predictive active goals | [[ca3_goals/ca3_predictive_active_goals_interim_analysis_20260924|Predictive CA3]] | Balanced 75M online spatial analysis; the declared 300M outcome is unreported | Interim remains the primary result report |
| DG capacity and goal conditioning | [[dg_representation/dg_capacity_goal_conditioning_interim_20260911|Capacity and goal conditioning]]; [selected exact endpoint](results/A0_poster_analysis_20260926/batch_summary.md#main-quantitative-findings) | Balanced 25M matrix, restricted 75M direct subset, and later selected endpoints; no matched-age F64 waypoint endpoint contrast | Interim remains the primary full-matrix report |
| Navigation8 screen | [[controllers/navigation8_algorithm_screen_interim_20260916|Navigation8 screen]] | Complete three-seed 75M online snapshot; no shared exact saved terminal checkpoint for the selected seeds | Interim remains the primary result report |
| Easy landmark maze | [[environments_transfer/easy_landmark_maze_qualification_analysis_20260924|Landmark maze qualification]] | 2M qualification; production effect has no matched report here | Qualification is not a production outcome |
| Directional/predictive recruitment | [[hrl_graph/directional_predictive_recruitment_interim_20260904|Directional/predictive diagnosis]] | Early graph and representation diagnosis with causal comparison unresolved | Retained separately from its implementation record |

For code-level architecture definitions, use [[../04_implementation/architecture/README|Architecture Reference]]. For metric meanings and denominators, use [[../04_implementation/IntrMotiv_metric_reference|Metric Reference]] and [[../04_implementation/IntrMotiv_metrics_guidebook|Metrics Guidebook]].

## Complete Markdown index

Read each row from left to right: the result or synthesis owner comes first,
then its supporting plans, run records, and dated probes. `RESULT` reports an
outcome; `PLAN` proposes work; `RUN` records implementation or submission;
`TELEMETRY` is a scoped measurement; `AUDIT` checks a claim or contract;
`SYNTHESIS` compares lines; `ARTIFACT` belongs to a pinned output bundle.
Dates identify evidence age. The [owner table](#report-status-and-ownership)
above determines which findings are current.

### A. Cross-study syntheses and historical briefings

| Study line or question | Files in reading order |
| --- | --- |
| Early batch statistics | [SYNTHESIS: batch statistics](syntheses/recent_batch_statistics_report.md) |
| September design review | [SYNTHESIS: recent batch design audit](syntheses/recent_batches_design_audit_20260906.md) |
| 10 September briefing | [SYNTHESIS: architectures and fields](syntheses/02_architectures_losses_place_fields_trajectories_and_graphs_20260910.md); [SYNTHESIS: mean and silent units](syntheses/03_mean_punishment_and_silent_units_jannek_comparison_20260910.md); [SYNTHESIS: three-goal capacity](syntheses/04_three_goal_context_conditioning_and_dg_capacity_20260910.md); [SYNTHESIS: CA3 feedback](syntheses/05_ca3_feedback_matched_results_and_full_state_gap_20260913.md); [SYNTHESIS: goal-set controls](syntheses/06_high_option_success_goal_sets_and_controls_20260914.md); [AUDIT: FiLM and input weights](syntheses/07_film_goal_parameters_and_ca3_depth_weights_20260914.md) |
| Late-training signals | [SYNTHESIS: outliers](syntheses/late_training_outliers_20260908.md); [TELEMETRY: spatial gallery](syntheses/late_outlier_spatial_gallery_20260908.md); [AUDIT: target-hit lift](syntheses/late_target_hit_lift_audit_20260908.md) |
| September architecture and poster selection | [SYNTHESIS: three architecture batches](syntheses/recent_architecture_batches_synthesis_20260924.md); [AUDIT: poster candidates](syntheses/poster_candidate_screening_20260925.md); [AUDIT: exemplar snapshot](syntheses/poster_exemplar_analysis_20260925.md) |

### B. Early HRL, graph, and recruitment studies

| Study line or question | Files in reading order |
| --- | --- |
| First HRL iteration | [RESULT: batch 1 and next iteration](hrl_graph/hrl_batch1_results_and_next_iteration.md); [RUN: iteration 2 and retained questions](hrl_graph/hrl_iteration2_implementation_and_batch.md); [PLAN: compatibility pointer](hrl_graph/next_iteration_plan_before_iterative_update.md) |
| Manager design | [RUN: frontier isolation](hrl_graph/frontier_manager_isolation_20260831.md); [RUN: manager exploration](hrl_graph/hrl_manager_exploration_batch.md); [PLAN: topological frontier](hrl_graph/topological_frontier_planning_batch.md) |
| Controllability and graph recruitment | [RUN: edge exploration](hrl_graph/controllability_edge_exploration_20260903.md); [RESULT: graph-stabilized recruitment](hrl_graph/graph_stabilized_recruitment_20260903.md); [TELEMETRY: aligned 75M fields](hrl_graph/graph_stabilized_recruitment_place_field_telemetry_20260903.md) |
| Directional and predictive recruitment | [RESULT: interim diagnosis](hrl_graph/directional_predictive_recruitment_interim_20260904.md); [RUN: batch definition](hrl_graph/directional_predictive_recruitment_20260904.md) |

### C. DG representation and spatial health

| Study line or question | Files in reading order |
| --- | --- |
| Anti-collapse and regularizers | [RESULT: online and place-field anti-collapse](dg_representation/dg_anti_collapse_results.md); [RESULT: encourage-regularizer interim and 50M telemetry](dg_representation/encourage_dg_regularizers_interim_analysis.md); [AUDIT: DGP trivial minimum](dg_representation/dgp_interim_failure_audit_20260907.md) |
| Structural diversity and manager exploration | [RESULT: online and 10k-decision spatial results](dg_representation/dg_structural_and_manager_exploration_results.md); [RUN: structural-diversity batch](dg_representation/dg_structural_diversity_batch.md) |
| Corrected core | [RESULT: historical re-evaluation](corrected_core_reevaluation_20260901.md); [TELEMETRY: selected candidates](dg_representation/corrected_core_candidate_place_field_telemetry_20260902.md) |
| DG capacity and goal conditioning | [RESULT: interim factorial](dg_representation/dg_capacity_goal_conditioning_interim_20260911.md); [AUDIT: later run health](dg_representation/dg_capacity_health_20260913.md); [PLAN: matrix](dg_representation/dg_capacity_goal_conditioning_plan_20260910.md); [RUN: launch](dg_representation/dg_capacity_goal_conditioning_launch_20260910.md) |
| DG neighborhood and update mechanics | [RUN: G500 qualification](dg_representation/dg_neighborhood_g500_20260914.md); [AUDIT: normalization gradients](dg_representation/landmark_normalization_gradient_audit.md); [AUDIT: encoder/decoder update contract](dg_representation/encoder_decoder_update_contract_batch_diagnosis.md) |
| Minimal mechanism test | [RESULT: thresholded rotation toy](dg_representation/threshold_rotation_toy_report.md) |

### D. Controller, replay, and goal-control studies

| Study line or question | Files in reading order |
| --- | --- |
| Native DDQN/HER | [RUN: recurrent DDQN/HER](controllers/intrmotiv_ddqn_her_implementation_20260911.md); [RUN: Sample Factory integration](controllers/intrmotiv_ddqn_sample_factory_integration_20260911.md); [AUDIT: v2 repair](controllers/intrmotiv_ddqn_her_v2_repair_20260911.md); [AUDIT: throughput](controllers/intrmotiv_ddqn_throughput_20260911.md); [AUDIT: metric consistency](controllers/intrmotiv_ddqn_metric_consistency_20260912.md); [RUN: production](controllers/intrmotiv_ddqn_sf_production_20260912.md) |
| Full-system and off-policy control | [RUN: full-system controller](controllers/intrmotiv_full_system_controller_20260912.md); [RUN: CRL+/L3P+ baselines](controllers/offpolicy_crl_l3p_implementation_20260911.md); [RUN: RR1 extension](controllers/controller_rr1_extension_20260913.md) |
| CPU controller screen | [RESULT: CPU2048](controllers/cpu2048_analysis_20260917.md); [RUN: selected CPU comparison](controllers/controller_cpu_selected_20260914.md) |
| Navigation8 screen | [RESULT: 75M interim](controllers/navigation8_algorithm_screen_interim_20260916.md); [RUN: implementation](controllers/navigation8_algorithm_screen_implementation_20260909.md) |
| Target-control representation | [TELEMETRY: provisional HER place fields](controllers/target_control_her_provisional_place_field_telemetry_20260902.md) |
| Persistent intrinsic control | [RESULT: 8–9 September status and learning audit](controllers/persistent_intrinsic_control_status_20260909.md); [RUN: implementation](controllers/persistent_intrinsic_control_implementation.md) |

### E. CA3 memory and contextual goals

| Study line or question | Files in reading order |
| --- | --- |
| Finite-memory novelty | [RUN: memory novelty and minimal control](ca3_goals/ca3_memory_novelty_goal_implementation.md) |
| Predictive active goals | [RESULT: matched 75M interim](ca3_goals/ca3_predictive_active_goals_interim_analysis_20260924.md); [RUN: active-goal batch](ca3_goals/ca3_predictive_active_goals_20260922.md) |
| State-goal follow-up | [RESULT: matched follow-up](ca3_goals/ca3_followup_analysis_20260926.md); [RUN: state-goal release](ca3_goals/ca3_state_goal_followup_20260922.md) |

### F. Environment, reward, and transfer

| Study line or question | Files in reading order |
| --- | --- |
| Corridor geometry | [RESULT: 100M analysis](environments_transfer/corridor_geometry_analysis_20260921.md); [RUN: geometry and qualification](environments_transfer/corridor_geometry_20260919.md) |
| Easy landmark maze | [RESULT: 2M qualification](environments_transfer/easy_landmark_maze_qualification_analysis_20260924.md); [RUN: cue screen](environments_transfer/easy_landmark_maze_implementation_20260923.md) |
| Initial fixed-reward transfer | [RESULT: latest common step](environments_transfer/fixed_reward_transfer_latest_common_20260911.md); [RUN: implementation](environments_transfer/fixed_reward_transfer_implementation_20260910.md); [RUN: repeat-8 replacement](environments_transfer/fixed_reward_transfer_repeat8_launch_20260910.md); [PLAN: timing replan](environments_transfer/fixed_reward_transfer_timing_replan_20260910.md) |
| DG-peak and five-cue transfer | [RESULT: five-cue campaign and controls](cued_reward5_transfer_20260925.md); [RUN: DG-peak execution](environments_transfer/fixed_reward_dg_peak_transfer_execution_20260924.md); [AUDIT: site candidates](environments_transfer/fixed_reward_site_candidate_audit_20260924.md) |

### G. Nested Markdown in pinned data and result bundles

These files stay with their source tables and figures. They are indexed here
for discovery, while the bundle manifests remain authoritative for replay.

| Bundle | Files |
| --- | --- |
| Study data | [ARTIFACT: CPU2048 atlas](data/cpu2048_analysis_20260917/atlas.md); [ARTIFACT: full-system request](data/intrmotiv_full_system_controller_20260912/request.md); [ARTIFACT: Navigation8 atlas](data/navigation8_algorithm_screen_interim_20260916/visual_atlas_75m.md) |
| A0 poster overview | [ARTIFACT: summary](results/A0_poster_analysis_20260926/summary.md); [ARTIFACT: batch results](results/A0_poster_analysis_20260926/batch_summary.md); [ARTIFACT: cross-run scatter](results/A0_poster_analysis_20260926/cross_run_scatter/report.md); [ARTIFACT: flat/goal comparison](results/A0_poster_analysis_20260926/flat_goal_comparison/report.md) |
| C15 variants | [ARTIFACT: report](results/A0_poster_analysis_20260926/c15_variants/report.md); [ARTIFACT: run index](results/A0_poster_analysis_20260926/c15_variants/run_index.md); [ARTIFACT: source index](results/A0_poster_analysis_20260926/c15_variants/source_index.md); [ARTIFACT: variant catalogue](results/A0_poster_analysis_20260926/c15_variants/variant_catalogue.md); [ARTIFACT: pinned corrected-core source](results/A0_poster_analysis_20260926/c15_variants/source_inputs/06_experiments/corrected_core_reevaluation_20260901.md) |
| Exemplar gallery | [ARTIFACT: run index](results/A0_poster_analysis_20260926/exemplar_gallery/run_index.md); [ARTIFACT: selection](results/A0_poster_analysis_20260926/exemplar_gallery/selection_summary.md); [ARTIFACT: historical screen](results/A0_poster_analysis_20260926/exemplar_gallery/historical_extension_screen.md); [ARTIFACT: extension report](results/A0_poster_analysis_20260926/exemplar_gallery/historical_extension/report.md); [ARTIFACT: extension run index](results/A0_poster_analysis_20260926/exemplar_gallery/historical_extension/run_index.md) |

## 1. Canonical factorization

Every experiment should be described along the same axes.

| Factor                   | Questions to record                                                                                          |
| ------------------------ | ------------------------------------------------------------------------------------------------------------ |
| Environment / task       | Open field, corridor, easy-landmark maze, rewarded transfer; action set; frameskip/repeat; cue manipulation  |
| Visual input             | Frozen ImageNet trunk; any cue/instruction/depth channels; whether privileged pose is telemetry-only         |
| DG representation        | F; threshold/normalization; gradient owner; encoder objective; recruitment/retirement; contextual modulation |
| CA3 / memory state       | Fixed shift register parameters; raw CA3 versus learned readout; history horizon                             |
| Goal representation      | DG ID, FiLM target ID, continuous CA3/readout state, contextual anchor, external reward instruction          |
| Worker / controller      | PPO, direct DDQN, stored-state DDQN, HER; flat versus target-conditioned decoder                             |
| Manager / graph          | None, direct target, frontier/waypoint, passive/controllable graph, contextual graph; validation rules       |
| Reward / supervision     | Dense temporal-distance, hit, first-outcome, external reward, encoder credit, predictive auxiliary loss      |
| Replay / update contract | On-policy, empirical HER, stored replay, cadence, target-network refresh, representation-generation barrier  |
| Evaluation               | Online spatial telemetry, frozen matched rollouts, command intervention, transfer performance, coverage      |

A comparison is clean only when its row changes the intended factor while the other important axes are held fixed. Historical family names often change several factors simultaneously.

## 2. Architecture-family dictionary

| Family / shorthand       | DG and memory                                                            | Goal representation                         | Controller / manager                                        | Main architectural distinction                                                            |
| ------------------------ | ------------------------------------------------------------------------ | ------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Flat intrinsic           | Sparse DG + fixed CA3                                                    | None                                        | PPO, no target graph                                        | Dense temporal-distance worker reward; no explicit destination                            |
| SCR                      | F16 DG; arrival credit; direction-sensitive recruitment                  | DG target ID                                | PPO + graph/direct-target HRL                               | Representation protection/recruitment is the main intervention                            |
| SAT                      | SCR-like representation with open endpoint retirement; FiLM in key cells | DG target ID via FiLM                       | PPO + graph HRL                                             | Adds a more permissive retirement gate and compact goal interface                         |
| DGP                      | F16 DG with PPO-to-DG JOINT in historical cells                          | DG target ID, LEG or FiLM                   | PPO; HIT/FIRST outcome variants                             | Couples worker objective directly to DG and can build dense graphs without causal control |
| CPD                      | F16 DG with CA3-history-dependent feedback                               | DG target ID / FiLM depending cell          | PPO                                                         | Tests CA3 feedback-history routing such as DIRECT versus BPTT                             |
| W_REF                    | Frozen/shared reference detector variants                                | Reference-defined target                    | PPO; no comparable learned graph payload                    | Tests reference-state routing rather than self-organized graph learning                   |
| Direct F16 DDQN          | F16 DG + fixed CA3                                                       | DG target ID                                | Stored-state DDQN, optional HER                             | Replaces PPO control with off-policy Q learning while keeping direct landmark goals       |
| Waypoint decoder F64     | F64 DG + goal-independent memory                                         | DG target ID at decoder                     | Waypoint manager + stored-state DDQN, optional HER          | Larger landmark vocabulary and decoder-only goal conditioning                             |
| CA3 predictive/readout   | F64 DG + learned z = W S from CA3                                        | DG ID or continuous readout state           | Stored DDQN+HER + waypoint/context graph                    | Separates current state representation, goal representation, and contextual recognition   |
| CA3 state-goal follow-up | Same predictive readout, H32                                             | Continuous readout goal + contextual anchor | Stored DDQN+HER                                             | Factorial anchor maintenance FIXED/EMA × candidate admission DOM/UNIQUE                   |
| Reward transfer          | Source DG/policy/graph may be frozen, tuned, or discarded                | Fixed or cued external reward site          | External-reward PPO or transferred waypoint/controller path | Tests reuse after reward specification rather than intrinsic-training quality itself      |

## 3. What the chronological development actually changed

### 3.1 Representation formation

The earliest question was whether elapsed-time feedback could distribute sparse DG events. The central representation factors became: encoder feedback sign/centering, batch recruitment, population/collision regularization, normalization, temporal exclusion, and retirement. The cleanest representation-focused resources are:

- [[syntheses/03_mean_punishment_and_silent_units_jannek_comparison_20260910|Mean, punishment, and silent units]] — why suppression plus weak recruitment can create silence.
- [[dg_representation/dg_anti_collapse_results|DG anti-collapse online and place-field results]].
- [[dg_representation/dg_structural_and_manager_exploration_results|DG structural and manager exploration results]].
- [[dg_representation/landmark_normalization_gradient_audit|Normalization/gradient audit]].
- [[syntheses/recent_batches_design_audit_20260906|Recent batch design audit]] — ARR/SRC, FiLM, retirement, graph false positives.
- [[controllers/navigation8_algorithm_screen_interim_20260916|Navigation8 screen]] — selected architecture families under one newer action/temporal regime.

**Current lesson:** low map overlap, mono-field structure, broad exploration, and graph connectivity are distinct outcomes. None should be used as a proxy for the others.

### 3.2 Goal representation and aliasing

The project then moved from “one DG unit = one destination” toward contextual goals.

- [[syntheses/04_three_goal_context_conditioning_and_dg_capacity_20260910|Three-goal context conditioning and capacity]] shows that task context can alter DG and decoder state, but capacity and downstream width co-vary.
- [[syntheses/05_ca3_feedback_matched_results_and_full_state_gap_20260913|CA3 feedback matched comparison]] tests feedback-history gradient routing while holding the broader CPD architecture fixed.
- [[syntheses/07_film_goal_parameters_and_ca3_depth_weights_20260914|FiLM parameter audit]] shows that goal-dependent parameters exist, but parameter variation is not behavioral control.
- [[ca3_goals/ca3_predictive_active_goals_interim_analysis_20260924|CA3 predictive active goals]] separates shadow readout, worker state, continuous goal, contextual recognition, action conditioning, and horizon.
- [[ca3_goals/ca3_followup_analysis_20260926|CA3 state-goal follow-up]] isolates FIXED/EMA anchor maintenance and DOM/UNIQUE contextual candidate admission, with the earlier CPU interim evidence integrated.

**Current lesson:** instantaneous DG identity is often too aliased to serve as a robust goal. The newer architecture keeps DG as a sparse address/event code while using an observed CA3 state and learned predictive readout to define goal identity.

### 3.3 Controller learning and replay

Controller changes should not be mixed with representation changes.

- [[controllers/intrmotiv_full_system_controller_20260912|Full-system controller integration]] records the transition from PPO-only control to direct/waypoint stored-state DDQN and HER.
- [[controllers/controller_cpu_selected_20260914|Selected CPU comparison]] defines DDQN/HER and cadence manipulations.
- [[controllers/cpu2048_analysis_20260917|CPU2048 analysis]] compares Direct F16 versus Waypoint F64, DDQN versus HER, and cadence 64 versus 2048.
- [[controllers/intrmotiv_ddqn_her_implementation_20260911|Native recurrent DDQN/HER implementation]] and [[controllers/intrmotiv_ddqn_metric_consistency_20260912|metric consistency audit]] document the separate native off-policy path.

**Current lesson:** DDQN/HER changes the learning problem and replay support, not only optimizer choice. Direct F16 versus Waypoint F64 also changes capacity, manager, and goal interface, so it is a family comparison rather than a capacity ablation.

### 3.4 Graphs, managers, and apparent control

The graph can look strong even when commands do not causally control destinations.

- [[syntheses/06_high_option_success_goal_sets_and_controls_20260914|High option success and matched controls]] shows that eventual target activation can be high while FIRST/outcome specificity is weak.
- [[syntheses/recent_batches_design_audit_20260906|Design audit]] documents dense-graph false positives and the distinction between target sensitivity and target-specific outcomes.
- [[hrl_graph/controllability_edge_exploration_20260903|Controllability/edge exploration]], [[hrl_graph/directional_predictive_recruitment_20260904|directional/predictive recruitment]], and [[hrl_graph/topological_frontier_planning_batch|topological frontier planning]] contain the mechanism-level graph experiments.

**Current lesson:** edge count, reachability, SCC size, and option success are not sufficient evidence for command-conditioned navigation. Prefer matched commanded-versus-shuffled first-outcome tests.

### 3.5 Environment geometry and perceptual landmarks

Environment manipulations are orthogonal to controller architecture and should be analyzed as such.

- [[environments_transfer/corridor_geometry_analysis_20260921|Corridor geometry analysis]] crosses architecture families with open/corridor geometry.
- [[environments_transfer/easy_landmark_maze_qualification_analysis_20260924|Easy landmark maze qualification]] crosses SCR/DGP/Waypoint with rich versus neutral visual cues while holding maze geometry fixed.

**Current lesson:** making sensory states easier to distinguish can help some architectures without repairing a fundamentally goal-insensitive controller; conversely, corridor geometry can reduce exploration even if it simplifies topology.

### 3.6 Transfer after reward specification

Transfer is a separate scientific question: whether intrinsically learned structure is useful when a downstream reward is introduced.

- [[environments_transfer/fixed_reward_transfer_latest_common_20260911|Fixed-reward latest-common comparison]] compares scratch, frozen DG, tuned DG, and policy transfer at a shared training window.
- [[environments_transfer/fixed_reward_dg_peak_transfer_execution_20260924|DG-peak transfer execution]] and [[environments_transfer/fixed_reward_transfer_implementation_20260910|transfer implementation]] document later transfer variants.
- [[cued_reward5_transfer_20260925|Five-cue reward transfer and frozen-DG controls]] covers the task, campaign, paired online control results, and CA3/depth input probes. The [exact 75M endpoint report](results/A0_poster_analysis_20260926/batch_summary.md#main-quantitative-findings) adds frozen and heldout comparisons.

**Current lesson:** the completed five-cue frozen-DG control endpoints show that stronger source-DG field and option metrics do not imply better downstream reward or heldout success. Analyze the wider eight-arm transfer matrix before claiming which source components help.

## 4. Current experiment matrix

| Study | Environment factor | Representation factor | Goal factor | Controller factor | Primary scientific question |
| --- | --- | --- | --- | --- | --- |
| Navigation8 screen | New action/repeat regime | SCR/SAT/DGP/CPD/W_REF bundles | Mostly DG-ID/reference goals | PPO families | Which historical architecture properties survive a matched newer navigation regime? |
| DG capacity / goal conditioning | Open field | F16/32/64 and DG goal write variants | DG ID | Direct/waypoint | Does more landmark capacity or context routing prevent control collapse? |
| CPU2048 | Open field | Direct F16 vs Waypoint F64 | DG ID, decoder-only waypoint | DDQN vs HER; cadence 64/2048 | Can off-policy control improve target sensitivity and representation/graph balance? |
| Corridor geometry | Open vs corridor layouts | SAT/DGP/Waypoint families | Family-specific | Family-specific | Does lower-dimensional topology help exploration/control? |
| Easy landmark maze | Same geometry, rich vs neutral cues | SCR/DGP/Waypoint | Family-specific | PPO or DDQN/HER by family | Does perceptual uniqueness rescue representation/control? |
| CA3 predictive active goals | Reward-free open field | Learned CA3 readout variants | DG ID vs continuous z-goal; contextual hits | Stored DDQN+HER | Does predictive CA3 state solve goal aliasing? |
| CA3 state-goal follow-up | Reward-free open field | Fixed predictive readout architecture | Continuous z-goal | Stored DDQN+HER | Which anchor maintenance and contextual admission rule preserves identity without killing graph evidence? |
| Reward transfer | Rewarded open field | Scratch vs transferred/tuned source structure | Fixed or instructed reward locations | External-reward learner / transferred controller | When does intrinsically learned structure accelerate downstream learning? |

## 5. Reading order

For a concise current view:

1. This synthesis.
2. [[../05_plans/poster_results_synthesis_20260922|Poster/paper scientific synthesis]] for the narrative and claim boundary.
3. [[syntheses/recent_batches_design_audit_20260906|Design audit]] for the core graph/control failure mode.
4. [[controllers/navigation8_algorithm_screen_interim_20260916|Navigation8]] and [[controllers/cpu2048_analysis_20260917|CPU2048]] for matched architecture/controller comparisons.
5. [[ca3_goals/ca3_predictive_active_goals_interim_analysis_20260924|Predictive CA3]] and [[ca3_goals/ca3_followup_analysis_20260926|matched state-goal follow-up]] for the current goal-representation line.
6. [[environments_transfer/easy_landmark_maze_qualification_analysis_20260924|easy-landmark cues]], [[environments_transfer/corridor_geometry_analysis_20260921|corridor geometry]], and [[cued_reward5_transfer_20260925|five-cue transfer]] for environment and transfer tests; use the [75M frozen-DG endpoint](results/A0_poster_analysis_20260926/batch_summary.md#main-quantitative-findings) for transfer outcomes.

Use the dated implementation/launch reports only when you need exact StudySpecs, job IDs, source revisions, qualification gates, or failure provenance.

## 6. Standard report structure going forward

Every result report should use this order:

1. **Status and evidence boundary** — completed/interim/qualification; exact common checkpoint/window. The primary comparison uses the newest checkpoint complete for the declared contrast. If a newer checkpoint is complete only for a scientifically valid subset, add it as a restricted follow-up rather than comparing each run at its individual latest age.
2. **Architecture factorization** — the table of environment, DG, CA3/state, goal, controller, graph/manager, replay, reward; explicitly state what differs and what is held fixed.
3. **Declared contrasts** — which comparisons are causal/matched and which are descriptive family comparisons.
4. **Results by scientific question** — representation, goal identity/control, external behavior, optimization/health.
5. **Interpretation boundary** — what the metrics do and do not establish.
6. **Provenance/resources** — StudySpec, source, data tables, figures, W&B, evaluation artifacts.
7. **Reusable lesson / next discriminating test**.

Chronological launch notes can remain inside the provenance section, but new architecture revisions should update the factor table rather than prepend another “latest update” paragraph.

**Checkpoint-refresh rule:** before reusing a report for the poster or a synthesis, re-audit the snapshot inventory. Preserve an older checkpoint only when it is the newest fully matched comparison or when it captures a scientifically unique transient state; label the latter explicitly. Do not silently substitute each run's latest checkpoint for a matched milestone.

## 7. Resource map

- [[../04_implementation/architecture/architectural_choices|Architectural choices]]
- [[../04_implementation/architecture/losses|Loss catalogue]]
- [[../04_implementation/IntrMotiv_metric_reference|Complete metric dictionary]]
- [[../04_implementation/IntrMotiv_metrics_guidebook|Metric interpretation guide]]
- [[../04_implementation/reusable_place_field_telemetry|Place-field telemetry protocol]]
- [[../04_implementation/standardized_study_workflow|Study/collection workflow]]
- [[../05_plans/poster_results_synthesis_20260922|Poster/paper synthesis]]
