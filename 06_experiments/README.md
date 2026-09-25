# IntrMotiv experiment reports: factorized synthesis

This is the canonical entry point for experiment reports. Dates are provenance; the scientific organization is by **which factor changed**. Historical reports are retained because they contain exact evidence, manifests, and implementation details, but their shorthand names should be decoded through the factorization below rather than treated as indivisible algorithms.

For code-level architecture definitions, use [[../04_implementation/architecture/README|Architecture Reference]]. For metric meanings and denominators, use [[../04_implementation/IntrMotiv_metric_reference|Metric Reference]] and [[../04_implementation/IntrMotiv_metrics_guidebook|Metrics Guidebook]].

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

- [[03_mean_punishment_and_silent_units_jannek_comparison_20260910|Mean, punishment, and silent units]] — why suppression plus weak recruitment can create silence.
- [[dg_anti_collapse_results|DG anti-collapse results]] and [[dg_anti_collapse_place_fields|place-field analysis]].
- [[dg_structural_and_manager_exploration_results|DG structural and manager exploration results]].
- [[landmark_normalization_gradient_audit|Normalization/gradient audit]].
- [[recent_batches_design_audit_20260906|Recent batch design audit]] — ARR/SRC, FiLM, retirement, graph false positives.
- [[navigation8_algorithm_screen_interim_20260916|Navigation8 screen]] — selected architecture families under one newer action/temporal regime.

**Current lesson:** low map overlap, mono-field structure, broad exploration, and graph connectivity are distinct outcomes. None should be used as a proxy for the others.

### 3.2 Goal representation and aliasing

The project then moved from “one DG unit = one destination” toward contextual goals.

- [[04_three_goal_context_conditioning_and_dg_capacity_20260910|Three-goal context conditioning and capacity]] shows that task context can alter DG and decoder state, but capacity and downstream width co-vary.
- [[05_ca3_feedback_matched_results_and_full_state_gap_20260913|CA3 feedback matched comparison]] tests feedback-history gradient routing while holding the broader CPD architecture fixed.
- [[07_film_goal_parameters_and_ca3_depth_weights_20260914|FiLM parameter audit]] shows that goal-dependent parameters exist, but parameter variation is not behavioral control.
- [[ca3_predictive_active_goals_interim_analysis_20260924|CA3 predictive active goals]] separates shadow readout, worker state, continuous goal, contextual recognition, action conditioning, and horizon.
- [[ca3_state_goal_followup_interim_analysis_20260924|CA3 state-goal follow-up]] isolates FIXED/EMA anchor maintenance and DOM/UNIQUE contextual candidate admission.

**Current lesson:** instantaneous DG identity is often too aliased to serve as a robust goal. The newer architecture keeps DG as a sparse address/event code while using an observed CA3 state and learned predictive readout to define goal identity.

### 3.3 Controller learning and replay

Controller changes should not be mixed with representation changes.

- [[intrmotiv_full_system_controller_20260912|Full-system controller integration]] records the transition from PPO-only control to direct/waypoint stored-state DDQN and HER.
- [[controller_cpu_selected_20260914|Selected CPU comparison]] defines DDQN/HER and cadence manipulations.
- [[cpu2048_analysis_20260917|CPU2048 analysis]] compares Direct F16 versus Waypoint F64, DDQN versus HER, and cadence 64 versus 2048.
- [[intrmotiv_ddqn_her_implementation_20260911|Native recurrent DDQN/HER implementation]] and [[intrmotiv_ddqn_metric_consistency_20260912|metric consistency audit]] document the separate native off-policy path.

**Current lesson:** DDQN/HER changes the learning problem and replay support, not only optimizer choice. Direct F16 versus Waypoint F64 also changes capacity, manager, and goal interface, so it is a family comparison rather than a capacity ablation.

### 3.4 Graphs, managers, and apparent control

The graph can look strong even when commands do not causally control destinations.

- [[06_high_option_success_goal_sets_and_controls_20260914|High option success and matched controls]] shows that eventual target activation can be high while FIRST/outcome specificity is weak.
- [[recent_batches_design_audit_20260906|Design audit]] documents dense-graph false positives and the distinction between target sensitivity and target-specific outcomes.
- [[controllability_edge_exploration_20260903|Controllability/edge exploration]], [[directional_predictive_recruitment_20260904|directional/predictive recruitment]], and [[topological_frontier_planning_batch|topological frontier planning]] contain the mechanism-level graph experiments.

**Current lesson:** edge count, reachability, SCC size, and option success are not sufficient evidence for command-conditioned navigation. Prefer matched commanded-versus-shuffled first-outcome tests.

### 3.5 Environment geometry and perceptual landmarks

Environment manipulations are orthogonal to controller architecture and should be analyzed as such.

- [[corridor_geometry_analysis_20260921|Corridor geometry analysis]] crosses architecture families with open/corridor geometry.
- [[easy_landmark_maze_qualification_analysis_20260924|Easy landmark maze qualification]] crosses SCR/DGP/Waypoint with rich versus neutral visual cues while holding maze geometry fixed.

**Current lesson:** making sensory states easier to distinguish can help some architectures without repairing a fundamentally goal-insensitive controller; conversely, corridor geometry can reduce exploration even if it simplifies topology.

### 3.6 Transfer after reward specification

Transfer is a separate scientific question: whether intrinsically learned structure is useful when a downstream reward is introduced.

- [[fixed_reward_transfer_latest_common_20260911|Fixed-reward latest-common comparison]] compares scratch, frozen DG, tuned DG, and policy transfer at a shared training window.
- [[fixed_reward_dg_peak_transfer_execution_20260924|DG-peak transfer execution]] and [[fixed_reward_transfer_implementation_20260910|transfer implementation]] document later transfer variants.
- [[cued_reward5_transfer_20260925|Five-cue reward transfer]] replaces the single fixed destination with five instructed reward locations and tests broader reuse.

**Current lesson:** scratch can optimize a narrow downstream task rapidly, so transfer should increasingly test many instructed destinations or reward changes where reusable representation/control structure has a plausible advantage.

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
3. [[recent_batches_design_audit_20260906|Design audit]] for the core graph/control failure mode.
4. [[navigation8_algorithm_screen_interim_20260916|Navigation8]] and [[cpu2048_analysis_20260917|CPU2048]] for matched architecture/controller comparisons.
5. [[ca3_predictive_active_goals_interim_analysis_20260924|Predictive CA3]] and [[ca3_state_goal_followup_interim_analysis_20260924|state-goal follow-up]] for the current goal-representation line.
6. [[easy_landmark_maze_qualification_analysis_20260924|easy-landmark cues]], [[corridor_geometry_analysis_20260921|corridor geometry]], and [[cued_reward5_transfer_20260925|five-cue transfer]] for environment and transfer tests.

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
