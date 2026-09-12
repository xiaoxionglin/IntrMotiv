# Restore the original IntrMotiv system; change controller learning only

**Coding-agent instructions — 12 September 2026**

## 0. Scope correction and definition of success

The user's intended intervention is:

> Take the promising, original IntrMotiv implementations in `xiaoxionglin/SF_hipposlam` and equip their existing controllers with off-policy Q-learning and HER. Preserve the rest of each implementation.

This supersedes the earlier instructions that made frozen DG, goals 1/4/11, a new first-arrival objective, and removal of the manager prerequisites for the main experiment. Those restrictions defined a diagnostic, not the requested algorithm upgrade.

**Do not deliver another frozen-reference local-control agent, even if it uses native Sample Factory processes.** Reusing Sample Factory execution and preserving IntrMotiv's algorithm are separate requirements. The native adapter already addresses much of the first; this task must address the second.

The required result is an opt-in controller-learning mode in the actual IntrMotiv training/model/learner path. With the new mode disabled, the original implementation must remain usable and unchanged in behavior. With it enabled, the parent's representation learning, intrinsic objectives, goal vocabulary, manager, graph, and exploration mechanisms must still operate as configured.

A frozen diagnostic, shadow learner, or tiny task may be used as a test. It is not the final implementation or the scientific comparison.

## 1. Establish the actual source and parent configurations

Use `xiaoxionglin/SF_hipposlam` as the implementation repository and `xiaoxionglin/IntrMotiv` as the accompanying study/notes repository. Inspect the real checkout used for the selected original runs, including tracked changes and necessary untracked files. Do not assume public `master` or the DDQN release copy exactly represents those runs.

Public `SF_hipposlam/master` inspected for this brief was commit `4e2f1123a8f9d8a66b607d8898776bcbd28a7d0b`. This is a reference for source discovery, **not an instruction to revert the runtime to that commit**. Earlier run records explicitly identify additional runtime modifications and an untracked goal-write module.

Read applicable `AGENTS.md`, the canonical workflow guide and `LATEST.md`. Start with these original implementation paths:

```text
sf_working_directories/IntrMotiv/dmlab/train_hipposlam.py
sf_working_directories/IntrMotiv/dmlab/custom_actor_critic.py
sf_working_directories/IntrMotiv/dmlab/custom_encoder.py
sf_working_directories/IntrMotiv/dmlab/custom_core.py
sf_working_directories/IntrMotiv/dmlab/custom_decoder.py
sf_working_directories/IntrMotiv/dmlab/custom_learner.py
sf_working_directories/IntrMotiv/dmlab/custom_params.py
sf_working_directories/IntrMotiv/dmlab/hrl_controllable_graph.py
sf_working_directories/IntrMotiv/dmlab/topological_frontier.py
sf_working_directories/IntrMotiv/dmlab/iterative_update.py
sf_working_directories/IntrMotiv/dmlab/dg_recruitment_graph.py
sf_working_directories/IntrMotiv/dmlab/reward_summaries.py
sf_working_directories/IntrMotiv/evaluation/
```

Also inspect the actual goal-write/context modules and any other enabled parent-specific modules, whether versioned in Git or included in the original source snapshot. These paths are a starting map, not permission to replace the existing code with look-alike implementations.

The original entry point already registers encoder, core, decoder, actor–critic and learner factories. Use these extension points. Review the native DDQN adapter for reusable transport and off-policy accounting; do not mistake its reduced scientific model for the model to preserve.

### Parent manifest

Resolve the currently selected promising configurations from canonical studies and actual run configs. Record each parent’s complete resolved config, source revision/diff and untracked-file hashes, checkpoint hash, source seed, frame count and enabled mechanisms.

Historical candidates already identified in this discussion include:

- `DGC_DIRECT_WORKER_F16_S99` at 25,001,984 frames.
- `DGC_WAYPOINT_DG_F64_S8` and `DGC_WAYPOINT_DG_F64_S99` at 25,001,984 frames.

These are a historical shortlist, not proof that they remain the current best. Do not reduce the task to the F16 diagnostic simply because that is the only parent accepted by the existing adapter. Include the selected direct and waypoint/goal-write families in the compatibility work. Do not combine their settings into a new hybrid architecture.

Separate **configuration replication** from **continuation of a selected checkpoint**. They answer different questions. Keep source seeds distinct from new learner seeds.

## 2. Make preservation a machine-checkable contract

Create a per-parent `preservation_manifest.json` and a generated config/parameter delta. Every difference must be classified as either a necessary controller-learning change or an explicitly approved experimental factor.

| Component | Required treatment |
|---|---|
| Environment, observations, depth/context bypass, action vectors and repeat, resets | Preserve the parent. Do not substitute a convenient diagnostic interface. |
| Frozen visual trunk | Keep the parent's preprocessing, parameters and normalization exactly as configured. |
| DG projection and normalization | Keep the parent's learning status, objectives, coefficients, optimizer schedule and normalization semantics. Do not freeze a trainable parent DG to simplify replay. |
| DG recruitment/retirement, contextual feedback, inhibition and other enabled mechanisms | Preserve when enabled; leave disabled when disabled. Do not add a new neurogenesis mechanism. |
| CA3 dynamics and memory size | Reuse the actual core, including distinct canonical and goal-conditioned worker traces where present. |
| Controller conditioning | Preserve FiLM/target trace, goal-write modulation, action/context information, and existing separate/shared branches. |
| Intrinsic rewards and worker reward | Reuse the source computation, scale and timing. Preserve `hit_distance`, temporal bonuses and exploration rewards when selected. Do not replace them with diagnostic reward 1 on first arrival. |
| Goal vocabulary and manager | Preserve the parent's eligibility, selection, frontiers, exploration fallback, RETURN/VALIDATE, waypoints and deadline rules. No hard-coded three-goal training registry. |
| Graph memory and updates | Preserve memory scope, confidence/reliability logic, real-data update cadence and generation invalidation. Replay is not new graph evidence. |
| Controller-to-DG gradient routing | Preserve STOP versus JOINT intent. Replacing the controller loss does not authorize disabling or enabling its representation gradient path silently. |
| Telemetry, checkpoints, evaluation and launching | Extend existing contracts; preserve the original metrics and execution workflow. |

Allowed changes are the action-value head/target network, Bellman loss, replay, HER and the action-selection/optimizer machinery necessary for off-policy controller learning. Additional clock inputs, extra hidden layers, fixed horizons, new auxiliary losses and global freezes are **not automatically allowed** merely because they existed in the diagnostic.

Expected behavior will change once a different controller selects actions. Preservation means the same surrounding mechanisms and objectives, not identical learning trajectories or identical rewards after behavior diverges.

## 3. Implement in the native original training path

Expose opt-in controller choices through the original parser, factories and training entry point. For example, an explicit PPO/DDQN controller mode plus a separate HER toggle. These are proposed new switches; document their real names after implementation.

The desired launch shape remains the original `sf_working_directories.IntrMotiv.dmlab.train_hipposlam` invocation with the original parent arguments and a small controller-specific delta. Do not make a fixed-reference `hpc_runs.intrmotiv_offpolicy.train` agent the only implementation of the feature. Shared off-policy utilities may remain in their existing package and be imported; there is no need to duplicate them or move files solely for appearance.

Use the existing SF runner, actors, inference, batching, parameter publication, checkpointing and summaries. Reuse the validated native transport adapter, ordered stream ingestion, exact update accounting and terminal-handling tests where compatible. Prefer extension hooks over changes to shared SF code. A small necessary shared-framework change is acceptable when opt-in, justified and covered by default-PPO regression tests; “all SF files unchanged” is not itself the scientific goal.

Keep the actual encoder/core/decoder and add the action-value output and target machinery at the controller boundary. Do not implement a second approximate CA3/manager pipeline in a standalone `QWorker` as the authoritative actor.

### Preserve existing useful behavior during integration

Initially support a shadow test in which the original controller still acts and the replay/Q side learns without affecting its parameters, graph or actions. This is an integration check and can supply a bounded, explicitly accounted warm-up. It must not silently become permanent PPO+DDQN hybrid training.

When DDQN takes control, remove PPO updates from that controller branch. If the parent already has a separately learned exploration branch, preserve that branch's intended behavior and update path; do not delete it or introduce an additional branch where the parent had none. A shared parent's no-target/exploration mode still needs its existing reward and behavior path represented by the chosen controller.

Retain compatible parent state and report new parameters. PPO policy logits are not Q-values; do not reinterpret them. If warm-up and cutover are used, record their duration, policy version and interaction cost. A fresh Q head necessarily changes action selection; describe that honestly rather than claiming policy-preserving conversion.

## 4. Keep representation learning on the original real-data schedule

Separate new-data work from replay work without duplicating the implementation:

```text
Original native SF actor:
  original observation -> original DG/core/manager -> selected controller -> action

Fresh physical rollout processing:
  original reward/event processing
  original graph and structural updates, at their original cadence
  original DG/auxiliary optimization and normalization schedule
  append complete, action-aligned physical experience to replay

Replay processing:
  sample original/HER controller examples
  reconstruct current/target model histories
  compute the parent-compatible controller target
  optimize the DDQN controller and its authorized gradient paths
  update the target model on its declared learner-update clock
```

Use source functions or minimally refactor shared pure calculations. Do not copy reward and DG-loss formulas into a second independent implementation.

Extra replay epochs must not multiply DG auxiliary updates, graph counts, visitation counts, normalization updates or retirement checks. Keep fresh-data DG batch formation and optimizer cadence traceable to the parent. For a JOINT parent, TD-gradient effects on DG are an intentional controller-gradient change and must be logged separately from its unchanged auxiliary update schedule. Do not substitute STOP as a convenience.

Do not run the PPO controller loss merely to obtain access to DG auxiliary logic. Separate the auxiliary path cleanly, including optimizer ownership and ordering, without accidentally stepping shared parameters twice.

## 5. Replay must support the actual adaptive representation

The cached frozen-preactivation format is insufficient for a parent whose DG learns. Store observations or the exact fixed-trunk inputs/features needed to rerun every trainable downstream component, together with the original depth/context bypass and relevant history. Cache only upstream computation that is genuinely fixed. Do not cache away a trainable depth or context module.

Preserve physical stream/episode/decision IDs, executed actions, the action-time command and manager mode, source/final-goal/waypoint identities as applicable, option/deadline metadata, real rewards/components, boundaries, policy/representation versions, and structural identity generations. Retain enough contiguous history to rebuild all enabled recurrent state. Do not join different streams or resets, or confuse an SF transport boundary with an environmental terminal.

Reconstruct the online model and target model using their declared parameter snapshots. Define target-network ownership explicitly, including any representation or write adapter that affects its value input. Model snapshots used to construct one target must be internally consistent.

For the simple fixed shift register a bounded washout prefix can be sufficient. Do not assume the diagnostic's 71-step prefix is sufficient for every contextual/recurrent parent. If exact finite washout does not apply, use and test an appropriate burn-in/state-version contract rather than substituting the simple core.

Replay must not update BatchNorm running statistics repeatedly. Preserve the parent's original normalization update schedule, and specify which immutable normalization snapshot is used for each replay calculation. Do not change the actor's normalization semantics merely to make a cache convenient.

After representation or goal-write parameters are published, prevent stale actor memory from silently mixing old and new encodings. Rebuild from buffered physical history or use a documented version-consistent update boundary. Do not silently zero recurrent memory mid-episode or freeze the encoder to avoid this requirement. Log rebuild/version failures.

### Label consistency without globally freezing DG

Keep raw experience and action-time labels for provenance. Define a single, documented snapshot-consistent procedure for constructing original and hindsight training examples. Where current detector recomputation changes achievements, rewards or termination, recompute the affected target and masks consistently or reject that example explicitly. Do not pair newly encoded state with incompatible old success labels without a specified semantics.

Preserve the parent's canonical recognition rule and the distinction between canonical recognition and goal-conditioned worker writes. Track continuous parameter versions separately from structural identity generations. A reassigned slot must not inherit another landmark's replay labels, goal parameters or graph evidence. Reuse existing generation/invalidation machinery and extend it to Q/target/replay state as necessary.

Do not turn all gradual DG updates into global retirement, clear the entire replay on every update, or introduce a permanent frozen reference network as a silent architecture change. If a selected parent's semantics cannot yet be supported, report the exact blocker and implement that support; do not declare a stripped-down substitute complete.

## 6. Define Bellman targets from the existing controller objective

Before coding the integrated loss, write a short objective/termination table from the parent code. Distinguish physical termination, truncation, manager retargeting, next-waypoint selection, option timeout, other-landmark arrival, and a replay chunk ending. Audit the original PPO credit-assignment boundaries rather than assuming they already terminate on every goal hit.

Use Double DQN's action selection/evaluation separation, but derive reward, masks, discount and successor conditioning from that table. In particular:

- Preserve the parent's actual reward, including goal-dependent bonuses and no-target exploration rewards. Do not replace it with the diagnostic's finite first-arrival task.
- Do not impose a fixed 64-decision deadline if the original manager uses learned/variable deadlines.
- Do not add finite-budget or episode-clock conditioning unless the chosen objective genuinely requires it. If required, expose it consistently and provide sufficient state interactions; document the minimum extra interface as a controller change.
- Never bootstrap an old goal's value against a different new goal accidentally. If the original objective deliberately continues through manager goal switches, implement and document that continuation rather than silently converting it to an option-terminal objective.
- For HER, specify the valid virtual worker task and its successor command/deadline context. If manager continuation cannot be reconstructed faithfully, do not invent it: define an explicitly bounded auxiliary worker target, keep original-return learning separate, and flag the objective difference for review before calling it a matched result.
- Do not apply the diagnostic's Q range of [0,1] to the original shaped reward. Derive diagnostics from the actual return definition.
- Do not route replay through PPO likelihood-ratio clipping or use PPO policy-age filters as off-policy validity criteria. Keep genuine version/identity/history checks.

Controller replacement unavoidably changes its optimizer and action selection. It must not silently change the task being optimized as well. If exact original return semantics and the proposed goal-local HER target differ, make that visible in the design and tests rather than hiding it in a `done` flag.

## 7. Add HER to the full worker, not to a three-goal replacement task

Real commands continue to come from the original manager and its full eligible vocabulary. A small evaluation panel may still be useful, but goals 1/4/11 must not constrain training by default.

Sample achieved goals from valid real history using the parent's canonical event/recognition function and identity generations. Recompute the goal-dependent reward components and outcome masks with the same source functions. Keep valid goal-independent components; skip or explicitly classify virtual examples whose required manager/mode context is unavailable. Do not fabricate physical actions, future state, return legs or validation successes.

Carry a coherent virtual command, start and deadline through every relabeled trajectory and suffix. Keep first relevant outcome/termination as defined by that parent's worker semantics; do not impose first-arrival termination on a parent that uses a different outcome rule. Never reset a virtual budget merely because a loss batch ends.

For `dg_goal_input=write` parents, reconstruct the worker trace under the virtual command history from the intervention point onward. Keep the physical pre-intervention history and canonical evidence distinct. Rebuild online and target traces separately when their write parameters differ. Test a relabel spanning multiple loss chunks. Changing only a one-hot at the decoder is not sufficient.

HER examples train the worker only. They do not count as new visits, completed intentional attempts, validated edges, DG recruitment opportunities or real exploration outcomes. Those updates come from actual accepted actor data exactly as in the parent.

Keep original-goal samples, failures and the parent's exploration-mode data. An 80% requested relabel rate is a hyperparameter, not an instruction to drop all difficult examples or change the actual manager's goals.

## 8. Preserve graph behavior without inheriting false controller guarantees

Retain the original graph/planning modules, passive evidence, lifecycle and exploration/validation policy. Do not remove the graph simply because its previous edges were learned with PPO.

At controller cutover, distinguish topological/recognition evidence from claims about a particular controller's reliability. Maintain controller-version provenance and use the original validation mechanism to refresh reliability under DDQN. This may require versioning or conservatively marking prior intentional-control evidence, but it must not erase unrelated topology or replace the manager with uniform commands.

Document any cutover reliability treatment and apply comparable treatment to controls when needed. Replaying a physical trajectory cannot repeatedly add the same evidence. Default PPO behavior must remain unchanged when the extension is disabled.

## 9. Acceptance tests: demonstrate preservation before performance

### A. Disabled-mode regression

Original checkpoints load. Original launch commands still work. With DDQN/HER off and identical seeds/inputs/actions, verify original logits/values, DG/core outputs, goal transitions, reward components, graph state and update routing. Preserve old checkpoint defaults and namespaces.

### B. Forced-action parity

On recorded physical inputs and forced actions, compare the untouched parent against the integrated path before the controller chooses different actions. Verify DG activity and normalization, canonical/worker traces, manager targets/modes/deadlines, reward timing, graph updates and structural events. This is the key test against silently changing the algorithm.

For a controlled batch, compare DG auxiliary losses and parameter updates with controller gradients isolated. Off-policy replay must not increase the count of those original updates.

### C. Adaptation and parent-specific support

Show that a trainable parent DG actually updates, that replay reconstructs changed representations instead of serving frozen cached preactivations, and that actor publication maintains a declared consistent memory. Test both direct and selected waypoint/goal-write parents. Exercise retirement/context paths only when enabled in selected parents; no invented all-features configuration is needed.

### D. Controller and HER correctness

Retain useful v2 tests: Double-DQN separation, target copies, sparse delayed rewards, correct suffix budgets when budgets apply, original/future-goal masking, final/reset observations, physical-order ingestion, online/target memory separation and train/evaluation parity. Adapt expected reward/termination to the real parent, not the diagnostic.

Use real production readout and reward functions in at least one multi-step test. One-step classification and finite loss alone are insufficient. Test that replay produces zero graph/normalization/structural side effects.

## 10. Scientific comparison and execution

For each selected original configuration, expose:

| Arm | Complete parent system | Controller learning | HER |
|---|---|---|---|
| Original reference | Yes | Original PPO/APPO | Original setting, recorded explicitly |
| DDQN | Yes | Replay-based Double DQN | Off |
| DDQN + HER | Yes | Same DDQN | On |

Do not substitute intact-parent zero-shot evaluation for a matched PPO training/continuation control. If a selected parent used the old biased PPO-HER auxiliary, record that fact and define the control separately; do not accidentally run both that auxiliary and DDQN HER on the same controller.

Hold each parent's non-controller configuration fixed. Use matched new interaction budgets, consistent parent checkpoints or matched initializations, and declared seed pairing. Report warm-up/cutover interactions and pretraining cost separately. Child seeds from one parent are not independent pretraining replicates.

Match actual TD positions/update intensity between DDQN and DDQN+HER; report PPO's different optimizer work rather than claiming numerical equality of unlike losses. Include performance per environment decision/frame, actual collection overhead, throughput, memory and replay cost.

Measure the original full-system outcomes: exploration coverage, intrinsic behavior, selected-goal/waypoint control, graph validation and route use, and representation diagnostics. Also use fixed matched-command evaluations across varied starts and goals, independently distinguishing detector activation from spatially grounded arrival. A three-goal diagnostic and lower TD loss do not establish improvement of the original algorithm.

Use tiny/synthetic tests, then bounded preflights with **the full selected parent mechanisms active**. Freeze-only preflights do not qualify adaptive production. Prepare the canonical study and print-only launch review. Do not start another large sweep until preservation and runtime gates pass and the experiment fits the user's existing resource authorization.

Work in an isolated branch/checkout with new output directories. Preserve all running jobs and existing artifacts. Use the established ordinary-job launcher and workspace-only bulk storage; do not create a new submission or telemetry framework.

## 11. Deliverables and stop conditions

Deliver:

1. An actual-source map and parent manifest, with unresolved runtime/public-source differences named explicitly.
2. A per-parent preservation table and machine-readable config delta: unchanged, necessary controller change, or separately approved change.
3. A small, reviewed patch in the original training/model/learner integration, reusing existing modules and validated off-policy infrastructure.
4. Disabled-mode, forced-action, DG-update, goal-write, reward, replay and terminal regression evidence.
5. A full-system smoke showing original DG learning, manager, graph, and relevant exploration mechanisms operating alongside DDQN/HER.
6. A canonical full-parent PPO/DDQN/DDQN+HER study with commands, checkpoint initialization/cutover policy, budgets and evaluation plan.
7. A concise report of what is complete, blocked and untested. Runtime speedups are not algorithm-preservation or navigation-success evidence.

**Do not claim completion if the implementation still freezes a trainable parent DG, removes its manager, restricts training to three goals, changes its reward to pure first-arrival success, or rejects a selected promising architecture by disabling its defining mechanism.** Report and fix the specific integration blocker instead.

## Source basis of this brief

This brief is an engineering instruction grounded in the user's correction, not a new performance claim. Sources inspected:

- Uploaded `Native Sample Factory DDQN/HER integration — 2026-09-11`: native runtime reuse, packet ordering, update accounting, and the explicit continued absence of adaptive DG and a manager; restricted parent support.
- Uploaded `DDQN/HER v2 repair and qualification`: diagnostic scope, coherent virtual attempts, target cadence tests, and normalization/replay limitations.
- Uploaded initial DDQN/HER implementation record: original live-source modifications, parent identity, and omitted adaptive/manager capabilities.
- `SF_hipposlam` original `train_hipposlam.py`, `custom_actor_critic.py`, and `custom_learner.py`: factory extension points, original decoders/branches/gradient boundaries, encoder objectives, recruitment and generation guards.
- `IntrMotiv/06_experiments/dg_capacity_goal_conditioning_interim_20260911.md`: historical candidate configurations, not a current definitive performance ranking.

The full deployed originals and live run artifacts must still be inspected by the coding agent. Public source alone is not proof of exactly which uncommitted runtime changes trained a checkpoint.
