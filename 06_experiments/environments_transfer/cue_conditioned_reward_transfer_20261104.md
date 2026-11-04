# Direct cue conditioning for external-reward transfer

Date: 2026-11-04. Status: design implemented; compute-node qualification in progress; production has not been submitted.

## Decision and hypothesis

Reuse the pretrained DG–CA3 representation and worker, but give each downstream reward cue a directly learned conditioning vector. Train the worker and a fresh value head on external reward. The first experiment keeps DG, its BatchNorm statistics, and the visual trunk fixed so the result measures adaptation of the control interface and worker.

The question is whether an unrestricted cue vector adapts faster than a softmax mixture of source-goal vectors, and whether pretrained worker weights help under either interface. This is an optimization and information-flow hypothesis, not a claim that the existing representation already supports every reward location.

The same mechanism applies to a task identifier in a nonspatial domain: a task-conditioned actor learns which actions produce external reward. Coordinates, reward-site distances, and privileged task labels beyond the supplied cue are excluded from policy inputs.

## Evidence motivating this experiment

- [Frozen DG alone](../results/A0_poster_analysis_20260926/batch_summary.md) has lower 0–75M reward AUC than calibrated random DG in all six downstream seed pairs. At 75M, D50 heldout success is 50.0% versus 58.3%; D51 is 47.5% versus 61.7%.
- [Transferred DG–worker–graph packages](../../05_plans/poster_20260929/worker_random_transfer/README.md) improve mean 0–75M logged reward by 6.1% and 15.8%, but their 0–10M means are 18.0% and 23.4% lower. Package heldout evaluation is still missing. Reuse is promising, but faster adaptation is unproved.
- The [five-cue audit](../cued_reward5_transfer_20260925.md#reward-learning-and-goal-selection-at-50m) found almost-uniform reward-manager selection in the frozen-DG controls: maximum probability 0.0167–0.0177 versus uniform 0.0159. The task cue reaches the worker through the selected goal. This motivates direct cue conditioning, but does not establish that manager selection is the only bottleneck.
- Existing flat `F_FULL` and `F_DG` runs already use differentiable cue mixtures while training DG. Their complete matched results remain unreported locally. This pilot differs by freezing the same source DG in all cells and directly comparing conditioning parameterizations. Historical unreported runs must not be resubmitted as duplicates.

## Model and learning rule

Let $S_t$ be canonical CA3 memory, $b_t$ the existing depth bypass, and $c$ the environment's number cue. The actor uses the existing FiLM-conditioned worker:

$$
\pi_\theta(a_t\mid S_t,b_t,e_c).
$$

Only the parameterization of the learned cue table changes:

$$
e_c=\operatorname{softmax}(q_c)\quad\text{or}\quad e_c=v_c\in\mathbb{R}^{64}.
$$

The unrestricted vector can move directly under external-reward gradients. The softmax vector stays on the probability simplex. Since the FiLM table is trainable, both interfaces can represent rich cue-dependent modulations; this comparison does not isolate a fixed convex-hull capacity limit. Near-uniform softmax changes gradient scaling as well as parameter geometry. A positive result would motivate a later learning-rate-matched diagnostic, not establish which effect caused it.

Every arm uses external-reward PPO, a fresh critic, fresh optimizer and counters, and a fresh five-cue mapping. Source transfer loads DG only for fresh-worker controls and DG plus decoder/action-head weights for pretrained-worker arms. The actor remains trainable in every cell. No source critic or graph is transferred. Intrinsic encoder objectives, graph selection, recruitment, predictive auxiliaries, and contextual-goal recognition are disabled in this batch.

### Matched initialization

The new opt-in `fixed_task_goal_init=matched_cues` contract generates the same seeded, near-uniform effective cue vectors in both parameterizations. The simplex arm stores their logits; the unrestricted arm stores the effective vectors. Both fresh-worker arms receive the same seeded, nonzero FiLM table. The source-worker arms load the same source table.

This is necessary because the legacy free-vector path initializes identical cue rows and zero fresh FiLM weights, whereas the legacy mixture path introduces distinct rows. Merely toggling the old flag confounds parameterization with initialization and initially blocks cue gradients in the free-vector fresh-worker arm. Existing configurations retain their old initialization semantics.

Qualification checks paired effective cue vectors and initial action distributions, nonzero cue/worker gradients, actor/recurrent replay consistency, and absence of unrelated initialization changes. Initial equality is numerical to the verified floating-point tolerance, not a claim that trajectories remain identical after learning starts.

## First batch

The canonical source of run names, arguments, factors, seeds, metrics, and contrasts is [the production StudySpec](../../hpc_runs/studies/cue_transfer_20261104.study.json). Its [qualification StudySpec](../../hpc_runs/studies/cue_transfer_qualification_20261104.study.json) contains the same four conditions at seed 42 with a short horizon. Both use schema `intrmotiv/study/v1` and workflow `1.14.1`; validated fingerprints and release evidence are recorded below after validation.

| Worker initialization | Cue interface | Interpretation |
| --- | --- | --- |
| Pretrained | Simplex mixture | Frozen-representation transfer reference |
| Pretrained | Unrestricted vector | Proposed low-effort adaptation interface |
| Fresh | Simplex mixture | Worker-transfer control under existing parameterization |
| Fresh | Unrestricted vector | Worker-transfer control under proposed parameterization |

All four cells share one frozen D51 source DG and its normalization state. These fresh-worker arms are **not full scratch models**. The factorial estimates worker initialization and cue-interface effects conditional on one selected source representation.

- Production: three paired downstream seeds, 42/1234/9999; 25M environment frames per run; 12 runs and 300M total downstream frames (75M decisions at frameskip 4).
- Qualification: four runs initially targeting 262,144 frames each; one prespecified-configuration continuation to 1,048,576 frames was added after the short audit lacked a positive logged minibatch. Qualification budgets are not performance comparisons. A successful parser or running Slurm job is insufficient: complete checkpoints, positive external-reward batches, parameter updates, frozen-state parity, and reward-path checks must pass before production submission.
- Production milestones: 1M, 5M, 10M, 15M, and 25M frames. This is a bounded adaptation pilot. A null at 25M does not establish a final transfer deficit; historical package gains appeared later. Any 75M continuation should retain the matched conditions and seeds rather than select favorable seeds.
- Environment: existing five invisible reward cells in the source visual arena, number cues 1–5, terminal +10 physical-entry reward. No new map or reward shaping is introduced.
- Source-compatible interface: ImageNet layer-2 ResNet-18 trunk; 64 DG units; canonical CA3 with R=8 and L=64; eight navigation actions; frameskip 4; source map-3 sensory instruction; pass-through depth explicitly selected with `depth_sensor_inverse=false`.
- Optimization: the historical external-reward PPO settings are retained, including learning rate 0.0002, gamma 0.99, GAE lambda 0.95, and reward scale 0.1. The exact complete arguments are owned by the StudySpecs.

The selected source is `00_FSCS_WAYPOINT_DECODER_F64_PPO_S123`, checkpoint `checkpoint_000021868_179142656.pth`, SHA-256 `2cbc50fca8658f938775496956502d24a52ed8e437037f0bf7fe0e32574ae694`. Its original pretraining without external task reward is additional lifetime compute. Downstream seed replication does not replicate source pretraining.

## Outcomes and interpretation

Primary outcome: paired 0–25M external-reward AUC, with 0–5M and 0–10M curves exposing early adaptation. Confirm behavior at the nominal 25M endpoint after verifying a shared actual saved age, using a prespecified matched reset panel. Report every downstream seed and cue, physical success counts, and horizon-censored time to reward. Cue counts need not be equal: average the five per-cue outcomes with equal weight, retain each denominator, and flag any missing cue rather than silently changing the estimand.

The principal contrasts are free minus simplex within each worker initialization, and pretrained minus fresh within each cue interface. Their interaction asks whether direct cue adaptation specifically unlocks pretrained control. A full-scratch utility comparison and multiple independent source checkpoints are subsequent tests if the pilot supports the mechanism.

Diagnostics:

1. Real PPO reward equals the separately retained external stream, including positive-reward batches; frozen DG weights and BatchNorm buffers remain exactly unchanged.
2. Cue vectors and worker weights move; effective FiLM modulations and action probabilities distinguish cues. One-step sensitivity is a mechanism check, not task success.
3. Heldout correct-cue versus shuffled-cue behavior uses identical physical resets and horizon. This evaluator extension must preserve the environment's true reward site while changing only the policy cue; it is not yet a completed evaluation.
4. Reuse canonical DG/pre-threshold field, trajectory, occupancy, and common-observation analyses. Graph analysis is explicitly inapplicable because no graph is used. Do not substitute DG detector hits for physical external reward.

Use the [standard study workflow](../../04_implementation/standardized_study_workflow.md) and [manifest-driven telemetry](../../04_implementation/reusable_place_field_telemetry.md). All bulk outputs, logs, caches, and temporary data belong under `/work/classic/fr_xl1014-corridor-geometry`; the historical checkpoint is a read-only input. Keep only lightweight manifests, certificates, summaries, and selected figures in this repository.

### Deferred evaluation ownership

The [post-training dispatcher](../../hpc_runs/post_cue_transfer_20261104.py) is intended to run after all twelve production jobs succeed. It reuses canonical checkpoint discovery, manifest rendering, and ordinary-job submission for 28 field rollouts: five milestones for seed 42 in each cell, plus terminal checkpoints for seeds 1234 and 9999. Before submitting, it rejects incomplete panels, reused milestone checkpoints, checkpoints earlier than their targets, and unequal actual saved ages. Actual frames remain in the manifests and evaluation audit; the nominal 25M label does not replace them.

The same dispatcher derives twelve terminal rows for the [heldout runner](../../hpc_runs/run_cue_transfer_heldout_single.sh), which retains the existing physical-reward protocol. The isolated evaluator now leaves terminal coordinates blank with `terminal_pose_valid=false` when the batched environment has already reset; success and latency are unchanged. True terminal-position claims remain unsupported by this path. Each run receives 100 stochastic-policy trials using requested reset seeds 61000–61099 and a 1,800-decision horizon. Validate identical realized starts, cues, reward centers, and engine seeds across compared cells before computing equal-cue-weighted contrasts. The dispatcher records both print-only plans before optional submission; its existence is not evidence that evaluation has run. Correct-cue versus shuffled-cue evaluation remains a separate pending extension and is not part of these twelve jobs.

## Route to a world model

The next separate stage reuses the existing `CA3StateReadout` and `CausalDGInnovationPredictor`, initially at one decision. With frozen DG labels, predict the next full DG injection from current CA3 and the executed action, then reuse the known CA3 update:

$$
\hat u_{t+1}=f_\phi(W S_t,a_t),\qquad
\hat S_{t+1}=\operatorname{CA3Step}(S_t,\hat u_{t+1}).
$$

Require heldout improvement over state-only and persistence baselines, active-event accuracy, and adequate action support. The current auxiliary loss detaches DG/CA3, so it trains only the readout/predictor and cannot by itself make DG task-adaptive. Earlier predictive-signature collisions also preclude assuming a useful planning state from latent variance alone.

Only after this prediction gate should reward/termination heads and short lookahead be added. Imagined values must receive all necessary predicted state, including any bypass information they consume. The reward-relocation test should retain dynamics and measure new real decisions to a fixed physical-success level against external-reward fine-tuning alone, reporting internal model computation separately. No world-model improvement is claimed by this first transfer batch.

## Qualification, deployment, and current status

The source checkpoint and its actual config were inspected on NEMO2 after manual login restored SSH access. Its SHA-256 matches the declared source, including the eight-action interface and legacy pass-through depth. All new outputs use the active corridor-geometry workspace; the old source checkpoint is read-only.

| Artifact | Verified identity |
| --- | --- |
| Production StudySpec | `06e57d73b85023442d5f432dc1d2f841364b6d530b9c4e5f2b76773fc5af89f9` |
| Qualification StudySpec | `2d6cecc51d40001c544a1dfac6f5cf1aa1d4f0e7eae728849fdf674074d94879` |
| Immutable training runtime | `98a610e1d9471b09ad6f120814ae7a399380c97a` on `codex/cue-transfer-20261104` |
| Training archive | `6a6a77fbf6964b7164427136026a1fac2947de53e6fae4213cb0ed58603724e3` |
| Separate evaluation runtime | `920f807124759abbdb6667e179f9c9e66747006b` |
| Evaluation archive | `ec3e1efd0855f616b4d87d2ad82bb7a5f513cbd6702d459c035b0bd6082f3f54` |

Training uses `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_cue_transfer_20261104`; evaluation uses its separate `source_cue_analysis_20261104` sibling. The original runtime checkout and training release remain unchanged. The [training release manifest](../data/cue_transfer_20261104/launch/training_release.json) pins its parent and source directory.

- Actual-source local probes passed: matching initial policies within each worker pair, all five cue gradients, selective source loading, fresh critic, and frozen DG weights/normalization buffers. [Probe evidence](../data/cue_transfer_20261104/qualification/local_model_parity.json) is explicitly synthetic RGBD tensor execution, not an environment rollout.
- The deployed training release passed 114 focused tests. The separate evaluation release passed seven additional regression tests on the desktop and NEMO2.
- [Qualification submission audit](../data/cue_transfer_20261104/qualification/submission_audit.json): four exact StudySpec commands, jobs `8563708`–`8563711`. The L40S-only queue delayed startup; a recorded execution-only adjustment allowed H200/L40S and shortened the reservation to 30 minutes. All four started on H200. Production retains its separate L40S resource profile.
- Native reward-engine search `8563712` was deliberately stopped after verifying cue changes, spawn exclusions, and +10 terminal cue-3 contact at seed 1292. Its sequential opposite-action probes are directionally biased and its near-spawn filter is sparse, so it cannot serve as an all-five-site validity gate. The unchanged level retains structural tests for every site; all-five physical contacts are not claimed. Checkpoint/reward-path qualification `8563716` depends on the four successful learners; the independent search was removed from that dependency. Native heldout reproducibility and ordinary field-smoke dispatch: `8563721`, dependent on checkpoint qualification. The first checkpoint audit passed three of four cells; `SOURCE_FREE` had three native reward pickups but no positive batch among its 12 logged minibatch summaries. The [single-cell continuation](../../hpc_runs/studies/cue_transfer_positive_20261104.study.json), SHA-256 `9adaf12a7abbb3602ca760cb1fd111969137badd4611ab8cb5057670fd0a9652`, resumes its existing state to 1,048,576 frames as job `8563731`. Only the horizon and explicit resume setting change. The original failed audit remains [archived](../data/cue_transfer_20261104/qualification/initial_transfer_audit.json); never-started native job `8563721` was cancelled after its dependency failed. Final combined qualification is pending.
- [Production print-only audit](../data/cue_transfer_20261104/launch/print_only_audit.json) passes for all twelve jobs. Production submission remains gated on compute-node qualification and ordinary evaluator smoke.

## Reusable lesson

Inspect actual gradient routing and effective initialization before treating two CLI settings as a controlled comparison. The existing conditioning and transfer code supply nearly all required behavior; the small new contract removes an initialization confound. Preserve source action timing and depth preprocessing explicitly, reuse the canonical StudySpec/launcher/auditor, and distinguish a submitted job from a qualified, advancing learner. The newer package-transfer result should remain linked from the experiment index so future work does not repeat its collection.
