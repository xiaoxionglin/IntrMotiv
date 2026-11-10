# Direct cue conditioning for external-reward transfer

Design date: 2026-11-04. Status checked: **2026-11-10, 22:36 UTC**. All **81 production training jobs**, three evaluation dispatchers, **189 field jobs and 93 heldout jobs completed successfully**. The outputs contain 189 field archives with trajectory CSVs and 324 protocol CSVs of 100 trials each. Scientific performance and field metrics have not yet been compared; the record below distinguishes current execution status from historical launch snapshots.

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

Every arm uses external-reward PPO, a fresh critic, fresh optimizer and counters, and a fresh five-cue mapping. Source transfer loads DG only for fresh-worker controls and DG plus decoder/action-head weights for pretrained-worker arms. The actor remains trainable in every cell. No source critic or graph is transferred. The frozen DG receives no intrinsic or PPO updates. Graph selection, recruitment, predictive auxiliaries, and contextual-goal recognition are disabled in this first batch. Its retained encoder-loss coefficient has no effect on frozen DG, but must be set explicitly when unfreezing it in the expansion below.

### Matched initialization

The new opt-in `fixed_task_goal_init=matched_cues` contract generates the same seeded, near-uniform effective cue vectors in both parameterizations. The simplex arm stores their logits; the unrestricted arm stores the effective vectors. Both fresh-worker arms receive the same seeded, nonzero FiLM table. The source-worker arms load the same source table.

This is necessary because the legacy free-vector path initializes identical cue rows and zero fresh FiLM weights, whereas the legacy mixture path introduces distinct rows. Merely toggling the old flag confounds parameterization with initialization and initially blocks cue gradients in the free-vector fresh-worker arm. Existing configurations retain their old initialization semantics.

Qualification checks paired effective cue vectors and initial action distributions, nonzero cue/worker gradients, actor/recurrent replay consistency, and absence of unrelated initialization changes. Initial equality is numerical to the verified floating-point tolerance, not a claim that trajectories remain identical after learning starts.

## First batch

The canonical source of run names, arguments, factors, seeds, metrics, and contrasts is [the production StudySpec](../../hpc_runs/studies/cue_transfer_20261104.study.json). Its [qualification StudySpec](../../hpc_runs/studies/cue_transfer_qualification_20261104.study.json) contains the same four conditions at seed 42 with a short horizon. Both use schema `intrmotiv/study/v1` and workflow `1.14.1`; validated fingerprints and release evidence are recorded below.

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

## Expansion to 81 training runs

The user requested approximately 80 runs on 4 November. Retain the original
12 jobs and add [23 conditions with the same three seeds](../../hpc_runs/studies/cue_transfer_expanded_20261104.study.json),
for **81 production runs in total**. Each targets 25M environment frames:
2.025B downstream frames, or 506.25M decisions at frameskip 4. Qualification
and offline prediction fits are separate diagnostics and are not counted as
production replicates. The extension is a nested set of causal comparisons,
not a complete factorial over all factors.

| Added family | New conditions | Runs | Question |
| --- | ---: | ---: | --- |
| DG learning routes, free cue | 8 | 24 | With fresh or source worker, what changes under BN-only, PPO-only, intrinsic-only, or joint DG adaptation? |
| PPO-adaptive DG, simplex cue | 2 | 6 | Does the cue-interface contrast persist when DG can learn from reward? |
| Fresh DG and worker | 4 | 12 | Compare once-calibrated frozen random DG with PPO-trained random DG, under both cue interfaces |
| Frozen source worker | 2 | 6 | Can the cue table and fresh critic adapt without changing the worker? |
| Credit horizon | 3 | 9 | Independently raise gamma to 0.999, GAE lambda to 0.99, or both, from the source-worker/free-cue reference |
| Global learning rate | 4 | 12 | Test 0.00005 and 0.0008 under each source-worker cue interface, around the original 0.0002 |
| **Added** | **23** | **69** | Original four conditions contribute the other 12 runs |

All conditions retain the frozen ImageNet visual trunk. “Fresh DG and worker”
is therefore scratch initialization of those modules, not training vision from
scratch. The source-DG arms all use the same checkpoint; downstream seeds do
not replace independent source-pretraining replication.

The four DG treatments explicitly control two gradient routes. PPO reaches DG
only with `ppo_dg_gradient=joint`; its original intrinsic loss is removed only
with `encoder_grad_coeff=0`. `extra_encoder_losses=False` disables additional
terms but does **not** remove the main intrinsic objective. Thus BN-only uses
`stop/0`, PPO-only `joint/0`, intrinsic-only `stop/1`, and joint `joint/1`, all
with `transfer_freeze_dg=False`. The audit checks weight changes separately
from running moments and verifies the actual zero/nonzero gradient routes.
BN-only still changes the representation through its normalization statistics.
The learner also retains its unit-row projection after each optimizer step.
For these already-normalized source weights, qualification found only
float32 roundoff (maximum absolute change $1.49\times10^{-8}$), with exactly
zero DG Adam moments. This is certified separately from learned DG updates;
BN-only does not promise bitwise projection identity. Frozen-DG controls do.

Interpret global learning-rate effects as optimizer robustness, not an isolated
softmax-geometry test: the actor, critic and cue parameters all change step
size. Gamma/lambda tests concern credit assignment and do not establish memory
necessity. Historical `F_DG`/`F_FULL` combined intrinsic and PPO updates; these
new matched-initialization route controls are not a resubmission of those runs.

The [23-cell qualification](../../hpc_runs/studies/cue_transfer_expanded_qualification_20261104.study.json)
uses one seed and 1,048,576 frames per condition. Its actual-source
[local tensor certificate](../data/cue_transfer_expanded_20261104/qualification/local_model_contract.json)
passes all 23 cells, including five-cue gradients, selective initialization,
fixed visual trunk/worker checks, and once-only random-DG calibration. A tiny
10-image probe initially had no active random-DG gradients; the common
100-image probe passed without changing the threshold or model. Native learner
activity, gradient routes, reward parity, and checkpoint restoration now pass
for all 23 cells; the deployment record below owns the production launch status.

## Outcomes and interpretation

Primary outcome: paired 0–25M external-reward AUC, with 0–5M and 0–10M curves exposing early adaptation. Confirm behavior at the nominal 25M endpoint after verifying a shared actual saved age, using a prespecified matched reset panel. Report every downstream seed and cue, physical success counts, and horizon-censored time to reward. Cue counts need not be equal: average the five per-cue outcomes with equal weight, retain each denominator, and flag any missing cue rather than silently changing the estimand.

The original principal contrasts are free minus simplex within each worker initialization, and pretrained minus fresh within each cue interface. Their interaction asks whether direct cue adaptation specifically unlocks pretrained control. The expansion adds the declared within-family contrasts against their matched original references; it does not support every cross-family comparison. Report all three seeds rather than selecting the best run. Multiple independent source checkpoints and training the visual backbone from scratch remain subsequent tests.

Diagnostics:

1. Real PPO reward equals the separately retained external stream, including positive-reward batches. DG weights, BatchNorm buffers, and gradient routes obey each condition's declared treatment; frozen source DG remains exact and calibrated random DG updates its moments once.
2. Cue vectors move; worker weights move when trainable and remain exact in the frozen-worker controls. Effective FiLM modulations and action probabilities distinguish cues. One-step sensitivity is a mechanism check, not task success.
3. Heldout correct-cue versus shuffled-policy-cue behavior uses identical physical resets and horizon, preserving the environment's true reward site. Additional zero-CA3 and zero-depth worker-input interventions assess policy dependence; zero-CA3 includes the current DG input as well as history, and these out-of-distribution masks do not isolate memory necessity. The extension is implemented; native qualification and production evaluation status are recorded below.
4. Reuse canonical DG/pre-threshold field, trajectory, occupancy, and common-observation analyses. Graph analysis is explicitly inapplicable because no graph is used. Do not substitute DG detector hits for physical external reward.

Use the [standard study workflow](../../04_implementation/standardized_study_workflow.md) and [manifest-driven telemetry](../../04_implementation/reusable_place_field_telemetry.md). All bulk outputs, logs, caches, and temporary data belong under `/work/classic/fr_xl1014-corridor-geometry`; the historical checkpoint is a read-only input. Keep only lightweight manifests, certificates, summaries, and selected figures in this repository.

### Deferred evaluation ownership

The [post-training dispatcher](../../hpc_runs/post_cue_transfer_20261104.py), Slurm job `8563772`, was submitted with an `afterok` dependency on all twelve production jobs. It reuses canonical checkpoint discovery, manifest rendering, and ordinary-job submission for 28 field rollouts: five milestones for seed 42 in each cell, plus terminal checkpoints for seeds 1234 and 9999. Before submitting, it rejects incomplete panels, reused milestone checkpoints, checkpoints earlier than their targets, and unequal actual saved ages. Actual frames remain in the manifests and evaluation audit; the nominal 25M label does not replace them.

The field evaluator retains its documented inclusive limit: `max_num_frames=10000` produces 10,001 decisions and occupancy samples. Report recorded sample counts.

The same dispatcher derives twelve terminal rows for the [heldout runner](../../hpc_runs/run_cue_transfer_heldout_single.sh), which retains the existing physical-reward protocol. The isolated evaluator now leaves terminal coordinates blank with `terminal_pose_valid=false` when the batched environment has already reset; success and latency are unchanged. True terminal-position claims remain unsupported by this path. Each run receives 100 stochastic-policy trials using requested reset seeds 61000–61099 and a 1,800-decision horizon. Validate identical realized starts, cues, reward centers, and engine seeds across compared cells before computing equal-cue-weighted contrasts. The dispatcher records both print-only plans before optional submission; its existence is not evidence that evaluation has run. These twelve original jobs remain correct-cue-only. The expanded release adds an intervention-only dispatcher for the original runs, without duplicating their standard evaluations. The 69 new runs received 161 field rollouts and 69 heldout jobs, each running all four protocols. The completed execution panel covers every one of the 81 production runs with the same correct-cue and three intervention protocols. Protocol settings and a manifest hash are persisted so a later single-row retry cannot silently change the test.

## Route to a world model

The separate prediction stage below reuses the existing `CA3StateReadout` and `CausalDGInnovationPredictor`, with one-decision prediction as its primary test. With frozen DG labels, predict the next full DG injection from current CA3 and the executed action, then reuse the known CA3 update in a subsequent recursive-model test:

$$
\hat u_{t+1}=f_\phi(W S_t,a_t),\qquad
\hat S_{t+1}=\operatorname{CA3Step}(S_t,\hat u_{t+1}).
$$

Require heldout improvement over state-only and persistence baselines, active-event accuracy, and adequate action support. The current auxiliary loss detaches DG/CA3, so it trains only the readout/predictor and cannot by itself make DG task-adaptive. Earlier predictive-signature collisions also preclude assuming a useful planning state from latent variance alone.

Only after this prediction gate should reward/termination heads and short lookahead be added. Imagined values must receive all necessary predicted state, including any bypass information they consume. The reward-relocation test should retain dynamics and measure new real decisions to a fixed physical-success level against external-reward fine-tuning alone, reporting internal model computation separately. No world-model improvement is claimed by this first transfer batch.

### Shared-trajectory prediction diagnostic

A separate frozen-actor collection uses the original `SOURCE_FREE` qualification
checkpoint at 1,081,344 frames, 60,000 decisions, and an independently seeded
20% uniform-action replacement probability. Full CA3 state is recorded before
the executed action and before the target observation is assimilated. Splits
hold out complete episodes; reset-crossing targets are excluded. Privileged
pose is telemetry only and never enters the predictor.

The existing readout/predictor is fit with and without actions, using the same
minibatches and paired readout initialization, with persistence as an additional
baseline. Three fit seeds test horizons 1 and 16. Report active-target and
zero-target errors separately, action support, action-shuffle sensitivity, and
the randomized-action subset when supported. H1 is primary; H16 uses the
recorded action prefix and is not a recursively imagined rollout. Shadow
prediction does not change the policy or DG. A predictive advantage would
support the next modeling step, not establish planning or better transfer.

Collection job `8563827` completed with 60,000 decisions across 37 episodes
(36 complete), all eight actions covered, and an unchanged model/buffer hash.
The episode split is 22 training, seven validation, and eight test episodes.
The [collection metadata](../data/cue_transfer_expanded_20261104/prediction/trajectory_metadata.json)
preserves action counts, reset seeds, causal alignment, and the dataset hash.

The first [paired fit](../data/cue_transfer_expanded_20261104/prediction/fit_results_8563828.json),
job `8563828`, passed execution/support checks but failed the overall predictive
utility check. Values below are held-out elementwise errors; trained models
are averaged over three fit seeds on the same test episodes.

| Horizon | Model | Overall MSE | Active-target MSE | Zero-target MSE |
| --- | --- | ---: | ---: | ---: |
| 1 | Action-conditioned | 0.7025 | 1.8871 | 0.6844 |
| 1 | State-only | 0.7230 | 1.8967 | 0.7050 |
| 1 | Persistence | 0.1016 | 3.9613 | 0.0424 |
| 16 | Action-conditioned | 0.8359 | 2.2725 | 0.8139 |
| 16 | State-only | 0.9753 | 2.3463 | 0.9543 |
| 16 | Persistence | 0.1472 | 5.3045 | 0.0681 |

Actions reduce active-target error relative to state-only by 0.51% at H1 and
3.15% at H16, but the active-weighted objective overpredicts inactive units.
Overall error is 6.9 and 5.7 times persistence, respectively. Shuffling actions
worsens active-target error while slightly improving overall error. These
results do not qualify the head as a useful world model.

The [objective control](../data/cue_transfer_expanded_20261104/prediction/objective_control_summary.json),
job `8563881`, then changed only the prediction objective and validation
selection to full-vector MSE. Dataset, splits, seeds, epochs, architecture, and
readout regularizers were identical; the original negative result is retained.

| Held-out full-vector MSE | Action-conditioned | State-only | Persistence | All-zero |
| --- | ---: | ---: | ---: | ---: |
| H1, all test transitions | 0.06015 | 0.06097 | 0.10155 | 0.08330 |
| H1, randomized-action subset | 0.05829 | 0.05995 | 0.08332 | 0.08198 |
| H16, recorded action prefixes | 0.07191 | 0.07422 | 0.14715 | 0.08364 |

The revised H1 action model improves overall MSE by 40.8% versus persistence,
27.8% versus zero, and 1.35% versus the matched state-only head. The action
benefit is positive in all three paired fit seeds and rises to 2.78% on
independently randomized actions. Shuffling action inputs now worsens full MSE
(2.36% at H1; 1.25% on randomized H1), consistent with modest action-dependent
predictive information. However, active-target error on randomized H1 remains
worse than persistence (3.436 versus 3.247); roughly 98.5% of target elements
are zero. Fit seeds share the same eight held-out episodes and are not
independent collection replicates. This selects a more useful offline
objective, without demonstrating recursive simulation, planning, or improved
external-reward learning.

## Qualification, deployment, and current status

The source checkpoint and saved config were verified on NEMO2 after manual login restored SSH access. Its declared SHA-256, eight-action interface, frameskip 4, and legacy pass-through depth match the downstream configuration. New outputs use the active corridor-geometry workspace; the historical checkpoint remains read-only.

### Original twelve-run release

| Artifact | Verified identity |
| --- | --- |
| Production StudySpec | `06e57d73b85023442d5f432dc1d2f841364b6d530b9c4e5f2b76773fc5af89f9` |
| Initial qualification StudySpec | `2d6cecc51d40001c544a1dfac6f5cf1aa1d4f0e7eae728849fdf674074d94879` |
| Single-cell continuation StudySpec | `9adaf12a7abbb3602ca760cb1fd111969137badd4611ab8cb5057670fd0a9652` |
| Training runtime / archive SHA-256 | `98a610e1d9471b09ad6f120814ae7a399380c97a` / `6a6a77fbf6964b7164427136026a1fac2947de53e6fae4213cb0ed58603724e3` |
| Evaluation runtime / archive SHA-256 | `920f807124759abbdb6667e179f9c9e66747006b` / `ec3e1efd0855f616b4d87d2ad82bb7a5f513cbd6702d459c035b0bd6082f3f54` |

Training uses `/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_cue_transfer_20261104`; evaluation uses the separate `source_cue_analysis_20261104` sibling. The continuation copy contains unchanged training code plus the two study-definition files committed in `55de52b3e18c1d6020f6f7be5bb17daf3f31a1b4`. The lightweight [training](../data/cue_transfer_20261104/launch/training_release.json), [evaluation](../data/cue_transfer_20261104/launch/evaluation_release.json), and [continuation](../data/cue_transfer_20261104/launch/continuation_release.json) release manifests preserve their exact identities. The original runtime checkout was not modified.

| Qualification | Verified result | Evidence |
| --- | --- | --- |
| Actual-source tensor probes | Paired initial policies match; all five cues receive gradients; selective loading and frozen DG/BN checks pass | [Synthetic RGBD probe](../data/cue_transfer_20261104/qualification/local_model_parity.json); this is not an environment rollout |
| Focused tests | 114 training/workflow tests plus 7 new evaluation tests pass locally and on NEMO2 | Immutable releases above |
| Real external-reward learning | All four cells pass reward parity, worker/cue movement and exact frozen DG/BN checks | [Combined qualification](../data/cue_transfer_20261104/qualification/verified/qualification_summary.json), job `8563732` |
| Exact learner restoration | Four of four model/buffer, optimizer and counter restores pass on GPU | Per-run certificates retained under [verified evidence](../data/cue_transfer_20261104/qualification/verified/reloads/); the combined qualification preserves original NEMO2 paths |
| Matched evaluator age | All four retained milestones are exactly 262,144 frames | [Canonical manifest](../data/cue_transfer_20261104/qualification/verified/telemetry/analysis_manifest.tsv) |
| Native heldout evaluator | Two complete trials reproduce exactly under the same reset and policy seeds; terminal-pose reporting passes | [Native certificate](../data/cue_transfer_20261104/qualification/native/CTQ_D51_SOURCE_FREE_S42/heldout_native_qualification.json), job `8563742`; this is an execution check, not a success-rate estimate |
| Ordinary field evaluator | Completed nominal 10k rollout: 10,001 aligned trajectory/occupancy samples, 64 DG units, valid pre-threshold maps | [Field certificate](../data/cue_transfer_20261104/qualification/field_smoke_certificate.json), ordinary job `8563754` |

The four initial learners (`8563708`–`8563711`) exited successfully. Their H200 qualification followed an execution-only H200/L40S override when the original L40S queue had no available GPU; production retains the L40S profile. The short [initial audit](../data/cue_transfer_20261104/qualification/initial_transfer_audit.json) passed three cells. `SOURCE_FREE` collected three physical rewards but its 12 sampled minibatch summaries contained no positive reward. The [single-cell continuation](../../hpc_runs/studies/cue_transfer_positive_20261104.study.json), job `8563731`, changed only the horizon and explicit resume setting. Its final 1,081,344-frame checkpoint passed, including three positive batches among 35 summaries. The other cells were certified at 294,912 frames. These unequal qualification durations do not enter the production comparison; heldout qualification still uses the common earlier milestone.

Native search `8563712` verified cue changes, spawn exclusion, and +10 terminal cue-3 contact at seed 1292, then was deliberately stopped. Its sparse near-spawn filter and sequential opposite-action probes confound task validity with a biased controller. Structural tests cover all five unchanged reward sites; all-five physical contacts are not claimed. See the [infrastructure finding](../../infra.md#native-reward-contact-rediscovery-mixed-task-validity-with-a-biased-controller--2026-11-04). The initial failed audit and never-started dependent checks remain historical attempts, not successful qualifications.

The [production print-only audit](../data/cue_transfer_20261104/launch/print_only_audit.json) and [submission audit](../data/cue_transfer_20261104/launch/submission_audit.json) pass for all twelve commands and workspace paths. Jobs `8563760`–`8563771` were accepted with the unchanged L40S profile: one GPU, 40 CPUs, 80 GB, and 24 hours per run. The [job manifest](../data/cue_transfer_20261104/launch/production_jobs.tsv) preserves each command and its job ID. At the [02:02 UTC scheduler snapshot](../data/cue_transfer_20261104/launch/scheduler_snapshot.json), all twelve were pending for priority; none had started training.

The [evaluation submission record](../data/cue_transfer_20261104/launch/post_evaluation_submission.json) and [submitted wrapper](../data/cue_transfer_20261104/launch/post_evaluation.sbatch) pin dispatcher `8563772`. Its dependency on all twelve training jobs was verified directly with `scontrol`. At that launch snapshot it was pending on the dependency; the 28 field and 12 heldout jobs had not yet been submitted. If dispatch later stops after partial submission, use the accepted-row submission manifests and the canonical submitter's `--row` option to schedule only missing rows; do not rerun the whole dispatcher blindly.

### Expanded qualification and deployment

The expansion preserves both original and new StudySpec fingerprints. New
studies declare workflow `1.15.0`; compatible qualification implementation
`1.15.2` fixes audit semantics without changing training code or arguments.

| Artifact | Verified identity |
| --- | --- |
| Added 69-run production StudySpec | `247f24f90a9e6b85e2c5badbf68a60ce0628448c8b5ede37e6712cea6de17cde` |
| Added 23-cell qualification StudySpec | `ae88a15fda89774ccc463cdd44a1a8c013741f9c6866ddf61ff6160dd503513d` |
| [Training release](../data/cue_transfer_expanded_20261104/launch/training_release.json) | Runtime `be7fcbab24c22872972696b6b5a38116258acc91`; archive `dbc0818680bae3568066059b07fde2f85afa5dc8428a725c7b7c6d154757a458` |
| [Qualified intervention release](../data/cue_transfer_expanded_20261104/qualification/native/analysis_fix_RELEASE.json) | Runtime `1cf646791dc07cebdaac838d2362606d4a5507d7`; archive `14808f1f22fdfa4de17f1a40031ee3d44128f7c9d798bf4e6a4f4cd860938ea0` |
| [Final qualification release](../data/cue_transfer_expanded_20261104/qualification/audit_release.json) | Runtime `d5fe0218fc012e12f82848f8f55cb5aa5fb1f650`; archive `323929b22ed8780ee5f0694af13599cb0c4b0261bfc63f24a15c6a29f68c5932` |

All 23 short learners completed successfully. Their [submission audit](../data/cue_transfer_expanded_20261104/qualification/submission/audit.json)
and [job manifest](../data/cue_transfer_expanded_20261104/qualification/submission/jobs.tsv)
retain two quota-bounded submission waves. Recovery gate `8563922` completed
successfully in 6m35s. Its [aggregate certificate](../data/cue_transfer_expanded_20261104/qualification/verified/job_8563922/qualification_summary.json)
passes all 23 fresh transfer audits and all 23 checkpoint restorations, with
nine earlier certificates reused and fourteen newly checked. The 46-row field
manifest has exact shared checkpoint ages of 524,288 and 1,048,576 frames. Exact
CUDA replay confirms the BN-only projection roundoff. Nineteen restores have
raw optimizer equality; four instead satisfy the explicit scheduled-LR init
transition, with raw equality correctly reported as false.

Three qualification corrections are retained as evidence:

- The original audit treated BN-only projection weights as bitwise fixed.
  The [numerical diagnosis](../data/cue_transfer_expanded_20261104/qualification/verified/job_8563876/bn_constraint_diagnosis.json)
  and exact CUDA replay instead identify the learner's existing unit-row
  normalization roundoff, with zero Adam moments and no learned projection
  change. Frozen-DG comparisons remain bitwise strict.
- Some reloads correctly restored scheduled `curr_lr`, replacing the previous
  minibatch's temporary valid-sample reduction. The revised certificate reports
  raw optimizer equality separately and verifies the complete expected
  learner-init transition, with all moments, counters and non-LR fields exact.
  It does not claim bitwise resumed trajectories or generic RNG replay.
- Native job `8563826` preserved valid correct/shuffled controls but rejected a
  ten-depth-plus-three-map bypass before applying its first mask. Corrected job
  `8563886` completed only the two missing masks. The [combined certificate](../data/cue_transfer_expanded_20261104/qualification/native/native_intervention_qualification.json)
  verifies matching starts, engine/policy seeds, true reward site, intended cue
  changes and terminal reporting across all four protocols. This is one trial
  per protocol, qualifying execution rather than performance.

The original serial gate `8563876` was intentionally cancelled after preserving
nine completed restore certificates. The recovery gate reuses only complete,
hash-verified certificates with their original job/release provenance, and runs
four independent missing checks concurrently. Native environment setup took
about 101 seconds per check while model restoration took under a second.
The [replacement receipt](../data/cue_transfer_expanded_20261104/qualification/verified/job_8563922/recovery_gate_submission.json)
also records successful accounting for all 23 training prerequisites; older
completed IDs had expired from Slurm's live dependency table.

The updated auditor/reloader passes 113 focused tests plus six negative
subtests locally and on NEMO2. Runtime [formatting](../data/cue_transfer_expanded_20261104/launch/formatting_ci_receipt_5567784b.json)
and [final functional CI](../data/cue_transfer_expanded_20261104/launch/auditor_1_15_2_ci_receipt_d5fe0218.json)
are green. Local copied tables preserve every parsed value; [normalization metadata](../data/cue_transfer_expanded_20261104/table_copy_normalization.json)
distinguishes LF Git copies from raw remote byte hashes.

### Accepted 81-run production batch

The [10 November execution check](../data/cue_transfer_expanded_20261104/status/execution_status_20261110.json)
verifies `COMPLETED/0:0` for all 81 training jobs, all three dispatchers and all
282 child evaluation jobs, using the canonical submission manifests and Slurm
accounting. All 189 field NPZs and their 189 trajectory CSVs are present. The
324 heldout protocol CSVs each contain 100 trial rows, for 32,400 recorded
trials. This check establishes execution and output coverage; it does not yet
establish checkpoint-content validity, reset matching across conditions,
adaptation gains, place-field quality, or a learned planning model. The full
scientific analysis remains to be completed under the standard workflow.

The [added-run submission audit](../data/cue_transfer_expanded_20261104/launch/submission_audit.json)
verifies all 69 commands against the unchanged expanded StudySpec, with valid
workspace output paths. The [job manifest](../data/cue_transfer_expanded_20261104/launch/production_jobs.tsv)
records jobs `8563937`–`8563940` and `8563949`–`8564013`. They use the qualified
training release, one L40S GPU, 40 CPUs, 80 GB and a 24-hour limit per run.
Together with original jobs `8563760`–`8563771`, these are exactly 81 unique
production jobs targeting 25M frames each. The [03:13 UTC scheduler snapshot](../data/cue_transfer_expanded_20261104/launch/scheduler_snapshot.json)
shows all 81 pending for priority and all three evaluation dispatchers pending
on dependencies. None had started training at that snapshot.

Dispatcher [8564015](../data/cue_transfer_expanded_20261104/launch/post_expanded_evaluation_submission.json)
depended on all 69 new jobs and submitted 161 field rollouts plus 69 heldout
jobs with four protocols each. Dispatcher [8564016](../data/cue_transfer_expanded_20261104/launch/post_parent_interventions_submission.json)
depended on the original twelve jobs and added only their three missing
intervention protocols. Original dispatcher `8563772` retains its 28 field and
twelve correct-cue heldout jobs. The dependency sets were verified with
`scontrol`. At launch only the dispatchers were scheduled; the 10 November
check above confirms that their evaluator jobs subsequently completed. The
full execution panel is **189 field rollouts and 93 heldout
jobs covering 324 protocol evaluations**, with 100 trials per protocol. Graph
analysis remains inapplicable because these conditions use no graph.

The runtime branch is pushed through documentation commit `fbcba79a`; its
[CI and remote-parity receipt](../data/cue_transfer_expanded_20261104/launch/docs_sync_ci_receipt_fbcba79a.json)
is green. The immutable training, evaluator and qualification releases retain
their separate identities in the table above.

## Reusable lesson

Inspect actual gradient routing and effective initialization before treating two CLI settings as a controlled comparison. The existing conditioning and transfer code supply nearly all required behavior; the small new contract removes an initialization confound. Preserve source action timing and depth preprocessing explicitly, reuse the canonical StudySpec/launcher/auditor, and distinguish a submitted job from a qualified, advancing learner. The newer package-transfer result should remain linked from the experiment index so future work does not repeat its collection.

The slow steps were repeated native DMLab initialization and biased reward-contact rediscovery. Use existing structural fixtures and bounded real learner/evaluator checks for an unchanged task. Authoritative launch evidence is the validated StudySpec, `audit-submission`, checkpoint-content audits, and `sacct`/`scontrol`; sparse minibatch summaries and job submission alone are insufficient. Consult the telemetry contract before asserting rollout counts: the existing inclusive limit records 10,001 decisions for a nominal 10k request.

For plasticity comparisons, trace the main loss as well as auxiliary switches:
`extra_encoder_losses=False` does not remove intrinsic DG credit. Sparse random
DG also needs enough common probe observations to exercise its gradients; native
learner evidence remains the release gate. Bounded submission waves reuse the
canonical manifest and resumer to respect partition quotas. Prediction models
share one recorded dataset and episode split so differences cannot come from
different collection policies.
