# DG capacity and goal conditioning: verified production launch

**Status: all 27 production runs are running and advancing.** Verified on
10 September 2026 at **20:44 CEST** (18:44 UTC). Every run has its preserved
frame-zero checkpoint, matches the declared configuration, and has nonzero
training progress. Observed progress ranged from 327,680 to 1,015,808 frames.

Production jobs: **8049414–8049440**. The
[authoritative job manifest](data/dg_capacity_goal_conditioning_20260910/production_jobs.tsv)
contains the exact run-to-job mapping and commands. The
[submission audit](data/dg_capacity_goal_conditioning_20260910/production_submission_audit.json)
and [startup audit](data/dg_capacity_goal_conditioning_20260910/production_startup_audit.json)
both pass. No startup tracebacks were found in stdout or stderr.

## Scientific scope and provenance

The [agreed plan](dg_capacity_goal_conditioning_plan_20260910.md) specifies three
arms: C15 direct with worker conditioning, C15 direct with DG + worker
conditioning, and waypoint execution with DG + worker conditioning. Each uses
DG 16/32/64 and seeds 8/99/123. All start fresh, use **frameskip 4, navigation8,
and a 300M-frame target per run**. Goal discrimination, waypoint worker-only,
compact goal embeddings, and new goal-selection heuristics are excluded.

Production StudySpec: `hpc_runs/studies/dg_capacity_goal_conditioning.study.json`.
Schema: `intrmotiv/study/v1`; workflow: **1.6.0**. SHA-256:
`eeb8bafd190ceb15da1fa6066532cb5f44b20c742cc4c1b9fb56b698fe01f47e`.

Final preflight StudySpec:
`hpc_runs/studies/dg_capacity_goal_conditioning_preflight.study.json`.
It declares compatible workflow 1.5.0 and SHA-256:
`dbbfc58291a8e76197f94f3e6208c2324f92234510055733796a440af5490069`.

All training, logging, checkpoints, caches, and analysis data resolve under
`/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/`.
The production group is `intrmotiv_dg_capacity_goal_conditioning_20260910`
in W&B project `SF_IntrMotiv_DGCapacityGoalConditioning`.

## Implementation

`GoalConditionedDGCore` preserves canonical DG/CA3 for graph evidence,
achievement, and encoder objectives. A separate worker trace receives
identity-initialized goal modulation of DG preactivations. The visual projection
and BatchNorm run once per observation; PPO cannot update the base DG through
this worker path. Existing worker FiLM remains enabled in every arm.

Recorded behavior goals enter the recurrent replay head before memory writes.
The persistent behavior descriptor remains the state suffix. The policy tail
consumes the worker trace; encoder losses consume the canonical prefix.

The waypoint arm enables existing passive discovery and deliberate edge
validation. Parser validation now allows that machinery with
`frontier_waypoint`. This contrast tests routing plus graph construction, not
routing alone. Direct C15 selection and timeout behavior remain the controls.

Initial checkpoints use a separate `initial_` prefix outside rolling retention.
Permanent checkpoints use the existing milestone format at the first learner
batch crossing 5M, 25M, 75M, 150M, and 300M; actual frame counts are retained.
The canonical selector discovers these milestones. Resume does not backfill
previous frame targets.

Runtime edits were context-applied over the existing NEMO2 worktree. No unrelated
source edits were reverted. The source archive under
`hpc_runs/source_snapshots/dg_capacity_goal_conditioning_20260910.tar.gz`
contains **153 relevant Python files**, all verified against deployed hashes.
The adjacent patch records this task's changes relative to its starting
snapshot. The final print-only directory also preserves the runtime Git revision
and tracked runtime diff.

## Verification

- **294 full IntrMotiv tests passed** on NEMO2. Subsequent evaluator-only outcome
  corrections passed all **four focused intervention tests**, including late
  arrivals and observed successes before later termination.
- **30 canonical/study tests passed locally and on NEMO2.** The workflow now
  supports both 75M and 300M intervention targets, with exactly 54 rows for the
  27 runs and rejection of missing or duplicate rows.
- All 27 final commands passed the actual parser, including state sizing,
  checkpoint targets, and a training-time ceiling longer than the allocation.
- Tests and source checks cover initial identity, canonical detector invariance,
  packed/plain replay, goal switches, resets, gradient boundaries, checkpoint
  restoration, and milestone/resume behavior. A real model smoke using the
  encoder, core, and decoder produced a nonzero modulation gradient.
- [Initial checkpoint comparisons](data/dg_capacity_goal_conditioning_20260910/initial_pair_comparison.json)
  found **all 113 common tensors exactly equal** between the worker-only arm
  and each DG-conditioned arm at every capacity.

All nine final preflights, jobs **8048800–8048808**, reached **2,031,616 frames**
and [exited cleanly](data/dg_capacity_goal_conditioning_20260910/completion_states.json).
Their [full runtime audit](data/dg_capacity_goal_conditioning_20260910/runtime_audit.json)
passes: initial identity, frozen visual trunk including normalization buffers,
trainable DG, finite learning, live nonzero modulation gradients, successful
waypoint validation/multihop execution, and both 1M and 2M spatial snapshots.

Real DMLab evaluation smoke **8049046** completed with exit 0. Its
[summary](data/dg_capacity_goal_conditioning_20260910/smoke_summary.json) verifies
exact observation-panel replay, canonical detector invariance, goal-sensitive
worker DG, identical physical/recurrent starts, and frozen policy/graph tensors.
Its two paired comparisons had zero arrival lift: this validates infrastructure,
not learned controllability.

The evaluator records unsupported panels, coverage, initial action total
variation, physical endpoints, arrival deadlines, and censoring. Legacy online
sensitivity/TV changes the worker command while holding replayed memory fixed;
it is a **readout-only** diagnostic in DG-conditioned arms. Use forced-goal
replay and matched-command interventions for the full DG-plus-worker effect.
These no-gradient diagnostics add no goal-discrimination loss.

## Submission and operational history

All paths below are relative to the workspace `train_dir/`.

- Final print-only review:
  `_slurm/intrmotiv_dg_capacity_goal_conditioning_20260910/20260910T173152Z`.
- Production submission:
  `_slurm/intrmotiv_dg_capacity_goal_conditioning_20260910/20260910T182743Z`.
  Its `scancel.sh` contains only this production submission's job IDs.
- Final preflight submission:
  `_slurm/intrmotiv_dg_capacity_goal_conditioning_preflight2_20260910/20260910T171148Z`.
- Real evaluator smoke:
  `analysis/dg_capacity_preflight/evaluation_smoke_8049046/`.

The first diagnostic preflights (8048750–8048758) were cancelled after exposing
missing frame-zero saves. Their definition and logs remain in the original
preflight submission directory, timestamp `20260910T165742Z`; they were not
counted as passed gates.

Final preflight 8048800 resumed from 720,896 frames on Genoa after measured Milan
throughput was too slow. Preflight 8048808 resumed from 1,572,864 frames before
its one-hour training ceiling, leaving enough frames to refill the 100k-sample
telemetry ring before 2M. Their job IDs, directories, seeds, and scientific
arguments were preserved. Pre-requeue logs and checkpoint hashes are in
`requeue_8048800/` and `requeue_8048808/` beside the final-preflight manifest.

Production retains the reviewed **40 CPUs, 80 GB, 96 hours, partition genoa**
for every job. Preflight memory peaks were about 19–21 GB. The last production
jobs waited for resources; no production resource requests were changed to
bypass that wait.

All CPU partitions have a four-day allocation limit. Measured waypoint-F64
throughput suggests that some 300M-frame runs will require checkpoint
continuation beyond one allocation. A timed-out allocation must not be treated
as a completed 300M replicate; resume the existing run with the established
launcher workflow. The startup objective is complete; the training and its
later scientific evaluations remain ongoing.

## Reusable lessons and review boundary

Use the actual recurrent write/replay path as evidence: overriding only decoder
goals is insufficient when commands affect DG writes. Keep independent canonical
grounding and test interventions from matched starts. Exercise telemetry manifest
generation before training; expansion alone missed the former single-target
restriction. Verify telemetry aliases before interpreting absent signals.

Request frame-zero and frame-target saves explicitly. A resumed telemetry ring
needs fresh valid samples; at repeat 4, a 100k-sample window requires at least
400k fresh frames. Preserve scheduler requeue provenance and inspect actual
runner timer semantics. These lessons are recorded in the canonical workflow
and reusable telemetry guides.

Review pretraining representation and controllability jointly before defining
or launching downstream transfer to the three-randomized-goal task.
