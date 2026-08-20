# HRL Iteration 2: Current Summary and Remaining Plan

Date: 2026-08-19  
Last updated: 2026-08-20
Batch: `intrmotiv_hrl_iteration2_20260819`  
W&B project: `SF_HRL_Intrinsic_ArchSearch`  
W&B group: `intrmotiv_hrl_iteration2_20260819`

This is the canonical experiment-status document. It combines the implementation
record, the previous-batch diagnosis, the corrected 2026-08-20 sweep status,
and the remaining plan. The detailed tensor-level architecture reference is
kept separately in `04_implementation/current_hrl_architecture_summary.md`.

## Purpose

This iteration corrects the main Batch 1 failure modes before another broad
architecture sweep. It keeps curiosity as the manager objective: travel time
does not reduce a target's novelty score and there is no per-step option cost.

## Implemented changes

### Controllable graph

- HRL state remains episode-local and packed into `rnn_states` for sampling /
  PPO replay consistency.
- State now includes an intended-success count matrix next to `Tctrl`.
- `Tctrl[i,j]` is the running mean arrival time for successful, deliberately
  selected options from source `i` to target `j`.
- Accidental DG arrivals do not update controllability.
- Unknown targets use a separate bootstrap horizon of 64 actions. `L` remains
  CA3 memory length, not the learned option horizon.
- Known targets expire at the learned mean time plus a 20% margin and two
  actions. A timeout triggers deterministic reselection.

### Target selection and reward

- A DG unit is target-eligible only after at least one episode-local visit.
- The manager selects the least-visited eligible unit. Controllability cost is
  not part of the novelty score.
- With no eligible non-source unit, no target is assigned until bootstrap
  evidence exists; silent arbitrary units are not selected.
- `hit` worker reward is `1` on an exact target hit and `0` otherwise.
- `hit_distance` adds `0.1` times a clipped, nonnegative legacy temporal bonus.
- Consequently, a successful option cannot produce negative intrinsic reward.

### Representation and predictor

- Visual features use `layer2_resnet18`, not the pretrained ResNet path.
- `encoder_batch_loss` is always enabled.
- Population usage, target-density, and multi-activation controls are enabled.
- Encoder feedback normalization is deliberately deferred for this iteration.
- A shadow predictor consumes the complete current CA3 state and target
  one-hot vector. It predicts hit-within-window probability and conditional
  hit time. Its auxiliary loss trains the predictor only; it does not yet
  change target choice, actions, rewards, or deadlines.

### Environment and objective

- Every episode uses
  `openfield_map2_fixed_loc3_fixedlength_noreward`: 7,200 engine frames / 900
  policy actions at frameskip 8, independent of goal contact.
- DMLab position is used only for telemetry and is removed before policy input.
- Episode statistics include unique grid cells, occupancy entropy, and spatial
  coverage AUC.
- PBT maximizes coverage AUC only when DG silent fraction is at most 0.5, active
  target fraction exceeds 0.5, and intrinsic reward has no negative samples.
  Invalid policies receive objective zero.

## Batch design

The 12 independent Slurm jobs are the factorial product:

| Factor | Values |
|---|---|
| Population seed | 8, 99, 123 |
| Encoder feedback | `encourage`, `mean` |
| Worker reward | `hit`, `hit_distance` |

Each job contains four PBT policies. Architecture is fixed at `F=16`, `L=64`,
DG threshold `2.0`. Each policy is configured for 12.5M environment steps, so
the nominal aggregate population budget is about 50M steps. PBT inheritance
can make the final physical-frame count somewhat larger because a replaced
learner loads the donor checkpoint counter. The eight-hour training limit is
a simple fallback, not a separate global-step accounting system.

Throughput settings are 32 rollout workers, two environments per worker, two
worker splits, one policy worker per policy, rollout/recurrence 64, and batch
size 2,048. Each Slurm job requests the established CPU envelope: 40 CPUs,
80 GB RAM, no GPU, and 12:30 hours on the `cpu` partition.

PBT starts at 2.5M steps per policy and evaluates every 1.25M steps. With four
policies and replacement fraction 0.2, one weak policy can inherit from one
strong policy at a replacement event.

## Preflight evidence

Slurm job `7827943` ran the real four-policy pipeline with shortened budgets
and completed with exit code 0 in 2:15.

- Final collected counters: `{0: 83968, 1: 124928, 2: 122880, 3: 67584}`.
- Aggregate throughput: approximately 3,267 frames/s.
- Stock PBT matched `intrmotiv_pbt_objective`, ranked policies, saved donor
  checkpoints, and loaded them into other learners.
- Latest policies had active-target fractions of 0.969-0.988.
- Target-hit rates were 0.008-0.043 per transition.
- Intrinsic reward negative fraction was zero for all four policies.
- CA3 predictor loss and hit-time metrics were present for all policies.
- HRL validity was one and coverage-based objectives were nonzero for all four
  policies.

The preflight also found and fixed one checkout-specific compatibility issue:
the IntrMotiv train-stat callback now appends to the existing runner handler
list, preserving Sample Factory's built-in handler.

## Previous production submission (2026-08-19)

Submission work directory:

`train_dir/_slurm/intrmotiv_hrl_iteration2_20260819/20260819T171434Z`

Production job IDs are `7827944` through `7827955`. Logs and separate stderr
files are under the submission work directory. The generated `jobs.tsv`
records the experiment name, command, Slurm script, and job ID for every run.

## Previous-batch validation

- IntrMotiv unit/integration tests: 18 passed.
- Production launcher dry run: 12 experiment commands and 12 Slurm scripts.
- SFgit W&B SDK: 0.24.1; authenticated entity verified before submission.
- Existing Batch 1 jobs and Jannek's directory were not modified or stopped.

## Historical next-iteration budget (superseded by the corrected sweep below)

- Retain both `hit` and `hit_distance` worker reward conditions. Treat each
  four-policy PBT population as one replicate; the three populations per
  iteration-2 condition are insufficient to eliminate `hit_distance` based on
  coverage differences.
- Treat 100M environment steps as the approximate budget for the four-policy
  PBT population, not as 100M for each policy. The population is the scientific
  replicate and its metrics should be averaged across policies.
- Sample Factory applies `train_for_env_steps` per policy and stops when all
  policies cross it. Configure `train_for_env_steps=25000000` for four PBT
  policies, yielding approximately 100M aggregate population steps. Estimate
  wall-clock time from measured throughput and request roughly 30 hours so
  more allocations can run concurrently; wall-clock time is not the intended
  stopping condition.
- Keep the frame target as the primary Sample Factory stopping condition, but
  omit `train_for_seconds` from the next batch. Sample Factory combines the two
  limits with OR semantics, and this checkout initializes
  `total_train_seconds` without updating it, so the time argument is currently
  ineffective and would become an unwanted early-stop condition if that
  upstream bug were fixed. Use the Slurm time limit as the operational ceiling.
- `env_steps` includes frameskip. At `env_frameskip=8`, the 100M population
  budget corresponds to approximately 12.5M policy decisions in aggregate,
  subject to the four-policy scheduling details.
- Iteration-2 throughput suggests a roughly 20-40 hour allocation for the
  100M population target. Request about 30 hours when queue concurrency is
  more valuable than a single long allocation.
- Validate completion using the local event/checkpoint counters and the
  population aggregate. W&B policy-prefixed series (`0/global_step` through
  `3/global_step`) should be averaged for reporting; do not require each one
  to reach 100M independently.
- Repair or replace the live TensorBoard-to-W&B synchronization path before
  relying on cloud curves. An iteration-2 cloud history truncated near 3.5M
  while its local histories reached 12.5-13.3M per policy. The preceding
  one-policy Batch 1 also truncated at 5.8M in W&B while its local event file
  reached 90.0M, so the frame stopping limit is not the cause.

### Confirmed W&B root cause

Both batches used W&B 0.24.1. W&B 0.25.0 fixes a regression introduced in
0.24.0 where live TensorBoard synchronization stops after reading 1 MiB from
an event file; the 0.24.1 release fixed a different upload-loss defect but did
not include this TensorBoard fix.

Parsing each event file only through its first 1 MiB reproduced the cloud
endpoint exactly:

| History              | Step at 1 MiB | W&B endpoint | Local endpoint |
| -------------------- | ------------: | -----------: | -------------: |
| Iteration 2 policy 0 |     3,538,944 |    3,538,944 |     12,877,824 |
| Iteration 2 policy 1 |     3,768,320 |    3,768,320 |     13,303,808 |
| Iteration 2 policy 2 |     3,768,320 |    3,768,320 |     13,205,504 |
| Iteration 2 policy 3 |     3,801,088 |    3,801,088 |     12,550,144 |
| Batch 1, one policy  |     5,865,472 |    5,865,472 |     90,013,696 |

Before the next run, upgrade the SFgit environment to W&B 0.25.0 or newer
and validate a short live TensorBoard run whose event file exceeds 1 MiB. The
completed local event files are intact and can be imported to repair the cloud
histories after validating the upgraded SDK on one run.

## Corrected current sweep (2026-08-20)

The current production batch supersedes the 2026-08-19 submission above:

| Setting | Current value |
|---|---|
| Batch/W&B group | `intrmotiv_hrl_iteration2_iterative_sweep3_20260820` |
| Jobs | 18: 3 seeds x 3 encoder methods x 2 worker reward modes |
| Seeds | 8, 99, 123 |
| Encoder methods | `punish`, `encourage`, `mean` |
| Worker rewards | `hit`, `hit_distance` |
| Policies per PBT population | 4 |
| Sample Factory target | 25M env steps per policy, about 100M total |
| Slurm allocation | CPU, 40 CPUs, 80 GB, 30 hours, no GPU |
| Environment | `openfield_map2_fixed_loc3_fixedlength_noreward` |
| Architecture | `F=16`, `L=64`, fixed `layer2_resnet18`, iterative update enabled |
| DG controls | `encoder_batch_loss=True`, density/usage/collision controls enabled |
| W&B SDK | 0.28.2 in `SFgit` |

The four policies are one scientific population. Metrics are averaged across
all four policy streams with
`analysis/aggregate_pbt_population.py`; their curves must not be treated as
four independent replicates.

All 18 jobs created four policy event streams (72 event files), but the batch
did not remain healthy. Each Runner stopped updating summaries and frame
counters about 30 minutes after launch. Slurm continued to report the jobs as
`RUNNING` because child processes remained alive; their report and heartbeat
queues later filled because the Runner no longer consumed messages. The jobs
therefore used wall time without producing additional usable training data.

The earlier W&B truncation was traced to TensorBoard synchronization in W&B
0.24.x stopping at the first 1 MiB of an event file. The current batch uses
W&B 0.28.2; local event files and population aggregation remain authoritative.

## Corrected-sweep interim analysis

### Validity and failure diagnosis

- Useful per-policy endpoints range from about 1.51M to 3.57M environment
  steps. Per-population totals are only about 7.5M-14.9M frames, or 7.5%-14.9%
  of the intended 100M population budget.
- In a representative run, the last Runner frame report was 14.61M aggregate
  frames at 02:48:39. All four milestone checkpoints were written between
  02:48:44 and 02:48:52. Multiple jobs then reported `OSError: [Errno 28] No
  space left on device` at 02:49:15. The first full report queues appeared at
  03:24 after rollout processes had failed.
- The 30-minute alignment is exact across the sweep and coincides with
  `save_milestones_sec=1800`. The synchronized milestone writes consumed the
  remaining capacity on the full home filesystem. Subsequent environment-cache
  and logging writes failed, stopping useful training.
- A live process inspection found the Runner sleeping in a futex, learner
  processes still present, and rollout processes eventually defunct. These
  were downstream symptoms of worker write failures. Do not treat Slurm
  elapsed time as training time for this batch.

Consequently, these data cannot establish convergence or long-run HRL
performance. They can only compare early dynamics at a common pre-PBT step.

### Matched comparison at 1.5M steps per policy

The four policies are averaged into one population, then the three population
seeds are averaged per condition. At 1.5M steps PBT has not started, so this
comparison isolates the configured encoder feedback and worker reward modes.

| Encoder | Worker | Coverage AUC | DG density | Multi-active | Silent units | Hit rate | Option success | Known edges |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| encourage | hit | 67.57 | 0.0579 | 0.2547 | 0.0052 | 0.00920 | 0.3801 | 0.0170 |
| encourage | hit_distance | 68.02 | 0.0597 | 0.2751 | 0.0000 | 0.01013 | 0.3805 | 0.0194 |
| mean | hit | 58.18 | 0.0053 | 0.0156 | 0.4583 | 0.00452 | 0.2264 | 0.0090 |
| mean | hit_distance | 57.86 | 0.0063 | 0.0183 | 0.3750 | 0.00541 | 0.2523 | 0.0108 |
| punish | hit | 58.12 | 0.0092 | 0.0304 | 0.4844 | 0.00566 | 0.2648 | 0.0082 |
| punish | hit_distance | 57.86 | 0.0037 | 0.0092 | 0.4115 | 0.00566 | 0.2717 | 0.0100 |

`encourage` is the only encoder method that reliably preserves DG activity.
It produces roughly ten times the DG density of `mean`/`punish`, almost no
silent units, about twice the target-hit rate, and approximately 9.5-10.2
additional coverage-AUC points. From 1.0M to 1.5M steps, encourage coverage
continues to rise by 2.7-3.3 points while mean and punish decline by roughly
1.3-1.9 points and their DG density falls. This is evidence of early DG
collapse under `mean` and `punish`, not merely a threshold offset.

The encourage effect is not yet uniform across seeds. Seeds 8 and 99 improve
strongly, while seed 123 is close to the collapsed methods at 1.5M. Three
population seeds are therefore enough to identify the failure mode but not to
make a precise effect-size claim.

### Worker reward and predictor

- `hit_distance` raises measured intrinsic reward but does not improve matched
  coverage at 1.5M: the worker-mode marginal coverage is 61.25 versus 61.29
  for `hit`. Keep the requested hit-distance diagnostics, but do not attribute
  an exploration benefit to the bonus from this batch.
- Across the 18 populations, coverage correlates with hit rate (0.79), known
  edges (0.78), multi-activation rate (0.76), option success (0.74), and DG
  density (0.70). The controllable graph is active when the representation is
  active; it cannot compensate for DG collapse.
- Raw predictor accuracy is misleading: encourage has lower accuracy because
  it generates substantially more positive hits, while collapsed methods can
  score well by predicting no hit. Future evaluation needs balanced accuracy,
  precision/recall or PR-AUC, calibration, and conditional time error.

### Decision from this batch

Use `encourage` as the default encoder feedback for the next valid long run.
Do not spend another full sweep budget on `mean` or `punish` unless the purpose
is specifically to study representation collapse. Before resubmission, make a
four-policy test cross the former 30-minute failure boundary and at least two
PBT periods while monitoring workspace capacity; only then launch the 100M
population runs.

This factorial sweep cannot measure the benefit of iterative encoder/decoder
updates because every condition enables them. After relocating all run data,
the highest-value architectural comparison is a matched `encourage` run with
iterative update on versus off, preserving the Jannek-compatible baseline and
all other settings. The present batch also has no non-HRL navigation baseline,
so coverage gains should not yet be attributed specifically to graph planning.

### Storage-exhaustion fix

Move `train_dir`, W&B local data, Slurm logs, caches, and all other bulk outputs
to an allocated NEMO2 workspace. Production checkpoint behavior, including the
1,800-second milestone interval, remains unchanged. Before another production
submission, verify every resolved output path and run a four-policy preflight
while monitoring workspace capacity.

## Remaining plan

### HRL behavior

- Keep the episode-local controllable graph in `rnn_states` for PPO replay
  consistency.
- Keep target selection curiosity-driven: `T_ctrl` is a feasibility gate and
  deadline source, not a cost or novelty ranking term.
- Prefer currently reachable targets using multi-hop closure; fall back to an
  observed frontier when no reachable target exists.
- Keep option deadlines experience-derived from successful travel times, with
  a small timeout margin and deterministic target switching.
- Preserve target/source ids, option resets, hits, controllability updates,
  and hit-distance diagnostics.

### Predictor

- Keep the CA3 predictor shadow/auxiliary-only during this iteration.
- It currently consumes the complete CA3 state plus target one-hot and predicts
  target hit probability and conditional hit time. Compare those predictions
  against `T_ctrl`; do not let the predictor replace the graph yet.
- Treat next-DG/F-state prediction and top-k successor metrics as a follow-up
  extension if the current predictor is insufficient for the planned
  successor-structure comparison.

### Accounting and evaluation

- Let the current sweep reach approximately 100M aggregate environment steps.
- Report each four-policy PBT population as one replicate and average its four
  policy streams.
- Compare all six conditions using population-level curves and diagnostics.
- Require sustained DG activity, nonzero target hits and low-level intrinsic
  reward, increasing controllable-graph coverage, and continued navigation
  improvement through the budget.
- If DG activity remains nonzero but target hits stay near zero, prioritize
  feasibility/planning and option timing. If DG activity collapses, debug
  encoder rewards and gradient ownership before interpreting HRL behavior.
