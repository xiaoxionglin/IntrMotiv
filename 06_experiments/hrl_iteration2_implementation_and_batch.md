# HRL Iteration 2: Implementation and Batch Record

Date: 2026-08-19  
Batch: `intrmotiv_hrl_iteration2_20260819`  
W&B project: `SF_HRL_Intrinsic_ArchSearch`  
W&B group: `intrmotiv_hrl_iteration2_20260819`

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

## Production submission

Submission work directory:

`train_dir/_slurm/intrmotiv_hrl_iteration2_20260819/20260819T171434Z`

Production job IDs are `7827944` through `7827955`. Logs and separate stderr
files are under the submission work directory. The generated `jobs.tsv`
records the experiment name, command, Slurm script, and job ID for every run.

## Validation

- IntrMotiv unit/integration tests: 18 passed.
- Production launcher dry run: 12 experiment commands and 12 Slurm scripts.
- SFgit W&B SDK: 0.24.1; authenticated entity verified before submission.
- Existing Batch 1 jobs and Jannek's directory were not modified or stopped.

## Next iteration training budget

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

| History | Step at 1 MiB | W&B endpoint | Local endpoint |
|---|---:|---:|---:|
| Iteration 2 policy 0 | 3,538,944 | 3,538,944 | 12,877,824 |
| Iteration 2 policy 1 | 3,768,320 | 3,768,320 | 13,303,808 |
| Iteration 2 policy 2 | 3,768,320 | 3,768,320 | 13,205,504 |
| Iteration 2 policy 3 | 3,801,088 | 3,801,088 | 12,550,144 |
| Batch 1, one policy | 5,865,472 | 5,865,472 | 90,013,696 |

Before the next run, upgrade the SFgit environment to W&B 0.25.0 or newer
and validate a short live TensorBoard run whose event file exceeds 1 MiB. The
completed local event files are intact and can be imported to repair the cloud
histories after validating the upgraded SDK on one run.
